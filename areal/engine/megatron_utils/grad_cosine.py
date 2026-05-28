# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


@dataclasses.dataclass(frozen=True)
class GradientSnapshot:
    names: tuple[str, ...]
    grads: tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class PendingGradientCosine:
    snapshot: GradientSnapshot
    current_is_valid: bool
    cosine: float | None


def _param_is_not_shared(param: nn.Parameter) -> bool:
    from megatron.core.transformer.module import param_is_not_shared

    return bool(param_is_not_shared(param))


def _param_is_not_tensor_parallel_duplicate(param: nn.Parameter) -> bool:
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel

    tp_group = mpu.get_tensor_model_parallel_group()
    try:
        return bool(
            tensor_parallel.param_is_not_tensor_parallel_duplicate(
                param,
                tp_group=tp_group,
            )
        )
    except TypeError:
        return bool(tensor_parallel.param_is_not_tensor_parallel_duplicate(param))


def _get_tensor_model_parallel_rank() -> int:
    from megatron.core import parallel_state as mpu

    return int(mpu.get_tensor_model_parallel_rank())


def _get_model_parallel_group():
    from megatron.core import parallel_state as mpu

    return mpu.get_model_parallel_group()


def _get_grad_stats_parallel_group(optimizer: Any):
    get_group = getattr(optimizer, "get_grad_stats_parallel_group", None)
    if get_group is not None:
        return get_group()
    return _get_model_parallel_group()


def _get_param_grad(param: nn.Parameter) -> torch.Tensor | None:
    main_grad = getattr(param, "main_grad", None)
    if main_grad is not None:
        return main_grad
    return param.grad


def _should_count_param(
    name: str,
    param: nn.Parameter,
    duplicated_param_names: set[str],
) -> bool:
    if not _param_is_not_shared(param):
        return False
    if name in duplicated_param_names:
        return _get_tensor_model_parallel_rank() == 0
    return _param_is_not_tensor_parallel_duplicate(param)


def _collect_gradient_snapshot(
    model: Iterable[nn.Module],
    duplicated_param_names: set[str],
) -> GradientSnapshot:
    names: list[str] = []
    grads: list[torch.Tensor] = []
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if not _should_count_param(name, param, duplicated_param_names):
                continue
            grad = _get_param_grad(param)
            if grad is None:
                continue
            names.append(name)
            grads.append(grad.detach().clone())
    return GradientSnapshot(names=tuple(names), grads=tuple(grads))


def _snapshot_device(
    snapshot: GradientSnapshot,
    fallback: torch.device | str,
) -> torch.device:
    for grad in snapshot.grads:
        return grad.device
    return torch.device(fallback)


def _local_reduction_stats(
    current: GradientSnapshot,
    previous: GradientSnapshot | None,
    device: torch.device,
) -> torch.Tensor:
    stats = torch.zeros(5, dtype=torch.float32, device=device)
    missing_previous_idx = 3
    mismatch_idx = 4

    if previous is None:
        stats[missing_previous_idx] = 1.0
    elif current.names != previous.names or len(current.grads) != len(previous.grads):
        stats[mismatch_idx] = 1.0
    elif any(
        cur.shape != prev.shape
        for cur, prev in zip(current.grads, previous.grads, strict=True)
    ):
        stats[mismatch_idx] = 1.0

    for grad in current.grads:
        grad_fp32 = grad.to(device=device, dtype=torch.float32)
        stats[1] += torch.sum(grad_fp32 * grad_fp32)

    if previous is None or stats[mismatch_idx].item() != 0.0:
        return stats

    for current_grad, previous_grad in zip(current.grads, previous.grads, strict=True):
        current_fp32 = current_grad.to(device=device, dtype=torch.float32)
        previous_fp32 = previous_grad.to(device=device, dtype=torch.float32)
        stats[0] += torch.sum(current_fp32 * previous_fp32)
        stats[2] += torch.sum(previous_fp32 * previous_fp32)
    return stats


def _all_reduce_stats(stats: torch.Tensor, group) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
    return stats


def _cosine_from_reduced_stats(
    *,
    dot: float,
    current_norm_sq: float,
    previous_norm_sq: float,
    missing_previous_count: float,
    mismatch_count: float,
    eps: float,
) -> float | None:
    if missing_previous_count != 0.0 or mismatch_count != 0.0:
        return None
    if current_norm_sq <= 0.0 or previous_norm_sq <= 0.0:
        return None
    if not (
        math.isfinite(dot)
        and math.isfinite(current_norm_sq)
        and math.isfinite(previous_norm_sq)
    ):
        return None
    return dot / (math.sqrt(current_norm_sq * previous_norm_sq) + eps)


class GradientCosineTracker:
    def __init__(self, eps: float = 1e-8):
        self._eps = eps
        self._previous: GradientSnapshot | None = None

    @torch.no_grad()
    def prepare(
        self,
        *,
        model: Iterable[nn.Module],
        optimizer: Any,
        duplicated_param_names: set[str],
        device: torch.device | str,
    ) -> PendingGradientCosine:
        snapshot = _collect_gradient_snapshot(model, duplicated_param_names)
        stats_device = _snapshot_device(snapshot, device)
        stats = _local_reduction_stats(snapshot, self._previous, stats_device)
        stats = _all_reduce_stats(stats, _get_grad_stats_parallel_group(optimizer))

        dot = float(stats[0].item())
        current_norm_sq = float(stats[1].item())
        previous_norm_sq = float(stats[2].item())
        missing_previous_count = float(stats[3].item())
        mismatch_count = float(stats[4].item())

        current_is_valid = math.isfinite(current_norm_sq) and current_norm_sq > 0.0
        cosine = _cosine_from_reduced_stats(
            dot=dot,
            current_norm_sq=current_norm_sq,
            previous_norm_sq=previous_norm_sq,
            missing_previous_count=missing_previous_count,
            mismatch_count=mismatch_count,
            eps=self._eps,
        )
        return PendingGradientCosine(
            snapshot=snapshot,
            current_is_valid=current_is_valid,
            cosine=cosine,
        )

    def finalize(
        self,
        pending: PendingGradientCosine,
        *,
        update_successful: bool,
    ) -> float | None:
        if not update_successful:
            return None
        if not pending.current_is_valid:
            self._previous = None
            return None
        self._previous = pending.snapshot
        return pending.cosine
