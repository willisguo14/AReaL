# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import ProcessGroup

from areal.engine.fsdp_utils.grad import (
    is_param_not_tensor_parallel_duplicate,
    to_local_if_dtensor,
)
from areal.infra.platforms import current_platform


@dataclasses.dataclass(frozen=True)
class GradientSnapshot:
    names: tuple[str, ...]
    grads: tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class PendingGradientCosine:
    snapshot: GradientSnapshot
    current_is_valid: bool
    cosine: float | None


def _get_tensor_parallel_rank(tp_group: ProcessGroup | None) -> int:
    if tp_group is None or not dist.is_available() or not dist.is_initialized():
        return 0
    return int(dist.get_rank(tp_group))


def _collect_gradient_snapshot(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    tensor_parallel_rank: int,
) -> GradientSnapshot:
    names: list[str] = []
    grads: list[torch.Tensor] = []

    for name, param in named_parameters:
        grad = param.grad
        if grad is None:
            continue
        if not is_param_not_tensor_parallel_duplicate(
            param,
            tensor_parallel_rank,
        ):
            continue
        local_grad = to_local_if_dtensor(grad)
        names.append(name)
        grads.append(local_grad.detach().clone())

    return GradientSnapshot(names=tuple(names), grads=tuple(grads))


def _stats_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        return current_platform.current_device()
    return torch.device(device)


def _sum_sq_on_stats_device(
    tensor: torch.Tensor,
    stats_device: torch.device,
) -> torch.Tensor:
    tensor_fp32 = tensor.detach().to(dtype=torch.float32)
    return torch.sum(tensor_fp32 * tensor_fp32).to(device=stats_device)


def _dot_on_stats_device(
    left: torch.Tensor,
    right: torch.Tensor,
    stats_device: torch.device,
) -> torch.Tensor:
    left_fp32 = left.detach().to(dtype=torch.float32)
    right_fp32 = right.detach().to(
        device=left_fp32.device,
        dtype=torch.float32,
    )
    return torch.sum(left_fp32 * right_fp32).to(device=stats_device)


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
        stats[1] += _sum_sq_on_stats_device(grad, device)

    if previous is None or stats[mismatch_idx].item() != 0.0:
        return stats

    for current_grad, previous_grad in zip(current.grads, previous.grads, strict=True):
        stats[0] += _dot_on_stats_device(current_grad, previous_grad, device)
        stats[2] += _sum_sq_on_stats_device(previous_grad, device)

    return stats


def _all_reduce_stats(
    stats: torch.Tensor,
    groups: tuple[ProcessGroup | None, ...],
) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return stats

    for group in groups:
        if group is not None:
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


class FSDPGradientCosineTracker:
    def __init__(self, eps: float = 1e-8):
        self._eps = eps
        self._previous: GradientSnapshot | None = None

    @torch.no_grad()
    def prepare(
        self,
        *,
        model: nn.Module,
        fsdp_group: ProcessGroup | None,
        tp_group: ProcessGroup | None,
        device: torch.device | str | int | None,
    ) -> PendingGradientCosine:
        tensor_parallel_rank = _get_tensor_parallel_rank(tp_group)
        snapshot = _collect_gradient_snapshot(
            model.named_parameters(),
            tensor_parallel_rank,
        )
        stats = _local_reduction_stats(
            snapshot,
            self._previous,
            _stats_device(device),
        )
        stats = _all_reduce_stats(stats, (fsdp_group, tp_group))

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
