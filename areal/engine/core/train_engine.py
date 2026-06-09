# SPDX-License-Identifier: Apache-2.0

"""Core operations for training engines.

This module provides stateless utility functions that are shared across
different training engine implementations (FSDP, Megatron, etc.).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils.data import (
    MicroBatchList,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)

__all__ = [
    "LOGP_GRAD_ABSMAX_KEY",
    "LOGP_GRAD_NORM_KEY",
    "LogprobGradAccumulator",
    "compute_total_loss_weight",
    "aggregate_eval_losses",
    "reorder_and_pad_outputs",
]


LOGP_GRAD_NORM_KEY = "logp_grad_norm"
LOGP_GRAD_ABSMAX_KEY = "logp_grad_absmax"


@dataclass(frozen=True)
class _ReductionGroup:
    group: dist.ProcessGroup | None
    sum_divisor: float = 1.0


class LogprobGradAccumulator:
    """Accumulate ||dL/dlogprobs|| and max |dL/dlogprobs| for one train step."""

    def __init__(self, device: torch.device | str | int | None = None) -> None:
        if device is None:
            device = current_platform.current_device()
        self._sum_sq = torch.zeros((), dtype=torch.float32, device=device)
        self._abs_max = torch.zeros((), dtype=torch.float32, device=device)

    def add(
        self,
        loss: torch.Tensor,
        logprobs: torch.Tensor,
        loss_scale: torch.Tensor | float,
    ) -> None:
        scaled_loss = loss * loss_scale
        grad = torch.autograd.grad(
            scaled_loss,
            logprobs,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None or grad.numel() == 0:
            return

        grad = grad.detach().to(dtype=torch.float32)
        self._sum_sq += torch.sum(grad * grad)
        self._abs_max = torch.maximum(self._abs_max, grad.abs().max())

    def summary(
        self,
        *groups: tuple[dist.ProcessGroup | None, float] | dist.ProcessGroup | None,
    ) -> dict[str, float]:
        sum_sq = self._sum_sq.detach().clone()
        abs_max = self._abs_max.detach().clone()

        for reduction in self._normalize_groups(groups):
            if (
                reduction.group is not None
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(sum_sq, op=dist.ReduceOp.SUM, group=reduction.group)
                dist.all_reduce(abs_max, op=dist.ReduceOp.MAX, group=reduction.group)
            if reduction.sum_divisor != 1.0:
                sum_sq /= reduction.sum_divisor

        sum_sq_value = max(float(sum_sq.item()), 0.0)
        return {
            LOGP_GRAD_NORM_KEY: math.sqrt(sum_sq_value),
            LOGP_GRAD_ABSMAX_KEY: float(abs_max.item()),
        }

    @staticmethod
    def _normalize_groups(
        groups: tuple[
            tuple[dist.ProcessGroup | None, float] | dist.ProcessGroup | None, ...
        ],
    ) -> tuple[_ReductionGroup, ...]:
        reductions = []
        for group in groups:
            if isinstance(group, tuple):
                process_group, divisor = group
            else:
                process_group, divisor = group, 1.0
            divisor = float(divisor)
            if divisor <= 0.0:
                raise ValueError(
                    f"logprob grad sum divisor must be positive, got {divisor}"
                )
            reductions.append(_ReductionGroup(process_group, divisor))
        return tuple(reductions)


def compute_total_loss_weight(
    mb_list: MicroBatchList,
    loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    dp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Compute total loss weight and all_reduce across data parallel group.

    This aggregates the loss weights from all micro-batches and reduces
    them across the data parallel group to get a global normalization factor.

    Parameters
    ----------
    mb_list : MicroBatchList
        The list of micro-batches.
    loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
        Function to compute loss weight for each micro-batch.
    dp_group : dist.ProcessGroup
        The data parallel process group for all_reduce.

    Returns
    -------
    torch.Tensor
        The total loss weight (scalar tensor) after all_reduce.
    """
    total_weight = (
        torch.stack([loss_weight_fn(mb) for mb in mb_list.mbs])
        .sum()
        .detach()
        .clone()
        .to(dtype=torch.float32)
    )
    dist.all_reduce(total_weight, group=dp_group)
    assert total_weight > 0, (
        "Global total loss weight must be positive after all_reduce"
    )
    return total_weight


def aggregate_eval_losses(
    losses: list[torch.Tensor] | None,
    dp_group: dist.ProcessGroup,
    is_pp_last_stage: bool = True,
    pp_group: dist.ProcessGroup | None = None,
    pp_src_rank: int | None = None,
) -> torch.Tensor:
    """Aggregate evaluation losses from micro-batches.

    Parameters
    ----------
    losses : list[torch.Tensor] | None
        List of loss tensors from each micro-batch. None on non-last PP stages.
    dp_group : dist.ProcessGroup
        The data parallel process group for all_reduce.
    is_pp_last_stage : bool
        Whether this rank is the last PP stage. True by default.
    pp_group : dist.ProcessGroup | None
        Pipeline parallel group for broadcast. None if PP broadcast is not required.
    pp_src_rank : int | None
        Global rank of last PP stage (required if pp_group is set).

    Returns
    -------
    torch.Tensor
        The aggregated loss after summing and all_reduce.
    """
    if is_pp_last_stage:
        assert losses is not None, "losses required on last PP stage"
        loss = torch.stack(losses).sum(dtype=torch.float32)
        dist.all_reduce(loss, group=dp_group)
    else:
        device = current_platform.current_device()
        loss = torch.empty(1, device=device, dtype=torch.float32)

    if pp_group is not None:
        assert pp_src_rank is not None, "pp_src_rank required when pp_group is set"
        dist.broadcast(loss, src=pp_src_rank, group=pp_group)

    return loss


def reorder_and_pad_outputs(
    outputs: list[torch.Tensor],
    output_seqlens: list[int],
    mb_list: MicroBatchList,
    aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
) -> torch.Tensor:
    """Aggregate, reorder, and pad forward outputs from micro-batches.

    This handles the output post-processing for forward_batch:
    1. Aggregate outputs from all micro-batches
    2. Unpack by sequence lengths
    3. Reorder to match original input order
    4. Pad and stack along batch dimension

    Parameters
    ----------
    outputs : list[torch.Tensor]
        List of output tensors from each micro-batch.
    output_seqlens : list[int]
        Sequence lengths for unpacking.
    mb_list : MicroBatchList
        The micro-batch list containing reordering indices.
    aggregate_fn : Callable[[list[Any]], Any], optional
        Function to aggregate outputs, by default torch.cat.

    Returns
    -------
    torch.Tensor
        The processed outputs, padded and stacked along batch dimension.
    """
    res = aggregate_fn(outputs)
    seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
    unpacked = unpack_sequence(res, lens=seqlens, dim=0)
    reordered = reorder_list(unpacked, mb_list.backward_indices)
    return pad_and_stack_tensors_along_first_dim(reordered)
