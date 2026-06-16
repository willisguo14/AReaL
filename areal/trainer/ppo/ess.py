# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ESSStats:
    ess: float
    ess_ratio: float
    ess_lr_scale: float
    valid_count: int


@dataclass(frozen=True)
class SequenceBehaviorMetrics:
    log_weight: torch.Tensor
    mean_log_ratio: torch.Tensor
    valid_mask: torch.Tensor


def _should_all_reduce(process_group: Any) -> bool:
    return process_group is not None and dist.is_available() and dist.is_initialized()


def _coerce_finite_scaling_value(scaling_config: Any, field_name: str) -> float:
    try:
        value = float(getattr(scaling_config, field_name))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite float") from exc

    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite, got {value}")

    return value


def _get_scaling_values(scaling_config: Any | None) -> tuple[float, float, float]:
    if scaling_config is None:
        return 1.0, 0.0, 1.0

    base_ess_ratio = _coerce_finite_scaling_value(scaling_config, "base_ess_ratio")
    min_lr_scale = _coerce_finite_scaling_value(scaling_config, "min_lr_scale")
    max_lr_scale = _coerce_finite_scaling_value(scaling_config, "max_lr_scale")

    if base_ess_ratio <= 0.0:
        raise ValueError(f"base_ess_ratio must be positive, got {base_ess_ratio}")
    if min_lr_scale < 0.0:
        raise ValueError(f"min_lr_scale must be non-negative, got {min_lr_scale}")
    if min_lr_scale > max_lr_scale:
        raise ValueError(
            f"min_lr_scale ({min_lr_scale}) cannot be greater than "
            f"max_lr_scale ({max_lr_scale})"
        )

    return base_ess_ratio, min_lr_scale, max_lr_scale


def _validate_ess_tensor_shapes(
    prox_logp: torch.Tensor, logprobs: torch.Tensor, loss_mask: torch.Tensor
) -> None:
    if not (prox_logp.shape == logprobs.shape == loss_mask.shape):
        raise ValueError(
            "prox_logp, logprobs, and loss_mask must have the same shape; "
            f"got prox_logp={tuple(prox_logp.shape)}, "
            f"logprobs={tuple(logprobs.shape)}, "
            f"loss_mask={tuple(loss_mask.shape)}"
        )

    if prox_logp.ndim != 2:
        raise ValueError(
            "prox_logp, logprobs, and loss_mask must be 2-D padded tensors "
            f"shaped [batch, sequence]; got ndim={prox_logp.ndim}"
        )


def _validate_ess_level(level: str) -> None:
    if level not in {"sequence", "token"}:
        raise ValueError(f"level must be one of 'sequence' or 'token'; got {level!r}")


def _masked_token_log_ratio(
    prox_logp: torch.Tensor,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_ess_tensor_shapes(prox_logp, logprobs, loss_mask)

    loss_mask = loss_mask.bool()
    token_log_ratio = prox_logp.float() - logprobs.float()
    token_log_ratio = torch.where(
        torch.isfinite(token_log_ratio),
        token_log_ratio,
        torch.zeros_like(token_log_ratio),
    )

    return token_log_ratio.masked_fill(~loss_mask, 0.0), loss_mask


def _compute_ess(
    *,
    level: str,
    prox_logp: torch.Tensor,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    scaling_config: Any | None = None,
    process_group: Any | None = None,
) -> ESSStats | None:
    _validate_ess_level(level)
    token_log_ratio, loss_mask = _masked_token_log_ratio(prox_logp, logprobs, loss_mask)

    if level == "sequence":
        valid_item_mask = loss_mask.any(dim=-1)
        log_weights = token_log_ratio.sum(dim=-1)[valid_item_mask]
    else:
        valid_item_mask = loss_mask
        log_weights = token_log_ratio[valid_item_mask]

    local_count = valid_item_mask.to(dtype=torch.float64).sum()
    has_valid_items = bool(valid_item_mask.any().item())

    if has_valid_items:
        log_weights = log_weights.to(dtype=torch.float64)
        local_max = log_weights.max()
    else:
        log_weights = torch.empty((0,), dtype=torch.float64, device=prox_logp.device)
        local_max = torch.tensor(
            float("-inf"), dtype=torch.float64, device=prox_logp.device
        )

    global_max = local_max.clone()
    should_all_reduce = _should_all_reduce(process_group)
    if should_all_reduce:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=process_group)

    global_count = local_count.clone()
    if should_all_reduce:
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=process_group)

    valid_count = int(global_count.item())
    if valid_count == 0:
        return None

    if has_valid_items:
        shifted = torch.exp(log_weights - global_max)
        shifted_sum = shifted.sum(dtype=torch.float64)
        shifted_sq_sum = (shifted * shifted).sum(dtype=torch.float64)
    else:
        shifted_sum = torch.tensor(0.0, dtype=torch.float64, device=prox_logp.device)
        shifted_sq_sum = torch.tensor(0.0, dtype=torch.float64, device=prox_logp.device)

    if should_all_reduce:
        dist.all_reduce(shifted_sum, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(shifted_sq_sum, op=dist.ReduceOp.SUM, group=process_group)

    ess = float((shifted_sum.square() / shifted_sq_sum).item())
    ess_ratio = ess / valid_count

    base_ess_ratio, min_lr_scale, max_lr_scale = _get_scaling_values(scaling_config)
    ess_lr_scale = math.sqrt(max(ess_ratio / base_ess_ratio, 0.0))
    ess_lr_scale = min(max(ess_lr_scale, min_lr_scale), max_lr_scale)

    return ESSStats(
        ess=ess,
        ess_ratio=ess_ratio,
        ess_lr_scale=float(ess_lr_scale),
        valid_count=valid_count,
    )


def compute_sequence_ess(
    *,
    prox_logp: torch.Tensor,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    scaling_config: Any | None = None,
    process_group: Any | None = None,
) -> ESSStats | None:
    return _compute_ess(
        level="sequence",
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
        scaling_config=scaling_config,
        process_group=process_group,
    )


def compute_token_ess(
    *,
    prox_logp: torch.Tensor,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    scaling_config: Any | None = None,
    process_group: Any | None = None,
) -> ESSStats | None:
    return _compute_ess(
        level="token",
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
        scaling_config=scaling_config,
        process_group=process_group,
    )


def compute_sequence_behavior_metrics(
    *,
    prox_logp: torch.Tensor,
    logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> SequenceBehaviorMetrics:
    token_log_ratio, loss_mask = _masked_token_log_ratio(prox_logp, logprobs, loss_mask)

    valid_mask = loss_mask.any(dim=-1)
    valid_token_count = loss_mask.sum(dim=-1).clamp(min=1)
    log_weight = token_log_ratio.sum(dim=-1)
    mean_log_ratio = log_weight / valid_token_count.to(dtype=log_weight.dtype)

    log_weight = torch.where(valid_mask, log_weight, 0.0).float()
    mean_log_ratio = torch.where(valid_mask, mean_log_ratio, 0.0).float()

    return SequenceBehaviorMetrics(
        log_weight=log_weight,
        mean_log_ratio=mean_log_ratio,
        valid_mask=valid_mask,
    )


def _add_finite_summary(
    summary: dict[str, float], name: str, values: list[float]
) -> None:
    finite_values = [float(value) for value in values if math.isfinite(value)]
    if not finite_values:
        return

    summary[f"{name}/avg"] = float(sum(finite_values) / len(finite_values))
    summary[f"{name}/min"] = float(min(finite_values))
    summary[f"{name}/max"] = float(max(finite_values))


def summarize_ess_stats(stats: list[ESSStats], *, level: str) -> dict[str, float]:
    _validate_ess_level(level)
    if not stats:
        return {}

    summary: dict[str, float] = {}
    _add_finite_summary(
        summary, f"ess_ratio_{level}", [stat.ess_ratio for stat in stats]
    )
    _add_finite_summary(summary, f"ess_{level}", [stat.ess for stat in stats])

    return summary


def summarize_ess_lr_scale_stats(stats: list[ESSStats]) -> dict[str, float]:
    if not stats:
        return {}

    summary: dict[str, float] = {}
    _add_finite_summary(summary, "ess_lr_scale", [stat.ess_lr_scale for stat in stats])

    return summary
