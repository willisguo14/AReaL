# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from areal.engine.core.per_trajectory import MaskedTensorSummary


@dataclass(frozen=True)
class PerTrajectoryFilterContext:
    grad_norm: float
    advantage_summary: MaskedTensorSummary | None = None
    behave_approx_kl_summary: MaskedTensorSummary | None = None
    advantage_mean_emitted: bool = False
    behave_approx_kl_mean_emitted: bool = False


FilterParams = dict[str, Any]
FilterFn = Callable[[FilterParams, PerTrajectoryFilterContext], bool]
ValidatorFn = Callable[[FilterParams], None]


def _config_field(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _known_rules() -> str:
    return ", ".join(PER_TRAJECTORY_FILTER_RULES)


def _params_dict(rule: str, params: Any) -> FilterParams:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ValueError(
            f"{rule} per-trajectory filter params must be a mapping or None, "
            f"got {type(params).__name__}"
        )
    return dict(params)


def _filter_rule_and_params(config: Any) -> tuple[str, FilterParams]:
    rule = _config_field(config, "rule")
    if not isinstance(rule, str) or not rule:
        raise ValueError(
            "unknown per-trajectory filter rule: rule must be a non-empty string; "
            f"known rules: {_known_rules()}"
        )
    if rule not in PER_TRAJECTORY_FILTER_RULES:
        raise ValueError(
            f"unknown per-trajectory filter rule {rule!r}; "
            f"known rules: {_known_rules()}"
        )
    return rule, _params_dict(rule, _config_field(config, "params", None))


def _finite_float(rule: str, name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"{rule} per-trajectory filter param {name!r} must be a finite number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{rule} per-trajectory filter param {name!r} must be finite")
    return result


def _optional_bound(rule: str, params: FilterParams, name: str) -> float | None:
    if name not in params:
        return None
    return _finite_float(rule, name, params[name])


def _reject_extra_params(
    rule: str,
    params: FilterParams,
    allowed: set[str],
) -> None:
    extra = sorted((key for key in params if key not in allowed), key=repr)
    if extra:
        rendered = ", ".join(repr(key) for key in extra)
        raise ValueError(
            f"{rule} per-trajectory filter does not accept params: {rendered}"
        )


def _validate_none(params: FilterParams) -> None:
    _reject_extra_params("none", params, set())


def _validate_grad_norm_max(params: FilterParams) -> None:
    _reject_extra_params("grad_norm_max", params, {"max"})
    if "max" not in params:
        raise ValueError(
            "grad_norm_max per-trajectory filter requires positive finite max"
        )
    max_value = _finite_float("grad_norm_max", "max", params["max"])
    if max_value <= 0.0:
        raise ValueError(
            "grad_norm_max per-trajectory filter max must be positive and finite"
        )


def _validate_kl_k1_range(params: FilterParams) -> None:
    _reject_extra_params("kl_k1_range", params, {"lower", "upper"})
    lower = _optional_bound("kl_k1_range", params, "lower")
    upper = _optional_bound("kl_k1_range", params, "upper")
    if lower is None and upper is None:
        raise ValueError(
            "kl_k1_range per-trajectory filter requires lower, upper, or both"
        )
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("kl_k1_range per-trajectory filter lower cannot exceed upper")


def _validate_advantage_mean_positive(params: FilterParams) -> None:
    _reject_extra_params("advantage_mean_positive", params, set())


def _finite_context_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _summary_mean(
    summary: MaskedTensorSummary | None,
    *,
    emitted: bool,
    name: str,
) -> float | None:
    if not emitted:
        raise RuntimeError(
            f"per-trajectory filter requires processed {name} to be emitted"
        )
    if summary is None or summary.count <= 0 or summary.mean is None:
        return None
    return _finite_context_float(summary.mean)


def _always_accept(
    params: FilterParams,
    context: PerTrajectoryFilterContext,
) -> bool:
    del params, context
    return True


def _grad_norm_max(
    params: FilterParams,
    context: PerTrajectoryFilterContext,
) -> bool:
    max_value = _finite_float("grad_norm_max", "max", params["max"])
    grad_norm = _finite_context_float(context.grad_norm)
    return grad_norm is not None and grad_norm <= max_value


def _kl_k1_range(
    params: FilterParams,
    context: PerTrajectoryFilterContext,
) -> bool:
    mean = _summary_mean(
        context.behave_approx_kl_summary,
        emitted=context.behave_approx_kl_mean_emitted,
        name="behave_approx_kl_mean",
    )
    if mean is None:
        return False

    lower = _optional_bound("kl_k1_range", params, "lower")
    upper = _optional_bound("kl_k1_range", params, "upper")
    if lower is not None and mean < lower:
        return False
    if upper is not None and mean > upper:
        return False
    return True


def _advantage_mean_positive(
    params: FilterParams,
    context: PerTrajectoryFilterContext,
) -> bool:
    del params
    mean = _summary_mean(
        context.advantage_summary,
        emitted=context.advantage_mean_emitted,
        name="advantage_mean",
    )
    return mean is not None and mean > 0.0


PER_TRAJECTORY_FILTER_RULES: dict[str, FilterFn] = {
    "none": _always_accept,
    "grad_norm_max": _grad_norm_max,
    "kl_k1_range": _kl_k1_range,
    "advantage_mean_positive": _advantage_mean_positive,
}

_VALIDATORS: dict[str, ValidatorFn] = {
    "none": _validate_none,
    "grad_norm_max": _validate_grad_norm_max,
    "kl_k1_range": _validate_kl_k1_range,
    "advantage_mean_positive": _validate_advantage_mean_positive,
}


def validate_per_trajectory_filter_config(config: Any) -> None:
    rule, params = _filter_rule_and_params(config)
    _VALIDATORS[rule](params)


def evaluate_per_trajectory_filters(
    filters: Sequence[Any],
    context: PerTrajectoryFilterContext,
) -> bool:
    for filter_config in filters:
        rule, params = _filter_rule_and_params(filter_config)
        _VALIDATORS[rule](params)
        if not PER_TRAJECTORY_FILTER_RULES[rule](params, context):
            return False
    return True
