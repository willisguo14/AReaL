# SPDX-License-Identifier: Apache-2.0

"""Core utilities for training engines."""

from areal.engine.core.per_trajectory import (
    LogprobSummary,
    MaskedTensorAccumulator,
    MaskedTensorSummary,
    PerTrajectoryRecord,
    PerTrajectoryTracer,
    attach_trainer_step_metadata,
    normalize_flush_threshold,
    per_trajectory_log_dir,
    preserve_rollout_logprobs,
    slice_trajectory,
    summarize_logprobs,
    trajectory_id_from_sample,
)
from areal.engine.core.train_engine import (
    LOGP_GRAD_ABSMAX_KEY,
    LOGP_GRAD_NORM_KEY,
    LogprobGradAccumulator,
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)

__all__ = [
    "aggregate_eval_losses",
    "compute_total_loss_weight",
    "LOGP_GRAD_ABSMAX_KEY",
    "LOGP_GRAD_NORM_KEY",
    "LogprobGradAccumulator",
    "LogprobSummary",
    "MaskedTensorAccumulator",
    "MaskedTensorSummary",
    "PerTrajectoryRecord",
    "PerTrajectoryTracer",
    "attach_trainer_step_metadata",
    "normalize_flush_threshold",
    "per_trajectory_log_dir",
    "preserve_rollout_logprobs",
    "reorder_and_pad_outputs",
    "slice_trajectory",
    "summarize_logprobs",
    "trajectory_id_from_sample",
]
