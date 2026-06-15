# Per-Trajectory Grad-Norm Filtering Design

## Summary

Add optional fixed-threshold grad-norm filtering to the existing Megatron actor
per-trajectory training path. When enabled, AReaL still runs each trajectory's
forward/backward pass to measure its exact gradient norm, but trajectories whose
per-trajectory grad norm exceeds the configured threshold are not added to the
accumulated gradient buffer used for the optimizer step.

This is post-backward gradient dropping. It is intentionally separate from
`actor.rejection_sampling`, which is pre-backward loss masking based on policy
divergence.

## Goals

- Support fixed absolute grad-norm filtering only for `actor.per_trajectory`.
- Keep the implementation localized to the existing per-trajectory Megatron path.
- Preserve current behavior when the new threshold is unset.
- Preserve the existing full-minibatch/global loss denominator.
- Add enough observability to understand which trajectories were filtered.

## Non-Goals

- Do not add grad-norm filtering to the normal microbatch training path.
- Do not merge grad-norm filtering into `RejectionSamplingConfig`.
- Do not implement percentile, top-k, adaptive, or batch-relative thresholds.
- Do not try to save forward/backward compute for grad-norm filtering, because the grad
  norm is only known after backward.
- Do not compute a counterfactual grad norm for trajectories already fully masked by
  rejection sampling.

## Configuration

Extend `PerTrajectoryConfig` with:

```python
max_grad_norm: float | None = None
```

Semantics:

- `None`: disabled, current behavior.
- positive float: drop any trajectory whose recorded per-trajectory `grad_norm` is
  greater than this threshold.
- values less than or equal to zero are invalid.

The field belongs under `actor.per_trajectory`, for example:

```yaml
actor:
  per_trajectory:
    enabled: true
    max_grad_norm: 10.0
```

## Data Flow

Current per-trajectory path:

1. Build `full_mb_list` and compute `total_loss_weight` over the full minibatch using
   the existing DP(+CP) normalization group. The per-trajectory path currently requires
   context parallel size 1, so this is effectively DP normalization in supported
   configurations.
1. For each trajectory:
   - slice one trajectory,
   - run forward/backward,
   - compute `trace_grad_norm`,
   - accumulate current grad buffers into `accum_buffers`,
   - write JSONL trace record,
   - clear current grad buffers.
1. Copy `accum_buffers` back to model grad buffers.
1. Run one optimizer step.

New path:

1. Keep the same full-minibatch `total_loss_weight`.
1. For each trajectory:
   - slice one trajectory,
   - run forward/backward,
   - compute `trace_grad_norm`,
   - set `grad_norm_filtered = trace_grad_norm > max_grad_norm` when enabled,
   - accumulate current grad buffers only when `grad_norm_filtered` is false,
   - write JSONL trace record with `grad_norm_filtered`,
   - clear current grad buffers.
1. Copy only kept accumulated grads back to model grad buffers.
1. Run one optimizer step.

The optimizer update becomes:

```text
G = sum_i keep_i * grad_i
keep_i = grad_norm_i <= max_grad_norm
```

Each `grad_i` remains normalized by the existing full-minibatch/global denominator.
Filtering changes the numerator by removing selected trajectory gradients; it does not
change the denominator.

## Rejection Sampling Interaction

Rejection sampling still runs inside the loss before the per-trajectory grad norm is
measured. Therefore the grad norm filter sees the actual gradient produced by the final
trajectory loss after rejection sampling has updated the loss mask or behavioral
importance weights.

If sequence-level rejection sampling fully masks a trajectory in the plain PPO/GRPO
actor loss, the scalar PPO loss is zero and the resulting PPO gradient norm is zero or
near zero. The grad-norm filter will usually keep that trajectory, but it contributes
zero to the update either way.

Auxiliary losses, such as teacher/distillation losses, may produce a nonzero total
gradient even when PPO rejection sampling masks the PPO term. This feature filters the
total trajectory gradient produced by the current loss function.

## Observability

Add one JSONL field to `PerTrajectoryRecord`:

```python
grad_norm_filtered: bool = False
```

Add scalar training stats only when grad-norm filtering is enabled:

- `grad_norm_filter_count`: number of trajectories dropped in the per-trajectory
  minibatch call.
- `grad_norm_filter_fraction`: dropped trajectories divided by total trajectories in
  that per-trajectory minibatch call.

These metrics are separate from existing rejection-sampling metrics such as
`rs_filtered_fraction`.

## Correctness Requirements

- With `max_grad_norm=None`, behavior and trace output remain backward-compatible except
  for the added JSONL field defaulting to `false`.
- A trajectory with `trace_grad_norm <= max_grad_norm` is accumulated exactly as it is
  today.
- A trajectory with `trace_grad_norm > max_grad_norm` is traced but not accumulated.
- The final optimizer step runs once after all trajectories, as it does today.
- The global denominator from `compute_total_loss_weight(...)` is unchanged.
- Filtering must use the same `trace_grad_norm` value written as `grad_norm` in the
  per-trajectory JSONL record.

## Test Plan

- Config validation rejects non-positive `max_grad_norm`.
- Existing per-trajectory tests pass when `max_grad_norm` is unset.
- Unit test a mixed batch where one trajectory is below threshold and one is above: only
  the kept trajectory affects gradient accumulation.
- JSONL records include `grad_norm_filtered` with `false` for kept trajectories and
  `true` for dropped trajectories.
- Returned train stats include `grad_norm_filter_count` and `grad_norm_filter_fraction`
  only when filtering is enabled.
