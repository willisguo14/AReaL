# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import getpass
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_BATCH_METADATA_KEYS = frozenset({"ids", "traj_uid", "uid", "rid", "task_id"})
_BATCH_1D_TENSOR_KEYS = frozenset(
    {
        "begin_of_trajectory",
        "kl_rewards",
        "rewards",
        "rid",
        "task_id",
        "task_reward",
        "tot_rewards",
        "trainer_global_step",
        "traj_uid",
        "uid",
        "versions",
    }
)


@dataclass(frozen=True)
class LogprobSummary:
    sum: float
    mean: float
    length: int


@dataclass(frozen=True)
class MaskedTensorSummary:
    min: float | None
    max: float | None
    mean: float | None
    count: int


class MaskedTensorAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0
        self.min: float | None = None
        self.max: float | None = None

    def add(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        if values.shape != mask.shape:
            raise ValueError(
                "masked tensor summary shape mismatch: "
                f"values shape {tuple(values.shape)} vs mask shape {tuple(mask.shape)}"
            )
        selected = values[mask.bool()].detach().to(dtype=torch.float32)
        if selected.numel() == 0:
            return

        self.total += float(selected.sum().item())
        self.count += int(selected.numel())
        min_value = float(selected.min().item())
        max_value = float(selected.max().item())
        self.min = min_value if self.min is None else min(self.min, min_value)
        self.max = max_value if self.max is None else max(self.max, max_value)

    def summary(self) -> MaskedTensorSummary:
        if self.count == 0:
            return MaskedTensorSummary(min=None, max=None, mean=None, count=0)
        return MaskedTensorSummary(
            min=self.min,
            max=self.max,
            mean=self.total / self.count,
            count=self.count,
        )


@dataclass(frozen=True)
class PerTrajectoryRecord:
    trainer_global_step: int
    minibatch_idx: int
    trajectory_idx: int
    trajectory_id: str
    grad_norm: float
    logprob_train_sum: float
    logprob_train_mean: float
    logprob_infer_sum: float
    logprob_infer_mean: float
    reward: float
    response_length: int
    behave_imp_weight_min: float | None = None
    behave_imp_weight_max: float | None = None
    behave_imp_weight_mean: float | None = None
    behave_approx_kl_min: float | None = None
    behave_approx_kl_max: float | None = None
    behave_approx_kl_mean: float | None = None


def normalize_flush_threshold(value: Any) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError, OverflowError):
        return 1


class PerTrajectoryTracer:
    """Buffered JSONL trace writer.

    This class assumes a single writer per path. Distributed callers should
    choose rank-specific paths before constructing tracers.
    """

    def __init__(
        self,
        path: str | Path,
        flush_threshold: Any = 1,
        enabled: bool = True,
    ) -> None:
        self.path = Path(path)
        self.flush_threshold = normalize_flush_threshold(flush_threshold)
        self.enabled = enabled
        self._buffer: list[PerTrajectoryRecord] = []

    def write(self, record: PerTrajectoryRecord) -> None:
        if not self.enabled:
            return
        self._buffer.append(record)
        if len(self._buffer) >= self.flush_threshold:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self._buffer:
            return

        lines = [
            json.dumps(
                dataclasses.asdict(record),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
            for record in self._buffer
        ]

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.writelines(lines)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()


def per_trajectory_log_dir(
    fileroot: str | Path,
    experiment_name: str,
    trial_name: str,
) -> Path:
    return (
        Path(fileroot)
        / "logs"
        / getpass.getuser()
        / str(experiment_name)
        / str(trial_name)
        / "per_trajectory"
        / "actor"
    )


def summarize_logprobs(
    logprobs: torch.Tensor,
    response_mask: torch.Tensor,
) -> LogprobSummary:
    selected = logprobs[response_mask.bool()].detach().to(dtype=torch.float32)
    response_length = int(selected.numel())
    if response_length == 0:
        raise ValueError("response_length must be positive")

    return LogprobSummary(
        sum=float(selected.sum().item()),
        mean=float(selected.mean().item()),
        length=response_length,
    )


def preserve_rollout_logprobs(batch: list[dict[str, Any]]) -> None:
    from areal.infra.rpc.rtensor import RTensor

    for trajectory in batch:
        logprobs = trajectory.get("logprobs")
        if "rollout_logprobs" not in trajectory:
            if isinstance(logprobs, torch.Tensor):
                trajectory["rollout_logprobs"] = logprobs.detach().clone()
            elif isinstance(logprobs, RTensor):
                trajectory["rollout_logprobs"] = logprobs

        loss_mask = trajectory.get("loss_mask")
        if "rollout_loss_mask" not in trajectory:
            if isinstance(loss_mask, torch.Tensor):
                trajectory["rollout_loss_mask"] = loss_mask.detach().clone()
            elif isinstance(loss_mask, RTensor):
                trajectory["rollout_loss_mask"] = loss_mask


def attach_trainer_step_metadata(
    batch: list[dict[str, Any]],
    global_step: int,
    fileroot: str | Path,
) -> None:
    for trajectory in batch:
        trajectory["trainer_global_step"] = torch.tensor(
            [int(global_step)],
            dtype=torch.long,
        )
        trajectory["trainer_fileroot"] = str(fileroot)


def _first_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.detach().flatten()[0].item()
    if isinstance(value, list | tuple):
        if not value:
            return None
        return _first_scalar(value[0])
    return value


def trajectory_id_from_sample(sample: dict[str, Any], fallback: str) -> str:
    for key in ("traj_uid", "uid", "rid", "task_id"):
        if key not in sample:
            continue
        value = _first_scalar(sample[key])
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return fallback


def slice_trajectory(
    batch: dict[str, Any],
    index: int,
    *,
    extra_batch_keys: set[str] | None = None,
) -> dict[str, Any]:
    if "attention_mask" not in batch:
        raise KeyError("attention_mask is required to infer batch size")

    attention_mask = batch["attention_mask"]
    if not isinstance(attention_mask, torch.Tensor):
        raise ValueError("attention_mask must be a tensor")
    if attention_mask.ndim < 2:
        raise ValueError("attention_mask must be at least 2D")

    batch_size = int(attention_mask.shape[0])
    if batch_size <= 0:
        raise ValueError("attention_mask must have a positive batch dimension")

    extra_keys = frozenset(extra_batch_keys or ())
    tensor_1d_keys = _BATCH_1D_TENSOR_KEYS | extra_keys
    list_keys = _BATCH_METADATA_KEYS | extra_keys

    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == batch_size
            and (value.ndim > 1 or key in tensor_1d_keys)
        ):
            sliced[key] = value[index].unsqueeze(0)
        elif (
            isinstance(value, list)
            and (key in list_keys or key.startswith("multi_modal_input"))
            and len(value) == batch_size
        ):
            sliced[key] = [value[index]]
        else:
            sliced[key] = value
    return sliced
