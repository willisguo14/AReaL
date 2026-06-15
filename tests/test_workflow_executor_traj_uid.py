import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from areal.api import RolloutWorkflow
from areal.infra.workflow_executor import WorkflowExecutor, ensure_trajectory_uids


def _trajectory(num_samples: int = 2) -> dict:
    return {
        "input_ids": torch.ones(num_samples, 4, dtype=torch.long),
        "attention_mask": torch.ones(num_samples, 4, dtype=torch.bool),
        "loss_mask": torch.ones(num_samples, 4, dtype=torch.long),
        "rewards": torch.arange(num_samples, dtype=torch.float32),
        "versions": torch.zeros(num_samples, 4, dtype=torch.long),
    }


def test_ensure_trajectory_uids_generates_task_sample_ids():
    traj = _trajectory(num_samples=3)

    ensure_trajectory_uids(traj, task_id=42)

    assert traj["traj_uid"] == ["42:0", "42:1", "42:2"]


def test_ensure_trajectory_uids_preserves_valid_custom_ids():
    traj = _trajectory(num_samples=2)
    traj["traj_uid"] = ["custom-a", "custom-b"]

    ensure_trajectory_uids(traj, task_id=42)

    assert traj["traj_uid"] == ["custom-a", "custom-b"]


def test_ensure_trajectory_uids_replaces_misaligned_custom_ids():
    traj = _trajectory(num_samples=2)
    traj["traj_uid"] = ["only-one"]

    ensure_trajectory_uids(traj, task_id=42)

    assert traj["traj_uid"] == ["42:0", "42:1"]


class _StaticWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        return _trajectory(num_samples=2)


def test_create_workflow_task_attaches_uids_before_return():
    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    executor.inference_engine = object()
    executor.config = type(
        "Config",
        (),
        {
            "check_trajectory_format": False,
            "dump_to_file": False,
            "enable_rollout_tracing": False,
        },
    )()
    executor.logger = type(
        "Logger",
        (),
        {"info": lambda *args, **kwargs: None, "warning": lambda *args, **kwargs: None},
    )()
    executor._expected_trajectory_keys = None

    class _Manager:
        def on_rollout_accepted(self):
            self.accepted = True

        def on_rollout_rejected(self):
            self.rejected = True

    executor._staleness_manager = _Manager()

    pending = type(
        "Pending",
        (),
        {
            "task_id": 77,
            "data": {},
            "workflow": _StaticWorkflow(),
            "should_accept_fn": None,
            "is_eval": False,
        },
    )()

    task = executor._create_workflow_task(pending)
    result = asyncio.run(task())

    assert result.trajectory["traj_uid"] == ["77:0", "77:1"]


def test_create_workflow_task_adds_uids_before_format_check():
    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    executor.inference_engine = object()
    executor.config = type(
        "Config",
        (),
        {
            "check_trajectory_format": True,
            "dump_to_file": False,
            "enable_rollout_tracing": False,
        },
    )()
    executor.logger = type(
        "Logger",
        (),
        {
            "info": lambda *args, **kwargs: None,
            "warning": lambda *args, **kwargs: None,
            "error": lambda *args, **kwargs: None,
        },
    )()
    executor._expected_trajectory_keys = set(_trajectory(num_samples=2).keys()) | {
        "traj_uid"
    }

    class _Manager:
        def on_rollout_accepted(self):
            self.accepted = True

        def on_rollout_rejected(self):
            self.rejected = True

    executor._staleness_manager = _Manager()

    pending = type(
        "Pending",
        (),
        {
            "task_id": 88,
            "data": {},
            "workflow": _StaticWorkflow(),
            "should_accept_fn": None,
            "is_eval": False,
        },
    )()

    task = executor._create_workflow_task(pending)
    result = asyncio.run(task())

    assert result is not None
    assert result.trajectory["traj_uid"] == ["88:0", "88:1"]


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


@pytest.mark.asyncio
async def test_dump_trajectory_writes_traj_uid(tmp_path):
    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    executor.config = SimpleNamespace()
    executor.inference_engine = SimpleNamespace(get_version=lambda: 0)
    executor.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    executor._get_dump_dir = lambda is_eval: str(tmp_path)
    executor._get_tokenizer = lambda: _Tokenizer()

    traj = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.long),
        "rewards": torch.tensor([1.0, 0.0]),
        "versions": torch.zeros(2, 4, dtype=torch.long),
    }

    success, reason = await executor._dump_trajectory(traj, task_id=9, is_eval=False)

    assert success, reason
    lines = (tmp_path / "0" / "9.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["traj_uid"] for record in records] == ["9:0", "9:1"]
