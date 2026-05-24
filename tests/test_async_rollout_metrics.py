# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from unittest.mock import Mock, patch

from areal.infra.async_task_runner import TimedResult
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import BatchTaskDispatcher, WorkflowExecutor


class VersionProvider:
    def __init__(self, version: int = 0):
        self.version = version

    def get_version(self) -> int:
        return self.version


@dataclass
class TaskInput:
    task_id: int
    accepted: bool = True
    delay: float = 0.0


def make_manager(
    *,
    max_concurrent_rollouts: int = 10,
    consumer_batch_size: int = 4,
    max_staleness: int = 2,
) -> StalenessManager:
    return StalenessManager(
        version_provider=VersionProvider(),
        max_concurrent_rollouts=max_concurrent_rollouts,
        consumer_batch_size=consumer_batch_size,
        max_staleness=max_staleness,
    )


def make_dispatcher(manager: StalenessManager) -> BatchTaskDispatcher:
    def task_factory(task_input: TaskInput):
        async def run():
            await asyncio.sleep(task_input.delay)
            if task_input.accepted:
                manager.on_rollout_accepted()
                return {"task_id": task_input.task_id}
            manager.on_rollout_rejected()
            return None

        return run

    return BatchTaskDispatcher(
        max_queue_size=16,
        task_factory=task_factory,
        staleness_manager=manager,
    )


def test_active_submit_and_wait_exports_snapshot_from_train_request_start():
    manager = make_manager(max_concurrent_rollouts=10, consumer_batch_size=4)
    dispatcher = make_dispatcher(manager)

    dispatcher._pending_results = {
        1: TimedResult(create_time=1, data={"task_id": 1}, task_id=1),
        2: TimedResult(create_time=2, data={"task_id": 2}, task_id=2),
    }

    results = dispatcher.active_submit_and_wait(
        iter([TaskInput(task_id=3)]),
        batch_size=1,
    )

    assert results == [{"task_id": 1}]
    assert len(dispatcher._pending_results) == 1

    stats = dispatcher.export_async_metrics()
    assert stats["async/train_request/ready_tasks"] == 2
    assert stats["async/train_request/inflight_tasks"] == 0
    assert stats["async/train_request/capacity_tasks"] == 10
    assert stats["async/train_request/accepted_tasks"] == 0
    assert stats["async/train_request/rejected_tasks"] == 0
    assert "async/train_request/task_latency/p50" not in stats


def test_train_request_metrics_use_deltas_and_latency_window():
    manager = make_manager(max_concurrent_rollouts=10, consumer_batch_size=8)
    dispatcher = make_dispatcher(manager)
    dispatcher.initialize(logger=Mock())

    try:
        dispatcher.submit_task_input(TaskInput(task_id=1, accepted=True, delay=0.01))
        dispatcher.submit_task_input(TaskInput(task_id=2, accepted=True, delay=0.02))
        dispatcher.submit_task_input(TaskInput(task_id=3, accepted=False, delay=0.03))

        results = dispatcher.wait_results(count=3, timeout=5.0)
        assert len(results) == 3

        first = dispatcher.capture_train_request_metrics()
        assert first["async/train_request/accepted_tasks"] == 2
        assert first["async/train_request/rejected_tasks"] == 1
        assert first["async/train_request/ready_tasks"] == 0
        assert first["async/train_request/inflight_tasks"] == 0
        assert first["async/train_request/capacity_tasks"] == 10
        assert (
            0
            < first["async/train_request/task_latency/p50"]
            <= first["async/train_request/task_latency/p95"]
            <= first["async/train_request/task_latency/max"]
        )

        second = dispatcher.capture_train_request_metrics()
        assert second["async/train_request/accepted_tasks"] == 0
        assert second["async/train_request/rejected_tasks"] == 0
        assert "async/train_request/task_latency/p50" not in second
    finally:
        dispatcher.destroy()


def test_workflow_executor_delegates_async_metric_export_to_dispatcher():
    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    executor._dispatcher = Mock()
    executor._dispatcher.export_async_metrics.return_value = {
        "async/train_request/ready_tasks": 4,
    }

    assert executor.export_async_metrics() == {
        "async/train_request/ready_tasks": 4,
    }


def test_rollout_controller_export_stats_includes_controller_async_metrics():
    controller = RolloutController.__new__(RolloutController)
    controller._dispatcher = Mock()
    controller._dispatcher.export_async_metrics.return_value = {
        "async/train_request/ready_tasks": 7,
    }
    controller._collective_rpc = Mock(
        return_value=[
            {"rollout/reward": 2.0, "rollout/reward__count": 2},
            {"rollout/reward": 4.0, "rollout/reward__count": 2},
        ]
    )

    stats = RolloutController.export_stats(controller)

    assert stats["rollout/reward"] == 3.0
    assert stats["async/train_request/ready_tasks"] == 7


def test_remote_engine_wrappers_export_async_metrics():
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.engine.vllm_remote import RemotevLLMEngine

    for engine_cls, module_name in (
        (RemoteSGLangEngine, "areal.engine.sglang_remote"),
        (RemotevLLMEngine, "areal.engine.vllm_remote"),
    ):
        engine = engine_cls.__new__(engine_cls)
        engine._engine = Mock()
        engine._engine.workflow_executor.export_async_metrics.return_value = {
            "async/train_request/ready_tasks": 5,
        }

        with patch(f"{module_name}.stats_tracker.export_all", return_value={}):
            assert engine_cls.export_stats(engine) == {
                "async/train_request/ready_tasks": 5,
            }
