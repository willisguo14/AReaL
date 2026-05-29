import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import areal.engine.fsdp_engine as fsdp_engine_module
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.fsdp_utils import grad_cosine
from areal.engine.fsdp_utils.grad_cosine import (
    FSDPGradientCosineTracker,
    _collect_gradient_snapshot,
    _cosine_from_reduced_stats,
)


class _TinyModule(nn.Module):
    def __init__(self, size: int = 2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))


class _ManyParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.kept = nn.Parameter(torch.tensor([1.0]))
        self.tp_duplicate = nn.Parameter(torch.tensor([1.0]))
        self.no_grad = nn.Parameter(torch.tensor([1.0]))


class _FakeMBList:
    def __init__(self):
        self.mbs = [{"loss_mask": torch.ones(1, dtype=torch.bool)}]

    def to(self, device):
        return self


@pytest.fixture(autouse=True)
def patch_distributed_helpers(monkeypatch):
    monkeypatch.setattr(grad_cosine, "_get_tensor_parallel_rank", lambda tp_group: 0)
    monkeypatch.setattr(
        grad_cosine,
        "_all_reduce_stats",
        lambda stats, groups: stats,
    )


def _module_with_grad(values: list[float]) -> _TinyModule:
    module = _TinyModule(size=len(values))
    module.weight.grad = torch.tensor(values, dtype=torch.float32)
    return module


def _prepare_and_finalize(
    tracker: FSDPGradientCosineTracker,
    module: nn.Module,
    *,
    update_successful: bool = True,
) -> float | None:
    pending = tracker.prepare(
        model=module,
        fsdp_group=None,
        tp_group=None,
        device=torch.device("cpu"),
    )
    return tracker.finalize(pending, update_successful=update_successful)


def test_cosine_from_reduced_stats_exact_values():
    assert _cosine_from_reduced_stats(
        dot=0.0,
        current_norm_sq=1.0,
        previous_norm_sq=1.0,
        missing_previous_count=0.0,
        mismatch_count=0.0,
        eps=1e-8,
    ) == pytest.approx(0.0)
    assert _cosine_from_reduced_stats(
        dot=-2.0,
        current_norm_sq=4.0,
        previous_norm_sq=1.0,
        missing_previous_count=0.0,
        mismatch_count=0.0,
        eps=1e-8,
    ) == pytest.approx(-1.0)
    assert _cosine_from_reduced_stats(
        dot=11.0,
        current_norm_sq=25.0,
        previous_norm_sq=5.0,
        missing_previous_count=0.0,
        mismatch_count=0.0,
        eps=1e-8,
    ) == pytest.approx(11.0 / math.sqrt(25.0 * 5.0))


@pytest.mark.parametrize(
    "dot,current_norm_sq,previous_norm_sq,missing_previous_count,mismatch_count",
    [
        (1.0, 0.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0, 0.0),
        (float("nan"), 1.0, 1.0, 0.0, 0.0),
        (1.0, float("inf"), 1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 0.0, 1.0),
    ],
)
def test_cosine_from_reduced_stats_omits_invalid_inputs(
    dot,
    current_norm_sq,
    previous_norm_sq,
    missing_previous_count,
    mismatch_count,
):
    assert (
        _cosine_from_reduced_stats(
            dot=dot,
            current_norm_sq=current_norm_sq,
            previous_norm_sq=previous_norm_sq,
            missing_previous_count=missing_previous_count,
            mismatch_count=mismatch_count,
            eps=1e-8,
        )
        is None
    )


def test_first_successful_step_stores_snapshot_without_emitting():
    tracker = FSDPGradientCosineTracker()

    first = _prepare_and_finalize(tracker, _module_with_grad([1.0, 2.0]))
    second = _prepare_and_finalize(tracker, _module_with_grad([1.0, 2.0]))

    assert first is None
    assert second == pytest.approx(1.0)


def test_tracker_emits_signed_consecutive_cosines():
    tracker = FSDPGradientCosineTracker()

    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0])) is None
    assert _prepare_and_finalize(
        tracker,
        _module_with_grad([0.0, 2.0]),
    ) == pytest.approx(0.0)
    assert _prepare_and_finalize(
        tracker,
        _module_with_grad([0.0, -3.0]),
    ) == pytest.approx(-1.0)


def test_unsuccessful_step_does_not_replace_previous_snapshot():
    tracker = FSDPGradientCosineTracker()

    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0])) is None
    assert (
        _prepare_and_finalize(
            tracker,
            _module_with_grad([0.0, 1.0]),
            update_successful=False,
        )
        is None
    )
    assert _prepare_and_finalize(
        tracker,
        _module_with_grad([1.0, 0.0]),
    ) == pytest.approx(1.0)


def test_successful_zero_gradient_clears_previous_snapshot():
    tracker = FSDPGradientCosineTracker()

    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0])) is None
    assert _prepare_and_finalize(tracker, _module_with_grad([0.0, 0.0])) is None
    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0])) is None


def test_snapshot_mismatch_omits_once_then_uses_new_snapshot():
    tracker = FSDPGradientCosineTracker()

    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0])) is None
    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0, 0.0])) is None
    assert _prepare_and_finalize(
        tracker,
        _module_with_grad([2.0, 0.0, 0.0]),
    ) == pytest.approx(1.0)


def test_collect_gradient_snapshot_filters_tp_duplicates(monkeypatch):
    module = _ManyParams()
    module.kept.grad = torch.ones_like(module.kept)
    module.tp_duplicate.grad = torch.ones_like(module.tp_duplicate)

    monkeypatch.setattr(
        grad_cosine,
        "is_param_not_tensor_parallel_duplicate",
        lambda param, tensor_parallel_rank: param is not module.tp_duplicate,
    )

    snapshot = _collect_gradient_snapshot(
        module.named_parameters(),
        tensor_parallel_rank=1,
    )

    assert snapshot.names == ("kept",)


def test_collect_gradient_snapshot_uses_local_grad_conversion(monkeypatch):
    module = _TinyModule(size=1)
    module.weight.grad = torch.tensor([1.0])
    local_grad = torch.tensor([3.0])
    calls = []

    def fake_to_local_if_dtensor(tensor):
        calls.append(tensor)
        return local_grad

    monkeypatch.setattr(grad_cosine, "to_local_if_dtensor", fake_to_local_if_dtensor)

    snapshot = _collect_gradient_snapshot(
        module.named_parameters(),
        tensor_parallel_rank=0,
    )
    local_grad.fill_(9.0)

    assert calls == [module.weight.grad]
    assert snapshot.names == ("weight",)
    assert torch.equal(snapshot.grads[0], torch.tensor([3.0]))


@pytest.mark.parametrize(
    ("update_successful", "expected_cosine"),
    [(1.0, 0.25), (0.0, None)],
)
def test_fsdp_train_batch_finalizes_tracker_after_optimizer_step(
    monkeypatch,
    update_successful,
    expected_cosine,
):
    events = []
    engine = FSDPEngine.__new__(FSDPEngine)
    fake_mb_list = _FakeMBList()

    class _RecordingTracker:
        def prepare(self, **kwargs):
            events.append("prepare")
            assert kwargs["model"] is engine.model
            assert kwargs["fsdp_group"] == "dp_sp_group"
            assert kwargs["tp_group"] == "tp_group"
            assert kwargs["device"] == torch.device("cpu")
            return "pending"

        def finalize(self, pending, *, update_successful):
            events.append("finalize")
            assert pending == "pending"
            return 0.25 if update_successful else None

    engine.grad_cosine_tracker = _RecordingTracker()
    engine.model = object()
    engine.device = torch.device("cpu")
    engine.parallel_helper = SimpleNamespace(dp_size=1)
    engine.dp_group = "dp_group"
    engine.world_mesh = {
        "dp_sp": SimpleNamespace(get_group=lambda: "dp_sp_group"),
        "tp": SimpleNamespace(get_group=lambda: "tp_group"),
    }
    engine._ensure_ready = lambda: events.append("ensure")
    engine.optimizer_zero_grad = lambda: events.append("zero_grad")
    engine._normalize_batch_input = lambda input_: (input_, None)
    engine._prepare_mb_list = lambda input_: fake_mb_list

    def fake_forward_backward_batch(mb_list, process_output, forward_only=False):
        events.append("forward_backward")
        assert mb_list is fake_mb_list
        assert forward_only is False

    def fake_optimizer_step():
        events.append("optimizer_step")
        return {
            "update_successful": update_successful,
            "grad_norm": 1.0,
            "lr": 1e-6,
        }

    engine.forward_backward_batch = fake_forward_backward_batch
    engine.optimizer_step = fake_optimizer_step
    monkeypatch.setattr(
        fsdp_engine_module,
        "compute_total_loss_weight",
        lambda mb_list, loss_weight_fn, dp_group: torch.tensor(1.0),
    )

    stats = FSDPEngine.train_batch(
        engine,
        {"loss_mask": torch.ones(1, dtype=torch.bool)},
        loss_fn=lambda *args: torch.tensor(0.0),
        loss_weight_fn=lambda x: torch.tensor(1),
    )

    assert events == [
        "ensure",
        "zero_grad",
        "forward_backward",
        "prepare",
        "optimizer_step",
        "finalize",
    ]
    assert stats["num_micro_batches"] == 1
    if expected_cosine is None:
        assert "grad_cos_sim" not in stats
    else:
        assert stats["grad_cos_sim"] == pytest.approx(expected_cosine)
