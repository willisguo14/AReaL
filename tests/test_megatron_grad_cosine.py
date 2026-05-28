import math

import pytest
import torch
from torch import nn

from areal.engine.megatron_utils import grad_cosine
from areal.engine.megatron_utils.grad_cosine import (
    GradientCosineTracker,
    _collect_gradient_snapshot,
    _cosine_from_reduced_stats,
)


class _FakeOptimizer:
    def get_grad_stats_parallel_group(self):
        return None


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))


class _ManyParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.kept = nn.Parameter(torch.tensor([1.0]))
        self.shared = nn.Parameter(torch.tensor([1.0]))
        self.tp_duplicate = nn.Parameter(torch.tensor([1.0]))
        self.areal_duplicate = nn.Parameter(torch.tensor([1.0]))


@pytest.fixture(autouse=True)
def patch_megatron_helpers(monkeypatch):
    monkeypatch.setattr(grad_cosine, "_param_is_not_shared", lambda param: True)
    monkeypatch.setattr(
        grad_cosine,
        "_param_is_not_tensor_parallel_duplicate",
        lambda param: True,
    )
    monkeypatch.setattr(grad_cosine, "_get_tensor_model_parallel_rank", lambda: 0)


def _module_with_grad(values: list[float]) -> _TinyModule:
    module = _TinyModule()
    module.weight.grad = torch.tensor(values, dtype=torch.float32)
    return module


def _prepare_and_finalize(
    tracker: GradientCosineTracker,
    module: nn.Module,
    *,
    update_successful: bool = True,
) -> float | None:
    pending = tracker.prepare(
        model=[module],
        optimizer=_FakeOptimizer(),
        duplicated_param_names=set(),
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
    tracker = GradientCosineTracker()

    first = _prepare_and_finalize(tracker, _module_with_grad([1.0, 2.0]))
    second = _prepare_and_finalize(tracker, _module_with_grad([1.0, 2.0]))

    assert first is None
    assert second == pytest.approx(1.0)


def test_tracker_emits_signed_consecutive_cosines():
    tracker = GradientCosineTracker()

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
    tracker = GradientCosineTracker()

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
    tracker = GradientCosineTracker()

    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0])) is None
    assert _prepare_and_finalize(tracker, _module_with_grad([0.0, 0.0])) is None
    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0])) is None


def test_successful_step_with_no_eligible_gradients_clears_previous_snapshot(
    monkeypatch,
):
    tracker = GradientCosineTracker()

    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0])) is None
    monkeypatch.setattr(grad_cosine, "_param_is_not_shared", lambda param: False)
    assert _prepare_and_finalize(tracker, _module_with_grad([2.0, 0.0])) is None
    monkeypatch.setattr(grad_cosine, "_param_is_not_shared", lambda param: True)
    assert _prepare_and_finalize(tracker, _module_with_grad([1.0, 0.0])) is None


def test_collect_gradient_snapshot_prefers_main_grad():
    module = _TinyModule()
    module.weight.grad = torch.tensor([1.0, 1.0])
    module.weight.main_grad = torch.tensor([2.0, 3.0])

    snapshot = _collect_gradient_snapshot(
        model=[module],
        duplicated_param_names=set(),
    )

    assert snapshot.names == ("weight",)
    assert torch.equal(snapshot.grads[0], torch.tensor([2.0, 3.0]))


def test_collect_gradient_snapshot_filters_shared_and_tp_duplicates(monkeypatch):
    module = _ManyParams()
    for param in module.parameters():
        param.grad = torch.ones_like(param)

    monkeypatch.setattr(
        grad_cosine,
        "_param_is_not_shared",
        lambda param: param is not module.shared,
    )
    monkeypatch.setattr(
        grad_cosine,
        "_param_is_not_tensor_parallel_duplicate",
        lambda param: param is not module.tp_duplicate,
    )

    snapshot = _collect_gradient_snapshot(
        model=[module],
        duplicated_param_names=set(),
    )

    assert snapshot.names == ("kept", "areal_duplicate")


def test_collect_gradient_snapshot_counts_areal_duplicate_only_on_tp_rank_zero(
    monkeypatch,
):
    module = _ManyParams()
    for param in module.parameters():
        param.grad = torch.ones_like(param)

    monkeypatch.setattr(grad_cosine, "_get_tensor_model_parallel_rank", lambda: 1)

    snapshot = _collect_gradient_snapshot(
        model=[module],
        duplicated_param_names={"areal_duplicate"},
    )

    assert "areal_duplicate" not in snapshot.names
