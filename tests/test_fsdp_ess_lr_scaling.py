# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import areal.engine.fsdp_engine as fsdp_engine_module
from areal.engine.fsdp_engine import FSDPEngine


class _FakeOptimizer:
    def __init__(self, lr: float = 0.01, *, raise_on_step: bool = False):
        self.param_groups = [{"lr": lr}]
        self.raise_on_step = raise_on_step
        self.step_lrs: list[float] = []
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self):
        self.step_calls += 1
        self.step_lrs.append(self.param_groups[0]["lr"])
        if self.raise_on_step:
            raise RuntimeError("step failed")

    def zero_grad(self):
        self.zero_grad_calls += 1


class _FakeMBList:
    def __init__(self):
        self.mbs = [{"loss_mask": torch.ones(1, dtype=torch.bool)}]

    def to(self, device):
        return self


def _make_optimizer_step_engine(optimizer: _FakeOptimizer) -> FSDPEngine:
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.optimizer = optimizer
    engine.optimizer_config = SimpleNamespace(gradient_clipping=1.0)
    engine.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [0.01])
    engine.model = SimpleNamespace(parameters=lambda: [])
    engine.world_mesh = {
        "dp_sp": SimpleNamespace(get_group=lambda: "dp_sp_group"),
        "tp": SimpleNamespace(get_group=lambda: "tp_group"),
    }
    engine.config = SimpleNamespace(
        fsdp=SimpleNamespace(offload_params=False, per_layer_optim_step=False)
    )
    return engine


@pytest.fixture
def finite_grad_norm(monkeypatch):
    monkeypatch.setattr(
        fsdp_engine_module,
        "fsdp2_clip_grad_norm",
        lambda *args, **kwargs: 1.0,
    )


def test_fsdp_engine_declares_optimizer_step_scale_support():
    assert FSDPEngine.supports_optimizer_step_scale is True


def test_optimizer_step_scales_lr_only_during_step(finite_grad_norm):
    optimizer = _FakeOptimizer(lr=0.01)
    engine = _make_optimizer_step_engine(optimizer)

    stats = FSDPEngine.optimizer_step(engine, optimizer_step_scale=0.25)

    assert optimizer.step_calls == 1
    assert optimizer.step_lrs == [pytest.approx(0.0025)]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert stats["update_successful"] == 1.0
    assert stats["lr"] == pytest.approx(0.01)


def test_optimizer_step_restores_lr_when_step_raises(finite_grad_norm):
    optimizer = _FakeOptimizer(lr=0.01, raise_on_step=True)
    engine = _make_optimizer_step_engine(optimizer)

    with pytest.raises(RuntimeError, match="step failed"):
        FSDPEngine.optimizer_step(engine, optimizer_step_scale=0.25)

    assert optimizer.step_calls == 1
    assert optimizer.step_lrs == [pytest.approx(0.0025)]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_optimizer_step_does_not_scale_when_grad_norm_nonfinite(monkeypatch):
    monkeypatch.setattr(
        fsdp_engine_module,
        "fsdp2_clip_grad_norm",
        lambda *args, **kwargs: float("nan"),
    )
    optimizer = _FakeOptimizer(lr=0.01)
    engine = _make_optimizer_step_engine(optimizer)

    stats = FSDPEngine.optimizer_step(engine, optimizer_step_scale=0.25)

    assert optimizer.step_calls == 0
    assert optimizer.step_lrs == []
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert stats["update_successful"] == 0.0


def test_train_batch_passes_optimizer_step_scale(monkeypatch):
    events = []
    received_optimizer_step_scales = []
    engine = FSDPEngine.__new__(FSDPEngine)
    fake_mb_list = _FakeMBList()

    class _NoopTracker:
        def prepare(self, **kwargs):
            events.append("prepare")
            return "pending"

        def finalize(self, pending, *, update_successful):
            events.append("finalize")
            assert pending == "pending"
            assert update_successful is True
            return None

    engine.grad_cosine_tracker = _NoopTracker()
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

    def fake_optimizer_step(*, optimizer_step_scale=1.0):
        events.append("optimizer_step")
        received_optimizer_step_scales.append(optimizer_step_scale)
        return {
            "update_successful": 1.0,
            "grad_norm": 1.0,
            "lr": 0.01,
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
        optimizer_step_scale=0.4,
    )

    assert events == [
        "ensure",
        "zero_grad",
        "forward_backward",
        "prepare",
        "optimizer_step",
        "finalize",
    ]
    assert received_optimizer_step_scales == [0.4]
    assert stats["num_micro_batches"] == 1
