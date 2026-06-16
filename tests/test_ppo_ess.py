# SPDX-License-Identifier: Apache-2.0

import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import ESSScalingConfig
from areal.trainer.ppo import actor as actor_module
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.ess import (
    ESSStats,
    compute_sequence_behavior_metrics,
    compute_sequence_ess,
    compute_token_ess,
    summarize_ess_lr_scale_stats,
    summarize_ess_stats,
)


def test_compute_sequence_ess_uniform_weights():
    logprobs = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    prox_logp = logprobs.clone()
    loss_mask = torch.ones_like(logprobs)

    stats = compute_sequence_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    assert stats is not None
    assert stats.ess == pytest.approx(2.0)
    assert stats.ess_ratio == pytest.approx(1.0)
    assert stats.ess_lr_scale == pytest.approx(1.0)
    assert stats.valid_count == 2


def test_compute_token_ess_uniform_weights():
    logprobs = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    prox_logp = logprobs.clone()
    loss_mask = torch.ones_like(logprobs)

    stats = compute_token_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    assert stats is not None
    assert stats.ess == pytest.approx(4.0)
    assert stats.ess_ratio == pytest.approx(1.0)
    assert stats.ess_lr_scale == pytest.approx(1.0)
    assert stats.valid_count == 4


def test_compute_sequence_ess_rejects_mismatched_tensor_shapes():
    prox_logp = torch.zeros((2, 2))
    logprobs = torch.zeros((2, 3))
    loss_mask = torch.ones((2, 2))

    with pytest.raises(ValueError, match="prox_logp.*logprobs.*loss_mask"):
        compute_sequence_ess(
            prox_logp=prox_logp,
            logprobs=logprobs,
            loss_mask=loss_mask,
        )


def test_compute_sequence_ess_rejects_flat_packed_tensors():
    prox_logp = torch.zeros(4)
    logprobs = torch.zeros_like(prox_logp)
    loss_mask = torch.ones_like(prox_logp)

    with pytest.raises(ValueError, match="2-D"):
        compute_sequence_ess(
            prox_logp=prox_logp,
            logprobs=logprobs,
            loss_mask=loss_mask,
        )


def test_compute_sequence_ess_dominant_weight_reduces_ratio():
    logprobs = torch.zeros((2, 1))
    prox_logp = torch.tensor([[math.log(100.0)], [math.log(1.0)]])
    loss_mask = torch.ones_like(logprobs)

    stats = compute_sequence_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    expected_ess = (101.0**2) / (100.0**2 + 1.0)
    assert stats is not None
    assert stats.ess == pytest.approx(expected_ess)
    assert stats.ess_ratio == pytest.approx(expected_ess / 2.0)
    assert stats.ess_lr_scale == pytest.approx(math.sqrt(expected_ess / 2.0))


def test_compute_token_ess_dominant_token_reduces_ratio():
    logprobs = torch.zeros((2, 2))
    prox_logp = torch.tensor(
        [
            [math.log(100.0), 0.0],
            [0.0, 0.0],
        ]
    )
    loss_mask = torch.ones_like(logprobs)

    stats = compute_token_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    expected_ess = (103.0**2) / (100.0**2 + 1.0 + 1.0 + 1.0)
    assert stats is not None
    assert stats.ess == pytest.approx(expected_ess)
    assert stats.ess_ratio == pytest.approx(expected_ess / 4.0)
    assert stats.ess_lr_scale == pytest.approx(math.sqrt(expected_ess / 4.0))


def test_sequence_and_token_ess_can_differ():
    logprobs = torch.zeros((2, 2))
    prox_logp = torch.tensor(
        [
            [math.log(3.0), math.log(3.0)],
            [0.0, 0.0],
        ]
    )
    loss_mask = torch.ones_like(logprobs)

    sequence_stats = compute_sequence_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )
    token_stats = compute_token_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    assert sequence_stats is not None
    assert token_stats is not None
    assert sequence_stats.ess_ratio == pytest.approx(
        ((9.0 + 1.0) ** 2 / (9.0**2 + 1.0)) / 2.0
    )
    assert token_stats.ess_ratio == pytest.approx(
        ((3.0 + 3.0 + 1.0 + 1.0) ** 2 / (3.0**2 + 3.0**2 + 1.0 + 1.0)) / 4.0
    )
    assert sequence_stats.ess_ratio != pytest.approx(token_stats.ess_ratio)


def test_compute_sequence_ess_applies_custom_config_lr_scale_clamps():
    logprobs = torch.zeros((2, 1))
    loss_mask = torch.ones_like(logprobs)

    min_clamped_stats = compute_sequence_ess(
        prox_logp=torch.tensor([[math.log(100.0)], [math.log(1.0)]]),
        logprobs=logprobs,
        loss_mask=loss_mask,
        scaling_config=ESSScalingConfig(min_lr_scale=0.8, max_lr_scale=1.0),
    )
    assert min_clamped_stats is not None
    assert min_clamped_stats.ess_lr_scale == pytest.approx(0.8)

    max_clamped_stats = compute_sequence_ess(
        prox_logp=logprobs.clone(),
        logprobs=logprobs,
        loss_mask=loss_mask,
        scaling_config=ESSScalingConfig(base_ess_ratio=0.25, max_lr_scale=0.9),
    )
    assert max_clamped_stats is not None
    assert max_clamped_stats.ess_lr_scale == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("scaling_config", "match"),
    [
        (
            SimpleNamespace(
                base_ess_ratio=0.0,
                min_lr_scale=0.0,
                max_lr_scale=1.0,
            ),
            "base_ess_ratio",
        ),
        (
            SimpleNamespace(
                base_ess_ratio=float("nan"),
                min_lr_scale=0.0,
                max_lr_scale=1.0,
            ),
            "base_ess_ratio",
        ),
        (
            SimpleNamespace(
                base_ess_ratio=1.0,
                min_lr_scale=float("nan"),
                max_lr_scale=1.0,
            ),
            "min_lr_scale",
        ),
        (
            SimpleNamespace(
                base_ess_ratio=1.0,
                min_lr_scale=0.0,
                max_lr_scale=float("inf"),
            ),
            "max_lr_scale",
        ),
        (
            SimpleNamespace(
                base_ess_ratio=1.0,
                min_lr_scale=1.1,
                max_lr_scale=1.0,
            ),
            "min_lr_scale.*max_lr_scale",
        ),
    ],
)
def test_compute_sequence_ess_rejects_invalid_scaling_config_values(
    scaling_config, match
):
    logprobs = torch.zeros((2, 1))
    prox_logp = torch.zeros_like(logprobs)
    loss_mask = torch.ones_like(logprobs)

    with pytest.raises(ValueError, match=match):
        compute_sequence_ess(
            prox_logp=prox_logp,
            logprobs=logprobs,
            loss_mask=loss_mask,
            scaling_config=scaling_config,
        )


def test_compute_sequence_ess_uses_full_sequence_sum():
    logprobs = torch.zeros((2, 2))
    prox_logp = torch.tensor(
        [
            [math.log(2.0), math.log(3.0)],
            [math.log(6.0), 0.0],
        ]
    )
    loss_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    stats = compute_sequence_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    assert stats is not None
    assert stats.ess == pytest.approx(2.0)
    assert stats.ess_ratio == pytest.approx(1.0)


def test_compute_sequence_behavior_metrics_uses_valid_sequence_values():
    logprobs = torch.zeros((3, 3))
    prox_logp = torch.tensor(
        [
            [math.log(2.0), math.log(3.0), 99.0],
            [math.log(4.0), 0.0, 0.0],
            [10.0, 20.0, 30.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    metrics = compute_sequence_behavior_metrics(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    assert torch.equal(metrics.valid_mask, torch.tensor([True, True, False]))
    torch.testing.assert_close(
        metrics.log_weight,
        torch.tensor([math.log(6.0), math.log(4.0), 0.0]),
    )
    torch.testing.assert_close(
        metrics.mean_log_ratio,
        torch.tensor([math.log(6.0) / 2.0, math.log(4.0) / 3.0, 0.0]),
    )


def test_compute_sequence_ess_excludes_empty_sequences():
    logprobs = torch.zeros((2, 2))
    prox_logp = torch.zeros_like(logprobs)
    loss_mask = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    stats = compute_sequence_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    assert stats is not None
    assert stats.valid_count == 1
    assert stats.ess == pytest.approx(1.0)
    assert stats.ess_ratio == pytest.approx(1.0)


def test_compute_sequence_ess_returns_none_for_fully_empty_minibatch():
    logprobs = torch.zeros((2, 2))
    prox_logp = torch.zeros_like(logprobs)
    loss_mask = torch.zeros_like(logprobs)

    stats = compute_sequence_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    assert stats is None


def test_compute_sequence_ess_sanitizes_nonfinite_token_ratios():
    logprobs = torch.tensor([[0.0, 0.0, float("nan")], [0.0, float("-inf"), 0.0]])
    prox_logp = torch.tensor([[float("inf"), 0.0, 0.0], [0.0, 0.0, float("nan")]])
    loss_mask = torch.ones_like(logprobs)

    stats = compute_sequence_ess(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    assert stats is not None
    assert math.isfinite(stats.ess)
    assert math.isfinite(stats.ess_ratio)
    assert math.isfinite(stats.ess_lr_scale)


def test_compute_sequence_behavior_metrics_sanitizes_nonfinite_token_ratios():
    logprobs = torch.tensor([[0.0, float("-inf"), 0.0]])
    prox_logp = torch.tensor([[float("inf"), 0.0, float("nan")]])
    loss_mask = torch.ones_like(logprobs)

    metrics = compute_sequence_behavior_metrics(
        prox_logp=prox_logp,
        logprobs=logprobs,
        loss_mask=loss_mask,
    )

    assert torch.equal(metrics.valid_mask, torch.tensor([True]))
    torch.testing.assert_close(metrics.log_weight, torch.tensor([0.0]))
    torch.testing.assert_close(metrics.mean_log_ratio, torch.tensor([0.0]))


def test_summarize_ess_stats_returns_level_named_avg_min_max():
    stats = [
        ESSStats(
            ess=1.0,
            ess_ratio=0.5,
            ess_lr_scale=0.25,
            valid_count=2,
        ),
        ESSStats(
            ess=3.0,
            ess_ratio=0.75,
            ess_lr_scale=0.5,
            valid_count=4,
        ),
    ]

    summary = summarize_ess_stats(stats, level="sequence")

    assert summary == {
        "ess_ratio_sequence/avg": pytest.approx(0.625),
        "ess_ratio_sequence/min": pytest.approx(0.5),
        "ess_ratio_sequence/max": pytest.approx(0.75),
        "ess_sequence/avg": pytest.approx(2.0),
        "ess_sequence/min": pytest.approx(1.0),
        "ess_sequence/max": pytest.approx(3.0),
    }


def test_summarize_ess_lr_scale_stats_returns_avg_min_max():
    stats = [
        ESSStats(
            ess=1.0,
            ess_ratio=0.5,
            ess_lr_scale=0.25,
            valid_count=2,
        ),
        ESSStats(
            ess=3.0,
            ess_ratio=0.75,
            ess_lr_scale=0.5,
            valid_count=4,
        ),
    ]

    summary = summarize_ess_lr_scale_stats(stats)

    assert summary["ess_lr_scale/avg"] == pytest.approx(0.375)
    assert summary["ess_lr_scale/min"] == pytest.approx(0.25)
    assert summary["ess_lr_scale/max"] == pytest.approx(0.5)


class _StatsRecorder:
    def __init__(self):
        self.denominator_calls = []
        self.stat_calls = []
        self.scalar_calls = []

    def denominator(self, **kwargs):
        self.denominator_calls.append(kwargs)

    def stat(self, **kwargs):
        self.stat_calls.append(kwargs)

    def scalar(self, **kwargs):
        self.scalar_calls.append(kwargs)

    @contextmanager
    def scope(self, name):
        yield


class _FakeEngine:
    def __init__(
        self,
        train_stats,
        *,
        data_parallel_group=None,
        supports_optimizer_step_scale=True,
    ):
        self._train_stats = list(train_stats)
        self.supports_optimizer_step_scale = supports_optimizer_step_scale
        self.data_parallel_group = data_parallel_group
        self.train_batch_kwargs = []

    def train(self):
        pass

    def get_version(self):
        return 7

    def train_batch(self, *args, **kwargs):
        self.train_batch_kwargs.append(kwargs)
        return dict(self._train_stats.pop(0))


def _make_actor(
    train_stats,
    *,
    data_parallel_group=None,
    ess_scaling=None,
    supports_optimizer_step_scale=True,
    use_decoupled_loss=True,
) -> PPOActor:
    actor = PPOActor.__new__(PPOActor)
    actor.config = SimpleNamespace(
        c_clip=None,
        eps_clip=0.2,
        eps_clip_higher=None,
        ess_scaling=ess_scaling,
        importance_sampling_level="token",
        log_agent_stats=False,
        mask_no_eos_with_zero=False,
        per_trajectory=SimpleNamespace(enabled=False),
        ppo_n_minibatches=len(train_stats),
        prox_logp_method=None,
        rejection_sampling=None,
        sapo_tau_neg=None,
        sapo_tau_pos=None,
        use_decoupled_loss=use_decoupled_loss,
        use_sapo_loss=False,
    )
    actor.engine = _FakeEngine(
        train_stats,
        data_parallel_group=data_parallel_group,
        supports_optimizer_step_scale=supports_optimizer_step_scale,
    )
    actor.m2_threshold = None
    return actor


def _make_data(*, include_prox_logp=True) -> dict[str, torch.Tensor]:
    data = {
        "attention_mask": torch.ones(2, 2, dtype=torch.bool),
        "loss_mask": torch.ones(2, 2, dtype=torch.bool),
        "rewards": torch.tensor([1.0, -1.0]),
        "advantages": torch.ones(2, 2),
        "kl_rewards": torch.full((2, 2), 0.1),
        "tot_rewards": torch.full((2, 2), 0.5),
        "logprobs": torch.zeros(2, 2),
    }
    if include_prox_logp:
        data["prox_logp"] = torch.zeros(2, 2)
    return data


def _make_data_without_loss_mask() -> dict[str, torch.Tensor]:
    data = _make_data()
    data.pop("loss_mask")
    return data


def _make_minibatch(
    *,
    prox_logp: torch.Tensor,
    logprobs: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if logprobs is None:
        logprobs = torch.zeros_like(prox_logp)
    if loss_mask is None:
        loss_mask = torch.ones_like(prox_logp, dtype=torch.bool)
    return {
        "prox_logp": prox_logp,
        "logprobs": logprobs,
        "loss_mask": loss_mask,
        "advantages": torch.ones_like(prox_logp),
    }


def _uniform_minibatch() -> dict[str, torch.Tensor]:
    return _make_minibatch(prox_logp=torch.zeros(2, 1))


def _skewed_minibatch() -> dict[str, torch.Tensor]:
    return _make_minibatch(prox_logp=torch.tensor([[math.log(3.0)], [math.log(1.0)]]))


def _multi_token_skewed_minibatch() -> dict[str, torch.Tensor]:
    return _make_minibatch(
        prox_logp=torch.tensor(
            [
                [math.log(3.0), math.log(3.0)],
                [0.0, 0.0],
            ]
        )
    )


def _empty_minibatch() -> dict[str, torch.Tensor]:
    return _make_minibatch(
        prox_logp=torch.zeros(2, 1),
        loss_mask=torch.zeros(2, 1, dtype=torch.bool),
    )


def _patch_actor_update_fakes(monkeypatch, recorder, minibatches):
    monkeypatch.setattr(actor_module, "stats_tracker", recorder)

    def split_fake(data, mb_spec):
        assert mb_spec.n_mbs == len(minibatches)
        return SimpleNamespace(mbs=minibatches)

    monkeypatch.setattr(
        actor_module, "split_padded_tensor_dict_into_mb_list", split_fake
    )


def _scalar_keys(recorder):
    return {key for call in recorder.scalar_calls for key in call}


def _scalar_call_with(recorder, key):
    matches = [call for call in recorder.scalar_calls if key in call]
    assert len(matches) == 1
    return matches[0]


def _stat_call_with(recorder, key):
    for call in recorder.stat_calls:
        if key in call:
            return call
    raise AssertionError(f"No stat call contained {key!r}: {recorder.stat_calls}")


def test_ess_collective_device_uses_engine_device_for_non_cpu_backend(monkeypatch):
    monkeypatch.setattr(
        actor_module,
        "dist",
        SimpleNamespace(
            get_backend=lambda group=None: "nccl",
            is_available=lambda: True,
            is_initialized=lambda: True,
        ),
        raising=False,
    )

    device = actor_module._ess_collective_device(
        {"logprobs": torch.zeros(1)},
        process_group="dp-group",
        engine=SimpleNamespace(device=torch.device("cuda:7")),
    )

    assert device == torch.device("cuda:7")


def test_ess_collective_device_uses_cpu_for_gloo_backend(monkeypatch):
    monkeypatch.setattr(
        actor_module,
        "dist",
        SimpleNamespace(
            get_backend=lambda group=None: "gloo",
            is_available=lambda: True,
            is_initialized=lambda: True,
        ),
        raising=False,
    )

    device = actor_module._ess_collective_device(
        {"logprobs": torch.zeros(1)},
        process_group="dp-group",
        engine=SimpleNamespace(device=torch.device("cuda:7")),
    )

    assert device == torch.device("cpu")


def test_ppo_update_logs_sequence_and_token_ess_metrics_without_scaling(monkeypatch):
    recorder = _StatsRecorder()
    minibatches = [_multi_token_skewed_minibatch()]
    _patch_actor_update_fakes(monkeypatch, recorder, minibatches)
    actor = _make_actor([{"loss": 1.0}], ess_scaling=None)

    actor._ppo_update(_make_data())

    assert all(
        "optimizer_step_scale" not in kwargs
        for kwargs in actor.engine.train_batch_kwargs
    )
    summary = _scalar_call_with(recorder, "ess_ratio_sequence/avg")
    assert {
        "ess_ratio_sequence/avg",
        "ess_ratio_sequence/min",
        "ess_ratio_sequence/max",
        "ess_sequence/avg",
        "ess_sequence/min",
        "ess_sequence/max",
        "ess_ratio_token/avg",
        "ess_ratio_token/min",
        "ess_ratio_token/max",
        "ess_token/avg",
        "ess_token/min",
        "ess_token/max",
    } <= set(summary)
    sequence_ess_ratio = ((9.0 + 1.0) ** 2 / (9.0**2 + 1.0)) / 2.0
    token_ess_ratio = (
        ((3.0 + 3.0 + 1.0 + 1.0) ** 2) / (3.0**2 + 3.0**2 + 1.0 + 1.0)
    ) / 4.0
    assert summary["ess_ratio_sequence/avg"] == pytest.approx(sequence_ess_ratio)
    assert summary["ess_ratio_token/avg"] == pytest.approx(token_ess_ratio)
    assert summary["ess_ratio_sequence/avg"] != pytest.approx(
        summary["ess_ratio_token/avg"]
    )
    legacy_metric_keys = {
        "ess_ratio" + "/avg",
        "ess" + "/avg",
    }
    assert legacy_metric_keys.isdisjoint(summary)
    keys = _scalar_keys(recorder)
    assert not any(key.startswith("ess_lr_scale/") for key in keys)
    assert not any(key.startswith("ess_lr/") for key in keys)


def test_ppo_update_logs_sequence_behavior_metrics_with_sequence_denominator(
    monkeypatch,
):
    recorder = _StatsRecorder()
    minibatches = [_multi_token_skewed_minibatch()]
    _patch_actor_update_fakes(monkeypatch, recorder, minibatches)
    actor = _make_actor([{"loss": 1.0}], ess_scaling=None)

    actor._ppo_update(_make_data())

    stat_call = _stat_call_with(recorder, "behave_seq_log_weight")
    assert stat_call["denominator"] == "n_valid_seqs"
    torch.testing.assert_close(
        stat_call["behave_seq_log_weight"],
        torch.tensor([math.log(9.0), 0.0]),
    )
    torch.testing.assert_close(
        stat_call["behave_seq_mean_log_ratio"],
        torch.tensor([math.log(9.0) / 2.0, 0.0]),
    )

    denominator_call = next(
        call for call in recorder.denominator_calls if "n_valid_seqs" in call
    )
    assert torch.equal(
        denominator_call["n_valid_seqs"],
        torch.tensor([True, True]),
    )


def test_ppo_update_sequence_scaling_uses_sequence_ess_by_default(monkeypatch):
    recorder = _StatsRecorder()
    minibatches = [_uniform_minibatch(), _multi_token_skewed_minibatch()]
    _patch_actor_update_fakes(monkeypatch, recorder, minibatches)
    actor = _make_actor(
        [{"loss": 1.0}, {"loss": 2.0}],
        ess_scaling=ESSScalingConfig(),
    )

    actor._ppo_update(_make_data())

    sequence_ess_ratio = ((9.0 + 1.0) ** 2 / (9.0**2 + 1.0)) / 2.0
    scales = [
        kwargs["optimizer_step_scale"] for kwargs in actor.engine.train_batch_kwargs
    ]
    assert scales == pytest.approx([1.0, math.sqrt(sequence_ess_ratio)])


def test_ppo_update_token_scaling_uses_token_ess(monkeypatch):
    recorder = _StatsRecorder()
    minibatches = [_uniform_minibatch(), _multi_token_skewed_minibatch()]
    _patch_actor_update_fakes(monkeypatch, recorder, minibatches)
    actor = _make_actor(
        [{"loss": 1.0}, {"loss": 2.0}],
        ess_scaling=ESSScalingConfig(level="token"),
    )

    actor._ppo_update(_make_data())

    token_ess_ratio = (
        ((3.0 + 3.0 + 1.0 + 1.0) ** 2) / (3.0**2 + 3.0**2 + 1.0 + 1.0)
    ) / 4.0
    scales = [
        kwargs["optimizer_step_scale"] for kwargs in actor.engine.train_batch_kwargs
    ]
    assert scales == pytest.approx([1.0, math.sqrt(token_ess_ratio)])


def test_ppo_update_logs_effective_lr_when_scaling_enabled(monkeypatch):
    recorder = _StatsRecorder()
    minibatches = [_uniform_minibatch(), _multi_token_skewed_minibatch()]
    _patch_actor_update_fakes(monkeypatch, recorder, minibatches)
    actor = _make_actor(
        [{"loss": 1.0, "lr": 0.01}, {"loss": 2.0, "lr": 0.01}],
        ess_scaling=ESSScalingConfig(),
    )

    actor._ppo_update(_make_data())

    sequence_ess_ratio = ((9.0 + 1.0) ** 2 / (9.0**2 + 1.0)) / 2.0
    scale_2 = math.sqrt(sequence_ess_ratio)
    summary = _scalar_call_with(recorder, "ess_lr_scale/avg")
    assert summary["ess_lr_scale/avg"] == pytest.approx((1.0 + scale_2) / 2)
    assert summary["ess_lr_scale/min"] == pytest.approx(scale_2)
    assert summary["ess_lr_scale/max"] == pytest.approx(1.0)
    assert summary["ess_lr/avg"] == pytest.approx((0.01 + 0.01 * scale_2) / 2)
    assert summary["ess_lr/min"] == pytest.approx(0.01 * scale_2)
    assert summary["ess_lr/max"] == pytest.approx(0.01)


def test_ppo_update_raises_when_scaling_enabled_without_prox_logp(monkeypatch):
    recorder = _StatsRecorder()
    _patch_actor_update_fakes(monkeypatch, recorder, [_uniform_minibatch()])
    actor = _make_actor([{"loss": 1.0}], ess_scaling=ESSScalingConfig())

    with pytest.raises(RuntimeError, match="prox_logp"):
        actor._ppo_update(_make_data(include_prox_logp=False))


def test_ppo_update_raises_when_scaling_enabled_without_outer_loss_mask(
    monkeypatch,
):
    recorder = _StatsRecorder()
    _patch_actor_update_fakes(monkeypatch, recorder, [_uniform_minibatch()])
    actor = _make_actor([{"loss": 1.0}], ess_scaling=ESSScalingConfig())

    with pytest.raises(RuntimeError, match="loss_mask"):
        actor._ppo_update(_make_data_without_loss_mask())


def test_ppo_update_raises_on_mixed_exact_ess_inputs_before_minibatch_split(
    monkeypatch,
):
    recorder = _StatsRecorder()
    monkeypatch.setattr(actor_module, "stats_tracker", recorder)

    def all_reduce_mixed_availability(tensor, op=None, group=None):
        assert group == "dp-group"
        tensor[0] = 1
        tensor[1] = 2

    monkeypatch.setattr(
        actor_module,
        "dist",
        SimpleNamespace(
            ReduceOp=SimpleNamespace(SUM=object()),
            all_reduce=all_reduce_mixed_availability,
            is_available=lambda: True,
            is_initialized=lambda: True,
        ),
        raising=False,
    )

    def split_should_not_run(data, mb_spec):
        raise AssertionError("minibatch split should not run")

    monkeypatch.setattr(
        actor_module, "split_padded_tensor_dict_into_mb_list", split_should_not_run
    )
    actor = _make_actor(
        [{"loss": 1.0}],
        data_parallel_group="dp-group",
        ess_scaling=None,
    )

    with pytest.raises(RuntimeError, match="exact ESS inputs.*DP ranks.*prox_logp"):
        actor._ppo_update(_make_data(include_prox_logp=False))
    assert actor.engine.train_batch_kwargs == []


def test_ppo_update_skips_ess_when_scaling_disabled_without_prox_logp(monkeypatch):
    recorder = _StatsRecorder()
    minibatches = [
        {
            "logprobs": torch.zeros(2, 1),
            "loss_mask": torch.ones(2, 1, dtype=torch.bool),
            "advantages": torch.ones(2, 1),
        }
    ]
    _patch_actor_update_fakes(monkeypatch, recorder, minibatches)
    actor = _make_actor([{"loss": 1.0}], ess_scaling=None)

    actor._ppo_update(_make_data(include_prox_logp=False))

    assert len(actor.engine.train_batch_kwargs) == 1
    assert not any(key.startswith("ess") for key in _scalar_keys(recorder))


def test_ppo_update_raises_when_scaling_enabled_for_unsupported_engine(monkeypatch):
    recorder = _StatsRecorder()
    _patch_actor_update_fakes(monkeypatch, recorder, [_uniform_minibatch()])
    actor = _make_actor(
        [{"loss": 1.0}],
        ess_scaling=ESSScalingConfig(),
        supports_optimizer_step_scale=False,
    )

    with pytest.raises(RuntimeError, match="FSDP"):
        actor._ppo_update(_make_data())


def test_ppo_update_uses_scale_one_for_empty_ess_minibatch(monkeypatch):
    recorder = _StatsRecorder()
    minibatches = [_empty_minibatch(), _skewed_minibatch()]
    _patch_actor_update_fakes(monkeypatch, recorder, minibatches)
    actor = _make_actor(
        [{"loss": 1.0, "lr": 0.01}, {"loss": 2.0, "lr": 0.01}],
        ess_scaling=ESSScalingConfig(),
    )

    actor._ppo_update(_make_data())

    scales = [
        kwargs["optimizer_step_scale"] for kwargs in actor.engine.train_batch_kwargs
    ]
    assert scales == pytest.approx([1.0, math.sqrt(0.8)])
    summary = _scalar_call_with(recorder, "ess_ratio_sequence/avg")
    assert summary["ess_ratio_sequence/avg"] == pytest.approx(0.8)
    assert summary["ess_ratio_sequence/min"] == pytest.approx(0.8)
    assert summary["ess_ratio_sequence/max"] == pytest.approx(0.8)
    assert summary["ess_sequence/avg"] == pytest.approx(1.6)
    assert summary["ess_sequence/min"] == pytest.approx(1.6)
    assert summary["ess_sequence/max"] == pytest.approx(1.6)
    assert summary["ess_ratio_token/avg"] == pytest.approx(0.8)
    assert summary["ess_ratio_token/min"] == pytest.approx(0.8)
    assert summary["ess_ratio_token/max"] == pytest.approx(0.8)
    assert summary["ess_token/avg"] == pytest.approx(1.6)
    assert summary["ess_token/min"] == pytest.approx(1.6)
    assert summary["ess_token/max"] == pytest.approx(1.6)
    assert summary["ess_lr_scale/avg"] == pytest.approx(math.sqrt(0.8))
    assert summary["ess_lr_scale/min"] == pytest.approx(math.sqrt(0.8))
    assert summary["ess_lr_scale/max"] == pytest.approx(math.sqrt(0.8))
    assert summary["ess_lr/avg"] == pytest.approx(0.01 * math.sqrt(0.8))
    assert summary["ess_lr/min"] == pytest.approx(0.01 * math.sqrt(0.8))
    assert summary["ess_lr/max"] == pytest.approx(0.01 * math.sqrt(0.8))


def test_ppo_update_omits_ess_summary_when_all_ess_minibatches_empty(monkeypatch):
    recorder = _StatsRecorder()
    minibatches = [_empty_minibatch(), _empty_minibatch()]
    _patch_actor_update_fakes(monkeypatch, recorder, minibatches)
    actor = _make_actor(
        [{"loss": 1.0, "lr": 0.01}, {"loss": 2.0, "lr": 0.02}],
        ess_scaling=ESSScalingConfig(),
    )

    actor._ppo_update(_make_data())

    scales = [
        kwargs["optimizer_step_scale"] for kwargs in actor.engine.train_batch_kwargs
    ]
    assert scales == pytest.approx([1.0, 1.0])
    keys = _scalar_keys(recorder)
    assert not any(key.startswith("ess_ratio_sequence/") for key in keys)
    assert not any(key.startswith("ess_sequence/") for key in keys)
    assert not any(key.startswith("ess_ratio_token/") for key in keys)
    assert not any(key.startswith("ess_token/") for key in keys)
    assert not any(key.startswith("ess_lr_scale/") for key in keys)
    assert not any(key.startswith("ess_lr/") for key in keys)
