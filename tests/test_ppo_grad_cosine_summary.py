import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from areal.trainer.ppo import actor as actor_module
from areal.trainer.ppo.actor import PPOActor, _summarize_grad_cos_sims


def test_summarize_grad_cos_sims_returns_abs_and_signed_extrema():
    summary = _summarize_grad_cos_sims([-0.8, 0.2, 0.5])

    assert summary == pytest.approx(
        {
            "grad_cos_sim_abs/max": 0.8,
            "grad_cos_sim_abs/min": 0.2,
            "grad_cos_sim_abs/avg": 0.5,
            "grad_cos_sim/max": 0.5,
            "grad_cos_sim/min": -0.8,
        }
    )


@pytest.mark.parametrize("values", [[], [math.nan], [math.inf], [-math.inf]])
def test_summarize_grad_cos_sims_omits_when_no_finite_values(values):
    assert _summarize_grad_cos_sims(values) == {}


class _StatsRecorder:
    def __init__(self):
        self.scalar_calls = []

    def denominator(self, **kwargs):
        pass

    def stat(self, **kwargs):
        pass

    def scalar(self, **kwargs):
        self.scalar_calls.append(kwargs)

    @contextmanager
    def scope(self, name):
        yield


class _FakeEngine:
    def __init__(self, train_stats):
        self._train_stats = list(train_stats)

    def train(self):
        pass

    def get_version(self):
        return 7

    def train_batch(self, *args, **kwargs):
        return dict(self._train_stats.pop(0))


def _make_actor(train_stats) -> PPOActor:
    actor = PPOActor.__new__(PPOActor)
    actor.config = SimpleNamespace(
        c_clip=None,
        eps_clip=0.2,
        eps_clip_higher=None,
        importance_sampling_level="token",
        log_agent_stats=False,
        mask_no_eos_with_zero=False,
        ppo_n_minibatches=len(train_stats),
        prox_logp_method=None,
        rejection_sampling=None,
        sapo_tau_neg=None,
        sapo_tau_pos=None,
        use_decoupled_loss=False,
        use_sapo_loss=False,
    )
    actor.engine = _FakeEngine(train_stats)
    actor.m2_threshold = None
    return actor


def _make_data() -> dict[str, torch.Tensor]:
    return {
        "attention_mask": torch.ones(2, 4, dtype=torch.bool),
        "loss_mask": torch.tensor(
            [[0, 1, 1, 0], [0, 1, 0, 0]],
            dtype=torch.bool,
        ),
        "rewards": torch.tensor([1.0, -1.0]),
        "advantages": torch.ones(2, 4),
        "kl_rewards": torch.full((2, 4), 0.1),
        "tot_rewards": torch.full((2, 4), 0.5),
    }


def test_ppo_update_pops_grad_cos_sim_and_emits_only_summary(monkeypatch):
    recorder = _StatsRecorder()
    monkeypatch.setattr(actor_module, "stats_tracker", recorder)
    monkeypatch.setattr(
        actor_module,
        "split_padded_tensor_dict_into_mb_list",
        lambda data, mb_spec: SimpleNamespace(
            mbs=[
                {"loss_mask": torch.ones(1, dtype=torch.bool)}
                for _ in range(mb_spec.n_mbs)
            ]
        ),
    )
    actor = _make_actor(
        [
            {"loss": 1.0, "grad_cos_sim": -0.8},
            {"loss": 2.0, "grad_cos_sim": 0.2},
            {"loss": 3.0, "grad_cos_sim": float("nan")},
        ]
    )

    actor._ppo_update(_make_data())

    assert all("grad_cos_sim" not in call for call in recorder.scalar_calls)
    summary_calls = [
        call for call in recorder.scalar_calls if "grad_cos_sim_abs/max" in call
    ]
    assert len(summary_calls) == 1
    assert summary_calls[0] == pytest.approx(
        {
            "grad_cos_sim_abs/max": 0.8,
            "grad_cos_sim_abs/min": 0.2,
            "grad_cos_sim_abs/avg": 0.5,
            "grad_cos_sim/max": 0.2,
            "grad_cos_sim/min": -0.8,
        }
    )


def test_ppo_update_omits_summary_when_no_finite_grad_cos_sim(monkeypatch):
    recorder = _StatsRecorder()
    monkeypatch.setattr(actor_module, "stats_tracker", recorder)
    monkeypatch.setattr(
        actor_module,
        "split_padded_tensor_dict_into_mb_list",
        lambda data, mb_spec: SimpleNamespace(
            mbs=[
                {"loss_mask": torch.ones(1, dtype=torch.bool)}
                for _ in range(mb_spec.n_mbs)
            ]
        ),
    )
    actor = _make_actor(
        [
            {"loss": 1.0},
            {"loss": 2.0, "grad_cos_sim": float("inf")},
        ]
    )

    actor._ppo_update(_make_data())

    assert all("grad_cos_sim" not in call for call in recorder.scalar_calls)
    assert all("grad_cos_sim_abs/max" not in call for call in recorder.scalar_calls)
