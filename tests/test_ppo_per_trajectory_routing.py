from contextlib import contextmanager
from unittest.mock import ANY, MagicMock, call

import pytest
import torch

from areal.api.cli_args import PerTrajectoryConfig, PPOActorConfig
from areal.trainer.ppo import actor as actor_module
from areal.trainer.ppo.actor import PPOActor


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


def _stat_side_effect(train_stats):
    stats = list(train_stats)

    def _next_stat(*args, **kwargs):
        return dict(stats.pop(0))

    return _next_stat


class _FakeEngine:
    supports_optimizer_step_scale = False

    def __init__(self, train_stats=None, per_trajectory_stats=None):
        self.train = MagicMock()
        self.get_version = MagicMock(return_value=3)
        self.train_batch = MagicMock(
            side_effect=_stat_side_effect(train_stats or [{"lr": 1.0}])
        )
        self.train_batch_per_trajectory = MagicMock(
            side_effect=_stat_side_effect(per_trajectory_stats or [{"lr": 1.0}])
        )


def _minimal_batch(batch_size=1):
    return {
        "input_ids": torch.arange(1, batch_size * 3 + 1).view(batch_size, 3),
        "attention_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0]], dtype=torch.bool).repeat(batch_size, 1),
        "advantages": torch.tensor([[0.0, 1.0, 0.0]]).repeat(batch_size, 1),
        "logprobs": torch.zeros(batch_size, 3),
        "prox_logp": torch.zeros(batch_size, 3),
        "rollout_logprobs": torch.zeros(batch_size, 3),
        "trainer_global_step": torch.full((batch_size,), 5),
        "rewards": torch.ones(batch_size),
        "tot_rewards": torch.zeros(batch_size, 3),
        "kl_rewards": torch.zeros(batch_size, 3),
    }


def test_per_trajectory_disabled_uses_normal_train_batch(monkeypatch):
    recorder = _StatsRecorder()
    monkeypatch.setattr(actor_module, "stats_tracker", recorder)
    engine = _FakeEngine(train_stats=[{"lr": 1.0}])
    actor = PPOActor(PPOActorConfig(disable_dropout=True, ppo_n_minibatches=1), engine)

    actor._ppo_update(_minimal_batch())

    engine.train_batch.assert_called_once_with(ANY, loss_fn=ANY, loss_weight_fn=ANY)
    assert set(engine.train_batch.call_args.kwargs) == {"loss_fn", "loss_weight_fn"}
    normal_mb = engine.train_batch.call_args.args[0]
    assert "task_reward" not in normal_mb
    engine.train_batch_per_trajectory.assert_not_called()
    assert {"lr": 1.0} in recorder.scalar_calls


def test_per_trajectory_enabled_uses_replacement_path(monkeypatch):
    recorder = _StatsRecorder()
    monkeypatch.setattr(actor_module, "stats_tracker", recorder)
    engine = _FakeEngine(per_trajectory_stats=[{"lr": 2.0}, {"lr": 3.0}])
    actor = PPOActor(
        PPOActorConfig(
            disable_dropout=True,
            ppo_n_minibatches=2,
            per_trajectory=PerTrajectoryConfig(enabled=True),
        ),
        engine,
    )
    batch = _minimal_batch(batch_size=2)
    batch["rewards"] = torch.tensor([1.25, -0.5])

    def fake_splitter(data, mb_spec):
        assert "rewards" not in data
        assert torch.equal(data["task_reward"], torch.tensor([1.25, -0.5]))
        return type(
            "FakeMicroBatchList",
            (),
            {
                "forward_indices": [1, 0],
                "mbs": [
                    {"attention_mask": torch.ones(1, 3, dtype=torch.bool)},
                    {"attention_mask": torch.ones(1, 3, dtype=torch.bool)},
                ],
            },
        )()

    monkeypatch.setattr(
        actor_module,
        "split_padded_tensor_dict_into_mb_list",
        fake_splitter,
    )

    actor._ppo_update(batch)

    engine.train_batch.assert_not_called()
    engine.train_batch_per_trajectory.assert_has_calls(
        [
            call(ANY, minibatch_idx=0, loss_fn=ANY, loss_weight_fn=ANY),
            call(ANY, minibatch_idx=1, loss_fn=ANY, loss_weight_fn=ANY),
        ]
    )
    assert engine.train_batch_per_trajectory.call_count == 2
    assert [
        call.kwargs["minibatch_idx"]
        for call in engine.train_batch_per_trajectory.call_args_list
    ] == [0, 1]
    per_trajectory_mbs = [
        call.args[0] for call in engine.train_batch_per_trajectory.call_args_list
    ]
    assert all("rewards" not in mb for mb in per_trajectory_mbs)
    assert [float(mb["task_reward"].flatten()[0]) for mb in per_trajectory_mbs] == [
        pytest.approx(-0.5),
        pytest.approx(1.25),
    ]
    assert {"lr": 2.0} in recorder.scalar_calls
    assert {"lr": 3.0} in recorder.scalar_calls


def test_per_trajectory_metadata_reordered_with_minibatch_indices(monkeypatch):
    recorder = _StatsRecorder()
    monkeypatch.setattr(actor_module, "stats_tracker", recorder)
    engine = _FakeEngine(per_trajectory_stats=[{"lr": 2.0}, {"lr": 3.0}])
    actor = PPOActor(
        PPOActorConfig(
            disable_dropout=True,
            ppo_n_minibatches=2,
            per_trajectory=PerTrajectoryConfig(enabled=True),
        ),
        engine,
    )
    batch = _minimal_batch(batch_size=2)
    batch["rewards"] = torch.tensor([1.25, -0.5])
    batch["uid"] = ["uid-0", "uid-1"]
    batch["traj_uid"] = ["traj-0", "traj-1"]
    batch["trainer_global_step"] = torch.tensor([7, 8])

    def fake_splitter(data, mb_spec):
        assert data["uid"] == ["uid-0", "uid-1"]
        assert data["traj_uid"] == ["traj-0", "traj-1"]
        assert torch.equal(data["trainer_global_step"], torch.tensor([7, 8]))
        return type(
            "FakeMicroBatchList",
            (),
            {
                "forward_indices": [1, 0],
                "mbs": [
                    {
                        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
                        "uid": data["uid"],
                        "traj_uid": data["traj_uid"],
                        "trainer_global_step": data["trainer_global_step"],
                    },
                    {
                        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
                        "uid": data["uid"],
                        "traj_uid": data["traj_uid"],
                        "trainer_global_step": data["trainer_global_step"],
                    },
                ],
            },
        )()

    monkeypatch.setattr(
        actor_module,
        "split_padded_tensor_dict_into_mb_list",
        fake_splitter,
    )

    actor._ppo_update(batch)

    per_trajectory_mbs = [
        call.args[0] for call in engine.train_batch_per_trajectory.call_args_list
    ]
    assert [float(mb["task_reward"].flatten()[0]) for mb in per_trajectory_mbs] == [
        pytest.approx(-0.5),
        pytest.approx(1.25),
    ]
    assert [mb["uid"] for mb in per_trajectory_mbs] == [["uid-1"], ["uid-0"]]
    assert [mb["traj_uid"] for mb in per_trajectory_mbs] == [
        ["traj-1"],
        ["traj-0"],
    ]
    assert [mb["trainer_global_step"].tolist() for mb in per_trajectory_mbs] == [
        [8],
        [7],
    ]


def test_per_trajectory_scalar_metadata_from_concat_reordered_with_minibatches(
    monkeypatch,
):
    from areal.utils.data import concat_batch

    recorder = _StatsRecorder()
    monkeypatch.setattr(actor_module, "stats_tracker", recorder)
    engine = _FakeEngine(per_trajectory_stats=[{"lr": 2.0}, {"lr": 3.0}])
    actor = PPOActor(
        PPOActorConfig(
            disable_dropout=True,
            ppo_n_minibatches=2,
            per_trajectory=PerTrajectoryConfig(enabled=True),
        ),
        engine,
    )
    first = _minimal_batch(batch_size=1)
    first["uid"] = "uid-0"
    first["traj_uid"] = "traj-0"
    first["rid"] = 100
    first["task_id"] = "task-0"
    first["trainer_global_step"] = torch.tensor([7])
    second = _minimal_batch(batch_size=1)
    second["uid"] = "uid-1"
    second["traj_uid"] = "traj-1"
    second["rid"] = 101
    second["task_id"] = "task-1"
    second["trainer_global_step"] = torch.tensor([8])
    batch, _ = concat_batch([first, second])

    def fake_splitter(data, mb_spec):
        assert data["uid"] == ["uid-0", "uid-1"]
        assert data["traj_uid"] == ["traj-0", "traj-1"]
        assert data["rid"] == [100, 101]
        assert data["task_id"] == ["task-0", "task-1"]
        return type(
            "FakeMicroBatchList",
            (),
            {
                "forward_indices": [1, 0],
                "mbs": [
                    {"attention_mask": torch.ones(1, 3, dtype=torch.bool)},
                    {"attention_mask": torch.ones(1, 3, dtype=torch.bool)},
                ],
            },
        )()

    monkeypatch.setattr(
        actor_module,
        "split_padded_tensor_dict_into_mb_list",
        fake_splitter,
    )

    actor._ppo_update(batch)

    per_trajectory_mbs = [
        call.args[0] for call in engine.train_batch_per_trajectory.call_args_list
    ]
    assert [mb["uid"] for mb in per_trajectory_mbs] == [["uid-1"], ["uid-0"]]
    assert [mb["traj_uid"] for mb in per_trajectory_mbs] == [
        ["traj-1"],
        ["traj-0"],
    ]
    assert [mb["rid"] for mb in per_trajectory_mbs] == [[101], [100]]
    assert [mb["task_id"] for mb in per_trajectory_mbs] == [
        ["task-1"],
        ["task-0"],
    ]


def test_per_trajectory_task_reward_uses_raw_reward_before_overlong_penalty(
    monkeypatch,
):
    recorder = _StatsRecorder()
    monkeypatch.setattr(actor_module, "stats_tracker", recorder)
    engine = _FakeEngine(per_trajectory_stats=[{"lr": 2.0}, {"lr": 3.0}])
    actor = PPOActor(
        PPOActorConfig(
            disable_dropout=True,
            ppo_n_minibatches=2,
            overlong_reward_penalty=True,
            overlong_tokens=1,
            overlong_penalty_factor=10.0,
            per_trajectory=PerTrajectoryConfig(enabled=True),
        ),
        engine,
    )
    raw_rewards = torch.tensor([1.25, -0.5])
    penalty = torch.tensor([10.0, 20.0])
    batch = _minimal_batch(batch_size=2)
    batch["rewards"] = raw_rewards.clone()

    def fake_overlong_penalty(data, **kwargs):
        torch.testing.assert_close(data["rewards"], raw_rewards)
        data["rewards"].sub_(penalty)
        return data

    def fake_splitter(data, mb_spec):
        assert "rewards" not in data
        torch.testing.assert_close(data["task_reward"], raw_rewards)
        return type(
            "FakeMicroBatchList",
            (),
            {
                "forward_indices": [1, 0],
                "mbs": [
                    {"attention_mask": torch.ones(1, 3, dtype=torch.bool)},
                    {"attention_mask": torch.ones(1, 3, dtype=torch.bool)},
                ],
            },
        )()

    monkeypatch.setattr(
        actor_module,
        "reward_overlong_penalty",
        fake_overlong_penalty,
    )
    monkeypatch.setattr(
        actor_module,
        "split_padded_tensor_dict_into_mb_list",
        fake_splitter,
    )

    advantage_data = actor._compute_advantages(batch)
    torch.testing.assert_close(advantage_data["rewards"], raw_rewards - penalty)

    actor._ppo_update(advantage_data)

    per_trajectory_mbs = [
        call.args[0] for call in engine.train_batch_per_trajectory.call_args_list
    ]
    torch.testing.assert_close(per_trajectory_mbs[0]["task_reward"], raw_rewards[[1]])
    torch.testing.assert_close(per_trajectory_mbs[1]["task_reward"], raw_rewards[[0]])


def test_per_trajectory_enabled_requires_engine_method():
    engine = _FakeEngine()
    delattr(engine, "train_batch_per_trajectory")
    actor = PPOActor(
        PPOActorConfig(
            disable_dropout=True,
            ppo_n_minibatches=1,
            per_trajectory=PerTrajectoryConfig(enabled=True),
        ),
        engine,
    )

    with pytest.raises(RuntimeError, match="Megatron"):
        actor._ppo_update(_minimal_batch())


def test_invalid_config_without_per_trajectory_is_not_silently_masked():
    engine = _FakeEngine()
    actor = PPOActor(PPOActorConfig(disable_dropout=True, ppo_n_minibatches=1), engine)
    delattr(actor.config, "per_trajectory")

    with pytest.raises(AttributeError, match="per_trajectory"):
        actor._ppo_update(_minimal_batch())

    engine.train_batch.assert_not_called()
