import pytest

from areal.api.cli_args import (
    ESSScalingConfig,
    MicroBatchSpec,
    PerTrajectoryConfig,
    PPOActorConfig,
)


def test_per_trajectory_config_defaults_disabled():
    config = PerTrajectoryConfig()

    assert config.enabled is False
    assert config.flush_threshold == 256


def test_per_trajectory_config_accepts_enabled():
    config = PerTrajectoryConfig(enabled=True, flush_threshold=8)

    assert config.enabled is True
    assert config.flush_threshold == 8


def test_actor_config_accepts_disabled_per_trajectory_with_defaults():
    config = PPOActorConfig()

    assert config.per_trajectory.enabled is False


def test_enabled_per_trajectory_requires_dropout_disabled():
    with pytest.raises(ValueError, match="disable_dropout"):
        PPOActorConfig(
            per_trajectory=PerTrajectoryConfig(enabled=True),
            disable_dropout=False,
        )


def test_enabled_per_trajectory_rejects_m2():
    with pytest.raises(ValueError, match="m2_threshold"):
        PPOActorConfig(
            per_trajectory=PerTrajectoryConfig(enabled=True),
            disable_dropout=True,
            m2_threshold=0.04,
        )


def test_enabled_per_trajectory_rejects_ess_scaling():
    with pytest.raises(ValueError, match="ess_scaling"):
        PPOActorConfig(
            per_trajectory=PerTrajectoryConfig(enabled=True),
            disable_dropout=True,
            use_decoupled_loss=True,
            ess_scaling=ESSScalingConfig(),
        )


def test_enabled_per_trajectory_rejects_microbatch_granularity_grouping():
    with pytest.raises(ValueError, match="granularity"):
        PPOActorConfig(
            per_trajectory=PerTrajectoryConfig(enabled=True),
            disable_dropout=True,
            mb_spec=MicroBatchSpec(granularity=2),
        )
