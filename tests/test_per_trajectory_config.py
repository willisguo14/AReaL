import pytest

from areal.api.cli_args import (
    ESSScalingConfig,
    MicroBatchSpec,
    PerTrajectoryConfig,
    PerTrajectoryFilterConfig,
    PPOActorConfig,
)


def test_per_trajectory_config_defaults_disabled():
    config = PerTrajectoryConfig()

    assert config.enabled is False
    assert config.flush_threshold == 256
    assert config.mask_filters == []


def test_per_trajectory_config_accepts_enabled_with_mask_filters():
    config = PerTrajectoryConfig(
        enabled=True,
        flush_threshold=8,
        mask_filters=[
            PerTrajectoryFilterConfig(
                rule="grad_norm_exceeds_max",
                params={"max": 3.5},
            )
        ],
    )

    assert config.enabled is True
    assert config.flush_threshold == 8
    assert len(config.mask_filters) == 1
    assert config.mask_filters[0].rule == "grad_norm_exceeds_max"
    assert config.mask_filters[0].params == {"max": 3.5}


def test_per_trajectory_filter_config_accepts_generic_params():
    config = PerTrajectoryFilterConfig(
        rule="kl_k1_outside_range",
        params={"lower": -0.1, "upper": 0.2},
    )

    assert config.rule == "kl_k1_outside_range"
    assert config.params == {"lower": -0.1, "upper": 0.2}


@pytest.mark.parametrize("mask_filters", [None, object()])
def test_per_trajectory_config_rejects_invalid_mask_filters_list(mask_filters):
    with pytest.raises(ValueError, match="mask_filters.*sequence"):
        PerTrajectoryConfig(mask_filters=mask_filters)


@pytest.mark.parametrize("rule", ["missing", "", "grad_norm_max", "kl_k1_range"])
def test_per_trajectory_filter_config_rejects_unknown_rule(rule):
    with pytest.raises(ValueError, match="unknown per-trajectory filter rule"):
        PerTrajectoryFilterConfig(rule=rule)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_grad_norm_exceeds_max_filter_rejects_invalid_max(value):
    with pytest.raises(ValueError, match="grad_norm_exceeds_max.*max"):
        PerTrajectoryFilterConfig(
            rule="grad_norm_exceeds_max",
            params={"max": value},
        )


def test_grad_norm_exceeds_max_filter_requires_max_param():
    with pytest.raises(ValueError, match="grad_norm_exceeds_max.*max"):
        PerTrajectoryFilterConfig(rule="grad_norm_exceeds_max", params={})


def test_kl_k1_outside_range_filter_requires_at_least_one_bound():
    with pytest.raises(ValueError, match="kl_k1_outside_range.*lower.*upper"):
        PerTrajectoryFilterConfig(rule="kl_k1_outside_range", params={})


def test_kl_k1_outside_range_filter_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lower.*upper"):
        PerTrajectoryFilterConfig(
            rule="kl_k1_outside_range",
            params={"lower": 0.2, "upper": -0.2},
        )


def test_kl_k1_zscore_filter_accepts_n():
    config = PerTrajectoryFilterConfig(
        rule="kl_k1_zscore_exceeds",
        params={"n": 2.0},
    )

    assert config.rule == "kl_k1_zscore_exceeds"
    assert config.params == {"n": 2.0}


@pytest.mark.parametrize("params", [{}, {"n": -1.0}, {"n": float("inf")}])
def test_kl_k1_zscore_filter_rejects_invalid_n(params):
    with pytest.raises(ValueError, match="kl_k1_zscore_exceeds.*n"):
        PerTrajectoryFilterConfig(rule="kl_k1_zscore_exceeds", params=params)


def test_advantage_mean_positive_rejects_params():
    with pytest.raises(ValueError, match="does not accept params"):
        PerTrajectoryFilterConfig(
            rule="advantage_mean_positive",
            params={"threshold": 0.0},
        )


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


def test_enabled_per_trajectory_accepts_ess_scaling():
    config = PPOActorConfig(
        per_trajectory=PerTrajectoryConfig(enabled=True),
        disable_dropout=True,
        use_decoupled_loss=True,
        ess_scaling=ESSScalingConfig(),
    )

    assert config.per_trajectory.enabled is True
    assert config.ess_scaling is not None


def test_enabled_per_trajectory_rejects_microbatch_granularity_grouping():
    with pytest.raises(ValueError, match="granularity"):
        PPOActorConfig(
            per_trajectory=PerTrajectoryConfig(enabled=True),
            disable_dropout=True,
            mb_spec=MicroBatchSpec(granularity=2),
        )
