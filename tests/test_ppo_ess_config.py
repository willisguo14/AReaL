import pytest

from areal.api.cli_args import ESSScalingConfig, PPOActorConfig
from areal.utils.constants import PROX_LOGP_METHOD_LOGLINEAR


def test_ess_scaling_config_defaults():
    config = ESSScalingConfig()

    assert config.base_ess_ratio == 1.0
    assert config.min_lr_scale == 0.0
    assert config.max_lr_scale == 1.0


def test_ess_scaling_config_rejects_nonpositive_base_ratio():
    with pytest.raises(ValueError, match="base_ess_ratio"):
        ESSScalingConfig(base_ess_ratio=0.0)


def test_ess_scaling_config_rejects_negative_min_scale():
    with pytest.raises(ValueError, match="min_lr_scale"):
        ESSScalingConfig(min_lr_scale=-0.1)


def test_ess_scaling_config_rejects_min_greater_than_max():
    with pytest.raises(ValueError, match="min_lr_scale"):
        ESSScalingConfig(min_lr_scale=0.8, max_lr_scale=0.5)


def test_ppo_actor_config_rejects_non_decoupled_ess_scaling():
    with pytest.raises(ValueError, match="use_decoupled_loss"):
        PPOActorConfig(use_decoupled_loss=False, ess_scaling=ESSScalingConfig())


def test_ppo_actor_config_rejects_skip_forward_prox_method():
    with pytest.raises(ValueError, match="prox_logp_method"):
        PPOActorConfig(
            use_decoupled_loss=True,
            prox_logp_method=PROX_LOGP_METHOD_LOGLINEAR,
            ess_scaling=ESSScalingConfig(),
        )
