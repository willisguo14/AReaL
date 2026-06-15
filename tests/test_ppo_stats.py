from unittest.mock import MagicMock, patch

import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.trainer.ppo.critic import ppo_loss_fn
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils.stats_tracker import DistributedStatsTracker


def test_infer_token_denominator_prefers_attention_mask():
    input_data = {
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "input_ids": torch.tensor([[11, 12], [13, 14]]),
    }

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(5))

    assert n_tokens.shape == torch.Size([2, 3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_uses_input_ids_when_attention_mask_missing():
    input_data = {"input_ids": torch.tensor([[11, 12, 13], [14, 15, 16]])}

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(2, 3))

    assert n_tokens.shape == torch.Size([2, 3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_falls_back_for_padded_tree_input_ids():
    input_data = {"input_ids": torch.tensor([11, 12, 13, 0])}

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(3))

    assert n_tokens.shape == torch.Size([3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_falls_back_when_metadata_is_missing():
    fallback = torch.zeros(4)

    n_tokens = infer_token_denominator({"logprobs": torch.zeros(2)}, fallback=fallback)

    assert n_tokens.shape == torch.Size([4])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_grpo_loss_fn_uses_full_cu_seqlens_for_n_tokens():
    input_data = {
        "input_ids": torch.tensor([11, 12]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "logprobs": torch.zeros(2),
        "advantages": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
        "prox_logp": torch.zeros(2),
        "versions": torch.zeros(2, dtype=torch.int32),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()
        mock_tracker.scope = MagicMock()
        mock_tracker.scope.return_value.__enter__ = MagicMock()
        mock_tracker.scope.return_value.__exit__ = MagicMock()

        grpo_loss_fn(
            logprobs=torch.zeros(2),
            entropy=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )

    n_tokens = next(
        call.kwargs["n_tokens"]
        for call in mock_tracker.denominator.call_args_list
        if "n_tokens" in call.kwargs
    )
    assert n_tokens.shape == torch.Size([4])
    assert torch.all(n_tokens)


def test_grpo_loss_fn_trace_stat_callback_receives_behavior_stats():
    prox_logp = torch.log(torch.tensor([[1.0, 2.0, 3.0]]))
    input_data = {
        "input_ids": torch.tensor([[11, 12, 13]]),
        "logprobs": torch.zeros(1, 3),
        "advantages": torch.ones(1, 3),
        "loss_mask": torch.tensor([[True, True, False]]),
        "prox_logp": prox_logp,
    }
    captured_stats = []

    with patch("areal.trainer.ppo.actor.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()
        mock_tracker.scalar = MagicMock()

        grpo_loss_fn(
            logprobs=torch.zeros(1, 3),
            entropy=torch.zeros(1, 3),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            rejection_sampling=RejectionSamplingConfig(
                level="token",
                action="clamp",
                metric="ratio",
                upper=10.0,
            ),
            trace_stat_callback=captured_stats.append,
        )

    assert len(captured_stats) == 1
    stat = captured_stats[0]
    torch.testing.assert_close(
        stat["behave_imp_weight"],
        torch.tensor([[1.0, 2.0, 0.0]]),
    )
    torch.testing.assert_close(
        stat["behave_approx_kl"],
        torch.tensor([[0.0, torch.log(torch.tensor(2.0)).item(), 0.0]]),
    )
    assert torch.equal(
        stat["behave_mask"],
        torch.tensor([[True, True, False]]),
    )


def test_grpo_loss_fn_trace_stat_callback_receives_loss_advantage_and_entropy():
    entropy = torch.tensor([[0.25, 0.75, 9.0]])
    input_data = {
        "input_ids": torch.tensor([[11, 12, 13]]),
        "logprobs": torch.zeros(1, 3),
        "advantages": torch.tensor([[1.0, 3.0, 99.0]]),
        "loss_mask": torch.tensor([[True, True, False]]),
        "prox_logp": torch.zeros(1, 3),
    }
    captured_stats = []

    with patch("areal.trainer.ppo.actor.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()
        mock_tracker.scalar = MagicMock()

        grpo_loss_fn(
            logprobs=torch.zeros(1, 3),
            entropy=entropy,
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            importance_sampling_level="sequence",
            trace_stat_callback=captured_stats.append,
        )

    assert len(captured_stats) == 1
    stat = captured_stats[0]
    torch.testing.assert_close(stat["entropy"], entropy)
    torch.testing.assert_close(
        stat["loss_advantage"],
        torch.tensor([[2.0, 2.0, 0.0]]),
    )
    assert stat["loss_mask"].dtype == torch.bool
    assert torch.equal(
        stat["loss_mask"],
        torch.tensor([[True, True, False]]),
    )


def test_critic_loss_fn_uses_full_cu_seqlens_for_n_tokens():
    input_data = {
        "input_ids": torch.tensor([11, 12]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "values": torch.zeros(2),
        "returns": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
    }

    with patch("areal.trainer.ppo.critic.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()

        ppo_loss_fn(
            value=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
        )

    n_tokens = mock_tracker.denominator.call_args.kwargs["n_tokens"]
    assert n_tokens.shape == torch.Size([4])
    assert torch.all(n_tokens)


def test_grpo_loss_fn_uses_packed_denominator_for_tree_vocab_stats():
    tracker = DistributedStatsTracker()
    input_data = {
        "input_ids": torch.tensor([11, 12, 13, 0]),
        "logprobs": torch.zeros(3),
        "advantages": torch.ones(3),
        "loss_mask": torch.ones(3, dtype=torch.bool),
        "prox_logp": torch.zeros(3),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        grpo_loss_fn(
            logprobs=torch.zeros(3),
            entropy=torch.zeros(3),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            vocab_min_logits=torch.zeros(3),
            vocab_max_logits=torch.zeros(3),
        )

    stats = tracker.export(reset=True)
    assert "n_tokens" in stats
