import math

import pytest
import torch

from areal.engine.core import (
    LOGP_GRAD_ABSMAX_KEY,
    LOGP_GRAD_NORM_KEY,
    LogprobGradAccumulator,
)
from areal.utils.functional import ppo_actor_loss_fn


def _ppo_loss(logprobs: torch.Tensor) -> torch.Tensor:
    proximal_logprobs = torch.tensor([[0.0, -0.1, 0.2]], dtype=torch.float32)
    old_logprobs = torch.zeros_like(proximal_logprobs)
    advantages = torch.tensor([[1.5, -0.7, 2.0]], dtype=torch.float32)
    loss_mask = torch.tensor([[True, True, False]])
    loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=loss_mask,
    )
    return loss


def test_logprob_grad_accumulator_matches_autograd_and_finite_difference():
    logprobs = torch.tensor(
        [[0.05, -0.2, 0.3]], dtype=torch.float32, requires_grad=True
    )
    loss = _ppo_loss(logprobs)
    loss_scale = torch.tensor(3.25)

    accumulator = LogprobGradAccumulator("cpu")
    accumulator.add(loss, logprobs, loss_scale)
    stats = accumulator.summary()

    grad = torch.autograd.grad(loss * loss_scale, logprobs, retain_graph=True)[0]
    assert stats["logp_grad_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(grad)),
        rel=1e-6,
        abs=1e-7,
    )
    assert stats["logp_grad_absmax"] == pytest.approx(
        float(grad.abs().max()),
        rel=1e-6,
        abs=1e-7,
    )

    eps = 1e-3
    finite_diff = torch.zeros_like(logprobs)
    for idx in range(logprobs.numel()):
        plus = logprobs.detach().clone()
        minus = logprobs.detach().clone()
        plus.view(-1)[idx] += eps
        minus.view(-1)[idx] -= eps
        finite_diff.view(-1)[idx] = (
            (_ppo_loss(plus) * loss_scale) - (_ppo_loss(minus) * loss_scale)
        ) / (2 * eps)

    torch.testing.assert_close(grad, finite_diff, rtol=2e-3, atol=2e-3)


def test_logprob_grad_accumulator_sums_squares_across_microbatches():
    weights = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
    logprobs = torch.tensor([0.2, -0.1, 0.4, 0.7], dtype=torch.float32)
    total_weight = torch.tensor(float(logprobs.numel()))

    def local_loss(lp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (lp * w).sum() / lp.numel()

    full = logprobs.clone().requires_grad_(True)
    full_accum = LogprobGradAccumulator("cpu")
    full_accum.add(local_loss(full, weights), full, torch.tensor(1.0))
    full_stats = full_accum.summary()

    split_accum = LogprobGradAccumulator("cpu")
    for lp, w in zip(logprobs.chunk(2), weights.chunk(2), strict=True):
        local = lp.clone().requires_grad_(True)
        local_weight = torch.tensor(float(local.numel()))
        split_accum.add(
            local_loss(local, w),
            local,
            local_weight / total_weight,
        )
    split_stats = split_accum.summary()

    expected_norm = math.sqrt(float(torch.sum((weights / total_weight) ** 2)))
    assert full_stats["logp_grad_norm"] == pytest.approx(expected_norm)
    assert split_stats["logp_grad_norm"] == pytest.approx(expected_norm)
    assert split_stats["logp_grad_absmax"] == pytest.approx(
        float((weights / total_weight).abs().max())
    )


def test_logprob_grad_accumulator_merge_combines_sums_and_absmax():
    first = LogprobGradAccumulator("cpu")
    second = LogprobGradAccumulator("cpu")
    merged = LogprobGradAccumulator("cpu")

    first_logprobs = torch.tensor(
        [0.2, -0.1],
        dtype=torch.float32,
        requires_grad=True,
    )
    second_logprobs = torch.tensor(
        [0.7, -0.3],
        dtype=torch.float32,
        requires_grad=True,
    )

    first.add(
        (first_logprobs * torch.tensor([1.0, -2.0])).sum(),
        first_logprobs,
        0.5,
    )
    second.add(
        (second_logprobs * torch.tensor([3.0, -4.0])).sum(),
        second_logprobs,
        0.25,
    )

    first_stats = first.summary()
    second_stats = second.summary()

    merged.merge(first)
    merged.merge(second)
    merged_stats = merged.summary()

    expected_norm = math.sqrt(
        first_stats[LOGP_GRAD_NORM_KEY] ** 2 + second_stats[LOGP_GRAD_NORM_KEY] ** 2
    )
    expected_absmax = max(
        first_stats[LOGP_GRAD_ABSMAX_KEY],
        second_stats[LOGP_GRAD_ABSMAX_KEY],
    )
    assert merged_stats[LOGP_GRAD_NORM_KEY] == pytest.approx(expected_norm)
    assert merged_stats[LOGP_GRAD_ABSMAX_KEY] == pytest.approx(expected_absmax)
