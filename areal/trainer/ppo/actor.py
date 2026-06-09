# SPDX-License-Identifier: Apache-2.0

import functools
import math
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from areal.api import TrainEngine
from areal.api.cli_args import MicroBatchSpec, PPOActorConfig, RejectionSamplingConfig
from areal.engine.core import LOGP_GRAD_ABSMAX_KEY
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.trainer.ppo.ess import (
    compute_sequence_ess,
    compute_token_ess,
    summarize_ess_lr_scale_stats,
    summarize_ess_stats,
)
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import logging, stats_tracker
from areal.utils.constants import (
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_LOGP_METHOD_RECOMPUTE,
    ProxLogpMethod,
)
from areal.utils.data import (
    KLEstimator,
    Normalization,
    batched_call,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import (
    ppo_actor_loss_fn,
    reward_overlong_penalty,
    sapo_loss_fn,
)
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger("PPOActor")


_PER_TRAJECTORY_BATCH_LIST_METADATA_KEYS = frozenset(
    {"ids", "traj_uid", "uid", "rid", "task_id"}
)
_PER_TRAJECTORY_BATCH_TENSOR_METADATA_KEYS = frozenset(
    {
        "begin_of_trajectory",
        "rid",
        "task_id",
        "task_reward",
        "trainer_global_step",
        "traj_uid",
        "uid",
        "versions",
    }
)


def _insert_per_trajectory_minibatch_metadata(
    data: dict[str, Any],
    mb_inputs: Any,
) -> None:
    batch_size = int(data["attention_mask"].shape[0])
    forward_indices_raw = getattr(mb_inputs, "forward_indices", None)
    if forward_indices_raw is None:
        raise RuntimeError(
            "actor.per_trajectory requires minibatch forward_indices metadata."
        )
    forward_indices = list(forward_indices_raw)
    if len(forward_indices) != batch_size:
        raise RuntimeError(
            "actor.per_trajectory minibatch forward_indices must match batch size."
        )

    ordered_metadata: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if (
                key not in _PER_TRAJECTORY_BATCH_TENSOR_METADATA_KEYS
                or value.ndim == 0
                or int(value.shape[0]) != batch_size
            ):
                continue
            index = torch.as_tensor(
                forward_indices,
                dtype=torch.long,
                device=value.device,
            )
            ordered_metadata[key] = value.index_select(0, index)
        elif (
            isinstance(value, list)
            and key in _PER_TRAJECTORY_BATCH_LIST_METADATA_KEYS
            and len(value) == batch_size
        ):
            ordered_metadata[key] = [value[index] for index in forward_indices]

    offset = 0
    for mb in mb_inputs.mbs:
        mb_size = int(mb["attention_mask"].shape[0])
        for key, ordered_value in ordered_metadata.items():
            mb[key] = ordered_value[offset : offset + mb_size]
        offset += mb_size

    if offset != batch_size:
        raise RuntimeError(
            "actor.per_trajectory minibatch sequence counts must match batch size."
        )


def _summarize_grad_cos_sims(values: list[float]) -> dict[str, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return {}

    abs_values = [abs(value) for value in finite_values]
    return {
        "grad_cos_sim_abs/max": max(abs_values),
        "grad_cos_sim_abs/min": min(abs_values),
        "grad_cos_sim_abs/avg": sum(abs_values) / len(abs_values),
        "grad_cos_sim/max": max(finite_values),
        "grad_cos_sim/min": min(finite_values),
    }


def _has_exact_ess_inputs(data: dict[str, Any]) -> bool:
    return (
        data.get("prox_logp") is not None
        and data.get("logprobs") is not None
        and data.get("loss_mask") is not None
    )


def _missing_exact_ess_inputs(data: dict[str, Any]) -> list[str]:
    return [
        key for key in ("prox_logp", "logprobs", "loss_mask") if data.get(key) is None
    ]


def _should_all_reduce_exact_ess_inputs(process_group: Any) -> bool:
    return process_group is not None and dist.is_available() and dist.is_initialized()


def _first_tensor_device(data: dict[str, Any]) -> torch.device | None:
    for value in data.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def _dist_backend_name(process_group: Any) -> str:
    get_backend = getattr(dist, "get_backend", None)
    if get_backend is None:
        return ""
    try:
        backend = get_backend(process_group)
    except TypeError:
        backend = get_backend(group=process_group)
    return str(backend).lower()


def _ess_collective_device(
    data: dict[str, Any],
    *,
    process_group: Any,
    engine: Any,
) -> torch.device | None:
    if not _should_all_reduce_exact_ess_inputs(process_group):
        return None

    backend_name = _dist_backend_name(process_group)
    if "gloo" in backend_name:
        return torch.device("cpu")
    if backend_name:
        engine_device = getattr(engine, "device", None)
        if engine_device is not None:
            return torch.device(engine_device)
    return _first_tensor_device(data)


def _has_exact_ess_inputs_on_all_dp_ranks(
    data: dict[str, Any],
    process_group: Any,
    collective_device: torch.device | None,
) -> bool:
    local_has_exact_inputs = _has_exact_ess_inputs(data)
    if not _should_all_reduce_exact_ess_inputs(process_group):
        return local_has_exact_inputs

    counts = torch.tensor(
        [int(local_has_exact_inputs), 1],
        dtype=torch.int64,
        device=collective_device
        if collective_device is not None
        else _first_tensor_device(data),
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=process_group)

    exact_rank_count = int(counts[0].item())
    total_rank_count = int(counts[1].item())
    if exact_rank_count == total_rank_count:
        return True
    if exact_rank_count == 0:
        return False

    missing = _missing_exact_ess_inputs(data)
    local_state = (
        f"local missing: {', '.join(missing)}"
        if missing
        else "local rank has all exact ESS inputs"
    )
    raise RuntimeError(
        "Mixed exact ESS inputs across DP ranks would make ESS collectives unsafe; "
        f"{exact_rank_count}/{total_rank_count} DP ranks have exact ESS inputs "
        f"and {local_state}."
    )


def _ess_tensors_on_collective_device(
    mb: dict[str, Any],
    collective_device: torch.device | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prox_logp = mb["prox_logp"]
    logprobs = mb["logprobs"]
    loss_mask = mb["loss_mask"]
    if collective_device is None:
        return prox_logp, logprobs, loss_mask
    return (
        prox_logp.to(device=collective_device),
        logprobs.to(device=collective_device),
        loss_mask.to(device=collective_device),
    )


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature

        self.m2_threshold = config.m2_threshold

        # Log critical GSPO/GRPO configuration for reproducibility
        self._log_configuration()

    def _log_configuration(self):
        """Log PPO configuration including how proximal policy is computed."""
        config = self.config

        logger.info("=" * 70)
        logger.info("PPOActor Configuration")
        logger.info("=" * 70)

        # Log PPO mode and proximal policy computation
        if not config.use_decoupled_loss:
            logger.info("Mode: Standard PPO (on-policy)")
            if config.recompute_logprob:
                logger.info("  old_logp (π_old): RECOMPUTED from current policy")
            else:
                logger.info(
                    "  old_logp (π_old): FROM INFERENCE (cached during rollout)"
                )
        else:
            logger.info("Mode: Decoupled PPO (off-policy)")
            logger.info("  log_p_behave (π_behave): FROM INFERENCE (behavior policy)")

            # Log proximal policy computation method
            method_descriptions = {
                PROX_LOGP_METHOD_RECOMPUTE: "RECOMPUTED via forward pass (standard decoupled PPO)",
                PROX_LOGP_METHOD_LOGLINEAR: "LOG-LINEAR APPROXIMATION (no forward pass)",
                PROX_LOGP_METHOD_METRICS: "RECOMPUTED + APPROXIMATION METRICS (for evaluation)",
            }
            desc = method_descriptions.get(
                config.prox_logp_method, f"UNKNOWN ({config.prox_logp_method})"
            )
            logger.info(f"  Proximal policy (π_prox): {desc}")

            logger.info("  log_p_theta (π_θ): TRAINING FORWARD PASS (current policy)")

            if config.rejection_sampling is not None:
                rs = config.rejection_sampling
                logger.info(
                    f"  Rejection sampling: level={rs.level}, metric={rs.metric}, "
                    f"action={rs.action}, upper={rs.upper}"
                    + (f", lower={rs.lower}" if rs.lower is not None else "")
                    + (f", agg={rs.agg}" if rs.level == "sequence" else "")
                )

        # Log other critical config
        logger.info("=" * 70)
        logger.info("Training Parameters:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'token')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(f"  eps_clip: {config.eps_clip}")
        logger.info("=" * 70)

    @trace_perf("ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor] | None:
        return batched_call(self._compute_logp, data)

    def _compute_logp(self, data: dict[str, Any]) -> torch.Tensor | None:
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return batched_call(self._compute_advantages, data)

    def _compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length
        if self.config.per_trajectory.enabled and "task_reward" not in data:
            data["task_reward"] = data["rewards"].detach().clone()

        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp

        return data

    @trace_perf("ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._ppo_update, data, unpack=False)

    def _ppo_update(self, data: dict[str, Any]) -> None:
        ess_scaling = getattr(self.config, "ess_scaling", None)
        if ess_scaling is not None and not self.config.use_decoupled_loss:
            raise RuntimeError(
                "ess_scaling requires use_decoupled_loss=True because exact "
                "proximal log probabilities are required for ESS."
            )
        if ess_scaling is not None and not getattr(
            self.engine, "supports_optimizer_step_scale", False
        ):
            raise RuntimeError(
                "ess_scaling requires an FSDP engine that supports "
                "optimizer_step_scale."
            )
        data_parallel_group = getattr(self.engine, "data_parallel_group", None)
        ess_collective_device = _ess_collective_device(
            data,
            process_group=data_parallel_group,
            engine=self.engine,
        )
        has_exact_ess_inputs = _has_exact_ess_inputs_on_all_dp_ranks(
            data, data_parallel_group, ess_collective_device
        )
        if ess_scaling is not None and not has_exact_ess_inputs:
            missing = _missing_exact_ess_inputs(data)
            raise RuntimeError(
                "ess_scaling requires exact ESS inputs: prox_logp, logprobs, "
                f"and loss_mask. Missing: {', '.join(missing)}."
            )
        should_compute_ess = self.config.use_decoupled_loss and has_exact_ess_inputs
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=infer_token_denominator(data, loss_mask),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.rejection_sampling is not None:
            rs = self.config.rejection_sampling
            scalars["rs_upper"] = rs.upper
            if rs.lower is not None:
                scalars["rs_lower"] = rs.lower
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )
        ########## Logging code ends ##########

        if self.config.per_trajectory.enabled:
            task_reward = data.get("task_reward", reward_score)
            if isinstance(task_reward, torch.Tensor):
                task_reward = task_reward.detach().clone()
            data["task_reward"] = task_reward

        # Pop keys that are no longer needed after advantage computation
        # Note: "versions" is kept if needed for approximation/metrics in loss function
        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )
        if self.config.per_trajectory.enabled:
            _insert_per_trajectory_minibatch_metadata(data, mb_inputs)

        with stats_tracker.scope("update"):
            # Get current version for proximal approximation metrics
            current_version = self.engine.get_version()
            grad_cos_sims: list[float] = []
            sequence_ess_stats = []
            token_ess_stats = []
            selected_ess_stats = []
            ess_lrs: list[float] = []
            logp_grad_absmax_values: list[float] = []

            for mb_idx, mb in enumerate(mb_inputs.mbs):
                sequence_ess_stat = None
                token_ess_stat = None
                selected_ess_stat = None
                if should_compute_ess:
                    prox_logp, logprobs, loss_mask = _ess_tensors_on_collective_device(
                        mb, ess_collective_device
                    )
                    sequence_ess_stat = compute_sequence_ess(
                        prox_logp=prox_logp,
                        logprobs=logprobs,
                        loss_mask=loss_mask,
                        scaling_config=ess_scaling,
                        process_group=data_parallel_group,
                    )
                    token_ess_stat = compute_token_ess(
                        prox_logp=prox_logp,
                        logprobs=logprobs,
                        loss_mask=loss_mask,
                        scaling_config=ess_scaling,
                        process_group=data_parallel_group,
                    )
                    if sequence_ess_stat is not None:
                        sequence_ess_stats.append(sequence_ess_stat)
                    if token_ess_stat is not None:
                        token_ess_stats.append(token_ess_stat)
                    if ess_scaling is not None:
                        if ess_scaling.level == "sequence":
                            selected_ess_stat = sequence_ess_stat
                        elif ess_scaling.level == "token":
                            selected_ess_stat = token_ess_stat
                        else:
                            raise ValueError(
                                "ess_scaling.level must be one of 'sequence' or "
                                f"'token'; got {ess_scaling.level!r}"
                            )
                        if selected_ess_stat is not None:
                            selected_ess_stats.append(selected_ess_stat)

                train_batch_kwargs = {}
                if ess_scaling is not None:
                    train_batch_kwargs["optimizer_step_scale"] = (
                        selected_ess_stat.ess_lr_scale
                        if selected_ess_stat is not None
                        else 1.0
                    )

                actor_loss_fn = functools.partial(
                    grpo_loss_fn,
                    eps_clip=self.config.eps_clip,
                    eps_clip_higher=self.config.eps_clip_higher,
                    c_clip=self.config.c_clip,
                    rejection_sampling=self.config.rejection_sampling,
                    m2_threshold=self.m2_threshold,
                    importance_sampling_level=self.config.importance_sampling_level,
                    current_version=current_version,
                    prox_logp_method=self.config.prox_logp_method,
                    use_sapo_loss=self.config.use_sapo_loss,
                    sapo_tau_pos=self.config.sapo_tau_pos,
                    sapo_tau_neg=self.config.sapo_tau_neg,
                    use_decoupled_loss=self.config.use_decoupled_loss,
                )
                common_kwargs = {
                    "collect_logprob_grad_stats": True,
                    "loss_fn": actor_loss_fn,
                    "loss_weight_fn": lambda x: x["loss_mask"].count_nonzero(),
                    **train_batch_kwargs,
                }
                if self.config.per_trajectory.enabled:
                    train_batch_fn = getattr(
                        self.engine, "train_batch_per_trajectory", None
                    )
                    if train_batch_fn is None:
                        raise RuntimeError(
                            "actor.per_trajectory.enabled requires a Megatron "
                            "engine with train_batch_per_trajectory support."
                        )
                    train_stat = train_batch_fn(
                        mb,
                        minibatch_idx=mb_idx,
                        **common_kwargs,
                    )
                else:
                    train_stat = self.engine.train_batch(mb, **common_kwargs)
                grad_cos_sim = train_stat.pop("grad_cos_sim", None)
                if grad_cos_sim is not None and math.isfinite(float(grad_cos_sim)):
                    grad_cos_sims.append(float(grad_cos_sim))
                logp_grad_absmax = train_stat.pop(
                    LOGP_GRAD_ABSMAX_KEY,
                    None,
                )
                if logp_grad_absmax is not None:
                    value = float(logp_grad_absmax)
                    if math.isfinite(value):
                        logp_grad_absmax_values.append(value)
                if (
                    ess_scaling is not None
                    and selected_ess_stat is not None
                    and "lr" in train_stat
                ):
                    try:
                        effective_lr = float(train_stat["lr"]) * float(
                            train_batch_kwargs["optimizer_step_scale"]
                        )
                    except (TypeError, ValueError):
                        pass
                    else:
                        if math.isfinite(effective_lr):
                            ess_lrs.append(effective_lr)
                stats_tracker.scalar(**train_stat)

            ess_summary = {}
            ess_summary.update(
                summarize_ess_stats(sequence_ess_stats, level="sequence")
            )
            ess_summary.update(summarize_ess_stats(token_ess_stats, level="token"))
            if ess_scaling is not None:
                ess_summary.update(summarize_ess_lr_scale_stats(selected_ess_stats))
            if ess_scaling is not None and ess_lrs:
                ess_summary["ess_lr/avg"] = sum(ess_lrs) / len(ess_lrs)
                ess_summary["ess_lr/min"] = min(ess_lrs)
                ess_summary["ess_lr/max"] = max(ess_lrs)
            if ess_summary:
                stats_tracker.scalar(**ess_summary)

            grad_cos_summary = _summarize_grad_cos_sims(grad_cos_sims)
            if grad_cos_summary:
                stats_tracker.scalar(**grad_cos_summary)
            if logp_grad_absmax_values:
                stats_tracker.scalar(
                    **{LOGP_GRAD_ABSMAX_KEY: max(logp_grad_absmax_values)}
                )


class PPOActorController(TrainController):
    def compute_logp(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_logp", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def compute_advantages(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_advantages", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def ppo_update(self, *args, **kwargs) -> None:
        self._custom_function_call(
            "ppo_update", *args, rpc_meta={"broadcast": True}, **kwargs
        )


class PPOActorControllerV2(GatewayTrainController):
    def compute_logp(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/ppo/actor/compute_logp", payload)

    def compute_advantages(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/ppo/actor/compute_advantages", payload)

    def ppo_update(self, *args, **kwargs) -> None:
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post("/ppo/actor/update", payload)


def grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    rejection_sampling: RejectionSamplingConfig | None = None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    use_decoupled_loss: bool = False,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
    trace_stat_callback: Callable[[dict[str, Any]], None] | None = None,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp_gt = input_data.get("prox_logp")  # Could be None if skipped

    entropy = entropy.detach()

    # Resolve proximal log-probabilities based on method
    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    # Apply M2PO masking if threshold is set
    if m2_threshold is not None:
        loss_mask = _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold)

    # Use SAPO or PPO loss
    if use_sapo_loss:
        if use_decoupled_loss:
            raise ValueError(
                "SAPO is not compatible with `use_decoupled_loss=True`. "
                "Please set `actor.use_decoupled_loss=false` in your configuration."
            )
        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            tau_pos=sapo_tau_pos,
            tau_neg=sapo_tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )
    else:
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            loss_mask=loss_mask,
            c_clip=c_clip,
            proximal_logprobs=prox_logp,
            rejection_sampling=rejection_sampling,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )

    if trace_stat_callback is not None:
        trace_stat_callback(stat)

    # Joint Distillation KL Loss
    teacher_logp = input_data.get("teacher_logp")
    rkl_stat = None
    if teacher_logp is not None:
        # Coefficients for RL and Knowledge Distillation
        rl_loss_weight = input_data.get("rl_loss_weight", 1.0)
        distill_loss_weight = input_data.get("distill_loss_weight", 0.005)

        teacher_logp = (
            teacher_logp.detach()
        )  # detach to prevent gradient backprop to teacher

        if rl_loss_weight == 0:
            # Pure KD using reverse KL (importance-sampling)
            rkl_reward = teacher_logp - logprobs.detach()
            importance_weight = torch.exp(logprobs - old_logp)

            rkl_weighted_term = importance_weight * rkl_reward * loss_mask

            kd_coef = -1 * distill_loss_weight
            loss = kd_coef * rkl_weighted_term.sum() / loss_mask.sum().clamp(min=1)

            rkl_stat = -1 * rkl_weighted_term
        else:
            # KDRL: Knowledge Distillation + Reinforcement Learning (joint loss)
            rkl_penalty_per_token = (logprobs - teacher_logp) * loss_mask
            rkl_penalty = rkl_penalty_per_token.sum() / loss_mask.sum().clamp(min=1)

            loss = rl_loss_weight * loss + distill_loss_weight * rkl_penalty

            rkl_stat = rkl_penalty_per_token

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=infer_token_denominator(input_data, loss_mask),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    if rkl_stat is not None:
        stats_tracker.stat(
            rkl_loss=rkl_stat,
            denominator="n_valid_tokens",
        )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )
    if "filtered_fraction" in stat:
        stats_tracker.scalar(rs_filtered_fraction=stat["filtered_fraction"])

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    # Log SAPO-specific statistics
    if use_sapo_loss:
        stats_tracker.stat(
            sapo_soft_gate=stat["sapo_soft_gate"],
            sapo_scaled_gate_pos=stat["sapo_scaled_gate_pos"],
            sapo_scaled_gate_neg=stat["sapo_scaled_gate_neg"],
            denominator="n_valid_tokens",
        )
    else:
        # Log clipping statistics (PPO only)
        clip_mask = stat["clip_mask"]
        clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
        clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
        stats_tracker.stat(
            clipped_new_logp=clipped_new_logp,
            clipped_old_logp=clipped_old_logp,
            denominator="clipped_tokens",
        )

    # Log proximal approximation metrics
    compute_logp_mask = stat.get("behave_mask", loss_mask)
    _log_proximal_approximation_stats(
        prox_logp_method=prox_logp_method,
        prox_logp_gt=prox_logp_gt,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
        compute_logp_mask=compute_logp_mask,
    )

    # Log version staleness metrics
    if "versions" in input_data and current_version is not None:
        version_metrics_mask = stat.get("behave_mask", loss_mask)
        _log_version_staleness_stats(
            versions=input_data["versions"],
            current_version=current_version,
            version_metrics_mask=version_metrics_mask,
        )

    return loss


# =============================================================================
# Core Functions
# =============================================================================


def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute approximation(s) for proximal policy log-probabilities.

    This function approximates the log-probabilities of the proximal policy (one training step
    behind the current policy) using version-aware interpolation between the behavior policy
    (old_logp) and current policy (logprobs). This avoids the need for an expensive forward pass
    to compute the proximal policy's log-probabilities explicitly.

    Args:
        old_logp: log_p_behave from the rollout (behavior policy)
        logprobs: log_p_theta from current training forward pass
        versions: per-token policy versions from rollout (v_behave for each token)
        current_version: current training step version (v_theta)
        method: If specified, only compute this method. If None, compute all methods.

    Returns:
        Dictionary with approximation results. Single key if method specified, all methods otherwise.
    """
    # Assume proximal version is current_version - 1 (last broadcast)
    # In AReaL, proximal policy is the last updated/broadcast policy version
    v_proximal = current_version - 1

    # Extract version information
    v_behave = versions.float()
    v_theta = float(current_version)

    # CRITICAL: Only approximate generated tokens (version >= 0)
    # Prompt tokens (version < 0) must NOT be approximated - they have no generation version
    generated_tokens_mask = versions >= 0

    # Compute interpolation factor alpha
    # When v_behave == v_proximal: alpha=0 (use old_logp)
    # When v_behave == v_theta: alpha=1 (use logprobs)
    # For prompt tokens (version < 0): alpha=0 (no interpolation)
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    # Avoid division by zero AND exclude prompt tokens
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}

    # If method is specified, only compute that one
    # Otherwise compute all methods (for metrics comparison)
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            # Method 1: Log-linear interpolation in log-space (geometric mean in probability space)
            # log(p_prox) = (1-α)·log(p_behave) + α·log(p_theta)
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )

        elif m == PROX_APPROX_METHOD_LINEAR:
            # Method 2: Linear interpolation in probability space (arithmetic mean)
            # p_prox = (1-α)·p_behave + α·p_theta
            # Then convert back to log space: log(p_prox)
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)

        elif m == PROX_APPROX_METHOD_ROLLOUT:
            # Method 3: Use behavior policy from rollout as-is (no approximation)
            # p_prox = p_behave
            # Used for metrics comparison
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor:
    """
    Resolve the proximal policy log-probabilities based on the method.

    This function determines the final proximal log-probabilities to use for PPO training,
    either from ground truth (forward pass) or approximation methods.

    Args:
        prox_logp_gt: Ground truth proximal logp (from forward pass), or None if skipped.
        prox_logp_method: Method to use (recompute, loglinear, metrics).
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (should be detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.

    Returns:
        Resolved proximal log-probabilities tensor.

    Raises:
        ValueError: If configuration is invalid (e.g., missing required data).
        RuntimeError: If computation fails (None result, NaN, Inf).
    """
    prox_logp_is_none = prox_logp_gt is None

    # Validate configuration when prox_logp is None
    if prox_logp_is_none:
        if not ProxLogpMethod(prox_logp_method).skips_forward_pass():
            raise ValueError(
                f"prox_logp is None but prox_logp_method='{prox_logp_method}'. "
                "This indicates compute_logp() was skipped incorrectly."
            )
        if versions is None:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{prox_logp_method}' "
                "but versions not available. "
                "Cannot proceed without either ground truth or approximation."
            )

    # Determine prox_logp based on method
    prox_logp = prox_logp_gt  # Default to ground truth (could be None)

    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        # Use loglinear approximation (must compute if prox_logp is None)
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]
    elif prox_logp_method == PROX_LOGP_METHOD_METRICS:
        # Metrics mode: use recomputed prox_logp for training,
        # but will also compute approximation metrics later
        pass  # Use prox_logp_gt as-is (should be recomputed)
    # else: PROX_LOGP_METHOD_RECOMPUTE - use prox_logp_gt as-is

    # Safety check: ensure we have prox_logp
    if prox_logp is None:
        raise RuntimeError(
            f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'. "
            "This indicates configuration or computation error."
        )

    # Verify the value is valid
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(
            f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'. "
            "This indicates computation failed."
        )

    return prox_logp


def _apply_m2po_masking(
    old_logp: torch.Tensor,
    prox_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Apply M2PO (Second-Momentum PPO) masking to filter high-variance tokens.

    M2PO filters out tokens with high second-momentum (squared difference between
    old and proximal log-probabilities) to reduce gradient variance.

    Args:
        old_logp: Behavior policy log-probabilities.
        prox_logp: Proximal policy log-probabilities.
        loss_mask: Original loss mask [batch, seq_len].
        m2_threshold: Threshold for second-momentum filtering.

    Returns:
        Updated loss mask with M2PO filtering applied.
    """
    delta = old_logp - prox_logp
    m2 = delta * delta
    mask_flat = loss_mask.view(-1)
    m2_selected = m2.view(-1)[mask_flat]

    if m2_selected.numel() == 0:
        return loss_mask

    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)
    sorted_m2_loss_mask = _get_m2po_loss_mask(
        sorted_m2=sorted_m2, m2_threshold=m2_threshold
    )
    m2_selected_mask = sorted_m2_loss_mask[restored_indices]

    m2_full_flat = torch.zeros_like(
        mask_flat, dtype=torch.bool, device=loss_mask.device
    )
    m2_full_flat[mask_flat] = m2_selected_mask

    return m2_full_flat.view_as(loss_mask)


def _get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.
    Mask the tokens whose second-momentum is the largest, until the average second-momentum is below the threshold.
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold.
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask


# =============================================================================
# Logging Helper Functions
# =============================================================================

_EPSILON = 1e-8  # Small constant for numerical stability in relative error calculations


def _compute_importance_weight(
    logp_numerator: torch.Tensor,
    logp_denominator: torch.Tensor,
) -> torch.Tensor:
    """Compute importance weight as exp(logp_num - logp_denom)."""
    return torch.exp(logp_numerator - logp_denominator).float()


def _compute_approximation_errors(
    ground_truth: torch.Tensor,
    approximation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute error metrics between ground truth and approximation.

    Returns:
        Dictionary with abs_error, rel_error, and squared_error tensors.
    """
    diff = ground_truth - approximation
    abs_error = torch.abs(diff).float()
    rel_error = torch.abs(diff / (torch.abs(ground_truth) + _EPSILON)).float()
    squared_error = (diff * diff).float()
    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "squared_error": squared_error,
    }


def _tensor_scalar_stats(tensor: torch.Tensor) -> dict[str, float]:
    """
    Compute scalar statistics (avg, max, min) for a tensor.

    Args:
        tensor: Input tensor to compute statistics on.

    Returns:
        Dictionary with avg, max, min as Python floats.
    """
    t = tensor.float()
    return {
        "avg": t.mean().item(),
        "max": t.max().item(),
        "min": t.min().item(),
    }


def _log_approximation_metrics_for_method(
    method_name: str,
    approx_logp: torch.Tensor,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    prox_logp_gt: torch.Tensor | None = None,
) -> None:
    """
    Log metrics for a single approximation method.

    Args:
        method_name: Name of the approximation method (e.g., "loglinear").
        approx_logp: Approximated proximal log-probabilities.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities.
        prox_logp_gt: Ground truth proximal logp, or None if unavailable.
    """
    # Compute importance weights from approximation
    behave_imp_weight = _compute_importance_weight(approx_logp, old_logp)
    importance_weight = _compute_importance_weight(logprobs, approx_logp)

    metrics = {
        f"{method_name}/approx_logp": approx_logp.float(),
        f"{method_name}/behave_imp_weight": behave_imp_weight,
        f"{method_name}/importance_weight": importance_weight,
    }

    # Add error metrics if ground truth is available
    if prox_logp_gt is not None:
        # Log-probability errors
        logp_errors = _compute_approximation_errors(prox_logp_gt, approx_logp)
        metrics.update(
            {
                f"{method_name}/abs_error": logp_errors["abs_error"],
                f"{method_name}/rel_error": logp_errors["rel_error"],
                f"{method_name}/squared_error": logp_errors["squared_error"],
            }
        )

        # Ground truth importance weights for comparison
        behave_imp_weight_gt = _compute_importance_weight(prox_logp_gt, old_logp)
        importance_weight_gt = _compute_importance_weight(logprobs, prox_logp_gt)

        # Importance weight errors
        behave_errors = _compute_approximation_errors(
            behave_imp_weight_gt, behave_imp_weight
        )
        imp_errors = _compute_approximation_errors(
            importance_weight_gt, importance_weight
        )

        metrics.update(
            {
                f"{method_name}/behave_imp_weight_abs_error": behave_errors[
                    "abs_error"
                ],
                f"{method_name}/behave_imp_weight_rel_error": behave_errors[
                    "rel_error"
                ],
                f"{method_name}/importance_weight_abs_error": imp_errors["abs_error"],
                f"{method_name}/importance_weight_rel_error": imp_errors["rel_error"],
            }
        )

    stats_tracker.stat(**metrics, denominator="n_valid_tokens")


def _log_proximal_approximation_stats(
    prox_logp_method: str,
    prox_logp_gt: torch.Tensor | None,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
    compute_logp_mask: torch.Tensor,
) -> None:
    """
    Log proximal policy approximation metrics based on the method.

    Args:
        prox_logp_method: The proximal logp method being used.
        prox_logp_gt: Ground truth proximal logp, or None if skipped.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.
        compute_logp_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("compute_logp"):
        stats_tracker.denominator(n_valid_tokens=compute_logp_mask.bool())

        # Log ground truth when available
        if prox_logp_gt is not None:
            stats_tracker.stat(
                prox_logp_gt=prox_logp_gt.float(),
                denominator="n_valid_tokens",
            )

        # Skip if versions not available
        if versions is None or current_version is None:
            return

        if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
            # Loglinear mode: log approximation without error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=None,  # No ground truth in loglinear mode
                )

        elif prox_logp_method == PROX_LOGP_METHOD_METRICS and prox_logp_gt is not None:
            # Metrics mode: compute all methods with error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=None,  # Compute all methods
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=prox_logp_gt,
                )

        if logprobs is not None:
            # Log KL divergence estimators to check for policy drift between the
            # training-time policy (logprobs) and the inference-time policy (old_logp).
            log_ratio = (logprobs.float() - old_logp.float()).detach()

            # Implementation of different estimators for KL divergence.
            # See: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl
            kl_div_estimator_direct = -log_ratio
            kl_div_estimator_taylor = log_ratio**2 / 2.0
            kl_div_estimator_dual = log_ratio.exp() - 1 - log_ratio

            # Register these to TensorBoard
            stats_tracker.stat(
                kl_div_direct=kl_div_estimator_direct,
                kl_div_taylor=kl_div_estimator_taylor,
                kl_div_dual=kl_div_estimator_dual,
                denominator="n_valid_tokens",
            )


def _log_version_staleness_stats(
    versions: torch.Tensor,
    current_version: int,
    version_metrics_mask: torch.Tensor,
) -> None:
    """
    Log sample staleness metrics based on policy versions.

    Args:
        versions: Per-token policy versions from rollout.
        current_version: Current training version.
        version_metrics_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("version_stats"):
        stats_tracker.denominator(n_valid_tokens=version_metrics_mask.bool())

        v_proximal = current_version - 1
        v_theta = current_version
        v_behave = versions.float()

        # Filter to generated tokens only (version >= 0)
        valid_generated_mask = version_metrics_mask & (versions >= 0)

        if not valid_generated_mask.any():
            return

        # Compute staleness for valid tokens
        staleness_proximal = (v_proximal - v_behave)[valid_generated_mask]
        staleness_theta = (v_theta - v_behave)[valid_generated_mask]

        # Compute and log statistics
        proximal_stats = _tensor_scalar_stats(staleness_proximal)
        theta_stats = _tensor_scalar_stats(staleness_theta)

        stats_tracker.scalar(
            sample_staleness_proximal_avg=proximal_stats["avg"],
            sample_staleness_proximal_max=proximal_stats["max"],
            sample_staleness_proximal_min=proximal_stats["min"],
            sample_staleness_theta_avg=theta_stats["avg"],
            sample_staleness_theta_max=theta_stats["max"],
            sample_staleness_theta_min=theta_stats["min"],
            v_theta=v_theta,
            v_proximal=v_proximal,
            n_valid_generated_tokens=valid_generated_mask.sum().item(),
        )
