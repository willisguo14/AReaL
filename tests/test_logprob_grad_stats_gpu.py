import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    FSDPEngineConfig,
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.slow

MODEL_PATH_ENV = "AREAL_LOGPROB_GRAD_GPU_MODEL_PATH"
DEFAULT_MODEL_PATHS = (
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/",
    "/storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct/",
)


@contextmanager
def _distributed_env() -> Iterator[None]:
    old_env = {key: os.environ.get(key) for key in _DISTRIBUTED_ENV_KEYS}
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(find_free_ports(1)[0]),
        }
    )
    try:
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


_DISTRIBUTED_ENV_KEYS = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _batch(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [
                [101, 314, 1592, 653, 920, 88, 47, 13],
                [205, 444, 1729, 802, 377, 91, 58, 17],
            ],
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.ones((2, 8), dtype=torch.bool, device=device),
        "loss_mask": torch.tensor(
            [
                [False, True, True, True, False, True, True, False],
                [False, True, True, False, True, True, True, False],
            ],
            dtype=torch.bool,
            device=device,
        ),
        "advantages": torch.tensor(
            [
                [0.0, 0.75, -0.25, 0.50, 0.0, -0.10, 0.30, 0.0],
                [0.0, -0.55, 0.20, 0.0, 0.65, -0.35, 0.15, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        ),
    }


def _linear_actor_loss(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict[str, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    del entropy, kwargs
    mask = input_data["loss_mask"].bool()
    advantages = input_data["advantages"].to(dtype=logprobs.dtype)
    return -(logprobs[mask] * advantages[mask]).sum() / mask.count_nonzero()


def _loss_weight(mb: dict[str, torch.Tensor]) -> torch.Tensor:
    return mb["loss_mask"].count_nonzero()


def _assert_logprob_grad_stats(stats: dict[str, float], batch: dict[str, torch.Tensor]):
    valid_advantages = batch["advantages"][batch["loss_mask"]].float()
    expected_norm = (
        torch.linalg.vector_norm(valid_advantages) / valid_advantages.numel()
    )
    expected_abs_max = valid_advantages.abs().max() / valid_advantages.numel()

    assert stats["update_successful"] == 1.0
    assert stats["logp_grad_norm"] == pytest.approx(
        float(expected_norm),
        rel=2e-2,
        abs=1e-6,
    )
    assert stats["logp_grad_absmax"] == pytest.approx(
        float(expected_abs_max),
        rel=2e-2,
        abs=1e-6,
    )
    assert stats["grad_norm"] > 0.0


def _skip_without_cuda():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("CUDA GPU is required for this smoke test")


def _model_path_or_skip() -> str:
    candidates = [os.environ.get(MODEL_PATH_ENV), *DEFAULT_MODEL_PATHS]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    pytest.skip(
        f"set {MODEL_PATH_ENV} to a local causal-LM checkpoint for this smoke test"
    )


def test_fsdp_train_batch_reports_logprob_grad_stats_on_gpu():
    _skip_without_cuda()

    from areal.engine import FSDPEngine

    with _distributed_env():
        config = TrainEngineConfig(
            backend="fsdp:d1p1t1",
            experiment_name="logprob-grad-smoke",
            trial_name="fsdp",
            path=_model_path_or_skip(),
            dtype="bfloat16",
            disable_dropout=True,
            gradient_checkpointing=False,
            init_from_scratch=True,
            mb_spec=MicroBatchSpec(n_mbs=2, max_tokens_per_mb=64),
            optimizer=OptimizerConfig(lr=1e-6, weight_decay=0.0),
            fsdp=FSDPEngineConfig(),
        )
        engine = FSDPEngine(config)
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=2,
            train_batch_size=2,
        )
        engine.create_process_group()
        engine.initialize(None, ft_spec)
        try:
            batch = _batch(engine.device)
            stats = engine.train_batch(
                batch,
                loss_fn=_linear_actor_loss,
                loss_weight_fn=_loss_weight,
                collect_logprob_grad_stats=True,
            )
            _assert_logprob_grad_stats(stats, batch)
        finally:
            engine.destroy()
            assert not dist.is_initialized()


def test_megatron_train_batch_reports_loss_scale_invariant_logprob_grad_stats_on_gpu():
    _skip_without_cuda()
    missing = [
        module_name
        for module_name in ("mbridge", "megatron")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        pytest.skip(f"missing Megatron runtime dependencies: {', '.join(missing)}")

    from areal.engine import MegatronEngine

    with _distributed_env():
        config = TrainEngineConfig(
            backend="megatron:d1p1t1",
            experiment_name="logprob-grad-smoke",
            trial_name="megatron",
            path=_model_path_or_skip(),
            dtype="bfloat16",
            disable_dropout=True,
            gradient_checkpointing=False,
            init_from_scratch=True,
            mb_spec=MicroBatchSpec(n_mbs=2, max_tokens_per_mb=64),
            optimizer=OptimizerConfig(lr=1e-6, weight_decay=0.0),
            megatron=MegatronEngineConfig(),
        )
        engine = MegatronEngine(config)
        alloc_mode = ModelAllocation.from_str("megatron:d1p1t1")
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=2,
            train_batch_size=2,
        )
        engine.create_process_group(alloc_mode.parallel)
        engine.initialize(addr=None, ft_spec=ft_spec)
        try:
            original_get_loss_scale = engine.optimizer.get_loss_scale

            batch = _batch(engine.device)
            engine.optimizer.get_loss_scale = lambda: torch.tensor(
                1.0,
                device=engine.device,
            )
            stats_scale_1 = engine.train_batch(
                batch,
                loss_fn=_linear_actor_loss,
                loss_weight_fn=_loss_weight,
                collect_logprob_grad_stats=True,
            )
            _assert_logprob_grad_stats(stats_scale_1, batch)

            batch = _batch(engine.device)
            engine.optimizer.get_loss_scale = lambda: torch.tensor(
                8.0,
                device=engine.device,
            )
            stats_scale_8 = engine.train_batch(
                batch,
                loss_fn=_linear_actor_loss,
                loss_weight_fn=_loss_weight,
                collect_logprob_grad_stats=True,
            )
            _assert_logprob_grad_stats(stats_scale_8, batch)
            assert stats_scale_8["logp_grad_norm"] == pytest.approx(
                stats_scale_1["logp_grad_norm"],
                rel=2e-2,
                abs=1e-6,
            )
            assert stats_scale_8["logp_grad_absmax"] == pytest.approx(
                stats_scale_1["logp_grad_absmax"],
                rel=2e-2,
                abs=1e-6,
            )
        finally:
            engine.optimizer.get_loss_scale = original_get_loss_scale
            engine.destroy()
            assert not dist.is_initialized()
