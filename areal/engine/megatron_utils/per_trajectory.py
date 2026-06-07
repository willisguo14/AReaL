# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

__all__ = [
    "accumulate_grad_buffers",
    "allocate_grad_accum_buffers",
    "collect_grads_for_norm",
    "copy_accum_buffers_to_grad_buffers",
    "disable_dp_sync",
    "finalize_model_grads_without_dp",
    "finish_dp_grad_sync",
    "grad_norm_from_model_parallel_stats",
    "iter_grad_buffers",
    "restore_dp_sync",
    "zero_grad_accum_buffers",
]


def _as_modules(modules: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(modules, Iterable) and not isinstance(modules, nn.Module):
        return tuple(modules)
    return (modules,)


def _iter_buffer_objects(modules: Iterable[Any]) -> Iterator[Any]:
    for module in modules:
        yielded_buffer = False
        for attr_name in ("buffers", "expert_parallel_buffers"):
            buffers = getattr(module, attr_name, None)
            if buffers is None or callable(buffers):
                continue
            if isinstance(buffers, dict):
                buffer_iter = buffers.values()
            else:
                buffer_iter = buffers
            for buffer in buffer_iter:
                yielded_buffer = True
                yield buffer

        if yielded_buffer:
            continue

        param_and_grad_buffer = getattr(module, "param_and_grad_buffer", None)
        if param_and_grad_buffer is not None:
            yield param_and_grad_buffer


def iter_grad_buffers(modules: Iterable[Any]) -> Iterator[torch.Tensor]:
    """Yield Megatron grad-buffer tensors from DDP model chunks."""
    for buffer in _iter_buffer_objects(modules):
        yield buffer.grad_data


def allocate_grad_accum_buffers(modules: Iterable[Any]) -> list[torch.Tensor]:
    return [torch.zeros_like(grad_buffer) for grad_buffer in iter_grad_buffers(modules)]


def zero_grad_accum_buffers(buffers: Iterable[torch.Tensor]) -> None:
    buffers = list(buffers)
    if not buffers:
        return
    try:
        torch._foreach_zero_(buffers)
    except (RuntimeError, TypeError):
        for buffer in buffers:
            buffer.zero_()


def _materialize_grad_buffer_pairs(
    modules: Iterable[Any],
    accum_buffers: Iterable[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    grad_buffers = list(iter_grad_buffers(modules))
    accum_buffers = list(accum_buffers)
    if len(grad_buffers) != len(accum_buffers):
        raise ValueError(
            "grad buffer count mismatch: "
            f"{len(grad_buffers)} model grad buffers vs "
            f"{len(accum_buffers)} accumulation buffers"
        )
    return grad_buffers, accum_buffers


def accumulate_grad_buffers(
    modules: Iterable[Any],
    accum_buffers: Iterable[torch.Tensor],
    scale: float = 1.0,
) -> None:
    grad_buffers, accum_buffers = _materialize_grad_buffer_pairs(
        modules,
        accum_buffers,
    )
    if not grad_buffers:
        return

    try:
        torch._foreach_add_(accum_buffers, grad_buffers, alpha=scale)
    except (RuntimeError, TypeError):
        for accum_buffer, grad_buffer in zip(
            accum_buffers,
            grad_buffers,
            strict=True,
        ):
            accum_buffer.add_(grad_buffer, alpha=scale)


def copy_accum_buffers_to_grad_buffers(
    modules: Iterable[Any],
    accum_buffers: Iterable[torch.Tensor],
) -> None:
    grad_buffers, accum_buffers = _materialize_grad_buffer_pairs(
        modules,
        accum_buffers,
    )
    for grad_buffer, accum_buffer in zip(grad_buffers, accum_buffers, strict=True):
        grad_buffer.copy_(accum_buffer)


def _param_is_not_shared(param: nn.Parameter) -> bool:
    from megatron.core.transformer.module import param_is_not_shared

    return bool(param_is_not_shared(param))


def _param_is_not_tensor_parallel_duplicate(param: nn.Parameter) -> bool:
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel

    tp_group = mpu.get_tensor_model_parallel_group()
    try:
        return bool(
            tensor_parallel.param_is_not_tensor_parallel_duplicate(
                param,
                tp_group=tp_group,
            )
        )
    except TypeError:
        return bool(tensor_parallel.param_is_not_tensor_parallel_duplicate(param))


def _get_tensor_model_parallel_rank() -> int:
    from megatron.core import parallel_state as mpu

    return int(mpu.get_tensor_model_parallel_rank())


def _get_model_config(model_chunk: Any):
    from megatron.core.utils import get_model_config

    return get_model_config(model_chunk)


def _get_tensor_model_parallel_group():
    from megatron.core import parallel_state as mpu

    return mpu.get_tensor_model_parallel_group()


def _get_pipeline_model_parallel_group():
    from megatron.core import parallel_state as mpu

    return mpu.get_pipeline_model_parallel_group()


def _get_context_parallel_world_size() -> int:
    from megatron.core import parallel_state as mpu

    return int(mpu.get_context_parallel_world_size())


def _get_context_parallel_group():
    from megatron.core import parallel_state as mpu

    return mpu.get_context_parallel_group()


def _get_group_world_size(group: Any) -> int:
    return int(torch.distributed.get_world_size(group=group))


def _get_embedding_group():
    from megatron.core import parallel_state as mpu

    return mpu.get_embedding_group(check_initialized=False)


def _get_position_embedding_group():
    from megatron.core import parallel_state as mpu

    return mpu.get_position_embedding_group(check_initialized=False)


def _get_model_parallel_group():
    from megatron.core import parallel_state as mpu

    return mpu.get_model_parallel_group()


def _get_param_grad(param: nn.Parameter) -> torch.Tensor | None:
    main_grad = getattr(param, "main_grad", None)
    if main_grad is not None:
        return main_grad
    return param.grad


def _should_count_param(
    name: str,
    param: nn.Parameter,
    duplicated_param_names: set[str],
) -> bool:
    if not _param_is_not_shared(param):
        return False
    if name in duplicated_param_names:
        return _get_tensor_model_parallel_rank() == 0
    return _param_is_not_tensor_parallel_duplicate(param)


def collect_grads_for_norm(
    model: Iterable[nn.Module] | nn.Module,
    duplicated_param_names: set[str],
) -> list[torch.Tensor]:
    grads_for_norm: list[torch.Tensor] = []
    for model_chunk in _as_modules(model):
        for name, param in model_chunk.named_parameters():
            if not _should_count_param(name, param, duplicated_param_names):
                continue
            grad = _get_param_grad(param)
            if grad is not None:
                grads_for_norm.append(grad)
    return grads_for_norm


def _grad_stats_parallel_group(optimizer: Any):
    # Per-trajectory records are local to each DP replica; optimizer grad-stats
    # groups may include DP peers under Megatron distributed optimizer.
    return _get_model_parallel_group()


def _get_grad_norm_fp32():
    from megatron.core.optimizer.clip_grads import get_grad_norm_fp32

    return get_grad_norm_fp32


def _call_get_grad_norm_fp32(
    grads_for_norm: list[torch.Tensor],
    model_parallel_group: Any,
) -> float:
    get_grad_norm_fp32 = _get_grad_norm_fp32()
    try:
        parameters = inspect.signature(get_grad_norm_fp32).parameters
    except (TypeError, ValueError):
        return float(get_grad_norm_fp32(grads_for_norm, model_parallel_group))

    if "grad_stats_parallel_group" in parameters:
        norm = get_grad_norm_fp32(
            grads_for_norm,
            norm_type=2,
            grad_stats_parallel_group=model_parallel_group,
        )
    elif "data_parallel_group" in parameters:
        norm = get_grad_norm_fp32(
            grads_for_norm,
            data_parallel_group=None,
            model_parallel_group=model_parallel_group,
        )
    elif "model_parallel_group" in parameters:
        norm = get_grad_norm_fp32(
            grads_for_norm,
            model_parallel_group=model_parallel_group,
        )
    else:
        norm = get_grad_norm_fp32(grads_for_norm, model_parallel_group)
    return float(norm)


def grad_norm_from_model_parallel_stats(
    model: Iterable[nn.Module],
    optimizer: Any,
    duplicated_param_names: set[str],
    loss_scale: float,
    schedule_scale: float,
) -> float:
    grad_norm = _call_get_grad_norm_fp32(
        collect_grads_for_norm(model, duplicated_param_names),
        _grad_stats_parallel_group(optimizer),
    )
    total_scale = float(loss_scale) * float(schedule_scale)
    if total_scale == 0.0:
        raise ValueError("loss_scale * schedule_scale must be non-zero")
    return grad_norm / total_scale


def _allreduce_conditional_embedding_grads(model: list[Any], config: Any, pp_group):
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_conditional_embedding_grads,
    )

    return _allreduce_conditional_embedding_grads(model, config, pp_group)


def _allreduce_non_tensor_model_parallel_grads(model: list[Any], config: Any, tp_group):
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_non_tensor_model_parallel_grads,
    )

    return _allreduce_non_tensor_model_parallel_grads(model, config, tp_group)


def _allreduce_word_embedding_grads(
    model: list[Any],
    config: Any,
    embd_group,
    pp_group,
):
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_word_embedding_grads,
    )

    return _allreduce_word_embedding_grads(model, config, embd_group, pp_group)


def _allreduce_position_embedding_grads(
    model: list[Any],
    config: Any,
    pos_emb_group,
    pp_group,
):
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_position_embedding_grads,
    )

    return _allreduce_position_embedding_grads(
        model,
        config,
        pos_emb_group,
        pp_group,
    )


def _reset_model_temporary_tensors(config: Any, model: list[Any]) -> None:
    from megatron.core.distributed.finalize_model_grads import (
        reset_model_temporary_tensors,
    )

    reset_model_temporary_tensors(config, model)


def _get_main_grad_attr(param: nn.Parameter, use_custom_fsdp: bool = False) -> str:
    from megatron.core.distributed.finalize_model_grads import _get_main_grad_attr

    parameters = inspect.signature(_get_main_grad_attr).parameters
    if len(parameters) == 1:
        return str(_get_main_grad_attr(param))
    if "use_custom_fsdp" in parameters:
        return str(_get_main_grad_attr(param, use_custom_fsdp=use_custom_fsdp))
    return str(_get_main_grad_attr(param, use_custom_fsdp))


def get_attr_wrapped_model(model_chunk: Any, attr_name: str):
    from megatron.core.distributed.finalize_model_grads import get_attr_wrapped_model

    return get_attr_wrapped_model(model_chunk, attr_name)


def _unshard_if_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core.distributed.finalize_model_grads import _unshard_if_dtensor

    return _unshard_if_dtensor(tensor)


def _reshard_if_dtensor(
    tensor: torch.Tensor,
    orig_grad: torch.Tensor,
) -> torch.Tensor:
    from megatron.core.distributed.finalize_model_grads import _reshard_if_dtensor

    return _reshard_if_dtensor(tensor, orig_grad)


def _distributed_all_reduce(
    tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    group: Any,
) -> None:
    torch.distributed.all_reduce(tensor, op=op, group=group)


def _allreduce_grads_cp(
    model: list[Any],
    cp_group: Any = None,
    cp_world_size: int | None = None,
) -> None:
    if cp_world_size is None:
        cp_world_size = _get_context_parallel_world_size()
    if cp_world_size <= 1:
        return
    if cp_group is None:
        cp_group = _get_context_parallel_group()

    params_and_grads: list[tuple[nn.Parameter, str, torch.Tensor]] = []
    grads_avg: list[torch.Tensor] = []

    for model_chunk in model:
        ddp_config = getattr(model_chunk, "ddp_config", None)
        use_custom_fsdp = bool(getattr(ddp_config, "use_custom_fsdp", False))
        named_parameters = get_attr_wrapped_model(model_chunk, "named_parameters")
        for _name, param in named_parameters():
            if not param.requires_grad:
                continue
            grad_attr = _get_main_grad_attr(param, use_custom_fsdp=use_custom_fsdp)
            orig_grad = getattr(param, grad_attr, None)
            if orig_grad is None:
                continue
            grad = _unshard_if_dtensor(orig_grad)
            grads_avg.append(grad.data)
            params_and_grads.append((param, grad_attr, orig_grad))

    if not grads_avg:
        return

    coalesced = _flatten_dense_tensors(grads_avg)
    _distributed_all_reduce(
        coalesced,
        op=torch.distributed.ReduceOp.AVG,
        group=cp_group,
    )

    for (param, grad_attr, orig_grad), buf, synced in zip(
        params_and_grads,
        grads_avg,
        _unflatten_dense_tensors(coalesced, grads_avg),
        strict=True,
    ):
        buf.copy_(synced)
        setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))


def _pg_collection_group(pg_collection: Any, *names: str):
    for name in names:
        if not hasattr(pg_collection, name):
            continue
        group = getattr(pg_collection, name)
        if group is not None:
            return group
    raise NotImplementedError(
        "finalize_model_grads_without_dp requires pg_collection to provide "
        f"one of {names}"
    )


@contextmanager
def _timer(config: Any, name: str):
    timers = getattr(config, "timers", None)
    if timers is None:
        yield
        return

    timer = timers(name, log_level=1)
    timer.start(barrier=getattr(config, "barrier_with_L1_time", False))
    try:
        yield
    finally:
        timers(name).stop()


def finalize_model_grads_without_dp(
    model: Iterable[Any] | Any,
    num_tokens: Any = None,
    *,
    pg_collection: Any | None = None,
    **_schedule_kwargs: Any,
):
    model_chunks = list(_as_modules(model))
    config = _get_model_config(model_chunks[0])
    if num_tokens is not None:
        raise NotImplementedError(
            "finalize_model_grads_without_dp does not support num_tokens scaling"
        )
    if getattr(config, "moe_router_enable_expert_bias", False):
        raise NotImplementedError(
            "finalize_model_grads_without_dp does not support MoE expert bias update"
        )

    if pg_collection is None:
        tp_group = _get_tensor_model_parallel_group()
        pp_group = _get_pipeline_model_parallel_group()
        embd_group = _get_embedding_group()
        pos_emb_group = _get_position_embedding_group()
        cp_group = None
        cp_world_size = None
    else:
        tp_group = _pg_collection_group(pg_collection, "tp")
        pp_group = _pg_collection_group(pg_collection, "pp")
        embd_group = _pg_collection_group(pg_collection, "embd")
        pos_emb_group = _pg_collection_group(pg_collection, "pos_embd", "pos_emb")
        cp_group = _pg_collection_group(pg_collection, "cp")
        cp_world_size = _get_group_world_size(cp_group)

    with _timer(config, "context-parallel-grads-all-reduce"):
        _allreduce_grads_cp(
            model_chunks,
            cp_group=cp_group,
            cp_world_size=cp_world_size,
        )
    with _timer(config, "conditional-embedder-grads-all-reduce"):
        _allreduce_conditional_embedding_grads(model_chunks, config, pp_group)
    with _timer(config, "non-tensor-parallel-grads-all-reduce"):
        _allreduce_non_tensor_model_parallel_grads(model_chunks, config, tp_group)
    with _timer(config, "embedding-grads-all-reduce"):
        _allreduce_word_embedding_grads(model_chunks, config, embd_group, pp_group)
        _allreduce_position_embedding_grads(
            model_chunks,
            config,
            pos_emb_group,
            pp_group,
        )
    _reset_model_temporary_tensors(config, model_chunks)


def disable_dp_sync(model: Iterable[Any] | Any) -> tuple[Any, Any, Any]:
    model_chunks = _as_modules(model)
    for model_chunk in model_chunks:
        ddp_config = getattr(model_chunk, "ddp_config", None)
        if getattr(ddp_config, "overlap_grad_reduce", False):
            raise RuntimeError(
                "per-trajectory Megatron DP sync disabling does not support "
                "ddp_config.overlap_grad_reduce=True"
            )
    config = _get_model_config(model_chunks[0])
    original = (
        getattr(config, "no_sync_func", None),
        getattr(config, "grad_sync_func", None),
        getattr(config, "finalize_model_grads_func", None),
    )
    config.no_sync_func = nullcontext
    config.grad_sync_func = None
    config.finalize_model_grads_func = finalize_model_grads_without_dp
    return original


def restore_dp_sync(model: Iterable[Any] | Any, original: tuple[Any, Any, Any]) -> None:
    model_chunks = _as_modules(model)
    config = _get_model_config(model_chunks[0])
    (
        config.no_sync_func,
        config.grad_sync_func,
        config.finalize_model_grads_func,
    ) = original


def finish_dp_grad_sync(modules: Iterable[Any] | Any) -> None:
    modules = _as_modules(modules)
    for module in modules:
        ddp_config = getattr(module, "ddp_config", None)
        if not getattr(ddp_config, "overlap_grad_reduce", False):
            continue
        start_grad_sync = getattr(module, "start_grad_sync", None)
        if callable(start_grad_sync):
            start_grad_sync()
    for module in modules:
        finish_grad_sync = getattr(module, "finish_grad_sync", None)
        if callable(finish_grad_sync):
            finish_grad_sync()
