import builtins
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from areal.engine.megatron_utils import per_trajectory
from areal.engine.megatron_utils.per_trajectory import (
    accumulate_grad_buffers,
    allocate_grad_accum_buffers,
    collect_grads_for_norm,
    copy_accum_buffers_to_grad_buffers,
    disable_dp_sync,
    finalize_model_grads_without_dp,
    finish_dp_grad_sync,
    grad_norm_from_model_parallel_stats,
    iter_grad_buffers,
    restore_dp_sync,
    zero_grad_accum_buffers,
)


class _Buffer:
    def __init__(self, values):
        self.grad_data = torch.tensor(values, dtype=torch.float32)


class _Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.buffers = [_Buffer([1.0, 2.0])]


def test_grad_buffer_accumulation_and_copy_back():
    module = _Module()
    accum = allocate_grad_accum_buffers([module])

    assert torch.equal(accum[0], torch.zeros(2))

    accumulate_grad_buffers([module], accum)
    module.buffers[0].grad_data = torch.tensor([3.0, 4.0])
    accumulate_grad_buffers([module], accum, scale=0.5)

    assert torch.equal(accum[0], torch.tensor([2.5, 4.0]))

    module.buffers[0].grad_data.zero_()
    copy_accum_buffers_to_grad_buffers([module], accum)

    assert torch.equal(module.buffers[0].grad_data, torch.tensor([2.5, 4.0]))


def test_accumulate_grad_buffers_count_mismatch_is_atomic():
    module = _Module()
    accum = [torch.zeros(2), torch.full((2,), 9.0)]

    with pytest.raises(ValueError, match="grad buffer count"):
        accumulate_grad_buffers([module], accum)

    assert torch.equal(accum[0], torch.zeros(2))
    assert torch.equal(accum[1], torch.full((2,), 9.0))


def test_copy_accum_buffers_count_mismatch_is_atomic():
    module = _Module()
    module.buffers[0].grad_data.zero_()
    accum = [torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])]

    with pytest.raises(ValueError, match="grad buffer count"):
        copy_accum_buffers_to_grad_buffers([module], accum)

    assert torch.equal(module.buffers[0].grad_data, torch.zeros(2))


def test_zero_grad_accum_buffers():
    module = _Module()
    accum = allocate_grad_accum_buffers([module])
    accum[0].fill_(5.0)

    zero_grad_accum_buffers(accum)

    assert torch.equal(accum[0], torch.zeros(2))


def test_iter_grad_buffers_supports_param_and_grad_buffer():
    module = SimpleNamespace(param_and_grad_buffer=_Buffer([7.0]))

    buffers = list(iter_grad_buffers([module]))

    assert len(buffers) == 1
    assert torch.equal(buffers[0], torch.tensor([7.0]))


def test_iter_grad_buffers_supports_dict_and_expert_parallel_buffers():
    module = SimpleNamespace(
        buffers={
            "dense_a": _Buffer([1.0]),
            "dense_b": _Buffer([2.0]),
        },
        expert_parallel_buffers=[_Buffer([3.0])],
        param_and_grad_buffer=_Buffer([99.0]),
    )

    buffers = list(iter_grad_buffers([module]))

    assert len(buffers) == 3
    assert torch.equal(buffers[0], torch.tensor([1.0]))
    assert torch.equal(buffers[1], torch.tensor([2.0]))
    assert torch.equal(buffers[2], torch.tensor([3.0]))


def test_grad_norm_from_model_parallel_stats_uses_model_parallel_group(monkeypatch):
    tensor = torch.tensor([3.0, 4.0])

    def fake_get_grad_norm_fp32(grads, norm_type=2, grad_stats_parallel_group=None):
        assert grads == [tensor]
        assert norm_type == 2
        assert grad_stats_parallel_group == "model-parallel"
        return 5.0

    class OptimizerWithWorldGradStatsGroup:
        def get_grad_stats_parallel_group(self):
            return "world"

    monkeypatch.setattr(
        per_trajectory,
        "collect_grads_for_norm",
        lambda *args, **kwargs: [tensor],
    )
    monkeypatch.setattr(
        per_trajectory,
        "_get_model_parallel_group",
        lambda: "model-parallel",
    )
    monkeypatch.setattr(
        per_trajectory,
        "_get_grad_norm_fp32",
        lambda: fake_get_grad_norm_fp32,
    )

    norm = grad_norm_from_model_parallel_stats(
        model=[object()],
        optimizer=OptimizerWithWorldGradStatsGroup(),
        duplicated_param_names=set(),
        loss_scale=2.0,
        schedule_scale=5.0,
    )

    assert norm == pytest.approx(0.5)


def test_get_grad_norm_fp32_is_megatron_only(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("megatron"):
            raise ModuleNotFoundError("No module named 'megatron'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="megatron"):
        per_trajectory._get_grad_norm_fp32()


def test_disable_and_restore_dp_sync_use_model_config(monkeypatch):
    config = SimpleNamespace(
        no_sync_func=object(),
        grad_sync_func=object(),
        finalize_model_grads_func=object(),
    )
    original_fields = (
        config.no_sync_func,
        config.grad_sync_func,
        config.finalize_model_grads_func,
    )
    model = [object()]

    monkeypatch.setattr(per_trajectory, "_get_model_config", lambda chunk: config)

    original = disable_dp_sync(model)

    assert original == original_fields
    assert config.no_sync_func is nullcontext
    assert config.grad_sync_func is None
    assert config.finalize_model_grads_func is finalize_model_grads_without_dp

    restore_dp_sync(model, original)

    assert (
        config.no_sync_func,
        config.grad_sync_func,
        config.finalize_model_grads_func,
    ) == original_fields


def test_disable_dp_sync_rejects_overlap_grad_reduce(monkeypatch):
    config = SimpleNamespace(
        no_sync_func=object(),
        grad_sync_func=object(),
        finalize_model_grads_func=object(),
    )
    original_fields = (
        config.no_sync_func,
        config.grad_sync_func,
        config.finalize_model_grads_func,
    )
    chunk = SimpleNamespace(
        ddp_config=SimpleNamespace(overlap_grad_reduce=True),
    )

    monkeypatch.setattr(per_trajectory, "_get_model_config", lambda chunk: config)

    with pytest.raises(RuntimeError, match="overlap_grad_reduce"):
        disable_dp_sync([chunk])

    assert (
        config.no_sync_func,
        config.grad_sync_func,
        config.finalize_model_grads_func,
    ) == original_fields


def test_finish_dp_grad_sync_starts_only_when_overlap_enabled():
    class _Chunk:
        def __init__(self, overlap_grad_reduce):
            self.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap_grad_reduce)
            self.calls = []

        def start_grad_sync(self):
            self.calls.append("start")

        def finish_grad_sync(self):
            self.calls.append("finish")

    overlapped = _Chunk(overlap_grad_reduce=True)
    non_overlapped = _Chunk(overlap_grad_reduce=False)

    finish_dp_grad_sync([overlapped, non_overlapped])

    assert overlapped.calls == ["start", "finish"]
    assert non_overlapped.calls == ["finish"]


def test_finalize_model_grads_without_dp_calls_non_dp_helpers(monkeypatch):
    config = SimpleNamespace(
        timers=None,
        moe_router_enable_expert_bias=False,
        moe_router_load_balancing_type="none",
    )
    model = [object()]
    calls = []

    monkeypatch.setattr(per_trajectory, "_get_model_config", lambda chunk: config)
    monkeypatch.setattr(per_trajectory, "_get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        per_trajectory, "_get_tensor_model_parallel_group", lambda: "tp"
    )
    monkeypatch.setattr(
        per_trajectory, "_get_pipeline_model_parallel_group", lambda: "pp"
    )
    monkeypatch.setattr(per_trajectory, "_get_embedding_group", lambda: "embd")
    monkeypatch.setattr(per_trajectory, "_get_position_embedding_group", lambda: "pos")
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_conditional_embedding_grads",
        lambda model_arg, config_arg, pp_group: calls.append(
            ("conditional", model_arg, config_arg, pp_group)
        ),
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_non_tensor_model_parallel_grads",
        lambda model_arg, config_arg, tp_group: calls.append(
            ("non_tp", model_arg, config_arg, tp_group)
        ),
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_word_embedding_grads",
        lambda model_arg, config_arg, embd_group, pp_group: calls.append(
            ("word", model_arg, config_arg, embd_group, pp_group)
        ),
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_position_embedding_grads",
        lambda model_arg, config_arg, pos_group, pp_group: calls.append(
            ("position", model_arg, config_arg, pos_group, pp_group)
        ),
    )
    monkeypatch.setattr(
        per_trajectory,
        "_reset_model_temporary_tensors",
        lambda config_arg, model_arg: calls.append(("reset", config_arg, model_arg)),
    )

    finalize_model_grads_without_dp(model)

    assert calls == [
        ("conditional", model, config, "pp"),
        ("non_tp", model, config, "tp"),
        ("word", model, config, "embd", "pp"),
        ("position", model, config, "pos", "pp"),
        ("reset", config, model),
    ]


def test_finalize_model_grads_without_dp_accepts_schedule_kwargs_and_pg_collection(
    monkeypatch,
):
    config = SimpleNamespace(
        timers=None,
        moe_router_enable_expert_bias=False,
        moe_router_load_balancing_type="none",
    )
    model = [object()]
    pg_collection = SimpleNamespace(
        tp="tp-collection",
        pp="pp-collection",
        embd="embd-collection",
        pos_embd="pos-collection",
        cp="cp-collection",
    )
    calls = []

    monkeypatch.setattr(per_trajectory, "_get_model_config", lambda chunk: config)
    monkeypatch.setattr(
        per_trajectory,
        "_get_group_world_size",
        lambda group: 1 if group == "cp-collection" else 8,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_get_context_parallel_world_size",
        lambda: (_ for _ in ()).throw(AssertionError("fallback CP size used")),
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_conditional_embedding_grads",
        lambda model_arg, config_arg, pp_group: calls.append(("conditional", pp_group)),
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_non_tensor_model_parallel_grads",
        lambda model_arg, config_arg, tp_group: calls.append(("non_tp", tp_group)),
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_word_embedding_grads",
        lambda model_arg, config_arg, embd_group, pp_group: calls.append(
            ("word", embd_group, pp_group)
        ),
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_position_embedding_grads",
        lambda model_arg, config_arg, pos_group, pp_group: calls.append(
            ("position", pos_group, pp_group)
        ),
    )
    monkeypatch.setattr(
        per_trajectory,
        "_reset_model_temporary_tensors",
        lambda config_arg, model_arg: calls.append(("reset",)),
    )

    finalize_model_grads_without_dp(
        model,
        pg_collection=pg_collection,
        force_all_reduce=True,
    )

    assert calls == [
        ("conditional", "pp-collection"),
        ("non_tp", "tp-collection"),
        ("word", "embd-collection", "pp-collection"),
        ("position", "pos-collection", "pp-collection"),
        ("reset",),
    ]


def test_finalize_model_grads_without_dp_rejects_unknown_pg_collection(monkeypatch):
    config = SimpleNamespace(
        timers=None,
        moe_router_enable_expert_bias=False,
        moe_router_load_balancing_type="none",
    )

    monkeypatch.setattr(per_trajectory, "_get_model_config", lambda chunk: config)

    with pytest.raises(NotImplementedError, match="pg_collection"):
        finalize_model_grads_without_dp([object()], pg_collection=object())


def _install_fake_finalize_module(monkeypatch, fake_get_main_grad_attr):
    finalize_module = ModuleType("megatron.core.distributed.finalize_model_grads")
    finalize_module._get_main_grad_attr = fake_get_main_grad_attr
    megatron_module = ModuleType("megatron")
    megatron_module.__path__ = []
    core_module = ModuleType("megatron.core")
    core_module.__path__ = []
    distributed_module = ModuleType("megatron.core.distributed")
    distributed_module.__path__ = []

    monkeypatch.setitem(sys.modules, "megatron", megatron_module)
    monkeypatch.setitem(sys.modules, "megatron.core", core_module)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", distributed_module)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.distributed.finalize_model_grads",
        finalize_module,
    )


def test_get_main_grad_attr_supports_one_arg_megatron_helper(monkeypatch):
    calls = []

    def fake_get_main_grad_attr(param):
        calls.append(param)
        return "main_grad"

    _install_fake_finalize_module(monkeypatch, fake_get_main_grad_attr)

    param = nn.Parameter(torch.tensor([1.0]))

    assert (
        per_trajectory._get_main_grad_attr(param, use_custom_fsdp=True) == "main_grad"
    )
    assert calls == [param]


def test_get_main_grad_attr_supports_named_use_custom_fsdp(monkeypatch):
    calls = []

    def fake_get_main_grad_attr(param, *, use_custom_fsdp=False):
        calls.append((param, use_custom_fsdp))
        return "fsdp_grad" if use_custom_fsdp else "main_grad"

    _install_fake_finalize_module(monkeypatch, fake_get_main_grad_attr)

    param = nn.Parameter(torch.tensor([1.0]))

    assert (
        per_trajectory._get_main_grad_attr(param, use_custom_fsdp=True) == "fsdp_grad"
    )
    assert calls == [(param, True)]


def test_get_main_grad_attr_supports_positional_use_custom_fsdp(monkeypatch):
    calls = []

    def fake_get_main_grad_attr(param, custom_fsdp_flag):
        calls.append((param, custom_fsdp_flag))
        return "fsdp_grad" if custom_fsdp_flag else "main_grad"

    _install_fake_finalize_module(monkeypatch, fake_get_main_grad_attr)

    param = nn.Parameter(torch.tensor([1.0]))

    assert (
        per_trajectory._get_main_grad_attr(param, use_custom_fsdp=True) == "fsdp_grad"
    )
    assert calls == [(param, True)]


def test_finalize_model_grads_without_dp_allreduces_cp_main_grads(monkeypatch):
    class _Chunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.bias = nn.Parameter(torch.tensor([[3.0]]))
            self.weight.main_grad = torch.tensor([1.0, 2.0])
            self.bias.main_grad = torch.tensor([[3.0]])
            self.ddp_config = SimpleNamespace(use_custom_fsdp=False)

    config = SimpleNamespace(
        timers=None,
        moe_router_enable_expert_bias=False,
        moe_router_load_balancing_type="none",
    )
    model = [_Chunk()]
    calls = []

    monkeypatch.setattr(per_trajectory, "_get_model_config", lambda chunk: config)
    monkeypatch.setattr(per_trajectory, "_get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(per_trajectory, "_get_context_parallel_group", lambda: "cp")
    monkeypatch.setattr(
        per_trajectory,
        "_distributed_all_reduce",
        lambda tensor, op, group: (
            calls.append((tensor.clone(), op, group)),
            tensor.add_(10.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_get_main_grad_attr",
        lambda param, use_custom_fsdp=False: "main_grad",
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory,
        "get_attr_wrapped_model",
        lambda model_chunk, attr_name: getattr(model_chunk, attr_name),
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_unshard_if_dtensor",
        lambda tensor: tensor,
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_reshard_if_dtensor",
        lambda tensor, orig_grad: tensor,
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory, "_get_tensor_model_parallel_group", lambda: "tp"
    )
    monkeypatch.setattr(
        per_trajectory,
        "_get_pipeline_model_parallel_group",
        lambda: "pp",
    )
    monkeypatch.setattr(per_trajectory, "_get_embedding_group", lambda: "embd")
    monkeypatch.setattr(per_trajectory, "_get_position_embedding_group", lambda: "pos")
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_conditional_embedding_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_non_tensor_model_parallel_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_word_embedding_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_position_embedding_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_reset_model_temporary_tensors",
        lambda *args: None,
    )

    finalize_model_grads_without_dp(model)

    assert len(calls) == 1
    assert torch.equal(calls[0][0], torch.tensor([1.0, 2.0, 3.0]))
    assert calls[0][1] == torch.distributed.ReduceOp.AVG
    assert calls[0][2] == "cp"
    assert torch.equal(model[0].weight.main_grad, torch.tensor([11.0, 12.0]))
    assert torch.equal(model[0].bias.main_grad, torch.tensor([[13.0]]))


def test_finalize_model_grads_without_dp_allreduces_cp_pg_collection_group(
    monkeypatch,
):
    class _Chunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.weight.main_grad = torch.tensor([1.0, 2.0])
            self.ddp_config = SimpleNamespace(use_custom_fsdp=False)

    config = SimpleNamespace(
        timers=None,
        moe_router_enable_expert_bias=False,
        moe_router_load_balancing_type="none",
    )
    pg_collection = SimpleNamespace(
        tp="tp-collection",
        pp="pp-collection",
        embd="embd-collection",
        pos_embd="pos-collection",
        cp="cp-collection",
    )
    calls = []

    monkeypatch.setattr(per_trajectory, "_get_model_config", lambda chunk: config)
    monkeypatch.setattr(
        per_trajectory,
        "_get_group_world_size",
        lambda group: 2 if group == "cp-collection" else 8,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_get_context_parallel_group",
        lambda: (_ for _ in ()).throw(AssertionError("fallback CP group used")),
    )

    def fake_all_reduce(tensor, op, group):
        assert group == "cp-collection"
        calls.append((tensor.clone(), op, group))
        tensor.add_(10.0)

    monkeypatch.setattr(
        per_trajectory,
        "_distributed_all_reduce",
        fake_all_reduce,
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_get_main_grad_attr",
        lambda param, use_custom_fsdp=False: "main_grad",
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory,
        "get_attr_wrapped_model",
        lambda model_chunk, attr_name: getattr(model_chunk, attr_name),
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_unshard_if_dtensor",
        lambda tensor: tensor,
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_reshard_if_dtensor",
        lambda tensor, orig_grad: tensor,
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_conditional_embedding_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_non_tensor_model_parallel_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_word_embedding_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_position_embedding_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_reset_model_temporary_tensors",
        lambda *args: None,
    )
    model = [_Chunk()]

    finalize_model_grads_without_dp(model, pg_collection=pg_collection)

    assert len(calls) == 1
    assert torch.equal(calls[0][0], torch.tensor([1.0, 2.0]))
    assert calls[0][1] == torch.distributed.ReduceOp.AVG
    assert calls[0][2] == "cp-collection"
    assert torch.equal(model[0].weight.main_grad, torch.tensor([11.0, 12.0]))


def test_finalize_model_grads_without_dp_skips_cp_grad_sync_when_cp_disabled(
    monkeypatch,
):
    class _Chunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))
            self.weight.main_grad = torch.tensor([1.0])
            self.ddp_config = SimpleNamespace(use_custom_fsdp=False)

    config = SimpleNamespace(
        timers=None,
        moe_router_enable_expert_bias=False,
        moe_router_load_balancing_type="none",
    )
    model = [_Chunk()]
    calls = []

    monkeypatch.setattr(per_trajectory, "_get_model_config", lambda chunk: config)
    monkeypatch.setattr(per_trajectory, "_get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        per_trajectory,
        "_distributed_all_reduce",
        lambda *args: calls.append(args),
        raising=False,
    )
    monkeypatch.setattr(
        per_trajectory, "_get_tensor_model_parallel_group", lambda: "tp"
    )
    monkeypatch.setattr(
        per_trajectory,
        "_get_pipeline_model_parallel_group",
        lambda: "pp",
    )
    monkeypatch.setattr(per_trajectory, "_get_embedding_group", lambda: "embd")
    monkeypatch.setattr(per_trajectory, "_get_position_embedding_group", lambda: "pos")
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_conditional_embedding_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_non_tensor_model_parallel_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_word_embedding_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_allreduce_position_embedding_grads",
        lambda *args: None,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_reset_model_temporary_tensors",
        lambda *args: None,
    )

    finalize_model_grads_without_dp(model)

    assert calls == []
    assert torch.equal(model[0].weight.main_grad, torch.tensor([1.0]))


def test_finalize_model_grads_without_dp_rejects_dp_dependent_paths(monkeypatch):
    config = SimpleNamespace(
        timers=None,
        moe_router_enable_expert_bias=False,
        moe_router_load_balancing_type="none",
    )
    model = [object()]

    monkeypatch.setattr(per_trajectory, "_get_model_config", lambda chunk: config)

    with pytest.raises(NotImplementedError, match="num_tokens"):
        finalize_model_grads_without_dp(model, num_tokens=torch.tensor(1))

    config.moe_router_enable_expert_bias = True

    with pytest.raises(NotImplementedError, match="expert bias"):
        finalize_model_grads_without_dp(model)


def test_collect_grads_for_norm_uses_main_grad_and_lower_level_filters(monkeypatch):
    class _GradModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.keep = nn.Parameter(torch.tensor([1.0]))
            self.shared = nn.Parameter(torch.tensor([2.0]))
            self.tp_duplicate = nn.Parameter(torch.tensor([3.0]))
            self.duplicated_name = nn.Parameter(torch.tensor([4.0]))
            self.fallback_grad = nn.Parameter(torch.tensor([5.0]))
            self.no_grad = nn.Parameter(torch.tensor([3.0]))
            self.keep.main_grad = torch.tensor([4.0])
            self.shared.main_grad = torch.tensor([5.0])
            self.tp_duplicate.main_grad = torch.tensor([6.0])
            self.duplicated_name.main_grad = torch.tensor([7.0])
            self.fallback_grad.grad = torch.tensor([8.0])

    module = _GradModule()

    monkeypatch.setattr(
        per_trajectory,
        "_param_is_not_shared",
        lambda param: param is not module.shared,
    )
    monkeypatch.setattr(
        per_trajectory,
        "_param_is_not_tensor_parallel_duplicate",
        lambda param: param is not module.tp_duplicate,
    )
    monkeypatch.setattr(per_trajectory, "_get_tensor_model_parallel_rank", lambda: 0)

    grads = collect_grads_for_norm(
        [module],
        duplicated_param_names={"duplicated_name"},
    )

    assert grads == [
        module.keep.main_grad,
        module.duplicated_name.main_grad,
        module.fallback_grad.grad,
    ]


def test_collect_grads_for_norm_filters_duplicated_names_by_tp_rank(monkeypatch):
    class _GradModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.keep = nn.Parameter(torch.tensor([1.0]))
            self.duplicated_name = nn.Parameter(torch.tensor([2.0]))
            self.keep.main_grad = torch.tensor([3.0])
            self.duplicated_name.main_grad = torch.tensor([4.0])

    module = _GradModule()

    monkeypatch.setattr(per_trajectory, "_param_is_not_shared", lambda param: True)
    monkeypatch.setattr(
        per_trajectory,
        "_param_is_not_tensor_parallel_duplicate",
        lambda param: True,
    )
    monkeypatch.setattr(per_trajectory, "_get_tensor_model_parallel_rank", lambda: 0)

    rank_zero_grads = collect_grads_for_norm(
        [module],
        duplicated_param_names={"duplicated_name"},
    )

    monkeypatch.setattr(per_trajectory, "_get_tensor_model_parallel_rank", lambda: 1)

    rank_one_grads = collect_grads_for_norm(
        [module],
        duplicated_param_names={"duplicated_name"},
    )

    assert rank_zero_grads == [module.keep.main_grad, module.duplicated_name.main_grad]
    assert rank_one_grads == [module.keep.main_grad]
