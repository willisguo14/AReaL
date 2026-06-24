import contextlib
import copy
import importlib.util
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import areal.engine.core.per_trajectory as per_trajectory_module

_MISSING = object()


def _stub_module(name, restore):
    restore["modules"].setdefault(name, sys.modules.get(name, _MISSING))

    parent_name, _, child_name = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        restore["attrs"].setdefault(
            (parent_name, child_name),
            getattr(parent, child_name, _MISSING),
        )

    module = types.ModuleType(name)
    sys.modules[name] = module
    restore.setdefault("stubs", {})[name] = module
    if parent is not None:
        setattr(parent, child_name, module)
    return module


def _restore_modules(restore):
    for name, module in reversed(restore["modules"].items()):
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

    for (parent_name, child_name), value in reversed(restore["attrs"].items()):
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        restored_child_name = f"{parent_name}.{child_name}"
        restored_child = sys.modules.get(restored_child_name, _MISSING)
        stub_child = restore.get("stubs", {}).get(restored_child_name, _MISSING)
        current_value = getattr(parent, child_name, _MISSING)
        if value is _MISSING:
            if restored_child is _MISSING:
                if current_value is stub_child:
                    delattr(parent, child_name)
            elif current_value is _MISSING or current_value is stub_child:
                setattr(parent, child_name, restored_child)
        else:
            setattr(parent, child_name, value)


def _install_megatron_engine_import_stubs(restore):
    _stub_module("areal.models.mcore.bailing_moe_bridge", restore)

    module = _stub_module("areal.engine.megatron_utils.checkpointer", restore)
    module.MegatronCheckpointManager = type("MegatronCheckpointManager", (), {})

    module = _stub_module("areal.engine.megatron_utils.deterministic", restore)
    module.set_deterministic_algorithms = lambda *args, **kwargs: None

    module = _stub_module("areal.engine.megatron_utils.fp8", restore)
    module.FP8BlockwiseTensorHelper = type("FP8BlockwiseTensorHelper", (), {})

    module = _stub_module("areal.engine.megatron_utils.grad_cosine", restore)
    module.GradientCosineTracker = type("GradientCosineTracker", (), {})

    module = _stub_module("areal.engine.megatron_utils.megatron", restore)
    module.all_gather_param = lambda *args, **kwargs: None
    module.convert_to_hf = lambda *args, **kwargs: None
    module.get_named_parameters = lambda *args, **kwargs: []
    module.remove_padding = lambda *args, **kwargs: None

    module = _stub_module("areal.engine.megatron_utils.megatron_lora", restore)
    module.get_vllm_lora_target_modules = lambda *args, **kwargs: []

    module = _stub_module(
        "areal.engine.megatron_utils.packed_context_parallel",
        restore,
    )
    module._is_multi_modal_payload_key = lambda *args, **kwargs: False
    module.extract_vision_from_multi_modal = lambda *args, **kwargs: None
    module.packed_context_parallel_forward = lambda *args, **kwargs: None
    module.reassemble_cp_packed_logprobs = lambda *args, **kwargs: None
    module.split_packed_seqs_for_context_parallel = lambda *args, **kwargs: None

    module = _stub_module("areal.engine.megatron_utils.pipeline_parallel", restore)
    module.configure_pipeline_layer_splits = (
        lambda parallel_strategy, hf_config, tf_config: tf_config
    )

    module = _stub_module("areal.models.mcore.hf_load", restore)
    module.load_weights_from_hf_with_mbridge_fast = lambda *args, **kwargs: None

    module = _stub_module("areal.models.mcore.hf_save", restore)
    module.save_critic_value_head = lambda *args, **kwargs: None
    module.save_weights_to_hf_with_mbridge_fast = lambda *args, **kwargs: None

    module = _stub_module("areal.models.mcore.registry", restore)
    module.make_hf_and_mcore_config = lambda *args, **kwargs: (None, None)
    module.make_mcore_model = lambda *args, **kwargs: []

    module = _stub_module("areal.models.tree_attn.functional", restore)
    module._gather_packed_tree_logprobs = lambda *args, **kwargs: None
    module.gather_packed_tree_logprobs_entropy = lambda *args, **kwargs: None
    module.gather_packed_tree_vocab_stats = lambda *args, **kwargs: None
    module.merge_packed_tree_results = lambda *args, **kwargs: None

    module = _stub_module("areal.models.tree_attn.module", restore)
    module.build_tree_attn_kwargs = lambda *args, **kwargs: {}
    module.patch_bridge_for_tree_training = (
        lambda *args, **kwargs: contextlib.nullcontext()
    )

    module = _stub_module("areal.models.tree_attn.tree", restore)
    module.build_packed_tree_batch = lambda *args, **kwargs: None

    module = _stub_module("mbridge", restore)
    module.AutoBridge = SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)
    module = _stub_module("mbridge.core", restore)
    module.Bridge = object
    module.LLMBridge = object
    module.register_model = lambda *args, **kwargs: (lambda cls: cls)
    module = _stub_module("mbridge.core.bridge", restore)
    module.Bridge = object
    module = _stub_module("mbridge.core.util", restore)
    module.unwrap_model = lambda model: model

    module = _stub_module("megatron", restore)
    module.__path__ = []
    module = _stub_module("megatron.bridge", restore)
    module.AutoBridge = SimpleNamespace(from_hf_pretrained=lambda *args, **kwargs: None)
    module = _stub_module("megatron.bridge.peft", restore)
    module.__path__ = []
    module = _stub_module("megatron.bridge.peft.lora", restore)
    module.LoRA = type("LoRA", (), {})

    module = _stub_module("megatron.core", restore)
    module.__path__ = []
    module = _stub_module("megatron.core.parallel_state", restore)
    module.RankGenerator = type("RankGenerator", (), {})
    module.get_data_parallel_rank = lambda: 0
    module.get_tensor_model_parallel_rank = lambda: 0
    module.get_context_parallel_rank = lambda: 0
    _stub_module("megatron.core.tensor_parallel", restore)
    module = _stub_module("megatron.core.distributed", restore)
    module.DistributedDataParallel = type("DistributedDataParallel", (), {})
    module.finalize_model_grads = lambda *args, **kwargs: None
    module = _stub_module("megatron.core.optimizer", restore)
    module.OptimizerConfig = type("OptimizerConfig", (), {})
    module.get_megatron_optimizer = lambda *args, **kwargs: None
    module = _stub_module("megatron.core.optimizer_param_scheduler", restore)
    module.OptimizerParamScheduler = type("OptimizerParamScheduler", (), {})
    module = _stub_module("megatron.core.pipeline_parallel", restore)
    module.get_forward_backward_func = lambda *args, **kwargs: None
    module = _stub_module("megatron.core.transformer", restore)
    module.TransformerConfig = type("TransformerConfig", (), {})
    module = _stub_module("megatron.core.utils", restore)
    module.get_model_config = lambda model: None


def _load_megatron_engine_module():
    restore = {"modules": {}, "attrs": {}}
    module_name = "_test_megatron_per_trajectory_megatron_engine"
    restore["modules"][module_name] = sys.modules.get(module_name, _MISSING)

    try:
        _install_megatron_engine_import_stubs(restore)
        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).resolve().parents[1]
            / "areal"
            / "engine"
            / "megatron_engine.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_modules(restore)


megatron_engine = _load_megatron_engine_module()
MegatronEngine = megatron_engine.MegatronEngine


def test_train_batch_per_trajectory_method_exists():
    assert hasattr(MegatronEngine, "train_batch_per_trajectory")


def test_megatron_engine_declares_optimizer_step_scale_support():
    assert MegatronEngine.supports_optimizer_step_scale is True


class _GetterOnlyParamGroupsOptimizer:
    def __init__(self, lrs=(0.01, 0.02), *, raise_on_step=False):
        self._param_groups = [{"lr": lr} for lr in lrs]
        self.raise_on_step = raise_on_step
        self.step_lrs = []
        self.step_calls = 0

    @property
    def param_groups(self):
        return list(self._param_groups)

    def step(self):
        self.step_calls += 1
        self.step_lrs.append([group["lr"] for group in self.param_groups])
        if self.raise_on_step:
            raise RuntimeError("step failed")
        return True, 3.0, 0


class _FakeMBList:
    def __init__(self):
        torch = megatron_engine.torch
        self.mbs = [{"loss_mask": torch.ones(1, dtype=torch.bool)}]
        self.max_seqlen = 1

    def to(self, device):
        return self

    def __len__(self):
        return len(self.mbs)


def test_optimizer_step_scales_lr_only_during_megatron_step():
    optimizer = _GetterOnlyParamGroupsOptimizer()
    engine = object.__new__(MegatronEngine)
    engine.optimizer = optimizer

    stats = MegatronEngine.optimizer_step(engine, optimizer_step_scale=0.25)

    assert optimizer.step_calls == 1
    assert optimizer.step_lrs == [[pytest.approx(0.0025), pytest.approx(0.005)]]
    assert [group["lr"] for group in optimizer.param_groups] == [
        pytest.approx(0.01),
        pytest.approx(0.02),
    ]
    assert stats["update_successful"] == 1.0
    assert stats["grad_norm"] == pytest.approx(3.0)
    assert stats["lr"] == pytest.approx(0.01)


def test_optimizer_step_restores_lr_when_megatron_step_raises():
    optimizer = _GetterOnlyParamGroupsOptimizer(raise_on_step=True)
    engine = object.__new__(MegatronEngine)
    engine.optimizer = optimizer

    with pytest.raises(RuntimeError, match="step failed"):
        MegatronEngine.optimizer_step(engine, optimizer_step_scale=0.25)

    assert optimizer.step_calls == 1
    assert optimizer.step_lrs == [[pytest.approx(0.0025), pytest.approx(0.005)]]
    assert [group["lr"] for group in optimizer.param_groups] == [
        pytest.approx(0.01),
        pytest.approx(0.02),
    ]


def test_train_batch_passes_optimizer_step_scale(monkeypatch):
    torch = megatron_engine.torch
    events = []
    received_optimizer_step_scales = []
    engine = object.__new__(MegatronEngine)
    fake_mb_list = _FakeMBList()

    class _NoopTracker:
        def prepare(self, **kwargs):
            events.append("prepare")
            return "pending"

        def finalize(self, pending, *, update_successful):
            events.append("finalize")
            assert pending == "pending"
            assert update_successful is True
            return None

    engine.config = SimpleNamespace(is_critic=False)
    engine.device = "cpu"
    engine.model = object()
    engine.optimizer = SimpleNamespace(get_loss_scale=lambda: torch.tensor(1.0))
    engine._cached_duplicated_param_names = set()
    engine.grad_cosine_tracker = _NoopTracker()
    engine._ensure_ready = lambda: events.append("ensure")
    engine.optimizer_zero_grad = lambda: events.append("zero_grad")
    engine._normalize_batch_input = lambda input_: (input_, None)
    engine._prepare_mb_list = lambda input_: fake_mb_list
    engine.forward_backward_batch = (
        lambda mb_list, process_output, forward_only=False: events.append(
            "forward_backward"
        )
    )

    def fake_optimizer_step(*, optimizer_step_scale=1.0):
        events.append("optimizer_step")
        received_optimizer_step_scales.append(optimizer_step_scale)
        return {"update_successful": 1.0, "grad_norm": 1.0, "lr": 0.01}

    engine.optimizer_step = fake_optimizer_step
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_data_parallel_group",
        lambda *args, **kwargs: "dp_group",
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_data_parallel_world_size",
        lambda: 1,
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_context_parallel_world_size",
        lambda: 1,
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine,
        "compute_total_loss_weight",
        lambda mb_list, loss_weight_fn, group: torch.tensor(1.0),
    )

    stats = MegatronEngine.train_batch(
        engine,
        {"loss_mask": torch.ones(1, dtype=torch.bool)},
        loss_fn=lambda *args: torch.tensor(0.0),
        loss_weight_fn=lambda mb: mb["loss_mask"].count_nonzero(),
        optimizer_step_scale=0.4,
    )

    assert events == [
        "ensure",
        "zero_grad",
        "forward_backward",
        "prepare",
        "optimizer_step",
        "finalize",
    ]
    assert received_optimizer_step_scales == [0.4]
    assert stats["num_micro_batches"] == 1


def test_restore_modules_preserves_restored_parent_child_attribute(monkeypatch):
    parent = types.ModuleType("megatron")
    child = types.ModuleType("megatron.core")
    parent.core = child
    monkeypatch.setitem(sys.modules, "megatron", parent)
    monkeypatch.setitem(sys.modules, "megatron.core", child)
    restore = {"modules": {}, "attrs": {}}

    _stub_module("megatron", restore)
    _stub_module("megatron.core", restore)
    _restore_modules(restore)

    assert sys.modules["megatron"] is parent
    assert sys.modules["megatron.core"] is child
    assert parent.core is child


def _engine_stub(**overrides):
    engine = object.__new__(MegatronEngine)
    engine.config = SimpleNamespace(
        backend="megatron:d1p1t1",
        is_critic=False,
        disable_dropout=True,
        use_lora=False,
        experiment_name="exp",
        trial_name="trial",
        per_trajectory=SimpleNamespace(
            enabled=True,
            flush_threshold=256,
            mask_filters=[],
        ),
    )
    engine.parallel_strategy = SimpleNamespace(
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    engine.enable_tree_training = False
    engine.enable_fp8 = False
    engine.bridge_lora = None
    engine.mcore_config = SimpleNamespace(
        ddp=SimpleNamespace(overlap_grad_reduce=False)
    )
    engine.model = [SimpleNamespace()]
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


def _patch_megatron_ranks(monkeypatch, *, dp_rank=0, tp_rank=0, cp_rank=0):
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_data_parallel_rank",
        lambda: dp_rank,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_tensor_model_parallel_rank",
        lambda: tp_rank,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_context_parallel_rank",
        lambda: cp_rank,
    )


def test_reward_from_batch_uses_task_reward_fallback():
    reward = MegatronEngine._reward_from_batch(
        {"task_reward": megatron_engine.torch.tensor([2.5])}
    )

    assert reward == pytest.approx(2.5)


def test_reward_from_batch_prefers_rewards_over_task_reward():
    reward = MegatronEngine._reward_from_batch(
        {
            "rewards": megatron_engine.torch.tensor([1.25]),
            "task_reward": megatron_engine.torch.tensor([2.5]),
        }
    )

    assert reward == pytest.approx(1.25)


class _FakeMicroBatchList:
    def __init__(self, mbs):
        self.mbs = list(mbs)

    def __len__(self):
        return len(self.mbs)

    def to(self, device):
        self.device = device
        return self


class _FakeOptimizer:
    def get_loss_scale(self):
        return megatron_engine.torch.tensor(2.0)


class _FakeGradCosineTracker:
    def __init__(self):
        self.prepare_calls = []
        self.finalize_calls = []

    def prepare(self, **kwargs):
        self.prepare_calls.append(kwargs)
        return "grad-cosine-pending"

    def finalize(self, pending, *, update_successful):
        self.finalize_calls.append((pending, update_successful))
        return 0.25


class _FakePerTrajectoryTracer:
    def __init__(self):
        self.records = []
        self.closed = False

    def write(self, record):
        self.records.append(record)

    def close(self):
        self.closed = True


def _make_per_trajectory_input(torch):
    return {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.bool),
        "loss_mask": torch.tensor([[False, True], [False, True]]),
        "rollout_logprobs": torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
        "rollout_loss_mask": torch.tensor([[False, True], [False, True]]),
        "task_reward": torch.tensor([1.0, 0.0]),
        "trainer_global_step": torch.tensor([9, 9]),
        "trainer_fileroot": "/tmp/unused",
        "uid": ["traj-a", "traj-b"],
    }


def _setup_per_trajectory_filter_fixture(
    monkeypatch,
    *,
    grad_norms,
    grad_norm_limit=3.0,
):
    torch = megatron_engine.torch
    mb_spec = megatron_engine.MicroBatchSpec(
        n_mbs=2,
        max_tokens_per_mb=128,
        packing_algorithm="kk",
    )
    engine = _engine_stub()
    engine.config.mb_spec = mb_spec
    engine.config.per_trajectory.mask_filters = [
        SimpleNamespace(
            rule="grad_norm_exceeds_max",
            params={"max": grad_norm_limit},
        )
    ]
    engine.config.pad_to_maximum = False
    engine.device = torch.device("cpu")
    engine.is_offload = False
    engine.optimizer = _FakeOptimizer()
    engine.optimizer.param_groups = [{"lr": 0.01}]
    engine.grad_cosine_tracker = _FakeGradCosineTracker()
    engine._cached_duplicated_param_names = set()
    tracer = _FakePerTrajectoryTracer()
    events = []
    grad_norm_iter = iter(grad_norms)
    input_batched = _make_per_trajectory_input(torch)

    def fake_prepare_mb_list(input_, mb_spec=None):
        effective_spec = mb_spec or engine.config.mb_spec
        if int(input_["attention_mask"].shape[0]) == 1:
            mbs = [{"loss_mask": input_["loss_mask"]}]
        else:
            mbs = [
                {"loss_mask": input_["loss_mask"]} for _ in range(effective_spec.n_mbs)
            ]
        return _FakeMicroBatchList(mbs)

    def fake_forward_backward_batch(mb_list, process_output, forward_only=False):
        events.append(("forward_backward", len(mb_list), forward_only))
        for mb in mb_list.mbs:
            output = torch.zeros_like(mb["loss_mask"], dtype=torch.float32)
            process_output(output, mb)

    def fake_compute_logprobs_and_loss(
        output,
        inputs,
        loss_fn,
        loss_weight_fn,
        total_loss_weight,
        *,
        loss_multiplier=1.0,
        logprob_callback=None,
        loss_stat_callback=None,
        logprob_grad_accumulator=None,
        logprob_grad_loss_multiplier=None,
    ):
        events.append(
            (
                "loss",
                float(loss_multiplier),
                logprob_grad_accumulator is not None,
                logprob_grad_loss_multiplier,
            )
        )
        if logprob_callback is not None:
            logprob_callback(
                torch.tensor([[-0.5, -1.5]], dtype=torch.float32),
                inputs,
            )
        if loss_stat_callback is not None:
            loss_stat_callback(
                {
                    "loss_mask": torch.tensor([[True, True]]),
                    "entropy": torch.tensor([[0.25, 0.75]]),
                    "loss_advantage": torch.tensor([[1.0, 3.0]]),
                    "behave_mask": torch.tensor([[True, True]], dtype=torch.bool),
                    "behave_imp_weight": torch.tensor([[1.0, 3.0]]),
                    "behave_approx_kl": torch.tensor([[-0.5, 0.25]]),
                }
            )
        if logprob_grad_accumulator is not None:
            logprobs = torch.tensor(
                [[0.1, 0.2]],
                dtype=torch.float32,
                requires_grad=True,
            )
            logprob_grad_accumulator.add(logprobs.sum(), logprobs, 1.0)
        return output.sum()

    monkeypatch.setattr(engine, "_prepare_mb_list", fake_prepare_mb_list)
    monkeypatch.setattr(engine, "forward_backward_batch", fake_forward_backward_batch)
    monkeypatch.setattr(
        engine,
        "_compute_logprobs_and_loss",
        fake_compute_logprobs_and_loss,
    )
    monkeypatch.setattr(
        engine,
        "optimizer_zero_grad",
        lambda: events.append(("zero_grad",)),
    )
    monkeypatch.setattr(
        engine,
        "optimizer_step",
        lambda *, optimizer_step_scale=1.0: events.append(("optimizer_step",))
        or {"update_successful": 1.0, "grad_norm": 5.0, "lr": 0.01},
    )
    monkeypatch.setattr(engine, "_make_per_trajectory_tracer", lambda _input: tracer)
    monkeypatch.setattr(
        megatron_engine,
        "compute_total_loss_weight",
        lambda mb_list, loss_weight_fn, group: torch.tensor(2.0),
    )
    monkeypatch.setattr(
        megatron_engine,
        "allocate_grad_accum_buffers",
        lambda model: events.append(("allocate",)) or ["accum"],
    )
    monkeypatch.setattr(
        megatron_engine,
        "zero_grad_accum_buffers",
        lambda buffers: events.append(("zero_accum", tuple(buffers))),
    )
    monkeypatch.setattr(
        megatron_engine,
        "accumulate_grad_buffers",
        lambda model, buffers: events.append(("accumulate", tuple(buffers))),
    )
    monkeypatch.setattr(
        megatron_engine,
        "copy_accum_buffers_to_grad_buffers",
        lambda model, buffers: events.append(("copy_back", tuple(buffers))),
    )
    monkeypatch.setattr(
        megatron_engine,
        "grad_norm_from_model_parallel_stats",
        lambda **kwargs: next(grad_norm_iter),
    )
    monkeypatch.setattr(
        megatron_engine,
        "finish_dp_grad_sync",
        lambda model: events.append(("finish_dp_sync",)),
    )
    monkeypatch.setattr(
        megatron_engine,
        "restore_dp_sync",
        lambda model, state: events.append(("restore_dp_sync", state)),
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_data_parallel_group",
        lambda *args, **kwargs: "dp-group",
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_data_parallel_world_size",
        lambda *args, **kwargs: 1,
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_tensor_model_parallel_group",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_tensor_model_parallel_world_size",
        lambda *args, **kwargs: 1,
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.dist,
        "get_world_size",
        lambda group=None: 1,
    )
    return engine, input_batched, tracer, events


def _train_per_trajectory_fixture(
    engine,
    input_batched,
    *,
    collect_logprob=False,
    optimizer_step_scale=1.0,
):
    torch = megatron_engine.torch
    return engine.train_batch_per_trajectory(
        input_batched,
        loss_fn=lambda *args, **kwargs: torch.tensor(0.0),
        loss_weight_fn=lambda mb: mb["loss_mask"].count_nonzero(),
        minibatch_idx=4,
        optimizer_step_scale=optimizer_step_scale,
        collect_logprob_grad_stats=collect_logprob,
    )


def test_train_batch_per_trajectory_uses_single_mb_spec_for_sliced_trajectories(
    monkeypatch,
):
    torch = megatron_engine.torch
    mb_spec = megatron_engine.MicroBatchSpec(
        n_mbs=3,
        max_tokens_per_mb=128,
        packing_algorithm="kk",
    )
    engine = _engine_stub()
    engine.config.mb_spec = mb_spec
    engine.config.pad_to_maximum = False
    engine.device = torch.device("cpu")
    engine.is_offload = False
    engine.optimizer = _FakeOptimizer()
    engine.grad_cosine_tracker = _FakeGradCosineTracker()
    engine._cached_duplicated_param_names = set()
    tracer = _FakePerTrajectoryTracer()

    events = []
    prepare_effective_specs = []

    input_batched = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.bool),
        "loss_mask": torch.tensor([[False, True], [False, True]]),
        "logprobs": torch.tensor([[0.0, -1.0], [0.0, -2.0]]),
        "prox_logp": torch.tensor([[0.0, -0.5], [0.0, -3.5]]),
        "rollout_logprobs": torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
        "rollout_loss_mask": torch.tensor([[False, True], [False, True]]),
        "task_reward": torch.tensor([1.0, 0.0]),
        "trainer_global_step": torch.tensor([9, 9]),
        "trainer_fileroot": "/tmp/unused",
        "uid": ["traj-a", "traj-b"],
    }

    def fake_prepare_mb_list(input_, mb_spec=None):
        effective_spec = mb_spec or engine.config.mb_spec
        prepare_effective_specs.append(effective_spec)
        if int(input_["attention_mask"].shape[0]) == 1:
            mbs = [{"loss_mask": input_["loss_mask"]}]
        else:
            mbs = [
                {"loss_mask": input_["loss_mask"]} for _ in range(effective_spec.n_mbs)
            ]
        return _FakeMicroBatchList(mbs)

    def fake_forward_backward_batch(mb_list, process_output, forward_only=False):
        events.append(("forward_backward", len(mb_list), forward_only))
        for mb in mb_list.mbs:
            output = torch.zeros_like(mb["loss_mask"], dtype=torch.float32)
            process_output(output, mb)

    def fake_compute_logprobs_and_loss(
        output,
        inputs,
        loss_fn,
        loss_weight_fn,
        total_loss_weight,
        *,
        loss_multiplier=1.0,
        logprob_callback=None,
        loss_stat_callback=None,
    ):
        events.append(("loss", float(loss_multiplier)))
        if logprob_callback is not None:
            logprob_callback(
                torch.tensor([[-0.5, -1.5]], dtype=torch.float32),
                inputs,
            )
        if loss_stat_callback is not None:
            loss_stat_callback(
                {
                    "loss_mask": torch.tensor([[True, True]]),
                    "entropy": torch.tensor([[0.25, 0.75]]),
                    "loss_advantage": torch.tensor([[1.0, 3.0]]),
                    "behave_mask": torch.tensor([[True, True]], dtype=torch.bool),
                    "behave_imp_weight": torch.tensor([[1.0, 3.0]]),
                    "behave_approx_kl": torch.tensor([[-0.5, 0.25]]),
                }
            )
        return output.sum()

    def fake_zero_grad():
        events.append(("zero_grad",))

    def fake_optimizer_step(*, optimizer_step_scale=1.0):
        events.append(("optimizer_step",))
        return {"update_successful": 1.0, "grad_norm": 7.0, "lr": 0.01}

    monkeypatch.setattr(engine, "_prepare_mb_list", fake_prepare_mb_list)
    monkeypatch.setattr(engine, "forward_backward_batch", fake_forward_backward_batch)
    monkeypatch.setattr(
        engine,
        "_compute_logprobs_and_loss",
        fake_compute_logprobs_and_loss,
    )
    monkeypatch.setattr(engine, "optimizer_zero_grad", fake_zero_grad)
    monkeypatch.setattr(engine, "optimizer_step", fake_optimizer_step)
    monkeypatch.setattr(engine, "_make_per_trajectory_tracer", lambda _input: tracer)

    monkeypatch.setattr(
        megatron_engine,
        "compute_total_loss_weight",
        lambda mb_list, loss_weight_fn, group: torch.tensor(2.0),
    )
    monkeypatch.setattr(
        megatron_engine,
        "allocate_grad_accum_buffers",
        lambda model: events.append(("allocate",)) or ["accum"],
    )
    monkeypatch.setattr(
        megatron_engine,
        "zero_grad_accum_buffers",
        lambda buffers: events.append(("zero_accum", tuple(buffers))),
    )
    monkeypatch.setattr(
        megatron_engine,
        "accumulate_grad_buffers",
        lambda model, buffers: events.append(("accumulate", tuple(buffers))),
    )
    monkeypatch.setattr(
        megatron_engine,
        "copy_accum_buffers_to_grad_buffers",
        lambda model, buffers: events.append(("copy_back", tuple(buffers))),
    )
    monkeypatch.setattr(
        megatron_engine,
        "grad_norm_from_model_parallel_stats",
        lambda **kwargs: 7.0,
    )
    monkeypatch.setattr(
        megatron_engine,
        "disable_dp_sync",
        lambda model: events.append(("disable_dp_sync",)) or "dp-sync-state",
    )
    monkeypatch.setattr(
        megatron_engine,
        "finish_dp_grad_sync",
        lambda model: events.append(("finish_dp_sync",)),
    )
    monkeypatch.setattr(
        megatron_engine,
        "restore_dp_sync",
        lambda model, state: events.append(("restore_dp_sync", state)),
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_data_parallel_group",
        lambda *args, **kwargs: "dp-group",
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_data_parallel_world_size",
        lambda *args, **kwargs: 2,
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.dist,
        "get_world_size",
        lambda group=None: 2,
    )

    stats = engine.train_batch_per_trajectory(
        input_batched,
        loss_fn=lambda *args, **kwargs: torch.tensor(0.0),
        loss_weight_fn=lambda mb: mb["loss_mask"].count_nonzero(),
        minibatch_idx=4,
    )

    assert [spec.n_mbs for spec in prepare_effective_specs] == [3, 1, 1]
    assert [spec.max_tokens_per_mb for spec in prepare_effective_specs] == [
        128,
        128,
        128,
    ]
    assert [spec.packing_algorithm for spec in prepare_effective_specs] == [
        "kk",
        "kk",
        "kk",
    ]
    assert [event for event in events if event[0] == "forward_backward"] == [
        ("forward_backward", 1, False),
        ("forward_backward", 1, False),
    ]
    assert [event for event in events if event[0] == "loss"] == [
        ("loss", 4.0),
        ("loss", 4.0),
    ]
    assert [event for event in events if event[0] == "accumulate"] == [
        ("accumulate", ("accum",)),
        ("accumulate", ("accum",)),
    ]
    assert ("copy_back", ("accum",)) in events
    assert ("disable_dp_sync",) in events
    assert ("finish_dp_sync",) in events
    assert ("restore_dp_sync", "dp-sync-state") in events
    assert events[-1] == ("zero_grad",)
    assert [event for event in events if event[0] == "optimizer_step"] == [
        ("optimizer_step",)
    ]
    assert len(tracer.records) == 2
    assert tracer.closed is True
    assert [record.trajectory_id for record in tracer.records] == ["traj-a", "traj-b"]
    assert [record.grad_norm for record in tracer.records] == [3.5, 3.5]
    assert [record.accepted for record in tracer.records] == [True, True]
    assert [record.reward for record in tracer.records] == [1.0, 0.0]
    assert [record.logprob_train_sum for record in tracer.records] == [-1.5, -1.5]
    assert [record.valid_response_tokens for record in tracer.records] == [1, 1]
    assert [record.behave_seq_log_weight for record in tracer.records] == [0.5, -1.5]
    assert [record.behave_seq_mean_log_ratio for record in tracer.records] == [
        0.5,
        -1.5,
    ]
    assert [record.entropy_mean for record in tracer.records] == [0.5, 0.5]
    assert [record.advantage_min for record in tracer.records] == [1.0, 1.0]
    assert [record.advantage_max for record in tracer.records] == [3.0, 3.0]
    assert [record.advantage_mean for record in tracer.records] == [2.0, 2.0]
    assert [record.behave_imp_weight_min for record in tracer.records] == [1.0, 1.0]
    assert [record.behave_imp_weight_max for record in tracer.records] == [3.0, 3.0]
    assert [record.behave_imp_weight_mean for record in tracer.records] == [2.0, 2.0]
    assert [record.behave_approx_kl_min for record in tracer.records] == [-0.5, -0.5]
    assert [record.behave_approx_kl_max for record in tracer.records] == [0.25, 0.25]
    assert [record.behave_approx_kl_mean for record in tracer.records] == [
        -0.125,
        -0.125,
    ]
    assert stats["num_micro_batches"] == 3
    assert stats["grad_cos_sim"] == pytest.approx(0.25)
    assert "trajectory_filter_count" not in stats
    assert "trajectory_filter_fraction" not in stats


def test_train_batch_per_trajectory_filters_over_threshold_grad_norm(monkeypatch):
    torch = megatron_engine.torch
    mb_spec = megatron_engine.MicroBatchSpec(
        n_mbs=2,
        max_tokens_per_mb=128,
        packing_algorithm="kk",
    )
    engine = _engine_stub()
    engine.config.mb_spec = mb_spec
    engine.config.per_trajectory.mask_filters = [
        SimpleNamespace(rule="grad_norm_exceeds_max", params={"max": 3.0})
    ]
    engine.config.pad_to_maximum = False
    engine.device = torch.device("cpu")
    engine.is_offload = False
    engine.optimizer = _FakeOptimizer()
    engine.grad_cosine_tracker = _FakeGradCosineTracker()
    engine._cached_duplicated_param_names = set()
    tracer = _FakePerTrajectoryTracer()

    events = []
    grad_norms = iter([2.0, 5.0])
    input_batched = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.bool),
        "loss_mask": torch.tensor([[False, True], [False, True]]),
        "rollout_logprobs": torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
        "rollout_loss_mask": torch.tensor([[False, True], [False, True]]),
        "task_reward": torch.tensor([1.0, 0.0]),
        "trainer_global_step": torch.tensor([9, 9]),
        "trainer_fileroot": "/tmp/unused",
        "uid": ["traj-a", "traj-b"],
    }

    def fake_prepare_mb_list(input_, mb_spec=None):
        effective_spec = mb_spec or engine.config.mb_spec
        if int(input_["attention_mask"].shape[0]) == 1:
            mbs = [{"loss_mask": input_["loss_mask"]}]
        else:
            mbs = [
                {"loss_mask": input_["loss_mask"]} for _ in range(effective_spec.n_mbs)
            ]
        return _FakeMicroBatchList(mbs)

    def fake_forward_backward_batch(mb_list, process_output, forward_only=False):
        events.append(("forward_backward", len(mb_list), forward_only))
        for mb in mb_list.mbs:
            output = torch.zeros_like(mb["loss_mask"], dtype=torch.float32)
            process_output(output, mb)

    def fake_compute_logprobs_and_loss(
        output,
        inputs,
        loss_fn,
        loss_weight_fn,
        total_loss_weight,
        *,
        loss_multiplier=1.0,
        logprob_callback=None,
        loss_stat_callback=None,
        logprob_grad_accumulator=None,
        logprob_grad_loss_multiplier=None,
    ):
        events.append(
            (
                "loss",
                float(loss_multiplier),
                logprob_grad_accumulator is not None,
                logprob_grad_loss_multiplier,
            )
        )
        if logprob_callback is not None:
            logprob_callback(
                torch.tensor([[-0.5, -1.5]], dtype=torch.float32),
                inputs,
            )
        if loss_stat_callback is not None:
            loss_stat_callback(
                {
                    "loss_mask": torch.tensor([[True, True]]),
                    "entropy": torch.tensor([[0.25, 0.75]]),
                    "loss_advantage": torch.tensor([[1.0, 3.0]]),
                    "behave_mask": torch.tensor([[True, True]], dtype=torch.bool),
                    "behave_imp_weight": torch.tensor([[1.0, 3.0]]),
                    "behave_approx_kl": torch.tensor([[-0.5, 0.25]]),
                }
            )
        if logprob_grad_accumulator is not None:
            logprobs = torch.tensor(
                [[0.1, 0.2]],
                dtype=torch.float32,
                requires_grad=True,
            )
            logprob_grad_accumulator.add(logprobs.sum(), logprobs, 1.0)
        return output.sum()

    monkeypatch.setattr(engine, "_prepare_mb_list", fake_prepare_mb_list)
    monkeypatch.setattr(engine, "forward_backward_batch", fake_forward_backward_batch)
    monkeypatch.setattr(
        engine,
        "_compute_logprobs_and_loss",
        fake_compute_logprobs_and_loss,
    )
    monkeypatch.setattr(
        engine,
        "optimizer_zero_grad",
        lambda: events.append(("zero_grad",)),
    )
    monkeypatch.setattr(
        engine,
        "optimizer_step",
        lambda *, optimizer_step_scale=1.0: events.append(("optimizer_step",))
        or {"update_successful": 1.0, "grad_norm": 5.0, "lr": 0.01},
    )
    monkeypatch.setattr(engine, "_make_per_trajectory_tracer", lambda _input: tracer)
    monkeypatch.setattr(
        megatron_engine,
        "compute_total_loss_weight",
        lambda mb_list, loss_weight_fn, group: torch.tensor(2.0),
    )
    monkeypatch.setattr(
        megatron_engine,
        "allocate_grad_accum_buffers",
        lambda model: events.append(("allocate",)) or ["accum"],
    )
    monkeypatch.setattr(
        megatron_engine,
        "zero_grad_accum_buffers",
        lambda buffers: events.append(("zero_accum", tuple(buffers))),
    )
    monkeypatch.setattr(
        megatron_engine,
        "accumulate_grad_buffers",
        lambda model, buffers: events.append(("accumulate", tuple(buffers))),
    )
    monkeypatch.setattr(
        megatron_engine,
        "copy_accum_buffers_to_grad_buffers",
        lambda model, buffers: events.append(("copy_back", tuple(buffers))),
    )
    monkeypatch.setattr(
        megatron_engine,
        "grad_norm_from_model_parallel_stats",
        lambda **kwargs: next(grad_norms),
    )
    monkeypatch.setattr(
        megatron_engine,
        "finish_dp_grad_sync",
        lambda model: events.append(("finish_dp_sync",)),
    )
    monkeypatch.setattr(
        megatron_engine,
        "restore_dp_sync",
        lambda model, state: events.append(("restore_dp_sync", state)),
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_data_parallel_group",
        lambda *args, **kwargs: "dp-group",
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_data_parallel_world_size",
        lambda *args, **kwargs: 1,
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_tensor_model_parallel_group",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.mpu,
        "get_tensor_model_parallel_world_size",
        lambda *args, **kwargs: 1,
        raising=False,
    )
    monkeypatch.setattr(
        megatron_engine.dist,
        "get_world_size",
        lambda group=None: 1,
    )

    stats = engine.train_batch_per_trajectory(
        input_batched,
        loss_fn=lambda *args, **kwargs: torch.tensor(0.0),
        loss_weight_fn=lambda mb: mb["loss_mask"].count_nonzero(),
        minibatch_idx=4,
        collect_logprob_grad_stats=True,
    )

    assert [event for event in events if event[0] == "forward_backward"] == [
        ("forward_backward", 1, False),
        ("forward_backward", 1, False),
    ]
    assert [event for event in events if event[0] == "accumulate"] == [
        ("accumulate", ("accum",)),
    ]
    assert ("copy_back", ("accum",)) in events
    assert [record.grad_norm for record in tracer.records] == [2.0, 5.0]
    assert [record.accepted for record in tracer.records] == [True, False]
    assert stats["trajectory_filter_count"] == pytest.approx(1.0)
    assert stats["trajectory_filter_fraction"] == pytest.approx(0.5)
    assert stats["logp_grad_norm"] == pytest.approx(math.sqrt(2.0))
    assert stats["logp_grad_absmax"] == pytest.approx(1.0)


def test_train_batch_per_trajectory_passes_optimizer_step_scale(monkeypatch):
    engine, input_batched, _tracer, events = _setup_per_trajectory_filter_fixture(
        monkeypatch,
        grad_norms=[2.0, 3.0],
    )
    received_scales = []

    def fake_optimizer_step(*, optimizer_step_scale=1.0):
        events.append(("optimizer_step", optimizer_step_scale))
        received_scales.append(optimizer_step_scale)
        return {"update_successful": 1.0, "grad_norm": 5.0, "lr": 0.01}

    monkeypatch.setattr(engine, "optimizer_step", fake_optimizer_step)

    _train_per_trajectory_fixture(
        engine,
        input_batched,
        optimizer_step_scale=0.4,
    )

    assert received_scales == [0.4]
    assert ("optimizer_step", 0.4) in events


def test_train_batch_per_trajectory_and_composes_processed_summary_mask_filters(
    monkeypatch,
):
    torch = megatron_engine.torch
    engine, input_batched, tracer, events = _setup_per_trajectory_filter_fixture(
        monkeypatch,
        grad_norms=[2.0, 2.0],
    )
    engine.config.per_trajectory.mask_filters = [
        SimpleNamespace(rule="kl_k1_outside_range", params={"upper": 0.2}),
        SimpleNamespace(rule="advantage_mean_positive", params={}),
    ]
    trajectory_seen = {"idx": 0}

    def fake_compute_logprobs_and_loss(
        output,
        inputs,
        loss_fn,
        loss_weight_fn,
        total_loss_weight,
        *,
        loss_multiplier=1.0,
        logprob_callback=None,
        loss_stat_callback=None,
        logprob_grad_accumulator=None,
        logprob_grad_loss_multiplier=None,
    ):
        del loss_fn, loss_weight_fn, total_loss_weight
        del logprob_grad_accumulator, logprob_grad_loss_multiplier
        events.append(("loss", float(loss_multiplier)))
        if logprob_callback is not None:
            logprob_callback(torch.tensor([[-0.5, -1.5]], dtype=torch.float32), inputs)
        idx = trajectory_seen["idx"]
        trajectory_seen["idx"] += 1
        advantage = (
            torch.tensor([[1.0, 3.0]], dtype=torch.float32)
            if idx == 0
            else torch.tensor([[-3.0, -1.0]], dtype=torch.float32)
        )
        if loss_stat_callback is not None:
            loss_stat_callback(
                {
                    "loss_mask": torch.tensor([[True, True]]),
                    "entropy": torch.tensor([[0.25, 0.75]]),
                    "loss_advantage": advantage,
                    "behave_mask": torch.tensor([[True, True]], dtype=torch.bool),
                    "behave_imp_weight": torch.tensor([[1.0, 3.0]]),
                    "behave_approx_kl": torch.tensor(
                        [[0.3, 0.5]],
                        dtype=torch.float32,
                    ),
                }
            )
        return output.sum()

    monkeypatch.setattr(
        engine,
        "_compute_logprobs_and_loss",
        fake_compute_logprobs_and_loss,
    )

    stats = _train_per_trajectory_fixture(engine, input_batched)

    assert [event for event in events if event[0] == "accumulate"] == [
        ("accumulate", ("accum",)),
    ]
    assert [record.accepted for record in tracer.records] == [False, True]
    assert [record.advantage_mean for record in tracer.records] == [2.0, -2.0]
    assert stats["trajectory_filter_count"] == pytest.approx(1.0)
    assert stats["trajectory_filter_fraction"] == pytest.approx(0.5)


def test_train_batch_per_trajectory_all_filtered_skips_optimizer_step(monkeypatch):
    engine, input_batched, tracer, events = _setup_per_trajectory_filter_fixture(
        monkeypatch,
        grad_norms=[5.0, 7.0],
    )

    stats = _train_per_trajectory_fixture(
        engine,
        input_batched,
        collect_logprob=True,
    )

    assert [event for event in events if event[0] == "forward_backward"] == [
        ("forward_backward", 1, False),
        ("forward_backward", 1, False),
    ]
    assert [event for event in events if event[0] == "accumulate"] == []
    assert [event for event in events if event[0] == "copy_back"] == []
    assert [event for event in events if event[0] == "optimizer_step"] == []
    assert engine.grad_cosine_tracker.prepare_calls == []
    assert engine.grad_cosine_tracker.finalize_calls == []
    assert [record.accepted for record in tracer.records] == [False, False]
    assert stats["trajectory_filter_count"] == pytest.approx(2.0)
    assert stats["trajectory_filter_fraction"] == pytest.approx(1.0)
    assert stats["update_successful"] == pytest.approx(0.0)
    assert stats["grad_norm"] == pytest.approx(0.0)
    assert stats["lr"] == pytest.approx(0.01)
    assert stats["logp_grad_norm"] == pytest.approx(0.0)
    assert stats["logp_grad_absmax"] == pytest.approx(0.0)
    assert stats["num_micro_batches"] == 2


def test_train_batch_per_trajectory_filters_non_finite_grad_norm(monkeypatch):
    engine, input_batched, tracer, events = _setup_per_trajectory_filter_fixture(
        monkeypatch,
        grad_norms=[float("nan"), 2.0],
    )

    stats = _train_per_trajectory_fixture(engine, input_batched)

    assert [event for event in events if event[0] == "accumulate"] == [
        ("accumulate", ("accum",)),
    ]
    assert math.isnan(tracer.records[0].grad_norm)
    assert [record.accepted for record in tracer.records] == [False, True]
    assert stats["trajectory_filter_count"] == pytest.approx(1.0)
    assert stats["trajectory_filter_fraction"] == pytest.approx(0.5)


def test_train_batch_per_trajectory_reports_global_filter_stats(monkeypatch):
    engine, input_batched, tracer, events = _setup_per_trajectory_filter_fixture(
        monkeypatch,
        grad_norms=[2.0, 5.0],
    )
    all_reduce_inputs = []

    def fake_all_reduce(tensor, op=None, group=None):
        all_reduce_inputs.append((tensor.detach().clone(), op, group))
        tensor[0] = 3.0
        tensor[1] = 4.0

    monkeypatch.setattr(megatron_engine.dist, "is_available", lambda: True)
    monkeypatch.setattr(megatron_engine.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(megatron_engine.dist, "all_reduce", fake_all_reduce)

    stats = _train_per_trajectory_fixture(engine, input_batched)

    assert [event for event in events if event[0] == "accumulate"] == [
        ("accumulate", ("accum",)),
    ]
    assert [record.accepted for record in tracer.records] == [True, False]
    assert len(all_reduce_inputs) == 1
    assert all_reduce_inputs[0][0].tolist() == [1.0, 2.0]
    assert all_reduce_inputs[0][2] == "dp-group"
    assert stats["trajectory_filter_count"] == pytest.approx(3.0)
    assert stats["trajectory_filter_fraction"] == pytest.approx(0.75)


def test_validate_per_trajectory_support_accepts_dense_megatron():
    engine = _engine_stub()

    engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_unwrapped_ddp():
    engine = _engine_stub()
    engine.mcore_config.wrap_with_ddp = False

    with pytest.raises(RuntimeError, match="wrap_with_ddp|DDP wrapping"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_torch_fsdp2():
    engine = _engine_stub()
    engine.mcore_config.use_torch_fsdp2 = True

    with pytest.raises(RuntimeError, match="FSDP|use_torch_fsdp2"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_custom_fsdp():
    engine = _engine_stub()
    engine.mcore_config.use_custom_fsdp = True

    with pytest.raises(RuntimeError, match="FSDP|use_custom_fsdp"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_pipeline_parallel():
    engine = _engine_stub(
        parallel_strategy=SimpleNamespace(
            pipeline_parallel_size=2,
            expert_parallel_size=1,
        )
    )

    with pytest.raises(RuntimeError, match="pipeline_parallel_size"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_context_parallel():
    engine = _engine_stub()
    engine.parallel_strategy.context_parallel_size = 2

    with pytest.raises(RuntimeError, match="context_parallel_size|context parallel"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_fp8():
    engine = _engine_stub(enable_fp8=True)

    with pytest.raises(RuntimeError, match="FP8"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_lora():
    engine = _engine_stub()
    engine.config.use_lora = True

    with pytest.raises(RuntimeError, match="LoRA"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_bridge_lora():
    engine = _engine_stub(bridge_lora=object())

    with pytest.raises(RuntimeError, match="LoRA"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_tree_training():
    engine = _engine_stub(enable_tree_training=True)

    with pytest.raises(RuntimeError, match="tree"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_critics():
    engine = _engine_stub()
    engine.config.is_critic = True

    with pytest.raises(RuntimeError, match="actor-only"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_expert_parallelism():
    engine = _engine_stub(
        parallel_strategy=SimpleNamespace(
            pipeline_parallel_size=1,
            expert_parallel_size=2,
            expert_tensor_parallel_size=1,
        )
    )

    with pytest.raises(RuntimeError, match="expert parallelism"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_expert_tensor_parallelism():
    engine = _engine_stub(
        parallel_strategy=SimpleNamespace(
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            expert_tensor_parallel_size=2,
        )
    )

    with pytest.raises(RuntimeError, match="expert tensor parallelism"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_tf_config_moe_experts():
    engine = _engine_stub(tf_config=SimpleNamespace(num_moe_experts=8))

    with pytest.raises(RuntimeError, match="num_moe_experts"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_mcore_config_moe_experts():
    engine = _engine_stub(
        mcore_config=SimpleNamespace(
            ddp=SimpleNamespace(overlap_grad_reduce=False),
            num_moe_experts=8,
        )
    )

    with pytest.raises(RuntimeError, match="num_moe_experts"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_rejects_overlap_grad_reduce():
    engine = _engine_stub(
        mcore_config=SimpleNamespace(ddp=SimpleNamespace(overlap_grad_reduce=True))
    )

    with pytest.raises(RuntimeError, match="overlap_grad_reduce"):
        engine._validate_per_trajectory_supported()


def test_validate_per_trajectory_support_allows_missing_mcore_config():
    engine = _engine_stub()
    delattr(engine, "mcore_config")

    engine._validate_per_trajectory_supported()


@pytest.mark.parametrize(
    "bad_fileroot",
    ["", None, Path("/tmp/not-a-string")],
)
def test_per_trajectory_tracer_path_requires_trainer_fileroot(
    monkeypatch,
    bad_fileroot,
):
    engine = _engine_stub()
    _patch_megatron_ranks(monkeypatch, dp_rank=3, tp_rank=0)

    with pytest.raises(RuntimeError, match="trainer_fileroot"):
        engine._per_trajectory_tracer_path({"trainer_fileroot": bad_fileroot})


def test_per_trajectory_tracer_path_uses_ranked_actor_log_path(monkeypatch):
    engine = _engine_stub()
    _patch_megatron_ranks(monkeypatch, dp_rank=7, tp_rank=2)
    monkeypatch.setattr(per_trajectory_module.getpass, "getuser", lambda: "alice")

    path = engine._per_trajectory_tracer_path({"trainer_fileroot": "/tmp/areal"})

    assert path == (
        "/tmp/areal/logs/alice/exp/trial/per_trajectory/actor/dp_00007_tp_00002.jsonl"
    )


def test_make_per_trajectory_tracer_enables_tensor_and_context_parallel_rank_zero(
    monkeypatch,
):
    engine = _engine_stub()
    _patch_megatron_ranks(monkeypatch, dp_rank=1, tp_rank=0, cp_rank=0)
    monkeypatch.setattr(per_trajectory_module.getpass, "getuser", lambda: "alice")

    tracer = engine._make_per_trajectory_tracer({"trainer_fileroot": "/tmp/areal"})

    assert tracer.enabled is True
    assert tracer.flush_threshold == 256
    assert tracer.path == Path(
        "/tmp/areal/logs/alice/exp/trial/per_trajectory/actor/dp_00001_tp_00000.jsonl"
    )


def test_make_per_trajectory_tracer_disables_nonzero_context_parallel_rank(
    monkeypatch,
):
    engine = _engine_stub()
    _patch_megatron_ranks(monkeypatch, dp_rank=1, tp_rank=0, cp_rank=1)
    monkeypatch.setattr(per_trajectory_module.getpass, "getuser", lambda: "alice")

    tracer = engine._make_per_trajectory_tracer({"trainer_fileroot": "/tmp/areal"})

    assert tracer.enabled is False
    assert tracer.path == Path(
        "/tmp/areal/logs/alice/exp/trial/per_trajectory/actor/dp_00001_tp_00000.jsonl"
    )


def test_make_per_trajectory_tracer_treats_missing_context_rank_as_zero(
    monkeypatch,
):
    engine = _engine_stub()
    _patch_megatron_ranks(monkeypatch, dp_rank=1, tp_rank=0, cp_rank=1)
    monkeypatch.delattr(megatron_engine.mpu, "get_context_parallel_rank")
    monkeypatch.setattr(per_trajectory_module.getpass, "getuser", lambda: "alice")

    tracer = engine._make_per_trajectory_tracer({"trainer_fileroot": "/tmp/areal"})

    assert tracer.enabled is True


def test_make_per_trajectory_tracer_disables_nonzero_tensor_parallel_rank(
    monkeypatch,
):
    engine = _engine_stub()
    _patch_megatron_ranks(monkeypatch, dp_rank=1, tp_rank=1)
    monkeypatch.setattr(per_trajectory_module.getpass, "getuser", lambda: "alice")

    tracer = engine._make_per_trajectory_tracer({"trainer_fileroot": "/tmp/areal"})

    assert tracer.enabled is False
    assert tracer.path == Path(
        "/tmp/areal/logs/alice/exp/trial/per_trajectory/actor/dp_00001_tp_00001.jsonl"
    )


def test_make_per_trajectory_tracer_respects_config_disabled(monkeypatch):
    engine = _engine_stub()
    engine.config.per_trajectory.enabled = False
    _patch_megatron_ranks(monkeypatch, dp_rank=1, tp_rank=0)
    monkeypatch.setattr(per_trajectory_module.getpass, "getuser", lambda: "alice")

    tracer = engine._make_per_trajectory_tracer({"trainer_fileroot": "/tmp/areal"})

    assert tracer.enabled is False


def _assert_finite_trace_value(row, key):
    assert key in row
    assert not isinstance(row[key], bool)
    assert isinstance(row[key], int | float)
    assert math.isfinite(float(row[key]))
    return float(row[key])


def _assert_trace_mean_matches_sum(row, prefix, *, expected_sum=None):
    response_length = row["response_length"]
    assert response_length == 6
    trace_sum = _assert_finite_trace_value(row, f"{prefix}_sum")
    trace_mean = _assert_finite_trace_value(row, f"{prefix}_mean")
    if expected_sum is not None:
        assert trace_sum == pytest.approx(expected_sum, rel=1e-6, abs=1e-7)
    expected_mean = trace_sum / response_length
    if expected_sum is not None:
        expected_mean = expected_sum / response_length
    assert trace_mean == pytest.approx(
        expected_mean,
        rel=1e-6,
        abs=1e-7,
    )


_EXPECTED_PER_TRAJECTORY_TRACE_ROWS = (
    {
        "trainer_global_step": 3,
        "minibatch_idx": 0,
        "trajectory_idx": 0,
        "trajectory_id": "traj-a",
        "response_length": 6,
        "accepted": True,
        "reward": 0.75,
        "logprob_infer_sum": -2.39,
    },
    {
        "trainer_global_step": 3,
        "minibatch_idx": 0,
        "trajectory_idx": 1,
        "trajectory_id": "traj-b",
        "response_length": 6,
        "accepted": True,
        "reward": -0.25,
        "logprob_infer_sum": -2.26,
    },
)


def _assert_trace_field_matches(row, key, expected):
    assert key in row
    assert type(row[key]) is type(expected)
    assert row[key] == expected


def _assert_per_trajectory_trace_row(row, expected):
    for key in (
        "trainer_global_step",
        "minibatch_idx",
        "trajectory_idx",
        "trajectory_id",
        "response_length",
    ):
        _assert_trace_field_matches(row, key, expected[key])
    _assert_trace_field_matches(row, "accepted", expected["accepted"])
    reward = _assert_finite_trace_value(row, "reward")
    assert reward == pytest.approx(expected["reward"], rel=0.0, abs=0.0)
    _assert_finite_trace_value(row, "grad_norm")
    _assert_trace_mean_matches_sum(row, "logprob_train")
    _assert_trace_mean_matches_sum(
        row,
        "logprob_infer",
        expected_sum=expected["logprob_infer_sum"],
    )


def _assert_per_trajectory_trace_records(tmp_path):
    trace_files = sorted(
        tmp_path.glob("logs/*/per-traj-test/trial0/per_trajectory/actor/*.jsonl")
    )
    assert len(trace_files) == 1

    rows = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2

    assert len({row["trajectory_id"] for row in rows}) == len(rows)
    for row, expected in zip(rows, _EXPECTED_PER_TRAJECTORY_TRACE_ROWS):
        _assert_per_trajectory_trace_row(row, expected)


def _valid_per_trajectory_trace_rows():
    return [
        {
            "trainer_global_step": 3,
            "minibatch_idx": 0,
            "trajectory_idx": 0,
            "trajectory_id": "traj-a",
            "accepted": True,
            "grad_norm": 1.25,
            "logprob_train_sum": -3.0,
            "logprob_train_mean": -0.5,
            "logprob_infer_sum": -2.39,
            "logprob_infer_mean": -2.39 / 6,
            "reward": 0.75,
            "response_length": 6,
        },
        {
            "trainer_global_step": 3,
            "minibatch_idx": 0,
            "trajectory_idx": 1,
            "trajectory_id": "traj-b",
            "accepted": True,
            "grad_norm": 2.5,
            "logprob_train_sum": -1.2,
            "logprob_train_mean": -0.2,
            "logprob_infer_sum": -2.26,
            "logprob_infer_mean": -2.26 / 6,
            "reward": -0.25,
            "response_length": 6,
        },
    ]


def _write_per_trajectory_trace_rows(tmp_path, rows):
    trace_dir = (
        tmp_path
        / "logs"
        / "alice"
        / "per-traj-test"
        / "trial0"
        / "per_trajectory"
        / "actor"
    )
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "dp_00000_tp_00000.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_assert_per_trajectory_trace_records_validates_jsonl_rows(tmp_path):
    _write_per_trajectory_trace_rows(tmp_path, _valid_per_trajectory_trace_rows())

    _assert_per_trajectory_trace_records(tmp_path)


def test_assert_per_trajectory_trace_records_rejects_wrong_second_row_metadata(
    tmp_path,
):
    rows = _valid_per_trajectory_trace_rows()
    rows[1]["minibatch_idx"] = 1
    _write_per_trajectory_trace_rows(tmp_path, rows)

    with pytest.raises(AssertionError):
        _assert_per_trajectory_trace_records(tmp_path)


def test_assert_per_trajectory_trace_records_rejects_non_finite_logprob(
    tmp_path,
):
    rows = _valid_per_trajectory_trace_rows()
    rows[0]["logprob_train_sum"] = float("inf")
    rows[0]["logprob_train_mean"] = float("inf")
    _write_per_trajectory_trace_rows(tmp_path, rows)

    with pytest.raises(AssertionError):
        _assert_per_trajectory_trace_records(tmp_path)


def _clone_batch_value(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_batch_value(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_batch_value(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_batch_value(item, torch) for item in value)
    return copy.deepcopy(value)


def _snapshot_model_parameters(engine, torch):
    assert engine.model is not None
    return {
        name: param.detach().cpu().clone()
        for name, param in engine.model.named_parameters()
    }


def _restore_model_parameters(engine, snapshot, torch):
    assert engine.model is not None
    with torch.no_grad():
        for name, param in engine.model.named_parameters():
            param.copy_(snapshot[name].to(device=param.device, dtype=param.dtype))


_DISTRIBUTED_ENV_KEYS = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _capture_distributed_env():
    return {key: os.environ.get(key) for key in _DISTRIBUTED_ENV_KEYS}


def _restore_distributed_env(snapshot):
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _capture_rng_state(torch):
    import numpy as np

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
        state["torch_deterministic_algorithms_warn_only"] = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        state["cudnn_deterministic"] = cudnn_backend.deterministic
        state["cudnn_benchmark"] = cudnn_backend.benchmark
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state, torch):
    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "torch_deterministic_algorithms" in state:
        torch.use_deterministic_algorithms(
            state["torch_deterministic_algorithms"],
            warn_only=state.get("torch_deterministic_algorithms_warn_only", False),
        )
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        if "cudnn_deterministic" in state:
            cudnn_backend.deterministic = state["cudnn_deterministic"]
        if "cudnn_benchmark" in state:
            cudnn_backend.benchmark = state["cudnn_benchmark"]


def _destroy_engine_and_process_group(engine, dist, *, dist_was_initialized):
    try:
        if engine is not None:
            engine.destroy()
    finally:
        if dist.is_initialized() and not dist_was_initialized:
            dist.destroy_process_group()


class _FakeDistForCleanup:
    def __init__(self, *, initialized):
        self.initialized = initialized
        self.destroy_calls = 0

    def is_initialized(self):
        return self.initialized

    def destroy_process_group(self):
        self.destroy_calls += 1
        self.initialized = False


class _FakeEngineForCleanup:
    def __init__(self):
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1


def test_process_group_cleanup_respects_preexisting_dist_ownership():
    preexisting_dist = _FakeDistForCleanup(initialized=True)
    preexisting_engine = _FakeEngineForCleanup()

    _destroy_engine_and_process_group(
        preexisting_engine,
        preexisting_dist,
        dist_was_initialized=True,
    )

    assert preexisting_engine.destroy_calls == 1
    assert preexisting_dist.destroy_calls == 0
    assert preexisting_dist.is_initialized() is True

    test_owned_dist = _FakeDistForCleanup(initialized=True)
    test_owned_engine = _FakeEngineForCleanup()

    _destroy_engine_and_process_group(
        test_owned_engine,
        test_owned_dist,
        dist_was_initialized=False,
    )

    assert test_owned_engine.destroy_calls == 1
    assert test_owned_dist.destroy_calls == 1
    assert test_owned_dist.is_initialized() is False


_AREAL_SEEDING_ATTRS = ("_SEED", "_BASE_SEED", "_SHUFFLER")


def _capture_areal_seeding_state(seeding_module):
    return {
        "attrs": {
            name: getattr(seeding_module, name, _MISSING)
            for name in _AREAL_SEEDING_ATTRS
        },
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


def _restore_areal_seeding_state(seeding_module, snapshot):
    for name, value in snapshot["attrs"].items():
        if value is _MISSING:
            if hasattr(seeding_module, name):
                delattr(seeding_module, name)
        else:
            setattr(seeding_module, name, value)

    python_hash_seed = snapshot["PYTHONHASHSEED"]
    if python_hash_seed is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = python_hash_seed


def test_distributed_env_restore_restores_original_values(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "original-world")
    monkeypatch.setenv("MASTER_ADDR", "original-master")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)

    snapshot = _capture_distributed_env()
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
    )

    _restore_distributed_env(snapshot)

    assert os.environ["WORLD_SIZE"] == "original-world"
    assert os.environ["MASTER_ADDR"] == "original-master"
    assert "RANK" not in os.environ
    assert "LOCAL_RANK" not in os.environ
    assert "MASTER_PORT" not in os.environ


def test_rng_state_restore_restores_python_numpy_torch_and_determinism():
    import numpy as np

    torch = megatron_engine.torch
    outer_state = _capture_rng_state(torch)
    try:
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.use_deterministic_algorithms(True, warn_only=True)
        captured_state = _capture_rng_state(torch)

        expected_python = random.random()
        expected_numpy = np.random.random(3)
        expected_torch = torch.rand(3)

        random.seed(4321)
        np.random.seed(4321)
        torch.manual_seed(4321)
        torch.use_deterministic_algorithms(False)

        _restore_rng_state(captured_state, torch)

        assert random.random() == expected_python
        np.testing.assert_array_equal(np.random.random(3), expected_numpy)
        torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0.0, atol=0.0)
        assert torch.are_deterministic_algorithms_enabled() is True
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
            assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    finally:
        _restore_rng_state(outer_state, torch)


def test_areal_seeding_state_restore_restores_globals_and_pythonhashseed(monkeypatch):
    shuffler = object()
    seeding_module = SimpleNamespace(_SEED=11, _BASE_SEED=22, _SHUFFLER=shuffler)

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    snapshot = _capture_areal_seeding_state(seeding_module)
    seeding_module._SEED = 101
    seeding_module._BASE_SEED = 202
    seeding_module._SHUFFLER = object()
    os.environ["PYTHONHASHSEED"] = "999"

    _restore_areal_seeding_state(seeding_module, snapshot)

    assert seeding_module._SEED == 11
    assert seeding_module._BASE_SEED == 22
    assert seeding_module._SHUFFLER is shuffler
    assert "PYTHONHASHSEED" not in os.environ

    os.environ["PYTHONHASHSEED"] = "original-hash-seed"
    snapshot = _capture_areal_seeding_state(seeding_module)
    os.environ["PYTHONHASHSEED"] = "mutated-hash-seed"

    _restore_areal_seeding_state(seeding_module, snapshot)

    assert os.environ["PYTHONHASHSEED"] == "original-hash-seed"


def _clone_optimizer_state_value(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            copy.deepcopy(key): _clone_optimizer_state_value(item, torch)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_optimizer_state_value(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_state_value(item, torch) for item in value)
    return copy.deepcopy(value)


def _optimizer_state_summary(optimizer, torch):
    return _clone_optimizer_state_value(optimizer.state_dict(), torch)


def _optimizer_child_path(path, key):
    child = str(key)
    return child if not path else f"{path}.{child}"


def _optimizer_index_path(path, index):
    return f"{path}[{index}]" if path else f"[{index}]"


def _assert_optimizer_state_values_close(expected, actual, torch, *, path=""):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), (
            f"{path}: type mismatch, expected tensor, got {type(actual).__name__}"
        )
        assert tuple(actual.shape) == tuple(expected.shape), (
            f"{path}: tensor shape mismatch, expected {tuple(expected.shape)}, "
            f"got {tuple(actual.shape)}"
        )
        assert actual.dtype == expected.dtype, (
            f"{path}: tensor dtype mismatch, expected {expected.dtype}, "
            f"got {actual.dtype}"
        )
        if expected.is_floating_point():
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-4,
                atol=1e-7,
                msg=f"{path}: floating tensor mismatch",
            )
        else:
            assert torch.equal(actual, expected), f"{path}: tensor value mismatch"
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), (
            f"{path}: type mismatch, expected dict, got {type(actual).__name__}"
        )
        assert actual.keys() == expected.keys(), (
            f"{path}: dict keys mismatch, expected {list(expected.keys())}, "
            f"got {list(actual.keys())}"
        )
        for key in sorted(expected.keys(), key=repr):
            _assert_optimizer_state_values_close(
                expected[key],
                actual[key],
                torch,
                path=_optimizer_child_path(path, key),
            )
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), (
            f"{path}: type mismatch, expected list, got {type(actual).__name__}"
        )
        assert len(actual) == len(expected), (
            f"{path}: list length mismatch, expected {len(expected)}, got {len(actual)}"
        )
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_optimizer_state_values_close(
                expected_item,
                actual_item,
                torch,
                path=_optimizer_index_path(path, index),
            )
        return

    if isinstance(expected, tuple):
        assert isinstance(actual, tuple), (
            f"{path}: type mismatch, expected tuple, got {type(actual).__name__}"
        )
        assert len(actual) == len(expected), (
            f"{path}: tuple length mismatch, expected {len(expected)}, "
            f"got {len(actual)}"
        )
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_optimizer_state_values_close(
                expected_item,
                actual_item,
                torch,
                path=_optimizer_index_path(path, index),
            )
        return

    assert actual == expected, (
        f"{path}: value mismatch, expected {expected!r}, got {actual!r}"
    )


def _assert_optimizer_state_summaries_close(normal, per_trajectory, torch):
    _assert_optimizer_state_values_close(normal, per_trajectory, torch)


class _FakeOptimizerWithState:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


def test_optimizer_state_comparison_checks_all_nested_tensors():
    torch = megatron_engine.torch
    normal_state = {
        "state": {f"param{i}": {"exp_avg": torch.tensor([float(i)])} for i in range(9)}
    }
    per_trajectory_state = copy.deepcopy(normal_state)
    per_trajectory_state["state"]["param8"]["exp_avg"] = torch.tensor([99.0])

    normal = _optimizer_state_summary(_FakeOptimizerWithState(normal_state), torch)
    per_trajectory = _optimizer_state_summary(
        _FakeOptimizerWithState(per_trajectory_state),
        torch,
    )

    with pytest.raises(AssertionError, match="state\\.param8\\.exp_avg"):
        _assert_optimizer_state_summaries_close(normal, per_trajectory, torch)


def _assert_parameter_snapshots_close(expected, actual, torch):
    assert set(actual) == set(expected)
    for name, expected_param in expected.items():
        torch.testing.assert_close(
            actual[name].to(dtype=torch.float32),
            expected_param.to(dtype=torch.float32),
            rtol=1e-4,
            atol=1e-7,
            msg=f"parameter mismatch for {name}",
        )


def _make_per_trajectory_equivalence_batch(torch, device, trainer_fileroot):
    input_ids = torch.tensor(
        [
            [101, 314, 1592, 653, 920, 88, 47, 13],
            [205, 444, 1729, 802, 377, 91, 58, 17],
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask = torch.tensor(
        [
            [False, True, True, True, True, True, True, False],
            [False, True, True, True, True, True, True, False],
        ],
        dtype=torch.bool,
        device=device,
    )
    old_logprobs = torch.tensor(
        [
            [-0.31, -0.27, -0.41, -0.36, -0.45, -0.52, -0.38, -0.44],
            [-0.22, -0.34, -0.29, -0.48, -0.33, -0.57, -0.25, -0.39],
        ],
        dtype=torch.float32,
        device=device,
    )
    advantages = torch.tensor(
        [
            [0.0, 0.75, -0.25, 0.50, -0.10, 0.30, -0.45, 0.0],
            [0.0, -0.55, 0.20, 0.65, -0.35, 0.15, 0.40, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "advantages": advantages,
        "logprobs": old_logprobs.clone(),
        "rollout_logprobs": old_logprobs.clone(),
        "rollout_loss_mask": loss_mask.clone(),
        "prox_logp": old_logprobs.clone(),
        "trainer_global_step": torch.tensor([3, 3], dtype=torch.long, device=device),
        "trainer_fileroot": str(trainer_fileroot),
        "task_reward": torch.tensor([0.75, -0.25], dtype=torch.float32, device=device),
        "uid": ["traj-a", "traj-b"],
    }


def _deterministic_actor_loss(logprobs, entropy, input_data, **kwargs):
    del entropy, kwargs
    mask = input_data["loss_mask"].bool()
    old_logprobs = input_data["rollout_logprobs"].to(dtype=logprobs.dtype)
    advantages = input_data["advantages"].to(dtype=logprobs.dtype)
    token_loss = -(logprobs - old_logprobs) * advantages
    return token_loss[mask].mean()


def _masked_token_count(mb):
    return mb["loss_mask"].count_nonzero()


def _run_train_with_step_count(engine, method_name, batch, torch):
    original_optimizer_step = engine.optimizer_step
    step_count = 0

    def counted_optimizer_step(*, optimizer_step_scale=1.0):
        nonlocal step_count
        step_count += 1
        return original_optimizer_step(optimizer_step_scale=optimizer_step_scale)

    engine.optimizer_step = counted_optimizer_step
    try:
        method = getattr(engine, method_name)
        kwargs = {}
        if method_name == "train_batch_per_trajectory":
            kwargs["minibatch_idx"] = 0
        stats = method(
            _clone_batch_value(batch, torch),
            loss_fn=_deterministic_actor_loss,
            loss_weight_fn=_masked_token_count,
            **kwargs,
        )
    finally:
        engine.optimizer_step = original_optimizer_step

    assert step_count == 1
    assert stats["update_successful"] == 1.0
    return stats


def _reset_megatron_optimizer(engine, ft_spec):
    checkpointer = getattr(engine, "checkpointer", None)
    if checkpointer is not None:
        checkpointer.close()
        engine.checkpointer = None
    engine.optimizer = None
    engine.lr_scheduler = None
    engine._create_optimizer(ft_spec)


def _qwen3_model_path_or_skip():
    local_path = Path("/storage/openpsi/models/Qwen__Qwen3-0.6B/")
    if local_path.exists():
        return str(local_path)

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id="Qwen/Qwen3-0.6B",
            ignore_patterns=["*.gguf", "*.ggml", "consolidated*"],
            local_files_only=True,
        )
    except Exception as exc:
        pytest.skip(f"Qwen3-0.6B model is unavailable locally: {exc}")


def _missing_megatron_runtime_dependencies():
    return tuple(
        module_name
        for module_name in ("mbridge", "megatron")
        if importlib.util.find_spec(module_name) is None
    )


def test_missing_megatron_runtime_dependency_check_is_narrow(monkeypatch):
    checked_modules = []

    def fake_find_spec(module_name):
        checked_modules.append(module_name)
        if module_name == "megatron":
            return object()
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    assert _missing_megatron_runtime_dependencies() == ("mbridge",)
    assert checked_modules == ["mbridge", "megatron"]


@pytest.mark.slow
def test_per_trajectory_update_matches_normal_train_batch(tmp_path):
    import torch
    import torch.distributed as dist

    pre_test_dist_initialized = dist.is_initialized()
    pre_test_env = _capture_distributed_env()
    pre_test_rng_state = _capture_rng_state(torch)
    pre_test_seeding_state = None
    seeding_module = None
    engine = None
    try:
        missing_dependencies = _missing_megatron_runtime_dependencies()
        if missing_dependencies:
            pytest.skip(
                "real Megatron runtime dependencies unavailable: missing "
                f"{', '.join(missing_dependencies)}"
            )

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            pytest.skip("real Megatron equivalence test requires a CUDA device")

        model_path = _qwen3_model_path_or_skip()

        from areal.api import FinetuneSpec
        from areal.api.alloc_mode import ModelAllocation
        from areal.api.cli_args import (
            MegatronEngineConfig,
            MicroBatchSpec,
            OptimizerConfig,
            PerTrajectoryConfig,
            PPOActorConfig,
        )
        from areal.engine import MegatronEngine as RealMegatronEngine
        from areal.utils import seeding as seeding_module
        from areal.utils.network import find_free_ports

        pre_test_seeding_state = _capture_areal_seeding_state(seeding_module)

        os.environ.update(
            {
                "WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(find_free_ports(1)[0]),
            }
        )

        config = PPOActorConfig(
            backend="megatron:d1p1t1",
            experiment_name="per-traj-test",
            trial_name="trial0",
            path=model_path,
            disable_dropout=True,
            mb_spec=MicroBatchSpec(n_mbs=2, max_tokens_per_mb=32),
            optimizer=OptimizerConfig(lr=1e-6, weight_decay=0.0),
            megatron=MegatronEngineConfig(),
            per_trajectory=PerTrajectoryConfig(enabled=True, flush_threshold=1),
        )
        alloc_mode = ModelAllocation.from_str("megatron:d1p1t1")
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2)

        seeding_module.set_random_seed(0, key="per-trajectory-equivalence")
        engine = RealMegatronEngine(config)
        engine.create_process_group(alloc_mode.parallel)
        engine.initialize(addr=None, ft_spec=ft_spec)

        batch = _make_per_trajectory_equivalence_batch(
            torch,
            engine.device,
            tmp_path,
        )
        initial_parameters = _snapshot_model_parameters(engine, torch)
        initial_rng_state = _capture_rng_state(torch)

        _restore_rng_state(initial_rng_state, torch)
        normal_stats = _run_train_with_step_count(
            engine,
            "train_batch",
            batch,
            torch,
        )
        normal_parameters = _snapshot_model_parameters(engine, torch)
        normal_optimizer = _optimizer_state_summary(engine.optimizer, torch)

        _restore_model_parameters(engine, initial_parameters, torch)
        del initial_parameters
        _reset_megatron_optimizer(engine, ft_spec)

        _restore_rng_state(initial_rng_state, torch)
        per_trajectory_stats = _run_train_with_step_count(
            engine,
            "train_batch_per_trajectory",
            batch,
            torch,
        )
        _assert_per_trajectory_trace_records(tmp_path)
        per_trajectory_parameters = _snapshot_model_parameters(engine, torch)
        per_trajectory_optimizer = _optimizer_state_summary(engine.optimizer, torch)

        assert (
            per_trajectory_stats["num_micro_batches"]
            == normal_stats["num_micro_batches"]
        )
        assert per_trajectory_stats["lr"] == pytest.approx(
            normal_stats["lr"],
            rel=0.0,
            abs=0.0,
        )
        assert per_trajectory_stats["grad_norm"] == pytest.approx(
            normal_stats["grad_norm"],
            rel=1e-4,
            abs=1e-7,
        )
        _assert_parameter_snapshots_close(
            normal_parameters,
            per_trajectory_parameters,
            torch,
        )
        _assert_optimizer_state_summaries_close(
            normal_optimizer,
            per_trajectory_optimizer,
            torch,
        )
    finally:
        try:
            _destroy_engine_and_process_group(
                engine,
                dist,
                dist_was_initialized=pre_test_dist_initialized,
            )
        finally:
            _restore_rng_state(pre_test_rng_state, torch)
            if pre_test_seeding_state is not None:
                _restore_areal_seeding_state(
                    seeding_module,
                    pre_test_seeding_state,
                )
            _restore_distributed_env(pre_test_env)
