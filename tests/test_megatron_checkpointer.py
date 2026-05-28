import importlib
import sys
import types


def _install_fake_megatron(monkeypatch):
    fake_dp_cp_group = object()

    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    dist_checkpointing = types.ModuleType("megatron.core.dist_checkpointing")
    mapping = types.ModuleType("megatron.core.dist_checkpointing.mapping")
    serialization = types.ModuleType("megatron.core.dist_checkpointing.serialization")
    strategies = types.ModuleType("megatron.core.dist_checkpointing.strategies")
    async_utils = types.ModuleType(
        "megatron.core.dist_checkpointing.strategies.async_utils"
    )
    fully_parallel = types.ModuleType(
        "megatron.core.dist_checkpointing.strategies.fully_parallel"
    )
    mpu = types.ModuleType("megatron.core.mpu")
    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")

    class FakeShardedObject:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeFullyParallelLoadStrategyWrapper:
        def __init__(self, strategy, process_group):
            self.strategy = strategy
            self.process_group = process_group

    class FakeFullyParallelSaveStrategyWrapper:
        def __init__(self, strategy, process_group):
            self.strategy = strategy
            self.process_group = process_group

    class FakeCudaRngTracker:
        def get_states(self):
            return {}

        def set_states(self, state):
            self.state = state

    class FakeAsyncCallsQueue:
        pass

    mapping.ShardedObject = FakeShardedObject
    serialization.get_default_load_sharded_strategy = lambda ckpt_dir: object()
    serialization.get_default_save_sharded_strategy = lambda backend: object()
    async_utils.AsyncCallsQueue = FakeAsyncCallsQueue
    async_utils.AsyncRequest = object
    fully_parallel.FullyParallelLoadStrategyWrapper = (
        FakeFullyParallelLoadStrategyWrapper
    )
    fully_parallel.FullyParallelSaveStrategyWrapper = (
        FakeFullyParallelSaveStrategyWrapper
    )
    mpu.get_data_parallel_group = lambda with_context_parallel=False: fake_dp_cp_group
    mpu.get_pipeline_model_parallel_rank = lambda: 0
    mpu.get_pipeline_model_parallel_world_size = lambda: 1
    mpu.get_tensor_model_parallel_rank = lambda: 0
    mpu.get_tensor_model_parallel_world_size = lambda: 1
    mpu.get_data_parallel_rank = lambda with_context_parallel=False: 0
    mpu.get_data_parallel_world_size = lambda: 1
    mpu.set_virtual_pipeline_model_parallel_rank = lambda rank: None
    tensor_parallel.get_cuda_rng_tracker = lambda: FakeCudaRngTracker()

    core.dist_checkpointing = dist_checkpointing
    core.mpu = mpu
    core.tensor_parallel = tensor_parallel
    dist_checkpointing.mapping = mapping
    dist_checkpointing.serialization = serialization
    dist_checkpointing.load_content_metadata = lambda ckpt_dir: None
    strategies.async_utils = async_utils
    strategies.fully_parallel = fully_parallel

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.dist_checkpointing": dist_checkpointing,
        "megatron.core.dist_checkpointing.mapping": mapping,
        "megatron.core.dist_checkpointing.serialization": serialization,
        "megatron.core.dist_checkpointing.strategies": strategies,
        "megatron.core.dist_checkpointing.strategies.async_utils": async_utils,
        "megatron.core.dist_checkpointing.strategies.fully_parallel": fully_parallel,
        "megatron.core.mpu": mpu,
        "megatron.core.tensor_parallel": tensor_parallel,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("areal.engine.megatron_utils.checkpointer", None)

    return fake_dp_cp_group


class RecordingModel:
    def __init__(self):
        self.calls = []

    def sharded_state_dict(self, **kwargs):
        self.calls.append(kwargs)
        return {"weight": object()}


class RecordingOptimizer:
    def __init__(self):
        self.calls = []

    def sharded_state_dict(self, model_state_dict, is_loading=False, metadata=None):
        self.calls.append(
            {
                "model_state_dict": model_state_dict,
                "is_loading": is_loading,
                "metadata": metadata,
            }
        )
        return {"param_state": object()}


class LoadableModel:
    def __init__(self):
        self.loaded_state_dicts = []

    def load_state_dict(self, state_dict):
        self.loaded_state_dicts.append(state_dict)


class LoadableOptimizer:
    def __init__(self):
        self.loaded_state_dicts = []

    def load_state_dict(self, state_dict):
        self.loaded_state_dicts.append(state_dict)


def test_generate_state_dict_passes_megatron_metadata_for_load(monkeypatch):
    fake_dp_cp_group = _install_fake_megatron(monkeypatch)
    checkpointer = importlib.import_module("areal.engine.megatron_utils.checkpointer")
    monkeypatch.setattr(checkpointer.torch.distributed, "barrier", lambda: None)

    model = RecordingModel()
    optimizer = RecordingOptimizer()
    manager = checkpointer.MegatronCheckpointManager.__new__(
        checkpointer.MegatronCheckpointManager
    )
    manager.model = [model]
    manager.optimizer = optimizer
    manager.lr_scheduler = None

    manager.generate_state_dict(
        with_model=True,
        with_optimizer=True,
        with_rng=False,
        is_loading=True,
    )

    optimizer_call = optimizer.calls[0]
    metadata = optimizer_call["metadata"]
    assert optimizer_call["is_loading"] is True
    assert metadata["distrib_optim_sharding_type"] == "dp_reshardable"
    assert metadata["singleton_local_shards"] is False
    assert metadata["chained_optim_avoid_prefix"] is True
    assert metadata["dp_cp_group"] is fake_dp_cp_group
    assert model.calls[0]["metadata"] is metadata


def test_clean_sharded_state_dict_metadata_removes_process_group(monkeypatch):
    _install_fake_megatron(monkeypatch)
    checkpointer = importlib.import_module("areal.engine.megatron_utils.checkpointer")
    process_group = object()

    cleaned = checkpointer._clean_sharded_state_dict_metadata(
        {
            "distrib_optim_sharding_type": "dp_reshardable",
            "singleton_local_shards": False,
            "chained_optim_avoid_prefix": True,
            "dp_cp_group": process_group,
        }
    )

    assert cleaned == {
        "distrib_optim_sharding_type": "dp_reshardable",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
    }


def test_save_checkpoint_persists_serializable_metadata(monkeypatch, tmp_path):
    fake_dp_cp_group = _install_fake_megatron(monkeypatch)
    checkpointer = importlib.import_module("areal.engine.megatron_utils.checkpointer")
    monkeypatch.setattr(checkpointer.torch.distributed, "barrier", lambda: None)
    captured = {}

    manager = checkpointer.MegatronCheckpointManager.__new__(
        checkpointer.MegatronCheckpointManager
    )
    manager.use_dist_checkpointing = True
    manager.async_save = False
    manager._async_queue = None

    def fake_generate_state_dict(
        with_model=True,
        with_optimizer=True,
        with_rng=True,
        is_loading=False,
        metadata=None,
    ):
        captured["generate"] = {
            "with_model": with_model,
            "with_optimizer": with_optimizer,
            "with_rng": with_rng,
            "is_loading": is_loading,
            "metadata": metadata,
        }
        return {"state": object()}

    def fake_save_dist_checkpointing(
        sharded_state_dict,
        ckpt_path,
        async_save=False,
        content_metadata=None,
    ):
        captured["save"] = {
            "sharded_state_dict": sharded_state_dict,
            "ckpt_path": ckpt_path,
            "async_save": async_save,
            "content_metadata": content_metadata,
        }
        return None

    monkeypatch.setattr(manager, "generate_state_dict", fake_generate_state_dict)
    monkeypatch.setattr(
        checkpointer, "save_dist_checkpointing", fake_save_dist_checkpointing
    )

    manager.save_checkpoint(
        str(tmp_path), with_model=True, with_optimizer=True, with_rng=True
    )

    runtime_metadata = captured["generate"]["metadata"]
    assert captured["generate"]["is_loading"] is False
    assert runtime_metadata["distrib_optim_sharding_type"] == "dp_reshardable"
    assert runtime_metadata["dp_cp_group"] is fake_dp_cp_group
    assert captured["save"]["content_metadata"] == {
        "distrib_optim_sharding_type": "dp_reshardable",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
    }


def test_load_checkpoint_uses_checkpoint_metadata_for_loading(monkeypatch, tmp_path):
    fake_dp_cp_group = _install_fake_megatron(monkeypatch)
    checkpointer = importlib.import_module("areal.engine.megatron_utils.checkpointer")
    captured = {}
    checkpoint_metadata = {
        "distrib_optim_sharding_type": "dp_reshardable",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
    }

    model = LoadableModel()
    optimizer = LoadableOptimizer()
    manager = checkpointer.MegatronCheckpointManager.__new__(
        checkpointer.MegatronCheckpointManager
    )
    manager.model = [model]
    manager.optimizer = optimizer
    manager.lr_scheduler = None
    manager.rank = 0
    manager.use_dist_checkpointing = True
    manager.use_checkpoint_opt_param_scheduler = False
    manager._async_queue = None
    loaded_model_state = {"weight": object()}
    loaded_optimizer_state = {"param_state": object()}

    def fake_generate_state_dict(
        with_model=True,
        with_optimizer=True,
        with_rng=True,
        is_loading=False,
        metadata=None,
    ):
        captured["generate"] = {
            "with_model": with_model,
            "with_optimizer": with_optimizer,
            "with_rng": with_rng,
            "is_loading": is_loading,
            "metadata": metadata,
        }
        return {"template": object()}

    def fake_load_dist_checkpointing(sharded_state_dict, ckpt_dir):
        captured["load"] = {
            "sharded_state_dict": sharded_state_dict,
            "ckpt_dir": ckpt_dir,
        }
        return {"model": loaded_model_state, "optimizer": loaded_optimizer_state}

    monkeypatch.setattr(manager, "generate_state_dict", fake_generate_state_dict)
    monkeypatch.setattr(
        checkpointer, "load_dist_checkpointing", fake_load_dist_checkpointing
    )
    monkeypatch.setattr(
        checkpointer.dist_checkpointing,
        "load_content_metadata",
        lambda ckpt_dir: checkpoint_metadata,
    )

    manager.load_checkpoint(
        str(tmp_path), with_model=True, with_optimizer=True, with_rng=False
    )

    metadata = captured["generate"]["metadata"]
    assert captured["generate"]["is_loading"] is True
    assert metadata["distrib_optim_sharding_type"] == "dp_reshardable"
    assert metadata["singleton_local_shards"] is False
    assert metadata["chained_optim_avoid_prefix"] is True
    assert metadata["dp_cp_group"] is fake_dp_cp_group
    assert model.loaded_state_dicts == [loaded_model_state]
    assert optimizer.loaded_state_dicts == [loaded_optimizer_state]


def test_generate_state_dict_collects_all_data_parallel_rng_states(monkeypatch):
    _install_fake_megatron(monkeypatch)
    checkpointer = importlib.import_module("areal.engine.megatron_utils.checkpointer")
    monkeypatch.setattr(checkpointer.torch.distributed, "barrier", lambda: None)
    captured = {}

    manager = checkpointer.MegatronCheckpointManager.__new__(
        checkpointer.MegatronCheckpointManager
    )
    manager.model = []
    manager.optimizer = None
    manager.lr_scheduler = None

    def fake_get_rng_state(use_dist_ckpt=True, data_parallel_random_init=False):
        captured["use_dist_ckpt"] = use_dist_ckpt
        captured["data_parallel_random_init"] = data_parallel_random_init
        return "rng-state"

    monkeypatch.setattr(manager, "get_rng_state", fake_get_rng_state)

    state_dict = manager.generate_state_dict(
        with_model=False,
        with_optimizer=False,
        with_rng=True,
    )

    assert state_dict["rng_state"] == "rng-state"
    assert captured == {
        "use_dist_ckpt": True,
        "data_parallel_random_init": True,
    }


def test_load_checkpoint_restores_local_data_parallel_rng_state(monkeypatch, tmp_path):
    _install_fake_megatron(monkeypatch)
    checkpointer = importlib.import_module("areal.engine.megatron_utils.checkpointer")
    captured = {}

    manager = checkpointer.MegatronCheckpointManager.__new__(
        checkpointer.MegatronCheckpointManager
    )
    manager.model = []
    manager.optimizer = None
    manager.lr_scheduler = None
    manager.rank = 0
    manager.use_dist_checkpointing = True
    manager.use_checkpoint_opt_param_scheduler = False
    manager._async_queue = None

    def fake_load_dist_checkpointing(sharded_state_dict, ckpt_dir):
        return {"rng_state": ["dp0-rng", "dp1-rng"]}

    def fake_load_rng_states(rng_states, data_parallel_random_init=False):
        captured["rng_states"] = rng_states
        captured["data_parallel_random_init"] = data_parallel_random_init

    monkeypatch.setattr(manager, "generate_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        checkpointer, "load_dist_checkpointing", fake_load_dist_checkpointing
    )
    monkeypatch.setattr(manager, "load_rng_states", fake_load_rng_states)

    manager.load_checkpoint(
        str(tmp_path),
        with_model=False,
        with_optimizer=False,
        with_rng=True,
    )

    assert captured == {
        "rng_states": ["dp0-rng", "dp1-rng"],
        "data_parallel_random_init": True,
    }


def test_load_rng_states_selects_dp_rank_with_legacy_fallback(monkeypatch):
    _install_fake_megatron(monkeypatch)
    checkpointer = importlib.import_module("areal.engine.megatron_utils.checkpointer")
    captured = {}

    monkeypatch.setattr(
        checkpointer.mpu,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 1,
    )
    monkeypatch.setattr(checkpointer, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(
        checkpointer.random,
        "setstate",
        lambda state: captured.setdefault("random", []).append(state),
    )
    monkeypatch.setattr(
        checkpointer.np.random,
        "set_state",
        lambda state: captured.setdefault("np", []).append(state),
    )
    monkeypatch.setattr(
        checkpointer.torch,
        "set_rng_state",
        lambda state: captured.setdefault("torch", []).append(state),
    )

    class FakeCudaRngTracker:
        def set_states(self, state):
            captured.setdefault("tracker", []).append(state)

    monkeypatch.setattr(
        checkpointer.tensor_parallel,
        "get_cuda_rng_tracker",
        lambda: FakeCudaRngTracker(),
    )

    def rng_state(label):
        return {
            "random_rng_state": f"random-{label}",
            "np_rng_state": f"np-{label}",
            "torch_rng_state": f"torch-{label}",
            "rng_tracker_states": f"tracker-{label}",
        }

    manager = checkpointer.MegatronCheckpointManager.__new__(
        checkpointer.MegatronCheckpointManager
    )

    manager.load_rng_states(
        [rng_state("dp0"), rng_state("dp1")],
        data_parallel_random_init=True,
    )
    manager.load_rng_states([rng_state("legacy")], data_parallel_random_init=True)

    assert captured == {
        "random": ["random-dp1", "random-legacy"],
        "np": ["np-dp1", "np-legacy"],
        "torch": ["torch-dp1", "torch-legacy"],
        "tracker": ["tracker-dp1", "tracker-legacy"],
    }
