import pytest


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("areal.engine.fsdp_engine", "FSDPEngine"),
        ("areal.engine.megatron_engine", "MegatronEngine"),
        ("areal.engine.sglang_remote", "RemoteSGLangEngine"),
        ("areal.engine.vllm_remote", "RemotevLLMEngine"),
    ],
)
def test_clear_batches_noops_without_shard_ids(module_name, class_name):
    module = pytest.importorskip(module_name)
    engine_cls = getattr(module, class_name)

    assert engine_cls.clear_batches(object()) is None
