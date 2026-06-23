import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import areal.engine.core.per_trajectory as per_trajectory_module
from areal.engine.core.per_trajectory import (
    MaskedTensorAccumulator,
    MaskedTensorSummary,
    PerTrajectoryRecord,
    PerTrajectoryTracer,
    attach_trainer_step_metadata,
    normalize_flush_threshold,
    per_trajectory_log_dir,
    preserve_rollout_logprobs,
    slice_trajectory,
    summarize_logprobs,
    trajectory_id_from_sample,
)
from areal.engine.core.per_trajectory_filters import (
    PerTrajectoryFilterContext,
    evaluate_per_trajectory_mask_filters,
)


def _record(**overrides):
    values = {
        "trainer_global_step": 1,
        "minibatch_idx": 0,
        "trajectory_idx": 0,
        "trajectory_id": "a",
        "grad_norm": 1.5,
        "logprob_train_sum": -1.0,
        "logprob_train_mean": -0.5,
        "logprob_infer_sum": -2.0,
        "logprob_infer_mean": -1.0,
        "reward": 1.0,
        "response_length": 2,
    }
    values.update(overrides)
    return PerTrajectoryRecord(**values)


def _filter_context(**overrides):
    values = {
        "grad_norm": 1.0,
        "advantage_summary": None,
        "behave_approx_kl_summary": None,
        "advantage_mean_emitted": False,
        "behave_approx_kl_mean_emitted": False,
    }
    values.update(overrides)
    return PerTrajectoryFilterContext(**values)


def _masked_summary(mean, count=1):
    if count == 0:
        return MaskedTensorSummary(min=None, max=None, mean=None, count=0)
    return MaskedTensorSummary(min=mean, max=mean, mean=mean, count=count)


def test_per_trajectory_record_defaults_accepted_true():
    record = _record()

    assert record.accepted is True
    assert asdict(record)["accepted"] is True


def test_tracer_writes_accepted_flag(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = PerTrajectoryTracer(path=path, flush_threshold=1, enabled=True)

    tracer.write(_record(accepted=False))

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["accepted"] is False
    assert "grad_norm_filtered" not in row


def test_normalize_flush_threshold_falls_back_to_one():
    assert normalize_flush_threshold(0) == 1
    assert normalize_flush_threshold(-5) == 1
    assert normalize_flush_threshold("bad") == 1
    assert normalize_flush_threshold(17) == 17


def test_core_package_reexports_normalize_flush_threshold():
    from areal.engine.core import normalize_flush_threshold as exported

    assert exported("4") == 4


def test_evaluate_per_trajectory_mask_filters_empty_list_masks_nothing():
    assert not evaluate_per_trajectory_mask_filters(
        [],
        _filter_context(grad_norm=float("nan")),
    )


@pytest.mark.parametrize(
    ("grad_norm", "expected"),
    [
        (2.0, False),
        (2.5, False),
        (2.6, True),
        (float("inf"), True),
        (float("nan"), True),
    ],
)
def test_evaluate_per_trajectory_mask_filters_grad_norm_exceeds_max(
    grad_norm,
    expected,
):
    assert (
        evaluate_per_trajectory_mask_filters(
            [SimpleNamespace(rule="grad_norm_exceeds_max", params={"max": 2.5})],
            _filter_context(grad_norm=grad_norm),
        )
        is expected
    )


@pytest.mark.parametrize(
    ("mean", "params", "expected"),
    [
        (0.1, {"lower": 0.0}, False),
        (-0.1, {"lower": 0.0}, True),
        (0.1, {"upper": 0.2}, False),
        (0.3, {"upper": 0.2}, True),
        (0.1, {"lower": 0.0, "upper": 0.2}, False),
        (0.3, {"lower": 0.0, "upper": 0.2}, True),
    ],
)
def test_evaluate_per_trajectory_mask_filters_kl_k1_outside_range(
    mean,
    params,
    expected,
):
    assert (
        evaluate_per_trajectory_mask_filters(
            [SimpleNamespace(rule="kl_k1_outside_range", params=params)],
            _filter_context(
                behave_approx_kl_summary=_masked_summary(mean),
                behave_approx_kl_mean_emitted=True,
            ),
        )
        is expected
    )


def test_evaluate_per_trajectory_mask_filters_kl_k1_outside_range_requires_emitted_summary():
    with pytest.raises(RuntimeError, match="behave_approx_kl_mean"):
        evaluate_per_trajectory_mask_filters(
            [SimpleNamespace(rule="kl_k1_outside_range", params={"upper": 0.2})],
            _filter_context(behave_approx_kl_summary=_masked_summary(0.1)),
        )


def test_evaluate_per_trajectory_mask_filters_kl_k1_outside_range_does_not_mask_empty_summary():
    assert not evaluate_per_trajectory_mask_filters(
        [SimpleNamespace(rule="kl_k1_outside_range", params={"upper": 0.2})],
        _filter_context(
            behave_approx_kl_summary=_masked_summary(None, count=0),
            behave_approx_kl_mean_emitted=True,
        ),
    )


@pytest.mark.parametrize(
    ("zscore", "expected"),
    [
        (1.5, False),
        (2.0, False),
        (2.1, True),
        (float("nan"), False),
    ],
)
def test_evaluate_per_trajectory_mask_filters_kl_k1_zscore_exceeds(zscore, expected):
    assert (
        evaluate_per_trajectory_mask_filters(
            [SimpleNamespace(rule="kl_k1_zscore_exceeds", params={"n": 2.0})],
            _filter_context(
                kl_k1_mean=0.2,
                kl_k1_batch_mean=0.0,
                kl_k1_batch_std=0.1,
                kl_k1_zscore=zscore,
            ),
        )
        is expected
    )


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (_masked_summary(0.1), True),
        (_masked_summary(0.0), False),
        (_masked_summary(-0.1), False),
        (_masked_summary(None, count=0), False),
    ],
)
def test_evaluate_per_trajectory_mask_filters_advantage_mean_positive(
    summary,
    expected,
):
    assert (
        evaluate_per_trajectory_mask_filters(
            [SimpleNamespace(rule="advantage_mean_positive", params={})],
            _filter_context(
                advantage_summary=summary,
                advantage_mean_emitted=True,
            ),
        )
        is expected
    )


def test_evaluate_per_trajectory_mask_filters_advantage_mean_positive_requires_emitted_summary():
    with pytest.raises(RuntimeError, match="advantage_mean"):
        evaluate_per_trajectory_mask_filters(
            [SimpleNamespace(rule="advantage_mean_positive", params={})],
            _filter_context(advantage_summary=_masked_summary(0.1)),
        )


def test_evaluate_per_trajectory_mask_filters_and_composes_rules():
    assert not evaluate_per_trajectory_mask_filters(
        [
            SimpleNamespace(rule="kl_k1_outside_range", params={"upper": 0.2}),
            SimpleNamespace(rule="advantage_mean_positive", params={}),
        ],
        _filter_context(
            behave_approx_kl_summary=_masked_summary(0.1),
            behave_approx_kl_mean_emitted=True,
            advantage_summary=_masked_summary(0.1),
            advantage_mean_emitted=True,
        ),
    )
    assert evaluate_per_trajectory_mask_filters(
        [
            SimpleNamespace(rule="kl_k1_outside_range", params={"upper": 0.2}),
            SimpleNamespace(rule="advantage_mean_positive", params={}),
        ],
        _filter_context(
            behave_approx_kl_summary=_masked_summary(0.3),
            behave_approx_kl_mean_emitted=True,
            advantage_summary=_masked_summary(0.1),
            advantage_mean_emitted=True,
        ),
    )


@pytest.mark.parametrize(
    ("kl_mean", "advantage_mean", "expected_masked"),
    [
        (0.3, 0.1, True),
        (0.3, -0.1, False),
        (0.1, 0.1, False),
        (0.1, -0.1, False),
    ],
)
def test_evaluate_per_trajectory_mask_filters_masks_only_outside_kl_with_positive_advantage(
    kl_mean,
    advantage_mean,
    expected_masked,
):
    assert (
        evaluate_per_trajectory_mask_filters(
            [
                SimpleNamespace(rule="kl_k1_outside_range", params={"upper": 0.2}),
                SimpleNamespace(rule="advantage_mean_positive", params={}),
            ],
            _filter_context(
                behave_approx_kl_summary=_masked_summary(kl_mean),
                behave_approx_kl_mean_emitted=True,
                advantage_summary=_masked_summary(advantage_mean),
                advantage_mean_emitted=True,
            ),
        )
        is expected_masked
    )


def test_evaluate_per_trajectory_mask_filters_accepts_mapping_configs():
    assert evaluate_per_trajectory_mask_filters(
        [{"rule": "grad_norm_exceeds_max", "params": {"max": 2.0}}],
        _filter_context(grad_norm=2.5),
    )


def test_per_trajectory_log_dir_uses_expected_layout(monkeypatch):
    monkeypatch.setattr(per_trajectory_module.getpass, "getuser", lambda: "alice")

    assert per_trajectory_log_dir("/tmp/areal", "exp", "trial") == Path(
        "/tmp/areal/logs/alice/exp/trial/per_trajectory/actor"
    )


def test_summarize_logprobs_uses_original_response_mask():
    logprobs = torch.tensor([[0.0, -0.2, -0.3, -9.0]], requires_grad=True)
    mask = torch.tensor([[False, True, True, False]])

    summary = summarize_logprobs(logprobs, mask)

    assert isinstance(summary.sum, float)
    assert isinstance(summary.mean, float)
    assert summary.sum == pytest.approx(-0.5)
    assert summary.mean == pytest.approx(-0.25)
    assert summary.length == 2
    assert logprobs.grad is None


def test_summarize_logprobs_rejects_empty_response():
    with pytest.raises(ValueError, match="response_length"):
        summarize_logprobs(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))


def test_masked_tensor_accumulator_summarizes_selected_values():
    accumulator = MaskedTensorAccumulator()

    accumulator.add(
        torch.tensor([[1.0, 3.0, 9.0]]),
        torch.tensor([[True, True, False]]),
    )
    accumulator.add(
        torch.tensor([[-2.0, 7.0]]),
        torch.tensor([[True, False]]),
    )

    summary = accumulator.summary()
    assert summary.count == 3
    assert summary.min == pytest.approx(-2.0)
    assert summary.max == pytest.approx(3.0)
    assert summary.mean == pytest.approx(2.0 / 3.0)


def test_masked_tensor_accumulator_empty_mask_returns_null_summary():
    accumulator = MaskedTensorAccumulator()

    accumulator.add(
        torch.tensor([[1.0, 3.0]]),
        torch.tensor([[False, False]]),
    )

    summary = accumulator.summary()
    assert summary.count == 0
    assert summary.min is None
    assert summary.max is None
    assert summary.mean is None


def test_masked_tensor_accumulator_rejects_shape_mismatch():
    accumulator = MaskedTensorAccumulator()

    with pytest.raises(ValueError, match="shape mismatch"):
        accumulator.add(torch.ones(1, 2), torch.ones(2, dtype=torch.bool))


def test_preserve_rollout_logprobs_copies_before_mutation():
    first_loss_mask = torch.tensor([[False, True]])
    second_loss_mask = torch.tensor([[True, False]])
    batch = [
        {
            "logprobs": torch.tensor([[1.0, 2.0]], requires_grad=True),
            "loss_mask": first_loss_mask,
        },
        {"logprobs": torch.tensor([[3.0, 4.0]]), "loss_mask": second_loss_mask},
    ]

    preserve_rollout_logprobs(batch)
    with torch.no_grad():
        batch[0]["logprobs"].zero_()
    batch[0]["loss_mask"].fill_(False)
    batch[1]["loss_mask"] = torch.zeros_like(second_loss_mask)

    assert batch[0]["rollout_logprobs"].requires_grad is False
    assert batch[0]["rollout_logprobs"].grad_fn is None
    assert torch.equal(batch[0]["rollout_logprobs"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(batch[1]["rollout_logprobs"], torch.tensor([[3.0, 4.0]]))
    assert batch[0]["rollout_loss_mask"].data_ptr() != first_loss_mask.data_ptr()
    assert torch.equal(batch[0]["rollout_loss_mask"], torch.tensor([[False, True]]))
    assert torch.equal(batch[1]["rollout_loss_mask"], torch.tensor([[True, False]]))


def test_preserved_rollout_logprobs_survive_actor_overwrite():
    batch = [{"logprobs": torch.tensor([[1.0, 2.0]])}]

    preserve_rollout_logprobs(batch)
    batch[0]["logprobs"] = torch.tensor([[9.0, 9.0]])

    assert torch.equal(batch[0]["rollout_logprobs"], torch.tensor([[1.0, 2.0]]))


def test_preserve_rollout_logprobs_aliases_rtensors_without_localizing(monkeypatch):
    from areal.infra.rpc.rtensor import RTensor, TensorShardInfo

    def fail_to_local(self):
        raise AssertionError("preserve_rollout_logprobs must not localize RTensors")

    monkeypatch.setattr(RTensor, "to_local", fail_to_local)
    logprobs = RTensor(
        shard=TensorShardInfo(shard_id="logprobs", node_addr="node:1"),
        data=torch.empty((1, 2), device="meta"),
    )
    loss_mask = RTensor(
        shard=TensorShardInfo(shard_id="loss_mask", node_addr="node:1"),
        data=torch.empty((1, 2), dtype=torch.bool, device="meta"),
    )
    batch = [{"logprobs": logprobs, "loss_mask": loss_mask}]

    preserve_rollout_logprobs(batch)

    assert batch[0]["rollout_logprobs"] is logprobs
    assert batch[0]["rollout_loss_mask"] is loss_mask
    assert logprobs.data.is_meta
    assert loss_mask.data.is_meta


def test_trainer_rollout_metadata_helpers_are_gated_by_per_trajectory_enabled():
    from areal.trainer import rl_trainer

    maybe_attach = getattr(
        rl_trainer,
        "_maybe_attach_per_trajectory_rollout_metadata",
        None,
    )
    assert callable(maybe_attach)

    disabled_batch = [
        {
            "logprobs": torch.tensor([[1.0, 2.0]]),
            "loss_mask": torch.tensor([[False, True]]),
        }
    ]
    disabled_config = SimpleNamespace(
        actor=SimpleNamespace(per_trajectory=SimpleNamespace(enabled=False)),
        cluster=SimpleNamespace(fileroot="/tmp/disabled"),
    )

    maybe_attach(disabled_batch, disabled_config, global_step=3)

    assert "rollout_logprobs" not in disabled_batch[0]
    assert "rollout_loss_mask" not in disabled_batch[0]
    assert "trainer_global_step" not in disabled_batch[0]
    assert "trainer_fileroot" not in disabled_batch[0]

    enabled_batch = [
        {
            "logprobs": torch.tensor([[1.0, 2.0]]),
            "loss_mask": torch.tensor([[False, True]]),
        }
    ]
    enabled_config = SimpleNamespace(
        actor=SimpleNamespace(per_trajectory=SimpleNamespace(enabled=True)),
        cluster=SimpleNamespace(fileroot="/tmp/enabled"),
    )

    maybe_attach(enabled_batch, enabled_config, global_step=4)

    assert torch.equal(enabled_batch[0]["rollout_logprobs"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(
        enabled_batch[0]["rollout_loss_mask"],
        torch.tensor([[False, True]]),
    )
    assert torch.equal(enabled_batch[0]["trainer_global_step"], torch.tensor([4]))
    assert enabled_batch[0]["trainer_fileroot"] == "/tmp/enabled"


def test_attach_trainer_step_metadata_uses_tensor_for_batching():
    batch = [{"input_ids": torch.ones(1, 2)}, {"input_ids": torch.ones(1, 3)}]

    attach_trainer_step_metadata(batch, global_step=11, fileroot="/tmp/areal")

    assert torch.equal(batch[0]["trainer_global_step"], torch.tensor([11]))
    assert torch.equal(batch[1]["trainer_global_step"], torch.tensor([11]))
    assert batch[0]["trainer_global_step"].dtype == torch.long
    assert batch[1]["trainer_global_step"].dtype == torch.long
    assert batch[0]["trainer_fileroot"] == "/tmp/areal"
    assert batch[1]["trainer_fileroot"] == "/tmp/areal"


def test_trajectory_id_prefers_known_keys():
    assert (
        trajectory_id_from_sample({"traj_uid": ["traj-7"]}, fallback="fallback")
        == "traj-7"
    )
    assert trajectory_id_from_sample({"uid": ["uid-1"]}, fallback="fallback") == "uid-1"
    assert (
        trajectory_id_from_sample({"rid": torch.tensor([9])}, fallback="fallback")
        == "9"
    )
    assert (
        trajectory_id_from_sample({"task_id": ["task-1"]}, fallback="fallback")
        == "task-1"
    )
    assert trajectory_id_from_sample({}, fallback="fallback") == "fallback"


def test_trajectory_id_uses_priority_and_first_scalar():
    assert (
        trajectory_id_from_sample(
            {
                "task_id": ["task-1"],
                "rid": torch.tensor([7, 8]),
                "uid": ["uid-1", "uid-2"],
                "traj_uid": ["traj-1", "traj-2"],
            },
            fallback="fallback",
        )
        == "traj-1"
    )
    assert (
        trajectory_id_from_sample({"rid": torch.tensor([7, 8])}, fallback="fallback")
        == "7"
    )
    assert (
        trajectory_id_from_sample({"uid": ["uid-1", "uid-2"]}, fallback="fallback")
        == "uid-1"
    )


def test_concat_batch_preserves_scalar_trace_metadata_as_per_sample_lists():
    from areal.utils.data import concat_batch

    trajs = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
            "ids": "id-0",
            "uid": "uid-0",
            "traj_uid": "traj-0",
            "rid": 100,
            "task_id": "task-0",
            "shared_scalar": "keep-first",
        },
        {
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "ids": "id-1",
            "uid": "uid-1",
            "traj_uid": "traj-1",
            "rid": 101,
            "task_id": "task-1",
            "shared_scalar": "drop-second",
        },
    ]

    batched, _ = concat_batch(trajs)

    assert batched["ids"] == ["id-0", "id-1"]
    assert batched["uid"] == ["uid-0", "uid-1"]
    assert batched["traj_uid"] == ["traj-0", "traj-1"]
    assert batched["rid"] == [100, 101]
    assert batched["task_id"] == ["task-0", "task-1"]
    assert batched["shared_scalar"] == "keep-first"


def test_split_batch_slices_scalar_trace_metadata_without_reconcat_duplication():
    from areal.utils.data import concat_batch, split_batch

    trajs = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
            "ids": "id-0",
            "uid": "uid-0",
            "traj_uid": "traj-0",
            "rid": 100,
            "task_id": "task-0",
            "debug_tags": ["tag-0"],
            "shared_scalar": "keep-first",
        },
        {
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "ids": "id-1",
            "uid": "uid-1",
            "traj_uid": "traj-1",
            "rid": 101,
            "task_id": "task-1",
            "debug_tags": ["tag-1"],
            "shared_scalar": "drop-second",
        },
    ]

    batched, meta = concat_batch(trajs)
    recovered = split_batch(batched, meta)

    assert [traj["ids"] for traj in recovered] == [["id-0"], ["id-1"]]
    assert [traj["uid"] for traj in recovered] == [["uid-0"], ["uid-1"]]
    assert [traj["traj_uid"] for traj in recovered] == [["traj-0"], ["traj-1"]]
    assert [traj["rid"] for traj in recovered] == [[100], [101]]
    assert [traj["task_id"] for traj in recovered] == [["task-0"], ["task-1"]]
    assert recovered[0]["debug_tags"] == ["tag-0", "tag-1"]
    assert recovered[1]["debug_tags"] == ["tag-0", "tag-1"]
    assert recovered[0]["debug_tags"] is not recovered[1]["debug_tags"]
    assert recovered[0]["shared_scalar"] == "keep-first"
    assert recovered[1]["shared_scalar"] == "keep-first"

    repacked, _ = concat_batch(recovered)

    assert repacked["ids"] == ["id-0", "id-1"]
    assert repacked["uid"] == ["uid-0", "uid-1"]
    assert repacked["traj_uid"] == ["traj-0", "traj-1"]
    assert repacked["rid"] == [100, 101]
    assert repacked["task_id"] == ["task-0", "task-1"]


def test_slice_trajectory_slices_tensors_and_lists():
    non_batch_tensor = torch.tensor([[5, 6], [7, 8], [9, 10]])
    non_batch_vector = torch.tensor([10, 20])
    non_batch_list = ["x", "y", "z"]
    global_tags = ["keep-a", "keep-b"]
    custom_stat = torch.tensor([1.0, 2.0])
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1], [0, 0]], dtype=torch.bool),
        "ids": ["a", "b"],
        "uid": ["uid-a", "uid-b"],
        "rid": torch.tensor([101, 102]),
        "begin_of_trajectory": torch.tensor([True, False]),
        "rewards": torch.tensor([1.0, 2.0]),
        "task_reward": torch.tensor([1.0, 0.0]),
        "trainer_global_step": torch.tensor([5, 5]),
        "multi_modal_input_images": ["image-a", "image-b"],
        "multi_modal_input_features": [{"feature": "a"}, {"feature": "b"}],
        "custom_stat": custom_stat,
        "non_batch_tensor": non_batch_tensor,
        "non_batch_vector": non_batch_vector,
        "non_batch_list": non_batch_list,
        "global_tags": global_tags,
        "scalar": 3,
    }

    sliced = slice_trajectory(batch, 1)

    assert torch.equal(sliced["input_ids"], torch.tensor([[3, 4]]))
    assert torch.equal(
        sliced["attention_mask"], torch.tensor([[1, 0]], dtype=torch.bool)
    )
    assert torch.equal(sliced["loss_mask"], torch.tensor([[0, 0]], dtype=torch.bool))
    assert sliced["ids"] == ["b"]
    assert sliced["uid"] == ["uid-b"]
    assert torch.equal(sliced["rid"], torch.tensor([102]))
    assert torch.equal(sliced["begin_of_trajectory"], torch.tensor([False]))
    assert torch.equal(sliced["rewards"], torch.tensor([2.0]))
    assert torch.equal(sliced["task_reward"], torch.tensor([0.0]))
    assert torch.equal(sliced["trainer_global_step"], torch.tensor([5]))
    assert sliced["multi_modal_input_images"] == ["image-b"]
    assert sliced["multi_modal_input_features"] == [{"feature": "b"}]
    assert torch.equal(sliced["custom_stat"], custom_stat)
    assert torch.equal(sliced["non_batch_tensor"], non_batch_tensor)
    assert torch.equal(sliced["non_batch_vector"], non_batch_vector)
    assert sliced["non_batch_list"] == non_batch_list
    assert sliced["global_tags"] == global_tags
    assert sliced["scalar"] == 3


def test_slice_trajectory_slices_extra_batch_keys():
    batch = {
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "custom_stat": torch.tensor([1.0, 2.0]),
        "custom_labels": ["a", "b"],
        "global_labels": ["keep-a", "keep-b"],
        "unlisted_stat": torch.tensor([3.0, 4.0]),
    }

    sliced = slice_trajectory(
        batch, 1, extra_batch_keys={"custom_stat", "custom_labels"}
    )

    assert torch.equal(sliced["custom_stat"], torch.tensor([2.0]))
    assert sliced["custom_labels"] == ["b"]
    assert sliced["global_labels"] == ["keep-a", "keep-b"]
    assert torch.equal(sliced["unlisted_stat"], torch.tensor([3.0, 4.0]))


def test_slice_trajectory_rejects_malformed_attention_mask():
    with pytest.raises(KeyError, match="attention_mask"):
        slice_trajectory({"input_ids": torch.ones(2, 3)}, 0)

    with pytest.raises(ValueError, match="attention_mask.*tensor"):
        slice_trajectory({"attention_mask": [[1, 1], [1, 0]]}, 0)

    with pytest.raises(ValueError, match="attention_mask.*2D"):
        slice_trajectory({"attention_mask": torch.ones(2, dtype=torch.bool)}, 0)

    with pytest.raises(ValueError, match="positive"):
        slice_trajectory({"attention_mask": torch.empty(0, 3, dtype=torch.bool)}, 0)


def test_tracer_flushes_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = PerTrajectoryTracer(path=path, flush_threshold=2, enabled=True)

    tracer.write(
        _record(
            trajectory_id="a",
            trajectory_idx=0,
            grad_norm=1.5,
        )
    )
    assert not path.exists()

    tracer.write(
        _record(
            trajectory_id="b",
            trajectory_idx=1,
            grad_norm=2.5,
            logprob_train_sum=-3.0,
            logprob_train_mean=-1.5,
            logprob_infer_sum=-4.0,
            logprob_infer_mean=-2.0,
            reward=0.0,
            response_length=2,
        )
    )
    tracer.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["trajectory_id"] for row in rows] == ["a", "b"]
    assert rows[0]["grad_norm"] == 1.5


def test_tracer_disabled_mode_is_noop(tmp_path):
    path = tmp_path / "nested" / "trace.jsonl"
    tracer = PerTrajectoryTracer(path=path, flush_threshold=1, enabled=False)

    tracer.write(_record())
    tracer.flush()
    tracer.close()

    assert not path.exists()
    assert not path.parent.exists()


def test_tracer_explicit_flush_creates_parent_and_writes_sorted_jsonl(tmp_path):
    path = tmp_path / "nested" / "trace.jsonl"
    record = _record(trajectory_id="traj-é", grad_norm=3.25)
    tracer = PerTrajectoryTracer(path=path, flush_threshold=10, enabled=True)

    tracer.write(record)
    assert not path.exists()

    tracer.flush()

    assert path.parent.exists()
    expected = json.dumps(
        asdict(record),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    assert path.read_text(encoding="utf-8") == (expected + "\n")
    assert ": " not in expected
    assert ", " not in expected
    assert expected.startswith('{"accepted":')


def test_tracer_serializes_non_finite_values_as_null(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = PerTrajectoryTracer(path=path, flush_threshold=10, enabled=True)
    tracer.write(
        _record(
            accepted=False,
            grad_norm=float("nan"),
            entropy_mean=float("inf"),
            advantage_min=float("-inf"),
        )
    )

    tracer.flush()

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["accepted"] is False
    assert row["grad_norm"] is None
    assert row["entropy_mean"] is None
    assert row["advantage_min"] is None
    assert "grad_norm_filtered" not in row


def test_tracer_serializes_mixed_finite_and_non_finite_rows(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = PerTrajectoryTracer(path=path, flush_threshold=10, enabled=True)
    tracer.write(_record(trajectory_id="valid"))
    tracer.write(_record(trajectory_id="bad", grad_norm=float("nan")))

    tracer.flush()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["trajectory_id"] for row in rows] == ["valid", "bad"]
    assert rows[0]["grad_norm"] == 1.5
    assert rows[1]["grad_norm"] is None


def test_tracer_close_flushes_partial_buffer(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = PerTrajectoryTracer(path=path, flush_threshold=10, enabled=True)

    tracer.write(_record(trajectory_id="partial"))
    assert not path.exists()

    tracer.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["trajectory_id"] for row in rows] == ["partial"]
