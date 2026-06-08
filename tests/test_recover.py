"""Tests for the recovery configuration and functionality."""

import os
import tempfile
from unittest.mock import Mock

import pytest

from areal.api.cli_args import RecoverConfig
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.utils.recover import (
    RecoverHandler,
    RecoverInfo,
    check_if_auto_recover,
    check_if_recover,
)


class TestRecoverConfig:
    """Tests for RecoverConfig dataclass validation."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
        )
        assert config.mode == "disabled"
        assert config.retries == 3

    def test_history_default_values(self):
        """Test recovery history defaults preserve legacy behavior."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
        )
        assert config.keep_last == 1
        assert config.load_step is None

    @pytest.mark.parametrize("keep_last", [0, -1, -5, True])
    def test_invalid_keep_last(self, keep_last):
        """Test invalid history retention values are rejected."""
        with pytest.raises(ValueError) as exc_info:
            RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot="/tmp",
                keep_last=keep_last,
            )
        assert "recover.keep_last" in str(exc_info.value)
        assert "positive integer" in str(exc_info.value)

    @pytest.mark.parametrize("load_step", [-1, -10, True])
    def test_invalid_load_step(self, load_step):
        """Test invalid exact history load steps are rejected."""
        with pytest.raises(ValueError) as exc_info:
            RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot="/tmp",
                load_step=load_step,
            )
        assert "recover.load_step" in str(exc_info.value)
        assert "non-negative integer" in str(exc_info.value)

    @pytest.mark.parametrize("load_step", [0, 1, 1010, None])
    def test_valid_load_step(self, load_step):
        """Test exact history load step accepts non-negative integers and None."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
            load_step=load_step,
        )
        assert config.load_step == load_step

    @pytest.mark.parametrize("mode", ["on", "off", "auto", "disabled"])
    def test_valid_modes(self, mode):
        """Test that all valid modes are accepted."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
            mode=mode,
        )
        assert config.mode == mode

    @pytest.mark.parametrize("mode", ["fault", "resume", "invalid", "ON", "OFF", ""])
    def test_invalid_modes(self, mode):
        """Test that invalid modes raise ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot="/tmp",
                mode=mode,
            )
        error_msg = str(exc_info.value)
        assert f"Invalid recover mode '{mode}'" in error_msg
        assert "fault" in error_msg and "resume" in error_msg  # Migration hint


class TestCheckIfRecover:
    """Tests for the check_if_recover function."""

    @pytest.mark.parametrize("mode", ["disabled", "off"])
    def test_disabled_modes_return_false(self, mode):
        """Test that disabled modes always return False."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
            mode=mode,
        )
        # Should return False regardless of run_id
        assert check_if_recover(config, 0) is False
        assert check_if_recover(config, 1) is False
        assert check_if_recover(config, 10) is False

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_enabled_modes_check_for_checkpoint(self, mode):
        """Test that enabled modes check for existing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode=mode,
            )
            # No checkpoint exists, should return False
            assert check_if_recover(config, 0) is False

    @pytest.mark.parametrize("run_id", [0, 1, 5, 100])
    def test_run_id_parameter_unused(self, run_id):
        """Test that run_id parameter doesn't affect the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with disabled mode
            config_disabled = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="disabled",
            )
            assert check_if_recover(config_disabled, run_id) is False

            # Test with enabled mode (no checkpoint)
            config_enabled = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            # Result should be the same regardless of run_id
            result = check_if_recover(config_enabled, run_id)
            assert result == check_if_recover(config_enabled, 0)


class TestCheckIfAutoRecover:
    """Tests for the check_if_auto_recover function."""

    @staticmethod
    def _make_config(
        tmpdir: str,
        keep_last: int = 1,
        load_step: int | None = None,
    ) -> RecoverConfig:
        return RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot=tmpdir,
            mode="on",
            keep_last=keep_last,
            load_step=load_step,
        )

    @staticmethod
    def _write_valid_recovery_root(root: str, step: int) -> None:
        os.makedirs(os.path.join(root, "default", "recover_checkpoint"))
        TestRecoverHandler._write_recover_info(root, step)

    def test_no_checkpoint_returns_false(self):
        """Test that missing checkpoint returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            assert check_if_auto_recover(config) is False

    def test_empty_directory_returns_false(self):
        """Test that empty directory (no checkpoint) returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            assert check_if_auto_recover(config) is False

    def test_history_checkpoint_returns_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5)
            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            history_root = RecoverHandler.history_step_path(trial_root, 10)
            self._write_valid_recovery_root(history_root, 10)
            with open(os.path.join(history_root, ".complete"), "w") as f:
                f.write("10\n")

            assert check_if_auto_recover(config) is True

    def test_exact_missing_history_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5, load_step=10)

            with pytest.raises(ValueError) as exc_info:
                check_if_auto_recover(config)

            assert "Requested recovery step 10" in str(exc_info.value)

    def test_history_check_does_not_create_checkpoint_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5)
            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            TestRecoverHandler._write_recover_info(trial_root, 10)

            assert check_if_auto_recover(config) is False
            assert not os.path.exists(
                os.path.join(trial_root, "recover_info", "recover_checkpoint")
            )


class TestModeEquivalence:
    """Tests to verify mode equivalences (on=auto, off=disabled)."""

    def test_on_equals_auto(self):
        """Test that 'on' and 'auto' modes behave identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_on = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            config_auto = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="auto",
            )
            # Both should return the same result
            assert check_if_recover(config_on, 0) == check_if_recover(config_auto, 0)

    def test_off_equals_disabled(self):
        """Test that 'off' and 'disabled' modes behave identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_off = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="off",
            )
            config_disabled = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="disabled",
            )
            # Both should return False
            assert check_if_recover(config_off, 0) is False
            assert check_if_recover(config_disabled, 0) is False


class TestRecoverHandler:
    @staticmethod
    def _make_config(
        tmpdir: str,
        mode: str = "on",
        keep_last: int = 1,
        load_step: int | None = None,
    ) -> RecoverConfig:
        return RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot=tmpdir,
            mode=mode,
            keep_last=keep_last,
            load_step=load_step,
        )

    @staticmethod
    def _make_step_info(step: int) -> StepInfo:
        return StepInfo(
            epoch=0,
            epoch_step=step,
            global_step=step,
            steps_per_epoch=100,
        )

    @staticmethod
    def _write_recover_info(recovery_root: str, step: int) -> None:
        info = RecoverInfo(
            last_step_info=TestRecoverHandler._make_step_info(step),
            saver_info={},
            evaluator_info={},
            stats_logger_info={},
            dataloader_info={},
            checkpoint_info={
                "epoch": {
                    "frequence_seconds": None,
                    "frequency_steps": None,
                    "start_time": 0.0,
                    "steps": 0,
                    "last_time": 0.0,
                    "last_steps": 0,
                    "interval_steps": None,
                    "interval_seconds": None,
                    "initial_value": False,
                },
                "step": {
                    "frequence_seconds": None,
                    "frequency_steps": None,
                    "start_time": 0.0,
                    "steps": 0,
                    "last_time": 0.0,
                    "last_steps": 0,
                    "interval_steps": None,
                    "interval_seconds": None,
                    "initial_value": False,
                },
                "time": {
                    "frequence_seconds": None,
                    "frequency_steps": None,
                    "start_time": 0.0,
                    "steps": 0,
                    "last_time": 0.0,
                    "last_steps": 0,
                    "interval_steps": None,
                    "interval_seconds": None,
                    "initial_value": False,
                },
            },
        )
        info.dump(os.path.join(recovery_root, "recover_info"))

    @staticmethod
    def _mock_stateful_component() -> Mock:
        component = Mock()
        component.state_dict.return_value = {}
        component.load_state_dict.return_value = None
        return component

    @staticmethod
    def _write_history_dir(tmpdir: str, step: int, complete: bool = True) -> str:
        config = TestRecoverHandler._make_config(tmpdir)
        trial_root = RecoverHandler.trial_root_path(
            config.experiment_name,
            config.trial_name,
            config.fileroot,
        )
        history_dir = RecoverHandler.history_step_path(trial_root, step)
        os.makedirs(os.path.join(history_dir, "default", "recover_checkpoint"))
        if complete:
            with open(os.path.join(history_dir, ".complete"), "w") as f:
                f.write(f"{step}\n")
        return history_dir

    @staticmethod
    def _make_handler(tmpdir: str, mode: str) -> RecoverHandler:
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot=tmpdir,
            mode=mode,
        )
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=8,
            train_batch_size=2,
        )
        return RecoverHandler(config, ft_spec)

    @staticmethod
    def _make_gateway_controller() -> GatewayTrainController:
        return GatewayTrainController.__new__(GatewayTrainController)

    def test_dump_keep_last_one_uses_legacy_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=1)
            config.freq_steps = 1
            handler = RecoverHandler(
                config,
                FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2),
            )
            engine = Mock()

            handler.dump(
                engine,
                self._make_step_info(10),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
            )

            saved_meta = engine.save.call_args.args[0]
            assert saved_meta.path.endswith("test_trial/default/recover_checkpoint")
            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            assert os.path.exists(os.path.join(trial_root, "recover_info"))
            assert not os.path.exists(os.path.join(trial_root, "recover_history"))

    def test_dump_history_writes_complete_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5)
            config.freq_steps = 1
            handler = RecoverHandler(
                config,
                FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2),
            )
            engine = Mock()

            handler.dump(
                engine,
                self._make_step_info(10),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
            )

            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            history_dir = RecoverHandler.history_step_path(trial_root, 10)
            saved_meta = engine.save.call_args.args[0]
            assert saved_meta.path == os.path.join(
                history_dir, "default", "recover_checkpoint"
            )
            assert os.path.exists(os.path.join(history_dir, "recover_info"))
            assert os.path.exists(os.path.join(history_dir, ".complete"))

    def test_dump_history_replacing_step_removes_complete_before_save(
        self, monkeypatch
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5)
            config.freq_steps = 1
            handler = RecoverHandler(
                config,
                FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2),
            )
            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            history_dir = RecoverHandler.history_step_path(trial_root, 10)
            os.makedirs(history_dir)
            marker_path = RecoverHandler.complete_marker_path(history_dir)
            with open(marker_path, "w") as f:
                f.write("10\n")

            barrier = Mock()
            monkeypatch.setattr("areal.utils.recover.dist.is_initialized", lambda: True)
            monkeypatch.setattr("areal.utils.recover.dist.get_rank", lambda: 0)
            monkeypatch.setattr("areal.utils.recover.dist.barrier", barrier)

            engine = Mock()

            def fail_save(_meta):
                assert not os.path.exists(marker_path)
                barrier.assert_called_once_with()
                raise RuntimeError("save failed")

            engine.save.side_effect = fail_save

            with pytest.raises(RuntimeError, match="save failed"):
                handler.dump(
                    engine,
                    self._make_step_info(10),
                    self._mock_stateful_component(),
                    self._mock_stateful_component(),
                    self._mock_stateful_component(),
                    self._mock_stateful_component(),
                )

            assert not os.path.exists(marker_path)

    def test_dump_history_prunes_older_complete_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=2)
            config.freq_steps = 1
            handler = RecoverHandler(
                config,
                FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2),
            )

            for step in [10, 20, 30]:
                handler.dump(
                    Mock(),
                    self._make_step_info(step),
                    self._mock_stateful_component(),
                    self._mock_stateful_component(),
                    self._mock_stateful_component(),
                    self._mock_stateful_component(),
                )

            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            assert RecoverHandler.complete_history_steps(trial_root) == [
                (20, RecoverHandler.history_step_path(trial_root, 20)),
                (30, RecoverHandler.history_step_path(trial_root, 30)),
            ]
            assert not os.path.exists(RecoverHandler.history_step_path(trial_root, 10))

    def test_history_step_path_is_zero_padded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            assert RecoverHandler.history_step_path(trial_root, 1010).endswith(
                "recover_history/globalstep_00001010"
            )

    def test_complete_history_steps_ignore_incomplete_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            complete_dir = self._write_history_dir(tmpdir, 10, complete=True)
            self._write_history_dir(tmpdir, 20, complete=False)

            steps = RecoverHandler.complete_history_steps(trial_root)

            assert steps == [(10, complete_dir)]

    def test_complete_history_steps_ignore_noncanonical_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            history_root = RecoverHandler.history_root_path(trial_root)
            noncanonical_dir = os.path.join(history_root, "globalstep_10")
            os.makedirs(noncanonical_dir)
            with open(os.path.join(noncanonical_dir, ".complete"), "w") as f:
                f.write("10\n")
            complete_dir = self._write_history_dir(tmpdir, 20, complete=True)

            steps = RecoverHandler.complete_history_steps(trial_root)

            assert steps == [(20, complete_dir)]

    def test_select_recovery_root_without_history_falls_back_without_mutating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )

            selected = RecoverHandler.select_recovery_root(config)

            assert selected.root == trial_root
            assert selected.is_history is False
            assert selected.step is None
            assert not os.path.exists(trial_root)

    def test_select_latest_complete_history_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5)
            self._write_history_dir(tmpdir, 10, complete=True)
            expected = self._write_history_dir(tmpdir, 20, complete=True)
            self._write_history_dir(tmpdir, 30, complete=False)

            selected = RecoverHandler.select_recovery_root(config)

            assert selected.root == expected
            assert selected.is_history is True
            assert selected.step == 20

    def test_select_exact_complete_history_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5, load_step=10)
            expected = self._write_history_dir(tmpdir, 10, complete=True)
            self._write_history_dir(tmpdir, 20, complete=True)

            selected = RecoverHandler.select_recovery_root(config)

            assert selected.root == expected
            assert selected.is_history is True
            assert selected.step == 10

    def test_select_exact_missing_history_root_fails_with_available_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5, load_step=30)
            self._write_history_dir(tmpdir, 10, complete=True)
            self._write_history_dir(tmpdir, 20, complete=True)

            with pytest.raises(ValueError) as exc_info:
                RecoverHandler.select_recovery_root(config)

            msg = str(exc_info.value)
            assert "Requested recovery step 30" in msg
            assert "Available complete recovery steps: [10, 20]" in msg

    def test_select_exact_incomplete_history_root_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5, load_step=30)
            self._write_history_dir(tmpdir, 10, complete=True)
            self._write_history_dir(tmpdir, 30, complete=False)

            with pytest.raises(ValueError) as exc_info:
                RecoverHandler.select_recovery_root(config)

            msg = str(exc_info.value)
            assert "Requested recovery step 30 exists but is incomplete" in msg
            assert "Available complete recovery steps: [10]" in msg

    def test_load_uses_latest_complete_history_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5)
            handler = RecoverHandler(
                config,
                FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2),
            )
            old_root = self._write_history_dir(tmpdir, 10, complete=True)
            new_root = self._write_history_dir(tmpdir, 20, complete=True)
            self._write_recover_info(old_root, 10)
            self._write_recover_info(new_root, 20)
            engine = Mock()

            recover_info = handler.load(
                engine,
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
            )

            assert recover_info.last_step_info.global_step == 20
            loaded_meta = engine.load.call_args.args[0]
            assert loaded_meta.path == os.path.join(
                new_root, "default", "recover_checkpoint"
            )

    def test_load_uses_exact_complete_history_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5, load_step=10)
            handler = RecoverHandler(
                config,
                FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2),
            )
            old_root = self._write_history_dir(tmpdir, 10, complete=True)
            new_root = self._write_history_dir(tmpdir, 20, complete=True)
            self._write_recover_info(old_root, 10)
            self._write_recover_info(new_root, 20)
            engine = Mock()

            recover_info = handler.load(
                engine,
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
            )

            assert recover_info.last_step_info.global_step == 10
            loaded_meta = engine.load.call_args.args[0]
            assert loaded_meta.path == os.path.join(
                old_root, "default", "recover_checkpoint"
            )

    def test_load_with_exact_missing_step_fails_instead_of_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5, load_step=10)
            handler = RecoverHandler(
                config,
                FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2),
            )

            with pytest.raises(ValueError) as exc_info:
                handler.load(
                    Mock(),
                    self._mock_stateful_component(),
                    self._mock_stateful_component(),
                    self._mock_stateful_component(),
                    self._mock_stateful_component(),
                )

            assert "Requested recovery step 10" in str(exc_info.value)

    def test_load_falls_back_to_legacy_when_history_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, keep_last=5)
            handler = RecoverHandler(
                config,
                FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2),
            )
            trial_root = RecoverHandler.trial_root_path(
                config.experiment_name,
                config.trial_name,
                config.fileroot,
            )
            os.makedirs(os.path.join(trial_root, "default", "recover_checkpoint"))
            self._write_recover_info(trial_root, 7)
            engine = Mock()

            recover_info = handler.load(
                engine,
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
                self._mock_stateful_component(),
            )

            assert recover_info.last_step_info.global_step == 7
            loaded_meta = engine.load.call_args.args[0]
            assert loaded_meta.path == os.path.join(
                trial_root, "default", "recover_checkpoint"
            )

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_load_rejects_gateway_train_controller(self, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, mode)

            with pytest.raises(NotImplementedError) as exc_info:
                handler.load(
                    self._make_gateway_controller(),
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                )

            assert "GatewayTrainController" in str(exc_info.value)
            assert '`_version="v2"`' in str(exc_info.value)

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_dump_rejects_gateway_train_controller(self, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, mode)
            step_info = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )

            with pytest.raises(NotImplementedError) as exc_info:
                handler.dump(
                    self._make_gateway_controller(),
                    step_info,
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                )

            assert "GatewayTrainController" in str(exc_info.value)
            assert "recover.mode" in str(exc_info.value)
