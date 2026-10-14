# Copyright © 2024 Apple Inc.

"""Tests measurement utils for GCP."""
# pylint: disable=protected-access

import contextlib
from unittest import mock

from absl import flags, logging
from absl.testing import absltest, parameterized

from axlearn.cloud.gcp import measurement as measurement_gcp
from axlearn.cloud.gcp.measurement import GoodputRecorder
from axlearn.common import elastic_utils, measurement
from axlearn.common.config import RequiredFieldMissingError


class GoodputRecorderTest(parameterized.TestCase):
    """Tests GoodputRecorder."""

    @parameterized.parameters(
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
            ],
            expected_rolling_window_size=[],
            expected_jax_backend=None,
        ),
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
                "rolling_window_size=1,2,3",
                "jax_backend=proxy",
            ],
            expected_rolling_window_size=[1, 2, 3],
            expected_jax_backend="proxy",
        ),
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
                "enable_monitoring=true",
            ],
            expected_rolling_window_size=[],
            expected_jax_backend=None,
            expected_enable_monitoring=True,
        ),
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
                "enable_monitoring=false",
            ],
            expected_rolling_window_size=[],
            expected_jax_backend=None,
            expected_enable_monitoring=False,
        ),
        dict(
            recorder_spec=[
                "name=test-name",
                "upload_dir=/test/path",
                "upload_interval=15",
                "enable_checkpoint_logging=false",
            ],
            expected_rolling_window_size=[],
            expected_jax_backend=None,
            expected_enable_checkpoint_logging=False,
        ),
    )
    def test_from_flags(
        self,
        recorder_spec,
        expected_rolling_window_size,
        expected_jax_backend,
        expected_enable_monitoring=True,
        expected_enable_checkpoint_logging=True,
    ):
        """Tests that flags are correctly parsed into the config."""
        mock_fv = mock.MagicMock(spec=flags.FlagValues)
        mock_fv.recorder_spec = recorder_spec
        mock_fv.jax_backend = "tpu"

        recorder = GoodputRecorder.from_flags(mock_fv)

        self.assertEqual("test-name", recorder.config.name)
        self.assertEqual("/test/path", recorder.config.upload_dir)
        self.assertEqual(15, recorder.config.upload_interval)
        self.assertEqual(expected_rolling_window_size, recorder.config.rolling_window_size)
        self.assertEqual(expected_jax_backend, recorder.config.jax_backend)
        self.assertEqual(expected_enable_monitoring, recorder.config.enable_monitoring)
        self.assertEqual(
            expected_enable_checkpoint_logging, recorder.config.enable_checkpoint_logging
        )

    def test_from_flags_missing_required(self):
        """Tests that missing required flags raise an error."""
        mock_fv = mock.MagicMock(spec=flags.FlagValues)
        mock_fv.recorder_spec = ["name=test-name"]  # Missing upload_dir/interval
        mock_fv.jax_backend = "tpu"
        with self.assertRaisesRegex(RequiredFieldMissingError, "upload_dir"):
            GoodputRecorder.from_flags(mock_fv)

    @parameterized.parameters(
        dict(
            event=measurement.EventType.JOB,
            expected_start="record_job_start_time",
            expected_end="record_job_end_time",
            args=(),
            kwargs={},
            expect_end_call=True,
        ),
        dict(
            event=measurement.EventType.STEP,
            expected_start="record_step_start_time",
            expected_end=None,
            args=(123,),
            kwargs={},
            expect_end_call=False,
        ),
        dict(
            event=measurement.EventType.ACCELERATOR_INIT,
            expected_start="record_tpu_init_start_time",
            expected_end="record_tpu_init_end_time",
            args=(),
            kwargs={},
            expect_end_call=True,
        ),
        dict(
            event=measurement.EventType.TRAINING_PREPARATION,
            expected_start="record_training_preparation_start_time",
            expected_end="record_training_preparation_end_time",
            args=(),
            kwargs={},
            expect_end_call=True,
        ),
        dict(
            event=measurement.EventType.DATA_LOADING,
            expected_start="record_data_loading_start_time",
            expected_end="record_data_loading_end_time",
            args=(),
            kwargs={},
            expect_end_call=True,
        ),
        dict(
            event=measurement.EventType.CUSTOM_BADPUT_EVENT,
            expected_start="record_custom_badput_event_start_time",
            expected_end="record_custom_badput_event_end_time",
            args=(),
            kwargs={"custom_badput_event_type": "TEST_TYPE"},
            expect_end_call=True,
        ),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_record_event_context_manager_success(
        self, _, event, expected_start, expected_end, args, kwargs, expect_end_call
    ):
        """Tests that record_event calls correct start and end methods with args and kwargs."""
        cfg = GoodputRecorder.default_config().set(
            name="test",
            upload_dir="/tmp/test",
            upload_interval=1,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.goodput.GoodputRecorder") as mock_recorder_cls:
            mock_instance = mock_recorder_cls.return_value

            start_mock = mock.MagicMock()
            setattr(mock_instance, expected_start, start_mock)
            if expect_end_call and expected_end:
                end_mock = mock.MagicMock()
                setattr(mock_instance, expected_end, end_mock)

            with recorder.record_event(event, *args, **kwargs):
                pass

            mock_recorder_cls.assert_called_once()
            start_mock.assert_called_once_with(*args, **kwargs)
            if expect_end_call and expected_end:
                end_mock.assert_called_once_with(*args, **kwargs)

    def test_record_event_context_manager_handles_runtime_error(self):
        cfg = GoodputRecorder.default_config().set(
            name="test",
            upload_dir="/tmp/test",
            upload_interval=1,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("jax.process_index", return_value=0):
            with (
                mock.patch("ml_goodput_measurement.goodput.GoodputRecorder") as mock_recorder_cls,
                mock.patch.object(logging, "warning") as mock_warning,
            ):
                mock_instance = mock_recorder_cls.return_value

                def raise_runtime_error(*args, **kwargs):
                    raise RuntimeError("mocked error")

                mock_instance.record_job_start_time.side_effect = raise_runtime_error
                mock_instance.record_job_end_time.side_effect = raise_runtime_error
                # Should not crash here.
                with recorder.record_event(measurement.EventType.JOB):
                    pass

                # Assert warnings were logged for start and end failures
                assert mock_warning.call_count == 2
                start_call = mock_warning.call_args_list[0]
                end_call = mock_warning.call_args_list[1]

                assert "Failed to record" in start_call.args[0]
                assert "Failed to record" in end_call.args[0]

    @parameterized.parameters(
        dict(is_pathways_job=False, mock_jax_backend="tpu"),
        dict(is_pathways_job=True, mock_jax_backend="proxy"),
        dict(is_pathways_job=False, mock_jax_backend=None),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_goodput(self, _, is_pathways_job, mock_jax_backend):
        """Tests the _maybe_monitor_goodput context manager."""
        cfg = GoodputRecorder.default_config().set(
            name="test-monitor",
            upload_dir="/test",
            upload_interval=30,
            jax_backend=mock_jax_backend,
            enable_monitoring=True,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value
            with recorder._maybe_monitor_goodput():
                pass

            # Verify that GoodputMonitor was instantiated with the correct parameters.
            mock_monitor_cls.assert_called_once_with(
                job_name="test-monitor",
                logger_name="goodput_logger_test-monitor",
                tensorboard_dir="/test",
                upload_interval=30,
                monitoring_enabled=True,
                pathway_enabled=is_pathways_job,
                include_badput_breakdown=True,
            )
            mock_monitor_instance.start_goodput_uploader.assert_called_once()
            mock_monitor_instance.stop_goodput_uploader.assert_called_once()

    @parameterized.parameters(
        dict(
            is_rolling_window_enabled=True,
            rolling_window_size=[10, 20],
            is_pathways_job=False,
            mock_jax_backend="tpu",
        ),
        dict(
            is_rolling_window_enabled=False,
            rolling_window_size=[],
            is_pathways_job=False,
            mock_jax_backend="tpu",
        ),
        dict(
            is_rolling_window_enabled=True,
            rolling_window_size=[50],
            is_pathways_job=True,
            mock_jax_backend="proxy",
        ),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_rolling_window(
        self,
        mock_process_index,
        is_rolling_window_enabled,
        rolling_window_size,
        is_pathways_job,
        mock_jax_backend,
    ):  # pylint: disable=unused-argument
        """Tests the rolling window monitoring."""
        cfg = GoodputRecorder.default_config().set(
            name="test-rolling",
            upload_dir="/test",
            upload_interval=30,
            rolling_window_size=rolling_window_size,
            jax_backend=mock_jax_backend,
            enable_monitoring=True,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value
            if not is_rolling_window_enabled:
                with recorder._maybe_monitor_rolling_window_goodput():
                    pass
                mock_monitor_cls.assert_not_called()
                return
            with recorder._maybe_monitor_rolling_window_goodput():
                pass

            mock_monitor_cls.assert_called_once_with(
                job_name="test-rolling",
                logger_name="goodput_logger_test-rolling",
                tensorboard_dir="/test/rolling_window_test-rolling",
                upload_interval=30,
                monitoring_enabled=True,
                pathway_enabled=is_pathways_job,
                include_badput_breakdown=True,
            )

            mock_monitor_instance.start_rolling_window_goodput_uploader.assert_called_with(
                rolling_window_size
            )
            mock_monitor_instance.stop_rolling_window_goodput_uploader.assert_called_once()

    @mock.patch("jax.process_index", return_value=1)
    def test_non_zero_process_index_skips_monitoring(self, mock_process_index):  # pylint: disable=unused-argument
        """Tests that monitoring is skipped on non-zero process indices."""
        cfg = GoodputRecorder.default_config().set(
            name="test", upload_dir="/test", upload_interval=30, enable_monitoring=True
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            # Test cumulative goodput monitoring.
            with recorder._maybe_monitor_goodput():
                pass
            mock_monitor_cls.assert_not_called()

            cfg_rolling = GoodputRecorder.default_config().set(
                name="test-rolling-skip",
                upload_dir="/test",
                upload_interval=30,
                rolling_window_size=[10, 20],
                enable_monitoring=True,
            )
            recorder_rolling = GoodputRecorder(cfg_rolling)
            with recorder_rolling._maybe_monitor_rolling_window_goodput():
                pass
            mock_monitor_cls.assert_not_called()

    @parameterized.parameters(
        dict(
            rolling_window_size=[5, 10],
            jax_backend="tpu",
            expected_monitor_calls=2,  # Cumulative & Rolling Window
            expect_rolling=True,
            expect_cumulative=True,
        ),
        dict(
            rolling_window_size=[],
            jax_backend="tpu",
            expected_monitor_calls=1,  # Cumulative only
            expect_rolling=False,
            expect_cumulative=True,
        ),
        dict(
            rolling_window_size=[5, 10],
            jax_backend=None,  # Disables Pathways
            expected_monitor_calls=2,
            expect_rolling=True,
            expect_cumulative=True,
        ),
        dict(
            rolling_window_size=[],
            jax_backend=None,
            expected_monitor_calls=1,
            expect_rolling=False,
            expect_cumulative=True,
        ),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_all(
        self,
        _,
        rolling_window_size,
        jax_backend,
        expected_monitor_calls,
        expect_rolling,
        expect_cumulative,
    ):
        """Tests all goodput monitoring with various configs."""
        cfg = GoodputRecorder.default_config().set(
            name="test-all",
            upload_dir="/test",
            upload_interval=30,
            rolling_window_size=rolling_window_size,
            jax_backend=jax_backend,
            enable_monitoring=True,
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value

            with recorder.maybe_monitor_all():
                pass

            self.assertEqual(mock_monitor_cls.call_count, expected_monitor_calls)

            if expect_cumulative:
                mock_monitor_instance.start_goodput_uploader.assert_called_once()
                mock_monitor_instance.stop_goodput_uploader.assert_called_once()
            else:
                mock_monitor_instance.start_goodput_uploader.assert_not_called()
                mock_monitor_instance.stop_goodput_uploader.assert_not_called()

            if expect_rolling:
                mock_monitor_instance.start_rolling_window_goodput_uploader.assert_called_once_with(
                    rolling_window_size
                )
                mock_monitor_instance.stop_rolling_window_goodput_uploader.assert_called_once()
            else:
                mock_monitor_instance.start_rolling_window_goodput_uploader.assert_not_called()
                mock_monitor_instance.stop_rolling_window_goodput_uploader.assert_not_called()

    @mock.patch("jax.process_index", return_value=0)
    def test_enable_monitoring_enabled_by_default(self, _):
        """Tests that monitoring is enabled by default (enable_monitoring=True)."""
        cfg = GoodputRecorder.default_config().set(
            name="test-default-enabled",
            upload_dir="/test",
            upload_interval=30,
            # Enable_monitoring defaults to True.
            rolling_window_size=[10, 20],
        )
        recorder = GoodputRecorder(cfg)

        # Verify the flag defaults to True
        self.assertTrue(recorder.config.enable_monitoring)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            # Test that cumulative goodput monitoring is active by default.
            with recorder._maybe_monitor_goodput():
                pass
            mock_monitor_cls.assert_called()

            # Test that rolling window monitoring is active by default.
            with recorder._maybe_monitor_rolling_window_goodput():
                pass
            mock_monitor_cls.assert_called()

            # Test that maybe_monitor_all is skipped
            with recorder.maybe_monitor_all():
                pass
            mock_monitor_cls.assert_called()

    @mock.patch("jax.process_index", return_value=0)
    def test_enable_monitoring_explicitly_disabled(self, _):
        """Tests that monitoring is disabled when enable_monitoring=False."""
        cfg = GoodputRecorder.default_config().set(
            name="test-explicitly-disabled",
            upload_dir="/test",
            upload_interval=30,
            enable_monitoring=False,  # Explicitly disabled
            rolling_window_size=[10, 20],
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            # Test cumulative goodput monitoring is skipped
            with recorder._maybe_monitor_goodput():
                pass
            mock_monitor_cls.assert_not_called()

            # Test rolling window monitoring is skipped
            with recorder._maybe_monitor_rolling_window_goodput():
                pass
            mock_monitor_cls.assert_not_called()

            # Test maybe_monitor_all is skipped
            with recorder.maybe_monitor_all():
                pass
            mock_monitor_cls.assert_not_called()

    @mock.patch("jax.process_index", return_value=0)
    def test_enable_monitoring_explicitly_enabled(self, _):
        """Tests that monitoring works when enable_monitoring=True."""
        cfg = GoodputRecorder.default_config().set(
            name="test-enabled",
            upload_dir="/test",
            upload_interval=30,
            enable_monitoring=True,  # Explicitly enabled
            rolling_window_size=[10, 20],
        )
        recorder = GoodputRecorder(cfg)

        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor_instance = mock_monitor_cls.return_value

            # Test cumulative goodput monitoring works
            with recorder._maybe_monitor_goodput():
                pass

            # Should be called once for cumulative monitoring
            self.assertEqual(mock_monitor_cls.call_count, 1)
            mock_monitor_instance.start_goodput_uploader.assert_called_once()
            mock_monitor_instance.stop_goodput_uploader.assert_called_once()

    @mock.patch("jax.process_index", return_value=0)
    def test_record_event_works_with_monitoring_disabled(self, _):
        """Tests that record_event still works when monitoring is disabled."""
        cfg = GoodputRecorder.default_config().set(
            name="test-recording-only",
            upload_dir="/test",
            upload_interval=30,
            enable_monitoring=False,  # Monitoring disabled
        )
        recorder = GoodputRecorder(cfg)

        # Verify that goodput recording still works (not monitoring/uploading)
        with mock.patch("ml_goodput_measurement.goodput.GoodputRecorder") as mock_recorder_cls:
            mock_instance = mock_recorder_cls.return_value
            mock_instance.record_job_start_time = mock.MagicMock()
            mock_instance.record_job_end_time = mock.MagicMock()

            # Record event should work
            with recorder.record_event(measurement.EventType.JOB):
                pass

            # Verify goodput recording happened
            mock_recorder_cls.assert_called_once()
            mock_instance.record_job_start_time.assert_called_once()
            mock_instance.record_job_end_time.assert_called_once()

        # Verify no monitoring/uploading happened
        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            with recorder.maybe_monitor_all():
                pass
            mock_monitor_cls.assert_not_called()

    @parameterized.parameters(
        dict(enable_checkpoint_logging=True, process_index=0, expect_logger=True),
        dict(enable_checkpoint_logging=False, process_index=0, expect_logger=False),
        # As in MaxText, every process logs its checkpoint statistics.
        dict(enable_checkpoint_logging=True, process_index=1, expect_logger=True),
    )
    def test_checkpoint_logger(self, enable_checkpoint_logging, process_index, expect_logger):
        """Tests that the checkpoint logger writes to the goodput log."""
        cfg = GoodputRecorder.default_config().set(
            name="test-ckpt-logger",
            upload_dir="/test",
            upload_interval=30,
            enable_checkpoint_logging=enable_checkpoint_logging,
        )
        recorder = GoodputRecorder(cfg)
        with (
            mock.patch("jax.process_index", return_value=process_index),
            mock.patch("orbax.checkpoint.logging.CloudLogger") as mock_logger_cls,
            mock.patch.object(measurement_gcp.logging, "info") as mock_info,
        ):
            logger = recorder.checkpoint_logger()
        if not expect_logger:
            self.assertIsNone(logger)
            mock_logger_cls.assert_not_called()
            return
        self.assertIs(mock_logger_cls.return_value, logger)
        mock_info.assert_any_call(
            "Successfully set up checkpoint cloud logger: %s", "goodput_logger_test-ckpt-logger"
        )
        options = mock_logger_cls.call_args.kwargs["options"]
        # Same names as the goodput recorder, whose log the library reads.
        self.assertEqual("test-ckpt-logger", options.job_name)
        self.assertEqual("goodput_logger_test-ckpt-logger", options.logger_name)

    @mock.patch("jax.process_index", return_value=0)
    def test_checkpoint_logger_unavailable(self, _):
        """Tests that no logger is returned if it cannot be created."""
        cfg = GoodputRecorder.default_config().set(
            name="test-ckpt-logger", upload_dir="/test", upload_interval=30
        )
        recorder = GoodputRecorder(cfg)
        with mock.patch(
            "orbax.checkpoint.logging.CloudLogger", side_effect=RuntimeError("no credentials")
        ):
            self.assertIsNone(recorder.checkpoint_logger())

    @mock.patch("jax.process_index", return_value=0)
    def test_record_end_job_flushes(self, _):
        """Tests that goodput logs are flushed at the end of the job."""
        cfg = GoodputRecorder.default_config().set(
            name="test-flush", upload_dir="/test", upload_interval=30
        )
        recorder = GoodputRecorder(cfg)
        # Flushing before any event is recorded is a no-op.
        recorder.flush()
        with mock.patch(
            "ml_goodput_measurement.src.goodput_elastic.ElasticGoodputRecorder"
        ) as mock_recorder_cls:
            mock_instance = mock_recorder_cls.return_value
            recorder.record(measurement.Event.START_JOB)
            mock_instance.flush.assert_not_called()
            recorder.record(measurement.Event.END_JOB)
            mock_instance.record_job_end_time.assert_called_once()
            mock_instance.flush.assert_called_once()
            # `flush()` can also be called directly, e.g. on the exception path.
            recorder.flush()
            self.assertEqual(2, mock_instance.flush.call_count)

    @mock.patch("jax.process_index", return_value=0)
    def test_record_without_elastic_library(self, _):
        """Tests the fallback when ml-goodput-measurement has no elastic recorder."""
        cfg = GoodputRecorder.default_config().set(
            name="test-no-elastic", upload_dir="/test", upload_interval=30
        )
        recorder = GoodputRecorder(cfg)
        with (
            mock.patch(f"{measurement_gcp.__name__}.goodput_elastic", None),
            mock.patch(f"{measurement_gcp.__name__}.monitoring_elastic", None),
            mock.patch("ml_goodput_measurement.goodput.GoodputRecorder") as mock_recorder_cls,
            mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls,
        ):
            mock_instance = mock_recorder_cls.return_value
            # The base recorder has neither the elastic methods nor `flush`.
            del mock_instance.record_elastic_wait_start_time
            del mock_instance.flush
            recorder.record(measurement.Event.START_JOB)
            mock_recorder_cls.assert_called_once()
            # Elastic events are ignored instead of raising.
            recorder.record(measurement.Event.START_ELASTIC_WAIT, "elastic_wait")
            recorder.record(measurement.Event.END_JOB)
            mock_instance.record_job_end_time.assert_called_once()
            with recorder.maybe_monitor_all():
                pass
            mock_monitor_cls.assert_called_once()

    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_all_reentrant(self, _):
        """Tests that only the outermost monitoring context starts and stops the uploaders."""
        cfg = GoodputRecorder.default_config().set(
            name="test-reentrant", upload_dir="/test", upload_interval=30, rolling_window_size=[5]
        )
        recorder = GoodputRecorder(cfg)
        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor = mock_monitor_cls.return_value
            with recorder.maybe_monitor_all():
                mock_monitor.start_goodput_uploader.assert_called_once()
                with recorder.maybe_monitor_all():
                    pass
                # The inner context (e.g. `SpmdTrainer.run`) must not trigger the final upload
                # before the outer one (e.g. `launch_trainer.run_trainer`) records END_JOB.
                mock_monitor.stop_goodput_uploader.assert_not_called()
                mock_monitor.stop_rolling_window_goodput_uploader.assert_not_called()
            mock_monitor.start_goodput_uploader.assert_called_once()
            mock_monitor.stop_goodput_uploader.assert_called_once()
            mock_monitor.stop_rolling_window_goodput_uploader.assert_called_once()
            # The recorder can be monitored again afterwards.
            with recorder.maybe_monitor_all():
                pass
            self.assertEqual(2, mock_monitor.stop_goodput_uploader.call_count)

    @parameterized.parameters(
        dict(elastic=True, has_elastic_library=True, expect_elastic=True),
        # A recorded event alone does not make a job elastic.
        dict(elastic=False, has_elastic_library=True, expect_elastic=False),
        dict(elastic=True, has_elastic_library=False, expect_elastic=False),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_goodput_elastic(self, _, elastic, has_elastic_library, expect_elastic):
        """Tests that elastic jobs use the elastic monitor, which uploads the slice efficiency."""
        cfg = GoodputRecorder.default_config().set(
            name="test-monitor", upload_dir="/test", upload_interval=30, rolling_window_size=[5]
        )
        recorder = GoodputRecorder(cfg)
        with (
            mock.patch.object(
                elastic_utils, "elastic_manager", mock.MagicMock() if elastic else None
            ),
            mock.patch(
                "ml_goodput_measurement.src.monitoring_elastic.ElasticGoodputMonitor"
            ) as mock_elastic_cls,
            mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls,
            mock.patch("ml_goodput_measurement.src.goodput_elastic.ElasticGoodputRecorder"),
            contextlib.ExitStack() as stack,
        ):
            if not has_elastic_library:
                stack.enter_context(
                    mock.patch(f"{measurement_gcp.__name__}.monitoring_elastic", None)
                )
            recorder.record(measurement.Event.START_JOB)
            with recorder.maybe_monitor_all():
                pass
        if not expect_elastic:
            mock_elastic_cls.assert_not_called()
            self.assertEqual(2, mock_monitor_cls.call_count)
            return
        mock_monitor_cls.assert_not_called()
        # Both the cumulative and the rolling window monitors.
        self.assertEqual(2, mock_elastic_cls.call_count)
        mock_elastic_cls.assert_any_call(
            job_name="test-monitor",
            logger_name="goodput_logger_test-monitor",
            tensorboard_dir="/test",
            upload_interval=30,
            monitoring_enabled=True,
            include_badput_breakdown=True,
            include_slice_efficiency=True,
        )

    @parameterized.parameters(
        dict(recorder_spec=[], expected={}),
        dict(
            recorder_spec=[
                "upload_interval=15",
                "enable_gcp_goodput_metrics=false",
                "enable_gcp_step_deviation_metrics=false",
                "include_step_deviation=true",
                "step_deviation_interval_seconds=60",
            ],
            expected=dict(
                upload_interval=15,
                enable_gcp_goodput_metrics=False,
                enable_gcp_step_deviation_metrics=False,
                include_step_deviation=True,
                step_deviation_interval_seconds=60,
            ),
        ),
    )
    def test_from_flags_monitoring_options(self, recorder_spec, expected):
        """Tests the parsing and the defaults (as in MaxText) of the monitoring options."""
        mock_fv = mock.MagicMock(spec=flags.FlagValues)
        mock_fv.recorder_spec = ["name=test-name", "upload_dir=/test/path", *recorder_spec]
        recorder = GoodputRecorder.from_flags(mock_fv)
        defaults = dict(
            upload_interval=30,
            enable_gcp_goodput_metrics=True,
            enable_gcp_step_deviation_metrics=True,
            include_step_deviation=False,
            step_deviation_interval_seconds=30,
        )
        for key, value in {**defaults, **expected}.items():
            self.assertEqual(value, getattr(recorder.config, key), key)

    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_all_gcp_options(self, _):
        """Tests that both monitors get the Google Cloud Monitoring options."""
        cfg = GoodputRecorder.default_config().set(
            name="test-gcp",
            upload_dir="/test",
            rolling_window_size=[5],
            enable_gcp_goodput_metrics=False,
        )
        recorder = GoodputRecorder(cfg)
        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            with recorder.maybe_monitor_all():
                pass
        self.assertEqual(2, mock_monitor_cls.call_count)
        for call in mock_monitor_cls.call_args_list:
            self.assertEqual(
                measurement_gcp.goodput_monitoring.GCPOptions(
                    enable_gcp_goodput_metrics=False, enable_gcp_step_deviation_metrics=True
                ),
                call.kwargs["gcp_options"],
            )
            self.assertNotIn("include_step_deviation", call.kwargs)

    @parameterized.parameters(
        dict(include_step_deviation=False, start_error=None, stop_error=None),
        dict(include_step_deviation=True, start_error=None, stop_error=None),
        # Failures of the step deviation uploader do not interrupt training.
        dict(include_step_deviation=True, start_error=RuntimeError("start"), stop_error=None),
        dict(include_step_deviation=True, start_error=None, stop_error=RuntimeError("stop")),
    )
    @mock.patch("jax.process_index", return_value=0)
    def test_maybe_monitor_goodput_step_deviation(
        self, _, include_step_deviation, start_error, stop_error
    ):
        """Tests the step deviation uploader."""
        cfg = GoodputRecorder.default_config().set(
            name="test-step-deviation",
            upload_dir="/test",
            include_step_deviation=include_step_deviation,
            step_deviation_interval_seconds=60,
        )
        recorder = GoodputRecorder(cfg)
        with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor_cls:
            mock_monitor = mock_monitor_cls.return_value
            mock_monitor.start_step_deviation_uploader.side_effect = start_error
            mock_monitor.stop_step_deviation_uploader.side_effect = stop_error
            with recorder._maybe_monitor_goodput():
                mock_monitor.stop_step_deviation_uploader.assert_not_called()
        kwargs = mock_monitor_cls.call_args.kwargs
        mock_monitor.stop_goodput_uploader.assert_called_once()
        if not include_step_deviation:
            self.assertNotIn("include_step_deviation", kwargs)
            mock_monitor.start_step_deviation_uploader.assert_not_called()
            mock_monitor.stop_step_deviation_uploader.assert_not_called()
            return
        self.assertTrue(kwargs["include_step_deviation"])
        self.assertEqual(60, kwargs["step_deviation_interval_seconds"])
        mock_monitor.start_step_deviation_uploader.assert_called_once()
        if start_error:
            mock_monitor.stop_step_deviation_uploader.assert_not_called()
        else:
            mock_monitor.stop_step_deviation_uploader.assert_called_once()

    @mock.patch("jax.process_index", return_value=0)
    def test_logs_created_classes(self, _):
        """Tests that the recorder and monitor classes are logged when created."""
        cfg = GoodputRecorder.default_config().set(name="test-log", upload_dir="/test")
        recorder = GoodputRecorder(cfg)
        with (
            mock.patch("ml_goodput_measurement.src.goodput_elastic.ElasticGoodputRecorder"),
            mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor"),
            mock.patch.object(logging, "info") as mock_info,
        ):
            recorder.record(measurement.Event.START_JOB)
            with recorder.maybe_monitor_all():
                pass
        created = [c.args for c in mock_info.call_args_list if "created" in c.args[0]]
        self.assertEqual(2, len(created))
        for args in created:
            self.assertEqual(("test-log", "goodput_logger_test-log"), args[2:])


if __name__ == "__main__":
    absltest.main()
