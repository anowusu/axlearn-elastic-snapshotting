# Copyright © 2024 Apple Inc.

"""Measurement utils for GCP.

    For detailed documentation and advanced usage, please refer to:
    axlearn/docs/05-Goodput-Monitoring.md

    Example:

    # Enable Goodput when launching an AXLearn training job
    axlearn gcp launch run --instance_type=tpu-v5litepod-16 \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile \
        -- python3 -m my_training_job \
        --recorder_type=axlearn.cloud.gcp.measurement:goodput \
        --recorder_spec=name=my-run-with-goodput \
        --recorder_spec=upload_dir=my-output-directory/summaries \
        --recorder_spec=upload_interval=30 \
        --recorder_spec=rolling_window_size=86400,604800

"""

import contextlib
import os
from typing import Any, Optional, Sequence

import jax
from absl import flags, logging
from ml_goodput_measurement import monitoring as goodput_monitoring
from ml_goodput_measurement.src import goodput_elastic, monitoring_elastic

from axlearn.cloud.common.utils import parse_kv_flags, to_bool
from axlearn.common import measurement_base
from axlearn.common.config import REQUIRED, Required, config_class, maybe_set_config


@measurement_base.register_recorder("goodput_elastic")
@measurement_base.register_recorder("goodput")
class GoodputRecorder(measurement_base.Recorder):
    """Records overall training goodput."""

    @config_class
    class Config(measurement_base.Recorder.Config):
        """Configures GoodputRecorder.

        Attributes:
            upload_dir: Directory to store metrics for the monitor.
            upload_interval: Time interval (seconds) for monitoring uploads.
                See "How to Monitor Cumulative Goodput Metrics" in
                docs/05-Goodput-Monitoring.md for more details.
            rolling_window_size: A sequence of integers defining the rolling window sizes in
                seconds.
                See "How to Monitor Rolling Window Goodput Metrics" in
                docs/05-Goodput-Monitoring.md for more details.
            jax_backend: Jax backend type to infer Pathways environment.
            enable_monitoring: Whether to enable goodput monitoring/uploading.
            enable_checkpoint_logging: Whether to log Orbax checkpoint statistics for goodput.
            enable_gcp_goodput_metrics: Whether to upload goodput metrics to Google Cloud
                Monitoring.
        """

        upload_dir: Required[str] = REQUIRED
        upload_interval: int = 30
        rolling_window_size: Sequence[int] = []
        jax_backend: Optional[str] = None
        # Enable or disable monitoring. Recording is always enabled.
        enable_monitoring: bool = True
        enable_checkpoint_logging: bool = True
        enable_gcp_goodput_metrics: bool = True

    @classmethod
    def from_flags(cls, fv: flags.FlagValues) -> "GoodputRecorder":
        """Converts flags to a recorder.

        `fv.recorder_spec` will be interpreted as a list of `key=value` pairs; config names
        corresponding to keys will be set to the corresponding values. A GoodputRecorder can
        additionally take in following Tensorboard configs in the recorder_spec:
        - upload_dir: The directory to write Tensorboard data to.
        - upload_interval: The time interval in seconds at which to query and upload data
            to Tensorboard (default: 30).
        - rolling_window_size: Comma-separated list of integers representing rolling window
            sizes in seconds.
        - jax_backend: The type of jax backend.
        - enable_monitoring: Boolean to enable/disable goodput monitoring (default: true).
        - enable_checkpoint_logging: Boolean to enable/disable checkpoint logging (default: true).
        - enable_gcp_goodput_metrics: Boolean to enable/disable goodput metrics in Google Cloud
            Monitoring (default: true).
        """
        cfg: measurement_base.Recorder.Config = cls.default_config()
        parsed_flags = parse_kv_flags(fv.recorder_spec, delimiter="=")
        if "upload_interval" in parsed_flags:
            parsed_flags["upload_interval"] = int(parsed_flags["upload_interval"])
        if "rolling_window_size" in parsed_flags and isinstance(
            parsed_flags["rolling_window_size"], str
        ):
            parsed_flags["rolling_window_size"] = [
                int(x) for x in parsed_flags["rolling_window_size"].split(",")
            ]
        for key in ("enable_monitoring", "enable_checkpoint_logging", "enable_gcp_goodput_metrics"):
            if key in parsed_flags:
                parsed_flags[key] = to_bool(parsed_flags[key])
        return maybe_set_config(cfg, **parsed_flags).instantiate()

    def __init__(self, cfg):
        super().__init__(cfg)
        self._recorder: Optional[goodput_elastic.ElasticGoodputRecorder] = None
        self._monitor: Optional[goodput_monitoring.GoodputMonitor] = None
        self._rolling_window_monitor: Optional[goodput_monitoring.GoodputMonitor] = None
        self._job_name = cfg.name
        self._logger_name = f"goodput_logger_{cfg.name}"
        # Depth of nested `maybe_monitor_all()` contexts.
        self._monitoring_depth = 0

    def checkpoint_logger(self) -> Optional[Any]:
        """Returns an Orbax logger that writes checkpoint statistics to the goodput log.

        ml-goodput-measurement computes the checkpoint badput from these statistics. As in MaxText,
        every process logs; the library averages the entries of a step.

        Returns:
            An Orbax `CloudLogger`, or None if disabled or unavailable.
        """
        if not self.config.enable_checkpoint_logging:
            return None
        try:
            # pylint: disable-next=import-outside-toplevel
            from orbax.checkpoint import logging as ocp_logging

            checkpoint_logger = ocp_logging.CloudLogger(
                options=ocp_logging.CloudLoggerOptions(
                    job_name=self._job_name, logger_name=self._logger_name
                )
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Checkpoint goodput logging is disabled: %s", e)
            return None
        logging.info("Successfully set up checkpoint cloud logger: %s", self._logger_name)
        return checkpoint_logger

    def _get_recorder(self) -> goodput_elastic.ElasticGoodputRecorder:
        """Returns the goodput recorder, which is created lazily to not invoke jax before setup.

        Both `record()` and `record_event()` use it. The elastic recorder also records the elastic
        events and works for non-elastic jobs.
        """
        if self._recorder is None:
            if jax.process_index() == 0:
                logging.info("Lazily instantiating goodput recorder.")
            self._recorder = goodput_elastic.ElasticGoodputRecorder(
                job_name=self._job_name,
                logger_name=self._logger_name,
                logging_enabled=(jax.process_index() == 0),
            )
        return self._recorder

    @contextlib.contextmanager
    def record_event(self, event: measurement_base.EventType, *args, **kwargs):
        """Records a goodput event using a context manager."""
        recorder = self._get_recorder()
        start_method_name = f"record_{event.value}_start_time"
        end_method_name = f"record_{event.value}_end_time"

        record_event_start = getattr(recorder, start_method_name, None)
        record_event_end = getattr(recorder, end_method_name, None)

        if record_event_start:
            try:
                record_event_start(*args, **kwargs)
            except RuntimeError as e:
                logging.warning(
                    "Failed to record start of event %s. Error: %s", event.value, e, exc_info=True
                )
        # pylint: disable=try-except-raise
        try:
            yield  # Run the user code in the context
        except Exception:
            raise
        else:
            if record_event_end:
                try:
                    record_event_end(*args, **kwargs)
                except RuntimeError as e:
                    logging.warning(
                        "Failed to record end of event %s. Error: %s", event.value, e, exc_info=True
                    )
        # pylint: enable=try-except-raise

    def _create_monitor(self, tensorboard_dir: str) -> goodput_monitoring.GoodputMonitor:
        """Creates a goodput monitor, with the slice efficiency for elastic jobs.

        As in MaxText, falls back to the base monitor if the elastic monitor cannot be created.
        """
        # Imported lazily, since it imports pathwaysutils.
        # pylint: disable-next=import-outside-toplevel
        from axlearn.common import elastic_utils

        kwargs = dict(
            job_name=self._job_name,
            logger_name=self._logger_name,
            tensorboard_dir=tensorboard_dir,
            upload_interval=self.config.upload_interval,
            monitoring_enabled=True,
            include_badput_breakdown=True,
            **self._monitor_options(),
        )
        pathway_enabled = self.config.jax_backend == "proxy"
        # The trainer creates the elastic manager before monitoring starts.
        if elastic_utils.elastic_enabled():
            try:
                monitor = monitoring_elastic.ElasticGoodputMonitor(
                    include_slice_efficiency=True, **kwargs
                )
                logging.info("Goodput: using ElasticGoodputMonitor for %s.", tensorboard_dir)
                return monitor
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning(
                    "Failed to create the elastic goodput monitor, using the base monitor: %s", e
                )
            # Elastic training runs on Pathways.
            pathway_enabled = True
        monitor = goodput_monitoring.GoodputMonitor(pathway_enabled=pathway_enabled, **kwargs)
        logging.info("Goodput: using GoodputMonitor for %s.", tensorboard_dir)
        return monitor

    @contextlib.contextmanager
    def _maybe_monitor_goodput(self, *args, **kwargs):
        """Monitor cumulative goodput if enabled.

        Instantiate ml-goodput-measurement's GoodputMonitor to asynchronously calculate
        Goodput, Badput, Step & Disruption Information at the upload_interval to the
        specified TensorBoard directory and Google Cloud Monitoring.
        Note: This function requires initialization of distributed JAX before it is called.
        If there are internal GCP errors from querying and uploading data, these will be
        logged without affecting the workload. GoodputMonitor logs will provide further
        information if data is not being uploaded correctly.

        Default behavior is to push metrics to Google Cloud Monitoring.
        This behavior can be overridden by configuring `goodput_monitoring.GCPOptions`
        """
        if not self.config.enable_monitoring or jax.process_index() != 0:
            yield
            return
        try:
            if self._monitor is None:
                self._monitor = self._create_monitor(self.config.upload_dir)

            self._monitor.start_goodput_uploader(*args, **kwargs)
            logging.info("Started Goodput upload to Tensorboard & GCM in the background!")
            yield
        finally:
            if self._monitor:
                self._monitor.stop_goodput_uploader()
                logging.info("Flushed final metrics and safe exited from Goodput monitoring.")

    def _monitor_options(self) -> dict[str, Any]:
        """Returns the monitor arguments that differ from the library defaults."""
        if self.config.enable_gcp_goodput_metrics:
            return {}
        return dict(gcp_options=goodput_monitoring.GCPOptions(enable_gcp_goodput_metrics=False))

    @contextlib.contextmanager
    def _maybe_monitor_rolling_window_goodput(self):
        """Monitor rolling window goodput if enabled."""
        if (
            not self.config.enable_monitoring
            or not self.config.rolling_window_size
            or jax.process_index() != 0
        ):
            yield
            return
        try:
            if self._rolling_window_monitor is None:
                rolling_window_tensorboard_dir = os.path.join(
                    self.config.upload_dir, f"rolling_window_{self.config.name}"
                )
                self._rolling_window_monitor = self._create_monitor(rolling_window_tensorboard_dir)
            self._rolling_window_monitor.start_rolling_window_goodput_uploader(
                self.config.rolling_window_size
            )
            logging.info("Started Rolling Window Goodput monitoring in the background!")
            yield
        finally:
            if self._rolling_window_monitor:
                self._rolling_window_monitor.stop_rolling_window_goodput_uploader()
                logging.info(
                    "Flushed final metrics and safe exited from Rolling Window Goodput monitoring."
                )

    def maybe_monitor_all(self):
        goodput_monitor_manager = self._maybe_monitor_goodput()
        rolling_goodput_monitor_manager = self._maybe_monitor_rolling_window_goodput()

        @contextlib.contextmanager
        def monitor_goodput():
            # Only the outermost context (e.g. in `launch_trainer.run_trainer`) starts and stops
            # the uploaders, so that the final upload happens after END_JOB is recorded.
            outermost = self._monitoring_depth == 0
            self._monitoring_depth += 1
            try:
                with contextlib.ExitStack() as stack:
                    if outermost:
                        stack.enter_context(goodput_monitor_manager)
                        stack.enter_context(rolling_goodput_monitor_manager)
                    yield
            finally:
                self._monitoring_depth -= 1

        return monitor_goodput()

    def record(self, event: measurement_base.Event, *args, **kwargs):
        """Records a goodput event.

        Goodput is best effort: a failure to record an event, e.g. in an elastic recovery callback,
        is logged and never fails training.
        """
        try:
            self._record(event, *args, **kwargs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.log_every_n_seconds(
                logging.WARNING, "Failed to record goodput event %s: %s", 60, event, e
            )

    def _record(self, event: measurement_base.Event, *args, **kwargs):
        recorder = self._get_recorder()
        if event == measurement_base.Event.START_JOB:
            recorder.record_job_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_JOB:
            recorder.record_job_end_time(*args, **kwargs)
            self.flush()
        elif event == measurement_base.Event.START_STEP:
            recorder.record_step_start_time(*args, **kwargs)
        elif event == measurement_base.Event.START_ACCELERATOR_INIT:
            recorder.record_tpu_init_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_ACCELERATOR_INIT:
            recorder.record_tpu_init_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_TRAINING_PREPARATION:
            recorder.record_training_preparation_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_TRAINING_PREPARATION:
            recorder.record_training_preparation_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_DATA_LOADING:
            recorder.record_data_loading_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_DATA_LOADING:
            recorder.record_data_loading_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_CUSTOM_BADPUT_EVENT:
            recorder.record_custom_badput_event_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_CUSTOM_BADPUT_EVENT:
            recorder.record_custom_badput_event_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_ELASTIC_WAIT:
            recorder.record_elastic_wait_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_ELASTIC_WAIT:
            recorder.record_elastic_wait_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_ELASTIC_REINIT:
            recorder.record_elastic_reinit_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_ELASTIC_REINIT:
            recorder.record_elastic_reinit_end_time(*args, **kwargs)
        elif event == measurement_base.Event.RECORD_SLICE_COUNTS:
            recorder.record_elastic_slice_counts(*args, **kwargs)
        else:
            logging.log_first_n(
                logging.WARNING,
                "Ignoring unknown event %s",
                1,
                event,
            )

    def flush(self):
        """Flushes the goodput logs, which are written asynchronously."""
        if self._recorder is not None:
            self._recorder.flush()

    def start_monitoring(self, **kwargs):
        """Deprecated: `start_monitoring()` is not used in GoodputRecorder.
        Use the maybe_monitor_all context manager instead.
        """
        pass
