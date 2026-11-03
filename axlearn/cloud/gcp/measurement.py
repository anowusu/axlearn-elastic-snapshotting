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
import functools
import os
from typing import Any, Callable, Optional, Sequence

import jax
from absl import flags, logging
from ml_goodput_measurement import goodput
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
            enable_gcp_step_deviation_metrics: Whether to upload step deviation metrics to Google
                Cloud Monitoring.
            include_step_deviation: Whether to compute and upload the step time deviation.
            step_deviation_interval_seconds: Time interval (seconds) for step deviation uploads.
        """

        upload_dir: Required[str] = REQUIRED
        upload_interval: int = 30
        rolling_window_size: Sequence[int] = []
        jax_backend: Optional[str] = None
        # Enable or disable monitoring. Recording is always enabled.
        enable_monitoring: bool = True
        enable_checkpoint_logging: bool = True
        enable_gcp_goodput_metrics: bool = True
        enable_gcp_step_deviation_metrics: bool = True
        include_step_deviation: bool = False
        step_deviation_interval_seconds: int = 30

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
        - enable_gcp_step_deviation_metrics: Boolean to enable/disable step deviation metrics in
            Google Cloud Monitoring (default: true).
        - include_step_deviation: Boolean to enable/disable step deviation monitoring
            (default: false).
        - step_deviation_interval_seconds: The time interval in seconds at which to query and
            upload the step deviation (default: 30).
        """
        cfg: measurement_base.Recorder.Config = cls.default_config()
        parsed_flags = parse_kv_flags(fv.recorder_spec, delimiter="=")
        for key in ("upload_interval", "step_deviation_interval_seconds"):
            if key in parsed_flags:
                parsed_flags[key] = int(parsed_flags[key])
        if "rolling_window_size" in parsed_flags and isinstance(
            parsed_flags["rolling_window_size"], str
        ):
            parsed_flags["rolling_window_size"] = [
                int(x) for x in parsed_flags["rolling_window_size"].split(",")
            ]
        for key in (
            "enable_monitoring",
            "enable_checkpoint_logging",
            "enable_gcp_goodput_metrics",
            "enable_gcp_step_deviation_metrics",
            "include_step_deviation",
        ):
            if key in parsed_flags:
                parsed_flags[key] = to_bool(parsed_flags[key])
        return maybe_set_config(cfg, **parsed_flags).instantiate()

    def __init__(self, cfg):
        super().__init__(cfg)
        self._recorder: Optional[goodput.GoodputRecorder] = None
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

    @contextlib.contextmanager
    def record_event(self, event: measurement_base.EventType, *args, **kwargs):
        """Records a goodput event using a context manager."""
        # Lazily instantiate the recorder if it hasn't been already.
        if self._recorder is None:
            if jax.process_index() == 0:
                logging.info("Lazily instantiating goodput recorder.")
            self._recorder = goodput.GoodputRecorder(
                job_name=self._job_name,
                logger_name=self._logger_name,
                logging_enabled=(jax.process_index() == 0),
            )

        start_method_name = f"record_{event.value}_start_time"
        end_method_name = f"record_{event.value}_end_time"

        record_event_start = getattr(self._recorder, start_method_name, None)
        record_event_end = getattr(self._recorder, end_method_name, None)

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

    def _monitor_class(self) -> Callable[..., goodput_monitoring.GoodputMonitor]:
        """Returns the goodput monitor class, with the slice efficiency for elastic jobs."""
        # Imported lazily, since it imports pathwaysutils.
        # pylint: disable-next=import-outside-toplevel
        from axlearn.common import elastic_utils

        # The trainer creates the elastic manager before monitoring starts.
        if elastic_utils.elastic_enabled():
            return functools.partial(
                monitoring_elastic.ElasticGoodputMonitor, include_slice_efficiency=True
            )
        return functools.partial(
            goodput_monitoring.GoodputMonitor, pathway_enabled=self.config.jax_backend == "proxy"
        )

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
                self._monitor = self._monitor_class()(
                    job_name=self._job_name,
                    logger_name=self._logger_name,
                    tensorboard_dir=self.config.upload_dir,
                    upload_interval=self.config.upload_interval,
                    monitoring_enabled=True,
                    include_badput_breakdown=True,
                    **self._monitor_options(step_deviation=True),
                )

            self._monitor.start_goodput_uploader(*args, **kwargs)
            logging.info("Started Goodput upload to Tensorboard & GCM in the background!")
            if self.config.include_step_deviation:
                self._monitor.start_step_deviation_uploader()
            yield
        finally:
            if self._monitor:
                self._monitor.stop_goodput_uploader()
                if self.config.include_step_deviation:
                    self._monitor.stop_step_deviation_uploader()
                logging.info("Flushed final metrics and safe exited from Goodput monitoring.")

    def _monitor_options(self, *, step_deviation: bool = False) -> dict[str, Any]:
        """Returns the monitor arguments that differ from the library defaults."""
        cfg: GoodputRecorder.Config = self.config
        options = {}
        if not (cfg.enable_gcp_goodput_metrics and cfg.enable_gcp_step_deviation_metrics):
            options["gcp_options"] = goodput_monitoring.GCPOptions(
                enable_gcp_goodput_metrics=cfg.enable_gcp_goodput_metrics,
                enable_gcp_step_deviation_metrics=cfg.enable_gcp_step_deviation_metrics,
            )
        if step_deviation and cfg.include_step_deviation:
            options.update(
                include_step_deviation=True,
                step_deviation_interval_seconds=cfg.step_deviation_interval_seconds,
            )
        return options

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
                self._rolling_window_monitor = self._monitor_class()(
                    job_name=self._job_name,
                    logger_name=self._logger_name,
                    tensorboard_dir=rolling_window_tensorboard_dir,
                    upload_interval=self.config.upload_interval,
                    monitoring_enabled=True,
                    include_badput_breakdown=True,
                    **self._monitor_options(),
                )
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
        """Deprecated: `record()` is not used in GoodputRecorder.
        Use the record_event context manager instead.
        """
        # Lazily instantiate the recorder. This avoids invoking jax before setup is complete.
        if self._recorder is None:
            cfg: GoodputRecorder.Config = self.config
            self._recorder = goodput_elastic.ElasticGoodputRecorder(
                job_name=cfg.name,
                logger_name=f"goodput_logger_{cfg.name}",
                logging_enabled=(jax.process_index() == 0),
            )

        if event == measurement_base.Event.START_JOB:
            self._recorder.record_job_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_JOB:
            self._recorder.record_job_end_time(*args, **kwargs)
            self.flush()
        elif event == measurement_base.Event.START_STEP:
            self._recorder.record_step_start_time(*args, **kwargs)
        elif event == measurement_base.Event.START_ACCELERATOR_INIT:
            self._recorder.record_tpu_init_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_ACCELERATOR_INIT:
            self._recorder.record_tpu_init_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_TRAINING_PREPARATION:
            self._recorder.record_training_preparation_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_TRAINING_PREPARATION:
            self._recorder.record_training_preparation_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_DATA_LOADING:
            self._recorder.record_data_loading_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_DATA_LOADING:
            self._recorder.record_data_loading_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_CUSTOM_BADPUT_EVENT:
            self._recorder.record_custom_badput_event_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_CUSTOM_BADPUT_EVENT:
            self._recorder.record_custom_badput_event_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_ELASTIC_WAIT:
            self._recorder.record_elastic_wait_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_ELASTIC_WAIT:
            self._recorder.record_elastic_wait_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_ELASTIC_REINIT:
            self._recorder.record_elastic_reinit_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_ELASTIC_REINIT:
            self._recorder.record_elastic_reinit_end_time(*args, **kwargs)
        elif event == measurement_base.Event.RECORD_SLICE_COUNTS:
            self._recorder.record_elastic_slice_counts(*args, **kwargs)
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
