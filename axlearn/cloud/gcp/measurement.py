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
try:
    from ml_goodput_measurement.src import goodput_elastic, monitoring_elastic
except ImportError:
    goodput_elastic = monitoring_elastic = None

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
        """

        upload_dir: Required[str] = REQUIRED
        upload_interval: Required[int] = REQUIRED
        rolling_window_size: Sequence[int] = []
        jax_backend: Optional[str] = None
        # Enable or disable monitoring. Recording is always enabled.
        enable_monitoring: bool = True
        enable_checkpoint_logging: bool = True

    @classmethod
    def from_flags(cls, fv: flags.FlagValues) -> "GoodputRecorder":
        """Converts flags to a recorder.

        `fv.recorder_spec` will be interpreted as a list of `key=value` pairs; config names
        corresponding to keys will be set to the corresponding values. A GoodputRecorder can
        additionally take in following Tensorboard configs in the recorder_spec:
        - upload_dir: The directory to write Tensorboard data to.
        - upload_interval: The time interval in seconds at which to query and upload data
            to Tensorboard.
        - rolling_window_size: Comma-separated list of integers representing rolling window
            sizes in seconds.
        - jax_backend: The type of jax backend.
        - enable_monitoring: Boolean to enable/disable goodput monitoring (default: true).
        - enable_checkpoint_logging: Boolean to enable/disable checkpoint logging (default: true).
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
        for key in ("enable_monitoring", "enable_checkpoint_logging"):
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
        # Depth of active `maybe_monitor_all()` contexts. Only the outermost one starts and stops
        # the uploaders, so that the final upload happens after the caller records END_JOB.
        self._monitoring_depth = 0

    def checkpoint_logger(self) -> Optional[Any]:
        """Returns an Orbax logger that writes checkpoint statistics to the goodput log.

        ml-goodput-measurement computes the checkpoint badput from these statistics.

        As in MaxText, every process logs its statistics; the library averages the entries of a
        step across processes.

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
        """Returns the goodput monitor class to instantiate."""
        # Imported lazily: it imports pathwaysutils, which only elastic jobs need.
        # pylint: disable-next=import-outside-toplevel
        from axlearn.common import elastic_utils

        # The trainer creates the elastic manager before monitoring starts. Older
        # ml-goodput-measurement releases have no elastic monitor.
        if monitoring_elastic is not None and elastic_utils.elastic_enabled():
            # Also upload the slice efficiency computed from the slice counts, as MaxText does.
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
                )

            self._monitor.start_goodput_uploader(*args, **kwargs)
            logging.info("Started Goodput upload to Tensorboard & GCM in the background!")
            yield
        finally:
            if self._monitor:
                self._monitor.stop_goodput_uploader()
                logging.info("Flushed final metrics and safe exited from Goodput monitoring.")

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
            # Re-entrant: only the outermost context starts and stops the uploaders, so that an
            # inner one (e.g. `SpmdTrainer.run`) does not trigger the final upload before the
            # outer one (e.g. `launch_trainer.run_trainer`) has recorded END_JOB.
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
            # Older ml-goodput-measurement releases have no elastic recorder.
            recorder_cls = (
                goodput_elastic.ElasticGoodputRecorder
                if goodput_elastic is not None
                else goodput.GoodputRecorder
            )
            self._recorder = recorder_cls(
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
        elif method_name := {
            measurement_base.Event.START_ELASTIC_WAIT: "record_elastic_wait_start_time",
            measurement_base.Event.END_ELASTIC_WAIT: "record_elastic_wait_end_time",
            measurement_base.Event.START_ELASTIC_REINIT: "record_elastic_reinit_start_time",
            measurement_base.Event.END_ELASTIC_REINIT: "record_elastic_reinit_end_time",
            measurement_base.Event.RECORD_SLICE_COUNTS: "record_elastic_slice_counts",
        }.get(event):
            if (fn := getattr(self._recorder, method_name, None)) is not None:
                fn(*args, **kwargs)
            else:
                logging.log_first_n(
                    logging.WARNING,
                    "Ignoring elastic event %s: ml-goodput-measurement has no elastic recorder.",
                    1,
                    event,
                )
        else:
            logging.log_first_n(
                logging.WARNING,
                "Ignoring unknown event %s",
                1,
                event,
            )

    def flush(self):
        """Flushes pending goodput log writes.

        Logs are written asynchronously (ml-goodput-measurement >= 0.2), so call this before the
        process exits, whether or not the job ended gracefully. A no-op on older versions.
        """
        if self._recorder is None:
            return
        if (flush := getattr(self._recorder, "flush", None)) is not None:
            flush()

    def start_monitoring(self, **kwargs):
        """Deprecated: `start_monitoring()` is not used in GoodputRecorder.
        Use the maybe_monitor_all context manager instead.
        """
        pass
