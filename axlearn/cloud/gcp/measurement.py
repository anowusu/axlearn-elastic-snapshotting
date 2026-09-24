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
from typing import Optional, Sequence

import jax
from absl import flags, logging
from ml_goodput_measurement import goodput
from ml_goodput_measurement import monitoring as goodput_monitoring

try:
    from ml_goodput_measurement.src import goodput_elastic
    from ml_goodput_measurement.src import monitoring_elastic
except ImportError:
    goodput_elastic = None
    monitoring_elastic = None

from axlearn.cloud.common.utils import parse_kv_flags, to_bool
from axlearn.common import measurement_base
from axlearn.common.config import REQUIRED, Required, config_class, maybe_set_config


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
            step_deviation_interval_seconds: Interval in seconds for step deviation uploads.
        """

        upload_dir: Required[str] = REQUIRED
        upload_interval: Required[int] = REQUIRED
        rolling_window_size: Sequence[int] = []
        jax_backend: Optional[str] = None
        # Enable or disable monitoring. Recording is always enabled.
        enable_monitoring: bool = True
        step_deviation_interval_seconds: int = 10

    @classmethod
    def from_flags(cls, fv: flags.FlagValues) -> "GoodputRecorder":
        """Converts flags to a recorder."""
        cfg: measurement_base.Recorder.Config = cls.default_config()
        expanded_specs = []
        for spec in fv.recorder_spec:
            for part in spec.split(","):
                if "=" in part or not expanded_specs:
                    expanded_specs.append(part)
                else:
                    expanded_specs[-1] = f"{expanded_specs[-1]},{part}"
        parsed_flags = parse_kv_flags(expanded_specs, delimiter="=")
        if "upload_interval" in parsed_flags:
            parsed_flags["upload_interval"] = int(parsed_flags["upload_interval"])
        if "step_deviation_interval_seconds" in parsed_flags:
            parsed_flags["step_deviation_interval_seconds"] = int(
                parsed_flags["step_deviation_interval_seconds"]
            )
        if "rolling_window_size" in parsed_flags and isinstance(
            parsed_flags["rolling_window_size"], str
        ):
            parsed_flags["rolling_window_size"] = [
                int(x) for x in parsed_flags["rolling_window_size"].split(",")
            ]
        if "enable_monitoring" in parsed_flags:
            parsed_flags["enable_monitoring"] = to_bool(parsed_flags["enable_monitoring"])
        rec_name = parsed_flags.get("name", "")
        if not rec_name or rec_name.endswith("_"):
            trainer_dir = getattr(fv, "trainer_dir", None) if fv is not None else None
            fallback_suffix = (
                os.path.basename(trainer_dir.rstrip("/"))
                if isinstance(trainer_dir, str) and trainer_dir.strip("/")
                else os.environ.get("HOSTNAME", "default")
            )
            new_name = f"{rec_name or 'goodput_'}{fallback_suffix}"
            logging.warning(
                "Goodput recorder name %r appears incomplete (e.g. unset $RUN); "
                "auto-resolving to %r to prevent cross-run log collisions.",
                rec_name,
                new_name,
            )
            parsed_flags["name"] = new_name
        return maybe_set_config(cfg, **parsed_flags).instantiate()

    def __init__(self, cfg):
        super().__init__(cfg)
        self._recorder: Optional[goodput.GoodputRecorder] = None
        self._monitor: Optional[goodput_monitoring.GoodputMonitor] = None
        self._rolling_window_monitor: Optional[goodput_monitoring.GoodputMonitor] = None
        self._monitoring_active: bool = False
        self._job_name = cfg.name
        self._logger_name = f"goodput_logger_{cfg.name}"

    def _get_or_create_recorder(self):
        if self._recorder is None:
            if jax.process_index() == 0:
                logging.info("Lazily instantiating goodput recorder.")
            recorder_cls = (
                goodput_elastic.ElasticGoodputRecorder
                if goodput_elastic is not None
                else goodput.GoodputRecorder
            )
            self._recorder = recorder_cls(
                job_name=self._job_name,
                logger_name=self._logger_name,
                logging_enabled=(jax.process_index() == 0),
            )
        return self._recorder

    def flush(self):
        """Flushes buffered Cloud Logging entries if a recorder is instantiated."""
        if self._recorder is not None and hasattr(self._recorder, "flush"):
            self._recorder.flush()

    @contextlib.contextmanager
    def record_event(self, event: measurement_base.EventType, *args, **kwargs):
        """Records a goodput event using a context manager."""
        recorder = self._get_or_create_recorder()

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

    @contextlib.contextmanager
    def _maybe_monitor_goodput(self, *args, **kwargs):
        """Monitor cumulative goodput if enabled."""
        if not self.config.enable_monitoring or jax.process_index() != 0:
            yield
            return
        try:
            if self._monitor is None:
                if monitoring_elastic is not None:
                    self._monitor = monitoring_elastic.ElasticGoodputMonitor(
                        job_name=self._job_name,
                        logger_name=self._logger_name,
                        tensorboard_dir=self.config.upload_dir,
                        upload_interval=self.config.upload_interval,
                        monitoring_enabled=True,
                        include_badput_breakdown=True,
                        include_step_deviation=True,
                        include_slice_efficiency=True,
                        step_deviation_interval_seconds=self.config.step_deviation_interval_seconds,
                        gcp_options=monitoring_elastic.GCPOptions(
                            enable_gcp_goodput_metrics=True,
                            enable_gcp_step_deviation_metrics=True,
                        ),
                    )
                else:
                    self._monitor = goodput_monitoring.GoodputMonitor(
                        job_name=self._job_name,
                        logger_name=self._logger_name,
                        tensorboard_dir=self.config.upload_dir,
                        upload_interval=self.config.upload_interval,
                        monitoring_enabled=True,
                        pathway_enabled=self.config.jax_backend == "proxy",
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
                self._rolling_window_monitor = goodput_monitoring.GoodputMonitor(
                    job_name=self._job_name,
                    logger_name=self._logger_name,
                    tensorboard_dir=rolling_window_tensorboard_dir,
                    upload_interval=self.config.upload_interval,
                    monitoring_enabled=True,
                    pathway_enabled=self.config.jax_backend == "proxy",
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

    @contextlib.contextmanager
    def maybe_monitor_all(self):
        if self._monitoring_active:
            yield
            return
        self._monitoring_active = True
        try:
            with self._maybe_monitor_goodput(), self._maybe_monitor_rolling_window_goodput():
                yield
        finally:
            self._monitoring_active = False

    def record(self, event: measurement_base.Event, *args, **kwargs):
        """Records a goodput event."""
        recorder = self._get_or_create_recorder()

        if event == measurement_base.Event.START_JOB:
            recorder.record_job_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_JOB:
            recorder.record_job_end_time(*args, **kwargs)
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
        elif event == measurement_base.Event.START_ELASTIC_WAIT and hasattr(
            recorder, "record_elastic_wait_start_time"
        ):
            recorder.record_elastic_wait_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_ELASTIC_WAIT and hasattr(
            recorder, "record_elastic_wait_end_time"
        ):
            recorder.record_elastic_wait_end_time(*args, **kwargs)
        elif event == measurement_base.Event.START_ELASTIC_REINIT and hasattr(
            recorder, "record_elastic_reinit_start_time"
        ):
            recorder.record_elastic_reinit_start_time(*args, **kwargs)
        elif event == measurement_base.Event.END_ELASTIC_REINIT and hasattr(
            recorder, "record_elastic_reinit_end_time"
        ):
            recorder.record_elastic_reinit_end_time(*args, **kwargs)
        elif event == measurement_base.Event.RECORD_SLICE_COUNTS and hasattr(
            recorder, "record_elastic_slice_counts"
        ):
            recorder.record_elastic_slice_counts(*args, **kwargs)
        else:
            logging.log_first_n(
                logging.WARNING,
                "Ignoring unknown event %s",
                1,
                event,
            )

    def start_monitoring(self, **kwargs):
        """Deprecated: `start_monitoring()` is not used in GoodputRecorder.
        Use the maybe_monitor_all context manager instead.
        """
        pass
