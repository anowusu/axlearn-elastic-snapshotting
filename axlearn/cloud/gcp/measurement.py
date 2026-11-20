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
import datetime
import json
import os
from typing import Any, Optional, Sequence

import jax
from absl import flags, logging
from ml_goodput_measurement import monitoring as goodput_monitoring
from ml_goodput_measurement.src import (
    exclusion_utils,
    goodput,
    goodput_elastic,
    goodput_utils,
    monitoring_elastic,
)
from tensorboardX import writer as tbx_writer

from axlearn.cloud.common.utils import parse_kv_flags, to_bool
from axlearn.common import measurement_base
from axlearn.common.config import REQUIRED, Required, config_class, maybe_set_config

_JOB_START_TIME = "job_start_time"
_JOB_END_TIME = "job_end_time"
_JOB_PREEMPTED_TIME = "job_preempted_time"
_STEP_START_TIME = "step_start_time"
_STEP_COUNT = "step_count"
_DEFAULT_GCS_SYNC_INTERVAL_SECONDS = 120


def _entry_timestamp(entry: dict[str, Any]) -> Optional[float]:
    for key, val in entry.items():
        if isinstance(val, (int, float)) and (key.endswith("_time") or key.endswith("_timestamp")):
            return float(val)
    return None


def _extract_preemption_exclusions(
    entries: Sequence[dict[str, Any]], query_ts: Optional[float] = None
) -> list[tuple[float, float]]:
    """Extracts normalized unscheduled [preemption_ts, next_job_start_ts] exclusion windows."""
    raw: list[tuple[datetime.datetime, datetime.datetime]] = []
    utc = datetime.timezone.utc
    seen_start = False
    pending_preempt: Optional[float] = None
    last_ts: Optional[float] = None
    for entry in entries:
        if _JOB_START_TIME in entry:
            ts = float(entry[_JOB_START_TIME])
            if seen_start:
                gap_start = pending_preempt if pending_preempt is not None else last_ts
                if gap_start is not None and ts > gap_start:
                    raw.append(
                        (
                            datetime.datetime.fromtimestamp(gap_start, tz=utc),
                            datetime.datetime.fromtimestamp(ts, tz=utc),
                        )
                    )
            seen_start = True
            pending_preempt = None
            last_ts = ts
        elif _JOB_PREEMPTED_TIME in entry:
            pending_preempt = last_ts = float(entry[_JOB_PREEMPTED_TIME])
        elif (ts := _entry_timestamp(entry)) is not None:
            last_ts = max(last_ts, ts) if last_ts is not None else ts
    if pending_preempt is not None and query_ts is not None and query_ts > pending_preempt:
        raw.append(
            (
                datetime.datetime.fromtimestamp(pending_preempt, tz=utc),
                datetime.datetime.fromtimestamp(query_ts, tz=utc),
            )
        )
    return exclusion_utils.normalize_intervals(raw)


def _excluded_infra_recovery(
    entries: Sequence[dict[str, Any]], exclusions: list[tuple[float, float]]
) -> float:
    """Computes overlap between exclusions and GoodputCalculator's disruption->start gaps."""
    if not exclusions:
        return 0.0
    step_starts: dict[int, float] = {}
    job_start: Optional[float] = None
    excluded = 0.0
    for entry in entries:
        if _JOB_START_TIME in entry:
            job_start = float(entry[_JOB_START_TIME])
        if _STEP_START_TIME in entry:
            step = int(entry[_STEP_COUNT])
            ts = float(entry[_STEP_START_TIME])
            if step in step_starts:
                last_disrupted_ts = list(step_starts.values())[-1]
                if job_start is not None and job_start > last_disrupted_ts:
                    excluded += exclusion_utils.get_exclusion_overlap(
                        last_disrupted_ts, job_start, exclusions
                    )
                step_starts = {step: ts}
            else:
                step_starts[step] = ts
    return excluded


def _compute_efficiency_with_exclusions(
    slice_records: list[tuple[float, int, int, int]],
    interval_start_ts: float,
    interval_end_ts: float,
    exclusions: list[tuple[float, float]],
) -> tuple[float, float]:
    """Computes time-weighted slice efficiency excluding unscheduled preemption gaps."""
    if not slice_records or interval_end_ts <= interval_start_ts:
        return 0.0, 0.0
    initial = None
    for rec in slice_records:
        if rec[0] <= interval_start_ts:
            initial = rec
    timeline: list[tuple[float, int, int, int]] = []
    if initial is not None:
        timeline.append((interval_start_ts, initial[1], initial[2], initial[3]))
    for rec in slice_records:
        if interval_start_ts < rec[0] < interval_end_ts:
            timeline.append(rec)
    if not timeline:
        return 0.0, 0.0
    scheduled_duration = 0.0
    weighted_stepping = weighted_available = 0.0
    for idx, (ts, stepping, available, total) in enumerate(timeline):
        next_ts = timeline[idx + 1][0] if idx + 1 < len(timeline) else interval_end_ts
        excl_dur = exclusion_utils.get_exclusion_overlap(ts, next_ts, exclusions)
        adj_dur = max(0.0, (next_ts - ts) - excl_dur)
        scheduled_duration += adj_dur
        weighted_stepping += (stepping / total) * adj_dur
        weighted_available += (available / total) * adj_dur
    if scheduled_duration <= 0.0:
        return 0.0, 0.0
    return weighted_stepping / scheduled_duration, weighted_available / scheduled_duration


class _PreemptionGoodputMixin:
    """Adjusts Goodput across Kueue/VM preemptions by excluding unscheduled queue-wait gaps."""

    _goodput_cache: Any
    _goodput_cache_lock: Any
    _latest_exclusions: list[tuple[float, float]] = []
    _latest_excluded_time: float = 0.0

    def _sanitize_superseded_job_end(self) -> list[dict[str, Any]]:
        with self._goodput_cache_lock:
            entries = list(self._goodput_cache.get_cached_entries())
            latest_start = max(
                (float(e[_JOB_START_TIME]) for e in entries if _JOB_START_TIME in e),
                default=None,
            )
            if latest_start is None:
                return entries
            end_dt = self._goodput_cache.get_job_end_time()
            has_stale = any(
                _JOB_END_TIME in e and float(e[_JOB_END_TIME]) < latest_start for e in entries
            )
            if has_stale or (end_dt is not None and end_dt.timestamp() < latest_start):
                cleaned = [
                    {k: v for k, v in e.items() if k != _JOB_END_TIME}
                    if _JOB_END_TIME in e and float(e[_JOB_END_TIME]) < latest_start
                    else e
                    for e in entries
                ]
                cleaned = [e for e in cleaned if set(e.keys()) - {"job_name"}]
                path = getattr(self._goodput_cache, "_local_timeline_path", None)
                if path and os.path.exists(path):
                    with open(path, "w", encoding="utf-8") as fh:
                        for item in cleaned:
                            fh.write(json.dumps(item) + "\n")
                self._goodput_cache._job_end_time = None
                for item in cleaned:
                    if _JOB_END_TIME in item:
                        self._goodput_cache._job_end_time = datetime.datetime.fromtimestamp(
                            float(item[_JOB_END_TIME]), tz=datetime.timezone.utc
                        )
                return cleaned
            return entries

    def _fetch_new_entries(self, query_time: datetime.datetime):
        new_entries = super()._fetch_new_entries(query_time)  # type: ignore[misc]
        entries = self._sanitize_superseded_job_end()
        self._latest_exclusions = _extract_preemption_exclusions(entries, query_time.timestamp())
        return new_entries

    def _get_total_job_time(self, query_time: datetime.datetime) -> float:
        raw_total = super()._get_total_job_time(query_time)  # type: ignore[misc]
        with self._goodput_cache_lock:
            start_dt = self._goodput_cache.get_job_start_time()
        if start_dt is None or not self._latest_exclusions or raw_total <= 0.0:
            self._latest_excluded_time = 0.0
            return raw_total
        start_ts = start_dt.timestamp()
        excl = exclusion_utils.get_exclusion_overlap(
            start_ts, start_ts + raw_total, self._latest_exclusions
        )
        self._latest_excluded_time = excl
        return max(0.0, raw_total - excl)

    def _get_current_productive_and_unproductive_time(self, interval_query: Optional[bool] = False):
        sup = super()
        prod, unprod, max_step, last_step = sup._get_current_productive_and_unproductive_time(
            interval_query
        )
        if not interval_query and self._latest_exclusions:
            with self._goodput_cache_lock:
                entries = list(self._goodput_cache.get_cached_entries())
            excl_infra = _excluded_infra_recovery(entries, self._latest_exclusions)
            bt_infra = goodput_utils.BadputType.INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION
            if excl_infra > 0.0 and bt_infra in unprod:
                unprod[bt_infra] = max(0.0, float(unprod[bt_infra]) - excl_infra)
        return prod, unprod, max_step, last_step

    def get_job_goodput(self, *args, **kwargs):
        res = super().get_job_goodput(*args, **kwargs)  # type: ignore[misc]
        with self._goodput_cache_lock:
            info = self._goodput_cache.get_goodput_info()
            if info is not None:
                info.total_excluded_time = self._latest_excluded_time
        return res

    def get_job_goodput_details(self):
        details = super().get_job_goodput_details()  # type: ignore[misc]
        with self._goodput_cache_lock:
            info = self._goodput_cache.get_goodput_info()
            entries = list(self._goodput_cache.get_cached_entries())
            start_dt = self._goodput_cache.get_job_start_time()
        if info is not None:
            details[goodput_utils.MetricType.TOTAL_EXCLUDED_TIME.value] = info.total_excluded_time
        has_eff = "stepping_slice_efficiency" in details
        if start_dt and entries and self._latest_exclusions and has_eff:
            slice_records = goodput_elastic.ElasticGoodputCalculator._extract_slice_count_entries(
                entries
            )
            if slice_records:
                raw_span = (
                    details.get(goodput_utils.MetricType.TOTAL_ELAPSED_TIME.value, 0.0)
                    + self._latest_excluded_time
                )
                s_eff, a_eff = _compute_efficiency_with_exclusions(
                    slice_records,
                    start_dt.timestamp(),
                    start_dt.timestamp() + raw_span,
                    self._latest_exclusions,
                )
                details["stepping_slice_efficiency"] = s_eff
                details["available_slice_efficiency"] = a_eff
        return details


class PreemptionAwareGoodputCalculator(_PreemptionGoodputMixin, goodput.GoodputCalculator):
    """GoodputCalculator that excludes unscheduled preemption queue-wait gaps."""


class PreemptionAwareElasticGoodputCalculator(
    _PreemptionGoodputMixin, goodput_elastic.ElasticGoodputCalculator
):
    """ElasticGoodputCalculator that excludes unscheduled preemption queue-wait gaps."""


def _configure_preemption_aware_monitor(monitor: Any, *, elastic: bool) -> Any:
    """Wires preemption-aware calculator, faster GCS cache sync, and safe parent writer cleanup."""
    worker_cfg = getattr(monitor, "_worker_config", None)
    if not isinstance(worker_cfg, dict):
        return monitor
    calc_cls = (
        PreemptionAwareElasticGoodputCalculator if elastic else PreemptionAwareGoodputCalculator
    )
    worker_cfg["calculator_class"] = calc_cls
    if (
        worker_cfg.get("gcs_sync_interval_seconds") == 3600
        and "ML_GOODPUT_GCS_SYNC_INTERVAL_SECONDS" not in os.environ
    ):
        worker_cfg["gcs_sync_interval_seconds"] = _DEFAULT_GCS_SYNC_INTERVAL_SECONDS
    monitor._goodput_calculator = calc_cls(
        job_name=worker_cfg["job_name"],
        logger_name=worker_cfg["logger_name"],
        using_pathways=True if elastic else worker_cfg.get("pathway_enabled", False),
        gcs_path=worker_cfg.get("tensorboard_dir"),
        cache_dir=worker_cfg.get("cache_dir", "/tmp"),
        max_logs_retention_period=worker_cfg.get("max_logs_retention_period"),
    )
    # Close unused parent SummaryWriter without flushing empty buffer over worker's GCS object.
    parent_writer = getattr(monitor, "_writer", None)
    if isinstance(parent_writer, tbx_writer.SummaryWriter):
        for fw in (parent_writer.all_writers or {}).values():
            ev_w = getattr(getattr(fw, "event_writer", None), "_ev_writer", None)
            if hasattr(ev_w, "flush"):
                ev_w.flush = lambda: None
        parent_writer.close()
    return monitor


class _SafeLogger:
    """Wraps an Orbax logger so that failing to log never fails a checkpoint save or restore."""

    def __init__(self, logger: Any):
        self._logger = logger

    def log_entry(self, entry: Any, *args, **kwargs):
        try:
            self._logger.log_entry(entry, *args, **kwargs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Failed to log checkpoint statistics: %s", e)


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
        return _SafeLogger(checkpoint_logger)

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
                return _configure_preemption_aware_monitor(monitor, elastic=True)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning(
                    "Failed to create the elastic goodput monitor, using the base monitor: %s", e
                )
            # Elastic training runs on Pathways.
            pathway_enabled = True
        monitor = goodput_monitoring.GoodputMonitor(pathway_enabled=pathway_enabled, **kwargs)
        logging.info("Goodput: using GoodputMonitor for %s.", tensorboard_dir)
        return _configure_preemption_aware_monitor(monitor, elastic=False)

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

    def record_preemption(self, timestamp: Optional[datetime.datetime] = None) -> None:
        """Records a preemption/eviction start timestamp and flushes goodput logs."""
        try:
            if jax.process_index() != 0:
                return
            # pylint: disable-next=import-outside-toplevel
            from axlearn.common import elastic_utils

            mgr = elastic_utils.get_manager()
            if mgr is not None:
                self._record(
                    measurement_base.Event.RECORD_SLICE_COUNTS, 0, mgr.total_slice_count, 0
                )
            recorder = self._get_recorder()
            cloud_logger = getattr(recorder, "_cloud_logger", None)
            if cloud_logger is not None:
                ts = timestamp or datetime.datetime.now(datetime.timezone.utc)
                cloud_logger.write_cloud_logging_entry(
                    {"job_name": self._job_name, _JOB_PREEMPTED_TIME: ts.timestamp()}
                )
            self.flush()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Failed to record goodput preemption: %s", e)

    def start_monitoring(self, **kwargs):
        """Deprecated: `start_monitoring()` is not used in GoodputRecorder.
        Use the maybe_monitor_all context manager instead.
        """
        pass
