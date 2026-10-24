# Copyright © 2025 Apple Inc.

"""Elastic training utilities for the Pathways backend (no-ops on McJAX)."""

import contextlib
import math
from typing import Any, Callable, Optional

import jax
from absl import logging

try:
    import pathwaysutils  # pytype: disable=import-error
    from pathwaysutils.elastic import elastic, manager  # pytype: disable=import-error

    ScaleUpSignalError = manager.ScaleUpSignalError
except ImportError:
    pathwaysutils = elastic = manager = None

    class ScaleUpSignalError(Exception):  # type: ignore[no-redef]
        """Raised to interrupt training so that it can resume on newly available slices."""


elastic_manager: Any = None


def _elastic_event_cleanup():
    """Deletes live device arrays on an elastic event while preserving host-resident snapshots."""
    for array in jax.live_arrays():
        sharding = getattr(array, "sharding", None)
        on_host = getattr(sharding, "memory_kind", None) == "pinned_host" or any(
            getattr(d, "platform", None) == "cpu" for d in getattr(sharding, "device_set", ())
        )
        if not on_host:
            with contextlib.suppress(Exception):
                array.delete()


def elastic_enabled() -> bool:
    return elastic_manager is not None


def ensure_elastic_manager_initialized(enabled: bool) -> Any:
    global elastic_manager  # pylint: disable=global-statement
    if enabled and elastic_manager is None:
        if manager is None or not pathwaysutils.is_pathways_backend_used():
            logging.warning("Elastic training requires the Pathways backend; running without it.")
        else:
            elastic_manager = manager.Manager()
            manager._elastic_event_cleanup = _elastic_event_cleanup  # pylint: disable=protected-access
    return elastic_manager


def live_devices() -> list[jax.Device]:
    if elastic_manager is None:
        return jax.devices()
    active = elastic_manager.active_slice_indices
    if slice_to_devices := getattr(elastic_manager, "slice_to_devices", None):
        return [d for s in sorted(active) for d in slice_to_devices.get(s, ())]
    return [d for d in jax.devices() if getattr(d, "slice_index", 0) in active]


def total_slice_count() -> Optional[int]:
    return None if elastic_manager is None else elastic_manager.total_slice_count


def wait_for_slices(slice_count: Optional[int] = None, *, poll_interval: float = 5):
    if elastic_manager is not None and elastic is not None:
        elastic_manager.active_slice_indices = elastic.wait_for_slices(
            slice_count=slice_count or elastic_manager.total_slice_count,
            slice_to_devices=elastic_manager.slice_to_devices, poll_interval=poll_interval,
        )


def grad_accumulation_steps(base_steps: int, *, device_batch_sizes: Optional[tuple[int, int]] = None) -> int:
    if elastic_manager is None:
        return base_steps
    full, live = device_batch_sizes or (elastic_manager.active_slice_count * max(1, base_steps), elastic_manager.total_slice_count * max(1, base_steps))
    steps = math.ceil(live / max(1, full // max(1, base_steps)))
    while live % steps:
        steps += 1
    return base_steps if steps <= max(1, base_steps) else steps


def is_scale_up_event() -> bool:
    if elastic_manager is None:
        return False
    event = getattr(elastic_manager, "new_slice_event", None)
    return bool(getattr(elastic_manager, "available_inactive_slices", None)) or (event is not None and event.is_set())


def elastic_retry(
    *, min_slice_count: int = -1, timeout_seconds: Optional[float] = None, max_retries: Optional[int] = None,
    poll_interval: float = 5, pre_callback: Optional[Callable[[], Any]] = None, on_elastic_event_callback: Optional[Callable[[], Any]] = None,
) -> Callable[[Callable], Callable]:
    if elastic_manager is None:
        return lambda fn: fn
    return elastic_manager.elastic_retry(
        max_retries=max_retries, timeout=timeout_seconds,
        minimum_slice_count=None if min_slice_count == -1 else min_slice_count,
        poll_interval=poll_interval, pre_callback=pre_callback, on_elastic_event_callback=on_elastic_event_callback,
    )
