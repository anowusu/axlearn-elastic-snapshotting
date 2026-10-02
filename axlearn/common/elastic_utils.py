# Copyright © 2025 Apple Inc.

"""Elastic training utilities for the Pathways backend.

This module mirrors `maxtext.utils.elastic_utils` (same concepts, config names and
`pathwaysutils` entry points) so that AXLearn and MaxText elastic training behave alike:

* `elastic_manager`: a module-level `pathwaysutils.elastic.manager.Manager`.
* `live_devices()`: the devices of the currently active slices (`jax.devices()` when not elastic).
* `elastic_retry()`: `Manager.elastic_retry` driven by MaxText's `elastic_*` parameters.
* `is_scale_up_event()` / `ScaleUpSignalError`: cooperative scale-up to newly available slices.

Elastic training is Pathways-specific. On McJAX, or when `pathwaysutils` is not installed,
`elastic_enabled()` is False and every helper degrades to a no-op, so a single trainer config can
be launched on either backend (on McJAX a slice loss is handled by the usual job restart and
checkpoint restore).

Batch size: MaxText keeps the per-device batch fixed, so its global batch shrinks with the number
of live slices. AXLearn keeps the global batch fixed and instead scales the number of gradient
accumulation steps by `total_slices / active_slices` (`grad_accumulation_steps()`), so the
optimizer sees identical batches regardless of how many slices are active.
"""

from typing import Any, Callable, Optional

import jax
from absl import logging

try:
    import pathwaysutils  # pytype: disable=import-error
    from pathwaysutils.elastic import elastic, manager  # pytype: disable=import-error

    ScaleUpSignalError = manager.ScaleUpSignalError
except ImportError:  # `pathwaysutils` is only installed with the `tpu` extra.
    pathwaysutils = elastic = manager = None

    class ScaleUpSignalError(Exception):  # type: ignore[no-redef]
        """Raised to interrupt training so that it can resume on newly available slices."""


# The elastic manager, created by `ensure_elastic_manager_initialized`.
elastic_manager: Any = None


def _elastic_event_cleanup():
    """Replaces `manager._elastic_event_cleanup`, which deletes *every* live array.

    Host-resident arrays are kept alive: they hold the in-memory snapshots (`snapshot.Snapshotter`)
    that make recovery from an elastic event fast. JAX caches are also kept so that an unchanged
    mesh can reuse its compiled train step.
    """
    for array in jax.live_arrays():
        sharding = getattr(array, "sharding", None)
        on_host = getattr(sharding, "memory_kind", None) == "pinned_host" or any(
            getattr(d, "platform", None) == "cpu" for d in getattr(sharding, "device_set", ())
        )
        if not on_host:
            try:
                array.delete()
            except Exception:  # pylint: disable=broad-exception-caught
                pass


def elastic_enabled() -> bool:
    """Returns whether elastic training is active, i.e. an elastic manager exists."""
    return elastic_manager is not None


def ensure_elastic_manager_initialized(enabled: bool) -> Any:
    """Creates the module-level manager if `enabled` and the Pathways backend is used."""
    global elastic_manager  # pylint: disable=global-statement
    if enabled and elastic_manager is None:
        if manager is None or not pathwaysutils.is_pathways_backend_used():
            logging.warning("Elastic training requires the Pathways backend; running without it.")
        else:
            elastic_manager = manager.Manager()
            # pylint: disable-next=protected-access
            manager._elastic_event_cleanup = _elastic_event_cleanup
    return elastic_manager


def live_devices() -> list[jax.Device]:
    """Returns the devices of the active slices (all devices when not elastic)."""
    if elastic_manager is None:
        return jax.devices()
    active = elastic_manager.active_slice_indices
    return [d for d in jax.devices() if getattr(d, "slice_index", 0) in active]


def wait_for_slices(slice_count: Optional[int] = None, *, poll_interval: float = 5):
    """Blocks until at least `slice_count` (default: all) slices are active."""
    if elastic_manager is not None:
        elastic_manager.active_slice_indices = elastic.wait_for_slices(
            slice_count=slice_count or elastic_manager.total_slice_count,
            slice_to_devices=elastic_manager.slice_to_devices,
            poll_interval=poll_interval,
        )


def grad_accumulation_steps(base_steps: int) -> int:
    """Returns the gradient accumulation steps keeping the global batch fixed on the active slices.

    Args:
        base_steps: The configured number of minibatch steps (0 if gradient accumulation is off).
    """
    if elastic_manager is None:
        return base_steps
    total, active = elastic_manager.total_slice_count, elastic_manager.active_slice_count
    return base_steps if active >= total else max(1, base_steps) * -(-total // active)


def is_scale_up_event() -> bool:
    """Returns whether an inactive slice became available while training on fewer slices."""
    if elastic_manager is None:
        return False
    event = getattr(elastic_manager, "new_slice_event", None)  # pathwaysutils < 0.1.12.
    return bool(getattr(elastic_manager, "available_inactive_slices", None)) or (
        event is not None and event.is_set()
    )


def elastic_retry(
    *,
    min_slice_count: int = -1,
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    poll_interval: float = 5,
    pre_callback: Optional[Callable[[], Any]] = None,
    on_elastic_event_callback: Optional[Callable[[], Any]] = None,
) -> Callable[[Callable], Callable]:
    """Returns a decorator that retries a function across slice-down and scale-up events.

    The parameters follow MaxText's `elastic_min_slice_count` (-1 waits for all slices, i.e.
    pause and resume), `elastic_timeout_seconds` and `elastic_max_retries` (None: unlimited).
    `pre_callback` runs before every attempt, once the slices are active;
    `on_elastic_event_callback` runs right after an elastic event. Identity when not elastic.
    """
    if elastic_manager is None:
        return lambda fn: fn
    return elastic_manager.elastic_retry(
        max_retries=max_retries,
        timeout=timeout_seconds,
        minimum_slice_count=None if min_slice_count == -1 else min_slice_count,
        poll_interval=poll_interval,
        pre_callback=pre_callback,
        on_elastic_event_callback=on_elastic_event_callback,
    )
