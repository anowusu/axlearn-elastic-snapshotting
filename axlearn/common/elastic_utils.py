# Copyright © 2025 Apple Inc.

"""Elastic training utilities for the Pathways backend (no-ops on McJAX)."""

from typing import Any, Optional

from absl import logging
from pathwaysutils.elastic import elastic, manager

# Optional explicit manager reference; falls back to `manager.get_default_manager()` when unset.
elastic_manager: Optional[manager.Manager] = None


def get_manager() -> Any:
    if elastic_manager is not None:
        return elastic_manager
    return getattr(manager, "get_default_manager", lambda: None)()


def elastic_enabled() -> bool:
    return get_manager() is not None


def available_slice_count() -> Optional[int]:
    """Returns the number of healthy slices, or None when not elastic or if the check fails."""
    mgr = get_manager()
    if mgr is None:
        return None
    try:
        return len(elastic.get_active_slice_indices(slice_to_devices=mgr.slice_to_devices))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to check the available slices: %s", e)
        return None
