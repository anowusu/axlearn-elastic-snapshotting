# Copyright © 2025 Apple Inc.

"""Elastic training utilities for the Pathways backend (no-ops on McJAX)."""

from typing import Optional

from absl import logging
from pathwaysutils.elastic import elastic, manager

# The elastic manager of the `Snapshotter`, set when it is created on Pathways.
elastic_manager: Optional[manager.Manager] = None


def elastic_enabled() -> bool:
    return elastic_manager is not None


def available_slice_count() -> Optional[int]:
    """Returns the number of healthy slices, or None when not elastic or if the check fails."""
    if elastic_manager is None:
        return None
    try:
        return len(
            elastic.get_active_slice_indices(slice_to_devices=elastic_manager.slice_to_devices)
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to check the available slices: %s", e)
        return None
