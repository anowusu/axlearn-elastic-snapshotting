# Copyright © 2025 Apple Inc.

"""In-memory host-pinned snapshot checkpointer for fast elastic recovery."""

import contextlib
import gc
import queue
import threading
from typing import Any, Optional, Union

from absl import logging
import jax
try:
    from pathwaysutils.experimental import split_by_mesh_axis  # pytype: disable=import-error
except ImportError:
    split_by_mesh_axis = None
import numpy as np

from axlearn.common.checkpointer import BaseCheckpointer, every_n_steps_policy
from axlearn.common.config import ConfigOr, config_class, maybe_instantiate
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, TensorSpec


def _free_pinned_host_arrays(tree: Any):
    for x in jax.tree_util.tree_leaves(tree):
        if isinstance(x, jax.Array) and getattr(x.sharding, "memory_kind", None) == "pinned_host":
            with contextlib.suppress(Exception):
                x.delete()


class Snapshotter(BaseCheckpointer):
    """Manages asynchronous backups of JAX array states to pinned host memory."""

    @config_class
    class Config(BaseCheckpointer.Config):
        dir: Optional[str] = None
        save_policy: ConfigOr[Any] = every_n_steps_policy(5)

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._save_policy = maybe_instantiate(self.config.save_policy)
        self._latest_snapshot: Optional[tuple[Nested[Tensor], int]] = None
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._generation = 0
        threading.Thread(target=self._worker, name=f"{self.path()}_worker", daemon=True).start()

    @property
    def latest_snapshot_step(self) -> Optional[int]:
        with self._lock:
            return None if self._latest_snapshot is None else self._latest_snapshot[1]

    def _worker(self):
        while True:
            pinned_state, step, task_gen = self._queue.get()
            try:
                if task_gen == self._generation:
                    jax.block_until_ready(pinned_state)
                    with self._lock:
                        if task_gen == self._generation:
                            self._latest_snapshot, pinned_state = (pinned_state, step), None
                    logging.info("[ELASTIC] Snapshot at step %d is ready.", step)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("[ELASTIC] Failed to secure snapshot at step %d: %s", step, type(e))
            finally:
                _free_pinned_host_arrays(pinned_state)
                self._queue.task_done()

    def evict(self):
        self._cancel_pending()
        with self._lock:
            old, self._latest_snapshot = self._latest_snapshot, None
        _free_pinned_host_arrays(old)
        gc.collect()

    def save(self, *, step: int, state: Nested[Tensor], evaler_summaries: Optional[dict[str, Any]] = None, force: bool = False):
        if not force and not self._save_policy(step=step, evaler_summaries=evaler_summaries or {}):
            return
        self.evict()
        pinned_state = jax.tree.map(
            lambda x: jax.device_put(x, x.sharding.with_memory_kind("pinned_host"))
            if isinstance(x, jax.Array) and hasattr(x, "sharding")
            else x,
            state,
        )
        self._queue.put((pinned_state, step, self._generation))
        if force:
            self.wait_until_finished()

    def _cancel_pending(self):
        with self._lock:
            self._generation += 1
        self._queue.join()

    def restore(self, *, step: Optional[int] = None, state: Union[Nested[Tensor], Nested[TensorSpec]]) -> tuple[Optional[int], Nested[Tensor]]:
        self._cancel_pending()
        with self._lock:
            if self._latest_snapshot is None or (step is not None and step != self._latest_snapshot[1]):
                return None, state
            pinned_state, snap_step = self._latest_snapshot

        def restore_leaf(x, spec):
            if not isinstance(x, jax.Array) or not hasattr(x.sharding, "mesh"):
                return x
            sharding = spec.sharding.with_memory_kind("device")
            live_ids = {d.id for d in sharding.mesh.devices.flat}
            replica, src_by_region = None, {}
            with contextlib.suppress(Exception):
                for r in (
                    split_by_mesh_axis.split_by_mesh_axis(x, "data")
                    if x.sharding.mesh.shape.get("data", 1) > 1 and split_by_mesh_axis is not None
                    else (x,)
                ):
                    if {d.id for d in r.sharding.mesh.devices.flat}.issubset(live_ids):
                        with contextlib.suppress(jax.errors.JaxRuntimeError):
                            jax.block_until_ready(r)
                            for s in getattr(r, "addressable_shards", ()):
                                src_by_region.setdefault(s.index, s.data)
                            if src_by_region:
                                replica = r
                                break
            if not src_by_region:
                raise ValueError("No healthy shards in snapshot")
            if np.array_equal(replica.sharding.mesh.devices.flat, sharding.mesh.devices.flat):
                return jax.device_put(replica, sharding)
            regions = sharding.addressable_devices_indices_map(tuple(spec.shape))
            shards, cache = [], {}
            for dev in sharding.addressable_devices:
                key = regions[dev]
                dev_s = jax.sharding.SingleDeviceSharding(dev).with_memory_kind("device")
                shard = jax.device_put(cache.get(key, src_by_region[key]), dev_s)
                cache.setdefault(key, shard)
                shards.append(shard)
            return jax.make_array_from_single_device_arrays(spec.shape, sharding, shards)

        try:
            restored = jax.tree.map(restore_leaf, pinned_state, state)
            jax.block_until_ready(restored)
        except (ValueError, jax.errors.JaxRuntimeError) as e:
            if isinstance(e, jax.errors.JaxRuntimeError) and "Full data loss while cloning" not in str(e):
                raise
            logging.warning("[ELASTIC] Snapshot at step %d has no healthy shards: %s", snap_step, e)
            self.evict()
            return None, state
        return snap_step, restored

    def wait_until_finished(self):
        self._queue.join()

    def stop(self, *, has_exception: bool = False):
        (self._cancel_pending if has_exception else self.wait_until_finished)()
