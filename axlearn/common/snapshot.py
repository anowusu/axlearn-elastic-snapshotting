# Copyright © 2025 Apple Inc.

"""In-memory host-pinned snapshot checkpointer and Pathways pause-resume helpers."""

import contextlib
import queue
import threading
from typing import Any, Callable, Optional, Union

from absl import logging
import jax
import numpy as np

import pathwaysutils
from pathwaysutils.elastic import manager
from pathwaysutils.experimental import split_by_mesh_axis

from orbax.checkpoint import pathways as orbax_pathways

from axlearn.common.checkpointer import BaseCheckpointer, every_n_steps_policy
from axlearn.common.config import ConfigOr, config_class, config_for_function, maybe_instantiate
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, TensorSpec


def _free_arrays(tree: Any):
    for x in jax.tree.leaves(tree):
        if isinstance(x, jax.Array) and getattr(x.sharding, "memory_kind", None) == "pinned_host":
            with contextlib.suppress(Exception):
                x.delete()


def _to_pinned_host(tree: Any):
    return jax.tree.map(
        lambda x: jax.device_put(x, x.sharding.with_memory_kind("pinned_host"))
        if isinstance(x, jax.Array) and hasattr(x, "sharding") else x, tree)


class Snapshotter(BaseCheckpointer):
    """Manages asynchronous backups of JAX array states to pinned host memory."""

    @config_class
    class Config(BaseCheckpointer.Config):
        dir: Optional[str] = None
        save_policy: ConfigOr[Any] = config_for_function(every_n_steps_policy).set(n=5)
        timeout_seconds: int = 3600
        max_retries: int = 100

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._save_policy = maybe_instantiate(self.config.save_policy)
        self._latest_snapshot: Optional[tuple[Nested[Tensor], int]] = None
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=1)
        self._generation = 0
        self._manager = None
        if pathwaysutils.is_pathways_backend_used():
            self._manager = manager.Manager(wait_for_slices=True, poll_interval=5)
            with contextlib.suppress(Exception):
                orbax_pathways.register_type_handlers(
                    use_single_replica_array_handler=False,
                    primary_host=None,
                    checkpointing_impl=orbax_pathways.CheckpointingImpl.COLOCATED_PYTHON,
                    array_metadata_store=None,
                    enable_write_sharding_file=False,
                )
        threading.Thread(target=self._worker, name=f"{self.path()}_worker", daemon=True).start()

    def elastic_retry(self, **kwargs) -> Callable[[Callable], Callable]:
        if self._manager is None:
            return lambda fn: fn
        return self._manager.elastic_retry(
            max_retries=self.config.max_retries, timeout=self.config.timeout_seconds, **kwargs)

    @property
    def latest_snapshot_step(self) -> Optional[int]:
        with self._lock:
            return None if self._latest_snapshot is None else self._latest_snapshot[1]

    def _worker(self):
        while True:
            pinned_state, step, task_gen = self._queue.get()
            old = None
            try:
                if task_gen == self._generation:
                    jax.block_until_ready(pinned_state)
                    with self._lock:
                        if task_gen == self._generation:
                            old = self._latest_snapshot
                            self._latest_snapshot, pinned_state = (pinned_state, step), None
                    logging.info("[ELASTIC] Snapshot at step %d is ready.", step)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("[ELASTIC] Failed to secure snapshot at step %d: %s", step, type(e))
            finally:
                _free_arrays((old, pinned_state))
                self._queue.task_done()

    def _cancel_pending(self):
        with self._lock:
            self._generation += 1
        self._queue.join()

    def evict(self, *, free: bool = True):
        self._cancel_pending()
        with self._lock:
            old, self._latest_snapshot = self._latest_snapshot, None
        if free:
            _free_arrays(old)

    def save(
        self, *, step: int, state: Nested[Tensor], evaler_summaries: Optional[dict[str, Any]] = None,
        force: bool = False,
    ):
        if not force and not self._save_policy(step=step, evaler_summaries=evaler_summaries or {}):
            return
        self._cancel_pending()
        self._queue.put((_to_pinned_host(state), step, self._generation))
        if force:
            self.wait_until_finished()

    def _restore_replica(
        self,
        replica: jax.Array,
        sharding: jax.sharding.Sharding,
        global_shape: tuple[int, ...],
    ) -> Optional[jax.Array]:
        try:
            jax.block_until_ready(replica)
            shards_by_idx = {
                s.index: s.data for s in getattr(replica, "addressable_shards", ())
            }
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        if not shards_by_idx:
            return None
        if np.array_equal(replica.sharding.mesh.devices.flat, sharding.mesh.devices.flat):
            return jax.device_put(replica, sharding)
        regions = sharding.addressable_devices_indices_map(global_shape)
        local_ids = {d.id for d in replica.sharding.mesh.devices.flat}
        shards, cache = [], {}
        for d in sorted(sharding.addressable_devices, key=lambda d: d.id not in local_ids):
            dev_sharding = jax.sharding.SingleDeviceSharding(d).with_memory_kind("device")
            region = regions[d]
            shard = jax.device_put(cache.get(region, shards_by_idx[region]), dev_sharding)
            cache.setdefault(region, shard)
            shards.append(shard)
        return jax.make_array_from_single_device_arrays(global_shape, sharding, shards)

    def _restore_leaf(self, x: Any, spec: Any) -> Any:
        if not isinstance(x, jax.Array) or not hasattr(x.sharding, "mesh"):
            return x
        sharding = spec.sharding.with_memory_kind("device")
        has_data_axis = (
            pathwaysutils.is_pathways_backend_used() and x.sharding.mesh.shape.get("data", 1) > 1
        )
        replicas = split_by_mesh_axis.split_by_mesh_axis(x, "data") if has_data_axis else (x,)
        for replica in replicas:
            restored = self._restore_replica(replica, sharding, tuple(x.shape))
            if restored is not None:
                return restored
        raise ValueError("No healthy shards in snapshot")

    def restore(
        self, *, step: Optional[int] = None, state: Union[Nested[Tensor], Nested[TensorSpec]]
    ):
        self._cancel_pending()
        with self._lock:
            snap = self._latest_snapshot
            if not snap or step not in (None, snap[1]):
                return None, state
            pinned_state, snap_step = snap

        try:
            restored = jax.tree.map(self._restore_leaf, pinned_state, state)
            jax.block_until_ready(restored)
        except ValueError as e:
            logging.warning("[ELASTIC] Snapshot at step %d has no healthy shards: %s", snap_step, e)
            self.evict()
            return None, state
        logging.info(
            "[ELASTIC] Restored snapshot at step %d onto %d live devices",
            snap_step,
            len(jax.devices()),
        )
        return snap_step, restored

    def wait_until_finished(self):
        self._queue.join()

    def stop(self, *, has_exception: bool = False):
        if has_exception:
            self._cancel_pending()
        else:
            self.wait_until_finished()
