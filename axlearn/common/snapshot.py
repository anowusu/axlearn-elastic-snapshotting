# Copyright © 2025 Apple Inc.

"""In-memory host-pinned snapshot checkpointer for fast elastic recovery."""

import contextlib
import queue
import threading
from typing import Any, Optional, Union

from absl import logging
import jax
import jax.numpy as jnp
try:
    from pathwaysutils.experimental import split_by_mesh_axis  # pytype: disable=import-error
except ImportError:
    split_by_mesh_axis = None
import numpy as np

from axlearn.common import elastic_utils
from axlearn.common.checkpointer import BaseCheckpointer, every_n_steps_policy
from axlearn.common.config import ConfigOr, config_class, maybe_instantiate
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, TensorSpec

_HBM_GIB = {"v5e": 15.5, "v5 lite": 15.5, "v6e": 31.5, "v6 lite": 31.5, "v4": 31.5, "v5": 94.5, "7x": 190.0}


def _free_arrays(tree: Any, memory_kind: str = "pinned_host"):
    for x in jax.tree_util.tree_leaves(tree):
        if isinstance(x, jax.Array) and getattr(x.sharding, "memory_kind", None) == memory_kind:
            with contextlib.suppress(Exception):
                x.delete()


def _to_pinned_host(tree: Any):
    return jax.tree.map(
        lambda x: jax.device_put(x, x.sharding.with_memory_kind("pinned_host")) if isinstance(x, jax.Array) and hasattr(x, "sharding") else x,
        tree,
    )


def _hbm_limit_bytes(device: jax.Device) -> int:
    with contextlib.suppress(Exception):
        if limit := (device.memory_stats() or {}).get("bytes_limit"):
            return limit
    kind = getattr(device, "device_kind", "")
    return int(next((gib for k, gib in _HBM_GIB.items() if k in kind), 0) * 2**30)


class Snapshotter(BaseCheckpointer):
    """Manages asynchronous backups of JAX array states to pinned host memory."""

    @config_class
    class Config(BaseCheckpointer.Config):
        dir: Optional[str] = None
        save_policy: ConfigOr[Any] = every_n_steps_policy(5)
        stage_in_hbm: bool = False

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._save_policy = maybe_instantiate(self.config.save_policy)
        self._latest_snapshot: Optional[tuple[Nested[Tensor], int]] = None
        self._held_snapshot: Optional[tuple[Nested[Tensor], int]] = None
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._generation = 0
        self._stage_in_hbm = cfg.stage_in_hbm
        self._copy_fn = jax.jit(lambda xs: [jnp.copy(x) for x in xs])
        threading.Thread(target=self._worker, name=f"{self.path()}_worker", daemon=True).start()

    @property
    def latest_snapshot_step(self) -> Optional[int]:
        with self._lock:
            return None if self._latest_snapshot is None else self._latest_snapshot[1]

    @property
    def latest_snapshot_state(self) -> Optional[Nested[Tensor]]:
        with self._lock:
            return None if self._latest_snapshot is None else self._latest_snapshot[0]

    def _worker(self):
        while True:
            src, step, task_gen, staged, hold_prev = self._queue.get()
            pinned_state, to_free = None if staged else src, []
            try:
                if task_gen == self._generation:
                    if staged:
                        jax.block_until_ready(src)
                        pinned_state = _to_pinned_host(src)
                    jax.block_until_ready(pinned_state)
                    with self._lock:
                        if task_gen == self._generation:
                            if hold_prev:
                                if self._held_snapshot is None and self._latest_snapshot is not None:
                                    self._held_snapshot = self._latest_snapshot
                                elif self._latest_snapshot is not None:
                                    to_free.append(self._latest_snapshot)
                            else:
                                to_free.extend([self._held_snapshot, self._latest_snapshot])
                                self._held_snapshot = None
                            self._latest_snapshot, pinned_state = (pinned_state, step), None
                    logging.info("[ELASTIC] Snapshot at step %d is ready.", step)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("[ELASTIC] Failed to secure snapshot at step %d: %s", step, type(e))
            finally:
                if staged:
                    _free_arrays(src, memory_kind="device")
                _free_arrays([to_free, pinned_state])
                self._queue.task_done()

    def evict(self, *, free: bool = True):
        self._cancel_pending()
        with self._lock:
            old, self._latest_snapshot, self._held_snapshot = (self._latest_snapshot, self._held_snapshot), None, None
        if free:
            _free_arrays(old)

    def _stage(self, state: Nested[Tensor]) -> Nested[Tensor]:
        copies = iter(self._copy_fn([x for x in jax.tree.leaves(state) if isinstance(x, jax.Array)]))
        return jax.tree.map(lambda x: next(copies) if isinstance(x, jax.Array) else x, state)

    def save(self, *, step: int, state: Nested[Tensor], evaler_summaries: Optional[dict[str, Any]] = None, force: bool = False, hold_previous: bool = False):
        if not force and not self._save_policy(step=step, evaler_summaries=evaler_summaries or {}):
            return
        self._cancel_pending()
        src = self._stage(state) if self._stage_in_hbm else _to_pinned_host(state)
        self._queue.put((src, step, self._generation, self._stage_in_hbm, hold_previous))
        if force:
            self.wait_until_finished()

    def check_hbm_headroom(self, compiled: Optional[jax.stages.Compiled], *, device: jax.Device):
        self._stage_in_hbm = self.config.stage_in_hbm
        mem = getattr(compiled, "memory_analysis", lambda: None)()
        if not self._stage_in_hbm or mem is None:
            return
        required = mem.argument_size_in_bytes + 2 * mem.output_size_in_bytes - mem.alias_size_in_bytes + mem.temp_size_in_bytes + mem.generated_code_size_in_bytes
        if 0 < (limit := _hbm_limit_bytes(device)) < required:
            logging.warning("[ELASTIC] Disabling stage_in_hbm: needs %.1f of %.1f GiB HBM.", required / 2**30, limit / 2**30)
            self._stage_in_hbm = False

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
            replica, src_by_region = x, {}
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
                with contextlib.suppress(Exception):
                    for s in getattr(x, "addressable_shards", ()):
                        if s.device.id in live_ids:
                            src_by_region.setdefault(s.index, s.data)
            if not src_by_region:
                raise ValueError("No healthy shards in snapshot")
            if np.array_equal(replica.sharding.mesh.devices.flat, sharding.mesh.devices.flat):
                return jax.device_put(replica, sharding)
            regions = sharding.addressable_devices_indices_map(tuple(x.shape))
            local_ids = {d.id for d in replica.sharding.mesh.devices.flat}
            shards, cache = [], {}
            for dev in sorted(sharding.addressable_devices, key=lambda d: d.id not in local_ids):
                key = regions[dev]
                dev_s = jax.sharding.SingleDeviceSharding(dev).with_memory_kind("device")
                shard = jax.device_put(cache.get(key, src_by_region[key]), dev_s)
                cache.setdefault(key, shard)
                shards.append(shard)
            return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)

        try:
            restored = jax.tree.map(restore_leaf, pinned_state, state)
            jax.block_until_ready(restored)
        except (ValueError, jax.errors.JaxRuntimeError) as e:
            if isinstance(e, jax.errors.JaxRuntimeError) and "Full data loss while cloning" not in str(e):
                raise
            logging.warning("[ELASTIC] Snapshot at step %d has no healthy shards: %s", snap_step, e)
            self.evict()
            return None, state
        logging.info("[ELASTIC] [POLICY_A] Restored snapshot at step %d onto %d live devices", snap_step, len(elastic_utils.live_devices()))
        return snap_step, restored

    def wait_until_finished(self):
        self._queue.join()

    def stop(self, *, has_exception: bool = False):
        (self._cancel_pending if has_exception else self.wait_until_finished)()
