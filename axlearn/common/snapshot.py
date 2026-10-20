# Copyright © 2025 Apple Inc.
"""In-memory host-pinned snapshot checkpointer and Pathways pause-resume helpers."""

import contextlib, queue, threading
from typing import Any, Callable, Optional, Union
from absl import logging
import jax, jax.numpy as jnp, numpy as np

try:
    import pathwaysutils  # pytype: disable=import-error
    from pathwaysutils.elastic import elastic, manager  # pytype: disable=import-error
    from pathwaysutils.experimental import split_by_mesh_axis  # pytype: disable=import-error
except ImportError:
    pathwaysutils = elastic = manager = split_by_mesh_axis = None

from axlearn.common.checkpointer import BaseCheckpointer, every_n_steps_policy
from axlearn.common.config import ConfigOr, config_class, config_for_function, maybe_instantiate
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, TensorSpec

_HBM_GIB = {"v5e": 15.5, "v5 lite": 15.5, "v6e": 31.5, "v6 lite": 31.5, "v4": 31.5, "v5": 94.5, "7x": 191}
elastic_manager: Any = None

def _elastic_event_cleanup():
    for a in jax.live_arrays():
        s = getattr(a, "sharding", None)
        if getattr(s, "memory_kind", None) != "pinned_host" and not any(
            getattr(d, "platform", None) == "cpu" for d in getattr(s, "device_set", ())):
            with contextlib.suppress(Exception):
                a.delete()

def ensure_elastic_manager_initialized() -> Any:
    global elastic_manager  # pylint: disable=global-statement
    if elastic_manager is None and manager is not None and pathwaysutils.is_pathways_backend_used():
        elastic_manager = manager.Manager()
        manager._elastic_event_cleanup = _elastic_event_cleanup  # pylint: disable=protected-access
    return elastic_manager

def wait_for_slices(poll_interval: float = 5):
    if ensure_elastic_manager_initialized() is not None and elastic is not None:
        elastic_manager.active_slice_indices = elastic.wait_for_slices(
            slice_count=elastic_manager.total_slice_count,
            slice_to_devices=elastic_manager.slice_to_devices, poll_interval=poll_interval)

def elastic_retry(**kwargs) -> Callable[[Callable], Callable]:
    return elastic_manager.elastic_retry(**kwargs) if elastic_manager is not None else lambda fn: fn

def _free_arrays(tree: Any, memory_kind: str = "pinned_host"):
    for x in jax.tree.leaves(tree):
        if isinstance(x, jax.Array) and getattr(x.sharding, "memory_kind", None) == memory_kind:
            with contextlib.suppress(Exception):
                x.delete()

def _to_pinned_host(tree: Any):
    return jax.tree.map(lambda x: jax.device_put(x, x.sharding.with_memory_kind("pinned_host"))
                        if isinstance(x, jax.Array) and hasattr(x, "sharding") else x, tree)


class Snapshotter(BaseCheckpointer):
    """Manages asynchronous backups of JAX array states to pinned host memory."""

    @config_class
    class Config(BaseCheckpointer.Config):
        dir: Optional[str] = None
        save_policy: ConfigOr[Any] = config_for_function(every_n_steps_policy).set(n=5)
        stage_in_hbm: bool = False
        defer_during_checkpoint: bool = True
        timeout_seconds: int = 3600
        max_retries: int = 100

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._save_policy = maybe_instantiate(self.config.save_policy)
        self._latest_snapshot: Optional[tuple[Nested[Tensor], int]] = None
        self._lock, self._queue = threading.Lock(), queue.Queue(maxsize=1)
        self._generation, self._stage_in_hbm = 0, cfg.stage_in_hbm
        self._copy_fn = jax.jit(lambda xs: [jnp.copy(x) for x in xs])
        if pathwaysutils is not None and pathwaysutils.is_pathways_backend_used():
            with contextlib.suppress(Exception):
                from orbax.checkpoint import pathways  # pylint: disable=import-outside-toplevel
                pathways.register_type_handlers(
                    use_single_replica_array_handler=False, primary_host=None,
                    checkpointing_impl=pathways.CheckpointingImpl.COLOCATED_PYTHON,
                    array_metadata_store=None, enable_write_sharding_file=False)
        threading.Thread(target=self._worker, name=f"{self.path()}_worker", daemon=True).start()

    @property
    def latest_snapshot_step(self) -> Optional[int]:
        with self._lock:
            return None if self._latest_snapshot is None else self._latest_snapshot[1]

    def _worker(self):
        while True:
            src, step, task_gen, staged, free_old = self._queue.get()
            pinned_state, old = None if staged else src, None
            try:
                if task_gen == self._generation:
                    if staged:
                        jax.block_until_ready(src)
                        pinned_state = _to_pinned_host(src)
                    jax.block_until_ready(pinned_state)
                    with self._lock:
                        if task_gen == self._generation:
                            old = self._latest_snapshot if free_old else None
                            self._latest_snapshot, pinned_state = (pinned_state, step), None
                    logging.info("[ELASTIC] Snapshot at step %d is ready.", step)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("[ELASTIC] Failed to secure snapshot at step %d: %s", step, type(e))
            finally:
                if staged:
                    _free_arrays(src, memory_kind="device")
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
    def _stage(self, state: Nested[Tensor]) -> Nested[Tensor]:
        copies = iter(self._copy_fn([x for x in jax.tree.leaves(state) if isinstance(x, jax.Array)]))
        return jax.tree.map(lambda x: next(copies) if isinstance(x, jax.Array) else x, state)
    def save(self, *, step: int, state: Nested[Tensor], evaler_summaries: Optional[dict[str, Any]] = None,
             force: bool = False, hold_previous: bool = False):
        if not force and not self._save_policy(step=step, evaler_summaries=evaler_summaries or {}):
            return
        (self._cancel_pending if self._stage_in_hbm or hold_previous else self.evict)()
        src = self._stage(state) if self._stage_in_hbm else _to_pinned_host(state)
        self._queue.put((src, step, self._generation, self._stage_in_hbm, not hold_previous))
        if force:
            self.wait_until_finished()
    def check_hbm_headroom(self, compiled: Optional[jax.stages.Compiled], *, device: jax.Device):
        self._stage_in_hbm = self.config.stage_in_hbm
        if not self._stage_in_hbm or (mem := getattr(compiled, "memory_analysis", lambda: None)()) is None:
            return
        req = (mem.argument_size_in_bytes + 2 * mem.output_size_in_bytes - mem.alias_size_in_bytes
               + mem.temp_size_in_bytes + mem.generated_code_size_in_bytes)
        limit = None
        with contextlib.suppress(Exception):
            limit = (device.memory_stats() or {}).get("bytes_limit")
        limit = limit or int(next((g for k, g in _HBM_GIB.items() if k in getattr(device, "device_kind", "")), 0) * 2**30)
        if 0 < limit < req:
            logging.warning("[ELASTIC] Disabling stage_in_hbm: need %.1f GiB of %.1f GiB HBM.", req / 2**30, limit / 2**30)
            self._stage_in_hbm = False
    def restore(self, *, step: Optional[int] = None, state: Union[Nested[Tensor], Nested[TensorSpec]]):
        self._cancel_pending()
        with self._lock:
            if not self._latest_snapshot or step not in (None, self._latest_snapshot[1]):
                return None, state
            pinned_state, snap_step = self._latest_snapshot
        def restore_leaf(x, spec):
            if not isinstance(x, jax.Array) or not hasattr(x.sharding, "mesh"):
                return x
            sharding = spec.sharding.with_memory_kind("device")
            replicas = (split_by_mesh_axis.split_by_mesh_axis(x, "data")
                        if x.sharding.mesh.shape.get("data", 1) > 1 and split_by_mesh_axis is not None else (x,))
            for r in replicas:
                with contextlib.suppress(Exception):
                    jax.block_until_ready(r)
                    if not (by_idx := {s.index: s.data for s in getattr(r, "addressable_shards", ())}):
                        continue
                    if np.array_equal(r.sharding.mesh.devices.flat, sharding.mesh.devices.flat):
                        return jax.device_put(r, sharding)
                    regions = sharding.addressable_devices_indices_map(tuple(x.shape))
                    local_ids = {d.id for d in r.sharding.mesh.devices.flat}
                    shards, cache = [], {}
                    for d in sorted(sharding.addressable_devices, key=lambda d: d.id not in local_ids):
                        dev_s = jax.sharding.SingleDeviceSharding(d).with_memory_kind("device")
                        shard = jax.device_put(cache.get(regions[d], by_idx[regions[d]]), dev_s)
                        cache.setdefault(regions[d], shard)
                        shards.append(shard)
                    return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)
            raise ValueError("No healthy shards in snapshot")
        try:
            restored = jax.tree.map(restore_leaf, pinned_state, state)
            jax.block_until_ready(restored)
        except (ValueError, jax.errors.JaxRuntimeError) as e:
            if isinstance(e, jax.errors.JaxRuntimeError) and "Full data loss while cloning" not in str(e):
                raise
            logging.warning("[ELASTIC] Snapshot at step %d has no healthy shards: %s", snap_step, e)
            self.evict()
            return None, state
        logging.info("[ELASTIC] Restored snapshot at step %d onto %d live devices", snap_step, len(jax.devices()))
        return snap_step, restored
    def wait_until_finished(self):
        self._queue.join()
    def stop(self, *, has_exception: bool = False):
        (self._cancel_pending if has_exception else self.wait_until_finished)()
