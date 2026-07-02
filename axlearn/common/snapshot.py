# Copyright © 2024 Apple Inc.

"""Manages asynchronous backups of JAX array states to pinned host memory."""

import logging
import queue
import threading
from typing import Any, Callable, Optional

import concurrent.futures
import time
from etils import epath
import jax
from orbax.checkpoint.experimental.v1 import training  # pytype: disable=import-error
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types  # pytype: disable=import-error
from pathwaysutils.experimental import concatenate_by_mesh_axis  # pytype: disable=import-error
from pathwaysutils.experimental import split_by_mesh_axis  # pytype: disable=import-error
import numpy as np

_logger = logging.getLogger(__name__)


def get_local_numpy_array(x: jax.Array) -> np.ndarray:
  """Retrieve process-local addressable data of a sharded array as a numpy array."""
  if not x.addressable_shards:
    return None

  # Bounding box calculation of all local addressable shards
  indices = [shard.index for shard in x.addressable_shards]
  local_slices = []
  for dim in range(x.ndim):
    dim_slices = [idx[dim] for idx in indices]
    start = min((s.start if s.start is not None else 0) for s in dim_slices)
    stop = max((s.stop if s.stop is not None else x.shape[dim]) for s in dim_slices)
    local_slices.append(slice(start, stop))

  local_array_shape = tuple(s.stop - s.start for s in local_slices)
  local_np = np.empty(local_array_shape, dtype=x.dtype)

  for shard in x.addressable_shards:
    # Get relative slice in local array
    relative_slices = tuple(
        slice(
            (s.start if s.start is not None else 0) - local_slices[dim].start,
            (s.stop if s.stop is not None else x.shape[dim]) - local_slices[dim].start
        )
        for dim, s in enumerate(shard.index)
    )
    # Convert shard data to numpy locally
    local_np[relative_slices] = np.array(shard.data)

  return local_np


def to_local_numpy(pytree: Any, is_cancelled: Optional[Callable[[], bool]] = None) -> Any:
  """Convert all JAX arrays in pytree to process-local numpy arrays without network sync."""
  leaves, treedef = jax.tree_util.tree_flatten(pytree)
  def leaf_fn(x):
    if is_cancelled and is_cancelled():
      raise RuntimeError("Snapshotting cancelled.")
    if isinstance(x, jax.Array):
      return get_local_numpy_array(x)
    return x
  from axlearn.common.utils import is_pathways_proxy
  if is_pathways_proxy():
    # Under Pathways proxy, excessive multi-threaded fetching can saturate the gRPC
    # connection and starve keep-alive signals. We use a limited pool of 16 workers
    # to speed up transfer while maintaining connection health.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
      numpy_leaves = list(executor.map(leaf_fn, leaves))
  else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
      numpy_leaves = list(executor.map(leaf_fn, leaves))
  return jax.tree_util.tree_unflatten(treedef, numpy_leaves)


class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  def __init__(self, *, replica_axis_index: int = 0):
    self._latest_snapshot: tuple[tree_types.PyTree, int] | None = None  # Pinned host JAX arrays
    self._latest_ready_snapshot: dict[str, Any] | None = None  # CPU NumPy state
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self.replica_axis_index = replica_axis_index
    self._cancelled = False
    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def cancel(self) -> None:
    """Cancels any in-progress or queued snapshot operations."""
    with self._lock:
      self._cancelled = True
      # Clear the queue
      try:
        while True:
          self._queue.get_nowait()
          self._queue.task_done()
      except queue.Empty:
        pass

  def _worker(self):
    print("[*] Snapshotter worker thread started", flush=True)
    while True:
      pinned_state, step = self._queue.get()
      print(f"[*] Snapshotter worker got item for step {step}", flush=True)
      with self._lock:
        if self._cancelled:
          self._queue.task_done()
          continue
      try:
        logging.info(
            "[*] [Snapshot Thread] Waiting for snapshot at step %d to be ready...",
            step,
        )
        jax.block_until_ready(pinned_state)
        with self._lock:
          if self._cancelled:
            raise RuntimeError("Cancelled")
        logging.info(
            "[*] [Snapshot Thread] Snapshot at step %d is ready. Converting to local NumPy...",
            step,
        )
        # Perform JAX Array -> local NumPy array conversion in background thread
        start_time = time.time()
        numpy_state = to_local_numpy(pinned_state, is_cancelled=lambda: self._cancelled)
        duration = time.time() - start_time
        with self._lock:
          self._latest_snapshot = (pinned_state, step)
          self._latest_ready_snapshot = {
              "trainer_state": numpy_state,
              "step": step,
          }
        logging.info(
            "[*] [Snapshot Thread] Secured NumPy snapshot for step %d in %.3f seconds.",
            step,
            duration,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "[*] [Snapshot Thread] Failed to secure snapshot at step %d: %s.",
            step,
            e,
        )
      finally:
        self._queue.task_done()

  def save_pytree(
      self, step: int, state: tree_types.PyTreeOf[jax.Array]
  ) -> None:
    """Move arrays onto CPU worker devices."""
    print(f"[*] save_pytree called for step {step}", flush=True)
    if self._queue.full():
      logging.warning("Snapshotter busy. Skipping snapshot for step %d", step)
      return

    pinned_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host"), state
    )

    pinned_state = jax.device_put(state, pinned_shardings)

    self._queue.put((pinned_state, step))

  @classmethod
  def heal_pytree(
      cls,
      host_snapshot: tree_types.PyTree,
      abstract_target_state: tree_types.PyTreeOf[jax.Array],
      *,
      replica_axis_index: int = 0,
  ) -> tree_types.PyTree:
    """Heals a PyTree host snapshot by discarding dead shards and re-sharding onto the target mesh.

    Architectural Decisions & Integration Points:
    This function implements the "Split-Filter-Stitch" resilience algorithm.
    When a hardware failure (TPU node drop) occurs, JAX throws a runtime error.
    The host memory snapshot contains replicated arrays (e.g. data parallel shards) on host RAM.
    Some of these shards are located on nodes that just died, while others are on surviving nodes.

    1. SPLIT: We split the globally sharded host array along the replica/data axis to isolate
       per-replica array shards.
    2. FILTER: We check accessibility of each replica shard using `jax.block_until_ready()`.
       If a shard is hosted on a dead physical node, it raises a JaxRuntimeError and we discard it.
       If it succeeds, it means the replica was hosted on a surviving node and is preserved.
    3. STITCH: We concatenate the surviving shards back together to form a consolidated replica,
       then call a fresh `jax.device_put` compile step to project this consolidated state onto
       the newly configured/recovered TPU mesh using the abstract target sharding.
    """
    # Check if pathwaysutils' split/concatenate functions are available and working.
    use_pathways = False
    is_snapshot_backend_alive = True
    try:
      from pathwaysutils import jax as pw_jax
      if type(pw_jax.concatenate_by_mesh_axis).__name__ != "_FakeJaxFunction":
        use_pathways = True
    except (ImportError, AttributeError):
      pass

    if use_pathways:
      try:
        from jax._src import xla_bridge as xb
        active_backend = xb.get_backend()
        first_array = None
        for leaf in jax.tree.leaves(host_snapshot):
          if hasattr(leaf, "sharding") and leaf.sharding is not None:
            first_array = leaf
            break
        if first_array is not None:
          if first_array.devices():
            snapshot_backend = first_array.devices()[0].client
            if snapshot_backend is not active_backend:
              is_snapshot_backend_alive = False
              _logger.info("Host snapshot backend is dead/stale. Disabling pathways split/concat.")
      except Exception as e:
        _logger.warning("Failed to check snapshot backend health: %s", e)

    if use_pathways and is_snapshot_backend_alive and first_array is not None:
      from pathwaysutils.experimental import concatenate_by_mesh_axis
      from pathwaysutils.experimental import split_by_mesh_axis

      def is_replica_active(arr):
        # Helper to check if a shard is hosted on a healthy surviving node.
        try:
          jax.block_until_ready(arr)
          return True
        except jax.errors.JaxRuntimeError as _:
          # Shard is unreachable due to dead node, discard it.
          return False

      def get_active_pytree(x):
        # Return non-sharded or non-JAX objects unchanged.
        if not hasattr(x, "sharding") or x.sharding is None:
          return x

        # Determine the mesh axis name representing replica/data parallelism.
        mesh_axis_name = x.sharding.mesh.axis_names[replica_axis_index]

        # Split the global array into list of array shards along the data axis.
        all_replicas = split_by_mesh_axis.split_by_mesh_axis(
            x,
            mesh_axis_name,
        )

        # Filter out shards that belong to dead nodes.
        active_replicas = [
            replica for replica in all_replicas if is_replica_active(replica)
        ]

        if not active_replicas:
          raise RuntimeError("No active replicas found; recovery is impossible.")

        # Concatenate the surviving replicas back together.
        reconstructed_state = concatenate_by_mesh_axis.concatenate_by_mesh_axis(
            active_replicas,
            mesh_axis_name,
        )
        return reconstructed_state

      # 1 & 2: Split and filter the entire host PyTree state.
      surviving_shards = jax.tree.map(get_active_pytree, host_snapshot)

      # 3: Stitch and compile device transfers to the new TPU mesh sharding.
      healed_state = jax.tree.map(
          lambda state, abstract: jax.device_put(state, abstract.sharding),
          surviving_shards,
          abstract_target_state,
      )
      return healed_state

    else:
      # FALLBACK: Pure JAX/Numpy implementation for CPU/Single-node or standard JAX environments.
      # Since we are on a surviving host, the local addressable shards of host_snapshot are intact.
      # We extract the local addressable data from the host_snapshot arrays as numpy arrays,
      # and then call jax.device_put to reshard them onto the new mesh.

      def heal_leaf(x, abstract):
        if not hasattr(abstract, "sharding") or abstract.sharding is None:
          # Non-JAX target, return as is
          return x

        # If it is a JAX array target, but the source is a numpy array (process local data)
        # or if the source is still a jax.Array (which shouldn't happen if we converted to local NumPy, but just in case)
        if isinstance(x, jax.Array):
          try:
            if is_snapshot_backend_alive:
              jax.block_until_ready(x)
            local_np = get_local_numpy_array(x)
          except Exception as e:
            _logger.warning("Global array access failed: %s. Attempting local shard extraction...", e)
            local_np = get_local_numpy_array(x)
        else:
          local_np = x

        if local_np is None:
          # Construct an empty numpy array of the expected local shape to participate in collective creation.
          try:
            from jax._src.sharding_impls import num_addressable_indices
            local_shape = tuple(
                num_addressable_indices(abstract.sharding, dim, abstract.shape)
                for dim in range(len(abstract.shape))
            )
          except Exception:
            # Fallback to zero-shape for all dimensions if lookup fails
            local_shape = tuple(0 for _ in abstract.shape)
          local_np = np.empty(local_shape, dtype=abstract.dtype)

        # Assemble global jax.Array from process-local NumPy slices to avoid shape mismatch
        return jax.make_array_from_process_local_data(
            sharding=abstract.sharding,
            local_data=local_np,
            global_shape=tuple(abstract.shape),
        )

      return jax.tree.map(heal_leaf, host_snapshot, abstract_target_state)

  def load_pytree(
      self,
      abstract_state: tree_types.PyTreeOf[jax.Array],
      *,
      reset_snapshot_state: bool = True,
  ) -> tree_types.PyTree:
    """Move arrays from workers onto TPU devices.

    Uses `abstract_state.sharding` to properly re-partition onto the new mesh.

    Args:
      abstract_state: An abstract representation of the state, used to provide
        the target shardings for the restored arrays on the TPU devices.
      reset_snapshot_state: If True, clears snapshot history and resets it to
        contain only the returned restored state (in host-pinned memory).

    Returns:
      The restored array state.

    Raises:
      RuntimeError: If no snapshots are available to restore from.
    """
    with self._lock:
      if self._latest_ready_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state = self._latest_ready_snapshot["trainer_state"]
      step = self._latest_ready_snapshot["step"]

    _logger.info("Restoring from snapshot at step %d...", step)
    restored_state = self.heal_pytree(
        pinned_state, abstract_state, replica_axis_index=self.replica_axis_index
    )

    if reset_snapshot_state:
      # Re-shard target state back to host pinned memory for the next snapshots
      host_target_shardings = jax.tree.map(
          lambda x: x.sharding.with_memory_kind("pinned_host"), abstract_state
      )
      host_target_state = jax.device_put(restored_state, host_target_shardings)
      numpy_state = to_local_numpy(host_target_state)
      with self._lock:
        self._latest_snapshot = (host_target_state, step)
        self._latest_ready_snapshot = {
            "trainer_state": numpy_state,
            "step": step,
        }

    return restored_state

  def join(self) -> None:
    """Blocks until all snapshots in the queue are ready and secured."""
    self._queue.join()

  @property
  def latest(self) -> training.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_ready_snapshot is None:
        return None
      step = self._latest_ready_snapshot["step"]
    return training.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )