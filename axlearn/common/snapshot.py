# Copyright © 2024 Apple Inc.

"""Manages asynchronous backups of JAX array states to pinned host memory."""

import logging
import queue
import threading
from typing import Any

from etils import epath
import jax
from orbax.checkpoint.experimental.v1 import training  # pytype: disable=import-error
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types  # pytype: disable=import-error
from pathwaysutils.experimental import concatenate_by_mesh_axis  # pytype: disable=import-error
from pathwaysutils.experimental import split_by_mesh_axis  # pytype: disable=import-error

_logger = logging.getLogger(__name__)


class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  def __init__(self, *, replica_axis_index: int = 0):
    self._latest_snapshot: tuple[tree_types.PyTree, int] | None = None
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self.replica_axis_index = replica_axis_index
    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    while True:
      pinned_state, step = self._queue.get()
      try:
        _logger.info(
            "[*] [Snapshot Thread] Waiting for snapshot at step %d to be ready...",
            step,
        )
        jax.block_until_ready(pinned_state)
        _logger.info(
            "[*] [Snapshot Thread] Snapshot at step %d is ready and secured.",
            step,
        )
        with self._lock:
          self._latest_snapshot = (pinned_state, step)
      except Exception as e:  # pylint: disable=broad-except
        _logger.warning(
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
    if self._queue.full():
      _logger.warning("Snapshotter busy. Skipping snapshot for step %d", step)
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
    try:
      from pathwaysutils import jax as pw_jax
      if type(pw_jax.concatenate_by_mesh_axis).__name__ != "_FakeJaxFunction":
        use_pathways = True
    except (ImportError, AttributeError):
      pass

    if use_pathways:
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
      import numpy as np

      def get_local_numpy_array(x: jax.Array) -> np.ndarray:
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

      def heal_leaf(x, abstract):
        if not hasattr(x, "sharding") or x.sharding is None:
          # Non-JAX array (e.g. Python scalar or metadata), return as is or put to device
          return jax.device_put(x, abstract.sharding) if hasattr(abstract, "sharding") else x

        # Check if the array is healthy and block until ready
        try:
          jax.block_until_ready(x)
          # If block_until_ready succeeds, the array is healthy.
          # We can convert it to a local numpy array.
          # To avoid cross-host gather (which might hang if a host is dead),
          # we reconstruct the local addressable data from its addressable shards.
          local_np = get_local_numpy_array(x)
        except Exception as e:
          # If the global array access fails, try to extract surviving local shards
          _logger.warning("Global array access failed: %s. Attempting local shard extraction...", e)
          local_np = get_local_numpy_array(x)

        if local_np is None:
          raise RuntimeError("Failed to retrieve any local addressable data for array.")

        # Put the local numpy array onto the new TPU mesh sharding
        return jax.device_put(local_np, abstract.sharding)

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
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

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
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return restored_state

  def join(self) -> None:
    """Blocks until all snapshots in the queue are ready and secured."""
    self._queue.join()

  @property
  def latest(self) -> training.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step = self._latest_snapshot
    return training.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )