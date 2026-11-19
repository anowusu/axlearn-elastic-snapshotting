"""Manages asynchronous backups of JAX array states to pinned host memory."""

from collections.abc import Callable
import contextlib
import logging
import queue
import threading
from typing import Any

from etils import epath
import jax
from orbax.checkpoint._src import sharding_utils
from orbax.checkpoint.experimental.v1._src.training.metadata import types as training_metadata_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


def is_shardable_array(x: ...) -> bool:  # pyrefly: ignore[invalid-annotation]
  """Returns True if x is a concrete shardable array."""
  return isinstance(x, jax.Array)


def _is_prng_key(x: Any) -> bool:
  """Returns True if x has a JAX PRNGKeyArray dtype."""
  return (
      hasattr(x, "dtype")
      and hasattr(x, "shape")
      and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
  )


def _unwrap_if_prng_key(x: Any) -> Any:
  """Extracts the underlying key data buffer if x is a PRNGKeyArray."""
  return jax.random.key_data(x) if _is_prng_key(x) else x


def _wrap_if_prng_key(x: Any, orig_x: Any) -> Any:
  """Wraps raw array x into a PRNGKeyArray if orig_x was a PRNGKeyArray."""
  if _is_prng_key(orig_x):
    return jax.random.wrap_key_data(x, impl=jax.random.key_impl(orig_x))
  return x


def _pin_host_copy(array: Any) -> Any:
  """Copies the unwrapped JAX `array` into pinned host memory.

  The result never aliases `array`, even when `array` is already in pinned host
  memory (for example, host-offloaded parameters or optimizer state), so a
  snapshot keeps its buffers when the caller donates `array`.

  Args:
    array: The unwrapped JAX array to copy.

  Returns:
    A new JAX array in pinned host memory.
  """
  return jax.device_put(
      array, array.sharding.with_memory_kind("pinned_host"), may_alias=False
  )


def _copy_arrays(state: tree_types.PyTree) -> tree_types.PyTree:
  """Copies array leaves so that the caller may donate or mutate `state`."""

  def _copy_leaf(leaf: Any) -> Any:
    if not is_shardable_array(leaf):
      return leaf
    array = _unwrap_if_prng_key(leaf)
    if sharding_utils.is_in_device_memory(array):
      return _wrap_if_prng_key(jax.numpy.copy(array), leaf)
    return _wrap_if_prng_key(_pin_host_copy(array), leaf)

  return jax.tree.map(_copy_leaf, state)


def _find_named_mesh(tree: Any) -> Any:
  """Returns the mesh of the first `NamedSharding` leaf in `tree`, or None."""
  for leaf in jax.tree.leaves(tree):
    sharding = getattr(leaf, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
      return sharding.mesh
  return None


def _put_arrays(state: Any, target: Any = None) -> Any:
  """Moves array leaves of `state` onto `target` shardings via device memory."""
  leaves, tree_def = jax.tree.flatten(state)
  targets, target_def = jax.tree.flatten(state if target is None else target)
  if tree_def != target_def:
    raise ValueError("Snapshot PyTree structure does not match target.")
  idxs, curr, shardings = [], [], []
  for i, (leaf, target_leaf) in enumerate(zip(leaves, targets)):
    if not is_shardable_array(leaf):
      continue
    if target is None:
      sharding = leaf.sharding.with_memory_kind(None)
    elif isinstance(target_leaf, jax.sharding.Sharding):
      sharding = target_leaf
    else:
      if tuple(target_leaf.shape) != leaf.shape:
        raise RuntimeError("Restored replica shape mismatch.")
      sharding = target_leaf.sharding
      if not isinstance(sharding, jax.sharding.Sharding):
        raise TypeError(f"Expected jax.sharding.Sharding, got {sharding!r}.")
    idxs.append(i)
    curr.append(_unwrap_if_prng_key(leaf))
    shardings.append(sharding)
  stage1 = [
      arr.sharding.with_memory_kind(None)
      if arr.sharding != dst and arr.sharding.memory_kind == "pinned_host"
      else arr.sharding
      for arr, dst in zip(curr, shardings)
  ]
  stage2 = [dst.with_memory_kind(None) for dst in shardings]
  for stage in (stage1, stage2, shardings):
    move = [j for j, (x, t) in enumerate(zip(curr, stage)) if x.sharding != t]
    if move:
      moved = jax.device_put([curr[j] for j in move], [stage[j] for j in move])
      for j, array in zip(move, moved):
        curr[j] = array
  for i, array in zip(idxs, curr):
    leaves[i] = _wrap_if_prng_key(array, leaves[i])
  return jax.tree.unflatten(tree_def, leaves)


def _replicate_single_device_leaves(tree: Any) -> Any:
  """Replicates single-device array leaves across the mesh of `tree`."""
  mesh = _find_named_mesh(tree)
  if mesh is None:
    return tree
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def _target(leaf: Any) -> Any:
    if is_shardable_array(leaf) and isinstance(
        leaf.sharding, jax.sharding.SingleDeviceSharding
    ):
      return replicated
    return leaf

  return _put_arrays(tree, jax.tree.map(_target, tree))


def _extract_arrays(tree: Any) -> tuple[list[Any], Any]:
  """Unwraps `jax.Array` leaves of `tree` and returns `(arrays, rebuild_fn)`."""
  leaves, tree_def = jax.tree.flatten(tree)
  indices = [i for i, leaf in enumerate(leaves) if is_shardable_array(leaf)]

  def rebuild(arrays: Any) -> Any:
    restored = list(leaves)
    for i, array in zip(indices, arrays):
      restored[i] = _wrap_if_prng_key(array, leaves[i])
    return jax.tree.unflatten(tree_def, restored)

  return [_unwrap_if_prng_key(leaves[i]) for i in indices], rebuild


def _get_pathwaysutils():
  """Imports and returns pathwaysutils experimental modules."""
  try:
    # pylint: disable=g-import-not-at-top
    from pathwaysutils.experimental import concatenate_by_mesh_axis  # pyrefly: ignore[missing-import]
    from pathwaysutils.experimental import split_by_mesh_axis  # pyrefly: ignore[missing-import]
  except ImportError as e:
    raise ImportError(
        "Snapshotter requires pathwaysutils. Please ensure pathwaysutils is"
        " installed or linked via"
        " //third_party/py/orbax/checkpoint/experimental/v1:pathways_support."
    ) from e
  return concatenate_by_mesh_axis, split_by_mesh_axis


class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  def __init__(
      self,
      *,
      replica_axis_index: int = 0,
      copy_device_arrays_for_async_transfer: bool = False,
  ):
    """Initializes Snapshotter.

    Args:
      replica_axis_index: The index of the mesh axis representing data/model
        replicas. Typically corresponds to physical slices (connected by DCN).
      copy_device_arrays_for_async_transfer: Whether `save` copies device arrays
        on device, instead of transferring them to host, before returning to the
        caller. The copies are transferred to pinned host memory in the
        background and released when their transfer completes, so the caller may
        donate or mutate the originals without waiting for the transfer. This
        can double peak device memory usage. A `save` that takes the snapshot
        before it returns makes no copy.
    """
    self._concatenate_by_mesh_axis, self._split_by_mesh_axis = (
        _get_pathwaysutils()
    )
    # The replicas of the latest snapshot, its step and its full-mesh state.
    self._latest_snapshot: (
        tuple[list[tree_types.PyTree], int, tree_types.PyTree] | None
    ) = None
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self._generation = 0
    self.replica_axis_index = replica_axis_index
    self._copy_on_device = copy_device_arrays_for_async_transfer

    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    """Processes background snapshot requests from the queue."""
    while True:
      (
          full_mesh_pinned_state,
          per_replica_pinned_trees,
          step,
          generation,
          staged_copy,
      ) = self._queue.get()
      try:
        # `cancel_pending` ran after `save` queued this request, so discard it.
        if generation != self._generation:
          continue
        if staged_copy is not None:
          # `staged_copy` holds copies that only this request uses: wait for
          # them, start their transfer to pinned host memory and release them.
          jax.block_until_ready(jax.tree.map(_unwrap_if_prng_key, staged_copy))
          full_mesh_pinned_state, per_replica_pinned_trees = (
              self._to_snapshot_state(staged_copy, may_alias=True)
          )
          staged_copy = None
        jax.block_until_ready(
            jax.tree.map(
                _unwrap_if_prng_key,
                (full_mesh_pinned_state, per_replica_pinned_trees),
            )
        )
      except (
          jax.errors.JaxRuntimeError,
          RuntimeError,
          # Raised by `_to_snapshot_state`, which runs on this thread for a
          # staged request. An uncaught error would end this thread.
          ValueError,
          IndexError,
      ) as e:
        logging.exception("Failed to snapshot state at step %d: %s", step, e)
      else:
        with self._lock:
          if generation == self._generation:
            self._latest_snapshot = (
                per_replica_pinned_trees,
                step,
                full_mesh_pinned_state,
            )
      finally:
        del full_mesh_pinned_state, per_replica_pinned_trees
        del staged_copy
        self._queue.task_done()

  def _to_snapshot_state(
      self, state: tree_types.PyTree, may_alias: bool = False
  ) -> tuple[tree_types.PyTree, list[tree_types.PyTree]]:
    """Copies a PyTree to pinned host memory, whole and split into replicas.

    Args:
      state: The PyTree to be converted and split.
      may_alias: Whether arrays of `state` that already reside in pinned host
        memory may be used as they are instead of being copied.

    Returns:
      A tuple of `state` with its arrays in pinned host memory, and a list of
      PyTrees where each PyTree represents a single replica of `state`. In all
      of these PyTrees, single-device arrays are replicated across the mesh, and
      non-array leaves (e.g., scalars, strings, metadata, Python objects) are
      preserved.
    """
    replicated = _replicate_single_device_leaves(state)
    arrays, rebuild = _extract_arrays(replicated)
    if not arrays:
      return state, [state]
    axis, counts = "", set()
    for array in arrays:
      mesh = getattr(array.sharding, "mesh", None)
      axis = mesh.axis_names[self.replica_axis_index] if mesh else ""
      if axis and any(
          axis in jax.tree.leaves(dim) for dim in array.sharding.spec
      ):
        raise ValueError(f"Cannot shard along replica axis {axis!r}.")
      counts.add(mesh.shape[axis] if mesh else 1)
    if len(counts) > 1:
      raise ValueError(
          "All JAX arrays must have the same number of replicas along the"
          f" replica axis, got {sorted(counts)}."
      )
    pinned = jax.device_put(
        arrays,
        [array.sharding.with_memory_kind("pinned_host") for array in arrays],
        may_alias=may_alias,
    )
    pinned_state = rebuild(pinned)
    if counts == {1}:
      return pinned_state, [pinned_state]
    replicas = self._split_by_mesh_axis.split_by_mesh_axis(pinned, axis)
    return pinned_state, [rebuild(replica) for replica in replicas]

  def _enqueue_snapshot(
      self, step: int, state: tree_types.PyTree, copy_on_device: bool = False
  ) -> tree_types.PyTree | None:
    """Queues the snapshot of `state` at `step` for the worker thread.

    Args:
      step: The training step number associated with the state.
      state: The PyTree to be saved.
      copy_on_device: Whether to queue a copy of the device arrays of `state`,
        which the worker thread transfers to pinned host memory.

    Returns:
      `state` with its arrays in pinned host memory, as the snapshot stores it,
      or None with `copy_on_device`.
    """
    if copy_on_device:
      self._queue.put((None, None, step, self._generation, _copy_arrays(state)))
      return None
    pinned_state, per_replica_pinned_trees = self._to_snapshot_state(state)
    self._queue.put(
        (pinned_state, per_replica_pinned_trees, step, self._generation, None)
    )
    return pinned_state

  def save(
      self,
      step: int,
      state: tree_types.PyTree,
      *,
      wait: bool = False,
      checkpointer_wait_fn: Callable[[], Any] | None = None,
      checkpointer_save_fn: Callable[[tree_types.PyTree], Any] | None = None,
  ) -> Any:
    """Backs up JAX array states to pinned host memory.

    By default, the backup runs asynchronously. If previous snapshotting
    requests are still in progress, this request may be skipped.

    The saved state contains all replicas of user data. When restoring via
    `load_pytree`, snapshotter is able to reconstruct user data even if some
    replicas are unavailable, making it resilient to failures of some replicas.

    With `wait` or a checkpointer function, this method instead takes the
    snapshot of `step` before it returns, in the following order:

    1. Calls `checkpointer_wait_fn`, so that the previous checkpoint save
       releases its host arrays before the snapshot allocates new ones.
    2. Waits for the snapshotting requests in progress.
    3. Takes the snapshot.
    4. Calls `checkpointer_save_fn` with `state` backed by the host arrays of
       the snapshot, so that the checkpoint needs no second transfer from device
       memory. If the snapshot failed, passes `state` itself.

    Args:
      step: The training step number associated with the state.
      state: The PyTree to be saved.
      wait: Whether to take the snapshot of `step` before returning.
      checkpointer_wait_fn: Waits for the previous save of a checkpointer.
      checkpointer_save_fn: Saves the PyTree that it is given as a checkpoint.
        The arrays of that PyTree belong to the snapshot, so the function must
        not donate or delete them. Single-device arrays are replicated across
        the mesh.

    Returns:
      The result of `checkpointer_save_fn`, or None without that function.
    """
    if not (wait or checkpointer_wait_fn or checkpointer_save_fn):
      staged = self._copy_on_device
      # A staged request keeps a second copy of `state` in device memory until
      # its transfer to host finishes, so only one such request may be in
      # flight.
      busy = self._queue.unfinished_tasks > 0 if staged else self._queue.full()
      if busy:
        logging.warning("Snapshotter busy. Skipping snapshot for step %d", step)
        return None
      self._enqueue_snapshot(step, state, copy_on_device=staged)
      return None

    if checkpointer_wait_fn is not None:
      checkpointer_wait_fn()
    self.join()
    # The caller cannot donate `state` before this method returns, so the
    # snapshot needs no copy of `state` in device memory.
    pinned_state = self._enqueue_snapshot(step, state)
    self.join()
    if checkpointer_save_fn is None:
      return None
    with self._lock:
      latest_snapshot = self._latest_snapshot
    if latest_snapshot is None or latest_snapshot[2] is not pinned_state:
      # The snapshot failed, so the checkpoint transfers `state` itself.
      pinned_state = state
    return checkpointer_save_fn(pinned_state)

  def load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool | None = None,
  ) -> tree_types.PyTree:
    """Move arrays from workers onto TPU devices.

    Uses `abstract_state.sharding` to properly re-partition onto the new mesh.

    Args:
      abstract_state: An abstract representation of the state, used to provide
        the target shardings for the restored arrays on the TPU devices.
      reset_snapshot_state: If True, clears snapshot history and resets it to
        contain only the returned restored state (in host-pinned memory). If
        None, resets only when the mesh of `abstract_state` has more than one
        replica along the replica axis.

    Returns:
      The restored array state.

    Raises:
      RuntimeError: If no snapshots are available to restore from.
    """
    self.join()
    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      replica_trees, step, _ = self._latest_snapshot

    def is_replica_active(replica_tree):
      try:
        probe = next(
            filter(is_shardable_array, jax.tree.leaves(replica_tree)), None
        )
        if probe is not None:
          jax.block_until_ready(_unwrap_if_prng_key(probe)[...])
        return True
      except (jax.errors.JaxRuntimeError, RuntimeError):
        return False

    active_trees = [
        tree for tree in replica_trees if is_replica_active(tree)
    ]
    if not active_trees:
      raise RuntimeError("No active replicas found.")

    # TODO(b/574655289): Pathways on Cloud workers crash on cross-slice
    # transfers of pinned_host buffers over RpcTransport ("Attempting to
    # decrease reference on a buffer with zero reference count") until their
    # images include the PJRT C API fix. Until then, copy each replica to device
    # memory on its own devices, then concatenate the copies and reshard them in
    # device memory. In either memory, concatenate copies rather than the
    # snapshot's replicas: concatenating donates its inputs, and the snapshot
    # must survive a failed load (e.g. a slice going down) for it to be retried.
    device_state = active_trees[0]
    mesh = _find_named_mesh(abstract_state)
    rep_mesh = _find_named_mesh(device_state)
    idx = self.replica_axis_index
    _, rebuild = _extract_arrays(device_state)
    reps = [_extract_arrays(tree)[0] for tree in active_trees]
    if (
        reps[0]
        and mesh is not None
        and rep_mesh is not None
        and rep_mesh.axis_names == mesh.axis_names
        and rep_mesh.axis_sizes
        == (*mesh.axis_sizes[:idx], 1, *mesh.axis_sizes[idx + 1 :])
    ):
      subs = [
          jax.sharding.Mesh(
              mesh.devices.take([k], axis=idx),
              mesh.axis_names,
              axis_types=rep_mesh.axis_types,
          )
          for k in range(mesh.devices.shape[idx])
      ]
      by_devs = {tuple(rep[0].sharding.mesh.devices.flat): rep for rep in reps}
      keys = [tuple(sub.devices.flat) for sub in subs]
      missing = [sub for sub, key in zip(subs, keys) if key not in by_devs]
      donors = _put_arrays(reps if missing else [by_devs[k] for k in keys])
      by_devs = {
          tuple(rep[0].sharding.mesh.devices.flat): rep for rep in donors
      }
      filled = iter(())
      if missing:
        srcs = [
            [donors[(k + j) % len(donors)][j] for j in range(len(donors[0]))]
            for k in range(len(missing))
        ]
        dsts = [
            [jax.sharding.NamedSharding(s, a.sharding.spec) for a in donors[0]]
            for s in missing
        ]
        filled = iter(_put_arrays(srcs, dsts))
      slices = [by_devs[k] if k in by_devs else next(filled) for k in keys]
      concat = self._concatenate_by_mesh_axis.concatenate_by_mesh_axis
      axis = mesh.axis_names[idx]
      device_state = rebuild(
          slices[0] if len(subs) == 1 else concat(slices, axis)
      )
    restored_state = _put_arrays(device_state, abstract_state)
    unwrapped_restored = jax.tree.map(_unwrap_if_prng_key, restored_state)
    jax.block_until_ready(unwrapped_restored)

    if (should_reset := reset_snapshot_state) is None:
      mesh = _find_named_mesh(abstract_state)
      should_reset = bool(
          mesh and mesh.shape[mesh.axis_names[self.replica_axis_index]] > 1
      )
    if should_reset:
      full_mesh_pinned_state, per_replica_pinned_trees = (
          self._to_snapshot_state(restored_state)
      )
      with self._lock:
        self._latest_snapshot = (
            per_replica_pinned_trees,
            step,
            full_mesh_pinned_state,
        )

    return restored_state

  def try_load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool | None = None,
  ) -> tuple[int | None, tree_types.PyTree | None]:
    """Restores `(step, state)` from the latest snapshot, or `(None, None)`."""
    self.join()
    if (latest := self.latest) is None:
      return None, None
    try:
      return latest.step, self.load(
          abstract_state, reset_snapshot_state=reset_snapshot_state
      )
    except jax.errors.JaxRuntimeError:
      raise
    except RuntimeError as e:
      logging.warning("Snapshot restore failed at step %d: %s", latest.step, e)
      # Drop the unusable snapshot so that `latest` stops reporting it.
      with self._lock:
        self._latest_snapshot = None
      return None, None

  def join(self) -> None:
    """Blocks until all queued snapshot requests have been processed."""
    self._queue.join()

  def cancel_pending(self, timeout: float | None = 0.0) -> bool:
    """Cancels pending or in-flight snapshots, keeping the latest completed one.

    Args:
      timeout: Maximum seconds to wait for an in-flight snapshot to finish.
        Defaults to 0.0 seconds. If None, waits without a timeout.

    Returns:
      True if all queued requests finished within `timeout`, else False.
    """
    with self._lock, contextlib.suppress(queue.Empty):
      self._generation += 1
      while True:
        self._queue.get_nowait()
        self._queue.task_done()
    with self._queue.all_tasks_done:
      return self._queue.all_tasks_done.wait_for(
          lambda: not self._queue.unfinished_tasks, timeout=timeout
      )

  @property
  def latest(self) -> training_metadata_types.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step, _ = self._latest_snapshot
    return training_metadata_types.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )
