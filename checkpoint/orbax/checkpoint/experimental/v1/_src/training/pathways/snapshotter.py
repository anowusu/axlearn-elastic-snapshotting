"""Manages asynchronous backups of JAX array states to pinned host memory."""

from collections.abc import Sequence
import contextlib
import dataclasses
import logging
import queue
import threading
from typing import Any

from etils import epath
import jax
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
    return jax.random.wrap_key_data(x, dtype=orig_x.dtype)
  return x


def _copy_arrays(state: tree_types.PyTree) -> tree_types.PyTree:
  """Copies array leaves so that the caller may donate or mutate `state`."""
  arrays = _extract_arrays(state)
  host_indices = []
  for i, array in enumerate(arrays):
    if array.sharding.memory_kind in (None, "device"):
      arrays[i] = jax.numpy.copy(array)
    else:
      host_indices.append(i)
  if host_indices:
    host_copies = jax.device_put(
        [arrays[i] for i in host_indices],
        [
            arrays[i].sharding.with_memory_kind("pinned_host")
            for i in host_indices
        ],
        may_alias=False,
    )
    for i, copied in zip(host_indices, host_copies, strict=True):
      arrays[i] = copied
  return _replace_arrays(state, arrays)


def _is_replica_active(replica_tree: tree_types.PyTree) -> bool:
  """Returns True if `replica_tree`'s slice placement is reachable."""
  probe = next(
      filter(is_shardable_array, jax.tree.leaves(replica_tree)), None
  )
  if probe is None:
    return True
  try:
    jax.block_until_ready(_unwrap_if_prng_key(probe)[...])
    return True
  except (jax.errors.JaxRuntimeError, RuntimeError):
    return False


def _find_named_mesh(tree: tree_types.PyTree) -> jax.sharding.Mesh | None:
  """Returns the mesh of the first `NamedSharding` leaf in `tree`, or None."""
  for leaf in jax.tree.leaves(tree):
    sharding = getattr(leaf, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding) and isinstance(
        sharding.mesh, jax.sharding.Mesh
    ):
      return sharding.mesh
  return None


def _extract_arrays(tree: tree_types.PyTree) -> list[jax.Array]:
  """Returns unwrapped `jax.Array` leaves of `tree`."""
  return [
      _unwrap_if_prng_key(leaf)
      for leaf in jax.tree.leaves(tree)
      if is_shardable_array(leaf)
  ]


def _replace_arrays(
    tree: tree_types.PyTree, arrays: Sequence[jax.Array]
) -> tree_types.PyTree:
  """Returns `tree` with its `jax.Array` leaves replaced by `arrays`."""
  leaves, tree_def = jax.tree.flatten(tree)
  array_iter = iter(arrays)
  for i, leaf in enumerate(leaves):
    if is_shardable_array(leaf):
      leaves[i] = _wrap_if_prng_key(next(array_iter), leaf)
  return jax.tree.unflatten(tree_def, leaves)


def _can_match_target_submeshes(
    replica_arrays: Sequence[jax.Array],
    replica_mesh: jax.sharding.Mesh,
    target_mesh: jax.sharding.Mesh,
    axis_index: int,
) -> bool:
  """Returns whether replicas can match `target_mesh` slice-locally.

  Checks that `target_mesh` has matching axis names, axis types (needed by
  `Mesh.__eq__` and `concatenate_by_mesh_axis`), and per-slice device shape,
  and that every leaf uses `NamedSharding` on `replica_mesh` without sharding
  along the replica axis so surviving slices are interchangeable donors.

  Args:
    replica_arrays: Unwrapped arrays from one active replica.
    replica_mesh: Mesh of the active replica.
    target_mesh: Target mesh to restore onto.
    axis_index: Index of the replica axis in the mesh.
  """
  axis_name = replica_mesh.axis_names[axis_index]
  return (
      target_mesh.axis_names == replica_mesh.axis_names
      and target_mesh.axis_types == replica_mesh.axis_types
      and target_mesh.devices.take([0], axis=axis_index).shape
      == replica_mesh.devices.shape
      and all(
          isinstance(array.sharding, jax.sharding.NamedSharding)
          and array.sharding.mesh == replica_mesh
          and not any(
              axis_name in jax.tree.leaves(dim) for dim in array.sharding.spec
          )
          for array in replica_arrays
      )
  )


def _put_arrays(
    state: tree_types.PyTree,
    target_shardings: tree_types.PyTree | None = None,
) -> tree_types.PyTree:
  """Moves array leaves of `state` onto `target_shardings` via device memory."""
  leaves = jax.tree.leaves(state)
  arrays = [
      _unwrap_if_prng_key(leaf) for leaf in leaves if is_shardable_array(leaf)
  ]
  if target_shardings is None:
    shardings = [array.sharding.with_memory_kind(None) for array in arrays]
  else:
    shardings = [
        target_sharding
        for leaf, target_sharding in zip(
            leaves, jax.tree.leaves(target_shardings), strict=True
        )
        if is_shardable_array(leaf)
    ]

  def _move(stage_shardings: Sequence[Any]) -> None:
    to_move = [
        i
        for i, (array, sharding, target) in enumerate(
            zip(arrays, stage_shardings, shardings, strict=True)
        )
        if array.sharding != target and array.sharding != sharding
    ]
    if to_move:
      moved = jax.device_put(
          [arrays[i] for i in to_move],
          [stage_shardings[i] for i in to_move],
      )
      for i, array in zip(to_move, moved, strict=True):
        arrays[i] = array

  # Stage cross-device transfers through device memory (b/574655289),
  # skipping leaves already at their final target sharding.
  # 1. Move off-target host arrays onto their local device memory.
  _move([array.sharding.with_memory_kind(None) for array in arrays])
  # 2. Reshard off-target arrays across devices in device memory.
  _move([sharding.with_memory_kind(None) for sharding in shardings])
  # 3. Move arrays to their final target memory kind (e.g., pinned_host).
  _move(shardings)
  return _replace_arrays(state, arrays)


def _replicate_single_device_leaves(
    tree: tree_types.PyTree,
) -> tree_types.PyTree:
  """Replicates single-device array leaves across the mesh of `tree`."""
  mesh = _find_named_mesh(tree)
  if mesh is None:
    return tree
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  arrays = _extract_arrays(tree)
  shardings = [
      replicated
      if isinstance(array.sharding, jax.sharding.SingleDeviceSharding)
      else array.sharding
      for array in arrays
  ]
  return _replace_arrays(tree, _put_arrays(arrays, shardings))


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


@dataclasses.dataclass(frozen=True)
class _SnapshotRequest:
  """A queued background snapshot request."""

  step: int
  generation: int
  replica_trees: list[tree_types.PyTree] | None = None
  staged_copy: tree_types.PyTree | None = None


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
      copy_device_arrays_for_async_transfer: Whether non-blocking `save` copies
        device arrays on device before returning, transferring those copies to
        pinned host memory in the background so the caller may donate `state`.
    """
    self._concatenate_by_mesh_axis, self._split_by_mesh_axis = (
        _get_pathwaysutils()
    )
    self._latest_snapshot: tuple[list[tree_types.PyTree], int] | None = None
    self._lock = threading.Lock()
    self._queue: queue.Queue[_SnapshotRequest] = queue.Queue(maxsize=1)
    self._generation = 0
    self._pending_count = 0
    self.replica_axis_index = replica_axis_index
    self._copy_on_device = copy_device_arrays_for_async_transfer

    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    """Processes background snapshot requests from the queue."""
    while True:
      request = self._queue.get()
      replica_trees = request.replica_trees
      staged = request.staged_copy
      try:
        with self._lock:
          if request.generation != self._generation:
            continue
        if staged is not None:
          jax.block_until_ready(_extract_arrays(staged))
          with self._lock:
            if request.generation != self._generation:
              continue
          _, replica_trees = self._to_snapshot_state(staged, may_alias=True)
          staged = None
        jax.block_until_ready(_extract_arrays(replica_trees))
      except (jax.errors.JaxRuntimeError, RuntimeError):
        logging.exception("Failed to snapshot state at step %d", request.step)
      else:
        with self._lock:
          if (
              request.generation == self._generation
              and replica_trees is not None
          ):
            self._latest_snapshot = (replica_trees, request.step)
      finally:
        del request, replica_trees, staged
        with self._lock:
          self._pending_count -= 1
        self._queue.task_done()

  def _validate_snapshot_arrays(
      self,
      arrays: Sequence[jax.Array],
      mesh: jax.sharding.Mesh | None,
      *,
      allow_single_device: bool = False,
  ) -> None:
    """Validates replica axis index and replica count uniformity."""
    if mesh is None:
      return
    replica_counts = set()
    for array in arrays:
      if allow_single_device and isinstance(
          array.sharding, jax.sharding.SingleDeviceSharding
      ):
        continue
      if not isinstance(array.sharding, jax.sharding.NamedSharding):
        raise ValueError(
            "Expected NamedSharding when a mesh is present, got"
            f" {array.sharding}."
        )
      array_mesh = array.sharding.mesh
      replica_counts.add(
          array_mesh.shape[array_mesh.axis_names[self.replica_axis_index]]
      )
    if len(replica_counts) > 1:
      raise ValueError(
          "All JAX arrays must have the same number of replicas along the"
          f" replica axis, got {sorted(replica_counts)}."
      )

  def _to_snapshot_state(
      self, state: tree_types.PyTree, may_alias: bool = False
  ) -> tuple[tree_types.PyTree, list[tree_types.PyTree]]:
    """Copies a PyTree to pinned host memory, whole and split into replicas."""
    replicated = _replicate_single_device_leaves(state)
    arrays = _extract_arrays(replicated)
    if not arrays:
      return state, [state]
    mesh = _find_named_mesh(replicated)
    self._validate_snapshot_arrays(arrays, mesh)
    pinned = jax.device_put(
        arrays,
        [array.sharding.with_memory_kind("pinned_host") for array in arrays],
        may_alias=may_alias,
    )
    pinned_state = _replace_arrays(replicated, pinned)
    # split_by_mesh_axis does not short-circuit axis size 1 (still dispatches
    # IFRT RemapArrays).
    if mesh is None:
      return pinned_state, [pinned_state]
    axis = mesh.axis_names[self.replica_axis_index]
    if mesh.shape[axis] == 1:
      return pinned_state, [pinned_state]
    replicas = self._split_by_mesh_axis.split_by_mesh_axis(pinned, axis)
    return pinned_state, [
        _replace_arrays(replicated, replica) for replica in replicas
    ]

  def save(
      self,
      step: int,
      state: tree_types.PyTree,
      *,
      wait: bool = False,
  ) -> tree_types.PyTree | None:
    """Backs up JAX array states to pinned host memory.

    By default, the backup runs asynchronously and skips the request if busy.
    With `wait=True`, waits for pending snapshots, takes the snapshot of `step`
    synchronously, and returns `state` backed by the snapshot's pinned host
    arrays so callers (e.g., colocated Python checkpointers) can reuse the host
    buffers without a second device-to-host transfer. Callers must not donate
    or delete the returned arrays.

    Args:
      step: The training step number associated with the state.
      state: The PyTree to be saved.
      wait: Whether to take the snapshot of `step` synchronously and return the
        pinned host PyTree.

    Returns:
      The pinned host PyTree of `state` when `wait=True`, or None otherwise.
    """
    if not wait:
      with self._lock:
        # In copy-on-device mode, skip while any snapshot is in flight so at
        # most one extra HBM copy of `state` exists (queue.full() with
        # maxsize=1 would permit one in the worker plus one queued).
        busy = (
            self._pending_count > 0
            if self._copy_on_device
            else self._queue.full()
        )
        if busy:
          logging.warning(
              "Snapshotter busy. Skipping snapshot for step %d", step
          )
          return None
        generation = self._generation
      if self._copy_on_device:
        self._validate_snapshot_arrays(
            _extract_arrays(state),
            _find_named_mesh(state),
            allow_single_device=True,
        )
        request = _SnapshotRequest(
            step=step, generation=generation, staged_copy=_copy_arrays(state)
        )
      else:
        _, replica_trees = self._to_snapshot_state(state)
        request = _SnapshotRequest(
            step=step, generation=generation, replica_trees=replica_trees
        )
      with self._lock:
        self._pending_count += 1
      self._queue.put(request)
      return None

    self.join()
    with self._lock:
      generation = self._generation
    pinned_state, replica_trees = self._to_snapshot_state(state)
    jax.block_until_ready(_extract_arrays((pinned_state, replica_trees)))
    with self._lock:
      if generation == self._generation:
        self._latest_snapshot = (replica_trees, step)
    return pinned_state

  def load(
      self,
      abstract_state: tree_types.PyTree,
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
    _, restored_state = self._load(
        abstract_state,
        reset_snapshot_state=reset_snapshot_state,
        allow_missing=False,
    )
    return restored_state

  def _load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool,
      allow_missing: bool,
  ) -> tuple[int | None, tree_types.PyTree | None]:
    """Restores `(step, state)` from the latest snapshot."""
    self.join()
    with self._lock:
      if self._latest_snapshot is None:
        if allow_missing:
          return None, None
        raise RuntimeError("No snapshots available to restore from.")
      replica_trees, step = self._latest_snapshot

    active_trees = [tree for tree in replica_trees if _is_replica_active(tree)]
    if not active_trees:
      if allow_missing:
        logging.warning("No active snapshot replicas found at step %d.", step)
        with self._lock:
          if (
              self._latest_snapshot is not None
              and self._latest_snapshot[1] == step
          ):
            self._latest_snapshot = None
        return None, None
      raise RuntimeError("No active replicas found.")

    # TODO(b/574655289): Pathways on Cloud workers crash on cross-slice
    # transfers of pinned_host buffers over RpcTransport ("Attempting to
    # decrease reference on a buffer with zero reference count") until their
    # images include the PJRT C API fix. Until then, copy each replica to device
    # memory on its own devices, then concatenate the copies and reshard them in
    # device memory. In either memory, concatenate copies rather than the
    # snapshot's replicas: concatenating donates its inputs, and the snapshot
    # must survive a failed load (e.g. a slice going down) for it to be retried.
    mesh = _find_named_mesh(abstract_state)
    device_state = self._restore_replicas(active_trees, mesh)

    def _extract_target_sharding(leaf: Any, abs_leaf: Any) -> Any:
      if not is_shardable_array(leaf):
        return leaf
      if (target_shape := tuple(abs_leaf.shape)) != leaf.shape:
        raise ValueError(
            f"Restored replica shape mismatch: {target_shape} != {leaf.shape}."
        )
      return abs_leaf.sharding

    target_shardings = jax.tree.map(
        _extract_target_sharding, device_state, abstract_state
    )
    restored_state = _put_arrays(device_state, target_shardings)
    jax.block_until_ready(_extract_arrays(restored_state))

    if reset_snapshot_state:
      _, new_replica_trees = self._to_snapshot_state(restored_state)
      jax.block_until_ready(_extract_arrays(new_replica_trees))
      with self._lock:
        self._latest_snapshot = (new_replica_trees, step)

    return step, restored_state

  def _restore_replicas(
      self,
      active_trees: Sequence[tree_types.PyTree],
      target_mesh: jax.sharding.Mesh | None,
  ) -> tree_types.PyTree:
    """Moves `active_trees` to device memory and concatenates along slices."""
    replica_mesh = _find_named_mesh(active_trees[0])
    if replica_mesh is None:
      return _put_arrays(active_trees[0])

    axis_index = self.replica_axis_index
    axis_name = replica_mesh.axis_names[axis_index]
    replicas = [_extract_arrays(tree) for tree in active_trees]
    if target_mesh is None or not _can_match_target_submeshes(
        replicas[0], replica_mesh, target_mesh, axis_index
    ):
      slices = _put_arrays(replicas)
    else:
      submeshes = [
          jax.sharding.Mesh(
              target_mesh.devices.take([i], axis=axis_index),
              target_mesh.axis_names,
              axis_types=target_mesh.axis_types,
          )
          for i in range(target_mesh.devices.shape[axis_index])
      ]
      pinned_by_mesh = {
          _find_named_mesh(replica): replica for replica in replicas
      }
      missing = [
          submesh for submesh in submeshes if submesh not in pinned_by_mesh
      ]
      local_replicas = (
          replicas if missing else [pinned_by_mesh[s] for s in submeshes]
      )
      donors = _put_arrays(local_replicas)
      by_mesh = {donor[0].sharding.mesh: donor for donor in donors}
      if missing:
        specs = [array.sharding.spec for array in donors[0]]
        # Stripe leaves round-robin across surviving donors to spread DCN
        # egress bandwidth across slices.
        sources = [
            [donors[(k + j) % len(donors)][j] for j in range(len(specs))]
            for k in range(len(missing))
        ]
        target_shardings = [
            [jax.sharding.NamedSharding(submesh, spec) for spec in specs]
            for submesh in missing
        ]
        by_mesh.update(
            zip(missing, _put_arrays(sources, target_shardings), strict=True)
        )
      slices = [by_mesh[s] for s in submeshes]

    if len(slices) == 1:
      return _replace_arrays(active_trees[0], slices[0])
    concatenated = self._concatenate_by_mesh_axis.concatenate_by_mesh_axis(
        slices, axis_name
    )
    return _replace_arrays(active_trees[0], concatenated)

  def try_load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool | None = None,
  ) -> tuple[int | None, tree_types.PyTree | None]:
    """Restores `(step, state)` from the latest snapshot, or `(None, None)`.

    Args:
      abstract_state: An abstract representation of the state, used to provide
        the target shardings for the restored arrays on the TPU devices.
      reset_snapshot_state: If True, resets the snapshot to the restored state.
        If None, resets only when the mesh of `abstract_state` has more than
        one replica along the replica axis so newly recovered slices receive a
        local pinned-host backup while single-slice scale-downs skip the
        redundant D2H copy.

    Returns:
      `(step, restored_state)`, or `(None, None)` if no snapshot is available
      or all snapshot replicas are inactive (clearing that snapshot).

    Raises:
      jax.errors.JaxRuntimeError: If a JAX or Pathways runtime error occurs.
      ValueError: If `abstract_state` does not match the snapshot structure or
        shapes.
    """
    if reset_snapshot_state is None:
      mesh = _find_named_mesh(abstract_state)
      reset_snapshot_state = bool(
          mesh and mesh.shape[mesh.axis_names[self.replica_axis_index]] > 1
      )
    return self._load(
        abstract_state,
        reset_snapshot_state=reset_snapshot_state,
        allow_missing=True,
    )

  def join(self) -> None:
    """Blocks until all queued snapshot requests have been processed."""
    self._queue.join()

  def cancel_pending(self) -> None:
    """Cancels pending or in-flight snapshots, keeping the latest snapshot."""
    with self._lock:
      self._generation += 1
      with contextlib.suppress(queue.Empty):
        while True:
          self._queue.get_nowait()
          self._pending_count -= 1
          self._queue.task_done()

  @property
  def latest(self) -> training_metadata_types.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step = self._latest_snapshot
    return training_metadata_types.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )
