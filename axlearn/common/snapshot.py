# Copyright © 2024 Apple Inc.

"""Manages backups of JAX array states to pinned host memory."""

import logging
from typing import Any

from etils import epath
import jax
from orbax.checkpoint.experimental.v1 import training  # pytype: disable=import-error
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types  # pytype: disable=import-error

_logger = logging.getLogger(__name__)


class Snapshotter:
  """Manages backups of JAX array states to pinned host memory."""

  def __init__(self, *, replica_axis_index: int = 0):
    self._latest_snapshot: tuple[tree_types.PyTree, int] | None = None
    self.replica_axis_index = replica_axis_index

  def save_pytree(
      self, step: int, state: tree_types.PyTreeOf[jax.Array]
  ) -> None:
    """Move arrays onto CPU worker devices."""
    pinned_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host"), state
    )
    pinned_state = jax.device_put(state, pinned_shardings)
    self._latest_snapshot = (pinned_state, step)

  @classmethod
  def heal_pytree(
      cls,
      host_snapshot: tree_types.PyTree,
      abstract_target_state: tree_types.PyTreeOf[jax.Array],
      *,
      replica_axis_index: int = 0,
  ) -> tree_types.PyTree:
    """Heals a PyTree host snapshot by discarding dead shards and re-sharding onto the target mesh."""
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
        try:
          jax.block_until_ready(arr)
          return True
        except jax.errors.JaxRuntimeError as _:
          return False

      def get_active_pytree(x):
        if not hasattr(x, "sharding") or x.sharding is None:
          return x
        mesh_axis_name = x.sharding.mesh.axis_names[replica_axis_index]
        all_replicas = split_by_mesh_axis.split_by_mesh_axis(x, mesh_axis_name)
        active_replicas = [r for r in all_replicas if is_replica_active(r)]
        if not active_replicas:
          raise RuntimeError("No active replicas found.")
        return concatenate_by_mesh_axis.concatenate_by_mesh_axis(active_replicas, mesh_axis_name)

      surviving_shards = jax.tree.map(get_active_pytree, host_snapshot)
      return jax.tree.map(
          lambda state, abstract: jax.device_put(state, abstract.sharding) if hasattr(abstract, "sharding") and abstract.sharding is not None else state,
          surviving_shards,
          abstract_target_state,
      )
    else:
      return jax.tree.map(
          lambda state, abstract: jax.device_put(state, abstract.sharding) if hasattr(abstract, "sharding") and abstract.sharding is not None else state,
          host_snapshot,
          abstract_target_state,
      )

  def load_pytree(
      self,
      abstract_state: tree_types.PyTreeOf[jax.Array],
      *,
      reset_snapshot_state: bool = True,
  ) -> tree_types.PyTree:
    """Move arrays from workers onto TPU devices."""
    if self._latest_snapshot is None:
      raise RuntimeError("No snapshots available to restore from.")
    pinned_state, step = self._latest_snapshot

    _logger.info("Restoring from snapshot at step %d...", step)
    restored_state = self.heal_pytree(
        pinned_state, abstract_state, replica_axis_index=self.replica_axis_index
    )

    if reset_snapshot_state:
      host_target_shardings = jax.tree.map(
          lambda x: x.sharding.with_memory_kind("pinned_host"), abstract_state
      )
      host_target_state = jax.device_put(restored_state, host_target_shardings)
      self._latest_snapshot = (host_target_state, step)

    return restored_state

  def join(self) -> None:
    """Blocks until all snapshots in the queue are ready and secured."""
    pass

  @property
  def latest(self) -> training.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    if self._latest_snapshot is None:
      return None
    _, step = self._latest_snapshot
    return training.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )