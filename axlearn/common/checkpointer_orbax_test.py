# Copyright © 2024 Apple Inc.

"""Tests orbax checkpointer.

See also checkpointer_test.py for common checkpointing tests.
"""

# pylint: disable=protected-access

import os
import tempfile
from typing import Sequence

import jax
import orbax.checkpoint as ocp
from absl.testing import absltest
from jax import numpy as jnp
from jax.experimental import mesh_utils

from unittest import mock

from axlearn.common import test_utils, utils
from axlearn.common.checkpointer import every_n_steps_policy, read_index_file
from axlearn.common.checkpointer_orbax import OrbaxCheckpointer, _CheckpointManagerWithTrackerFile
from axlearn.common.config import config_for_class, config_for_function
from orbax.checkpoint.experimental.v1.training.pathways import Snapshotter


def _mesh(mesh_shape: Sequence[int]):
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return jax.sharding.Mesh(devices, ("data", "model"))


class OrbaxCheckpointerTest(test_utils.TestCase):
    def test_index(self):
        """Tests that index files saved with orbax can be read with `read_index_file`."""
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape), tempfile.TemporaryDirectory() as temp_dir:
            ckpt = (
                OrbaxCheckpointer.default_config()
                .set(name="test", dir=temp_dir)
                .instantiate(parent=None)
            )
            step = 123
            state = dict(x=jnp.ones([3, 2]))
            ckpt.save(step=step, state=state)
            ckpt.wait_until_finished()

            ref_index = read_index_file(os.path.join(temp_dir, "step_00000123", "index"))
            test_index = ckpt._manager.restore(
                step=step,
                # The input iterator is saved as part of `save_tf_savables`.
                args=ocp.args.Composite(
                    index=ocp.args.JsonSave(ckpt._get_spec(step=step, state=state))
                ),
            )
            self.assertEqual(ref_index, test_index["index"])

    def test_checkpoint_manager_with_tracker_file(self):
        """Test CheckpointManagerWithTrackerFile"""
        n = 5

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _CheckpointManagerWithTrackerFile(
                directory=temp_dir,
                options=ocp.CheckpointManagerOptions(
                    enable_async_checkpointing=True,
                ),
            )
            for step in range(1, n + 1):
                manager.save(
                    step,
                    args=ocp.args.Composite(
                        state=ocp.args.PyTreeSave(item={"x": jnp.ones([3, 2]) * step}),
                    ),
                )
                manager.wait_until_finished()

            # a new one is created, assuming the job is restarted
            manager = _CheckpointManagerWithTrackerFile(directory=temp_dir)

            assert manager.latest_step() == n
            assert manager.all_steps() == [n]

    def test_unified_snapshotter_save_restore_and_gcs_fallback(self):
        """Tests unified Snapshotter + CheckpointManager coordination in OrbaxCheckpointer."""
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return
        with _mesh(mesh_shape) as mesh, tempfile.TemporaryDirectory() as temp_dir:
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            ckpt: OrbaxCheckpointer = (
                OrbaxCheckpointer.default_config()
                .set(
                    name="test_unified",
                    dir=temp_dir,
                    save_policy=config_for_function(every_n_steps_policy).set(n=2, min_step=2),
                    snapshotter=config_for_class(Snapshotter).set(save_interval_steps=1),
                )
                .instantiate(parent=None)
            )
            ckpt.snapshotter._split_by_mesh_axis = mock.Mock(
                split_by_mesh_axis=lambda x, _: [x]
            )
            ckpt.snapshotter._concatenate_by_mesh_axis = None

            state1 = dict(x=jax.device_put(jnp.ones([2], dtype=jnp.float32), sharding))
            state2 = dict(x=jax.device_put(jnp.ones([2], dtype=jnp.float32) * 2, sharding))
            state3 = dict(x=jax.device_put(jnp.ones([2], dtype=jnp.float32) * 3, sharding))
            spec = dict(x=utils.TensorSpec(shape=(2,), dtype=jnp.float32, mesh_axes=None))

            # Step 1: non-checkpoint step snapshots in memory without writing to disk.
            ckpt.save(step=1, state=state1)
            ckpt.wait_until_finished()
            self.assertEqual(ckpt.snapshotter.latest.step, 1)
            self.assertEqual(ckpt.checkpoint_steps(temp_dir), [])

            # Step 2: checkpoint step shares pinned_host snapshot buffers and commits to disk.
            ckpt.save(step=2, state=state2)
            ckpt.wait_until_finished()
            self.assertEqual(ckpt.snapshotter.latest.step, 2)
            self.assertEqual(ckpt.checkpoint_steps(temp_dir), [2])

            # Step 3: newer in-memory snapshot takes precedence over step-2 checkpoint and omits
            # input_iter so live host iterators remain untouched.
            ckpt.save(step=3, state=state3)
            ckpt.wait_until_finished()
            restored_step, restored_state = ckpt.restore(
                step=None, state=dict(**spec, input_iter=object())
            )
            self.assertEqual(restored_step, 3)
            self.assertNotIn("input_iter", restored_state)
            self.assertNestedEqual(state3, restored_state)

            # Evicting the snapshot falls back to the step-2 persistent checkpoint and re-seeds
            # the snapshot at step 2.
            ckpt.snapshotter.evict()
            restored_step, restored_state = ckpt.restore(step=None, state=spec)
            self.assertEqual(restored_step, 2)
            self.assertNestedEqual(state2, restored_state)
            self.assertEqual(ckpt.snapshotter.latest.step, 2)
            ckpt.stop()


if __name__ == "__main__":
    absltest.main()
