# Copyright © 2024 Apple Inc.

"""Tests orbax checkpointer.

See also checkpointer_test.py for common checkpointing tests.
"""

# pylint: disable=protected-access

import os
import tempfile
from typing import Sequence
from unittest import mock

import jax
import orbax.checkpoint as ocp
from absl.testing import absltest
from jax import numpy as jnp
from jax.experimental import mesh_utils

from axlearn.common import checkpointer_orbax, test_utils
from axlearn.common.checkpointer import read_index_file
from axlearn.common.checkpointer_orbax import OrbaxCheckpointer, _CheckpointManagerWithTrackerFile
from axlearn.common.config import config_for_function


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

    def test_logger(self):
        """Tests that save and restore statistics are logged to `logger`."""
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        class _ListLogger(ocp.logging.AbstractLogger):
            def __init__(self):
                self.entries = []

            def log_entry(self, entry):
                self.entries.append(entry)

        logger = _ListLogger()
        with _mesh(mesh_shape), tempfile.TemporaryDirectory() as temp_dir:
            ckpt = (
                OrbaxCheckpointer.default_config()
                .set(name="test", dir=temp_dir, logger=config_for_function(lambda: logger))
                .instantiate(parent=None)
            )
            state = dict(x=jnp.ones([3, 2]))
            ckpt.save(step=1, state=state)
            ckpt.wait_until_finished()
            # The logger is kept when the manager is recreated.
            ckpt._create_manager()
            ckpt.restore(step=1, state=state)

        self.assertEqual(["save", "restore"], [entry["event_type"] for entry in logger.entries])
        self.assertIn("checkpoint_manager_blocking_start_time", logger.entries[0])
        self.assertIn("checkpoint_manager_start_time", logger.entries[1])

    def test_logger_restore_failure(self):
        """Tests that failing to log restore statistics does not fail the restore."""
        mesh_shape = (1, 1)
        if not test_utils.is_supported_mesh_shape(mesh_shape):
            return

        class _FailingRestoreLogger(ocp.logging.AbstractLogger):
            def __init__(self):
                self.entries = []

            def log_entry(self, entry):
                if entry["event_type"] == "restore":
                    raise RuntimeError("Cloud Logging unavailable")
                self.entries.append(entry)

        logger = _FailingRestoreLogger()
        with _mesh(mesh_shape), tempfile.TemporaryDirectory() as temp_dir:
            ckpt = (
                OrbaxCheckpointer.default_config()
                .set(name="test", dir=temp_dir, logger=config_for_function(lambda: logger))
                .instantiate(parent=None)
            )
            state = dict(x=jnp.ones([3, 2]))
            ckpt.save(step=1, state=state)
            ckpt.wait_until_finished()
            with mock.patch.object(checkpointer_orbax.logging, "warning") as mock_warning:
                step, restored = ckpt.restore(step=1, state=state)

        self.assertEqual(1, step)
        self.assertNestedAllClose(state, restored)
        self.assertEqual(["save"], [entry["event_type"] for entry in logger.entries])
        mock_warning.assert_any_call(
            "Failed to log checkpoint restore statistics: %s", mock.ANY
        )

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


if __name__ == "__main__":
    absltest.main()
