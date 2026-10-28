# Copyright © 2025 Apple Inc.

"""Tests for axlearn.common.snapshot (pinned-host Snapshotter and Pathways pause-resume helpers)."""

import os
from unittest import mock

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from axlearn.common.checkpointer import every_n_steps_policy
from axlearn.common.config import config_for_function
from axlearn.common.snapshot import Snapshotter


def _make_state(mesh: Mesh, *, scale: float = 1.0):
    sh = NamedSharding(mesh, P("fsdp"))
    return dict(
        model=dict(
            w=jax.device_put(jnp.arange(64, dtype=jnp.float32).reshape(8, 8) * scale, sh),
            b=jax.device_put(jnp.full((8,), 2.0 * scale, jnp.float32), sh),
        ),
        step=jax.device_put(jnp.asarray(3 * scale, jnp.float32), NamedSharding(mesh, P())),
    )


def _spec(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)


def _snapshotter(**kwargs) -> Snapshotter:
    cfg = Snapshotter.default_config().set(
        name="snap", save_policy=config_for_function(every_n_steps_policy).set(n=1), **kwargs
    )
    return cfg.instantiate(parent=None)


class SnapshotterTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.mesh = Mesh(np.array(jax.devices()[:8]).reshape(2, 4), ("data", "fsdp"))
        self.mesh.__enter__()

    def tearDown(self):
        self.mesh.__exit__(None, None, None)
        super().tearDown()

    def _assert_snapshot_equals(self, snap: Snapshotter, expected, step: int):
        snap.wait_until_finished()
        self.assertEqual(snap.latest_snapshot_step, step)
        pinned_state = snap._latest_snapshot[0]  # pylint: disable=protected-access
        for got, want in zip(jax.tree.leaves(pinned_state), jax.tree.leaves(expected)):
            self.assertEqual(got.sharding.memory_kind, "pinned_host")
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_save_supersede_and_restore(self):
        snap = _snapshotter()
        state = _make_state(self.mesh)
        snap.save(step=1, state=state)
        self._assert_snapshot_equals(snap, state, 1)
        first = snap._latest_snapshot[0]  # pylint: disable=protected-access

        state2 = _make_state(self.mesh, scale=2.0)
        snap.save(step=2, state=state2)
        self._assert_snapshot_equals(snap, state2, 2)
        self.assertTrue(all(x.is_deleted() for x in jax.tree.leaves(first)))

        step, restored = snap.restore(state=jax.tree.map(_spec, state2))
        self.assertEqual(step, 2)
        for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(state2)):
            self.assertEqual(got.sharding.memory_kind, "device")
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_failed_snapshot_preserves_previous(self):
        snap = _snapshotter()
        state1 = _make_state(self.mesh)
        snap.save(step=1, state=state1, force=True)
        # A slice lost while the new copy is in flight must not evict the previous snapshot.
        with mock.patch("jax.block_until_ready", side_effect=RuntimeError("slice down")):
            snap.save(step=2, state=_make_state(self.mesh, scale=2.0), force=True)
        self._assert_snapshot_equals(snap, state1, 1)

    def test_elastic_retry(self):
        snap = _snapshotter(max_retries=3, timeout_seconds=120)
        self.assertEqual(snap.elastic_retry()(lambda: 42)(), 42)
        mgr = mock.Mock()
        mgr.elastic_retry.side_effect = lambda **kw: lambda fn: fn
        snap._manager = mgr  # pylint: disable=protected-access
        self.assertEqual(snap.elastic_retry(pre_callback=None)(lambda: 42)(), 42)
        mgr.elastic_retry.assert_called_once_with(max_retries=3, timeout=120, pre_callback=None)


if __name__ == "__main__":
    absltest.main()
