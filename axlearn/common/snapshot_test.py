# Copyright © 2025 Apple Inc.

"""Tests for axlearn.common.snapshot (direct and HBM-staged in-memory snapshots)."""

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
        for got, want in zip(
            jax.tree.leaves(snap.latest_snapshot_state), jax.tree.leaves(expected)
        ):
            self.assertEqual(got.sharding.memory_kind, "pinned_host")
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    @parameterized.parameters(False, True)
    def test_save_supersede_and_restore(self, stage_in_hbm):
        snap = _snapshotter(stage_in_hbm=stage_in_hbm)
        state = _make_state(self.mesh)
        snap.save(step=1, state=state)
        self._assert_snapshot_equals(snap, state, 1)
        first = snap.latest_snapshot_state

        state2 = _make_state(self.mesh, scale=2.0)
        snap.save(step=2, state=state2)
        self._assert_snapshot_equals(snap, state2, 2)
        # The superseded host snapshot is released.
        self.assertTrue(all(x.is_deleted() for x in jax.tree.leaves(first)))

        step, restored = snap.restore(state=jax.tree.map(_spec, state2))
        self.assertEqual(step, 2)
        for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(state2)):
            self.assertEqual(got.sharding.memory_kind, "device")
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_staged_snapshot_survives_donating_step(self):
        """The staged copy decouples the snapshot from the live, donated trainer state."""
        snap = _snapshotter(stage_in_hbm=True)
        state = _make_state(self.mesh)
        expected = jax.tree.map(np.asarray, state)
        snap.save(step=1, state=state)
        # The save never blocks on the live arrays: donate them to a step right away.
        new_state = jax.jit(lambda s: jax.tree.map(lambda x: x * 10.0, s), donate_argnums=(0,))(
            state
        )
        np.testing.assert_array_equal(
            np.asarray(new_state["model"]["b"]), expected["model"]["b"] * 10.0
        )
        self._assert_snapshot_equals(snap, expected, 1)

    def test_staged_copies_are_released(self):
        snap = _snapshotter(stage_in_hbm=True)
        state = _make_state(self.mesh)
        staged = []
        orig = snap._stage  # pylint: disable=protected-access

        def spy(tree):
            staged.append(orig(tree))
            return staged[-1]

        with mock.patch.object(snap, "_stage", side_effect=spy):
            snap.save(step=1, state=state, force=True)
        self.assertLen(staged, 1)
        self.assertTrue(all(x.is_deleted() for x in jax.tree.leaves(staged[0])))
        self.assertFalse(any(x.is_deleted() for x in jax.tree.leaves(state)))
        self._assert_snapshot_equals(snap, state, 1)

    def test_cancelled_staged_task_frees_staging_copy(self):
        snap = _snapshotter(stage_in_hbm=True)
        staged = snap._stage(_make_state(self.mesh))  # pylint: disable=protected-access
        # A task from a stale generation must be dropped and its HBM copy freed.
        snap._queue.put((staged, 1, snap._generation - 1, True, False))  # pylint: disable=protected-access
        snap.wait_until_finished()
        self.assertIsNone(snap.latest_snapshot_step)
        self.assertTrue(all(x.is_deleted() for x in jax.tree.leaves(staged)))

    @parameterized.parameters(False, True)
    def test_failed_snapshot_preserves_previous(self, stage_in_hbm):
        snap = _snapshotter(stage_in_hbm=stage_in_hbm)
        state1 = _make_state(self.mesh)
        snap.save(step=1, state=state1, force=True)
        with mock.patch("jax.block_until_ready", side_effect=RuntimeError("slice down")):
            snap.save(step=2, state=_make_state(self.mesh, scale=2.0), force=True)
        self._assert_snapshot_equals(snap, state1, 1)

    @parameterized.parameters(False, True)
    def test_hold_previous_keeps_arrays_alive(self, stage_in_hbm):
        snap = _snapshotter(stage_in_hbm=stage_in_hbm)
        snap.save(step=1, state=_make_state(self.mesh), force=True)
        first = snap.latest_snapshot_state
        snap.save(step=2, state=_make_state(self.mesh, scale=2.0), force=True, hold_previous=True)
        second = snap.latest_snapshot_state
        self.assertEqual(snap.latest_snapshot_step, 2)
        self.assertFalse(any(x.is_deleted() for x in jax.tree.leaves(first)))
        # Subsequent hold_previous=True step keeps `first` alive while freeing intermediate `second`.
        snap.save(step=3, state=_make_state(self.mesh, scale=3.0), force=True, hold_previous=True)
        self.assertFalse(any(x.is_deleted() for x in jax.tree.leaves(first)))
        self.assertTrue(all(x.is_deleted() for x in jax.tree.leaves(second)))
        # Once checkpointing finishes (hold_previous=False), `first` is also freed.
        snap.save(step=4, state=_make_state(self.mesh, scale=4.0), force=True, hold_previous=False)
        self.assertTrue(all(x.is_deleted() for x in jax.tree.leaves(first)))

    @parameterized.parameters(
        dict(temp_gib=8, state_gib=2, kind="TPU v5e", expect_staged=True),
        dict(temp_gib=14, state_gib=3, kind="TPU v5e", expect_staged=False),
        dict(temp_gib=91, state_gib=14, kind="TPU v5p", expect_staged=False),
        dict(temp_gib=66, state_gib=12, kind="TPU v5p", expect_staged=True),
        dict(temp_gib=130, state_gib=25, kind="TPU 7x", expect_staged=True),
        dict(temp_gib=150, state_gib=25, kind="TPU 7x", expect_staged=False),
    )
    def test_check_hbm_headroom(self, temp_gib, state_gib, kind, expect_staged):
        snap = _snapshotter(stage_in_hbm=True)
        gib = 2**30
        compiled = mock.Mock()
        compiled.memory_analysis.return_value = mock.Mock(
            argument_size_in_bytes=state_gib * gib,
            output_size_in_bytes=state_gib * gib,
            alias_size_in_bytes=state_gib * gib,
            temp_size_in_bytes=temp_gib * gib,
            generated_code_size_in_bytes=0,
        )
        device = mock.Mock(device_kind=kind, memory_stats=mock.Mock(return_value=None))
        snap.check_hbm_headroom(compiled, device=device)
        self.assertEqual(snap._stage_in_hbm, expect_staged)  # pylint: disable=protected-access

    def test_check_hbm_headroom_prefers_memory_stats(self):
        snap = _snapshotter(stage_in_hbm=True)
        compiled = mock.Mock()
        compiled.memory_analysis.return_value = mock.Mock(
            argument_size_in_bytes=10,
            output_size_in_bytes=10,
            alias_size_in_bytes=10,
            temp_size_in_bytes=10,
            generated_code_size_in_bytes=0,
        )
        device = mock.Mock(
            device_kind="TPU v5p", memory_stats=mock.Mock(return_value={"bytes_limit": 25})
        )
        snap.check_hbm_headroom(compiled, device=device)
        self.assertFalse(snap._stage_in_hbm)  # pylint: disable=protected-access


if __name__ == "__main__":
    absltest.main()
