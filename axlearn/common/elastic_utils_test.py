# Copyright © 2025 Apple Inc.

"""Tests elastic_utils."""

import threading
from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import absltest

import numpy as np

from axlearn.common import elastic_utils, snapshot


class _FakeManager:
    """Mimics the parts of `pathwaysutils.elastic.manager.Manager` used by `elastic_utils`."""

    def __init__(self, *, total: int, active: list[int]):
        self.total_slice_count = total
        self.active_slice_count = len(active)
        self.active_slice_indices = set(active)
        self.new_slice_event = threading.Event()
        self.retry_kwargs = None

    def elastic_retry(self, **kwargs):
        self.retry_kwargs = kwargs
        return lambda fn: fn


class ElasticUtilsTest(absltest.TestCase):
    def tearDown(self):
        elastic_utils.elastic_manager = None
        super().tearDown()

    def test_disabled_is_noop(self):
        self.assertFalse(elastic_utils.elastic_enabled())
        self.assertEqual(elastic_utils.live_devices(), jax.devices())
        self.assertEqual(elastic_utils.grad_accumulation_steps(0), 0)
        self.assertEqual(elastic_utils.grad_accumulation_steps(4), 4)
        self.assertFalse(elastic_utils.is_scale_up_event())
        elastic_utils.wait_for_slices()

        def fn():
            return 1

        self.assertIs(elastic_utils.elastic_retry()(fn), fn)

    def test_not_initialized_without_pathways(self):
        # Without the Pathways backend (e.g. McJAX) elastic training is silently disabled.
        self.assertIsNone(elastic_utils.ensure_elastic_manager_initialized(True))
        self.assertFalse(elastic_utils.elastic_enabled())

    def test_grad_accumulation_steps_keeps_global_batch(self):
        elastic_utils.elastic_manager = _FakeManager(total=4, active=[0, 1, 2, 3])
        self.assertEqual(elastic_utils.grad_accumulation_steps(0), 0)
        self.assertEqual(elastic_utils.grad_accumulation_steps(3), 3)
        elastic_utils.elastic_manager = _FakeManager(total=4, active=[0, 2])
        self.assertEqual(elastic_utils.grad_accumulation_steps(0), 2)
        self.assertEqual(elastic_utils.grad_accumulation_steps(3), 6)
        # 4 slices worth of batch on 3 slices needs ceil(4 / 3) = 2 minibatches.
        elastic_utils.elastic_manager = _FakeManager(total=4, active=[1, 2, 3])
        self.assertEqual(elastic_utils.grad_accumulation_steps(0), 2)

    def test_total_slice_count(self):
        self.assertIsNone(elastic_utils.total_slice_count())
        elastic_utils.elastic_manager = _FakeManager(total=4, active=[0, 1])
        self.assertEqual(elastic_utils.total_slice_count(), 4)

    def test_grad_accumulation_steps_with_device_batch_sizes(self):
        elastic_utils.elastic_manager = _FakeManager(total=2, active=[0])
        # The per-device batch grows from 8 to 16: 2 minibatches of 8.
        self.assertEqual(elastic_utils.grad_accumulation_steps(0, device_batch_sizes=(8, 16)), 2)
        self.assertEqual(elastic_utils.grad_accumulation_steps(2, device_batch_sizes=(8, 16)), 4)
        # 8 -> 12 (3 -> 2 slices): 2 minibatches of 6.
        self.assertEqual(elastic_utils.grad_accumulation_steps(0, device_batch_sizes=(8, 12)), 2)
        # The steps must divide the live per-device batch: 5 minibatches of 1.
        self.assertEqual(elastic_utils.grad_accumulation_steps(0, device_batch_sizes=(3, 5)), 5)
        # An unchanged per-device batch keeps the configured steps.
        self.assertEqual(elastic_utils.grad_accumulation_steps(0, device_batch_sizes=(8, 8)), 0)
        self.assertEqual(elastic_utils.grad_accumulation_steps(2, device_batch_sizes=(8, 8)), 2)

    def test_live_devices_filters_inactive_slices(self):
        elastic_utils.elastic_manager = _FakeManager(total=2, active=[1])
        devices = [mock.Mock(slice_index=0), mock.Mock(slice_index=1)]
        with mock.patch.object(jax, "devices", return_value=devices):
            self.assertEqual(elastic_utils.live_devices(), devices[1:])

    def test_is_scale_up_event(self):
        mgr = elastic_utils.elastic_manager = _FakeManager(total=2, active=[0])
        self.assertFalse(elastic_utils.is_scale_up_event())
        mgr.new_slice_event.set()
        self.assertTrue(elastic_utils.is_scale_up_event())
        mgr.new_slice_event.clear()
        mgr.available_inactive_slices = frozenset({1})  # pathwaysutils >= 0.1.12.
        self.assertTrue(elastic_utils.is_scale_up_event())

    def test_elastic_retry_maps_maxtext_parameters(self):
        mgr = elastic_utils.elastic_manager = _FakeManager(total=2, active=[0, 1])
        elastic_utils.elastic_retry(min_slice_count=-1, timeout_seconds=300, max_retries=10)
        self.assertIsNone(mgr.retry_kwargs["minimum_slice_count"])
        self.assertEqual(mgr.retry_kwargs["timeout"], 300)
        self.assertEqual(mgr.retry_kwargs["max_retries"], 10)
        elastic_utils.elastic_retry(min_slice_count=1)
        self.assertEqual(mgr.retry_kwargs["minimum_slice_count"], 1)

    def test_event_cleanup_keeps_host_arrays(self):
        x = jnp.ones(4)  # On CPU every array is host-resident and must survive the cleanup.
        elastic_utils._elastic_event_cleanup()  # pylint: disable=protected-access
        self.assertFalse(x.is_deleted())

    def test_snapshot_restore_places_surviving_slice_first(self):
        # When Slice 0 is replaced and Slice 1 holds the surviving pinned_host replica,
        # Snapshotter.restore must populate Slice 1's device shard locally from its own
        # pinned_host shard first rather than caching Slice 0's shard and bouncing back.
        d0, d1 = mock.Mock(id=0), mock.Mock(id=1)
        pinned_shard = mock.Mock(name="slice1_pinned_shard")
        replica_shard = mock.Mock(index=(slice(None),), data=pinned_shard)
        replica = mock.Mock(
            sharding=mock.Mock(mesh=mock.Mock(devices=np.array([d1]))),
            addressable_shards=[replica_shard],
        )
        pinned_arr = mock.MagicMock(
            spec=jax.Array,
            sharding=mock.Mock(mesh=mock.Mock(shape={"data": 2}, devices=np.array([d0, d1]))),
        )
        target_sharding = mock.Mock()
        target_sharding.with_memory_kind.return_value = target_sharding
        target_sharding.mesh.devices = np.array([d0, d1])
        target_sharding.addressable_devices = [d0, d1]
        target_sharding.addressable_devices_indices_map.return_value = {
            d0: (slice(None),),
            d1: (slice(None),),
        }
        spec = mock.Mock(shape=(4,), sharding=target_sharding)

        snap = snapshot.Snapshotter.default_config().set(name="snap").instantiate(parent=None)
        snap._latest_snapshot = ({"w": pinned_arr}, 5)  # pylint: disable=protected-access

        put_calls = []

        def fake_device_put(src, dst_sharding):
            out = mock.Mock(name=f"hbm_shard_dev{dst_sharding.dev.id}")
            put_calls.append((src, dst_sharding.dev.id, out))
            return out

        def fake_single_device_sharding(dev):
            s = mock.Mock(dev=dev)
            s.with_memory_kind.return_value = s
            return s

        with (
            mock.patch.object(
                snapshot,
                "split_by_mesh_axis",
                mock.Mock(split_by_mesh_axis=mock.Mock(return_value=[replica])),
            ),
            mock.patch.object(jax.sharding, "SingleDeviceSharding", side_effect=fake_single_device_sharding),
            mock.patch.object(jax, "device_put", side_effect=fake_device_put),
            mock.patch.object(jax, "make_array_from_single_device_arrays", side_effect=lambda shape, sh, shards: shards),
            mock.patch.object(jax, "block_until_ready", side_effect=lambda x: x),
        ):
            step, _ = snap.restore(state={"w": spec})

        self.assertEqual(step, 5)
        self.assertLen(put_calls, 2)
        # First device_put is onto surviving Slice 1 (d1) directly from its local pinned_shard.
        self.assertEqual(put_calls[0][0], pinned_shard)
        self.assertEqual(put_calls[0][1], 1)
        # Second device_put is onto recovered Slice 0 (d0) from Slice 1's HBM shard.
        self.assertEqual(put_calls[1][0], put_calls[0][2])
        self.assertEqual(put_calls[1][1], 0)


if __name__ == "__main__":
    absltest.main()
