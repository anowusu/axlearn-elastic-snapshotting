# Copyright © 2025 Apple Inc.

"""Tests elastic_utils."""

import threading
from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import absltest

from axlearn.common import elastic_utils


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


if __name__ == "__main__":
    absltest.main()
