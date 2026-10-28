# Copyright © 2025 Apple Inc.

"""Tests elastic_utils."""

import os
import tempfile
import threading
from unittest import mock

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
from absl.testing import absltest

import numpy as np

from axlearn.common import elastic_utils, learner, optimizers, param_init, snapshot
from axlearn.common.checkpointer import Checkpointer, every_n_steps_policy
from axlearn.common.config import config_for_function
from axlearn.common.elastic_input import ElasticInput, ElasticSpmdInputDispatcher
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.trainer_test import DummyInput, DummyModel


class _FakeManager:
    """Mimics the parts of `pathwaysutils.elastic.manager.Manager` used by `elastic_utils`."""

    def __init__(self, *, total: int, active: list[int], slice_to_devices=None):
        self.total_slice_count = total
        self.active_slice_count = len(active)
        self.active_slice_indices = set(active)
        self.slice_to_devices = slice_to_devices or {}
        self.available_inactive_slices: frozenset[int] = frozenset()
        self.retry_kwargs = None

    @property
    def active_devices(self) -> list[jax.Device]:
        return [
            d
            for s in sorted(self.active_slice_indices)
            for d in self.slice_to_devices.get(s, ())
        ]

    def elastic_retry(self, **kwargs):
        self.retry_kwargs = kwargs
        return lambda fn: fn


def _elastic_trainer_fixture(**overrides):
    """Returns (devices, devices_per_slice, fake manager, SpmdTrainer config) for a 2-slice job.

    The fake manager's `elastic_retry` retries `fn` across slice-down and scale-up events the way
    `pathwaysutils` does, so `SpmdTrainer.run` can be exercised end to end on CPU devices.
    """
    devs = jax.devices()[:8] if jax.device_count() >= 8 else jax.devices()[:2]

    half = len(devs) // 2

    class _StepRetryManager(_FakeManager):
        def __init__(self):
            super().__init__(
                total=2,
                active=[0, 1],
                slice_to_devices={0: devs[:half], 1: devs[half:]},
            )
            self.default_device = devs[0]

        def elastic_retry(self, *, pre_callback=None, on_elastic_event_callback=None, **kwargs):
            self.retry_kwargs = kwargs

            def decorator(fn):
                def wrapped():
                    while True:
                        if pre_callback:
                            pre_callback()
                        try:
                            return fn()
                        except (jax.errors.JaxRuntimeError, elastic_utils.ScaleUpSignalError):
                            if on_elastic_event_callback:
                                on_elastic_event_callback()

                return wrapped

            return decorator

    mgr = elastic_utils.elastic_manager = _StepRetryManager()
    input_cfg = DummyInput.default_config().set(
        input_dispatcher=ElasticSpmdInputDispatcher.default_config().set(
            global_logical_batch_size=len(devs),
            num_max_slices=2,
        )
    )
    cfg = SpmdTrainer.default_config().set(
        name="elastic_aot_trainer",
        dir=tempfile.mkdtemp(),
        mesh_axis_names=("data", "model"),
        mesh_shape=(-1, half),
        batch_axis_names=("data",),
        model=DummyModel.default_config().set(
            dtype=jnp.float32, param_init=param_init.DefaultInitializer.default_config()
        ),
        input=ElasticInput.default_config().set(input=input_cfg),
        learner=learner.Learner.default_config().set(
            optimizer=config_for_function(optimizers.sgd_optimizer).set(
                learning_rate=0.1, decouple_weight_decay=True
            )
        ),
        checkpointer=Checkpointer.default_config().set(
            save_policy=config_for_function(every_n_steps_policy).set(n=100)
        ),
        snapshotter=snapshot.Snapshotter.default_config().set(
            save_policy=config_for_function(every_n_steps_policy).set(n=1)
        ),
        elastic_enabled=True,
        elastic_min_slice_count=1,
        max_step=4,
        **overrides,
    )
    return devs, half, mgr, cfg


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
        devices = [mock.Mock(slice_index=0), mock.Mock(slice_index=1)]
        elastic_utils.elastic_manager = _FakeManager(
            total=2, active=[1], slice_to_devices={0: [devices[0]], 1: [devices[1]]}
        )
        self.assertEqual(elastic_utils.live_devices(), devices[1:])

    def test_is_scale_up_event(self):
        mgr = elastic_utils.elastic_manager = _FakeManager(total=2, active=[0])
        self.assertFalse(elastic_utils.is_scale_up_event())
        mgr.available_inactive_slices = frozenset({1})
        self.assertTrue(elastic_utils.is_scale_up_event())

    def test_elastic_retry_maps_maxtext_parameters(self):
        mgr = elastic_utils.elastic_manager = _FakeManager(total=2, active=[0, 1])
        elastic_utils.elastic_retry(min_slice_count=-1, timeout_seconds=300, max_retries=10)
        self.assertIsNone(mgr.retry_kwargs["minimum_slice_count"])
        self.assertEqual(mgr.retry_kwargs["timeout"], 300)
        self.assertEqual(mgr.retry_kwargs["max_retries"], 10)
        elastic_utils.elastic_retry(min_slice_count=1)
        self.assertEqual(mgr.retry_kwargs["minimum_slice_count"], 1)

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
            shape=(4,),
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

    def test_spmd_trainer_aot_compile_and_zero_reinit_across_elastic_events(self):
        # Verifies that:
        # 1. _prepare_training AoT-compiles train steps for both N-1 and N slices at startup.
        # 2. @elastic_retry wrapping _step_loop performs zero re-initialization (init() called once,
        #    and trainer/model/learner/input_iter object IDs stay identical across scale-down & scale-up).
        # 3. Scale-down (N=2 -> N=1) and scale-up (N=1 -> N=2) hit _compiled_train_steps with 0 recompiles.
        if jax.device_count() < 2:
            self.skipTest("Requires at least 2 CPU devices")

        devs, half, mgr, cfg = _elastic_trainer_fixture()

        with mock.patch.object(jax, "devices", return_value=devs):
            trainer: SpmdTrainer = cfg.instantiate(parent=None)
            init_calls = [0]
            orig_init = trainer.init

            def counting_init(*args, **kwargs):
                init_calls[0] += 1
                return orig_init(*args, **kwargs)

            compile_calls = [0]
            orig_compile = trainer.compile_train_step

            def counting_compile(*args, **kwargs):
                compile_calls[0] += 1
                return orig_compile(*args, **kwargs)

            orig_run_step = trainer._run_step  # pylint: disable=protected-access
            preempted_once = [False]
            scaled_up_once = [False]
            compiles_after_prepare = [None]

            def hooked_run_step(input_batch, **kwargs):
                if compiles_after_prepare[0] is None:
                    compiles_after_prepare[0] = compile_calls[0]
                if trainer.step == 2 and not preempted_once[0]:
                    preempted_once[0] = True
                    mgr.active_slice_indices = {0}
                    mgr.active_slice_count = 1
                    raise jax.errors.JaxRuntimeError("UNAVAILABLE: slice 1 preempted at step 2")
                if trainer.step == 3 and not scaled_up_once[0]:
                    scaled_up_once[0] = True
                    mgr.active_slice_indices = {0, 1}
                    mgr.active_slice_count = 2
                    mgr.available_inactive_slices = frozenset({1})
                out = orig_run_step(input_batch, **kwargs)
                mgr.available_inactive_slices = frozenset()
                return out

            ids_before = (
                id(trainer),
                id(trainer.model),
                id(trainer.learner),
                id(trainer._input_iter),  # pylint: disable=protected-access
            )
            with (
                mock.patch.object(trainer, "init", side_effect=counting_init),
                mock.patch.object(trainer, "compile_train_step", side_effect=counting_compile),
                mock.patch.object(trainer, "_run_step", side_effect=hooked_run_step),
            ):
                trainer.run(jax.random.PRNGKey(0))
            ids_after = (
                id(trainer),
                id(trainer.model),
                id(trainer.learner),
                id(trainer._input_iter),  # pylint: disable=protected-access
            )

        self.assertEqual(trainer.step, 4)
        self.assertTrue(preempted_once[0])
        self.assertTrue(scaled_up_once[0])
        # 1. Zero re-initialization: init() ran only once at step 0, and all Python objects kept their IDs.
        self.assertEqual(init_calls[0], 1)
        self.assertEqual(ids_before, ids_after)
        # 2. Startup AoT compiled one N-1 subset and N slices = 2 compilations.
        self.assertEqual(compiles_after_prepare[0], 2)
        self.assertIsNotNone(trainer._compiled_train_step)  # pylint: disable=protected-access
        # 3. Scale-down (N=2 -> N=1) and scale-up (N=1 -> N=2) hit _compiled_train_steps with 0 recompiles.
        self.assertEqual(compile_calls[0], 2)

    def test_elastic_scale_grad_accumulation(self):
        # On half the slices the per-device batch doubles. By default the step is split into 2
        # minibatches to keep the microbatch constant; with scaling off the configured steps stay.
        if jax.device_count() < 2:
            self.skipTest("Requires at least 2 CPU devices")
        for scale, expected_steps in ((True, 2), (False, 0)):
            devs, half, mgr, cfg = _elastic_trainer_fixture(elastic_scale_grad_accumulation=scale)
            with mock.patch.object(jax, "devices", return_value=devs):
                trainer = cfg.instantiate(parent=None)
                mgr.active_slice_indices, mgr.active_slice_count = {0}, 1
                steps = trainer._reconfigure_elastic_mesh(devs[:half])  # pylint: disable=protected-access
            self.assertEqual(steps, expected_steps, msg=f"scale={scale}")

    def test_spmd_trainer_resets_stop_trace_step_across_elastic_events(self):
        # Verifies that resetting `stop_trace_step = None` on each `_step_loop` attempt prevents:
        # 1. Mid-trace slice-down (`JaxRuntimeError`) calling `stop_trace()` a second time after
        #    `Manager._elastic_event_cleanup()` already stopped the trace.
        # 2. Mid-trace scale-up (`ScaleUpSignalError`) crashing on `assert output is not None` when
        #    the restored step equals the previous attempt's `stop_trace_step`.
        # 3. Mid-trace rollback onto `start_trace_steps` skipping the fresh trace restart.
        if jax.device_count() < 2:
            self.skipTest("Requires at least 2 CPU devices")

        def run_case(*, start_steps, n_trace_steps, max_step, hook_fn, initial_active=(0, 1)):
            devs, _, mgr, cfg = _elastic_trainer_fixture(
                start_trace_steps=start_steps,
                n_steps_for_each_trace=n_trace_steps,
            )
            cfg.set(max_step=max_step)
            mgr.active_slice_indices = set(initial_active)
            mgr.active_slice_count = len(initial_active)
            trace_active = [False]
            counts = {"start": 0, "stop": 0}

            def fake_start_trace(*args, **kwargs):
                del args, kwargs
                if trace_active[0]:
                    raise RuntimeError("Trace already active")
                trace_active[0] = True
                counts["start"] += 1

            def fake_stop_trace(*args, **kwargs):
                del args, kwargs
                if not trace_active[0]:
                    raise RuntimeError("stop_trace called before a trace is being taken!")
                trace_active[0] = False
                counts["stop"] += 1

            orig_retry = mgr.elastic_retry

            def retry_with_profiler_cleanup(*, pre_callback=None, on_elastic_event_callback=None, **kw):
                def cleanup_and_cb():
                    # Mirrors pathwaysutils.elastic.manager._elastic_event_cleanup().
                    if trace_active[0]:
                        fake_stop_trace()
                    if on_elastic_event_callback:
                        on_elastic_event_callback()

                return orig_retry(
                    pre_callback=pre_callback,
                    on_elastic_event_callback=cleanup_and_cb,
                    **kw,
                )

            with (
                mock.patch.object(jax, "devices", return_value=devs),
                mock.patch.object(mgr, "elastic_retry", side_effect=retry_with_profiler_cleanup),
                mock.patch.object(jax.profiler, "start_trace", side_effect=fake_start_trace),
                mock.patch.object(jax.profiler, "stop_trace", side_effect=fake_stop_trace),
            ):
                trainer: SpmdTrainer = cfg.instantiate(parent=None)
                orig_run_step = trainer._run_step  # pylint: disable=protected-access
                with mock.patch.object(
                    trainer,
                    "_run_step",
                    side_effect=lambda batch, **kw: hook_fn(trainer, mgr, orig_run_step, batch, **kw),
                ):
                    trainer.run(jax.random.PRNGKey(0))

            self.assertEqual(trainer.step, max_step)
            self.assertFalse(trace_active[0])
            return counts

        # Case 1: Slice-down during step 3 (`stop_trace_step=3`) restores step 2 -> no double stop_trace().
        fired_down = [False]

        def slice_down_at_3(trainer, mgr, orig_run_step, batch, **kw):
            if trainer.step == 3 and not fired_down[0]:
                fired_down[0] = True
                mgr.active_slice_indices = {0}
                mgr.active_slice_count = 1
                raise jax.errors.JaxRuntimeError("UNAVAILABLE: slice 1 preempted mid-trace")
            return orig_run_step(batch, **kw)

        c1 = run_case(start_steps=[0], n_trace_steps=3, max_step=4, hook_fn=slice_down_at_3)
        self.assertTrue(fired_down[0])
        self.assertEqual(c1, {"start": 1, "stop": 1})

        # Case 2: Scale-up at end of step 2 (`stop_trace_step=2`) restores step 2 with `output=None`.
        fired_up = [False]

        def scale_up_at_2(trainer, mgr, orig_run_step, batch, **kw):
            if trainer.step == 2 and not fired_up[0]:
                fired_up[0] = True
                mgr.active_slice_indices = {0, 1}
                mgr.active_slice_count = 2
                mgr.available_inactive_slices = frozenset({1})
            out = orig_run_step(batch, **kw)
            mgr.available_inactive_slices = frozenset()
            return out

        c2 = run_case(
            start_steps=[0], n_trace_steps=2, max_step=4, hook_fn=scale_up_at_2, initial_active=(0,)
        )
        self.assertTrue(fired_up[0])
        self.assertEqual(c2, {"start": 1, "stop": 1})

        # Case 3: Slice-down during step 2 rolls back to `start_trace_steps=[1]` -> fresh trace runs.
        fired_rb = [False]

        def rollback_to_1(trainer, mgr, orig_run_step, batch, **kw):
            if trainer.step == 2 and not fired_rb[0]:
                fired_rb[0] = True
                mgr.active_slice_indices = {0}
                mgr.active_slice_count = 1
                raise jax.errors.JaxRuntimeError("UNAVAILABLE: slice 1 preempted at step 2")
            return orig_run_step(batch, **kw)

        c3 = run_case(start_steps=[1], n_trace_steps=2, max_step=4, hook_fn=rollback_to_1)
        self.assertTrue(fired_rb[0])
        self.assertEqual(c3, {"start": 2, "stop": 2})


if __name__ == "__main__":
    absltest.main()
