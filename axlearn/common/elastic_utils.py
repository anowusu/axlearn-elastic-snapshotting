# Copyright © 2025 Apple Inc.
"""Elastic training utilities for the Pathways backend (no-ops on McJAX)."""
# pylint: disable=protected-access

import contextlib
from typing import Any, Optional

import jax
import numpy as np
from absl import flags, logging
from pathwaysutils.elastic import elastic, manager

from axlearn.common import config, gradient_accumulation, measurement, utils

elastic_manager: Optional[manager.Manager] = None


def get_manager() -> Any:
    if elastic_manager is not None:
        return elastic_manager
    return getattr(manager, "get_default_manager", lambda: None)()


def elastic_enabled() -> bool:
    return get_manager() is not None


def available_slice_count() -> Optional[int]:
    """Returns the number of healthy slices, or None when not elastic or if the check fails."""
    mgr = get_manager()
    if mgr is None:
        return None
    try:
        return len(elastic.get_active_slice_indices(slice_to_devices=mgr.slice_to_devices))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to check the available slices: %s", e)
        return None


def configure_trainer(cfg: Any, fv: flags.FlagValues) -> None:
    cfg.elastic_enabled, cfg.elastic_min_slice_count = True, fv.elastic_min_slice_count
    cfg.elastic_scale_grad_accumulation = fv.elastic_scale_grad_accumulation
    cfg.checkpointer.skip_uncommitted_checkpoint = False
    if fv.elastic_backup_kind == "snapshot":
        from orbax.checkpoint.experimental.v1.training.pathways import Snapshotter

        dims = getattr(cfg.mesh_shape, "dcn_mesh_shape", cfg.mesh_shape)
        ax = dims.index(-1) if -1 in dims else next((i for i, d in enumerate(dims) if d != 1), 0)
        snap = config.config_for_class(Snapshotter).set(replica_axis_index=ax)
        snap.copy_device_arrays_for_async_transfer = fv.elastic_snapshot_stage_in_hbm
        cfg.checkpointer.set(replica_axis_index=ax, snapshotter=snap)
        cfg.checkpointer.snapshot_every_n_steps = fv.elastic_snapshot_every_n_steps


def input_mesh(trainer: Any, input_cfg: Any) -> Any:
    mgr, cfg = trainer._elastic_manager, trainer._config
    if mgr is None or jax.process_count() != 1:
        return contextlib.nullcontext()
    if disp := getattr(input_cfg, "input_dispatcher", None):
        s0 = getattr(cfg.learner.forward_fn_transformation, "steps", 0) or 1
        disp.global_logical_batch_size //= mgr.total_slice_count * s0
        trainer._chunked_input = True
    devs = utils.create_device_mesh(cfg.mesh_shape, devices=mgr.slice_to_devices[0])
    return jax.sharding.Mesh(devs, cfg.mesh_axis_names)


def next_batch(trainer: Any, it: Any, *, specs: Any = None) -> Any:
    k = getattr(trainer, "_elastic_chunks", 1)
    if specs is not None:
        return jax.tree.map(
            lambda x: jax.ShapeDtypeStruct((k * x.shape[0], *x.shape[1:]), x.dtype), specs
        )
    if k == 1:
        return next(it)
    return jax.tree.map(lambda *a: np.concatenate(a), *[next(it) for _ in range(k)])


def reconfigure_mesh(trainer: Any, live: Any) -> tuple[int, int]:
    cfg, mgr = trainer._config, trainer._elastic_manager
    if not isinstance(live, np.ndarray):
        live = utils.create_device_mesh(cfg.mesh_shape, devices=live)
    trainer._mesh = jax.sharding.Mesh(live, cfg.mesh_axis_names)
    with trainer.mesh():
        s0 = getattr(cfg.learner.forward_fn_transformation, "steps", 0) or 0
        sc = cfg.elastic_scale_grad_accumulation
        b, n, c = mgr.plan_step_scaling(s0, num_devices=live.size, scale=sc) if mgr else (1, s0, 1)
        trainer._elastic_chunks = c if getattr(trainer, "_chunked_input", False) else 1
        acc = gradient_accumulation.MetricAccumulator.default_config()
        xform = gradient_accumulation.with_minibatch_steps(steps=n, metric_accumulator=acc)
        trainer.learner._forward_fn_transformation = xform if n > 0 else lambda f: f
        specs = trainer._trainer_state_specs
        trainer._trainer_state_partition_specs = jax.tree.map(lambda s: s.sharding, specs)
        trainer._compiled_train_step = trainer._compiled_train_steps.get(
            tuple(d.id for d in live.flat)
        )
        trainer._jit_train_step = trainer._pjit_train_step()
    return b, n


def aot_compile(trainer: Any) -> None:
    if (mgr := trainer._elastic_manager) is None:
        return
    reconfigure_mesh(trainer, full := trainer._mesh.devices)
    if trainer.config.cache_compiled_train_step is not False:
        trainer._get_compiled_train_step_fn()
        for d in mgr.aot_slice_subsets(trainer._config.elastic_min_slice_count):
            b, n = mgr.compile_with_backoff(
                lambda d=d: (reconfigure_mesh(trainer, d), trainer._get_compiled_train_step_fn())[0]
            )
            trainer._step_log("[ELASTIC][AOT] devices=%d batch=%dx steps=%d", len(d), b, n)
        reconfigure_mesh(trainer, full)


def restore_after_event(trainer: Any) -> None:
    live = trainer._elastic_manager.active_devices
    changed = {d.id for d in live} != {d.id for d in trainer._mesh.devices.flat}
    if trainer._trainer_state is not None and not changed:
        return
    if trainer._elastic_wait_event_type is not None:
        trainer._maybe_record_event(
            measurement.Event.END_ELASTIC_WAIT, trainer._elastic_wait_event_type
        )
        trainer._elastic_wait_event_type = None
    trainer._maybe_record_event(measurement.Event.START_ELASTIC_REINIT)
    try:
        trainer._record_slice_counts(stepping=False)
        trainer._maybe_record_event(measurement.Event.START_ACCELERATOR_INIT)
        try:
            b, n = reconfigure_mesh(trainer, live) if changed else (1, None)
            sfx = f"batch={b}x minibatch_steps={n}" if changed else "unchanged; reusing mesh."
            trainer._step_log("[ELASTIC][SCALE] devices=%s %s", len(live), sfx)
        finally:
            trainer._maybe_record_event(measurement.Event.END_ACCELERATOR_INIT)
        trainer._maybe_record_event(measurement.Event.START_TRAINING_PREPARATION)
        try:
            trainer.checkpointer.cancel()
            if trainer.restore_checkpoint() is None:
                raise RuntimeError("Failed to restore from both snapshot and checkpoint.")
        finally:
            trainer._maybe_record_event(measurement.Event.END_TRAINING_PREPARATION)
    finally:
        trainer._maybe_record_event(measurement.Event.END_ELASTIC_REINIT)
        trainer._record_slice_counts(stepping=True)


def wrap_train_loop(trainer: Any, train_loop: Any) -> Any:
    if (mgr := trainer._elastic_manager) is None:
        return train_loop

    def _in_mesh():
        with trainer.mesh():
            return train_loop()

    kw = dict(minimum_slice_count=trainer._config.elastic_min_slice_count, poll_interval=5)
    return mgr.elastic_retry(
        **kw,
        pre_callback=lambda: restore_after_event(trainer),
        on_elastic_event_callback=trainer._on_elastic_event,
    )(_in_mesh)
