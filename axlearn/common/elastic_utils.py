# Copyright © 2025 Apple Inc.
"""Elastic training utilities for the Pathways backend (no-ops on McJAX)."""
# pylint: disable=protected-access

from collections.abc import Sequence
import itertools
from typing import Any, Optional

from absl import flags, logging
import jax
from pathwaysutils.elastic import elastic, manager

from axlearn.common import gradient_accumulation as ga
from axlearn.common import measurement, utils
from axlearn.common.config import config_for_class

# Optional explicit manager reference; falls back to `manager.get_default_manager()` when unset.
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
    # pylint: disable-next=import-error,import-outside-toplevel
    from orbax.checkpoint.experimental.v1.training.pathways import Snapshotter

    cfg.set(
        elastic_enabled=True, elastic_min_slice_count=fv.elastic_min_slice_count,
        elastic_scale_grad_accumulation=fv.elastic_scale_grad_accumulation,
    )
    cfg.checkpointer.skip_uncommitted_checkpoint = False
    if fv.elastic_backup_kind == "snapshot":
        dims = getattr(cfg.mesh_shape, "dcn_mesh_shape", cfg.mesh_shape)
        ax = dims.index(-1) if -1 in dims else next((i for i, d in enumerate(dims) if d != 1), 0)
        cfg.checkpointer.set(
            replica_axis_index=ax, snapshot_every_n_steps=fv.elastic_snapshot_every_n_steps,
            snapshotter=config_for_class(Snapshotter).set(
                replica_axis_index=ax,
                copy_device_arrays_for_async_transfer=fv.elastic_snapshot_stage_in_hbm,
            ),
        )


def reconfigure_mesh(trainer: Any, live_devs: Sequence[jax.Device]) -> int:
    cfg, mgr = trainer._config, trainer._elastic_manager
    devs = utils.create_device_mesh(mesh_shape=cfg.mesh_shape, devices=live_devs)
    trainer._mesh = jax.sharding.Mesh(devs, cfg.mesh_axis_names)
    with trainer.mesh():
        n = getattr(cfg.learner.forward_fn_transformation, "steps", 0) or 0
        if mgr is not None and cfg.elastic_scale_grad_accumulation:
            disp = getattr(trainer.input, "input_dispatcher", None)
            n = mgr.scale_grad_accumulation_steps(
                n, batch_sizes=getattr(disp, "device_batch_sizes", None)
            )
        acc = ga.MetricAccumulator.default_config()
        trainer.learner._forward_fn_transformation = (
            ga.with_minibatch_steps(steps=n, metric_accumulator=acc) if n > 0 else lambda f: f
        )
        specs = trainer._trainer_state_specs
        trainer._trainer_state_partition_specs = jax.tree.map(lambda s: s.sharding, specs)
        key = tuple(d.id for d in devs.flat)
        trainer._compiled_train_step = trainer._compiled_train_steps.get(key)
        trainer._jit_train_step = trainer._pjit_train_step()
    return n


def aot_compile(trainer: Any) -> None:
    mgr = trainer._elastic_manager
    if mgr is None or trainer.config.cache_compiled_train_step is False:
        return
    subsets = mgr.aot_slice_subsets(trainer._config.elastic_min_slice_count)
    for devs in itertools.chain(subsets, [list(trainer._mesh.devices.flat)]):
        n = reconfigure_mesh(trainer, devs)
        trainer._step_log("[ELASTIC][AOT] devices=%d minibatch_steps=%d", len(devs), n)
        trainer._get_compiled_train_step_fn()


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
            n = reconfigure_mesh(trainer, live) if changed else None
            sfx = f"minibatch_steps={n}" if changed else "unchanged; reusing mesh."
            trainer._step_log("[ELASTIC][SCALE] devices=%s %s", len(live), sfx)
        finally:
            trainer._maybe_record_event(measurement.Event.END_ACCELERATOR_INIT)
        trainer._maybe_record_event(measurement.Event.START_TRAINING_PREPARATION)
        try:
            trainer.checkpointer.cancel(wait=not changed)
            if trainer.restore_checkpoint() is None:
                raise RuntimeError("Failed to restore from both snapshot and checkpoint.")
        finally:
            trainer._maybe_record_event(measurement.Event.END_TRAINING_PREPARATION)
    finally:
        trainer._maybe_record_event(measurement.Event.END_ELASTIC_REINIT)
        trainer._record_slice_counts(stepping=True)


def wrap_train_loop(trainer: Any, train_loop: Any) -> Any:
    def _in_mesh():
        with trainer.mesh():
            return train_loop()
    return trainer._elastic_manager.elastic_retry(
        minimum_slice_count=trainer._config.elastic_min_slice_count, poll_interval=5,
        pre_callback=lambda: restore_after_event(trainer),
        on_elastic_event_callback=trainer._on_elastic_event,
    )(_in_mesh)
