# Copyright © 2023 Apple Inc.

"""Utilities to launch a trainer."""

import json
import os
from typing import Any, Optional

import jax
from absl import flags, logging

from axlearn.common import file_system as fs
from axlearn.common import measurement
from axlearn.common.config import TrainerConfigFn, get_named_trainer_config
from axlearn.common.trainer import SpmdTrainer, select_mesh_config
from axlearn.common.utils import MeshShape, get_data_dir, infer_mesh_shape

# Trainer-specific flags.
flags.DEFINE_string(
    "module",
    None,
    "The trainer config module. "
    "Only configs from the module will be loaded to avoid dependency on other modules.",
    required=True,
)
flags.DEFINE_alias("config_module", "module")
flags.DEFINE_string("config", None, "The trainer config name.", required=True)
flags.DEFINE_string(
    "trainer_dir",
    None,
    "The root directory of the trainer. "
    "Checkpoints will be stored in <dir>/checkpoints. "
    "Summaries will be stored in <dir>/summaries.",
    required=True,
)
flags.DEFINE_integer(
    "trainer_prng_seed",
    0,
    "The seed for jax.random.PRNGKey(). "
    "Used for initializing model parameters and pseudo-random number generation during training.",
)
flags.DEFINE_list("trace_at_steps", [], "Step numbers to start a 3-step profile at.")
flags.DEFINE_integer(
    "n_steps_for_each_trace",
    None,
    "Number of consecutive steps covered by each trace. If None, defaults to 3.",
)
flags.DEFINE_enum(
    "tpu_trace_mode",
    None,
    ["TRACE_ONLY_HOST", "TRACE_ONLY_XLA", "TRACE_COMPUTE", "TRACE_COMPUTE_AND_SYNC"],
    "TPU trace mode. If None, defaults to TRACE_ONLY_XLA. "
    "See https://docs.jax.dev/en/latest/profiling.html#tpu-options. ",
)
flags.DEFINE_enum(
    "host_tracer_level",
    None,
    ["0", "1", "2", "3"],
    "Host tracer level. Higher levels capture more host-side activity. "
    "If None, defaults to 2. See https://docs.jax.dev/en/latest/profiling.html#general-options.",
)
flags.DEFINE_enum(
    "device_tracer_level",
    None,
    ["0", "1"],
    "Device tracer level. If None, defaults to 1. "
    "See https://docs.jax.dev/en/latest/profiling.html#general-options.",
)
flags.DEFINE_enum(
    "python_tracer_level",
    None,
    ["0", "1"],
    "Python tracer level. If None, defaults to 0. "
    "See https://docs.jax.dev/en/latest/profiling.html#general-options.",
)
flags.DEFINE_list(
    "eval_trace_at_iters",
    [],
    "Evaluation iters to trace with the profiler each time the evaler is run. "
    "Each trace covers one eval batch. "
    "Traces will run for at most 3 unique steps.",
)
flags.DEFINE_integer(
    "trainer_watchdog_timeout_seconds",
    3600,
    "Timeout for the trainer watchdog in seconds. "
    "If the trainer.step does not increment within this interval, "
    "the watchdog will log the stack traces of all threads.",
)
flags.DEFINE_integer(
    "trainer_crash_on_hang_timeout_seconds",
    7200,
    "Timeout for crashing the trainer on hang in seconds. "
    "If the trainer hangs for longer than this interval, "
    "the trainer will crash to prevent indefinite hanging.",
)
flags.DEFINE_integer(
    "trainer_log_every_n_steps",
    None,
    "Logging frequency for the loss value during training. If None, defaults to every 100 steps.",
)
flags.DEFINE_enum(
    "device_monitor",
    "none",
    ["none", "tpu", "gpu"],
    "Whether to enable the device monitor. "
    "The device monitor collects the system metrics and logs them periodically. "
    "The device monitor also logs the idle status of the devices on the host, "
    "and trigger a watchdog if the devices are idle for 10 minutes.",
)
flags.DEFINE_string(
    "mesh_selector",
    None,
    "The mesh selector string. See `SpmdTrainer.Config.mesh_rules` for details.",
)
# Elastic training flags (Pathways only, a no-op on McJAX). Names follow MaxText's `elastic_*`
# config; see `axlearn.common.elastic_utils`.
flags.DEFINE_bool("elastic_enabled", False, "Enables elastic training on the Pathways backend.")
flags.DEFINE_integer(
    "elastic_min_slice_count", -1, "Minimum active slices to train on; -1 waits for all slices."
)
flags.DEFINE_float("elastic_timeout_seconds", None, "Max seconds to wait for slices per retry.")
flags.DEFINE_integer("elastic_max_retries", None, "Max retries after elastic events.")
flags.DEFINE_enum(
    "elastic_backup_kind",
    "snapshot",
    ["snapshot", "checkpoint"],
    "Recover elastic events from an in-memory snapshot or from the last checkpoint.",
)
flags.DEFINE_integer("elastic_snapshot_every_n_steps", 5, "Snapshot frequency in steps.")
flags.DEFINE_integer("save_every_n_steps", None, "Checkpoint frequency in steps.")

FLAGS = flags.FLAGS


def get_trainer_config(
    trainer_config_fn: Optional[TrainerConfigFn] = None,
    *,
    flag_values: flags.FlagValues = FLAGS,
) -> SpmdTrainer.Config:
    if trainer_config_fn is None:
        # Attempt a direct import. This is a common case for launching from pip package.
        try:
            trainer_config_fn = get_named_trainer_config(
                flag_values.config,
                config_module=flag_values.config_module,
            )
        except (ImportError, AttributeError, KeyError):
            logging.info(
                "Did not find config '%s' or module '%s' -- will continue searching.",
                flag_values.config,
                flag_values.config_module,
            )
            # Fallback to original strategy of importing from axlearn.experiments below.
            trainer_config_fn = None

    if trainer_config_fn is None:
        trainer_config_fn = get_named_trainer_config(
            flag_values.config,
            config_module=f"axlearn.experiments.{flag_values.config_module}",
        )
    trainer_config: SpmdTrainer.Config = trainer_config_fn()
    trainer_config.dir = trainer_config.dir or flag_values.trainer_dir
    if flag_values.mesh_selector is not None:
        select_mesh_config(trainer_config, mesh_selector=flag_values.mesh_selector)
    trainer_config.mesh_axis_names = trainer_config.mesh_axis_names or ("data", "model")
    trainer_config.mesh_shape = trainer_config.mesh_shape or (len(jax.devices()), 1)
    if isinstance(trainer_config.mesh_shape, MeshShape):
        trainer_config.mesh_shape = infer_mesh_shape(trainer_config.mesh_shape)
    trainer_config.start_trace_steps = [int(el) for el in flag_values.trace_at_steps]
    if flag_values["n_steps_for_each_trace"].present:
        trainer_config.n_steps_for_each_trace = int(flag_values.n_steps_for_each_trace)
    if flag_values["tpu_trace_mode"].present:
        trainer_config.tpu_trace_mode = flag_values.tpu_trace_mode
    if flag_values["host_tracer_level"].present:
        trainer_config.host_tracer_level = int(flag_values.host_tracer_level)
    if flag_values["device_tracer_level"].present:
        trainer_config.device_tracer_level = int(flag_values.device_tracer_level)
    if flag_values["python_tracer_level"].present:
        trainer_config.python_tracer_level = int(flag_values.python_tracer_level)
    if trainer_config.watchdog_timeout_seconds is None:
        trainer_config.watchdog_timeout_seconds = flag_values.trainer_watchdog_timeout_seconds
    if trainer_config.crash_on_hang_timeout_seconds is None:
        trainer_config.crash_on_hang_timeout_seconds = (
            flag_values.trainer_crash_on_hang_timeout_seconds
        )
    if trainer_config.log_every_n_steps is None:
        trainer_config.log_every_n_steps = flag_values.trainer_log_every_n_steps
    if flag_values.elastic_enabled:
        # pylint: disable=import-outside-toplevel
        from axlearn.common import elastic_utils
        from axlearn.common.checkpointer import Checkpointer, every_n_steps_policy
        from axlearn.common.checkpointer_orbax import OrbaxCheckpointer
        from axlearn.common.elastic_input import ElasticInput, ElasticSpmdInputDispatcher
        from axlearn.common.input_dispatch import SpmdInputDispatcher
        from axlearn.common.snapshot import Snapshotter

        # pylint: enable=import-outside-toplevel

        trainer_config.set(
            elastic_enabled=True,
            elastic_min_slice_count=flag_values.elastic_min_slice_count,
            elastic_timeout_seconds=flag_values.elastic_timeout_seconds,
            elastic_max_retries=flag_values.elastic_max_retries,
        )
        # On Pathways, keep the global batch constant across elastic events with `ElasticInput`
        # (McJAX configs set it up explicitly, with `num_max_slices`).
        dispatcher = getattr(trainer_config.input, "input_dispatcher", None)
        if elastic_utils.ensure_elastic_manager_initialized(True) is not None and isinstance(
            dispatcher, SpmdInputDispatcher.Config
        ):
            trainer_config.input.input_dispatcher = ElasticSpmdInputDispatcher.default_config().set(
                global_logical_batch_size=dispatcher.global_logical_batch_size,
                partition_spec=dispatcher.partition_spec,
            )
            trainer_config.input = ElasticInput.default_config().set(input=trainer_config.input)
        if getattr(trainer_config.checkpointer, "klass", None) is Checkpointer:
            trainer_config.checkpointer = OrbaxCheckpointer.default_config().set(
                save_policy=trainer_config.checkpointer.save_policy,
                keep_last_n=trainer_config.checkpointer.keep_last_n,
                keep_period=trainer_config.checkpointer.keep_every_n_steps,
                max_concurrent_save_gb=16,
            )
        if flag_values.elastic_backup_kind == "snapshot":
            trainer_config.snapshotter = Snapshotter.default_config().set(
                save_policy=every_n_steps_policy(flag_values.elastic_snapshot_every_n_steps)
            )
    if flag_values.save_every_n_steps is not None:
        n = int(flag_values.save_every_n_steps)
        if hasattr(trainer_config.checkpointer.save_policy, "n"):
            trainer_config.checkpointer.save_policy.set(n=n, min_step=n)
        for k in ("keep_every_n_steps", "keep_period"):
            if hasattr(trainer_config.checkpointer, k):
                setattr(trainer_config.checkpointer, k, n)
    for eval_cfg in trainer_config.evalers.values():
        eval_cfg.trace_at_iters = [int(el) for el in flag_values.eval_trace_at_iters]
    if flag_values.device_monitor == "tpu":
        # pylint: disable-next=wrong-import-position,import-outside-toplevel
        from axlearn.cloud.gcp.monitoring.tpu_device_monitor import create_tpu_monitor

        trainer_config.device_monitor = create_tpu_monitor()
    elif flag_values.device_monitor == "gpu":
        # pylint: disable-next=wrong-import-position,import-outside-toplevel
        from axlearn.common.monitoring.gpu_device_monitor import create_gpu_monitor

        trainer_config.device_monitor = create_gpu_monitor()
    if hasattr(trainer_config.checkpointer, "trainer_dir"):
        # Set trainer_dir if not already set.
        if not isinstance(trainer_config.checkpointer.trainer_dir, str):
            trainer_config.checkpointer.trainer_dir = trainer_config.dir
    return trainer_config


def run_trainer(trainer_config: SpmdTrainer.Config) -> Any:
    measurement.record_event(measurement.Event.START_JOB)
    trainer_config_debug_string = trainer_config.debug_string()
    logging.info("Trainer config:\n%s", trainer_config_debug_string)
    if jax.process_index() == 0:
        trainer_config_file = os.path.join(trainer_config.dir, "trainer_config")
        with fs.open(trainer_config_file, "w") as f:
            f.write(trainer_config_debug_string)

        config_file = os.path.join(trainer_config.dir, "launch_trainer_flags")
        with fs.open(config_file, "w") as f:
            json.dump(  # pytype: disable=wrong-arg-types
                {
                    **FLAGS.flag_values_dict(),
                    "data_dir": get_data_dir(),
                },
                f,
            )

    trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(seed=FLAGS.trainer_prng_seed)
    output = trainer.run(prng_key)
    measurement.record_event(measurement.Event.END_JOB)
    return output
