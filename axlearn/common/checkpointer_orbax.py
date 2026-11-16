# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# AI-Hypercomputer/maxtext:
# Copyright 2023–2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License").

"""Checkpointing utilities using orbax.

See also checkpointer.py for other checkpointing utilities and checkpointer_test.py for tests.
"""

import asyncio
import contextlib
import copy
import dataclasses
import functools
import os
import threading
from concurrent import futures
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
from absl import logging
from etils import epath
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.serialization.type_handlers import ArrayHandler
from orbax.checkpoint.checkpoint_manager import CheckpointInfo, _ShouldSaveFnPolicy
from tensorflow.python.checkpoint import async_checkpoint_helper

from axlearn.common import utils
from axlearn.common.checkpointer import (
    STEP_NUM_DIGITS,
    STEP_PREFIX,
    BaseCheckpointer,
    CheckpointValidationType,
    PythonSavable,
    check_state_structure,
    maybe_restore_python_savables,
    maybe_save_python_savables,
)
from axlearn.common.config import InstantiableConfig, config_class, maybe_instantiate
from axlearn.common.elastic_input import ElasticDatasetIterator
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, TensorSpec

import pathwaysutils
from orbax.checkpoint import pathways as orbax_pathways
from pathwaysutils import jax as pw_jax

try:
    # The import also registers the checkpoint handlers.
    import grain.python as grain

    _GrainIterator = Union[grain.DatasetIterator, grain.PyGrainDatasetIterator]
    _GRAIN_INSTALLED = True
except ImportError:
    logging.warning("grain is not installed; checkpointing grain iterators will not work.")
    _GRAIN_INSTALLED = False


class _TfIteratorHandler(ocp.type_handlers.TypeHandler):
    """Serializes tf.data.Iterator.

    Reference:
    https://orbax.readthedocs.io/en/latest/custom_handlers.html#custom-serialization-deserialization

    Note that since handlers are global instances (potentially shared by multiple checkpointer
    instances), we construct and cleanup the executor per-serialize/deserialize call.
    """

    def __init__(self):
        super().__init__()
        self._executor = futures.ThreadPoolExecutor()
        self._tf_ckpt_cache: dict[tf.data.Iterator, tf.train.Checkpoint] = {}

    # Must be a subclass of RestoreArgs for `PyTreeRestore` to recognize it.
    @dataclasses.dataclass
    class RestoreArgs(ocp.type_handlers.RestoreArgs):
        item: Optional[tf.data.Iterator] = None

    def typestr(self) -> str:
        return "TfIterator"

    def _get_or_create_tf_ckpt(self, value: tf.data.Iterator) -> tf.train.Checkpoint:
        # This is to avoid recreating `tf.train.Checkpoint` instances on each
        # checkpointing. When `enable_async` is `True` the resources will only
        # be released at the end of program.
        if value not in self._tf_ckpt_cache:
            self._tf_ckpt_cache[value] = tf.train.Checkpoint(value)
        return self._tf_ckpt_cache[value]

    def _sync_tf_ckpt_and_check_error(self, ckpt: tf.train.Checkpoint):
        # When `enable_async=True`, `ckpt.sync` will always return silently even
        # if it failed to save the checkpoint correctly. What makes it worse is
        # that Orbax Checkpoint Manager relies on the successful execution to
        # write the `commit_success.txt` file. Here we check and rethrow the
        # error if there's any.
        ckpt.sync()
        # pylint: disable=protected-access
        # pytype: disable=attribute-error
        assert isinstance(ckpt._async_checkpointer(), async_checkpoint_helper.AsyncCheckpointHelper)
        ckpt._async_checkpointer()._check_async_thread_error()
        # pytype: enable=attribute-error
        # pylint: enable=protected-access

    def _tasks(self, items, infos, *, save: bool) -> List[Callable[[], Any]]:
        tasks = []
        for value, info in zip(items, infos, strict=False):
            entries = [(value, jax.process_index())]
            if isinstance(value, ElasticDatasetIterator):
                entries = [(value.primary_iterator, jax.process_index())]
                if value.elastic_iterator is not None:
                    pids = value.elastic_process_ids if save else value.elastic_process_ids[:1]
                    if not save or value.is_primary_for_checkpoint:
                        entries.extend((value.elastic_iterator, pid) for pid in pids)
            base = os.path.dirname(info.parent_dir)
            for iterator, pid in entries:
                if isinstance(iterator, tf.data.Iterator):
                    ckpt = self._get_or_create_tf_ckpt(iterator)
                    (ckpt.write if save else ckpt.read)(
                        os.path.join(base, "tfds", f"tf_{pid}", info.name),
                        tf.train.CheckpointOptions(enable_async=True),
                    )
                    tasks.append(functools.partial(self._sync_tf_ckpt_and_check_error, ckpt))
                elif isinstance(iterator, PythonSavable) or (
                    _GRAIN_INSTALLED and isinstance(iterator, _GrainIterator)
                ):
                    io_fn = maybe_save_python_savables if save else maybe_restore_python_savables
                    path = os.path.join(base, "python", f"python_{pid}")
                    tasks.append(functools.partial(io_fn, {info.name: iterator}, dir=path))
                else:
                    raise NotImplementedError(f"Unsupported dataset iterator: {type(iterator)}")
        return tasks

    async def serialize(
        self,
        values: Sequence[tf.data.Iterator],
        infos: Sequence[ocp.type_handlers.ParamInfo],
        args: Optional[Sequence[ocp.args.PyTreeSave]],
    ) -> List[futures.Future]:
        """Serializes `values` into corresponding `info.path`s."""
        del args  # Unused.
        return [self._executor.submit(task) for task in self._tasks(values, infos, save=True)]

    async def deserialize(
        self,
        infos: Sequence[ocp.type_handlers.ParamInfo],
        args: Optional[Sequence[RestoreArgs]] = None,
    ) -> Sequence[tf.data.Iterator]:
        if args is None:
            raise ValueError(f"{self.RestoreArgs.__name__} should be supplied as args.")
        loop = asyncio.get_event_loop()
        tasks = self._tasks([arg.item for arg in args], infos, save=False)
        await asyncio.gather(*(loop.run_in_executor(self._executor, task) for task in tasks))
        return [arg.item for arg in args]

    async def metadata(
        self, infos: Sequence[ocp.type_handlers.ParamInfo]
    ) -> Sequence[ocp.metadata.Metadata]:
        return [ocp.metadata.Metadata(name=info.name, directory=info.path) for info in infos]


class _ElasticDatasetIteratorHandler(_TfIteratorHandler):
    def typestr(self) -> str:
        return "ElasticDatasetIterator"


ocp.type_handlers.register_type_handler(tf.data.Iterator, _TfIteratorHandler(), override=True)
ocp.type_handlers.register_type_handler(
    ElasticDatasetIterator, _ElasticDatasetIteratorHandler(), override=True
)
ocp.type_handlers.register_type_handler(
    jax.Array,
    ArrayHandler(
        array_metadata_store=array_metadata_store_lib.Store(),
        use_replica_parallel=False,
        enable_write_sharding_file=False,
    ),
    override=True,
)


if _GRAIN_INSTALLED:
    # TODO(markblee): Generalize to PythonSavableHandler.
    class _GrainDatasetIteratorHandler(ocp.type_handlers.TypeHandler):
        """Serializes grain dataset iterators."""

        def __init__(self):
            super().__init__()
            self._executor = futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="GrainDatasetIteratorHandler"
            )

        @dataclasses.dataclass
        class RestoreArgs(ocp.type_handlers.RestoreArgs):
            item: Optional[_GrainIterator] = None

        def typestr(self) -> str:
            return "DatasetIterator"

        def _ckpt_dir(self, info: ocp.type_handlers.ParamInfo) -> str:
            # Each worker writes its grain checkpoints under a different path.
            return os.path.join(
                os.path.dirname(info.parent_dir),
                "python",
                f"python_{jax.process_index()}",
            )

        async def serialize(
            self,
            values: Sequence[grain.DatasetIterator],
            infos: Sequence[ocp.type_handlers.ParamInfo],
            args: Optional[Sequence[ocp.args.PyTreeSave]],
        ) -> List[futures.Future]:
            """Serializes `values` into corresponding `info.path`s."""
            del args  # Unused.
            futs = []
            for value, info in zip(values, infos, strict=False):
                futs.append(
                    self._executor.submit(
                        maybe_save_python_savables,
                        {info.name: value},
                        dir=self._ckpt_dir(info),
                    )
                )

            return futs

        async def deserialize(
            self,
            infos: Sequence[ocp.type_handlers.ParamInfo],
            args: Optional[Sequence[RestoreArgs]] = None,
        ) -> Sequence[_GrainIterator]:
            if args is None:
                raise ValueError(f"{self.RestoreArgs.__name__} should be supplied as args.")

            await asyncio.gather(
                *(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        functools.partial(
                            maybe_restore_python_savables,
                            {info.name: arg.item},
                            dir=self._ckpt_dir(info),
                        ),
                    )
                    for arg, info in zip(args, infos, strict=False)
                )
            )

            return [arg.item for arg in args]

        async def metadata(
            self, infos: Sequence[ocp.type_handlers.ParamInfo]
        ) -> Sequence[ocp.metadata.Metadata]:
            return [ocp.metadata.Metadata(name=info.name, directory=info.path) for info in infos]

    ocp.type_handlers.register_type_handler(
        grain.DatasetIterator, _GrainDatasetIteratorHandler(), override=True
    )


def _iterator_restore_arg(x: Any) -> Optional[ocp.RestoreArgs]:
    """Returns the Orbax `RestoreArgs` for dataset iterator leaves, or None."""
    if isinstance(x, (tf.data.Iterator, ElasticDatasetIterator)):
        return _TfIteratorHandler.RestoreArgs(item=x)
    if _GRAIN_INSTALLED and isinstance(x, _GrainIterator):
        return _GrainDatasetIteratorHandler.RestoreArgs(item=x)
    return None


def _install_orbax_jax_compat_patches() -> None:
    """Installs a JAX < 0.10 fallback for `pathwaysutils.jax.concatenate_by_mesh_axis`."""
    if jax.__version_info__ >= (0, 10):
        return

    def concatenate_by_mesh_axis(arrays, sharded_dim_idxs, out_shardings, **_):
        outputs = []
        for replicas, dim, sharding in zip(arrays, sharded_dim_idxs, out_shardings):
            shape = list(replicas[0].shape)
            if dim >= 0:
                shape[dim] = sum(replica.shape[dim] for replica in replicas)
            shards = [shard.data for replica in replicas for shard in replica.addressable_shards]
            outputs.append(jax.make_array_from_single_device_arrays(tuple(shape), sharding, shards))
        return outputs

    pw_jax.concatenate_by_mesh_axis = concatenate_by_mesh_axis


_install_orbax_jax_compat_patches()


def _handle_cancelled_sidecar(err: jax.errors.JaxRuntimeError, on_cancelled: Callable[[], Any]):
    if "SidecarSession is cancelled" not in str(err):
        raise err
    logging.warning("Cancelled interrupted async checkpoint save: %s", err)
    on_cancelled()


class _CheckpointManagerWithTrackerFile(ocp.CheckpointManager):
    # pylint: disable=line-too-long
    """
    In some extreme cases, the number of available checkpoints may be quite
    large and the time spent on listing checkpoints can take > 10 minutes. This
    implementation extends the original CheckpointManager from Orbax to support
    reading only the latest checkpoint from a tracker file. So the time spent on
    initialization is constant and small. Inspired by
    [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/98d8c56dbdc9cc91b8a473debcf400958bba4524/megatron/training/checkpointing.py#L259).

    TODO(jtian22): Avoid overriding private methods or up merge it as a configurable option.
    """

    # pylint: enable=line-too-long

    def __init__(self, *args, tracker_filename="latest_checkpointed_step.txt", **kwargs):
        self._tracker_filename = tracker_filename
        super().__init__(*args, **kwargs)

    @property
    def tracker_file_path(self) -> epath.Path:
        return self.directory / self._tracker_filename

    def _load_checkpoint_infos(self, skip_metadata_read=False) -> List[CheckpointInfo]:
        """The original version looks up in the root directory and return all
        the successfully committed steps. Here we only return the latest one
        recorded in the track file. If that one is invalid or any exception is
        caught, we fallback to the original implementation.

        The checkpoint garbage collection thread will still work during current
        restart attempt, however steps prior to the one read from the tracker
        file at the start of current restart attempt will be skipped.

        TODO(jtian22): Improve GC of unnecessary checkpoints.
        """
        try:
            latest_step = int(self.tracker_file_path.read_text())
            latest_step_directory = self._get_write_step_directory(latest_step, self.directory)

            if ocp.step.is_path_finalized(latest_step_directory):
                mtime = self.tracker_file_path.stat().mtime
                latest_ckpt_info = CheckpointInfo(latest_step, datetime.fromtimestamp(mtime), None)
                return [latest_ckpt_info]
            else:
                raise ValueError(
                    (
                        f"The checkpoint {latest_step_directory} is corrupted!"
                        "The commit success file is missing!"
                    )
                )
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            logging.warning(
                (
                    "Failed to read latest checkpoint from tracker file %s!"
                    "Fallback to reading subfolders under %s"
                    "Error message: %s"
                ),
                self.tracker_file_path,
                self.directory,
                e,
            )
            return super()._load_checkpoint_infos(skip_metadata_read)

    def _finalize_checkpoint(self, step: int):
        super()._finalize_checkpoint(step)
        # In the worst (and rare) case, the checkpoint at `step` was just
        # successfully saved but the job then restarted and the tracker file was
        # not updated. Then we lost the progress up to the second latest
        # checkpoint.
        save_directory = self._get_write_step_directory(step, self.directory)
        if ocp.utils.is_primary_host(self._multiprocessing_options.primary_host):
            if ocp.step.is_path_finalized(save_directory):
                self.tracker_file_path.write_text(str(step))


class OrbaxCheckpointer(BaseCheckpointer):
    """A checkpointer that uses orbax CheckpointManager.

    NOTE: While this class uses index files to do additional validation on checkpoint state, the
    index file is not used for committing checkpoints (i.e., it is not reliable to check for the
    presence of 'index' file). In Orbax, checkpoints are committed differently on different
    filesystems. If the filesystem supports atomic renames, un-committed checkpoints are represented
    by a "tmp" prefix. OTOH, if the filesystem does not support atomic renames (such as GCS as of
    writing), committed checkpoints are represented by a commit file with the name
    "commit_success.txt". Thus, an incomplete checkpoint can still contain an "index" file (note
    that valid checkpoints should always contain an "index" file). In general, users should instead
    use `checkpoint_paths` to identify committed checkpoints.
    """

    @config_class
    class Config(BaseCheckpointer.Config):
        """Configures OrbaxCheckpointer."""

        # Keep this many past ckpts.
        keep_last_n: int = 1
        # If > 0, permanently retain checkpoints at step intervals divisible by this value.
        # This is the Orbax equivalent of `keep_every_n_steps` in the default AXLearn checkpointer.
        keep_period: Optional[int] = None
        # Checkpoint validation during restore.
        validation_type: CheckpointValidationType = CheckpointValidationType.EXACT
        # Will be passed to ocp.options.AsyncOptions(timeout_secs).
        # It is the timeout for async checkpointing operations in Orbax.
        async_timeout_secs: int = 600
        max_concurrent_save_gb: Optional[int] = None
        max_concurrent_restore_gb: Optional[int] = None
        save_device_host_concurrent_gb: Optional[int] = None
        # An AXLearn training job may use multiple TPU slices. When restoring, by default, all
        # slices read the checkpoint from GCS. If True, only the first replica loads from GCS and
        # broadcasts to other slices via the training cluster network.
        enable_single_replica_ckpt_restoring: bool = False
        # Defaults to the `data` dimension
        replica_axis_index: int = 1
        # The step to save may already exist in a incomplete state. This option
        # controls whether to skip that saving or not. The benefit of skipping
        # is to save the time on deleting an incomplete checkpoint folder.
        skip_uncommitted_checkpoint: bool = True
        # Read only the latest checkpoint recorded in the tracker file if set to `True`
        read_latest_checkpoint_from_tracker_file: bool = False
        # Optional Orbax Pathways Snapshotter config for host-RAM snapshots.
        snapshotter: Optional[InstantiableConfig[Any]] = None
        snapshot_every_n_steps: int = 1

    @classmethod
    def checkpoint_paths(cls, base_dir: str) -> List[str]:
        """See `BaseCheckpointer.checkpointer_paths`."""
        return [str(path) for path in ocp.utils.checkpoint_steps_paths(base_dir)]

    @classmethod
    def checkpoint_steps(cls, base_dir) -> list[int]:
        """See `BaseCheckpointer.checkpointer_steps`."""
        return ocp.utils.checkpoint_steps(base_dir)

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)

        cfg: OrbaxCheckpointer.Config = self.config
        save_policy = cfg.save_policy.instantiate()
        self._elastic_drain: Optional[threading.Thread] = None
        self._pending_drains: list[threading.Thread] = []

        if cfg.enable_single_replica_ckpt_restoring:
            array_handler = ocp.type_handlers.SingleReplicaArrayHandler(
                replica_axis_index=cfg.replica_axis_index,
                primary_replica_id=0,
                array_metadata_store=array_metadata_store_lib.Store(),
                use_replica_parallel=False,
                enable_write_sharding_file=False,
            )
            ocp.type_handlers.register_type_handler(jax.Array, array_handler, override=True)

        # self._eval_summaries will be set in save() and used by save_fn_with_summaries() to decide
        # whether to save at the step.
        #
        # We need this because Orbax only provides (current_step, last_saved_step) as args to the
        # save policy, so we use `self._eval_summaries` to capture evaler summaries.
        #
        # While we can check the eval summaries in `save()`, some Orbax features like
        # save-on-preemption requires the user to call `ocp.CheckpointManager.save()` at every step,
        # even if the verdict from `cfg.save_policy` is negative.
        self._eval_summaries = None

        self._name_format = ocp.step.standard_name_format(
            step_prefix=STEP_PREFIX,
            step_format_fixed_length=STEP_NUM_DIGITS,
        )

        # pylint: disable-next=unused-argument
        def save_fn_with_summaries(step: int, last_saved_step: Optional[int]) -> bool:
            is_save = save_policy(step=step, evaler_summaries=self._eval_summaries)

            if (
                is_save
                and cfg.skip_uncommitted_checkpoint
                and ocp.path.step.build_step_path(cfg.dir, self._name_format, step).exists()
            ):
                logging.warning(
                    (
                        "Step %s exists and will be skipped since"
                        "`skip_uncommitted_checkpoint` is configured to be `True`"
                    ),
                    step,
                )
                return False

            return is_save

        self._save_fn_with_summaries = save_fn_with_summaries
        self._iter_handler = ocp.PyTreeCheckpointHandler(use_ocdbt=False)
        self.snapshotter = maybe_instantiate(cfg.snapshotter)
        self._create_manager()

    @staticmethod
    def _merge_iterators(state: Nested[Any], target: Nested[Any]) -> Nested[Any]:
        return jax.tree.map(
            lambda kept, live: live if _iterator_restore_arg(live) else kept, state, target
        )

    def _snapshot_iterators(self, state: Nested[Any]) -> Nested[Any]:
        """Saves dataset iterators via `PyTreeCheckpointHandler` into a `/dev/shm` temp dir."""
        iters = {k: v for k, v in utils.flatten_items(state) if _iterator_restore_arg(v)}
        if not iters:
            return state
        tmp = TemporaryDirectory(dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        step_dir = epath.Path(tmp.name) / "s"
        step_dir.mkdir()
        self._iter_handler.save(step_dir, args=ocp.args.PyTreeSave(item=iters))
        return jax.tree.map(lambda x: tmp if _iterator_restore_arg(x) else x, state)

    def _restore_snapshot_iterators(self, snap: Nested[Any], target: Nested[Any]) -> Nested[Any]:
        """Restores dataset iterators in `target` from a snapshot's `/dev/shm` temp dir."""
        iters = {k: v for k, v in utils.flatten_items(target) if _iterator_restore_arg(v)}
        tmp = next((x for x in jax.tree.leaves(snap) if isinstance(x, TemporaryDirectory)), None)
        if iters and tmp is not None:
            self._iter_handler.restore(
                epath.Path(tmp.name) / "s",
                args=ocp.args.PyTreeRestore(
                    item=iters, restore_args=jax.tree.map(_iterator_restore_arg, iters)
                ),
            )
        return self._merge_iterators(snap, target)

    def _create_manager(self):
        cfg: OrbaxCheckpointer.Config = self.config
        is_proxy = bool(pathwaysutils.is_pathways_backend_used())
        if is_proxy:
            kwargs = dict(
                use_single_replica_array_handler=cfg.enable_single_replica_ckpt_restoring,
                checkpointing_impl=orbax_pathways.CheckpointingImpl.COLOCATED_PYTHON,
                primary_host=None,
                array_metadata_store=None,
                enable_write_sharding_file=False,
            )
            if cfg.enable_single_replica_ckpt_restoring:
                kwargs.update(replica_axis_index=cfg.replica_axis_index, primary_replica_id=0)
            orbax_pathways.register_type_handlers(**kwargs)
        CheckpointManager = (  # pylint: disable=invalid-name
            _CheckpointManagerWithTrackerFile
            if cfg.read_latest_checkpoint_from_tracker_file
            else ocp.CheckpointManager
        )
        host_gb = None if is_proxy else cfg.save_device_host_concurrent_gb
        self._manager = CheckpointManager(
            directory=cfg.dir,
            options=ocp.CheckpointManagerOptions(
                create=True,
                max_to_keep=cfg.keep_last_n,
                keep_period=cfg.keep_period,
                enable_async_checkpointing=True,
                step_name_format=self._name_format,
                should_save_fn=self._save_fn_with_summaries,
                enable_background_delete=True,
                async_options=ocp.options.AsyncOptions(
                    timeout_secs=cfg.async_timeout_secs,
                    create_directories_asynchronously=False,
                ),
                # Explicitly wrapped in `_ShouldSaveFnPolicy`, otherwise
                # `PreemptionCheckpointingPolicy` is auto injected
                save_decision_policy=_ShouldSaveFnPolicy(self._save_fn_with_summaries),
                lightweight_initialize=True,
                cleanup_tmp_directories=not hasattr(self, "_manager"),
            ),
            item_handlers={
                # NOTE: we make a relatively weak assumption that index files are JSON serialized
                # for simplicity. The test cases ensure that this is compatible with
                # `read_index_file`.
                "index": ocp.JsonCheckpointHandler(filename="index"),
                # Note that this defaults to use_ocdb=True. Note also that custom `TypeHandler`s are
                # ignored by `StandardCheckpointHandler`, so we use `PyTreeCheckpointHandler`.
                "state": ocp.PyTreeCheckpointHandler(
                    save_concurrent_gb=cfg.max_concurrent_save_gb,
                    restore_concurrent_gb=cfg.max_concurrent_restore_gb,
                    save_device_host_concurrent_gb=host_gb,
                    is_prioritized_key_fn=(lambda _: False) if self.snapshotter else None,
                ),
            },
        )

    @staticmethod
    def _cancel_manager(manager: ocp.CheckpointManager):
        with contextlib.suppress(Exception):
            manager.cancel()
        with contextlib.suppress(Exception):
            manager.close()

    def on_elastic_event(self):
        """Cancels pending snapshots and starts draining async saves in the background."""
        if self.snapshotter is not None:
            self.snapshotter.cancel_pending(timeout=0.0)
        if self._elastic_drain is None:
            self._elastic_drain = threading.Thread(
                target=self._cancel_manager, args=(self._manager,), daemon=True
            )
            self._elastic_drain.start()

    def _blocking_snapshot(self, step: int, state: Nested[Any]) -> None:
        """Takes a blocking host-RAM snapshot if `step` is not already snapshotted."""
        self.snapshotter.join()
        if getattr(self.snapshotter.latest, "step", None) != step:
            self.snapshotter.save(step, self._snapshot_iterators(state), wait=True)

    def _drain_pending(self):
        for thread in self._pending_drains:
            thread.join(timeout=self.config.async_timeout_secs)
        self._pending_drains = [t for t in self._pending_drains if t.is_alive()]

    def reset_after_elastic_event(self, *, devices_changed: bool = False):
        """Re-creates the underlying Orbax CheckpointManager after an elastic event."""
        if self.snapshotter is not None:
            self.snapshotter.cancel_pending(timeout=0.0 if devices_changed else 5.0)
        if self._elastic_drain is None:
            self.on_elastic_event()
        if not devices_changed:
            self._elastic_drain.join(timeout=self.config.async_timeout_secs)
        if self._elastic_drain.is_alive():
            self._pending_drains.append(self._elastic_drain)
        self._elastic_drain = None
        self._eval_summaries = None
        self._create_manager()

    def flush_for_scale_up(self, *, step: int, state: Nested[Tensor]) -> bool:
        """Ensures `state` at `step` is snapshotted before voluntary scale-up."""
        if self._manager.is_saving_in_progress():
            return False
        if self.snapshotter is not None:
            self._blocking_snapshot(step, state)
        return True

    def _get_spec(self, *, step: int, state: Nested[Any]) -> Nested[Any]:
        spec = {"index": [("step", step)]}
        for path, value in utils.flatten_items(state):
            if isinstance(value, (Tensor, TensorSpec, jax.ShapeDtypeStruct)):
                dtype = getattr(value.dtype, "dtype", value.dtype)
                spec["index"].append(
                    (path, {"dtype": str(dtype), "shape": str(tuple(value.shape))})
                )
            elif isinstance(value, (tf.data.Iterator, ElasticDatasetIterator, PythonSavable)):
                spec["index"].append((path, str(type(value))))
            else:
                spec["index"].append((path, value))
        return spec

    # pylint: disable-next=redefined-builtin
    def ckpt_dir(self, step: int, dir: Optional[str] = None) -> str:
        """Obtains the checkpoint dir for the given step."""
        if dir is None:
            dir = self._manager.directory
        return str(ocp.step.build_step_path(dir, self._name_format, step))

    def save(
        self,
        *,
        step: int,
        state: Nested[Tensor],
        evaler_summaries: Optional[Dict[str, Any]] = None,
    ):
        """See `BaseCheckpointer.save` for details.

        Checkpoint saving is handled by `orbax` checkpoint manager.
        """
        spec = self._get_spec(step=step, state=state)
        assert self._eval_summaries is None, self._eval_summaries
        self._eval_summaries = copy.deepcopy(evaler_summaries or {})

        try:
            if self.snapshotter is not None:
                if not self._manager.should_save(step):
                    latest = self.snapshotter.latest
                    every = self.config.snapshot_every_n_steps
                    if latest is None:
                        self._blocking_snapshot(step, state)
                    elif every > 0 and step > 0 and step % every == 0 and latest.step != step:
                        self.snapshotter.save(step, self._snapshot_iterators(state))
                    return

                def _wait_prior():
                    try:
                        self._manager.wait_until_finished()
                    except jax.errors.JaxRuntimeError as e:
                        _handle_cancelled_sidecar(e, self._manager.reload)

                state = self.snapshotter.save(
                    step,
                    self._snapshot_iterators(state),
                    checkpointer_wait_fn=_wait_prior,
                    checkpointer_save_fn=functools.partial(self._merge_iterators, target=state),
                )
            self._drain_pending()
            # Note that save() waits for prior serialization to finish.
            self._manager.save(
                step=step,
                # The input iterator is saved as part of `save_tf_savables`.
                args=ocp.args.Composite(
                    index=ocp.args.JsonSave(spec["index"]),
                    # TODO(markblee): Investigate save_args for chunk_byte_size and
                    # ocdbt_target_data_file_size:
                    # https://orbax.readthedocs.io/en/latest/optimized_checkpointing.html#custom-chunk-sizes
                    # https://orbax.readthedocs.io/en/latest/optimized_checkpointing.html#customizing-data-file-size
                    state=ocp.args.PyTreeSave(item=state),
                ),
            )
            # Exit early after pre-emption, equivalent to sys.exit():
            # https://orbax.readthedocs.io/en/latest/preemption_checkpointing.html
            if self._manager.reached_preemption(step):
                self._manager.wait_until_finished()
                raise SystemExit(f"Exiting after saving checkpoint at {step=} due to pre-emption.")
        except jax.errors.JaxRuntimeError as e:
            _handle_cancelled_sidecar(e, self.cancel)
        finally:
            self._eval_summaries = None

    def restore(
        self,
        *,
        step: Optional[int] = None,
        state: Union[Nested[Tensor], Nested[TensorSpec]],
    ) -> Tuple[Optional[int], Nested[Tensor]]:
        """See `BaseCheckpointer.restore` for details."""

        cfg: OrbaxCheckpointer.Config = self.config

        if step is None and self.snapshotter is not None:
            latest_ckpt = self._manager.latest_step()
            snap_step, snap_state = self.snapshotter.try_load(state)
            if snap_step is not None and (latest_ckpt is None or snap_step >= latest_ckpt):
                logging.info("[ELASTIC] Restored trainer_state from snapshot at step=%s", snap_step)
                return snap_step, self._restore_snapshot_iterators(snap_state, state)

        def _restore_args(x: Any) -> ocp.RestoreArgs:
            if isinstance(x, (Tensor, TensorSpec, jax.ShapeDtypeStruct)):
                arg = ocp.checkpoint_utils.construct_restore_args(
                    jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)
                )
                if cfg.enable_single_replica_ckpt_restoring and isinstance(
                    arg, ocp.type_handlers.ArrayRestoreArgs
                ):
                    mesh = x.sharding.mesh
                    arg = ocp.type_handlers.SingleReplicaArrayRestoreArgs(
                        restore_type=arg.restore_type,
                        dtype=arg.dtype,
                        mesh=arg.mesh,
                        mesh_axes=arg.mesh_axes,
                        sharding=arg.sharding,
                        global_shape=arg.global_shape,
                        shape=arg.shape,
                        strict=arg.strict,
                        single_replica_sharding=jax.sharding.NamedSharding(
                            jax.sharding.Mesh(
                                _replica_devices(mesh.devices, cfg.replica_axis_index),
                                mesh.axis_names,
                            ),
                            x.sharding.spec,
                        ),
                    )
                return arg
            return _iterator_restore_arg(x)

        restore_args = jax.tree.map(_restore_args, state)

        try:
            composite_state = self._manager.restore(
                step,
                args=ocp.args.Composite(
                    index=ocp.args.JsonRestore(None),
                    state=ocp.args.PyTreeRestore(item=state, restore_args=restore_args),
                ),
            )
        except FileNotFoundError as e:
            # Orbax raises FileNotFoundError if there are no checkpoints.
            if step is not None:
                raise ValueError(f"Failed to restore at step {step}.") from e
            logging.info("Could not find any completed checkpoints under %s: %s", cfg.dir, e)
            return None, state  # Return the input state.

        restored_index = composite_state["index"]
        restored_state = composite_state["state"]

        # If we successfully restored from step=None, use the restored step.
        if step is None:
            for k, v in restored_index:
                if k == "step":
                    step = v
                    break

        # Validate ckpt structure.
        check_state_structure(
            restored_index,
            target_structure=self._get_spec(step=step, state=state)["index"],
            validation=cfg.validation_type,
        )
        if step is not None and self.snapshotter is not None:
            self._blocking_snapshot(step, restored_state)
        return step, restored_state

    def wait_until_finished(self):
        """See `BaseCheckpointer.wait_until_finished` docstring for details."""
        self._manager.wait_until_finished()

    def cancel(self):
        """Cancels pending snapshots/saves and recreates the checkpoint manager."""
        if self.snapshotter is not None:
            self.snapshotter.cancel_pending(timeout=0.0)
        with contextlib.suppress(Exception):
            self._manager.cancel()
        with contextlib.suppress(Exception):
            self._manager.close()
        self._create_manager()

    def stop(self, *, has_exception: bool = False):
        """See `BaseCheckpointer.stop` for details."""
        if self.snapshotter is not None:
            if has_exception:
                self.snapshotter.cancel_pending(timeout=0.0)
            else:
                self.snapshotter.join()
        if has_exception:
            with contextlib.suppress(Exception):
                self._manager.cancel()
        self._drain_pending()
        try:
            self._manager.close()
        except Exception:  # pylint: disable=broad-exception-caught
            if not has_exception:
                raise
            logging.warning("Checkpointer close failed after error.", exc_info=True)


# Below are adapted from:
# https://github.com/AI-Hypercomputer/maxtext/blob/3d9378d77759a7756e20ae2940ce71dcaa17ef13/src/MaxText/checkpointing.py#L333-L356


def _find_idx(array: np.ndarray, replica_axis_idx: int) -> int:
    """Returns the index along given dimension that the current host belongs to."""
    idx = None
    for idx, val in np.ndenumerate(array):
        if val.process_index == jax.process_index():
            break
    return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int) -> np.ndarray:
    """Returns the devices from the replica that current host belongs to.

    Replicas are assumed to be restricted to the first axis.

    Args:
      device_array: devices of the mesh that can be obtained by mesh.devices()
      replica_axis_idx: axis dimension along which replica is taken

    Returns:
      devices inside the replica that current host is in
    """
    idx = _find_idx(device_array, replica_axis_idx)
    replica_result = np.take(device_array, idx, axis=replica_axis_idx)
    return np.expand_dims(replica_result, axis=replica_axis_idx)
