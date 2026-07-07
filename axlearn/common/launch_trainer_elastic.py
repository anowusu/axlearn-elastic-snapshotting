# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer with elastic monkeypatch."""

import asyncio
import concurrent.futures
import datetime
import logging
import os
import threading
import time
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging as absl_logging
from jax.experimental.array_serialization import serialization as jax_serde

from pathwaysutils.persistence import helper

# Monkeypatch jax.distributed.initialize to bypass it in proxy mode.
absl_logging.info("ELASTIC_LAUNCH: Mocking jax.distributed.initialize")
jax.distributed.initialize = lambda *args, **kwargs: absl_logging.info("ELASTIC_LAUNCH: jax.distributed.initialize bypassed")


class PathwaysGlobalAsyncCheckpointManager(jax_serde.GlobalAsyncCheckpointManagerBase):
    def __init__(self, timeout_secs: Union[int, float] = 300, *args, **kwargs):
        self._timeout = datetime.timedelta(seconds=timeout_secs) if timeout_secs else None
        self._cond = threading.Condition()
        self._pending_disk_writes = False
        self._exception = None
        self._write_future = None

    def check_for_errors(self):
        with self._cond:
            if self._exception is not None:
                raise self._exception

    def wait_until_finished(self):
        absl_logging.info("PathwaysGACM: wait_until_finished called")
        with self._cond:
            while self._pending_disk_writes:
                self._cond.wait()
        self.check_for_errors()
        absl_logging.info("PathwaysGACM: wait_until_finished completed")

    def _convert_specs(self, specs: Sequence[dict]) -> tuple[str, list[str]]:
        # Reconstruct full GCS paths
        full_paths = []
        for spec in specs:
            kv = spec.get("kvstore", {})
            driver = kv.get("driver")
            if driver == "gcs":
                bucket = kv.get("bucket")
                path = kv.get("path")
                full_paths.append(f"gs://{bucket}/{path}")
            elif driver == "gfile":
                full_paths.append(kv.get("path"))
            else:
                raise ValueError(f"Unsupported kvstore driver: {driver} in spec {spec}")

        # Find common path
        commonpath = os.path.commonpath([os.path.dirname(p) for p in full_paths])
        commonpath_len = len(commonpath)
        names = []
        for p in full_paths:
            names.append(p[commonpath_len:].lstrip("/"))
        
        return commonpath, names

    def serialize(
        self,
        arrays: Sequence[jax.Array],
        tensorstore_specs: Sequence[dict],
        *,
        on_commit_callback: Callable[[], None],
        additional_futures: Optional[list[concurrent.futures.Future]] = None,
    ) -> None:
        absl_logging.info("PathwaysGACM: serialize called with %d arrays", len(arrays))
        self.wait_until_finished()

        location, names = self._convert_specs(tensorstore_specs)
        absl_logging.info("PathwaysGACM: Writing to location: %s, names: %s", location, names)

        # Call write_arrays on all processes (TPU hosts)
        write_future = helper.write_arrays(location, names, arrays, self._timeout)

        self._pending_disk_writes = True
        
        def wait_thread():
            try:
                write_future.result()
                absl_logging.info("PathwaysGACM: write_arrays future finished successfully")
                
                # Wait for additional futures (e.g. TF state save)
                if additional_futures:
                    absl_logging.info("PathwaysGACM: waiting for %d additional futures", len(additional_futures))
                    for f in additional_futures:
                        f.result()
                
                # Execute commit callback ONLY on process 0 to avoid write conflicts
                if jax.process_index() == 0:
                    absl_logging.info("PathwaysGACM: process 0 executing commit callback")
                    on_commit_callback()
                    absl_logging.info("PathwaysGACM: commit callback executed successfully")
            except Exception as e:
                absl_logging.exception("PathwaysGACM: Exception in write thread")
                with self._cond:
                    self._exception = e
            finally:
                with self._cond:
                    self._pending_disk_writes = False
                    self._cond.notify_all()

        t = threading.Thread(target=wait_thread, name="pathways_gacm_commit_thread")
        t.start()

    def deserialize(
        self,
        shardings: Sequence[jax.sharding.Sharding],
        tensorstore_specs: Sequence[dict],
        global_shapes: Optional[Sequence[tuple]] = None,
        dtypes: Optional[Sequence[np.dtype]] = None,
        concurrent_gb: int = 32,
    ) -> Sequence[jax.Array]:
        absl_logging.info("PathwaysGACM: deserialize called")
        assert global_shapes, "global_shapes must be specified for Pathways persistence"
        assert dtypes, "dtypes must be specified for Pathways persistence"
        
        location, names = self._convert_specs(tensorstore_specs)
        absl_logging.info("PathwaysGACM: Reading from location: %s, names: %s", location, names)

        # Group by mesh to match _deserialize_on_single_mesh behavior
        mesh = shardings[0].mesh
        for s in shardings:
            assert s.mesh == mesh, "All shardings must be on the same mesh"

        np_dtypes = [np.dtype(d) for d in dtypes]

        # Call read_arrays on all processes (TPU hosts)
        arrays, read_future = helper.read_arrays(
            location,
            names,
            np_dtypes,
            global_shapes,
            shardings,
            mesh.devices,
            self._timeout,
        )
        
        absl_logging.info("PathwaysGACM: waiting for read_arrays to complete")
        read_future.result()
        absl_logging.info("PathwaysGACM: read_arrays completed successfully")
        return arrays


# Apply the monkeypatch to AXLearn array_serialization
absl_logging.info("ELASTIC_LAUNCH: Monkeypatching AXLearn checkpointer classes with PathwaysGACM")
from axlearn.common import array_serialization as axlearn_serde
axlearn_serde.GlobalAsyncCheckpointManager = PathwaysGlobalAsyncCheckpointManager
axlearn_serde.BoundedDataShardedAsyncCheckpointManager = PathwaysGlobalAsyncCheckpointManager


from axlearn.common import launch_trainer_main as m

if __name__ == "__main__":
    m.measurement.define_flags()
    app.run(m.main)
