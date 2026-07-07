# Refactoring Plan - Elastic Recovery (AxLearn)

## Goal
Modify the open-source AxLearn snapshotting and recovery logic to improve JAX integration, remove background NumPy threads, keep snapshots as JAX `pinned_host` memory arrays, use native JAX resharding (`split_by_mesh_axis` / `concatenate_by_mesh_axis`), delete socket cleanups, and use JAX device health pings during recovery.

## Steps
1. **snapshot.py**:
   - Keep snapshots as JAX `pinned_host` memory arrays via `jax.device_put` and `.with_memory_kind("pinned_host")`.
   - Remove background thread queues and NumPy conversions.
   - Use pathways utils resharding (`split_by_mesh_axis` / `concatenate_by_mesh_axis`) in `heal_pytree`.
2. **trainer.py**:
   - Delete obsolete OS network socket connection cleanups (`_cleanup_pathways_proxy_sockets` and `_wait_for_workers_ready`).
   - Update `elastic_training_loop` to use `elastic.wait_for_slices` for JAX device health pings.
3. **Verification**:
   - Run pytest on `trainer_test.py`.
