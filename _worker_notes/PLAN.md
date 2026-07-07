# Plan - Synthesis of Elastic Recovery Refactor

1. **Modify google3 `pathwaysutils` files**:
   - `third_party/pathways/jax/ifrt/BUILD`: Add visibility for `//third_party/py/pathwaysutils/...` to `users` package group.
   - `third_party/py/pathwaysutils/elastic/BUILD`: Add `//third_party/pathways/jax/ifrt:client` dependency.
   - `third_party/py/pathwaysutils/elastic/elastic.py`: Implement `PathwaysSliceHealthChecker` using per-slice `devices_placement_active` calls (Worker 0's strategy) and copybara import tags.
   - `third_party/py/pathwaysutils/elastic/manager.py`: Remove `jax.clear_caches()` calls.
   - `third_party/py/pathwaysutils/test/google_internal/elastic/BUILD`: Ensure `//third_party/py/pathwaysutils:_initialize` dependency is added to `elastic_test`.
   - `third_party/py/pathwaysutils/test/google_internal/elastic/elastic_test.py`: Mock `devices_placement_active` inside tests.
   - `third_party/py/pathwaysutils/test/google_internal/elastic/manager_test.py`: Update assertions to check that `clear_caches` is not called.

2. **Verify `axlearn` open-source changes**:
   - Verify that changes on branch `elastic-preemption-recovery-fixes-cleaned` compile and that `pytest common/trainer_test.py` passes.

3. **Verify google3 `pathwaysutils` changes**:
   - Run `blaze test //third_party/py/pathwaysutils/test/google_internal/elastic/...`.

4. **Write Handoff Report**:
   - Document final workspace path and verification results in `_worker_notes/README.md`.
