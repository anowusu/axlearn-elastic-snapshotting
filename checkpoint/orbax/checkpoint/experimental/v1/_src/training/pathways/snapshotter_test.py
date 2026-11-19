"""Tests for Snapshotter.

This module tests the Snapshot class, which manages asynchronous backups of JAX
array states to pinned host memory.
"""

import contextlib
import threading
import types
from unittest import mock

import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.training.pathways import snapshotter

from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized

SimpleNamespace = types.SimpleNamespace


def _create_sharded_pytree(devices: list[jax.Device]):
  slices = len({d.slice_index for d in devices})
  mesh = jax.sharding.Mesh(
      np.asarray(devices).reshape((slices, len(devices) // slices)),
      ('x', 'y'),
  )
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  sharded_y = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(None, 'y')
  )
  shardings = {
      'a': replicated,
      'b': replicated,
      'c': {
          'a': sharded_y,
          'e': sharded_y,
      },
      'x': replicated,
      'y': replicated,
  }
  pytree, _ = array_test_utils.create_numpy_pytree()
  pytree = jax.tree.map(
      array_test_utils.create_sharded_array, pytree, shardings
  )
  return pytree, jax.tree.map(array_test_utils.as_abstract_type, pytree)


# Meshes to save a snapshot on and to load it onto: the same mesh, fewer slices
# (scale down) and more slices (scale up).
_SAVE_AND_LOAD_MESHES = (
    dict(
        testcase_name='same_mesh',
        save_mesh='all_slices',
        load_mesh='all_slices',
    ),
    dict(
        testcase_name='scale_down',
        save_mesh='all_slices',
        load_mesh='two_slices',
    ),
    dict(
        testcase_name='scale_up',
        save_mesh='two_slices',
        load_mesh='all_slices',
    ),
)


class SnapshotterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.pytree, self.abstract_pytree = _create_sharded_pytree(jax.devices())
    self.pytree_2_slices, self.abstract_pytree_2_slices = (
        _create_sharded_pytree(
            [d for d in jax.devices() if d.slice_index in (0, 1)]
        )
    )

  @parameterized.named_parameters(
      dict(testcase_name='0', replica_axis_index=0),
      dict(testcase_name='1', replica_axis_index=1),
  )
  def test_save_restore_pytree(self, replica_axis_index):
    snapshot_manager = snapshotter.Snapshotter(
        replica_axis_index=replica_axis_index
    )

    snapshot_manager.save(0, self.pytree)
    snapshot_manager.join()
    snapshot_manager.save(1, self.pytree)
    snapshot_manager.join()

    self.assertIsNotNone(snapshot_manager._latest_snapshot)
    assert snapshot_manager._latest_snapshot is not None
    self.assertIsInstance(snapshot_manager._latest_snapshot[0], list)
    self.assertNotEmpty(snapshot_manager._latest_snapshot[0])
    self.assertEqual(
        snapshot_manager._latest_snapshot[0][0]['a'].sharding.memory_kind,
        'pinned_host',
    )
    self.assertEqual(
        self.pytree['a'].sharding.memory_kind,
        'device',
    )

    restored_pytree = snapshot_manager.load(self.abstract_pytree)
    # Check that restored array is on default memory (not pinned_host).
    # Since we restored to self.abstract_pytree, it should match self.pytree.
    test_utils.assert_tree_equal(self, self.pytree, restored_pytree)

    def assert_not_pinned_host(x):
      self.assertNotEqual(x.sharding.memory_kind, 'pinned_host')

    jax.tree.map(assert_not_pinned_host, restored_pytree)

  def test_scale_down(self):
    snapshot_manager = snapshotter.Snapshotter()

    snapshot_manager.save(0, self.pytree)
    snapshot_manager.join()

    self.assertEqual(
        snapshot_manager._latest_snapshot[0][0]['a'].sharding.memory_kind,  # pyrefly: ignore[unsupported-operation]
        'pinned_host',
    )
    restored_pytree = snapshot_manager.load(
        self.abstract_pytree_2_slices
    )
    expected_tree = jax.tree.map(
        lambda x, y: jax.device_put(x, y.sharding),
        self.pytree,
        self.abstract_pytree_2_slices,
    )
    test_utils.assert_tree_equal(self, expected_tree, restored_pytree)
    assert snapshot_manager._latest_snapshot is not None
    self.assertNotEmpty(snapshot_manager._latest_snapshot[0])
    self.assertEqual(
        snapshot_manager._latest_snapshot[0][0]['a'].sharding.memory_kind,  # pyrefly: ignore[unsupported-operation]
        'pinned_host',
    )

  def test_scale_up(self):
    snapshot_manager = snapshotter.Snapshotter(replica_axis_index=0)

    snapshot_manager.save(0, self.pytree_2_slices)
    snapshot_manager.join()

    restored_pytree = snapshot_manager.load(self.abstract_pytree)
    expected_pytree = jax.tree.map(
        lambda x, y: jax.device_put(x, y.sharding),
        self.pytree_2_slices,
        self.abstract_pytree,
    )
    test_utils.assert_tree_equal(self, expected_pytree, restored_pytree)

  def test_load_with_inactive_replicas(self):
    snapshot_manager = snapshotter.Snapshotter()
    snapshot_manager.save(0, self.pytree)
    snapshot_manager.join()

    assert snapshot_manager._latest_snapshot is not None
    original_block_until_ready = jax.block_until_ready
    call_count = 0

    def side_effect(tree):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise jax.errors.JaxRuntimeError('Replica 0 is inactive')
      return original_block_until_ready(tree)

    with mock.patch.object(jax, 'block_until_ready', side_effect=side_effect):
      restored_pytree = snapshot_manager.load(self.abstract_pytree)

    test_utils.assert_tree_equal(self, self.pytree, restored_pytree)

  def test_save_mismatched_replicas_raises_error(self):
    snapshot_manager = snapshotter.Snapshotter()
    mismatched_tree = {
        'four_slices': self.pytree['a'],
        'two_slices': self.pytree_2_slices['a'],
    }
    with self.assertRaisesRegex(
        ValueError, 'All JAX arrays must have the same number of replicas'
    ):
      snapshot_manager.save(0, mismatched_tree)

  def test_latest_property(self):
    snapshot_manager = snapshotter.Snapshotter()
    self.assertIsNone(snapshot_manager.latest)
    snapshot_manager.save(0, self.pytree)
    snapshot_manager.join()

    self.assertIsNotNone(snapshot_manager.latest)
    assert snapshot_manager.latest is not None
    self.assertEqual(snapshot_manager.latest.step, 0)

    snapshot_manager.save(1, self.pytree)
    snapshot_manager.join()

    assert snapshot_manager.latest is not None
    self.assertEqual(snapshot_manager.latest.step, 1)

    snapshot_manager.load(self.abstract_pytree)
    assert snapshot_manager.latest is not None
    self.assertEqual(snapshot_manager.latest.step, 1)

  def test_join(self):
    snapshot_manager = snapshotter.Snapshotter()
    with mock.patch.object(
        snapshot_manager._queue, 'join', wraps=snapshot_manager._queue.join
    ) as mock_join:
      snapshot_manager.save(0, self.pytree)
      snapshot_manager.join()
      mock_join.assert_called_once()
    self.assertIsNotNone(snapshot_manager._latest_snapshot)
    assert snapshot_manager.latest is not None
    self.assertEqual(snapshot_manager.latest.step, 0)

  def test_save_restore_pytree_with_scalars(self):
    snapshot_manager = snapshotter.Snapshotter()

    # Create a tree with an array and a scalar
    tree = {
        'array': self.pytree['a'],
        'scalar_int': 42,
        'scalar_float': 3.14,
        'string': 'hello',
        'object': SimpleNamespace(a=1, b=2),
        'object_with_sharding': SimpleNamespace(sharding=5),
    }
    abstract_tree = {
        'array': self.abstract_pytree['a'],
        'scalar_int': 42,
        'scalar_float': 3.14,
        'string': 'hello',
        'object': SimpleNamespace(a=1, b=2),
        'object_with_sharding': SimpleNamespace(sharding=5),
    }

    snapshot_manager.save(0, tree)
    snapshot_manager.join()

    self.assertIsNotNone(snapshot_manager._latest_snapshot)
    assert snapshot_manager._latest_snapshot is not None
    self.assertIsInstance(snapshot_manager._latest_snapshot[0], list)
    self.assertNotEmpty(snapshot_manager._latest_snapshot[0])
    for replica_tree in snapshot_manager._latest_snapshot[0]:
      self.assertEqual(replica_tree['scalar_int'], 42)
      self.assertEqual(replica_tree['scalar_float'], 3.14)
      self.assertEqual(replica_tree['string'], 'hello')
      self.assertIs(replica_tree['object'], tree['object'])
      self.assertIs(
          replica_tree['object_with_sharding'], tree['object_with_sharding']
      )

    restored_tree = snapshot_manager.load(abstract_tree)

    test_utils.assert_tree_equal(self, tree, restored_tree)

  def test_worker_survives_runtime_error_on_save(self):
    snapshot_manager = snapshotter.Snapshotter()
    original_block_until_ready = jax.block_until_ready
    call_count = 0

    def side_effect(arr):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise jax.errors.JaxRuntimeError('Simulated runtime error')
      return original_block_until_ready(arr)

    with mock.patch.object(jax, 'block_until_ready', side_effect=side_effect):
      snapshot_manager.save(0, self.pytree)
      snapshot_manager.join()
      self.assertIsNone(snapshot_manager._latest_snapshot)
      self.assertTrue(snapshot_manager._worker_thread.is_alive())

      snapshot_manager.save(1, self.pytree)
      snapshot_manager.join()
      self.assertIsNotNone(snapshot_manager._latest_snapshot)
      assert snapshot_manager.latest is not None
      self.assertEqual(snapshot_manager.latest.step, 1)
      self.assertTrue(snapshot_manager._worker_thread.is_alive())

  def test_save_restore_prng_key_array(self):
    snapshot_manager = snapshotter.Snapshotter()
    mesh = self.pytree['a'].sharding.mesh
    sharded_scalar_key = jax.device_put(
        jax.random.key(42),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
    sharded_array_key = jax.device_put(
        jax.random.split(jax.random.key(42), 4),
        self.pytree['a'].sharding,
    )
    tree = {'scalar_key': sharded_scalar_key, 'array_key': sharded_array_key}
    abstract_tree = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
        tree,
    )

    snapshot_manager.save(0, tree)
    snapshot_manager.join()

    self.assertIsNotNone(snapshot_manager._latest_snapshot)
    assert snapshot_manager._latest_snapshot is not None
    self.assertIsInstance(snapshot_manager._latest_snapshot[0], list)
    replica_0_state = snapshot_manager._latest_snapshot[0][0]

    for k in tree:
      self.assertEqual(
          replica_0_state[k].sharding.memory_kind,
          'pinned_host',
      )
      self.assertEqual(tree[k].sharding.memory_kind, 'device')

    restored_tree = snapshot_manager.load(abstract_tree)
    for k in tree:
      self.assertIsInstance(restored_tree[k], type(tree[k]))
      self.assertNotEqual(
          restored_tree[k].sharding.memory_kind, 'pinned_host'
      )
    test_utils.assert_tree_equal(self, tree, restored_tree)

  def test_load_does_not_donate_buffer(self):
    snapshot_manager = snapshotter.Snapshotter()
    snapshot_manager.save(0, self.pytree)
    snapshot_manager.join()

    restored_pytree = snapshot_manager.load(self.abstract_pytree)
    test_utils.assert_tree_equal(self, self.pytree, restored_pytree)

    # Verify that the snapshot buffer was not donated and can still be used.
    self.assertIsNotNone(snapshot_manager._latest_snapshot)
    assert snapshot_manager._latest_snapshot is not None
    for replica_tree in snapshot_manager._latest_snapshot[0]:
      replica_tree['a'][...].block_until_ready()

    # Verify that loading a second time immediately following succeeds.
    second_restored_pytree = snapshot_manager.load(self.abstract_pytree)
    test_utils.assert_tree_equal(self, self.pytree, second_restored_pytree)

  def test_load_without_reset_keeps_the_snapshot(self):
    snapshot_manager = snapshotter.Snapshotter()
    snapshot_manager.save(0, self.pytree)
    snapshot_manager.join()

    first_restored_pytree = snapshot_manager.load(
        self.abstract_pytree, reset_snapshot_state=False
    )
    # A load that fails (e.g. a slice goes down) is retried from the same
    # snapshot, so loading must not consume the snapshot's buffers.
    second_restored_pytree = snapshot_manager.load(
        self.abstract_pytree, reset_snapshot_state=False
    )

    test_utils.assert_tree_equal(self, self.pytree, first_restored_pytree)
    test_utils.assert_tree_equal(self, self.pytree, second_restored_pytree)

  @contextlib.contextmanager
  def _record_device_puts(self):
    """Records the `(array, target sharding)` of every `jax.device_put` call."""
    puts = []
    device_put = jax.device_put

    def recording_device_put(x, device=None, *args, **kwargs):
      arrays = jax.tree.leaves(x)
      targets = (
          [device] * len(arrays)
          if device is None or isinstance(device, jax.sharding.Sharding)
          else jax.tree.leaves(device)
      )
      puts.extend(zip(arrays, targets))
      return device_put(x, device, *args, **kwargs)

    with mock.patch.object(jax, 'device_put', side_effect=recording_device_put):
      yield puts

  def _save_and_get_target(self, save_mesh, load_mesh):
    """Saves a snapshot of the pytree on `save_mesh`.

    Args:
      save_mesh: Mesh of the saved pytree, `all_slices` or `two_slices`.
      load_mesh: Mesh to load onto, `all_slices` or `two_slices`.

    Returns:
      The snapshotter, the saved pytree, the abstract pytree on `load_mesh` and
      the saved pytree resharded onto `load_mesh`.
    """
    trees = {
        'all_slices': (self.pytree, self.abstract_pytree),
        'two_slices': (self.pytree_2_slices, self.abstract_pytree_2_slices),
    }
    saved_pytree, _ = trees[save_mesh]
    _, abstract_target = trees[load_mesh]
    snapshot_manager = snapshotter.Snapshotter()
    snapshot_manager.save(0, saved_pytree)
    snapshot_manager.join()
    expected_pytree = jax.tree.map(
        lambda x, y: jax.device_put(x, y.sharding),
        saved_pytree,
        abstract_target,
    )
    return snapshot_manager, saved_pytree, abstract_target, expected_pytree

  @parameterized.named_parameters(*_SAVE_AND_LOAD_MESHES)
  def test_load_concatenates_active_replicas_in_device_memory(
      self, save_mesh, load_mesh
  ):
    snapshot_manager, saved_pytree, abstract_target, expected_pytree = (
        self._save_and_get_target(save_mesh, load_mesh)
    )
    concatenate_module = snapshot_manager._concatenate_by_mesh_axis

    with mock.patch.object(
        concatenate_module,
        'concatenate_by_mesh_axis',
        wraps=concatenate_module.concatenate_by_mesh_axis,
    ) as mock_concatenate:
      restored_pytree = snapshot_manager.load(abstract_target)

    test_utils.assert_tree_equal(self, expected_pytree, restored_pytree)
    # One batched call across all array leaves, with one slice list per target
    # replica in device memory.
    num_target_replicas = abstract_target['a'].sharding.mesh.shape['x']
    self.assertLen(mock_concatenate.call_args_list, 1)
    slice_lists = mock_concatenate.call_args.args[0]
    self.assertLen(slice_lists, num_target_replicas)
    for slice_arrays in slice_lists:
      self.assertLen(slice_arrays, len(jax.tree.leaves(saved_pytree)))
      for array in slice_arrays:
        self.assertEqual(array.sharding.memory_kind, 'device')

  @parameterized.named_parameters(*_SAVE_AND_LOAD_MESHES)
  def test_load_without_reset_does_not_put_arrays_in_pinned_host(
      self, save_mesh, load_mesh
  ):
    snapshot_manager, _, abstract_target, expected_pytree = (
        self._save_and_get_target(save_mesh, load_mesh)
    )

    with self._record_device_puts() as puts:
      restored_pytree = snapshot_manager.load(
          abstract_target, reset_snapshot_state=False
      )

    test_utils.assert_tree_equal(self, expected_pytree, restored_pytree)
    # The arrays are reconstructed and resharded in device memory only; nothing
    # is moved to (or across slices in) pinned_host memory.
    self.assertNotEmpty(puts)
    for _, target in puts:
      self.assertNotEqual(getattr(target, 'memory_kind', None), 'pinned_host')

  @parameterized.named_parameters(*_SAVE_AND_LOAD_MESHES)
  def test_load_resets_snapshot_from_restored_device_state(
      self, save_mesh, load_mesh
  ):
    snapshot_manager, _, abstract_target, expected_pytree = (
        self._save_and_get_target(save_mesh, load_mesh)
    )

    with self._record_device_puts() as puts:
      restored_pytree = snapshot_manager.load(abstract_target)

    test_utils.assert_tree_equal(self, expected_pytree, restored_pytree)
    # The new snapshot is copied from the restored arrays in device memory onto
    # pinned_host memory of the same devices: no copy crosses slices.
    pinned_host_puts = [
        (x, target)
        for x, target in puts
        if getattr(target, 'memory_kind', None) == 'pinned_host'
    ]
    self.assertLen(pinned_host_puts, len(jax.tree.leaves(expected_pytree)))
    for x, target in pinned_host_puts:
      self.assertEqual(x.sharding.memory_kind, 'device')
      self.assertEqual(x.sharding.device_set, target.device_set)
    assert snapshot_manager._latest_snapshot is not None
    replica_trees, step, _ = snapshot_manager._latest_snapshot
    self.assertEqual(step, 0)
    self.assertLen(replica_trees, abstract_target['a'].sharding.mesh.shape['x'])
    for replica_tree in replica_trees:
      self.assertEqual(replica_tree['a'].sharding.memory_kind, 'pinned_host')

  def test_save_copies_arrays_already_in_pinned_host(self):
    manager = snapshotter.Snapshotter()
    pinned_sharding = self.pytree['a'].sharding.with_memory_kind('pinned_host')
    pinned_input = jax.device_put(self.pytree['a'], pinned_sharding)

    manager.save(1, {**self.pytree, 'a': pinned_input})
    manager.join()
    # The snapshot must hold a copy of `pinned_input`, not an alias of it:
    # deleting the input, as a later step that donates it does, must leave the
    # snapshot intact.
    pinned_input.delete()
    self.assertTrue(pinned_input.is_deleted())

    restored = manager.load(self.abstract_pytree, reset_snapshot_state=False)
    test_utils.assert_tree_equal(self, self.pytree, restored)

  @parameterized.named_parameters(
      ('replica_axis_0', 0, (2, 2, -1), (1, 2, -1), 'data'),
      ('replica_axis_1', 1, (1, 2, -1), (1, 1, -1), 'fsdp'),
  )
  def test_replica_axis_and_single_device_sharding(
      self, replica_axis_index, save_mesh_shape, load_mesh_shape, sharded_axis
  ):
    devices = np.array(jax.devices())
    axis_names = ('dp', 'data', 'fsdp')
    save_mesh = jax.sharding.Mesh(devices.reshape(save_mesh_shape), axis_names)
    load_mesh = jax.sharding.Mesh(
        devices[:4].reshape(load_mesh_shape), axis_names
    )
    partition_spec = jax.sharding.PartitionSpec(sharded_axis)
    saved_key_sharding = jax.sharding.SingleDeviceSharding(
        devices[4], memory_kind='pinned_host'
    )
    restored_key_sharding = jax.sharding.SingleDeviceSharding(
        devices[0], memory_kind='pinned_host'
    )
    key = jax.device_put(jax.random.key(9), saved_key_sharding)
    vector = jax.numpy.arange(4, dtype=np.float32)
    state = {
        'key': key,
        'vector': jax.device_put(
            vector, jax.sharding.NamedSharding(save_mesh, partition_spec)
        ),
    }
    abstract_state = {
        'key': jax.ShapeDtypeStruct(
            (), key.dtype, sharding=restored_key_sharding
        ),
        'vector': jax.ShapeDtypeStruct(
            (4,),
            np.float32,
            sharding=jax.sharding.NamedSharding(load_mesh, partition_spec),
        ),
    }
    manager = snapshotter.Snapshotter(replica_axis_index=replica_axis_index)

    with mock.patch.object(
        jax.random, 'wrap_key_data', wraps=jax.random.wrap_key_data
    ) as mock_wrap_key_data:
      manager.save(1, state)
      manager.join()
      restored = manager.load(abstract_state)

    self.assertEqual(restored['key'].sharding, restored_key_sharding)
    self.assertEqual(restored['key'].dtype, key.dtype)
    np.testing.assert_array_equal(
        jax.random.key_data(restored['key']), jax.random.key_data(key)
    )
    np.testing.assert_array_equal(restored['vector'], vector)
    # JAX releases before 0.10.1 do not accept `dtype` in `wrap_key_data`.
    mock_wrap_key_data.assert_called()
    for call in mock_wrap_key_data.call_args_list:
      self.assertNotIn('dtype', call.kwargs)

  def test_load_preserves_slice_locality_and_stripes_donors(self):
    manager = snapshotter.Snapshotter()
    with mock.patch.object(jax, 'device_put', wraps=jax.device_put) as mock_put:
      manager.save(0, self.pytree)
      manager.join()
    self.assertEqual(mock_put.call_count, 1)
    assert manager._latest_snapshot is not None
    for leaf in jax.tree.leaves(manager._latest_snapshot[0][0]):
      if isinstance(leaf, jax.Array):
        snapshotter._unwrap_if_prng_key(leaf).delete()
    with mock.patch.object(jax, 'device_put', wraps=jax.device_put) as mock_put:
      restored = manager.load(self.abstract_pytree, reset_snapshot_state=False)
    test_utils.assert_tree_equal(self, self.pytree, restored)
    self.assertEqual(mock_put.call_count, 2)
    local_call, fill_call = mock_put.call_args_list
    for src, dst in zip(*local_call.args[:2]):
      self.assertEqual(src.sharding.device_set, dst.device_set)
    donors = {
        next(iter(src.sharding.device_set)).slice_index
        for src in fill_call.args[0]
    }
    self.assertEqual(donors, {1, 2, 3})

  def test_try_load_keeps_non_array_leaves_and_handles_errors(self):
    manager = snapshotter.Snapshotter()
    target = {'a': self.abstract_pytree['a'], 'i': {'k': 0}}
    self.assertEqual(manager.try_load(target), (None, None))
    manager.save(5, {'a': self.pytree['a'], 'i': {'k': 9}})
    step, restored = manager.try_load(target)
    assert restored is not None
    self.assertEqual((step, restored['i']), (5, {'k': 9}))
    self.assertTrue(manager.cancel_pending(0.0))
    assert manager.latest is not None
    self.assertEqual(manager.latest.step, 5)
    # A different tree structure is a caller error: the snapshot stays.
    with self.assertRaises(ValueError):
      manager.try_load({'a': self.abstract_pytree['a']})
    assert manager.latest is not None
    self.assertEqual(manager.latest.step, 5)
    # A different shape cannot be restored: the snapshot is dropped.
    bad_shape = SimpleNamespace(
        shape=(99,), sharding=self.abstract_pytree['a'].sharding
    )
    self.assertEqual(
        (manager.try_load({'a': bad_shape, 'i': {'k': 0}}), manager.latest),
        ((None, None), None),
    )

  def test_save_with_wait_takes_the_snapshot_before_returning(self):
    manager = snapshotter.Snapshotter()
    device = jax.devices()[0]
    scalar = jax.device_put(1, jax.sharding.SingleDeviceSharding(device))
    iterator = object()
    state = {'a': self.pytree['a'], 'it': iterator, 's': scalar}
    self.assertIsNone(manager.save(2, state))
    self.assertIsNone(manager.save(3, state, wait=True))
    assert manager.latest is not None
    self.assertEqual(manager.latest.step, 3)
    # A save with a checkpointer function takes the snapshot of `state` again
    # and passes the state that the snapshot holds to the function.
    pinned_states = []
    with mock.patch.object(
        manager, '_to_snapshot_state', wraps=manager._to_snapshot_state
    ) as to_snapshot_state:
      manager.save(3, state, checkpointer_save_fn=pinned_states.append)
    self.assertEqual((to_snapshot_state.call_count, len(pinned_states)), (1, 1))
    pinned = pinned_states[0]
    assert manager._latest_snapshot is not None
    self.assertIs(pinned, manager._latest_snapshot[2])
    self.assertIs(pinned['it'], iterator)
    # The snapshot replicates a single-device array across the mesh.
    replicated = jax.sharding.NamedSharding(
        self.pytree['a'].sharding.mesh,
        jax.sharding.PartitionSpec(),
        memory_kind='pinned_host',
    )
    self.assertEqual(
        (pinned['a'].sharding, pinned['s'].sharding), (replicated, replicated)
    )

  def test_save_with_checkpointer_fns(self):
    manager = snapshotter.Snapshotter()
    wait_fn, iterator, arrays = (
        mock.MagicMock(),
        object(),
        {'a': self.pytree['a']},
    )

    def share(step, state):
      # Returns the state that the checkpoint save function is given.
      return manager.save(
          step,
          state,
          checkpointer_wait_fn=wait_fn,
          checkpointer_save_fn=lambda pinned_state: pinned_state,
      )

    first_pinned = share(1, {**arrays, 'it': iterator})
    manager.save(2, arrays)
    manager.join()
    assert manager._latest_snapshot is not None
    second_snapshot_state = manager._latest_snapshot[2]
    manager.save(4, arrays)
    second_pinned = share(4, arrays)
    with mock.patch.object(jax, 'block_until_ready', side_effect=RuntimeError):
      failed_pinned = share(5, arrays)
    manager.save(6, arrays)
    checks = (
        second_snapshot_state['a'] is not second_pinned['a'],
        not first_pinned['a'].is_deleted(),
        failed_pinned is arrays,
    )
    abstract_array = self.abstract_pytree_2_slices['a']
    _, restored = manager.try_load(
        {'a': abstract_array}, reset_snapshot_state=False
    )
    assert restored is not None
    # The snapshot of step 6 is on another mesh than `restored`.
    third_pinned = share(6, restored)
    same_devices = (
        third_pinned['a'].sharding.device_set
        == abstract_array.sharding.device_set
    )
    results = (
        first_pinned['it'],
        *checks,
        not second_pinned['a'].is_deleted(),
        same_devices,
    )
    self.assertEqual(results, (iterator, True, True, True, True, True))
    self.assertEqual(wait_fn.call_count, 4)

  def test_save_calls_checkpointer_fns_in_order_and_raises_their_errors(self):
    manager = snapshotter.Snapshotter()
    arrays = {'a': self.pytree['a']}
    calls = mock.Mock()
    calls.save.return_value = 'saved'
    with mock.patch.object(
        manager, '_to_snapshot_state', wraps=manager._to_snapshot_state
    ) as to_snapshot_state:
      calls.attach_mock(to_snapshot_state, 'snapshot')
      result = manager.save(
          3,
          arrays,
          checkpointer_wait_fn=calls.wait,
          checkpointer_save_fn=calls.save,
      )
    self.assertEqual(
        (result, [name for name, _, _ in calls.mock_calls]),
        ('saved', ['wait', 'snapshot', 'save']),
    )
    # An error of a checkpointer function reaches the caller, and the worker
    # keeps running.
    failing_save_fn = mock.Mock(side_effect=ValueError)
    with self.assertRaises(ValueError):
      manager.save(4, arrays, checkpointer_save_fn=failing_save_fn)
    self.assertIsNone(manager.save(5, arrays, wait=True))
    assert manager.latest is not None
    self.assertEqual(manager.latest.step, 5)

  def test_save_re_snapshots_when_same_step_state_changes(self):
    manager = snapshotter.Snapshotter()
    arr1 = self.pytree['a'] + 1
    arr2 = self.pytree['a'] + 2
    state1 = {'a': arr1, 'meta': 'v1'}
    manager.save(10, state1, wait=True)

    # Same step, shape, dtype, and sharding, but different array values.
    state2 = {'a': arr2, 'meta': 'v1'}
    saved_states = []
    manager.save(10, state2, checkpointer_save_fn=saved_states.append)
    test_utils.assert_tree_equal(self, state2, saved_states[-1])
    self.assertEqual(saved_states[-1]['a'].sharding.memory_kind, 'pinned_host')

    # Same step and array object, but a different non-array leaf.
    state3 = {'a': arr2, 'meta': 'v2'}
    manager.save(10, state3, checkpointer_save_fn=saved_states.append)
    test_utils.assert_tree_equal(self, state3, saved_states[-1])
    target = {'a': self.abstract_pytree['a'], 'meta': ''}
    restored = manager.load(target, reset_snapshot_state=False)
    test_utils.assert_tree_equal(self, state3, restored)

  def test_copy_device_arrays_for_async_transfer(self):
    manager = snapshotter.Snapshotter(
        copy_device_arrays_for_async_transfer=True
    )
    scalar = jax.device_put(
        1, jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )
    pinned_sharding = self.pytree['a'].sharding.with_memory_kind('pinned_host')
    pinned_input = jax.device_put(self.pytree['a'], pinned_sharding)
    release = threading.Event()
    self.addCleanup(release.set)
    original_block = jax.block_until_ready

    def side_effect(tree):
      release.wait()
      return original_block(tree)

    with mock.patch.object(jax, 'block_until_ready', side_effect=side_effect):
      manager.save(10, {**self.pytree, 'a': pinned_input, 's': scalar})
      # Skipped, because the copy for step 10 is still in device memory.
      manager.save(11, self.pytree)
      # The caller may delete or donate its arrays as soon as `save` returns.
      pinned_input.delete()
      release.set()
      manager.join()
    assert manager.latest is not None
    self.assertEqual(manager.latest.step, 10)
    target = {**self.abstract_pytree_2_slices, 's': scalar}
    restored = manager.load(target, reset_snapshot_state=False)
    test_utils.assert_tree_equal(
        self, {**self.pytree_2_slices, 's': scalar}, restored
    )
    # A save that takes the snapshot before it returns makes no on-device copy.
    pinned_states = []
    with mock.patch.object(
        snapshotter, '_copy_arrays', wraps=snapshotter._copy_arrays
    ) as copy_arrays:
      manager.save(12, self.pytree, checkpointer_save_fn=pinned_states.append)
    copy_arrays.assert_not_called()
    self.assertEqual(pinned_states[0]['a'].sharding.memory_kind, 'pinned_host')

  @parameterized.named_parameters(
      dict(testcase_name='mismatched_replicas', replica_axis_index=0),
      dict(testcase_name='replica_axis_out_of_range', replica_axis_index=5),
  )
  def test_worker_survives_split_errors_of_on_device_copies(
      self, replica_axis_index
  ):
    manager = snapshotter.Snapshotter(
        replica_axis_index=replica_axis_index,
        copy_device_arrays_for_async_transfer=True,
    )
    mismatched_tree = {
        'four_slices': self.pytree['a'],
        'two_slices': self.pytree_2_slices['a'],
    }
    # The worker thread splits the on-device copies into replicas, so it gets
    # the error of the split. It logs the error and processes the next request.
    for step in (1, 2):
      manager.save(step, mismatched_tree)
      manager.join()
    self.assertIsNone(manager.latest)
    self.assertTrue(manager._worker_thread.is_alive())


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  g3_multiprocessing.handle_test_main(googletest.main)
