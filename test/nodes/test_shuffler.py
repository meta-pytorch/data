import itertools

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes import Shuffler
from torchdata.nodes.adapters import IterableWrapper

from .utils import MockSource, StatefulRangeNode


class TestShuffler(TestCase):
    def test_shuffler_basic(self) -> None:
        # Test with a simple range and different buffer sizes
        for buffer_size in [1, 3, 10]:
            source = IterableWrapper(range(10))
            node = Shuffler(source, buffer_size=buffer_size, seed=42)

            results = list(node)

            # Results should contain all original items
            self.assertEqual(sorted(results), list(range(10)))

            # With buffer_size=1, no shuffling should occur
            if buffer_size == 1:
                self.assertEqual(results, list(range(10)))
            else:
                # With buffer_size > 1, results should be shuffled
                self.assertNotEqual(results, list(range(10)))

            # Verify yielded counter
            self.assertEqual(node._num_yielded, 10)

    def test_shuffler_default_buffer_size(self) -> None:
        # Test that default buffer size works
        source = IterableWrapper(range(50))
        node = Shuffler(source, seed=42)  # Using default buffer_size=1024

        results = list(node)

        # Results should contain all original items
        self.assertEqual(sorted(results), list(range(50)))
        # With default buffer size > 1, results should be shuffled
        self.assertNotEqual(results, list(range(50)))

        # Verify yielded counter
        self.assertEqual(node._num_yielded, 50)
        # Verify default buffer size
        self.assertEqual(node.buffer_size, 1024)

    def test_shuffler_deterministic(self) -> None:
        # Test that results are deterministic with the same seed
        source1 = IterableWrapper(range(20))
        source2 = IterableWrapper(range(20))

        node1 = Shuffler(source1, buffer_size=10, seed=12345)
        node2 = Shuffler(source2, buffer_size=10, seed=12345)

        results1 = list(node1)
        results2 = list(node2)

        # Results should be identical with the same seed
        self.assertEqual(results1, results2)

        # Results should be different with different seeds
        source3 = IterableWrapper(range(20))
        node3 = Shuffler(source3, buffer_size=10, seed=54321)
        results3 = list(node3)

        # Very unlikely that different seeds produce same shuffle
        self.assertNotEqual(results1, results3)

    def test_shuffler_with_mock_source(self) -> None:
        num_samples = 20
        source = MockSource(num_samples=num_samples)
        node = Shuffler(source, buffer_size=10, seed=42)

        results = list(node)
        self.assertEqual(len(results), num_samples)

        # Verify yielded counter
        self.assertEqual(node._num_yielded, num_samples)

        # Check that all items are present
        step_values = [result["step"] for result in results]
        self.assertEqual(sorted(step_values), list(range(num_samples)))

        # With seed=42, result should be shuffled
        self.assertNotEqual(step_values, list(range(num_samples)))

    def test_shuffler_empty_source(self) -> None:
        source = IterableWrapper([])
        node = Shuffler(source, buffer_size=5, seed=42)

        results = list(node)
        self.assertEqual(results, [])

        # Verify yielded counter with empty source
        self.assertEqual(node._num_yielded, 0)

    def test_store_buffer_enabled(self) -> None:
        """Test behavior when store_buffer=True (default)."""
        source = IterableWrapper(range(10))
        node = Shuffler(source, buffer_size=5, seed=42, store_buffer=True)

        # Consume some items
        items_before = []
        for _ in range(3):
            items_before.append(next(node))

        # Get state - should include buffer contents
        state = node.state_dict()
        self.assertIn(node.BUFFER_KEY, state)
        self.assertTrue(len(state[node.BUFFER_KEY]) <= 5)  # Buffer size constraint

        # Create new node and restore state
        new_source = IterableWrapper(range(10))
        new_node = Shuffler(new_source, buffer_size=5, seed=42, store_buffer=True)
        new_node.reset(state)

        # Should be able to continue from exact same state
        self.assertEqual(new_node._num_yielded, 3)
        self.assertTrue(len(new_node.buffer) <= 5)

    def test_store_buffer_disabled(self) -> None:
        """Test behavior when store_buffer=False."""
        source = IterableWrapper(range(10))
        node = Shuffler(source, buffer_size=5, seed=42, store_buffer=False)

        # Consume some items
        items_before = []
        for _ in range(3):
            items_before.append(next(node))

        # Get state - should NOT include buffer contents
        state = node.state_dict()
        self.assertNotIn(node.BUFFER_KEY, state)
        self.assertEqual(state[node.STORE_BUFFER_KEY], False)

        # Create new node and restore state
        new_source = IterableWrapper(range(10))
        new_node = Shuffler(new_source, buffer_size=5, seed=42, store_buffer=False)
        new_node.reset(state)

        # Counter should be preserved but buffer should be empty initially
        self.assertEqual(new_node._num_yielded, 3)
        self.assertEqual(len(new_node.buffer), 0)  # Buffer not restored

    @parameterized.expand(itertools.product([0, 3, 7]))
    def test_save_load_state_with_buffer_storage(self, midpoint: int) -> None:
        """Test that saving and loading state preserves the deterministic sequence when store_buffer=True."""
        n = 20
        source = StatefulRangeNode(n=n)
        node = Shuffler(source, buffer_size=5, seed=42, store_buffer=True)

        # First, collect all items without interruption
        source_copy = StatefulRangeNode(n=n)
        node_copy = Shuffler(source_copy, buffer_size=5, seed=42, store_buffer=True)
        expected_items = list(node_copy)

        # Now collect with save/load at midpoint
        items_before = []
        for _ in range(midpoint):
            try:
                items_before.append(next(node))
            except StopIteration:
                break

        # Save state and create a new node
        state = node.state_dict()
        new_source = StatefulRangeNode(n=n)
        new_node = Shuffler(new_source, buffer_size=5, seed=42, store_buffer=True)
        new_node.reset(state)

        # Collect remaining items
        items_after = []
        try:
            while True:
                items_after.append(next(new_node))
        except StopIteration:
            pass

        # Combined sequence should match the expected sequence exactly when buffer is stored
        combined = items_before + items_after
        self.assertEqual(len(combined), len(expected_items))

        # For dictionary items from StatefulRangeNode, compare the entire dictionaries
        if expected_items and isinstance(expected_items[0], dict):
            # StatefulRangeNode returns dictionaries with 'i' and 'resets' keys
            self.assertEqual(combined, expected_items)
        else:
            # For non-dictionary items, check exact match
            self.assertEqual(combined, expected_items)

        # Verify that the first items match exactly (deterministic order)
        if midpoint > 0:
            self.assertEqual(items_before, expected_items[:midpoint])

    @parameterized.expand(itertools.product([0, 3, 7]))
    def test_save_load_state_without_buffer_storage(self, midpoint: int) -> None:
        """Test that saving and loading state works when store_buffer=False."""
        n = 20
        source = StatefulRangeNode(n=n)
        node = Shuffler(source, buffer_size=5, seed=42, store_buffer=False)

        # Collect with save/load at midpoint
        items_before = []
        for _ in range(midpoint):
            try:
                items_before.append(next(node))
            except StopIteration:
                break

        # Save state and create a new node
        state = node.state_dict()
        new_source = StatefulRangeNode(n=n)
        new_node = Shuffler(new_source, buffer_size=5, seed=42, store_buffer=False)
        new_node.reset(state)

        # Collect remaining items
        items_after = []
        try:
            while True:
                items_after.append(next(new_node))
        except StopIteration:
            pass

        # Combined sequence will have fewer items than expected because
        # we don't preserve the buffer in the state. We expect to lose
        # approximately buffer_size items.
        combined = items_before + items_after
        self.assertLessEqual(len(combined), n)

        # For dictionary items from StatefulRangeNode, extract the 'i' values for comparison
        if combined and isinstance(combined[0], dict):
            # Verify each item has the expected format from StatefulRangeNode
            for item in combined:
                self.assertIsInstance(item, dict)
                self.assertIn("i", item)
                self.assertIn("resets", item)

            # Check that combined 'i' values are a subset of possible values
            combined_i_values = [item["i"] for item in combined]
            for i_value in combined_i_values:
                self.assertIn(i_value, range(n))
        else:
            # For non-dictionary items, check subset relationship
            for item in combined:
                self.assertIn(item, range(n))

    def test_shuffler_reset_state(self) -> None:
        # This test verifies that after resetting with a state,
        # the counter is preserved and buffer behavior depends on store_buffer setting

        source = IterableWrapper(range(10))
        node = Shuffler(source, buffer_size=5, seed=42, store_buffer=True)

        # Consume first three items
        for _ in range(3):
            next(node)

        # Check counter after consuming items
        self.assertEqual(node._num_yielded, 3)

        # Get state and reset
        state = node.state_dict()

        # Create a new node with a fresh source
        new_source = IterableWrapper(range(10))
        new_node = Shuffler(new_source, buffer_size=5, seed=42, store_buffer=True)
        new_node.reset(state)

        # Counter should be preserved after reset with state
        self.assertEqual(new_node._num_yielded, 3)

        # With store_buffer=True, buffer should be restored
        self.assertGreater(len(new_node.buffer), 0)

    def test_counter_reset(self) -> None:
        # Test that counter is properly reset
        source = IterableWrapper(range(10))
        node = Shuffler(source, buffer_size=5, seed=42)

        # Consume all items
        list(node)

        # Verify counter after first pass
        self.assertEqual(node._num_yielded, 10)

        # Reset without state
        node.reset()

        # Counter should be reset to 0
        self.assertEqual(node._num_yielded, 0)

        # Consume some items
        for _ in range(3):
            next(node)

        # Verify counter after partial consumption
        self.assertEqual(node._num_yielded, 3)

    def test_invalid_input(self) -> None:
        # Test with invalid buffer size
        source = IterableWrapper(range(10))

        # Buffer size must be at least 1
        with self.assertRaises(ValueError):
            Shuffler(source, buffer_size=0)

        with self.assertRaises(ValueError):
            Shuffler(source, buffer_size=-3)

    def test_buffer_size_mismatch(self) -> None:
        # Create a shuffler with initial buffer size
        source = IterableWrapper(range(10))
        node = Shuffler(source, buffer_size=5, seed=42)

        # Consume some items and get state
        for _ in range(3):
            next(node)
        state = node.state_dict()

        # Try to load state into a shuffler with different buffer size
        new_source = IterableWrapper(range(10))
        new_node = Shuffler(new_source, buffer_size=10, seed=42)  # Different buffer size

        # Should raise ValueError due to buffer size mismatch
        with self.assertRaises(ValueError):
            new_node.reset(state)
