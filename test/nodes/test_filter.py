# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes import Batcher, Filter
from torchdata.nodes.adapters import IterableWrapper

from .utils import MockSource, run_test_save_load_state, StatefulRangeNode


class TestFilter(TestCase):
    def test_filter_basic(self) -> None:
        # Test with a simple range
        source = IterableWrapper(range(10))
        node = Filter(source, lambda x: x % 2 == 0)  # Keep even numbers

        results = list(node)
        self.assertEqual(results, [0, 2, 4, 6, 8])

        # Verify counters
        self.assertEqual(node._num_yielded, 5)  # 5 even numbers were yielded
        self.assertEqual(node._num_filtered, 5)  # 5 odd numbers were filtered out

        # Test with a different predicate
        source = IterableWrapper(range(10))
        node = Filter(source, lambda x: x > 5)  # Keep numbers greater than 5

        results = list(node)
        self.assertEqual(results, [6, 7, 8, 9])

        # Verify counters
        self.assertEqual(node._num_yielded, 4)  # 4 numbers > 5 were yielded
        self.assertEqual(node._num_filtered, 6)  # 6 numbers <= 5 were filtered out

    def test_filter_with_mock_source(self) -> None:
        num_samples = 20
        source = MockSource(num_samples=num_samples)
        node = Filter(source, lambda x: x["step"] % 3 == 0)  # Keep items where step is divisible by 3

        # Test multi epoch
        for _ in range(2):
            node.reset()
            results = list(node)
            expected_steps = [i for i in range(num_samples) if i % 3 == 0]
            self.assertEqual(len(results), len(expected_steps))

            # Verify counters after each epoch
            self.assertEqual(node._num_yielded, len(expected_steps))
            self.assertEqual(node._num_filtered, num_samples - len(expected_steps))

            for i, result in enumerate(results):
                expected_step = expected_steps[i]
                self.assertEqual(result["step"], expected_step)
                self.assertEqual(result["test_tensor"].item(), expected_step)
                self.assertEqual(result["test_str"], f"str_{expected_step}")

    def test_filter_empty_result(self) -> None:
        source = IterableWrapper(range(10))
        node = Filter(source, lambda x: x > 100)  # No items will pass this filter

        results = list(node)
        self.assertEqual(results, [])

        # Verify counters when no items pass the filter
        self.assertEqual(node._num_yielded, 0)  # No items were yielded
        self.assertEqual(node._num_filtered, 10)  # All 10 items were filtered out

    @parameterized.expand(itertools.product([0, 3, 7]))
    def test_save_load_state(self, midpoint: int):
        n = 50
        source = StatefulRangeNode(n=n)
        node = Filter(source, lambda x: x["i"] % 3 == 0)  # Keep items where 'i' is divisible by 3
        run_test_save_load_state(self, node, midpoint)

    def test_filter_reset_state(self) -> None:
        source = IterableWrapper(range(10))
        node = Filter(source, lambda x: x % 2 == 0)

        # Consume first two items
        self.assertEqual(next(node), 0)
        self.assertEqual(next(node), 2)

        # Check counters after consuming two items
        self.assertEqual(node._num_yielded, 2)  # 2 even numbers were yielded
        self.assertEqual(node._num_filtered, 1)  # 1 odd number was filtered out

        # Get state and reset
        state = node.state_dict()
        node.reset(state)

        # Counters should be preserved after reset with state
        self.assertEqual(node._num_yielded, 2)
        self.assertEqual(node._num_filtered, 1)

        # Should continue from where we left off
        self.assertEqual(next(node), 4)
        self.assertEqual(next(node), 6)
        self.assertEqual(next(node), 8)

        # Counters should be updated after consuming more items
        self.assertEqual(node._num_yielded, 5)  # Total of 5 even numbers yielded
        self.assertEqual(node._num_filtered, 4)  # Total of 4 odd numbers filtered out

        # Should raise StopIteration after all items are consumed
        with self.assertRaises(StopIteration):
            next(node)

    def test_filter_with_batcher(self) -> None:
        # Test Filter node with Batcher

        # Create a source with numbers 0-19
        source = IterableWrapper(range(20))

        # Batch into groups of 4
        batch_node = Batcher(source, batch_size=4)

        # Filter to keep only batches where the sum is divisible by 10
        filter_node = Filter(batch_node, lambda batch: sum(batch) % 10 == 0)

        # Let's calculate the expected batches and their sums
        # Batch 1: [0, 1, 2, 3] -> sum = 6
        # Batch 2: [4, 5, 6, 7] -> sum = 22
        # Batch 3: [8, 9, 10, 11] -> sum = 38
        # Batch 4: [12, 13, 14, 15] -> sum = 54
        # Batch 5: [16, 17, 18, 19] -> sum = 70
        # Batches with sum divisible by 10: Batch 5 (70)

        results = list(filter_node)

        # We expect only one batch to pass the filter (sum divisible by 10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], [16, 17, 18, 19])  # sum = 70

        # Check that the filter node tracked both filtered and yielded items
        self.assertEqual(filter_node._num_yielded, 1)  # 1 batch was yielded
        self.assertEqual(filter_node._num_filtered, 4)  # 4 batches were filtered out

        # Verify total number of batches processed
        self.assertEqual(filter_node._num_yielded + filter_node._num_filtered, 5)  # Total of 5 batches

    def test_counter_reset(self) -> None:
        # Test that counters are properly reset
        source = IterableWrapper(range(10))
        node = Filter(source, lambda x: x % 2 == 0)

        # Consume all items
        list(node)

        # Verify counters after first pass
        self.assertEqual(node._num_yielded, 5)
        self.assertEqual(node._num_filtered, 5)

        # Reset without state
        node.reset()

        # Counters should be reset to 0
        self.assertEqual(node._num_yielded, 0)
        self.assertEqual(node._num_filtered, 0)

        # Consume some items
        next(node)  # 0
        next(node)  # 2

        # Verify counters after partial consumption
        self.assertEqual(node._num_yielded, 2)
        self.assertEqual(node._num_filtered, 1)
