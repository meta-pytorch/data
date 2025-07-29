# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper

from torchdata.nodes.round_robin_cycler import RoundRobinNode

from .utils import run_test_save_load_state


class TestRoundRobinNode(TestCase):
    def _create_test_nodes(self) -> list[IterableWrapper]:
        node_A = IterableWrapper([{"text": "A", "label": 0}])
        node_B = IterableWrapper(
            [
                {"text": "B", "label": 0},
                {"text": "B", "label": 1},
            ]
        )
        node_C = IterableWrapper(
            [
                {"text": "C", "label": 0},
                {"text": "C", "label": 1},
                {"text": "C", "label": 2},
            ]
        )
        return [node_A, node_B, node_C]

    def _create_single_node(self) -> list[IterableWrapper]:
        return [IterableWrapper([{"text": "A", "label": 0}])]

    def test_init(self) -> None:
        nodes = self._create_test_nodes()
        round_robin_node = RoundRobinNode(nodes)

        self.assertEqual(round_robin_node.source_nodes, nodes)
        self.assertEqual(round_robin_node._total_items_yielded, 0)
        self.assertEqual(round_robin_node._current_node_index, 0)
        self.assertEqual(round_robin_node._active_nodes, list(range(len(nodes))))
        self.assertEqual(round_robin_node._node_item_indices, {0: 0, 1: 0, 2: 0})

    def test_init_empty_source_list(self) -> None:
        with self.assertRaises(ValueError) as cm:
            RoundRobinNode([])
        self.assertIn("At least one source node must be provided", str(cm.exception))

    def test_next(self) -> None:
        nodes = self._create_test_nodes()
        round_robin_node = RoundRobinNode(nodes)

        self.assertEqual(round_robin_node.next(), {"text": "A", "label": 0})
        self.assertEqual(round_robin_node.next(), {"text": "B", "label": 0})
        self.assertEqual(round_robin_node.next(), {"text": "C", "label": 0})
        self.assertEqual(round_robin_node.next(), {"text": "B", "label": 1})
        self.assertEqual(round_robin_node.next(), {"text": "C", "label": 1})
        self.assertEqual(round_robin_node.next(), {"text": "C", "label": 2})

        with self.assertRaises(StopIteration):
            round_robin_node.next()

    def test_reset(self) -> None:
        nodes = self._create_test_nodes()
        round_robin_node = RoundRobinNode(nodes)

        for _ in range(2):
            round_robin_node.next()

        state = round_robin_node.get_state()

        round_robin_node.reset()

        self.assertEqual(round_robin_node._total_items_yielded, 0)
        self.assertEqual(round_robin_node._current_node_index, 0)
        self.assertEqual(round_robin_node._active_nodes, list(range(len(nodes))))
        self.assertEqual(round_robin_node._node_item_indices, {0: 0, 1: 0, 2: 0})

        restarted_round_robin_node = RoundRobinNode(nodes)
        restarted_round_robin_node.reset(state)

        self.assertEqual(restarted_round_robin_node._total_items_yielded, 2)
        self.assertEqual(restarted_round_robin_node._current_node_index, 2)

        self.assertEqual(next(restarted_round_robin_node), {"text": "C", "label": 0})

    def test_single_source_node(self) -> None:
        single_node = self._create_single_node()
        round_robin_node = RoundRobinNode(single_node)

        self.assertEqual(len(round_robin_node._active_nodes), 1)
        self.assertEqual(round_robin_node._node_item_indices, {0: 0})

        self.assertEqual(round_robin_node.next(), {"text": "A", "label": 0})

        with self.assertRaises(StopIteration):
            round_robin_node.next()

    def test_explicit_iterator_exhaustion(self) -> None:
        node_A = IterableWrapper([{"text": "A", "label": 0}])
        node_B = IterableWrapper([{"text": "B", "label": 0}, {"text": "B", "label": 1}])
        node_C = IterableWrapper(
            [
                {"text": "C", "label": 0},
                {"text": "C", "label": 1},
                {"text": "C", "label": 2},
            ]
        )

        round_robin_node = RoundRobinNode([node_A, node_B, node_C])

        self.assertEqual(round_robin_node.next(), {"text": "A", "label": 0})
        self.assertEqual(round_robin_node.next(), {"text": "B", "label": 0})
        self.assertEqual(round_robin_node.next(), {"text": "C", "label": 0})

        self.assertEqual(round_robin_node.next(), {"text": "B", "label": 1})
        self.assertEqual(round_robin_node.next(), {"text": "C", "label": 1})

        self.assertEqual(round_robin_node.next(), {"text": "C", "label": 2})

        with self.assertRaises(StopIteration):
            round_robin_node.next()

        self.assertEqual(len(round_robin_node._active_nodes), 0)
        self.assertEqual(round_robin_node._node_item_indices, {0: 1, 1: 2, 2: 3})

    def test_state_management(self) -> None:
        nodes = self._create_test_nodes()
        round_robin_node = RoundRobinNode(nodes)

        initial_state = round_robin_node.get_state()

        self.assertIn(RoundRobinNode.SOURCE_STATES_KEY, initial_state)
        self.assertIn(RoundRobinNode.ACTIVE_NODES_KEY, initial_state)
        self.assertIn(RoundRobinNode.NODE_ITEM_INDICES_KEY, initial_state)
        self.assertIn(RoundRobinNode.CURRENT_NODE_INDEX_KEY, initial_state)
        self.assertIn(RoundRobinNode.TOTAL_ITEMS_YIELDED_KEY, initial_state)

        self.assertEqual(initial_state[RoundRobinNode.TOTAL_ITEMS_YIELDED_KEY], 0)
        self.assertEqual(initial_state[RoundRobinNode.CURRENT_NODE_INDEX_KEY], 0)
        self.assertEqual(len(initial_state[RoundRobinNode.SOURCE_STATES_KEY]), len(nodes))
        self.assertEqual(initial_state[RoundRobinNode.ACTIVE_NODES_KEY], list(range(len(nodes))))
        self.assertEqual(initial_state[RoundRobinNode.NODE_ITEM_INDICES_KEY], {0: 0, 1: 0, 2: 0})

        round_robin_node.next()
        round_robin_node.next()

        updated_state = round_robin_node.get_state()

        self.assertEqual(updated_state[RoundRobinNode.TOTAL_ITEMS_YIELDED_KEY], 2)
        self.assertEqual(updated_state[RoundRobinNode.CURRENT_NODE_INDEX_KEY], 2)
        self.assertEqual(updated_state[RoundRobinNode.NODE_ITEM_INDICES_KEY], {0: 1, 1: 1, 2: 0})

    def test_error_handling_invalid_state(self) -> None:
        nodes = self._create_test_nodes()
        round_robin_node = RoundRobinNode(nodes)

        incomplete_state = {
            RoundRobinNode.TOTAL_ITEMS_YIELDED_KEY: 1,
        }

        with self.assertRaises(KeyError):
            round_robin_node.reset(incomplete_state)

    @parameterized.expand(itertools.product([0, 2, 4]))
    def test_save_load_state(self, midpoint: int) -> None:
        nodes = self._create_test_nodes()
        round_robin_node = RoundRobinNode(nodes)
        run_test_save_load_state(self, round_robin_node, midpoint)

    def test_round_robin_behavior_detailed(self) -> None:
        nodes = self._create_test_nodes()
        round_robin_node = RoundRobinNode(nodes)

        expected_sequence = [
            {"text": "A", "label": 0},
            {"text": "B", "label": 0},
            {"text": "C", "label": 0},
            {"text": "B", "label": 1},
            {"text": "C", "label": 1},
            {"text": "C", "label": 2},
        ]

        results = []
        for expected_item in expected_sequence:
            item = round_robin_node.next()
            results.append(item)
            self.assertEqual(item, expected_item)

        self.assertEqual(results, expected_sequence)
        self.assertEqual(round_robin_node._total_items_yielded, len(expected_sequence))

        self.assertEqual(len(round_robin_node._active_nodes), 0)

    def test_empty_nodes_mixed_with_valid_nodes(self) -> None:
        empty_node = IterableWrapper([])
        valid_node = IterableWrapper([{"text": "Valid", "label": 1}])
        another_empty = IterableWrapper([])

        round_robin_node = RoundRobinNode([empty_node, valid_node, another_empty])

        self.assertEqual(round_robin_node.next(), {"text": "Valid", "label": 1})

        with self.assertRaises(StopIteration):
            round_robin_node.next()
