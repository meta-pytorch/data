# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from unittest import mock

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.batch import Batcher
from torchdata.nodes.prefetch import Prefetcher

from .utils import IterInitError, MockSource, run_test_save_load_state, StatefulRangeNode


class TestPrefetcher(TestCase):
    def test_prefetcher(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size, drop_last=True)
        root = Prefetcher(node, prefetch_factor=2)

        # Test multi epoch shutdown and restart
        for _ in range(2):
            root.reset()
            results = list(root)
            self.assertEqual(len(results), 3)
            for i in range(3):
                for j in range(batch_size):
                    self.assertEqual(results[i][j]["step"], i * batch_size + j)
                    self.assertEqual(results[i][j]["test_tensor"], torch.tensor([i * batch_size + j]))
                    self.assertEqual(results[i][j]["test_str"], f"str_{i * batch_size + j}")

    def test_iter_init_error(self):
        node = IterInitError()
        root = Prefetcher(node, prefetch_factor=2)

        with self.assertRaisesRegex(ValueError, "Iter Init Error"):
            list(root)

    @parameterized.expand(itertools.product([0, 7, 32], [0, 1, 9]))
    def test_save_load_state_stateful(self, midpoint: int, snapshot_frequency: int):
        batch_size = 6
        n = 200
        src = StatefulRangeNode(n=n)
        node = Batcher(src, batch_size=batch_size, drop_last=False)
        node = Prefetcher(node, prefetch_factor=8, snapshot_frequency=snapshot_frequency)
        run_test_save_load_state(self, node, midpoint)

    def test_explicit_shutdown(self):
        """Test that the explicit shutdown method properly shuts down the node."""
        mock_source = mock.MagicMock()
        mock_source.shutdown = mock.MagicMock()
        node = Prefetcher(mock_source, prefetch_factor=2)
        node.reset()
        # Mock the _shutdown method of the iterator
        with mock.patch.object(node._it, "_shutdown") as mock_shutdown:
            node.shutdown()
            mock_shutdown.assert_called_once()
            mock_source.shutdown.assert_called_once()


class ResetFailureSource(StatefulRangeNode):
    fail_next_reset = False
    shutdown_calls = 0

    def reset(self, initial_state=None):
        if self.fail_next_reset:
            self.fail_next_reset = False
            raise ValueError("temporary reset failure")
        super().reset(initial_state)

    def shutdown(self):
        self.shutdown_calls += 1


class TestPrefetcherResetFailure(TestCase):
    @parameterized.expand([False, True])
    def test_cleanup_or_retry_after_failed_replacement(self, retry):
        source = ResetFailureSource(n=4)
        node = Prefetcher(source, prefetch_factor=2)
        node.reset()
        self.assertEqual(next(node)["i"], 0)
        source.fail_next_reset = True
        with self.assertRaisesRegex(ValueError, "temporary reset failure"):
            node.reset()
        if retry:
            node.reset()
            self.assertEqual([item["i"] for item in node], list(range(4)))
        node.shutdown()
        self.assertEqual(source.shutdown_calls, 1)
