# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes import IterableWrapper
from torchdata.nodes.batch import Batcher, Unbatcher

from .utils import MockSource, run_test_save_load_state


class TestBatcher(TestCase):
    def test_batcher(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size, drop_last=True)

        results = list(node)
        self.assertEqual(len(results), 3)
        for i in range(3):
            for j in range(batch_size):
                self.assertEqual(results[i][j]["step"], i * batch_size + j)
                self.assertEqual(results[i][j]["test_tensor"], torch.tensor([i * batch_size + j]))
                self.assertEqual(results[i][j]["test_str"], f"str_{i * batch_size + j}")

    def test_batcher_batch_size_zero_raises(self):
        source = IterableWrapper(range(10))
        with self.assertRaises(ValueError):
            Batcher(source, batch_size=0)

    def test_batcher_drop_last_false(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        root = Batcher(src, batch_size=batch_size, drop_last=False)

        results = list(root)
        self.assertEqual(len(results), 4)
        for i in range(4):
            n = batch_size if i < 3 else 2
            for j in range(n):
                self.assertEqual(results[i][j]["step"], i * batch_size + j)
                self.assertEqual(results[i][j]["test_tensor"], torch.tensor([i * batch_size + j]))
                self.assertEqual(results[i][j]["test_str"], f"str_{i * batch_size + j}")

    @parameterized.expand(itertools.product([0, 2], [True, False]))
    def test_save_load_state_fast_forward(self, midpoint: int, drop_last: bool):
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size, drop_last=drop_last)
        run_test_save_load_state(self, node, midpoint)


class TestUnbatcher(TestCase):
    def test_unbatcher(self) -> None:
        batch_size = 6
        n = 20
        src = MockSource(num_samples=n)
        node = Batcher(src, batch_size=batch_size, drop_last=False)
        node = Unbatcher(node)

        results = list(node)
        self.assertEqual(len(results), n)
        for i in range(n):
            self.assertEqual(results[i]["step"], i)
            self.assertEqual(results[i]["test_tensor"], torch.tensor([i]))
            self.assertEqual(results[i]["test_str"], f"str_{i}")

    @parameterized.expand(itertools.product([0, 2], [True, False]))
    def test_save_load_state_fast_forward(self, midpoint: int, drop_last: bool):
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size, drop_last=drop_last)
        node = Unbatcher(node)
        run_test_save_load_state(self, node, midpoint)
