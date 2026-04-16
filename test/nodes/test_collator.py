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
from torchdata.nodes.batch import Batcher, Collator

from .utils import MockSource, run_test_save_load_state


class TestCollator(TestCase):
    def test_collator_default_collate(self) -> None:
        src = MockSource(num_samples=10)
        node = Collator(src, batch_size=4, drop_last=True)

        results = list(node)
        self.assertEqual(len(results), 2)
        for i, batch in enumerate(results):
            self.assertIsInstance(batch, dict)
            expected_steps = list(range(i * 4, i * 4 + 4))
            self.assertEqual(batch["step"].tolist(), expected_steps)
            self.assertEqual(batch["test_tensor"].tolist(), [[s] for s in expected_steps])

    def test_collator_drop_last_false(self) -> None:
        src = MockSource(num_samples=10)
        node = Collator(src, batch_size=4, drop_last=False)

        results = list(node)
        self.assertEqual(len(results), 3)
        # Last batch has 2 items
        self.assertEqual(len(results[2]["step"]), 2)

    def test_collator_custom_collate_fn(self) -> None:
        def custom_collate(batch):
            return [item["step"] for item in batch]

        src = MockSource(num_samples=6)
        node = Collator(src, batch_size=3, drop_last=True, collate_fn=custom_collate)

        results = list(node)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], [0, 1, 2])
        self.assertEqual(results[1], [3, 4, 5])

    def test_collator_batch_size_zero_raises(self) -> None:
        source = IterableWrapper(range(10))
        with self.assertRaises(ValueError):
            Collator(source, batch_size=0)

    def test_collator_is_batcher_subclass(self) -> None:
        src = MockSource(num_samples=6)
        node = Collator(src, batch_size=3)
        self.assertIsInstance(node, Batcher)

    def test_collator_default_collate_tensors(self) -> None:
        tensors = [torch.tensor([i, i + 1]) for i in range(8)]
        src = IterableWrapper(tensors)
        node = Collator(src, batch_size=4, drop_last=True)

        results = list(node)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].shape, (4, 2))
        self.assertEqual(results[1].shape, (4, 2))

    @parameterized.expand(itertools.product([0, 2], [True, False]))
    def test_save_load_state(self, midpoint: int, drop_last: bool) -> None:
        src = MockSource(num_samples=20)
        node = Collator(src, batch_size=4, drop_last=drop_last)
        run_test_save_load_state(self, node, midpoint)
