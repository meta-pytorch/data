# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase

from torch.utils.data import DistributedSampler, RandomSampler
from torchdata.nodes.adapters import IterableWrapper, MapStyleWrapper, SamplerWrapper

from .utils import DummyIterableDataset, DummyMapDataset, run_test_save_load_state, StatefulRange


class TestIterableWrapper(TestCase):
    def test_iterable(self):
        n = 20
        node = IterableWrapper(range(n))
        for _ in range(2):
            node.reset()
            result = list(node)
            self.assertEqual(len(result), n)
            for i, j in enumerate(result):
                self.assertEqual(j, i)

    def test_generator(self):
        n = 20
        node = IterableWrapper(f"str_{i}" for i in range(n))
        result = list(node)
        self.assertEqual(len(result), n)
        for i, j in enumerate(result):
            self.assertEqual(j, f"str_{i}")

        # Second time iter is called on generator will raise StopIteration
        result = list(node)
        self.assertEqual(len(result), 0)

    def test_iterable_dataset(self):
        n = 20
        node = IterableWrapper(DummyIterableDataset(n, name="test"))
        for _ in range(2):
            node.reset()
            result = list(node)
            self.assertEqual(len(result), n)
            for i, row in enumerate(result):
                self.assertEqual(row["step"], i)
                self.assertEqual(row["test_tensor"].item(), i)
                self.assertEqual(row["test_str"], f"str_{i}")

    @parameterized.expand([0, 5])
    def test_save_load_state_fast_forward(self, midpoint: int):
        run_test_save_load_state(self, IterableWrapper(range(10)), midpoint)

    @parameterized.expand([0, 5])
    def test_save_load_state_stateful(self, midpoint: int):
        run_test_save_load_state(self, IterableWrapper(StatefulRange(10)), midpoint)


class TestMapStyle(TestCase):
    def test_default_sampler(self):
        n = 20
        node = MapStyleWrapper(DummyMapDataset(n), sampler=range(n))
        for _ in range(2):
            node.reset()
            result = list(node)
            self.assertEqual(len(result), n)
            for i, row in enumerate(result):
                self.assertEqual(row["step"], i)
                self.assertEqual(row["test_tensor"].item(), i)
                self.assertEqual(row["test_str"], f"str_{i}")

    def test_random_sampler(self):
        n = 20
        ds = DummyMapDataset(n)
        node = MapStyleWrapper(ds, sampler=RandomSampler(ds))
        results = []
        for _ in range(2):
            node.reset()
            result = list(node)
            results.append(result)
            self.assertEqual(len(result), n)
            self.assertEqual({row["step"] for row in result}, set(range(n)))
            self.assertEqual({row["test_tensor"].item() for row in result}, set(range(n)))
            self.assertEqual(
                {row["test_str"] for row in result},
                {f"str_{i}" for i in range(n)},
            )

        self.assertNotEqual(results[0], results[1])  # Should have different values per epoch

    def test_dict(self):
        n = 20
        orig_ds = DummyMapDataset(n)
        d = {f"i{i}": orig_ds[i] for i in range(n)}
        sampler = list(d.keys())
        node = MapStyleWrapper(d, sampler=sampler)
        for _ in range(2):
            node.reset()
            result = list(node)
            self.assertEqual(len(result), n)
            for i, row in enumerate(result):
                self.assertEqual(row["step"], i)
                self.assertEqual(row["test_tensor"].item(), i)
                self.assertEqual(row["test_str"], f"str_{i}")

    @parameterized.expand([0, 7])
    def test_save_load_state_fast_forward(self, midpoint: int):
        n = 20
        node = MapStyleWrapper(DummyMapDataset(n), sampler=range(n))
        run_test_save_load_state(self, node, midpoint)

    @parameterized.expand([0, 7])
    def test_save_load_state_stateful(self, midpoint: int):
        n = 20
        node = MapStyleWrapper(DummyMapDataset(n), sampler=StatefulRange(n))
        run_test_save_load_state(self, node, midpoint)


class TestSamplerWrapper(TestCase):
    def test_sampler_wrapper(self):
        n = 20
        ds = DummyMapDataset(n)

        node = SamplerWrapper(sampler=RandomSampler(ds))

        results = []
        for epoch in range(2):
            node.reset()
            self.assertEqual(node.epoch, epoch)
            result = list(node)
            results.append(result)
            self.assertEqual(len(result), n)
            self.assertEqual(set(result), set(range(n)))

        self.assertNotEqual(results[0], results[1])

    def test_distributed_sampler(self):
        # Distributed sampler has set_epoch method
        n = 40
        ds = DummyMapDataset(n)

        sampler = DistributedSampler(ds, rank=1, num_replicas=2)
        exp = []
        for epoch in range(4):
            sampler.set_epoch(epoch)
            exp.append(list(sampler))

        node = SamplerWrapper(sampler=sampler)

        for epoch in range(4):
            node.reset()
            result = list(node)
            self.assertEqual(result, exp[epoch])

    @parameterized.expand([0, 7])
    def test_save_load_state(self, midpoint: int):
        n = 20
        ds = DummyMapDataset(n)
        sampler = DistributedSampler(ds, rank=1, num_replicas=2)
        node = SamplerWrapper(sampler=sampler)
        run_test_save_load_state(self, node, midpoint)

    @parameterized.expand([0, 7])
    def test_save_load_state_with_updater(self, midpoint: int):
        n = 20
        ds = DummyMapDataset(n)
        initial_epoch = 2

        def epoch_updater(epoch):
            return epoch + 5

        sampler = DistributedSampler(ds, rank=1, num_replicas=2)
        node = SamplerWrapper(sampler=sampler, initial_epoch=initial_epoch, epoch_updater=epoch_updater)
        run_test_save_load_state(self, node, midpoint)
