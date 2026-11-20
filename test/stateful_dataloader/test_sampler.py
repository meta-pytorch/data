# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch

from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_TSAN, TestCase

from torch.utils.data import Dataset

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler, StatefulDistributedSampler


class MockDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.arange(size)  # Simple data that is easy to verify

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
@unittest.skipIf(
    TEST_WITH_ASAN,
    "DataLoader tests hang in ASAN, see: https://github.com/pytorch/pytorch/issues/66223",
)
class TestDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = MockDataset(100)
        self.persistent_workers = False

    def test_initialization_StatefulDistributedSampler(self):

        sampler = StatefulDistributedSampler(
            self.dataset,
            num_replicas=10,
            rank=0,
            shuffle=False,
            seed=42,
            drop_last=False,
        )
        self.assertEqual(sampler.dataset_size, len(self.dataset))
        self.assertEqual(sampler.num_replicas, 10)
        self.assertEqual(sampler.rank, 0)
        self.assertFalse(sampler.shuffle)
        self.assertEqual(sampler.seed, 42)
        self.assertFalse(sampler.drop_last)
        self.assertEqual(sampler.yielded, 0)
        self.assertIsNone(sampler.next_yielded)

    def test_dataloader_state_dict(self):
        sampler = StatefulDistributedSampler(self.dataset, num_replicas=1, rank=0, shuffle=False)
        dataloader = StatefulDataLoader(self.dataset, batch_size=10, sampler=sampler)
        # Partial iteration over the DataLoader
        iter_count = 5
        for i, _ in enumerate(dataloader):
            if i == iter_count - 1:
                break
        state_dict = dataloader.state_dict()
        new_sampler = StatefulDistributedSampler(self.dataset, num_replicas=1, rank=0, shuffle=False)

        new_dataloader = StatefulDataLoader(self.dataset, batch_size=10, sampler=new_sampler)
        new_dataloader.load_state_dict(state_dict)
        resumed_data = []
        for data in new_dataloader:
            resumed_data.append(data.tolist())
        expected_data = []
        full_dataloader = StatefulDataLoader(self.dataset, batch_size=10, sampler=sampler)
        for data in full_dataloader:
            expected_data.append(data.tolist())

        self.assertEqual(resumed_data, expected_data[iter_count:])

    def test_sampler_state_dict(self):

        sampler = StatefulDistributedSampler(self.dataset, num_replicas=10, rank=0)
        sampler.yielded = 5
        state_dict = sampler.state_dict()
        self.assertEqual(state_dict["yielded"], 5)

    def test_sampler_load_state_dict(self):

        sampler = StatefulDistributedSampler(self.dataset, num_replicas=10, rank=0)
        sampler.load_state_dict({"epoch": 0, "yielded": 3})
        self.assertEqual(sampler.next_yielded, 3)
        with self.assertRaises(ValueError):
            sampler.load_state_dict({"epoch": 0, "yielded": -1})

    def test_sampler_next_yielded(self):

        sampler = StatefulDistributedSampler(self.dataset, num_replicas=2, rank=0, shuffle=True, seed=42)
        iterator = iter(sampler)
        next(iterator)  # advance the iterator
        self.assertEqual(sampler.yielded, 1)
        self.assertIsNone(sampler.next_yielded)
        sampler.load_state_dict(
            {
                StatefulDistributedSampler._EPOCH: 0,
                StatefulDistributedSampler._YIELDED: 5,
            }
        )
        self.assertEqual(sampler.next_yielded, 5)
        iterator = iter(sampler)
        next(iterator)  # advance the iterator again
        self.assertEqual(sampler.yielded, 6)

    def test_drop_last_effect(self):
        num_replicas = 3
        total_samples = len(self.dataset)
        expected_length_with_drop = total_samples // num_replicas
        expected_length_without_drop = math.ceil(total_samples / num_replicas)

        sampler_with_drop = StatefulDistributedSampler(
            self.dataset, num_replicas=3, rank=0, drop_last=True, shuffle=False
        )
        dataloader_with_drop = StatefulDataLoader(self.dataset, sampler=sampler_with_drop)

        sampler_without_drop = StatefulDistributedSampler(
            self.dataset, num_replicas=3, rank=0, drop_last=False, shuffle=False
        )
        dataloader_without_drop = StatefulDataLoader(self.dataset, sampler=sampler_without_drop)

        # Collect all indices from dataloaders
        indices_with_drop = [data for batch in dataloader_with_drop for data in batch]
        indices_without_drop = [data for batch in dataloader_without_drop for data in batch]

        # Check the lengths of the outputs
        self.assertEqual(
            len(indices_with_drop),
            expected_length_with_drop,
            "Length with drop_last=True should match expected truncated length",
        )
        self.assertEqual(
            len(indices_without_drop),
            expected_length_without_drop,
            "Length with drop_last=False should match total dataset size",
        )

        self.assertTrue(
            len(indices_with_drop) <= len(indices_without_drop),
            "Drop last should result in fewer or equal indices",
        )

    def test_data_order_with_shuffle(self):
        sampler = StatefulDistributedSampler(self.dataset, num_replicas=1, rank=0, shuffle=True)
        indices = list(iter(sampler))
        data_sampled = [self.dataset[i] for i in indices]
        self.assertNotEqual(data_sampled, list(range(100)), "Data should be shuffled")

        dataloader = StatefulDataLoader(self.dataset, sampler=sampler)
        data_loaded = []
        for batch in dataloader:
            data_loaded.extend(batch)
        self.assertEqual(len(data_loaded), len(self.dataset), "All data should be loaded")
        self.assertEqual(
            data_loaded,
            data_sampled,
            "Data loaded by DataLoader should match data sampled by sampler",
        )

    def test_data_order_without_shuffle(self):
        sampler = StatefulDistributedSampler(self.dataset, num_replicas=1, rank=0, shuffle=False)
        indices = list(iter(sampler))
        data_sampled = [self.dataset[i] for i in indices]
        self.assertEqual(data_sampled, list(range(100)), "Data should not be shuffled")

        batch_size = 32
        dataloader = StatefulDataLoader(self.dataset, batch_size=batch_size, sampler=sampler)
        data_loaded = []
        for batch in dataloader:
            data_loaded.extend(batch)
        self.assertEqual(len(data_loaded), len(self.dataset), "All data should be loaded")
        self.assertEqual(
            data_loaded,
            data_sampled,
            "Data loaded by DataLoader should match data sampled by sampler",
        )
        self.assertEqual(
            data_loaded,
            list(range(100)),
            "Data loaded by DataLoader should be in original order",
        )

    def test_data_distribution_across_replicas(self):
        num_replicas = 5
        all_data = []
        for rank in range(num_replicas):
            sampler = StatefulDistributedSampler(self.dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
            dataloader = StatefulDataLoader(self.dataset, sampler=sampler)
            data_loaded = []
            for batch in dataloader:
                data_loaded.extend([int(x.item()) for x in batch])
            all_data.extend(data_loaded)
        self.assertEqual(
            sorted(all_data),
            list(range(100)),
            "All data points should be covered exactly once across all replicas",
        )

    def test_seed_replicability(self):
        # Test that the same seed will result in the same data order
        # We first pick a random number as seed, then use it to initialize two dataloaders
        min_seed, max_seed = 0, 1000  # [min_seed, max_seed)
        seed = torch.randint(min_seed, max_seed, (1,), dtype=torch.int64).item()
        torch.manual_seed(seed)

        dataloader1 = StatefulDataLoader(self.dataset, batch_size=1, shuffle=True)
        results1 = list(dataloader1)

        # Repeat the same process with the same seed
        torch.manual_seed(seed)
        dataloader2 = StatefulDataLoader(self.dataset, batch_size=1, shuffle=True)
        results2 = list(dataloader2)

        # Repeat the same process with a different seed, making sure that the seed is different
        min_seed, max_seed = 1000, 2000  # [min_seed, max_seed)
        seed = torch.randint(min_seed, max_seed, (1,), dtype=torch.int64).item()
        torch.manual_seed(seed)
        dataloader3 = StatefulDataLoader(self.dataset, batch_size=1, shuffle=True)
        results3 = list(dataloader3)

        self.assertEqual(results1, results2, "Data should be replicable with same seed")
        self.assertNotEqual(results1, results3, "Data should not be replicable with different seed")

    def test_StatefulDistributedSampler_initialization_with_dataset_size(self):
        sampler = StatefulDistributedSampler(dataset_size=100, num_replicas=2, rank=0, shuffle=False)
        self.assertEqual(sampler.dataset_size, 100)
        indices = list(iter(sampler))
        expected_indices = list(range(0, 100, 2))
        self.assertEqual(indices, expected_indices)

    def test_StatefulDistributedSampler_mismatched_dataset_and_dataset_size(self):
        dataset = MockDataset(100)
        with self.assertRaises(ValueError):
            StatefulDistributedSampler(dataset=dataset, dataset_size=50)

    def test_StatefulDistributedSampler_no_dataset_or_dataset_size(self):
        with self.assertRaises(ValueError):
            StatefulDistributedSampler()

    def test_StatefulDistributedSampler_drop_last_with_dataset_size(self):
        dataset_size = 100
        num_replicas = 3
        sampler = StatefulDistributedSampler(
            dataset_size=dataset_size,
            num_replicas=num_replicas,
            rank=0,
            drop_last=True,
            shuffle=False,
        )
        self.assertEqual(sampler.num_samples, 33)
        indices = list(iter(sampler))
        self.assertEqual(len(indices), 33)
        expected_indices = list(range(0, 99, num_replicas))
        self.assertEqual(indices, expected_indices)

    def test_StatefulDistributedSampler_dataloader_state_dict_with_dataset_size(self):
        dataset_size = 100
        sampler = StatefulDistributedSampler(dataset_size=dataset_size, num_replicas=1, rank=0, shuffle=False)
        dataset = MockDataset(dataset_size)
        dataloader = StatefulDataLoader(dataset, batch_size=10, sampler=sampler)
        iter_count = 5
        for i, _ in enumerate(dataloader):
            if i == iter_count - 1:
                break
        state_dict = dataloader.state_dict()
        new_sampler = StatefulDistributedSampler(dataset_size=dataset_size, num_replicas=1, rank=0, shuffle=False)
        new_dataloader = StatefulDataLoader(MockDataset(dataset_size), batch_size=10, sampler=new_sampler)
        new_dataloader.load_state_dict(state_dict)
        resumed_data = []
        for data in new_dataloader:
            resumed_data.append(data.tolist())
        expected_data = []
        full_dataloader = StatefulDataLoader(MockDataset(dataset_size), batch_size=10, sampler=sampler)
        for data in full_dataloader:
            expected_data.append(data.tolist())
        self.assertEqual(resumed_data, expected_data[iter_count:])

    def test_StatefulDistributedSampler_dataset_size_zero(self):
        sampler = StatefulDistributedSampler(dataset_size=0, num_replicas=1, rank=0)
        self.assertEqual(len(sampler), 0)
        indices = list(iter(sampler))
        self.assertEqual(len(indices), 0)

    def test_StatefulDistributedSampler_shuffle_with_dataset_size(self):
        dataset_size = 100
        sampler = StatefulDistributedSampler(dataset_size=dataset_size, num_replicas=1, rank=0, shuffle=True, seed=42)
        indices = list(iter(sampler))
        self.assertEqual(len(indices), dataset_size)
        self.assertEqual(sorted(indices), list(range(dataset_size)))
        sampler.set_epoch(1)
        indices_epoch_1 = list(iter(sampler))
        self.assertNotEqual(indices, indices_epoch_1)


if __name__ == "__main__":
    run_tests()
