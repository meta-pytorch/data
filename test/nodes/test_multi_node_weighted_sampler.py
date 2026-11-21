# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import pickle

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.loader import Loader
from torchdata.nodes.prefetch import Prefetcher

from torchdata.nodes.samplers.multi_node_weighted_sampler import MultiNodeWeightedSampler
from torchdata.nodes.samplers.stop_criteria import StopCriteria

from .utils import DummyIterableDataset, run_test_save_load_state


class TestMultiNodeWeightedSampler(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._num_samples = 10
        self._num_datasets = 4
        self._weights_fn = lambda i: 0.1 * (i + 1)
        self._num_epochs = 3
        self.datasets = {
            f"ds{i}": IterableWrapper(DummyIterableDataset(self._num_samples, f"ds{i}"))
            for i in range(self._num_datasets)
        }
        self.weights = {f"ds{i}": self._weights_fn(i) for i in range(self._num_datasets)}
        self.equal_weights = {f"ds{i}": 1.0 for i in range(self._num_datasets)}

    def test_torchdata_nodes_imports(self) -> None:
        try:
            from torchdata.nodes import MultiNodeWeightedSampler, StopCriteria  # noqa
        except ImportError:
            self.fail("MultiNodeWeightedSampler or StopCriteria failed to import")

    @parameterized.expand(
        itertools.product((StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, StopCriteria.CYCLE_FOREVER), (True, False))
    )
    def test_cycle_with_empty_sources(self, stop_criteria: str, empty_first: bool) -> None:
        entries = [("empty", []), ("data", range(4))]
        if not empty_first:
            entries.reverse()
        sources = {key: IterableWrapper(values) for key, values in entries}
        node = MultiNodeWeightedSampler(
            sources, {key: 1.0 for key in sources}, stop_criteria=stop_criteria, tag_output=True
        )
        loader = Loader(node, restart_on_stop_iteration=False)
        expected = [{"dataset_key": "data", "data": value} for value in range(4)]
        if stop_criteria == StopCriteria.CYCLE_FOREVER:
            self.assertEqual(list(itertools.islice(loader, 8)), expected * 2)
        else:
            self.assertEqual(list(loader), expected)

    @parameterized.expand([StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, StopCriteria.CYCLE_FOREVER])
    def test_cycle_all_empty_sources(self, stop_criteria: str) -> None:
        sources = {"a": IterableWrapper([]), "b": IterableWrapper([])}
        node = MultiNodeWeightedSampler(sources, {"a": 1.0, "b": 1.0}, stop_criteria=stop_criteria)
        self.assertEqual(list(node), [])

    @parameterized.expand([StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, StopCriteria.CYCLE_FOREVER])
    def test_resume_cycle_with_empty_sources(self, stop_criteria: str) -> None:
        def make_loader():
            sources = {"data": IterableWrapper(range(4)), "empty": IterableWrapper([])}
            node = MultiNodeWeightedSampler(sources, {"data": 1.0, "empty": 1.0}, stop_criteria=stop_criteria)
            return Loader(node, restart_on_stop_iteration=False)

        loader = make_loader()
        iterator = iter(loader)
        self.assertEqual(next(iterator), 0)
        self.assertEqual(next(iterator), 1)
        state = loader.state_dict()
        restored = make_loader()
        restored.load_state_dict(state)
        expected = [2, 3]
        if stop_criteria == StopCriteria.CYCLE_FOREVER:
            self.assertEqual(list(itertools.islice(restored, 6)), expected + list(range(4)))
        else:
            self.assertEqual(list(restored), expected)

    @parameterized.expand([StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, StopCriteria.CYCLE_FOREVER])
    def test_cycle_with_high_weight_empty_source(self, stop_criteria: str) -> None:
        sources = {"empty": IterableWrapper([]), "data": IterableWrapper(range(4))}
        node = MultiNodeWeightedSampler(sources, {"empty": 1e20, "data": 1.0}, stop_criteria=stop_criteria)
        expected = list(range(4))
        if stop_criteria == StopCriteria.CYCLE_FOREVER:
            self.assertEqual(list(itertools.islice(node, 8)), expected * 2)
        else:
            self.assertEqual(list(node), expected)

    def test_cycle_forever_with_high_weight_all_empty_sources(self) -> None:
        sources = {"a": IterableWrapper([]), "b": IterableWrapper([])}
        node = MultiNodeWeightedSampler(sources, {"a": 1e20, "b": 1.0}, stop_criteria=StopCriteria.CYCLE_FOREVER)
        self.assertEqual(list(node), [])

    @parameterized.expand([StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, StopCriteria.CYCLE_FOREVER])
    def test_resume_cycle_with_multiple_nonempty_sources(self, stop_criteria: str) -> None:
        def make_loader():
            sources = {
                "empty": IterableWrapper([]),
                "a": IterableWrapper(range(3)),
                "b": IterableWrapper(range(5)),
            }
            node = MultiNodeWeightedSampler(
                sources, {"empty": 1e20, "a": 1.0, "b": 3.0}, stop_criteria=stop_criteria, tag_output=True
            )
            return Loader(node, restart_on_stop_iteration=False)

        expected = list(itertools.islice(make_loader(), 20))
        if stop_criteria == StopCriteria.CYCLE_FOREVER:
            self.assertEqual(len(expected), 20)
        else:
            self.assertGreaterEqual(len(expected), 8)
            self.assertEqual(
                {(item["dataset_key"], item["data"]) for item in expected},
                {("a", value) for value in range(3)} | {("b", value) for value in range(5)},
            )
        loader = make_loader()
        prefix = list(itertools.islice(loader, 3))
        checkpoint = pickle.dumps(loader.state_dict())
        self.assertEqual(prefix, expected[:3])
        for _ in range(2):
            restored = make_loader()
            restored.load_state_dict(pickle.loads(checkpoint))
            self.assertEqual(list(itertools.islice(restored, 17)), expected[3:])

    def _setup_multi_node_weighted_sampler(
        self, num_samples, num_datasets, weights_fn, stop_criteria, seed=0
    ) -> Prefetcher:

        datasets = {f"ds{i}": IterableWrapper(DummyIterableDataset(num_samples, f"ds{i}")) for i in range(num_datasets)}
        weights = {f"ds{i}": weights_fn(i) for i in range(num_datasets)}
        node = MultiNodeWeightedSampler(datasets, weights, stop_criteria, seed=seed)
        return Prefetcher(node, prefetch_factor=3)

    def test_multi_node_weighted_sampler_weight_sampler_keys_mismatch(self) -> None:
        """Test validation logic for MultiNodeWeightedSampler if the keys of source_nodes and weights are not the same"""
        with self.assertRaisesRegex(
            ValueError,
            "keys of source_nodes and weights must be the same",
        ):
            MultiNodeWeightedSampler(
                self.datasets,
                {f"dummy{i}": self._weights_fn(i) for i in range(self._num_datasets)},
            )

    def test_multi_node_weighted_batch_sampler_invalid_weights_tensor_shape(
        self,
    ) -> None:
        """Test validation logic for MultiNodeWeightedSampler if the shape of the weights tensor is invalid"""
        with self.assertRaisesRegex(ValueError, " weights must be a 1d sequence, non-negative, and non-zero"):
            MultiNodeWeightedSampler(
                self.datasets,
                weights={f"ds{i}": [[1.0]] for i in range(self._num_datasets)},
            )

    def test_multi_node_weighted_batch_sampler_negative_weights(
        self,
    ) -> None:
        """Test validation logic for MultiNodeWeightedSampler if the value of the weights tensor is invalid"""
        with self.assertRaisesRegex(ValueError, " weights must be a 1d sequence, non-negative, and non-zero"):
            MultiNodeWeightedSampler(
                self.datasets,
                weights={f"ds{i}": -1 for i in range(self._num_datasets)},
            )

    def test_multi_node_weighted_batch_sampler_zero_weights(
        self,
    ) -> None:
        """Test validation logic for MultiNodeWeightedSampler if the value of the weights tensor is invalid"""
        with self.assertRaisesRegex(ValueError, " weights must be a 1d sequence, non-negative, and non-zero"):
            MultiNodeWeightedSampler(
                self.datasets,
                weights={f"ds{i}": 10 * i for i in range(self._num_datasets)},
            )

    @parameterized.expand(range(10))
    def test_multi_node_weighted_sampler_first_exhausted(self, seed) -> None:
        """Test MultiNodeWeightedSampler with stop criteria FIRST_DATASET_EXHAUSTED"""
        node = self._setup_multi_node_weighted_sampler(
            self._num_samples,
            self._num_datasets,
            self._weights_fn,
            stop_criteria=StopCriteria.FIRST_DATASET_EXHAUSTED,
            seed=seed,
        )

        for _ in range(self._num_epochs):
            results = list(node)

            datasets_in_results = [result["name"] for result in results]
            dataset_counts_in_results = [datasets_in_results.count(f"ds{i}") for i in range(self._num_datasets)]

            # Check max item count for dataset is exactly _num_samples
            self.assertEqual(max(dataset_counts_in_results), self._num_samples)

            # Check that max items are taken from at least one dataset
            self.assertGreaterEqual(dataset_counts_in_results.count(self._num_samples), 1)
            node.reset()
        node.shutdown()

    @parameterized.expand(range(10))
    def test_multi_node_weighted_sampler_all_dataset_exhausted(self, seed) -> None:
        """Test MultiNodeWeightedSampler with stop criteria ALL_DATASETS_EXHAUSTED"""
        node = self._setup_multi_node_weighted_sampler(
            self._num_samples,
            self._num_datasets,
            self._weights_fn,
            stop_criteria=StopCriteria.ALL_DATASETS_EXHAUSTED,
            seed=seed,
        )

        for _ in range(self._num_epochs):
            results = list(node)
            datasets_in_results = [result["name"] for result in results]
            dataset_counts_in_results = [datasets_in_results.count(f"ds{i}") for i in range(self._num_datasets)]

            # check each dataset appears exactly _num_samples times,
            # each dataset has _num_samples samples
            self.assertEqual(
                dataset_counts_in_results,
                [self._num_samples] * self._num_datasets,
            )

            # check that all datasets are exhausted
            self.assertEqual(sorted(set(datasets_in_results)), ["ds0", "ds1", "ds2", "ds3"])
            node.reset()
        node.shutdown()

    @parameterized.expand(range(10))
    def test_multi_node_weighted_sampler_cycle_until_all_exhausted(self, seed) -> None:
        """Test MultiNodeWeightedSampler with stop criteria CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED"""
        node = self._setup_multi_node_weighted_sampler(
            self._num_samples,
            self._num_datasets,
            self._weights_fn,
            stop_criteria=StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            seed=seed,
        )

        for _ in range(self._num_epochs):
            results = list(node)
            datasets_in_results = {result["name"] for result in results}

            # check that all datasets are exhausted
            self.assertEqual(sorted(datasets_in_results), ["ds0", "ds1", "ds2", "ds3"])
            node.reset()
        node.shutdown()

    @parameterized.expand(range(10))
    def test_multi_node_weighted_sampler_cycle_forever(self, seed) -> None:
        """Test MultiNodeWeightedSampler with stop criteria CYCLE_FOREVER"""
        node = MultiNodeWeightedSampler(
            self.datasets, self.equal_weights, stop_criteria=StopCriteria.CYCLE_FOREVER, seed=seed
        )

        num_yielded = 0
        _it = iter(node)
        while num_yielded < 256:  # total number of samples is 4 * 10 = 40, 256 is an arbitrary larger number
            next(_it)
            num_yielded += 1

        node_num_yielded = node.get_state()[MultiNodeWeightedSampler.NUM_YIELDED_KEY]
        self.assertEqual(node_num_yielded, num_yielded)

    @parameterized.expand([(1, 8), (8, 32)])
    def test_multi_node_weighted_batch_sampler_set_rank_world_size(self, rank, world_size):
        """Test MultiNodeWeightedSampler with different rank and world size"""
        node = MultiNodeWeightedSampler(
            self.datasets,
            self.weights,
            rank=rank,
            world_size=world_size,
        )
        self.assertEqual(node.rank, rank)
        self.assertEqual(node.world_size, world_size)

    def test_multi_node_weighted_batch_sampler_results_for_ranks(self):
        """Test MultiNodeWeightedSampler with different results for different ranks"""
        world_size = 8
        global_results = []
        for rank in range(world_size):
            node = MultiNodeWeightedSampler(
                self.datasets,
                self.weights,
                rank=rank,
                world_size=world_size,
            )
            results = list(node)
            global_results.append(results)

        unique_results = []
        for results in global_results:
            if results not in unique_results:
                unique_results.append(results)
        self.assertEqual(unique_results, global_results)

    def test_multi_node_weighted_batch_sampler_results_for_multiple_epochs(self):
        """Test MultiNodeWeightedSampler with different results in each epoch"""

        # Check for the MultiNodeWeightedSampler node only
        node = MultiNodeWeightedSampler(
            self.datasets,
            self.weights,
        )

        overall_results = []
        for _ in range(self._num_epochs):
            results = list(node)
            overall_results.append(results)
            node.reset()
        node.shutdown()

        unique_results = []
        for results in overall_results:
            if results not in unique_results:
                unique_results.append(results)

        self.assertEqual(unique_results, overall_results)

        # Check for MultiNodeWeightedSampler node along with Prefetcher node
        node = self._setup_multi_node_weighted_sampler(
            self._num_samples,
            self._num_datasets,
            self._weights_fn,
            stop_criteria=StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
        )

        overall_results = []
        for _ in range(self._num_epochs):
            results = list(node)
            overall_results.append(results)
            node.reset()
        node.shutdown()

        unique_results = []
        for results in overall_results:
            if results not in unique_results:
                unique_results.append(results)

        self.assertEqual(unique_results, overall_results)

    @parameterized.expand(
        itertools.product(
            [1, 4, 7],
            [
                StopCriteria.ALL_DATASETS_EXHAUSTED,
                StopCriteria.FIRST_DATASET_EXHAUSTED,
                StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            ],
        )
    )
    def test_save_load_state_mds_node_over_multiple_epochs(self, midpoint: int, stop_criteria: str):
        """Test MultiNodeWeightedSampler with saving and loading of state across multiple epochs"""
        node = MultiNodeWeightedSampler(
            self.datasets,
            self.weights,
            stop_criteria,
        )
        run_test_save_load_state(self, node, midpoint)

    @parameterized.expand(
        itertools.product(
            [1, 4, 7],
            [
                StopCriteria.ALL_DATASETS_EXHAUSTED,
                StopCriteria.FIRST_DATASET_EXHAUSTED,
                StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            ],
        )
    )
    def test_save_load_state_mds_node_over_multiple_epochs_with_prefetcher(self, midpoint: int, stop_criteria: str):
        node = self._setup_multi_node_weighted_sampler(
            self._num_samples,
            self._num_datasets,
            self._weights_fn,
            stop_criteria=stop_criteria,
        )
        run_test_save_load_state(self, node, midpoint)

    @parameterized.expand(
        itertools.product(
            [100, 500, 1200],
            [
                StopCriteria.ALL_DATASETS_EXHAUSTED,
                StopCriteria.FIRST_DATASET_EXHAUSTED,
                StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            ],
        )
    )
    def test_multi_node_weighted_large_sample_size_with_prefetcher(self, midpoint, stop_criteria) -> None:
        """Test MultiNodeWeightedSampler (larger sample sizes) with saving and loading of state across multiple epochs"""
        num_samples = 1500
        num_datasets = 5

        node = self._setup_multi_node_weighted_sampler(
            num_samples,
            num_datasets,
            self._weights_fn,
            stop_criteria,
        )
        run_test_save_load_state(self, node, midpoint)

    def test_multi_node_weighted_sampler_tag_output_dict_items(self) -> None:
        """Test MultiNodeWeightedSampler with tag_output=True for dictionary items"""
        node = MultiNodeWeightedSampler(
            self.datasets,
            self.weights,
            tag_output=True,
        )

        results = list(node)

        # Verify that each result has a 'dataset_key' key with the correct dataset name
        for result in results:
            self.assertIn("dataset_key", result)

            dataset_name = result["dataset_key"]
            self.assertIn(dataset_name, [f"ds{i}" for i in range(self._num_datasets)])

            self.assertIn("name", result)
            self.assertIn("test_tensor", result)

            self.assertEqual(dataset_name, result["name"])

    def test_multi_node_weighted_sampler_tag_output_non_dict_items(self) -> None:
        """Test MultiNodeWeightedSampler with tag_output=True for non-dictionary items"""
        non_dict_datasets = {f"ds{i}": IterableWrapper(range(i * 10, (i + 1) * 10)) for i in range(self._num_datasets)}

        node = MultiNodeWeightedSampler(
            non_dict_datasets,
            self.weights,
            tag_output=True,
        )

        results = list(node)

        # Verify that each result is now a dictionary with 'data' and 'dataset_key' keys
        for result in results:
            self.assertIsInstance(result, dict)

            self.assertIn("data", result)
            self.assertIn("dataset_key", result)

            dataset_name = result["dataset_key"]
            self.assertIn(dataset_name, [f"ds{i}" for i in range(self._num_datasets)])

    def test_multi_node_weighted_sampler_tag_output_false(self) -> None:
        """Test MultiNodeWeightedSampler with tag_output=False (default behavior)"""
        node = MultiNodeWeightedSampler(
            self.datasets,
            self.weights,
            tag_output=False,
        )

        results = list(node)

        # Verify that none of the results have a 'dataset' key
        for result in results:
            self.assertNotIn("dataset", result)

            # Check that the original data is preserved
            self.assertIn("name", result)
            self.assertIn("test_tensor", result)
