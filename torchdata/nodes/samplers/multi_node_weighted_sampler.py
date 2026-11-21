# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, Mapping, Optional

import torch
from torchdata.nodes.base_node import BaseNode, T
from torchdata.nodes.samplers.stop_criteria import StopCriteria

from .utils import _get_rank_seed, get_rank_and_world_size


class MultiNodeWeightedSampler(BaseNode[T]):
    """A node that samples from multiple datasets with weights.

    This node expects to take in a dictionary of source nodes, and a dictionary of weights.
    The keys of the source nodes and weights must be the same. The weights are used to sample
    from the source nodes. We use torch.multinomial to sample from the source nodes, please
    refer to https://pytorch.org/docs/stable/generated/torch.multinomial.html on how to use
    weights for sampling. `seed` is used to initialize the random number generator.

    The node implements the state using the following keys:

    - DATASET_NODE_STATES_KEY: A dictionary of states for each source node.
    - DATASETS_EXHAUSTED_KEY: A dictionary of booleans indicating whether each source node is exhausted.
    - EPOCH_KEY: An epoch counter used to initialize the random number generator.
    - NUM_YIELDED_KEY: The number of items yielded.
    - WEIGHTED_SAMPLER_STATE_KEY: The state of the weighted sampler.

    We support multiple stopping criteria:

    - CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED: Cycle through the source nodes until all datasets are exhausted. This is the default behavior.
    - FIRST_DATASET_EXHAUSTED: Stop when the first dataset is exhausted.
    - ALL_DATASETS_EXHAUSTED: Stop when all datasets are exhausted.

    On complete exhaustion of the source nodes, the node will raise StopIteration.

    Args:
        source_nodes (Mapping[str, BaseNode[T]]): A dictionary of source nodes.
        weights (Dict[str, float]): A dictionary of weights for each source node.
        stop_criteria (str): The stopping criteria. Default is CYCLE_UNTIL_ALL_DATASETS_EXHAUST
        rank (int): The rank of the current process. Default is None, in which case the rank
            will be obtained from the distributed environment.
        world_size (int): The world size of the distributed environment. Default is None, in
            which case the world size will be obtained from the distributed environment.
        seed (int): The seed for the random number generator. Default is 0.
        tag_output (bool): Whether to tag the output with the dataset name. Default is False.
    """

    DATASET_NODE_STATES_KEY = "dataset_node_states"
    DATASETS_EXHAUSTED_KEY = "datasets_exhausted"
    EPOCH_KEY = "epoch"
    NUM_YIELDED_KEY = "num_yielded"
    WEIGHTED_SAMPLER_STATE_KEY = "weighted_sampler_state"

    def __init__(
        self,
        source_nodes: Mapping[str, BaseNode[T]],
        weights: Dict[str, float],
        stop_criteria: str = StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        seed: int = 0,
        tag_output: bool = False,
    ) -> None:
        super().__init__()

        self.source_nodes = source_nodes
        self.weights = weights
        self.stop_criteria = stop_criteria
        self.dataset_names = list(self.source_nodes.keys())
        self._num_yielded = 0
        self._started = False
        self.seed = seed
        self.tag_output = tag_output

        # Setup rank and world size
        if rank is None or world_size is None:
            self.rank, self.world_size = get_rank_and_world_size()
        else:
            self.rank = rank
            self.world_size = world_size

        self._epoch = 0

        self._validate()

    def _validate(self) -> None:
        if self.stop_criteria not in [
            StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            StopCriteria.ALL_DATASETS_EXHAUSTED,
            StopCriteria.FIRST_DATASET_EXHAUSTED,
            StopCriteria.CYCLE_FOREVER,
        ]:
            raise ValueError(
                f"Invalid {self.stop_criteria=}. stop_criteria must be one of: CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, FIRST_DATASET_EXHAUSTED, ALL_DATASETS_EXHAUSTED"
            )

        # Validate if keys of source_nodes and weights are the same
        if set(self.dataset_names) != set(self.weights.keys()) or len(self.dataset_names) != len(self.weights):
            raise ValueError(
                f"Invalid {self.weights=}. For multi-dataset weighted sampling, keys of source_nodes and weights must be the same",
            )

        for weight in self.weights.values():
            if not isinstance(weight, float) or weight <= 0:
                raise ValueError(
                    f"""Invalid {self.weights=}. For multi-dataset weighted sampling, weights must be a 1d sequence, non-negative, and non-zero.
                    Weights are used to sample from source nodes. Zero weight means the source node will never be sampled from, and can cause
                    unexpected behavior depending on the stop criteris. Weights are used as inputs to torch.multinomial, please refer to
                    https://pytorch.org/docs/stable/generated/torch.multinomial.html on how to use weights for sampling."""
                )

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
            self._epoch = initial_state[self.EPOCH_KEY]
            self._weighted_sampler = self._get_new_weighted_sampler(initial_state)
            self._datasets_exhausted = initial_state[self.DATASETS_EXHAUSTED_KEY]
            for k in self.dataset_names:
                self.source_nodes[k].reset(initial_state[self.DATASET_NODE_STATES_KEY][k])
        else:
            # Force a fresh iterator from all source nodes
            self._num_yielded = 0

            if self._started:
                self._epoch += 1
            self._weighted_sampler = self._get_new_weighted_sampler()

            self._datasets_exhausted = {key: False for key in self.weights.keys()}
            for k in self.dataset_names:
                self.source_nodes[k].reset()
        self._started = False

    def _get_new_weighted_sampler(self, initial_state=None):
        return _WeightedSampler(
            weights=self.weights,
            seed=self.seed,
            rank=self.rank,
            world_size=self.world_size,
            epoch=self._epoch,
            initial_state=(initial_state[self.WEIGHTED_SAMPLER_STATE_KEY] if initial_state is not None else None),
        )

    def _check_for_stop_iteration(self) -> None:
        if self.stop_criteria == StopCriteria.CYCLE_FOREVER:
            # If StopCriteria is CYCLE_FOREVER, we should never raise StopIteration
            return

        if all(self._datasets_exhausted.values()):
            # Raise StopIteration if all datasets are exhausted,
            # this covers for both ALL_DATASETS_EXHAUSTED and CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED
            raise StopIteration()

        # Raise StopIteration is StopCriteria is FIRST_DATASET_EXHAUSTED and
        # the first dataset is exhausted. Doing this to correctly catch StopIteration
        # when trying next(it) on already exhausted iterator
        if self.stop_criteria == StopCriteria.FIRST_DATASET_EXHAUSTED and any(self._datasets_exhausted.values()):
            raise StopIteration()

        return

    def next(self) -> T:
        self._started = True
        empty_sources: Optional[set[str]] = None
        while True:
            self._check_for_stop_iteration()

            # Fetch the next item's key from the weighted sampler
            if empty_sources:
                key = self._weighted_sampler.sample_without(empty_sources)
            else:
                key = next(self._weighted_sampler)
            try:
                if self._datasets_exhausted[key] and self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED:
                    # Before fetching a new item check if key corresponds to an already
                    # exhaused dataset and StopCriteria is ALL_DATASETS_EXHAUSTED, move to next key
                    continue
                item = next(self.source_nodes[key])
            except StopIteration:
                # Mark the dataset as exhausted
                self._datasets_exhausted[key] = True

                # Based on updated _datasets_exhausted, use _check_for_stop_iteration to check if we should raise StopIteration
                self._check_for_stop_iteration()

                # If StopCriteria is ALL_DATASETS_EXHAUSTED, move to next key
                if self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED:
                    continue

                # If StopCriteria is CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED or CYCLE_FOREVER,
                # reset the iterator and try again
                self.source_nodes[key].reset()
                try:
                    item = next(self.source_nodes[key])
                except StopIteration:
                    if empty_sources is None:
                        empty_sources = set()
                    empty_sources.add(key)
                    continue
            break

        # If we did't throw StopIteration, increment the number of items yielded and return the item
        self._num_yielded += 1

        # If tag_output is True, add the dataset key to the output
        if self.tag_output:
            if isinstance(item, dict):  # type: ignore[used-before-def]
                item["dataset_key"] = key  # type: ignore[used-before-def]
            else:
                item = {"dataset_key": key, "data": item}

        return item

    def get_state(self) -> Dict[str, Any]:
        return {
            self.DATASETS_EXHAUSTED_KEY: copy.deepcopy(self._datasets_exhausted),
            self.DATASET_NODE_STATES_KEY: {k: self.source_nodes[k].state_dict() for k in self.dataset_names},
            self.EPOCH_KEY: self._epoch,
            self.NUM_YIELDED_KEY: self._num_yielded,
            self.WEIGHTED_SAMPLER_STATE_KEY: self._weighted_sampler.state_dict(),
        }


class _WeightedSampler:
    """A weighted sampler that samples from a list of weights.

    The class implements the state using the following keys:

    - g_state: The state of the random number generator.
    - g_rank_state: The state of the random number generator for the rank.
    - offset: The offset of the batch of indices.
    - resample_g_state: Optional RNG state for sampling after an empty source is encountered.

    Args:
        weights (Dict[str, float]): A dictionary of weights for each source node.
        seed (int): The seed for the random number generator.
        rank (int): The rank of the current process.
        world_size (int): The world size of the distributed environment.
        random_tensor_batch_size (int): Generating random numbers in batches is faster than individually.
            This setting controls the batch size, but is invisible to users and shouldn't need to be tuned. Default is 1000.
        initial_state (Optional[Dict[str, Any]]): The initial state of the sampler. Default is None.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        seed: int,
        rank: int,
        world_size: int,
        epoch: int,
        random_tensor_batch_size: int = 1000,
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        _names, _weights = [], []
        for name, weight in weights.items():
            _names.append(name)
            _weights.append(weight)

        self.names = _names
        self.weights = torch.tensor(_weights, dtype=torch.float64)

        self.random_tensor_batch_size = random_tensor_batch_size

        self._g = torch.Generator()
        self._g_rank = torch.Generator()

        self.epoch = epoch
        seed = _get_rank_seed(seed, self._g_rank, rank, world_size, self.epoch)
        self._g.manual_seed(seed)
        self._resample_seed = (seed + 1) % (1 << 64)
        self._resample_generator: Optional[torch.Generator] = None

        self._g_snapshot = self._g.get_state()
        if initial_state is not None:
            self._g.set_state(initial_state["g_state"])
            self._offset = initial_state["offset"]
        else:
            self._offset = 0

        self._batch_of_indices = self._get_batch_of_indices()
        if initial_state is not None and "resample_g_state" in initial_state:
            self._resample_generator = torch.Generator()
            self._resample_generator.set_state(initial_state["resample_g_state"])

    def sample_without(self, excluded: set[str]) -> str:
        # Rejection sampling may never find a low-weight nonempty source. Draw directly
        # from the remaining weights, using a separate checkpointed RNG so the normal
        # batched sampling sequence remains unchanged when no empty source is encountered.
        indices = [i for i, name in enumerate(self.names) if name not in excluded]
        if not indices:
            raise StopIteration
        if self._resample_generator is None:
            self._resample_generator = torch.Generator().manual_seed(self._resample_seed)
        index = int(
            torch.multinomial(
                self.weights[indices], num_samples=1, replacement=True, generator=self._resample_generator
            ).item()
        )
        return self.names[indices[index]]

    def _get_batch_of_indices(self) -> list[int]:
        self._g_snapshot = self._g.get_state()
        return torch.multinomial(
            self.weights,
            num_samples=self.random_tensor_batch_size,
            replacement=True,
            generator=self._g,
        ).tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if self._offset >= len(self._batch_of_indices):
            self._batch_of_indices = self._get_batch_of_indices()
            self._offset = 0
        item = self._batch_of_indices[self._offset]
        self._offset += 1
        return self.names[item]

    def state_dict(self):
        state = {
            "g_state": self._g_snapshot,
            "offset": self._offset,
        }
        if self._resample_generator is not None:
            state["resample_g_state"] = self._resample_generator.get_state()
        return state
