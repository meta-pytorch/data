# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
from typing import Any, Dict, List, Mapping, Optional, Union

from torchdata.nodes.base_node import BaseNode, T
from torchdata.nodes.samplers.stop_criteria import StopCriteria

logger = logging.getLogger(__name__)


class MultiNodeRoundRobinSampler(BaseNode[Union[T, Dict[str, Any]]]):
    """A node that samples from multiple datasets in a round robin fashion.
    This node expects to take in a list or dictionary of source nodes. If a list is provided, it assumed that the order of the source nodes will be the same when the sampler is reset.
    The node implements the state using the following keys:
    - CURRENT_DATASET_INDEX_KEY: The index of the current dataset.
    - DATASET_NODE_STATES_KEY: A dictionary of states for each source node.
    - DATASETS_EXHAUSTED_KEY: A dictionary of booleans indicating whether each source node is exhausted.

    We support multiple stopping criteria:
    - CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED: Cycle through the source nodes until all datasets
        are exhausted. This is the default behavior.
    - FIRST_DATASET_EXHAUSTED: Stop when the first dataset is exhausted.
    - ALL_DATASETS_EXHAUSTED: Stop when all datasets are exhausted.
    Exhaustion Handling:
        A dataset is only marked exhausted when it raises StopIteration.
        Example: With 4 nodes (1st has 1 item, others have 5):
        - 1st iteration: Yields 1st node's item (not yet exhausted)
        - Subsequent iterations: Yields from other nodes
        - When returning to 1st node, StopIteration marks it exhausted
        - Behavior depends on criteria: FIRST_DATASET_EXHAUSTED stops immediately, others continue

    Args:
        source_nodes (Mapping[str, BaseNode[T]]): A dictionary of source nodes.
        stop_criteria (str): The stopping criteria. Default is CYCLE_UNTIL_ALL_DATASETS_EXHAUST.
        tag_output (bool): Whether to tag the output with the dataset name. Default is False.

    Example:
        >>> # Dataset A: 1 element, Dataset B: 2 elements
        >>> sampler = MultiNodeRoundRobinSampler(
        ...     source_nodes={"A": A_node, "B": B_node},
        ...     stop_criteria=StopCriteria.FIRST_DATASET_EXHAUSTED
        ...     tag_output=True
        ... )
        >>> list(sampler)  # Yields: A, B, then A is exhausted
        ['dataset_key': 'ds0', 'data': 'A_item'}, {'dataset_key': 'ds1', 'data': 'B_item1'}]
        If using StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED and tag_output=False:
        >>> list(sampler)  # Yields: A, B,  A (exhausted), B , A, then B is exhausted
        [A_item, B_item1, A_item, B_item2, A_item ]
    """

    CURRENT_DATASET_INDEX_KEY = "current_dataset_index"
    DATASET_KEYS = "dataset_keys"
    DATASET_NODE_STATES_KEY = "dataset_node_states"
    DATASETS_EXHAUSTED_KEY = "datasets_exhausted"

    def __init__(
        self,
        source_nodes: Union[Mapping[str, BaseNode[T]], List[BaseNode[T]]],
        stop_criteria: str = StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
        tag_output: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(source_nodes, list):
            logger.warning(
                "source_nodes are provided as a list. Thus, if resetting the sampler with an initial_state, please make sure that the order of the source_nodes is same as in the previous state."
            )
            source_nodes = {f"ds_{i}": node for i, node in enumerate(source_nodes)}

        self.dataset_keys = list(source_nodes.keys())
        self.source_nodes = [source_nodes[k] for k in self.dataset_keys]
        self.num_datasets = len(self.source_nodes)
        self.stop_criteria = stop_criteria
        self._current_dataset_index = 0
        self._validate_stop_criteria()
        self._datasets_exhausted = [False for _ in range(self.num_datasets)]
        if not isinstance(tag_output, bool):
            raise TypeError(f"tag_output must be a boolean (True/False), got {type(tag_output)}")
        self.output_keys = ("dataset_key", "data") if tag_output else None

    def _validate_stop_criteria(self) -> None:
        if self.stop_criteria not in [
            StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            StopCriteria.ALL_DATASETS_EXHAUSTED,
            StopCriteria.FIRST_DATASET_EXHAUSTED,
        ]:
            raise ValueError(
                f"Invalid {self.stop_criteria=}. stop_criteria must be one of: CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, ALL_DATASETS_EXHAUSTED, FIRST_DATASET_EXHAUSTED"
            )

    def _validate_reset_dataset_keys(self, reset_keys) -> None:
        if self.dataset_keys != reset_keys:
            raise ValueError(
                f"Invalid {self.dataset_keys=}. There is a mismatch between the dataset keys in the state and the current dataset keys. \n {self.dataset_keys=} \n {reset_keys=}"
            )

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self._validate_reset_dataset_keys(initial_state[self.DATASET_KEYS])
            self._datasets_exhausted = initial_state[self.DATASETS_EXHAUSTED_KEY]
            for k in range(self.num_datasets):
                self.source_nodes[k].reset(initial_state[self.DATASET_NODE_STATES_KEY][k])
            self._current_dataset_index = initial_state[self.CURRENT_DATASET_INDEX_KEY]
        else:
            # Force a fresh iterator from all source nodes
            self._datasets_exhausted = [False for _ in range(self.num_datasets)]
            self._current_dataset_index = 0
            for k in range(self.num_datasets):
                self.source_nodes[k].reset()

    def _check_for_stop_iteration(self) -> None:
        if all(self._datasets_exhausted):
            # Raise StopIteration if all datasets are exhausted,
            # this covers for both ALL_DATASETS_EXHAUSTED and CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED
            raise StopIteration()
        # Raise StopIteration is StopCriteria is FIRST_DATASET_EXHAUSTED and
        # the first dataset is exhausted. Doing this to correctly catch StopIteration
        # when trying next(it) on already exhausted iterator
        if self.stop_criteria == StopCriteria.FIRST_DATASET_EXHAUSTED and any(self._datasets_exhausted):
            raise StopIteration()
        return

    def next(self) -> Union[T, Dict[str, Any]]:
        while True:
            self._check_for_stop_iteration()
            current_iterator = self.source_nodes[self._current_dataset_index]
            try:
                # Before fetching a new item check if the current dataset is already
                # exhaused and if StopCriteria is ALL_DATASETS_EXHAUSTED, move to next dataset
                if (
                    self._datasets_exhausted[self._current_dataset_index]
                    and self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED
                ):
                    self._current_dataset_index = (self._current_dataset_index + 1) % self.num_datasets
                    continue
                item = next(current_iterator)
            except StopIteration:
                # Mark the dataset as exhausted
                self._datasets_exhausted[self._current_dataset_index] = True
                # Based on updated _check_for_stop_iteration, check if we should raise StopIteration
                self._check_for_stop_iteration()
                # If StopCriteria is ALL_DATASETS_EXHAUSTED, move to next dataset
                if self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED:
                    continue
                # If StopCriteria is CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
                # reset the iterator and try again
                self.source_nodes[self._current_dataset_index].reset()
                try:
                    item = next(self.source_nodes[self._current_dataset_index])
                except StopIteration:
                    # An empty source cannot be restarted. Continue with the other datasets.
                    self._current_dataset_index = (self._current_dataset_index + 1) % self.num_datasets
                    continue
            break
        # Capture dataset information before incrementing index
        dataset_idx = self._current_dataset_index
        dataset_name = self.dataset_keys[dataset_idx]
        self._current_dataset_index = (dataset_idx + 1) % self.num_datasets
        # Wrap item in dictionary if tagging is enabled
        if self.output_keys is not None:
            return {
                self.output_keys[0]: dataset_name,
                self.output_keys[1]: item,
            }  # Type: ignore[return-value]
        return item

    def get_state(self) -> Dict[str, Any]:
        state = {
            self.CURRENT_DATASET_INDEX_KEY: self._current_dataset_index,
            self.DATASET_KEYS: copy.deepcopy(self.dataset_keys),
            self.DATASET_NODE_STATES_KEY: {k: self.source_nodes[k].state_dict() for k in range(self.num_datasets)},
            self.DATASETS_EXHAUSTED_KEY: copy.deepcopy(self._datasets_exhausted),
        }
        return state
