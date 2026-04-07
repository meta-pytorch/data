# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, TypeVar

from torchdata.nodes import BaseNode


T = TypeVar("T")


class Filter(BaseNode[T]):
    """Node that filters items from source node based on predicate function.

    This node applies a filter function to each item from the source node and only yields
    items that satisfy the condition (when filter_fn returns True). It keeps track of both
    the number of items that were filtered out (rejected) and the number of items that were
    yielded (accepted).

    Args:
        source_node (BaseNode[T]): The source node to filter items from.
        filter_fn (Callable[[T], bool]): A function that takes an item and returns True if the item
            should be included, False otherwise.
    """

    SOURCE_KEY = "source"
    NUM_FILTERED_KEY = "num_filtered"
    NUM_YIELDED_KEY = "num_yielded"

    def __init__(self, source_node: BaseNode[T], filter_fn: Callable[[T], bool]):
        super().__init__()
        self.source = source_node
        self.filter_fn = filter_fn
        self._num_filtered = 0  # Count of items that did NOT pass the filter
        self._num_yielded = 0  # Count of items that DID pass the filter

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset the node to its initial state or to the provided state.

        Args:
            initial_state: Optional state dictionary to restore from.
        """
        super().reset(initial_state)
        if initial_state is not None:
            self.source.reset(initial_state.get(self.SOURCE_KEY))
            self._num_filtered = initial_state.get(self.NUM_FILTERED_KEY, 0)
            self._num_yielded = initial_state.get(self.NUM_YIELDED_KEY, 0)
        else:
            self.source.reset(None)
            self._num_filtered = 0
            self._num_yielded = 0

    def next(self) -> T:
        """Get the next item that passes the filter.

        Returns:
            The next item that satisfies the filter condition.

        Raises:
            StopIteration: If there are no more items in the source node.
        """
        while True:
            item = next(self.source)
            if self.filter_fn(item):
                self._num_yielded += 1
                return item
            self._num_filtered += 1

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the node.

        Returns:
            A dictionary containing the state of the source node and counters.
        """
        return {
            self.SOURCE_KEY: self.source.state_dict(),
            self.NUM_FILTERED_KEY: self._num_filtered,
            self.NUM_YIELDED_KEY: self._num_yielded,
        }
