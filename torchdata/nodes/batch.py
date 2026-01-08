# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Sequence

from torchdata.nodes.base_node import BaseNode, T


class Batcher(BaseNode[List[T]]):
    """Batcher node batches the data from the source node into batches of size batch_size.
    If the source node is exhausted, it will return the batch or raise StopIteration.
    If drop_last is True, the last batch will be dropped if it is smaller than batch_size.
    If drop_last is False, the last batch will be returned even if it is smaller than batch_size.

    Args:
        source (BaseNode[T]): The source node to batch the data from.
        batch_size (int): The size of the batch.
        drop_last (bool): Whether to drop the last batch if it is smaller than batch_size. Default is True.
    """

    SOURCE_KEY = "source"

    def __init__(self, source: BaseNode[T], batch_size: int, drop_last: bool = True):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.source = source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self.source.reset(initial_state[self.SOURCE_KEY])
        else:
            self.source.reset()

    def next(self) -> List[T]:
        batch: List[T] = []
        while len(batch) < self.batch_size:
            try:
                item = next(self.source)
            except StopIteration:
                break
            batch.append(item)
            if len(batch) == self.batch_size:
                return batch

        if len(batch) == self.batch_size:
            return batch
        elif len(batch) and not self.drop_last:
            return batch
        else:
            raise StopIteration()

    def get_state(self) -> Dict[str, Any]:
        return {self.SOURCE_KEY: self.source.state_dict()}


class Unbatcher(BaseNode[T]):
    """Unbatcher will flatten batches pulled from source, and
    yields elements in sequential order when next() is called on it.

    Args:
        source (BaseNode[T]): The source node to pull batches from.
    """

    SOURCE_KEY = "source"
    BATCH_IDX_KEY = "batch_idx"

    def __init__(self, source: BaseNode[Sequence[T]]):
        super().__init__(self)
        self.source = source

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self.source.reset(initial_state[self.SOURCE_KEY])
            self._cached_state_dict = initial_state[self.SOURCE_KEY]
            try:
                self._batch = next(self.source)
                self._batch_idx = initial_state[self.BATCH_IDX_KEY]
            except StopIteration:
                # next(self.source) will be called upon subsequent self.next() call
                # and raise StopIteration in the correct place.
                self._batch = []
                self._batch_idx = 0
        else:
            self.source.reset()
            self._batch = []
            self._cached_state_dict = None
            self._batch_idx = 0

    def next(self) -> T:
        while self._batch_idx >= len(self._batch):
            self._cached_state_dict = self.source.state_dict()
            self._batch = next(self.source)
            self._batch_idx = 0

        self._batch_idx += 1
        return self._batch[self._batch_idx - 1]

    def get_state(self) -> Dict[str, Any]:
        if self._cached_state_dict is None:
            self._cached_state_dict = self.source.state_dict()

        return {
            self.SOURCE_KEY: self._cached_state_dict,
            self.BATCH_IDX_KEY: self._batch_idx,
        }
