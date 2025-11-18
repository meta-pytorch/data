# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, TypeVar

from torch.utils.data import Sampler

from torchdata.nodes.base_node import BaseNode, T

from .map import Mapper

from .types import Stateful

K = TypeVar("K", covariant=True)


class IterableWrapper(BaseNode[T]):
    """Thin Wrapper that converts any Iterable (including
    torch.utils.data.IterableDataset) in to a BaseNode.

    If iterable implements the Stateful Protocol, it will be saved and restored with its
    state_dict/load_state_dict methods.

    Args:
        iterable (Iterable[T]): Iterable to convert to BaseNode. IterableWrapper calls iter() on it.

    :warning: Note the distinction between state_dict/load_state_dict defined on Iterable, vs Iterator.
        Only the Iterable's state_dict/load_state_dict are used.
    """

    NUM_YIELDED_KEY = "_num_yielded"
    ITERABLE_KEY = "iterable"

    def __init__(self, iterable: Iterable[T]):
        super().__init__()
        self.iterable = iterable
        self._it: Optional[Iterator[T]] = None

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        self._num_yielded = 0
        self._it = None
        super().reset(initial_state)
        if initial_state is not None:
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
            if isinstance(self.iterable, Stateful):
                self.iterable.load_state_dict(initial_state[self.ITERABLE_KEY])
                self._it = iter(self.iterable)
            else:
                self._it = iter(self.iterable)
                # Naively fast-forwarding
                for i in range(self._num_yielded):
                    try:
                        next(self._it)
                    except StopIteration:
                        raise ValueError(
                            f"Tried to fast-forward {self._num_yielded} items during init but "
                            f"hit StopIteration after {i} items, this is likely a bug or malformed state_dict"
                        )
        else:
            self._it = iter(self.iterable)

    def next(self) -> T:
        item = next(self._it)  # type: ignore [arg-type, union-attr]
        self._num_yielded += 1
        return item

    def get_state(self) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {self.NUM_YIELDED_KEY: self._num_yielded}
        if isinstance(self.iterable, Stateful):
            state_dict[self.ITERABLE_KEY] = self.iterable.state_dict()
        return state_dict


def MapStyleWrapper(map_dataset: Mapping[K, T], sampler: Sampler[K]) -> BaseNode[T]:
    """Thin Wrapper that converts any MapDataset in to a torchdata.node
    If you want parallelism, copy this and replace Mapper with ParallelMapper.

    Args:
        map_dataset (Mapping[K, T]): - Apply map_dataset.__getitem__ to the outputs of sampler.
        sampler (Sampler[K]):
    """
    sampler_node: SamplerWrapper[K] = SamplerWrapper(sampler)
    mapper_node = Mapper(sampler_node, map_dataset.__getitem__)
    return mapper_node


class SamplerWrapper(BaseNode[T]):
    """
    Convert a sampler into a BaseNode. This is nearly identical to
    IterableWrapper except it includes a hook to call set_epoch on the sampler,
    if it supports it.

    Args:
        sampler (Sampler): Sampler to wrap.
        initial_epoch (int): initial epoch to set on the sampler
        epoch_updater (Optional[Callable[[int], int]] = None): callback to update epoch at start of new iteration. It's called at the beginning of each iterator request, except the first one.
    """

    NUM_YIELDED_KEY = "_num_yielded"
    EPOCH_KEY = "_epoch"
    SAMPLER_KEY = "_sampler"

    def __init__(
        self,
        sampler: Sampler[T],
        initial_epoch: int = 0,
        epoch_updater: Optional[Callable[[int], int]] = None,
    ):
        super().__init__()
        self.sampler = sampler
        self.epoch = initial_epoch
        self._num_yielded = 0
        self._started = False
        self.epoch_updater = epoch_updater or self._default_epoch_updater
        self._it: Optional[Iterator[T]] = None

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
            self.epoch = initial_state[self.EPOCH_KEY]
            if hasattr(self.sampler, "set_epoch"):
                # Always first set epoch, even if sampler is Stateful. Some Stateful
                # samplers may have some specific handling to overwrite this.
                self.sampler.set_epoch(self.epoch)
            if isinstance(self.sampler, Stateful):
                self.sampler.load_state_dict(initial_state[self.SAMPLER_KEY])
                self._it = iter(self.sampler)  # type: ignore [assignment]
            else:
                self._it = iter(self.sampler)
                for i in range(self._num_yielded):
                    try:
                        next(self._it)  # type: ignore [arg-type]
                    except StopIteration:
                        raise ValueError(
                            f"Tried to fast-forward {self._num_yielded} items during init but "
                            f"hit StopIteration after {i} items, this is likely a bug or malformed state_dict"
                        )
        else:
            self._num_yielded = 0
            if self._started:
                # Don't update epoch unless iterator has started
                self.epoch = self.epoch_updater(self.epoch)
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self.epoch)
            self._it = iter(self.sampler)
        self._started = False

    def next(self) -> T:
        self._started = True
        item = next(self._it)  # type: ignore [arg-type, union-attr]
        self._num_yielded += 1
        return item

    def get_state(self) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {
            self.NUM_YIELDED_KEY: self._num_yielded,
            self.EPOCH_KEY: self.epoch,
        }
        if isinstance(self.sampler, Stateful):
            state_dict[self.SAMPLER_KEY] = self.sampler.state_dict()
        return state_dict

    @classmethod
    def _default_epoch_updater(cls, epoch: int) -> int:
        return epoch + 1
