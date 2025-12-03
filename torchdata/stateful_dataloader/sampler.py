# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Any, Dict, Iterator, List, Optional, Sized

import torch.utils.data.sampler
from torch.utils.data import Dataset
from torch.utils.data.dataloader import _InfiniteConstantSampler
from torch.utils.data.sampler import Sampler

from .stateful import Stateful


class _StatefulRandomSamplerIterator(Iterator[int], Stateful):
    _GENERATOR = "generator"
    _YIELDED = "yielded"

    def __init__(self, sampler):
        self.sampler = sampler
        self.generator_state = self.sampler.generator.get_state()
        self.yielded = 0
        self.next_yielded = None
        self.n = len(sampler.data_source)
        self.replacement = sampler.replacement
        self.num_samples = sampler.num_samples
        self.chunk_size = 32
        self.perm: List[int] = self._get_perm()
        self.perm_index = 0
        self.chunk_index = 0

    def __iter__(self):
        return self

    def _get_perm(self) -> List[int]:
        if self.replacement:
            return torch.randint(
                high=self.n,
                size=(self.chunk_size,),
                dtype=torch.int64,
                generator=self.sampler.generator,
            ).tolist()
        else:
            return torch.randperm(self.n, generator=self.sampler.generator).tolist()

    def __next__(self):
        if self.yielded == self.num_samples:
            raise StopIteration()
        if self.perm_index == len(self.perm):
            self.perm = self._get_perm()
            self.perm_index = 0
        val = self.perm[self.perm_index]
        self.perm_index += 1
        self.yielded += 1
        return val

    def state_dict(self) -> dict:
        return {
            self._YIELDED: self.yielded,
            self._GENERATOR: self.generator_state,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.next_yielded = state_dict[self._YIELDED]
        self.generator_state = state_dict[self._GENERATOR]
        self.sampler.generator.set_state(self.generator_state)

        if self.next_yielded is not None:
            self.perm = self._get_perm()  # We want permutations from the latest generator state that's loaded
            for _ in range(self.next_yielded):
                next(self)
            self.yielded = self.next_yielded
            self.next_yielded = None


class RandomSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        if generator is None:
            # Prevoiusly the random seed was fixed as 1. We then changed it to system generated seed to ensure deterministic randomness.
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        self.generator = generator
        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        return _StatefulRandomSamplerIterator(self)

    def __len__(self) -> int:
        return self.num_samples


class _BatchSamplerIterator(Iterator[list[int]], Stateful):
    _SAMPLES_YIELDED = "samples_yielded"
    _SAMPLER_STATE = "sampler_state"
    _SAMPLER_ITER_STATE = "sampler_iter_state"

    def __init__(self, sampler, batch_size: int, drop_last: bool):
        self.sampler = sampler
        self.sampler_iter = iter(self.sampler)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.samples_yielded = 0

    def __next__(self) -> list[int]:
        batch = []
        try:
            for _ in range(self.batch_size):
                batch.append(next(self.sampler_iter))
                self.samples_yielded += 1
            return batch
        except StopIteration:
            if self.drop_last or len(batch) == 0:
                raise StopIteration
            else:
                return batch

    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {self._SAMPLES_YIELDED: self.samples_yielded}
        if isinstance(self.sampler, Stateful):
            sd[self._SAMPLER_STATE] = self.sampler.state_dict()
        if isinstance(self.sampler_iter, Stateful):
            sd[self._SAMPLER_ITER_STATE] = self.sampler_iter.state_dict()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.samples_yielded = state_dict[self._SAMPLES_YIELDED]
        if self._SAMPLER_STATE in state_dict:
            assert isinstance(self.sampler, Stateful)
            self.sampler.load_state_dict(state_dict[self._SAMPLER_STATE])
        self.sampler_iter = iter(self.sampler)
        if self._SAMPLER_ITER_STATE in state_dict:
            assert isinstance(self.sampler_iter, Stateful)
            self.sampler_iter.load_state_dict(state_dict[self._SAMPLER_ITER_STATE])

        if not (isinstance(self.sampler, Stateful) or isinstance(self.sampler_iter, Stateful)) and not isinstance(
            self.sampler, _InfiniteConstantSampler
        ):
            # We skip x samples if underlying sampler is not stateful
            for _ in range(self.samples_yielded):
                next(self.sampler_iter)

    def update_state_dict(self) -> None:
        if isinstance(self.sampler_iter, Stateful) and hasattr(self.sampler_iter, "update_state_dict"):
            self.sampler_iter.update_state_dict()


class BatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        return _BatchSamplerIterator(
            sampler=self.sampler,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )


class StatefulDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    _YIELDED = "yielded"
    _EPOCH = "epoch"

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.yielded = 0
        self.next_yielded = None

    def __iter__(self):
        self.yielded = 0
        if self.next_yielded is not None:
            self.yielded = self.next_yielded
            self.next_yielded = None
        it = super().__iter__()
        for idx in itertools.islice(it, self.yielded, None):
            self.yielded += 1
            yield idx

    def state_dict(self) -> Dict[str, Any]:
        return {self._YIELDED: self.yielded, self._EPOCH: self.epoch, "NEXT_YIELDED": self.next_yielded}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self._EPOCH not in state_dict:
            raise ValueError("Invalid state_dict")
        self.set_epoch(state_dict[self._EPOCH])
        if self._YIELDED not in state_dict:
            raise ValueError("Invalid state_dict")
        if state_dict[self._YIELDED] < 0:
            raise ValueError("Cannot load state_dict with negative yielded value")
        self.next_yielded = state_dict[self._YIELDED]
