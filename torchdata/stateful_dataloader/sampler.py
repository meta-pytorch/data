# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
from typing import Any, Dict, Iterator, List, Optional, Sized

import torch.distributed as dist

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


class StatefulDistributedSampler(Sampler[int]):
    _YIELDED = "yielded"
    _EPOCH = "epoch"

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        dataset_size: Optional[int] = None,
    ) -> None:

        # Validate inputs
        if dataset is None and dataset_size is None:
            raise ValueError("Either dataset or dataset_size must be provided.")

        if dataset_size is not None:
            if dataset is not None and (hasattr(dataset, "__len__") and dataset_size != len(dataset)):
                raise ValueError(
                    f"dataset_size must match the length of the dataset. {dataset_size=} and {len(dataset)=}"
                )
            self.dataset_size = dataset_size
        else:
            if dataset is not None and hasattr(dataset, "__len__"):
                self.dataset_size = len(dataset)
            else:
                raise ValueError("Either a dataset with the __len__ method or dataset_size must be provided.")

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.dataset_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.dataset_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.dataset_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        self.yielded = 0
        self.next_yielded = None

    def __iter__(self):
        self.yielded = 0
        if self.next_yielded is not None:
            self.yielded = self.next_yielded
            self.next_yielded = None
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(self.dataset_size))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        it = iter(indices)

        for idx in itertools.islice(it, self.yielded, None):
            self.yielded += 1
            yield idx

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

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
