# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import math
import time
import unittest
from copy import deepcopy

from typing import Iterator

import torch
import torch.utils.data

from parameterized import parameterized
from torch.testing._internal.common_utils import IS_MACOS, TEST_CUDA, TestCase
from torchdata.stateful_dataloader import Stateful, StatefulDataLoader


class DummyIterator(Iterator, Stateful):
    def __init__(self, samples, shuffle, include_generator):
        self.samples = samples
        self.shuffle = shuffle
        self.include_generator = include_generator
        self.size = len(self.samples)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.samples):
            raise StopIteration
        if self.shuffle:
            i = torch.randint(self.size, (1,)).item()
        else:
            i = self.i
        sample = self.samples[i]
        self.i += 1
        return sample

    def state_dict(self):
        sd = {"nest": {"i": self.i}}
        if self.include_generator:
            sd["nest"]["g"] = torch.get_rng_state()
        return sd

    def load_state_dict(self, state_dict):
        self.i = state_dict["nest"]["i"]
        if self.include_generator:
            torch.set_rng_state(state_dict["nest"]["g"])


class DummySamplerIterator(Iterator, Stateful):
    def __init__(self, size):
        self.size = size
        self.i = 0

    def __next__(self):
        idx = self.i
        if idx >= self.size:
            raise StopIteration
        self.i += 1
        return idx

    def state_dict(self):
        return {"i": self.i}

    def load_state_dict(self, state_dict):
        self.i = state_dict["i"]


class DummySampler(torch.utils.data.Sampler):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        return DummySamplerIterator(self.size)

    def __len__(self):
        return self.size


class DummyIteratorIterableDataset(torch.utils.data.IterableDataset, Iterator, Stateful):
    def __init__(self, samples, shuffle, include_generator):
        self.samples = samples
        self.shuffle = shuffle
        self.include_generator = include_generator
        self.size = len(self.samples)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.samples):
            raise StopIteration
        if self.shuffle:
            i = torch.randint(self.size, (1,)).item()
        else:
            i = self.i
        sample = self.samples[i]
        self.i += 1
        return sample

    def state_dict(self):
        sd = {"nest": {"i": self.i}}
        if self.include_generator:
            sd["nest"]["g"] = torch.get_rng_state()
        return sd

    def load_state_dict(self, state_dict):
        self.i = state_dict["nest"]["i"]
        if self.include_generator:
            torch.set_rng_state(state_dict["nest"]["g"])


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, sizes_for_all_workers, shuffle=False, include_generator=True):
        self.sizes_for_all_workers = sizes_for_all_workers
        self.shuffle = shuffle
        self.include_generator = include_generator

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
        else:
            worker_id = 0
            self.sizes_for_all_workers = [sum(self.sizes_for_all_workers)]

        start = sum(self.sizes_for_all_workers[:worker_id])
        iter_data = list(range(start, start + self.sizes_for_all_workers[worker_id]))
        return DummyIterator(iter_data, self.shuffle, self.include_generator)


class DummyMapDataset(torch.utils.data.Dataset):
    def __init__(self, size, shuffle, include_generator=True):
        self.size = size
        self.data = [{"id": i, "strcol": f"strcol_{i}", "listcol": [i, i + 1, i + 2]} for i in range(size)]
        self.shuffle = shuffle
        self.include_generator = include_generator

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.shuffle:
            i = torch.randint(self.size, (1,)).item()
        return self.data[i]

    def state_dict(self):
        if self.include_generator:
            return {
                "g": torch.get_rng_state(),
            }
        else:
            return {}

    def load_state_dict(self, state_dict):
        if self.include_generator:
            torch.set_rng_state(state_dict["g"])


class DynamicStateIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, samples):
        self.samples = samples
        self.size = len(self.samples)
        self.i = 0
        self.state = {}

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.samples):
            raise StopIteration

        sample = self.samples[self.i]
        self.i += 1
        return sample

    def state_dict(self):
        state = {"i": self.i}
        for a in range(self.i):
            state[str(a)] = {a: list(range(a))}
            state[f"t{str(a)}"] = torch.tensor(a, dtype=torch.int8)
        return state

    def load_state_dict(self, state_dict):
        self.i = state_dict["i"]
        self.state = state_dict


def identity(x):
    return x


class TestStatefulDataLoaderIterable_shard0(TestCase):
    def _get_dataset(self, shuffle):
        return DummyIterableDataset([0, 100, 37], shuffle=shuffle)

    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        dataset = self._get_dataset(shuffle)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        exp = []
        for data in it:
            exp.append(data)

        # Restore new instance from state
        batches = []
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        for batch in iter(dl):
            batches.append(batch)

        self.assertEqual(exp, batches)

    def test_no_mp(self):
        for batch_size, interrupt in itertools.product([None, 7], [0, 1, 10]):
            with self.subTest(batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=0,
                    batch_size=batch_size,
                    pw=False,
                    interrupt=interrupt,
                )

    def test_mp_x(self):
        for batch_size, interrupt in itertools.product([None, 7], [0, 1, 10]):
            with self.subTest(batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=3,
                    batch_size=batch_size,
                    pw=False,
                    interrupt=interrupt,
                )

    def test_mp_pw(self):
        for batch_size, interrupt in itertools.product([None, 7], [0, 1, 10]):
            with self.subTest(batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=3,
                    batch_size=batch_size,
                    pw=True,
                    interrupt=interrupt,
                )

    def test_mp_every_n_steps(self):
        batch_size = 7
        for every_n_steps, interrupt in itertools.product([2, 5], [0, 1, 10]):
            with self.subTest(every_n_steps=every_n_steps, batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=3,
                    batch_size=batch_size,
                    pw=True,
                    interrupt=interrupt,
                )

    def test_random_state(self):
        for num_workers, interrupt in itertools.product([0, 3], [0, 1, 10]):
            with self.subTest(num_workers=num_workers, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=num_workers,
                    batch_size=7,
                    pw=False,
                    interrupt=interrupt,
                    shuffle=True,
                )


class TestStatefulDataLoaderMap_shard1(TestStatefulDataLoaderIterable_shard0):
    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        if num_workers == 0:
            return
        dataset = DummyMapDataset(100, shuffle=shuffle)
        generator = torch.Generator()
        generator.manual_seed(13)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            batch_size=batch_size,
            sampler=sampler,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )

        if interrupt is None:
            interrupt = len(dl)

        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        exp = []
        for batch in it:
            exp.append(batch)

        # Restore new instance from state
        generator = torch.Generator()
        generator.manual_seed(13)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            batch_size=batch_size,
            sampler=sampler,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        batches = []
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)


class TestStatefulSampler_shard1(TestStatefulDataLoaderIterable_shard0):
    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        dataset = DummyMapDataset(100, shuffle=shuffle)
        sampler = DummySampler(len(dataset))
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            batch_size=batch_size,
            sampler=sampler,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )

        if interrupt is None:
            interrupt = len(dl)

        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        exp = []
        for batch in it:
            exp.append(batch)

        # Restore new instance from state
        sampler = DummySampler(len(dataset))
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            batch_size=batch_size,
            sampler=sampler,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        batches = []
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)


class GeneratorIterable(torch.utils.data.IterableDataset):
    def __init__(self, sizes_for_all_workers, increment_epoch=False):
        self.sizes_for_all_workers = sizes_for_all_workers
        self.i = 0
        self.resume = None
        self.epoch = 0
        self.increment_epoch = increment_epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
        else:
            worker_id = 0
            self.sizes_for_all_workers = [sum(self.sizes_for_all_workers)]
        self.i = 0
        if self.resume is not None:
            self.i = self.resume
            self.resume = None
        start = sum(self.sizes_for_all_workers[:worker_id])
        skip = self.i
        for i in range(start + skip, start + self.sizes_for_all_workers[worker_id]):
            self.i += 1
            yield (i, self.epoch)

        # To save end-of-epoch state properly, reset variables here so loading
        # will begin from the correct position and epoch
        self.i = 0
        if self.increment_epoch:
            self.epoch += 1

    def state_dict(self):
        return {"i": self.i, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.resume = state["i"]
        self.epoch = state["epoch"]


class GeneratorIterableNoState(torch.utils.data.IterableDataset):
    def __init__(self, sizes_for_all_workers):
        self.sizes_for_all_workers = sizes_for_all_workers

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
        else:
            worker_id = 0
            self.sizes_for_all_workers = [sum(self.sizes_for_all_workers)]
        start = sum(self.sizes_for_all_workers[:worker_id])
        yield from range(start, start + self.sizes_for_all_workers[worker_id])


class GeneratorSampler(torch.utils.data.Sampler):
    def __init__(self, limit):
        self.limit = limit
        self.i = 0
        self.resume = None
        self.epoch = 0

    def __iter__(self):
        if self.resume is not None:
            self.i = self.resume
            self.resume = None
        skip = self.i
        for i in range(skip, self.limit):
            self.i += 1
            yield i

        self.i = 0
        self.epoch += 1

    def state_dict(self):
        return {"i": self.i, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.resume = state["i"]
        self.epoch = state["epoch"]


class TestStatefulDataLoaderGenerator_shard2(TestStatefulDataLoaderIterable_shard0):
    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        dataset = GeneratorIterable([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        exp = list(dl)

        if interrupt is None:
            interrupt = len(exp)

        dataset = GeneratorIterable([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        batches = []
        it = iter(dl)
        for _ in range(interrupt):
            batches.append(next(it))
        state_dict = dl.state_dict()

        self.assertEqual(batches, exp[:interrupt])

        # Restore new instance from state
        dataset = GeneratorIterable([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)


class TestStatefulDataLoaderGeneratorNoState_shard2(TestStatefulDataLoaderIterable_shard0):
    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        dataset = GeneratorIterableNoState([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        exp = list(dl)

        if interrupt is None:
            interrupt = len(exp)

        dataset = GeneratorIterableNoState([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        batches = []
        it = iter(dl)
        for _ in range(interrupt):
            batches.append(next(it))
        state_dict = dl.state_dict()
        self.assertEqual(batches, exp[:interrupt])

        # Restore new instance from state
        dataset = GeneratorIterableNoState([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)


class TestSnapshotZero_shard2(TestCase):
    def test_generator(self):
        num_workers = 3
        every_n_steps = 10
        for pw in [False, True]:
            dataset = GeneratorIterable([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = list(it)

            dl.load_state_dict(state0)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_iterable(self):
        num_workers = 3
        every_n_steps = 10
        for pw in [False, True]:
            dataset = DummyIterableDataset([0, 100, 37], shuffle=True)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = list(it)

            dl.load_state_dict(state0)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map(self):
        num_workers = 3
        every_n_steps = 10
        for pw in [False, True]:
            dataset = DummyMapDataset(100, shuffle=True)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = list(it)

            dl.load_state_dict(state0)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map_shuffle(self):
        num_workers = 3
        every_n_steps = 10
        for pw in [False, True]:
            dataset = DummyMapDataset(100, shuffle=False)
            dl = StatefulDataLoader(
                dataset=dataset,
                shuffle=True,  # Use default RandomSampler
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = list(it)

            dl.load_state_dict(state0)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map_iterrupted_shuffle(self):
        every_n_steps = 10

        for pw, num_workers, every_n_steps in itertools.product([False, True], [0, 2], [1, 15]):
            dataset = DummyMapDataset(10, shuffle=True)
            dl = StatefulDataLoader(
                dataset=dataset,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw if num_workers > 0 else False,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = []
            for _ in range(4):
                exp.append(next(it))
            state1 = dl.state_dict()

            dl.load_state_dict(state1)
            it = iter(dl)
            for data in it:
                exp.append(data)

            dl.load_state_dict(state0)
            batches = []
            for data in iter(dl):
                batches.append(data)

            self.assertEqual(batches, exp)


class TestSnapshotEnd_shard2(TestCase):
    def test_generator(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = GeneratorIterable([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            exp = list(dl)
            state_end = dl.state_dict()

            batches = list(dl)  # simple restart
            self.assertEqual(batches, exp)

            dataset = GeneratorIterable([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            it = iter(dl)
            for _ in range(2):
                next(it)
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_generator_no_state(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = GeneratorIterableNoState([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            exp = list(dl)
            state_end = dl.state_dict()

            batches = list(dl)  # simple restart
            self.assertEqual(batches, exp)

            dataset = GeneratorIterableNoState([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            it = iter(dl)
            for _ in range(2):
                next(it)
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_iterable(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = DummyIterableDataset([0, 100, 37], shuffle=True)
            g = torch.Generator()
            g.manual_seed(4)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                generator=g,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            list(dl)
            state_end = dl.state_dict()
            exp = list(dl)

            g.manual_seed(4)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                generator=g,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = DummyMapDataset(100, shuffle=True)
            generator = torch.Generator()
            generator.manual_seed(15)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                generator=generator,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            list(dl)
            state_end = dl.state_dict()
            exp = list(dl)

            generator.manual_seed(15)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                generator=generator,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map_shuffle(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = DummyMapDataset(100, shuffle=False)
            dl = StatefulDataLoader(
                dataset=dataset,
                shuffle=True,  # Use default RandomSampler
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            list(dl)
            state_end = dl.state_dict()
            exp = list(dl)

            dataset = DummyMapDataset(100, shuffle=False)
            dl = StatefulDataLoader(
                dataset=dataset,
                shuffle=True,  # Use default RandomSampler
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)


class TestNumWorkersMismatch_shard3(TestCase):
    def test_num_workers_mismatch(self):
        for initial_num_workers, num_workers in ((0, 3), (3, 0)):
            if initial_num_workers == num_workers:
                continue
            dataset = DummyMapDataset(100, shuffle=False)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=initial_num_workers,
                collate_fn=identity,
                multiprocessing_context=("forkserver" if IS_MACOS and initial_num_workers else None),
            )
            state = dl.state_dict()

            iter(dl)
            state = dl.state_dict()
            self.assertTrue(len(state) > 0)

            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            dl.load_state_dict(state)
            try:
                iter(dl)
                raise Exception("Expected AssertionError to be thrown")
            except AssertionError:
                continue
            self.assertTrue(False, "Error should be of type AssertionError")


class TestConcurrentDataLoaders_shard3(TestCase):
    def test_two_dataloaders(self) -> None:
        dataset = DummyMapDataset(100, shuffle=False)
        sdl = StatefulDataLoader(
            dataset=dataset,
            num_workers=2,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS else None,
        )
        exp = list(sdl)

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=2,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS else None,
        )
        data = list(dl)
        self.assertEqual(data, exp)


class TestFastStateDictRequest_shard3(TestCase):
    def _run_test(self, snapshot_every_n_steps, interrupt):
        num_workers = 4
        dataset = DummyIterableDataset([25, 25, 25, 25], shuffle=True)

        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=4,
            collate_fn=identity,
            persistent_workers=True,
            multiprocessing_context="forkserver" if IS_MACOS else None,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        for _ in range(2):
            next(it)
        exp = list(it)

        dl.load_state_dict(state_dict)
        # new iter after load_state_dict, ask for state dict before num_workers batches
        # are yielded to ensure old worker states are stored properly
        it = iter(dl)
        for _ in range(2):
            next(it)

        state_dict2 = dl.state_dict()
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=4,
            collate_fn=identity,
            persistent_workers=True,
            multiprocessing_context="forkserver" if IS_MACOS else None,
        )
        dl.load_state_dict(state_dict2)
        data = list(dl)

        self.assertEqual(data, exp)

    def test_fast_state_dict_request(self) -> None:
        self._run_test(0, 11)

    def test_fast_state_dict_request_skip_steps(self) -> None:
        self._run_test(17, 19)


class TestJsonSerDe_shard3(TestCase):
    def _run_test_iterable(self, num_workers):
        interrupt = 4
        dataset = DummyIterableDataset([0, 100, 37], shuffle=False, include_generator=False)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        list(dl)

        exp = []
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        ser = json.dumps(state_dict)
        for data in it:
            exp.append(data)

        # Restore new instance from state
        batches = []
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        deser = json.loads(ser)
        dl.load_state_dict(deser)
        for batch in iter(dl):
            batches.append(batch)

        self.assertEqual(exp, batches)

    def _run_test_map(self, num_workers):
        interrupt = 4
        dataset = DummyMapDataset(100, shuffle=False, include_generator=False)
        sampler = DummySampler(100)
        dl = StatefulDataLoader(
            dataset=dataset,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        list(dl)

        exp = []
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        ser = json.dumps(state_dict)
        for data in it:
            exp.append(data)

        # Restore new instance from state
        batches = []
        dl = StatefulDataLoader(
            dataset=dataset,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        deser = json.loads(ser)
        dl.load_state_dict(deser)
        for batch in iter(dl):
            batches.append(batch)

        self.assertEqual(exp, batches)

    def test_json_serde_single_process(self):
        self._run_test_iterable(0)

    def test_json_serde_multi_process(self):
        self._run_test_iterable(3)

    def test_json_serde_single_process_map(self):
        self._run_test_map(0)

    def test_json_serde_multi_process_map(self):
        self._run_test_map(3)


class ErrorDataset(torch.utils.data.Dataset):
    def __getitem__(self, index: int):
        raise ValueError("Iteration error")

    def __len__(self):
        return 10


ERROR_MSG = "Worker init error"


def error_worker_init_fn(worker_id):
    raise ValueError(ERROR_MSG)


class TestInitialState_shard0(TestCase):
    def test_initial_state(self):
        for pw in [False, True]:
            num_workers = 4
            dataset = DummyMapDataset(100, shuffle=False)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                persistent_workers=pw,
                collate_fn=identity,
                multiprocessing_context="forkserver" if IS_MACOS else None,
                pin_memory=TEST_CUDA,
            )
            state = dl.state_dict()
            self.assertEqual(len(state["_snapshot"]["_worker_snapshots"]), num_workers)

            exp = list(dl)

            it = iter(dl)
            for _ in range(2):
                next(it)

            dl.load_state_dict(state)
            data = list(dl)
            self.assertEqual(data, exp)

            data2 = list(dl)
            self.assertEqual(data2, exp)

    def test_load_state_after_initial_state_dict(self):
        for pw, interrupt in itertools.product([False, True], [2, 9]):
            num_workers = 4
            dataset = DummyMapDataset(100, shuffle=True)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                persistent_workers=pw,
                collate_fn=identity,
                multiprocessing_context="forkserver" if IS_MACOS else None,
                pin_memory=TEST_CUDA,
            )

            it = iter(dl)
            for _ in range(interrupt):
                next(it)
            state = dl.state_dict()
            exp = list(it)

            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                persistent_workers=pw,
                collate_fn=identity,
                multiprocessing_context="forkserver" if IS_MACOS else None,
                pin_memory=TEST_CUDA,
            )
            state0 = dl.state_dict()
            self.assertEqual(len(state0["_snapshot"]["_worker_snapshots"]), num_workers)
            dl.load_state_dict(state)
            data = list(dl)
            self.assertEqual(data, exp)

    def test_load_state_before_initial_state_dict(self):
        for pw, interrupt in itertools.product([False, True], [2, 9]):
            num_workers = 4
            dataset = DummyMapDataset(100, shuffle=True)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                persistent_workers=pw,
                collate_fn=identity,
                multiprocessing_context="forkserver" if IS_MACOS else None,
                pin_memory=TEST_CUDA,
            )

            it = iter(dl)
            for _ in range(interrupt):
                next(it)
            state = dl.state_dict()
            exp = list(it)

            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                persistent_workers=pw,
                collate_fn=identity,
                multiprocessing_context="forkserver" if IS_MACOS else None,
                pin_memory=TEST_CUDA,
            )
            dl.load_state_dict(state)
            state0 = dl.state_dict()
            self.assertEqual(state0, state)
            data = list(dl)
            self.assertEqual(data, exp)

    def test_init_error(self):
        num_workers = 4
        dataset = DummyMapDataset(100, shuffle=True)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS else None,
            worker_init_fn=error_worker_init_fn,
            pin_memory=TEST_CUDA,
        )
        with self.assertRaisesRegex(ValueError, ERROR_MSG):
            iter(dl)

    def test_iteration_error(self):
        num_workers = 4
        dataset = ErrorDataset()
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS else None,
            pin_memory=TEST_CUDA,
        )
        it = iter(dl)
        with self.assertRaisesRegex(ValueError, "Iteration error"):
            next(it)

    def test_load_then_state(self):
        for pw in itertools.product([False, True]):
            num_workers = 4
            dataset = DummyMapDataset(100, shuffle=True)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                persistent_workers=pw,
                collate_fn=identity,
                multiprocessing_context="forkserver" if IS_MACOS else None,
                pin_memory=TEST_CUDA,
            )

            state0 = dl.state_dict()
            exp = list(dl)

            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                persistent_workers=pw,
                collate_fn=identity,
                multiprocessing_context="forkserver" if IS_MACOS else None,
                pin_memory=TEST_CUDA,
            )
            it = iter(dl)
            for _ in range(3):
                next(it)
            dl.load_state_dict(state0)
            state1 = dl.state_dict()
            self.assertEqual(state1, state0)

            batches = list(dl)
            self.assertEqual(batches, exp)


class TestStatefulDataLoaderIterable2_shard0(TestStatefulDataLoaderIterable_shard0):
    # Perform sanity test checks with the iterable dataset that is also an iterator
    def _get_dataset(self, shuffle):
        return DummyIteratorIterableDataset(list(range(100)), shuffle=shuffle, include_generator=True)


class TestDynamicStateIterableDataset_shard0(TestCase):
    def test(self):
        dataset = DynamicStateIterableDataset(list(range(100)))
        num_workers = 2
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        it = iter(dl)
        # Fetch at least one batch from each worker
        for _ in range((num_workers + 1) * 2):
            next(it)
        state_dict = dl.state_dict()
        worker_state = state_dict["_snapshot"]["_worker_snapshots"]["worker_0"]["fetcher_state"]["dataset_iter_state"]
        self.assertEqual(len(worker_state), 7)
        deep_copy_state_dict = deepcopy(state_dict)

        # Iterate a few more steps and ensure earlier state_dict hasn't changed
        for _ in range(num_workers * 2):
            next(it)
        next_state_dict = dl.state_dict()
        self.assertEqual(state_dict, deep_copy_state_dict)
        self.assertFalse(state_dict == next_state_dict)
        worker_state = next_state_dict["_snapshot"]["_worker_snapshots"]["worker_0"]["fetcher_state"][
            "dataset_iter_state"
        ]
        self.assertEqual(len(worker_state), 11)

        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        it = iter(dl)
        exp = []
        for _ in range(num_workers):
            exp.extend(next(it))
        state_dict = dl.state_dict()
        self.assertEqual(exp, [3, 3])
        worker_state = state_dict["_snapshot"]["_worker_snapshots"]["worker_0"]["fetcher_state"]["dataset_iter_state"]
        self.assertEqual(len(worker_state), 9)


class TestDatasetIteratorStateDuplication_shard0(TestCase):
    def test(self):
        dataset = DummyIteratorIterableDataset(list(range(100)), shuffle=True, include_generator=True)
        for num_workers in (0, 2):
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            it = iter(dl)
            # Fetch at least one batch from each worker
            for _ in range(num_workers + 1):
                next(it)
            state_dict = dl.state_dict()

            if num_workers > 0:
                for i in range(num_workers):
                    # Ensure worker state is stored only once if the dataset is also the iterator
                    self.assertEqual(
                        state_dict["_snapshot"]["_worker_snapshots"][f"worker_{i}"]["dataset_state"],
                        None,
                    )
                    self.assertTrue(
                        state_dict["_snapshot"]["_worker_snapshots"][f"worker_{i}"]["fetcher_state"][
                            "dataset_iter_state"
                        ]
                    )
            else:
                self.assertEqual(state_dict["dataset_state"], None)
                self.assertTrue(state_dict["fetcher_state"]["dataset_iter_state"])


class PeriodicStateIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.state = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
        }
        self.step = 0
        self.limit = 100

    def __iter__(self):
        for _ in range(self.step, self.limit):
            self.step += 1
            y = self.state[self.step % len(self.state)]
            self.state[self.step % len(self.state)] = 1 - y

            yield self.state[self.step % len(self.state)]

    def state_dict(self):
        return {
            "state": self.state,
            "step": self.step,
        }

    def load_state_dict(self, state):
        self.state = state["state"]
        self.step = state["step"]


class TestFastStateDictRequestRoundRobin_shard3(TestCase):
    def _run_test(self, snapshot_every_n_steps, interrupt):
        num_workers = 4
        dataset = PeriodicStateIterableDataset()

        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=1,
            collate_fn=identity,
            persistent_workers=True,
            multiprocessing_context="forkserver" if IS_MACOS else None,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        for _ in range(2):
            next(it)
        exp_state_dict = dl.state_dict()
        exp = list(it)

        dl.load_state_dict(state_dict)
        # new iter after load_state_dict, ask for state dict before num_workers batches
        # are yielded to ensure old worker states are stored properly
        it = iter(dl)
        for _ in range(2):
            next(it)

        state_dict2 = dl.state_dict()

        dataset = PeriodicStateIterableDataset()
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=1,
            collate_fn=identity,
            persistent_workers=True,
            multiprocessing_context="forkserver" if IS_MACOS else None,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )
        dl.load_state_dict(state_dict2)
        data = list(dl)

        self.assertEqual(data, exp)

        dataset = PeriodicStateIterableDataset()
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=1,
            collate_fn=identity,
            persistent_workers=True,
            multiprocessing_context="forkserver" if IS_MACOS else None,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )
        dl.load_state_dict(state_dict)
        it = iter(dl)
        for _ in range(2):
            next(it)

        state_dict3 = dl.state_dict()
        self.assertEqual(state_dict3, exp_state_dict)

    def test_fast_state_dict_request(self) -> None:
        # these test settings will trigger a failure if
        # the worker/main incremental state_dicts are out of sync during initialization
        self._run_test(1, 15)

    def test_fast_state_dict_request_skip_steps(self) -> None:
        self._run_test(17, 19)


class TestMultiEpochSDL_shard0(TestCase):
    def get_map_dl(self, data_size, num_workers, batch_size, shuffle):
        dataset = DummyMapDataset(data_size, shuffle=False)
        return StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
        )

    def _run(self, data_size, num_workers, batch_size, shuffle):
        # For reproducibility of testing, fixing the seed
        torch.manual_seed(0)
        dataloader1 = self.get_map_dl(
            data_size=data_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        # Run through the dataloader for 2 epochs and count the number of items yielded
        num_items_yielded = 0
        dataloader1_items = []
        for _ in range(2):
            for batch in dataloader1:
                dataloader1_items.append(batch)
                num_items_yielded += 1
        # Save the state dict
        state_dict = dataloader1.state_dict()
        # Create a new StatefulDataLoader instance and load the state dict
        new_dataloader1 = self.get_map_dl(
            data_size=data_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        new_dataloader1.load_state_dict(state_dict)
        # Run through the new dataloader for another 2 epochs and count the number of items yielded
        additional_num_items_yielded = 0
        for _ in range(2):
            epoch_num_items_yielded = 0
            for batch in new_dataloader1:
                dataloader1_items.append(batch)
                epoch_num_items_yielded += 1
            additional_num_items_yielded += epoch_num_items_yielded
        # Check that the total number of items yielded is correct
        self.assertEqual(num_items_yielded + additional_num_items_yielded, data_size * 4)

        # now run a second dataloder for 4 epochs and check if the order is same.
        # we need to fix the seed again since we want to bring the initial conditions to the same state as at the time of instantiating the first dataloader
        torch.manual_seed(0)
        dataloader2 = self.get_map_dl(
            data_size=data_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        dataloader2_items = []
        for _ in range(4):
            for batch in dataloader2:
                dataloader2_items.append(batch)

        self.assertEqual(dataloader1_items, dataloader2_items)

    @parameterized.expand(itertools.product([100], [0, 2], [1], [False, True]))
    def test_multi_epoch_sdl(self, datasize, num_workers, batch_size, shuffle):
        self._run(datasize, num_workers, batch_size, shuffle)


class TestEndOfEpochBehavior_shard0(TestCase):
    def get_map_dl(self, data_size, num_workers, batch_size, shuffle):
        dataset = DummyMapDataset(data_size, shuffle=False)
        return StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
        )

    def _count_items_yielded(self, data_loader: StatefulDataLoader) -> int:
        num_items_yielded = 0
        for _ in data_loader:
            num_items_yielded += 1
        return num_items_yielded

    def _run(self, data_size, num_workers, batch_size, shuffle):
        dataloader = self.get_map_dl(
            data_size=data_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        # Run through the dataloader for 1 epoch and count the number of items yielded
        num_items_yielded = 0

        for _ in dataloader:
            num_items_yielded += 1
            sd_in = dataloader.state_dict()
        sd_out = dataloader.state_dict()

        self.assertEqual(num_items_yielded, data_size)

        # Create a new StatefulDataLoader instance and load the state dict saved before the end of epoch
        dataloader_sd_in = self.get_map_dl(
            data_size=data_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        dataloader_sd_in.load_state_dict(sd_in)

        # Run through the new dataloader for 1 epoch and count the number of items yielded
        # num_items_yielded should be 0 since the state dict was saved before the end of epoch
        num_items_yielded = self._count_items_yielded(dataloader_sd_in)
        self.assertEqual(num_items_yielded, 0)

        # Create a new StatefulDataLoader instance and load the state dict saved after the end of epoch
        dataloader_sd_out = self.get_map_dl(
            data_size=data_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        dataloader_sd_out.load_state_dict(sd_out)

        # Run through the new dataloader for 1 epoch and count the number of items yielded
        # num_items_yielded should be data_size since the state dict was saved after the end of epoch
        num_items_yielded = self._count_items_yielded(dataloader_sd_out)
        self.assertEqual(num_items_yielded, data_size)

    @parameterized.expand(itertools.product([100], [0, 2], [1], [False, True]))
    def test_end_of_epoch_behavior(self, datasize, num_workers, batch_size, shuffle):
        self._run(datasize, num_workers, batch_size, shuffle)


class TestNotStatefulSamplerSDL_shard0(TestCase):
    def get_map_dl(self, data_size, num_workers, batch_size, sampler_cls):
        dataset = DummyMapDataset(data_size, shuffle=False)
        sampler = sampler_cls(dataset)
        return StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=sampler,
            multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
        )

    def _run(self, data_size, num_workers, batch_size, interrupt, sampler_cls):
        torch.manual_seed(0)  # Fixing seed for deterministic results
        dataloader1 = self.get_map_dl(
            data_size=data_size,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler_cls=sampler_cls,
        )
        # interrupt the dataloader after interrupt batches and save the state dict
        results_dataloader1 = []
        for i, batch in enumerate(dataloader1):
            results_dataloader1.append(batch)
            if i == interrupt:
                break
        state_dict = dataloader1.state_dict()

        torch.manual_seed(
            0
        )  # We need to fix seed again so that before fast forwarding we are at the same state of gen as before
        resumed_dataloader1 = self.get_map_dl(
            data_size=data_size,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler_cls=sampler_cls,
        )
        resumed_dataloader1.load_state_dict(state_dict)

        for batch in resumed_dataloader1:
            results_dataloader1.append(batch)

        # now start a completely new dataloader and get all the batches
        torch.manual_seed(0)
        dataloader2 = self.get_map_dl(
            data_size=data_size,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler_cls=sampler_cls,
        )
        results_dataloader2 = []
        for batch in dataloader2:
            results_dataloader2.append(batch)
        self.assertEqual(results_dataloader1, results_dataloader2)

    @parameterized.expand(
        itertools.product(
            [100],
            [0, 2],
            [1],
            [10, 50, 80],
            [torch.utils.data.RandomSampler, torch.utils.data.SequentialSampler],
        )
    )
    def test_notstatefulSDL(self, data_size, num_workers, batch_size, interrupt, sampler_cls):
        self._run(100, 0, 1, interrupt, sampler_cls)


class TestMultiEpochState_shard0(TestCase):
    def get_iterable_dl(self, pw, num_workers):
        data_size = [25, 50, 100, 75]
        if num_workers == 0:
            data_size = [sum(data_size)]
        dataset = GeneratorIterable(data_size, True)
        return StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            persistent_workers=pw,
            collate_fn=identity,
            multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
        )

    def _run(self, pw: bool, num_workers: int):
        dl = self.get_iterable_dl(pw, num_workers)
        exp0 = list(dl)
        state1 = dl.state_dict()
        exp1 = list(dl)
        exp2 = list(dl)
        self.assertEqual(exp0, [[(x[0][0], x[0][1] - 1)] for x in exp1])
        self.assertEqual(exp0, [[(x[0][0], x[0][1] - 2)] for x in exp2])

        dl = self.get_iterable_dl(pw, num_workers)
        it = iter(dl)
        for _ in range(2):
            next(it)
        dl.load_state_dict(state1)
        it = iter(dl)
        data1 = list(it)
        data2 = list(dl)
        self.assertEqual(data1, exp1)
        self.assertEqual(data2, exp2)

    def test_inline(self):
        self._run(False, 0)

    def test_pw(self):
        self._run(True, 4)


class CountIterCalls(torch.utils.data.IterableDataset):
    def __init__(self, length):
        self.length = length
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(list(range(self.length)))

    def state_dict(self):
        return {"iter_calls": self.iter_calls}

    def load_state_dict(self, state_dict):
        pass


class CountIterCallsIter(torch.utils.data.IterableDataset):
    def __init__(self, length):
        self.length = length
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        worker_id = 0
        if torch.utils.data.get_worker_info() is not None:
            worker_id = torch.utils.data.get_worker_info().id
        num_workers = 1
        if torch.utils.data.get_worker_info() is not None:
            num_workers = torch.utils.data.get_worker_info().num_workers

        num_samples = (int)(self.length / num_workers)
        self.iter_state = IterationState(num_samples * worker_id, num_samples * (worker_id + 1))
        return self

    def __next__(self):
        if self.iter_state.curr >= self.iter_state.end:
            raise StopIteration
        value = self.iter_state.curr
        self.iter_state.curr += 1
        return value

    def state_dict(self):
        return {"state": self.iter_state.get_state(), "iter_calls": self.iter_calls}

    def load_state_dict(self, state_dict):
        self.iter_state.set_state(state_dict["state"])


class TestSingleIterCalled_shard0(TestCase):
    def _get_iter_calls(self, state):
        if "dataset_state" in state:
            w_states = [state]
        else:
            w_states = list(state["_snapshot"]["_worker_snapshots"].values())

        if w_states[0]["dataset_state"] is not None:
            return [x["dataset_state"]["iter_calls"] for x in w_states]
        return [x["fetcher_state"]["dataset_iter_state"]["iter_calls"] for x in w_states]

    def _run_test(self, num_workers, dataset, expected_iter_calls):
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
        )
        iter(dl)
        state = dl.state_dict()
        # Ensure iter is called only once per worker
        self.assertEqual(self._get_iter_calls(state), [expected_iter_calls[0]] * max(1, num_workers))

        dl2 = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
        )
        dl2.load_state_dict(state)
        iter(dl2)
        state2 = dl2.state_dict()
        # Ensure that iter is called only once per worker even when dataloader resumes from a state
        self.assertEqual(self._get_iter_calls(state2), [expected_iter_calls[1]] * max(1, num_workers))

    def test_inline(self):
        self._run_test(0, CountIterCalls(100), [1, 2])

    def test_mp(self):
        self._run_test(2, CountIterCalls(100), [1, 1])

    def test_inline_iter(self):
        self._run_test(0, CountIterCallsIter(100), [1, 2])

    def test_mp_iter(self):
        self._run_test(2, CountIterCallsIter(100), [1, 1])


class IterationState:
    def __init__(self, start, end):
        self.curr = start
        self.end = end

    def set_state(self, state):
        self.curr = state["curr"]
        self.end = state["end"]

    def get_state(self):
        return {"curr": self.curr, "end": self.end}


class TestStateInitializationDataset(TestCase):
    def _run_test(self, num_workers, dataset):
        length = dataset.length

        # Ensure test is run with compatible parameters as the test and dataset used in the test doesn't cover all the corner cases
        if num_workers > 0:
            self.assertTrue(length % num_workers == 0)
        self.assertTrue(length > 30)

        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
        )
        it = iter(dl)
        data = []

        for _ in range(length - 30):
            data.extend(next(it))
        state = dl.state_dict()

        # Resume from state
        dl2 = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
        )
        dl2.load_state_dict(state)
        it = iter(dl2)

        for _ in range(30):
            data.extend(next(it))

        # Order could be different for multiworker case as the data comes from different workers, so use set to check equality instead of list
        self.assertEqual(set(data), set(range(length)))

    def test_inline(self):
        self._run_test(0, CountIterCallsIter(100))

    def test_mp(self):
        self._run_test(2, CountIterCallsIter(100))


class _TestSlowIndexDataset(torch.utils.data.Dataset):
    def __init__(self, end: int, slow_index: int):
        self.end = end
        self.slow_index = slow_index
        self._worker_id = None

    def __getitem__(self, idx):
        if idx == self.slow_index:
            time.sleep(1.0)
        return idx

    def __len__(self):
        return self.end


class _TestSlowIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.mid = math.ceil((self.end - self.start) / 2)

    def give_data(self, iter_start, iter_end):
        for i in range(iter_start, iter_end):
            if i == self.mid:
                time.sleep(1.0)
            yield i

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        iter_start = self.start + worker_id * per_worker
        iter_end = min(iter_start + per_worker, self.end)
        return self.give_data(iter_start, iter_end)


class TestOutOfOrderWithCheckpointing(TestCase):
    def test_out_of_order_index_ds(self):
        dataset = _TestSlowIndexDataset(end=10, slow_index=0)
        dataloader = StatefulDataLoader(
            dataset,
            num_workers=2,
            in_order=False,
        )

        # worker_id = 0 gets 'stuck' on 0 and also has 2 in it's queue
        # due to prefetch_factor being 2
        output = []
        for i, data in enumerate(dataloader):
            output.append(data)
            if i == 3:
                state_dict = dataloader.state_dict()
                break

        # 0 is the slow index, assert it isn't in the output before the pause
        self.assertNotIn(0, output)

        new_dataloader = StatefulDataLoader(dataset, num_workers=2, in_order=False)
        new_dataloader.load_state_dict(state_dict)
        for data in new_dataloader:
            output.append(data)

        self.assertEqual(len(output), 10)
        self.assertNotEqual(output, list(range(10)))
        self.assertEqual(sorted(output), list(range(10)))

    def test_out_of_order_iterable_ds_one_completed_worker(self):
        dataset = _TestSlowIterableDataset(start=0, end=10)
        dataloader = StatefulDataLoader(
            dataset,
            num_workers=2,
            prefetch_factor=2,
            in_order=False,
        )

        # break later on, as one of the workers will be finished
        output = []
        for i, data in enumerate(dataloader):
            output.append(data)
            if i == 7:
                state_dict = dataloader.state_dict()
                break

        worker_0_ended = state_dict["_snapshot"]["_worker_snapshots"]["worker_0"]["fetcher_state"]["fetcher_ended"]
        worker_1_ended = state_dict["_snapshot"]["_worker_snapshots"]["worker_1"]["fetcher_state"]["fetcher_ended"]
        self.assertTrue(worker_0_ended)
        self.assertFalse(worker_1_ended)

        new_dataloader = StatefulDataLoader(dataset, batch_size=1, num_workers=2, in_order=False)
        new_dataloader.load_state_dict(state_dict)
        for data in new_dataloader:
            output.append(data)

        self.assertEqual(len(output), 10)
        self.assertEqual(output, list(range(10)))
        self.assertNotEqual(output, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9])

    def test_out_of_order_iterable_ds_no_completed_workers(self):
        dataset = _TestSlowIterableDataset(start=0, end=10)
        dataloader = StatefulDataLoader(
            dataset,
            num_workers=2,
            prefetch_factor=2,
            in_order=False,
        )

        # break early - both workers will resume
        output = []
        for i, data in enumerate(dataloader):
            output.append(data)
            if i == 3:
                state_dict = dataloader.state_dict()
                break

        worker_0_ended = state_dict["_snapshot"]["_worker_snapshots"]["worker_0"]["fetcher_state"]["fetcher_ended"]
        worker_1_ended = state_dict["_snapshot"]["_worker_snapshots"]["worker_1"]["fetcher_state"]["fetcher_ended"]
        self.assertFalse(worker_0_ended)
        self.assertFalse(worker_1_ended)

        new_dataloader = StatefulDataLoader(dataset, batch_size=1, num_workers=2, in_order=False)
        new_dataloader.load_state_dict(state_dict)
        for data in new_dataloader:
            output.append(data)

        self.assertEqual(len(output), 10)
        self.assertEqual(output, list(range(10)))
        self.assertNotEqual(output, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9])


if __name__ == "__main__":
    unittest.main()
