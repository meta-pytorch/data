# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import unittest
from multiprocessing.context import SpawnContext
from typing import Any, List, Optional, Tuple
from unittest.mock import patch

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader


class _MapDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return {"idx": index}


class _WorkerSeedDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> int:
        return torch.initial_seed()


class _StartTracker:
    def __init__(self, concurrent_starts: int) -> None:
        self._lock = threading.Lock()
        self._barrier = None
        if concurrent_starts > 1:
            self._barrier = threading.Barrier(concurrent_starts)
        self.active_starts = 0
        self.max_active_starts = 0

    def start_entered(self) -> None:
        with self._lock:
            self.active_starts += 1
            self.max_active_starts = max(self.max_active_starts, self.active_starts)
        if self._barrier is not None:
            self._barrier.wait(timeout=30)

    def start_exited(self) -> None:
        with self._lock:
            self.active_starts -= 1


class _ProcessProxy:
    def __init__(
        self,
        process: Any,
        tracker: _StartTracker,
        fail_start: bool = False,
    ) -> None:
        self._process = process
        self._tracker = tracker
        self._fail_start = fail_start

    @property
    def daemon(self) -> bool:
        return self._process.daemon

    @daemon.setter
    def daemon(self, value: bool) -> None:
        self._process.daemon = value

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid

    def start(self) -> None:
        self._tracker.start_entered()
        try:
            if self._fail_start:
                raise RuntimeError("worker start failed")
            self._process.start()
        finally:
            self._tracker.start_exited()

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    def join(self, timeout: Optional[float] = None) -> None:
        self._process.join(timeout)


class WorkerStartTest(unittest.TestCase):
    def test_invalid_parallelism_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            StatefulDataLoader(
                _MapDataset(1),
                spawn_worker_start_parallelism=0,
            )

    def test_spawn_starts_respect_parallelism_and_preserve_order(self) -> None:
        for parallelism in (1, 2, 4):
            with self.subTest(parallelism=parallelism):
                rows, max_active_starts = self._run_with_tracked_starts(parallelism)

                self.assertEqual(rows, list(range(20)))
                self.assertEqual(max_active_starts, parallelism)

    def test_parallel_spawn_preserves_worker_seeds(self) -> None:
        def worker_seeds(parallelism: int) -> List[int]:
            loader = StatefulDataLoader(
                _WorkerSeedDataset(),
                batch_size=1,
                num_workers=4,
                multiprocessing_context=SpawnContext(),
                generator=torch.Generator().manual_seed(42),
                spawn_worker_start_parallelism=parallelism,
            )
            return [int(seed) for seed in loader]

        self.assertEqual(worker_seeds(1), worker_seeds(4))

    def test_parallel_spawn_preserves_resume_and_persistent_workers(self) -> None:
        def create_loader() -> StatefulDataLoader:
            return StatefulDataLoader(
                _MapDataset(20),
                batch_size=2,
                num_workers=4,
                multiprocessing_context=SpawnContext(),
                persistent_workers=True,
                spawn_worker_start_parallelism=4,
            )

        loader = create_loader()
        loader_iter = iter(loader)
        rows = []
        for _ in range(3):
            rows.extend(next(loader_iter)["idx"].tolist())
        state_dict = loader.state_dict()

        resumed_loader = create_loader()
        resumed_loader.load_state_dict(state_dict)
        rows.extend(row for batch in resumed_loader for row in batch["idx"].tolist())

        self.assertEqual(rows, list(range(20)))
        self.assertEqual(
            [row for batch in resumed_loader for row in batch["idx"].tolist()],
            list(range(20)),
        )

    def test_parallel_start_failure_cleans_up_started_workers(self) -> None:
        self._assert_start_failure_cleans_up_workers(parallelism=2)

    def test_serial_start_failure_cleans_up_started_workers(self) -> None:
        self._assert_start_failure_cleans_up_workers(parallelism=1)

    def _assert_start_failure_cleans_up_workers(self, parallelism: int) -> None:
        context = SpawnContext()
        original_process = context.Process
        tracker = _StartTracker(parallelism)
        processes = []

        def create_process(*args, **kwargs):
            process = _ProcessProxy(
                original_process(*args, **kwargs),
                tracker,
                fail_start=len(processes) == 1,
            )
            processes.append(process)
            return process

        with patch.object(SpawnContext, "Process", side_effect=create_process):
            loader = StatefulDataLoader(
                _MapDataset(20),
                batch_size=2,
                num_workers=4,
                multiprocessing_context=context,
                spawn_worker_start_parallelism=parallelism,
            )

            with self.assertRaisesRegex(RuntimeError, "worker start failed"):
                next(iter(loader))

        self.assertEqual(len(processes), 4 if parallelism > 1 else 2)
        self.assertEqual(tracker.max_active_starts, parallelism)
        for process in processes:
            if process.pid is not None:
                self.assertFalse(process.is_alive())

    def _run_with_tracked_starts(self, parallelism: int) -> Tuple[List[int], int]:
        context = SpawnContext()
        original_process = context.Process
        tracker = _StartTracker(parallelism)

        def create_process(*args, **kwargs):
            return _ProcessProxy(original_process(*args, **kwargs), tracker)

        with patch.object(SpawnContext, "Process", side_effect=create_process):
            loader = StatefulDataLoader(
                _MapDataset(20),
                batch_size=2,
                num_workers=4,
                multiprocessing_context=context,
                spawn_worker_start_parallelism=parallelism,
            )
            rows = [row for batch in loader for row in batch["idx"].tolist()]
        return rows, tracker.max_active_starts
