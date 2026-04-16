# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict, List

import torch
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase
from torchdata.nodes import get_worker_info, IterableWrapper, ParallelMapper

from .utils import MockSource


def _capture_worker_info(item: Dict[str, Any]) -> Dict[str, Any]:
    """UDF that augments the item with the current worker's WorkerInfo fields."""
    info = get_worker_info()
    item = dict(item)
    if info is None:
        item["worker_id"] = None
        item["num_workers"] = None
        item["seed"] = None
    else:
        item["worker_id"] = info.id
        item["num_workers"] = info.num_workers
        item["seed"] = info.seed
    return item


def _capture_torch_worker_info(item: Dict[str, Any]) -> Dict[str, Any]:
    """UDF that reads worker info via torch.utils.data.get_worker_info() (process workers only)."""
    info = torch.utils.data.get_worker_info()
    item = dict(item)
    item["worker_id"] = info.id if info is not None else None
    item["num_workers"] = info.num_workers if info is not None else None
    return item


class TestGetWorkerInfo(TestCase):
    def test_none_outside_worker(self) -> None:
        self.assertIsNone(get_worker_info())

    def test_thread_workers(self) -> None:
        num_workers = 3
        src = MockSource(num_samples=12)
        node = ParallelMapper(src, _capture_worker_info, num_workers=num_workers, method="thread", in_order=False)

        results = list(node)
        self.assertEqual(len(results), 12)

        seen_ids = {r["worker_id"] for r in results}
        # All worker ids must be in [0, num_workers)
        self.assertTrue(seen_ids.issubset(set(range(num_workers))), f"Unexpected worker ids: {seen_ids}")
        # Every item reports the correct num_workers
        for r in results:
            self.assertEqual(r["num_workers"], num_workers)
            self.assertIsNotNone(r["seed"])

    def test_thread_workers_each_id_used(self) -> None:
        num_workers = 4
        src = MockSource(num_samples=40)
        node = ParallelMapper(src, _capture_worker_info, num_workers=num_workers, method="thread", in_order=False)

        results = list(node)
        seen_ids = {r["worker_id"] for r in results}
        # With 40 items and 4 workers we expect all 4 worker ids to appear
        self.assertEqual(seen_ids, set(range(num_workers)))

    def test_thread_workers_unique_seeds(self) -> None:
        num_workers = 4
        src = MockSource(num_samples=40)
        node = ParallelMapper(src, _capture_worker_info, num_workers=num_workers, method="thread", in_order=False)

        results = list(node)
        # Group seeds by worker id; each worker should report one unique seed
        seeds_by_worker: Dict[int, set] = {}
        for r in results:
            seeds_by_worker.setdefault(r["worker_id"], set()).add(r["seed"])
        # Each worker has exactly one seed
        for wid, seed_set in seeds_by_worker.items():
            self.assertEqual(len(seed_set), 1, f"Worker {wid} reported multiple seeds: {seed_set}")
        # Seeds differ across workers
        all_seeds = [next(iter(s)) for s in seeds_by_worker.values()]
        self.assertEqual(len(set(all_seeds)), len(all_seeds), f"Workers share seeds: {all_seeds}")

    @unittest.skipIf(IS_WINDOWS, "forkserver not supported on Windows")
    def test_process_workers_get_worker_info(self) -> None:
        """torch.utils.data.get_worker_info() works correctly in process workers."""
        num_workers = 2
        src = MockSource(num_samples=8)
        node = ParallelMapper(
            src,
            _capture_torch_worker_info,
            num_workers=num_workers,
            method="process",
            multiprocessing_context="forkserver",
            in_order=False,
        )

        results = list(node)
        self.assertEqual(len(results), 8)
        seen_ids = {r["worker_id"] for r in results}
        self.assertTrue(seen_ids.issubset(set(range(num_workers))))
        for r in results:
            self.assertEqual(r["num_workers"], num_workers)

    @unittest.skipIf(IS_WINDOWS, "forkserver not supported on Windows")
    def test_process_workers_torchdata_get_worker_info(self) -> None:
        """torchdata.nodes.get_worker_info() works correctly in process workers."""
        num_workers = 2
        src = MockSource(num_samples=8)
        node = ParallelMapper(
            src,
            _capture_worker_info,
            num_workers=num_workers,
            method="process",
            multiprocessing_context="forkserver",
            in_order=False,
        )

        results = list(node)
        self.assertEqual(len(results), 8)
        seen_ids = {r["worker_id"] for r in results}
        self.assertTrue(seen_ids.issubset(set(range(num_workers))))
        for r in results:
            self.assertEqual(r["num_workers"], num_workers)

    def test_num_workers_zero_no_worker_info(self) -> None:
        """With num_workers=0 (inline), get_worker_info() returns None inside UDF."""
        src = MockSource(num_samples=5)
        node = ParallelMapper(src, _capture_worker_info, num_workers=0)
        results = list(node)
        for r in results:
            self.assertIsNone(r["worker_id"])
            self.assertIsNone(r["num_workers"])
