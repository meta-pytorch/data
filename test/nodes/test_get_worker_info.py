# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict

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
    else:
        item["worker_id"] = info.id
        item["num_workers"] = info.num_workers
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
        num_workers = 4
        src = MockSource(num_samples=40)
        node = ParallelMapper(src, _capture_worker_info, num_workers=num_workers, method="thread", in_order=False)

        results = list(node)
        self.assertEqual(len(results), 40)

        # All reported worker ids must be in [0, num_workers)
        for r in results:
            self.assertIn(r["worker_id"], set(range(num_workers)))
            self.assertEqual(r["num_workers"], num_workers)

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
        for r in results:
            self.assertIn(r["worker_id"], set(range(num_workers)))
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
        for r in results:
            self.assertIn(r["worker_id"], set(range(num_workers)))
            self.assertEqual(r["num_workers"], num_workers)

    def test_num_workers_zero_no_worker_info(self) -> None:
        """With num_workers=0 (inline), get_worker_info() returns None inside UDF."""
        src = MockSource(num_samples=5)
        node = ParallelMapper(src, _capture_worker_info, num_workers=0)
        results = list(node)
        for r in results:
            self.assertIsNone(r["worker_id"])
            self.assertIsNone(r["num_workers"])
