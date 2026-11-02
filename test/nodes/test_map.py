# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import threading
import unittest
from typing import List, Optional
from unittest import mock

from parameterized import parameterized
from torch.testing._internal.common_utils import IS_WINDOWS, TEST_CUDA, TestCase
from torchdata.nodes.batch import Batcher

from torchdata.nodes.map import Mapper, ParallelMapper
from torchdata.nodes.pin_memory import PinMemory
from torchdata.nodes.prefetch import Prefetcher

from .utils import MockSource, RandomSleepUdf, run_test_save_load_state, StatefulRangeNode, udf_raises


class TestMap(TestCase):
    def _test_exception_handling_mapper(self, pin_memory, method):
        batch_size = 6
        multiprocessing_context = None if IS_WINDOWS else "forkserver"
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size)
        node = ParallelMapper(
            node,
            udf_raises,
            num_workers=2,
            method=method,
            multiprocessing_context=multiprocessing_context,
        )
        node = Mapper(node, udf_raises)
        if pin_memory:
            node = PinMemory(node)
        node = Prefetcher(node, prefetch_factor=2)

        with self.assertRaisesRegex(ValueError, "test exception"):
            list(node)

    def test_exception_handling_mapper(self):
        self._test_exception_handling_mapper(False, "thread")

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_exception_handling_mapper_cuda(self):
        self._test_exception_handling_mapper(True, "thread")

    def test_exception_handling_mapper_multiprocess(self):
        self._test_exception_handling_mapper(False, "process")

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_exception_handling_mapper_multiprocess_cuda(self):
        self._test_exception_handling_mapper(True, "process")

    def _test_map(self, in_order, method, prebatch) -> None:
        batch_size = 6
        n = 80
        multiprocessing_context = None if IS_WINDOWS else "forkserver"
        src = MockSource(num_samples=n)
        node = Batcher(src, batch_size=batch_size, drop_last=False)
        node = ParallelMapper(
            node,
            RandomSleepUdf(),
            num_workers=4,
            in_order=in_order,
            method=method,
            multiprocessing_context=multiprocessing_context,
            prebatch=prebatch,
        )
        node = Prefetcher(node, prefetch_factor=2)

        results: List[List[dict]] = [[], []]
        for epoch in range(2):
            node.reset()
            for batch in node:
                results[epoch].extend(batch)

        for result in results:
            self.assertEqual(len(result), n, epoch)
            if in_order:
                for i, row in enumerate(result):
                    self.assertEqual(row["step"], i, epoch)
                    self.assertEqual(row["test_tensor"].item(), i, epoch)
                    self.assertEqual(row["test_str"], f"str_{i}", epoch)
            else:
                self.assertEqual({row["step"] for row in result}, set(range(n))), epoch
                self.assertEqual(
                    {row["test_tensor"].item() for row in result},
                    set(range(n)),
                    epoch,
                )
                self.assertEqual(
                    {row["test_str"] for row in result},
                    {f"str_{i}" for i in range(n)},
                    epoch,
                )

    def test_in_order_threads(self):
        self._test_map(True, "thread", None)

    def test_out_of_order_threads(self):
        self._test_map(False, "thread", None)

    def test_in_order_process(self):
        self._test_map(True, "process", None)

    def test_out_of_order_process(self):
        self._test_map(False, "process", None)

    def test_in_order_thread_prebatch(self):
        self._test_map(True, "thread", 3)

    def test_out_of_order_thread_prebatch(self):
        self._test_map(False, "thread", 3)

    def test_in_order_process_prebatch(self):
        self._test_map(True, "process", 3)

    def test_out_of_order_process_prebatch(self):
        self._test_map(False, "process", 3)

    @parameterized.expand(
        itertools.product(
            [0, 7, 13],
            [True],  # TODO: define and fix in_order = False
            [0, 1, 9],  # TODO: define and fix in_order = False
            [None, 3],  # prebatch
        )
    )
    def test_save_load_state_thread(
        self,
        midpoint: int,
        in_order: bool,
        snapshot_frequency: int,
        prebatch: Optional[int],
    ):
        method = "thread"
        batch_size = 6
        n = 80
        src = StatefulRangeNode(n=n)
        node = Batcher(src, batch_size=batch_size, drop_last=False)
        node = ParallelMapper(
            node,
            RandomSleepUdf(),
            num_workers=4,
            in_order=in_order,
            method=method,
            snapshot_frequency=snapshot_frequency,
            prebatch=prebatch,
        )
        node = Prefetcher(node, prefetch_factor=2)
        run_test_save_load_state(self, node, midpoint)

    @parameterized.expand(
        itertools.product(
            [0, 7, 13],
            [True],  # TODO: define and fix in_order = False
            [0, 1, 9],  # TODO: define and fix in_order = False
            [None, 3],  # prebatch
        )
    )
    def test_save_load_state_process(
        self,
        midpoint: int,
        in_order: bool,
        snapshot_frequency: int,
        prebatch: Optional[int],
    ):
        method = "process"
        batch_size = 6
        n = 80
        multiprocessing_context = None if IS_WINDOWS else "forkserver"
        src = StatefulRangeNode(n=n)
        node = Batcher(src, batch_size=batch_size, drop_last=False)
        node = ParallelMapper(
            node,
            RandomSleepUdf(),
            num_workers=4,
            in_order=in_order,
            method=method,
            multiprocessing_context=multiprocessing_context,
            snapshot_frequency=snapshot_frequency,
            prebatch=prebatch,
        )
        node = Prefetcher(node, prefetch_factor=2)
        run_test_save_load_state(self, node, midpoint)

    def test_thread_pool_executor_shutdown_on_del(self):
        """Test that the ThreadPoolExecutor is properly shut down when the iterator is deleted."""
        # Create a ParallelMapper with method="thread"
        src = MockSource(num_samples=10)
        node = ParallelMapper(
            src,
            RandomSleepUdf(),
            num_workers=2,
            method="thread",
        )

        # Reset the node to create the iterator
        node.reset()

        # We need to consume some items to ensure the ThreadPoolExecutor is created
        # and the worker threads are started
        for _ in range(5):
            next(node)

        # Use mock.patch to intercept the ThreadPoolExecutor.shutdown method
        with mock.patch("concurrent.futures.ThreadPoolExecutor.shutdown") as mock_shutdown:
            # Delete the node, which should trigger the shutdown of the ThreadPoolExecutor
            del node

            # Verify that shutdown was called
            mock_shutdown.assert_called()

    def test_thread_pool_executor_shutdown_on_exception(self):
        """Test that the ThreadPoolExecutor is properly shut down when the iterator is deleted."""
        # Create a ParallelMapper with method="thread"
        src = MockSource(num_samples=10)
        node = ParallelMapper(
            src,
            udf_raises,
            num_workers=2,
            method="thread",
        )

        # Reset the node to create the iterator
        node.reset()

        # Use mock.patch to intercept the ThreadPoolExecutor.shutdown method
        with mock.patch("concurrent.futures.ThreadPoolExecutor.shutdown") as mock_shutdown:
            # Consumer the iterator to ensure the ThreadPoolExecutor is created
            # and exception is raised
            try:
                next(node)
            except ValueError:
                pass

            # Verify that shutdown was called
            mock_shutdown.assert_called()

    def test_thread_pool_executor_cancel_futures_shutdown(self):
        """Test that the ThreadPoolExecutor shutdown respects the cancel_futures parameter."""
        src = MockSource(num_samples=10)
        node = ParallelMapper(
            src,
            RandomSleepUdf(),
            num_workers=2,
            method="thread",
        )

        node.reset()
        parallel_mapper_iter = node._it._it  # type: ignore

        for _ in range(5):
            next(node)

        # Test cancel_futures=True
        with mock.patch("concurrent.futures.ThreadPoolExecutor.shutdown") as mock_shutdown:
            parallel_mapper_iter._shutdown(cancel_futures=True)
            mock_shutdown.assert_called_with(wait=True, cancel_futures=True)

        node.reset()
        parallel_mapper_iter = node._it._it  # type: ignore

        for _ in range(5):
            next(node)

        # Test cancel_futures=False
        with mock.patch("concurrent.futures.ThreadPoolExecutor.shutdown") as mock_shutdown:
            parallel_mapper_iter._shutdown(cancel_futures=False)
            mock_shutdown.assert_called_with(wait=True)

    @parameterized.expand(itertools.product([None, 3], [0, 1]))
    def test_explicit_shutdown_thread(self, prebatch, num_workers):
        """Public shutdown must stop actual workers and release the source."""
        source = MockSource(num_samples=20)
        source.shutdown = mock.MagicMock()
        node = ParallelMapper(
            source,
            lambda item: item,
            num_workers=num_workers,
            method="thread",
            prebatch=prebatch,
        )
        previous_threads = set(threading.enumerate())
        node.reset()
        next(node)
        worker_threads = set(threading.enumerate()) - previous_threads
        # Retain the inner mapper solely to clean up if the regression fails.
        inner = node._it if prebatch is None else node._it.source
        try:
            node.shutdown()
            self.assertFalse(any(thread.is_alive() for thread in worker_threads))
            source.shutdown.assert_called_once()
        finally:
            inner.shutdown()

    def test_explicit_shutdown_process(self):
        """Test that the explicit shutdown method properly shuts down the node with process method."""
        multiprocessing_context = None if IS_WINDOWS else "forkserver"

        mock_source = mock.MagicMock()
        mock_source.shutdown = mock.MagicMock()

        node = ParallelMapper(
            mock_source,
            RandomSleepUdf(),
            num_workers=2,
            method="process",
            multiprocessing_context=multiprocessing_context,
        )

        node.reset()

        # Mock the _shutdown method of the iterator
        with mock.patch.object(node._it._it, "_shutdown") as mock_shutdown:
            node.shutdown()
            mock_shutdown.assert_called_once_with(cancel_futures=True)
            mock_source.shutdown.assert_called_once()
