# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.constants import QUEUE_TIMEOUT
from torchdata.nodes.exception_wrapper import StartupExceptionWrapper
from torchdata.nodes.snapshot_store import QueueSnapshotStore


class TestQueueSnapshotStore(TestCase):
    def test_snapshot_store(self) -> None:
        for _ in range(100):
            store = QueueSnapshotStore()
            store.append({"a": 1}, 0)
            store.append({"a": 2}, 10)

            self.assertEqual(len(store._q.queue), 2)

            val = store.pop_version(0)
            self.assertEqual(val, {"a": 1})
            self.assertEqual(len(store._q.queue), 1)
            val = store.pop_version(1)
            self.assertIsNone(val)
            self.assertEqual(len(store._q.queue), 1)
            val = store.pop_version(7)
            self.assertIsNone(val)
            self.assertEqual(len(store._q.queue), 1)
            val = store.pop_version(10)
            self.assertEqual(val, {"a": 2})
            self.assertEqual(len(store._q.queue), 0)

            val = store.pop_version(11)
            self.assertIsNone(val)
            self.assertEqual(len(store._q.queue), 0)

            with self.assertRaisesRegex(ValueError, "is not strictly greater than"):
                store.append({"a": 3}, 3)

            self.assertEqual(len(store._q.queue), 0)

            with self.assertRaisesRegex(ValueError, "is not strictly greater than"):
                store.append({"a": 4}, 10)
            self.assertEqual(len(store._q.queue), 0)

            store.append({"a": 4}, 11)
            store.append({"a": 5}, 19)
            val = store.pop_version(19)
            self.assertEqual(val, {"a": 5})
            self.assertEqual(len(store._q.queue), 0)

    def test_init_error(self) -> None:
        for _ in range(10):
            store = QueueSnapshotStore()
            sleep_time = 0.1
            thread = threading.Thread(target=_worker_init_error, args=(store, sleep_time))
            thread.start()
            with self.assertRaisesRegex(RuntimeError, "Test Startup Exception"):
                store.get_initial_snapshot(thread, sleep_time)
            thread.join()

    def test_timeout_error(self) -> None:
        for _ in range(10):
            store = QueueSnapshotStore()
            sleep_time = 0.1
            thread = threading.Thread(target=_worker_raises_after, args=(sleep_time,))
            thread.start()
            with self.assertRaisesRegex(RuntimeError, "Failed to get initial snapshot"):
                store.get_initial_snapshot(thread, sleep_time * 0.1)
            thread.join()

    def test_thread_dead_error(self) -> None:
        # Test when thread is alive for longer than QUEUE_TIMEOUT but dies afterwards
        for _ in range(10):  # Should be reliable
            store = QueueSnapshotStore()
            thread = threading.Thread(target=_worker_raises_after, args=(QUEUE_TIMEOUT * 3.0,))
            thread.start()
            with self.assertRaisesRegex(RuntimeError, r"thread.is_alive\(\)=False"):
                store.get_initial_snapshot(thread, QUEUE_TIMEOUT * 10.0)
            thread.join()

    def test_future_dead_error(self) -> None:
        # Test when thread is alive for longer than QUEUE_TIMEOUT but dies afterwards
        for _ in range(10):  # Should be reliable
            store = QueueSnapshotStore()
            pool = ThreadPoolExecutor()
            future = pool.submit(_worker_raises_after, QUEUE_TIMEOUT * 3.0)
            with self.assertRaisesRegex(RuntimeError, r"thread.is_alive\(\)=False"):
                store.get_initial_snapshot(future, QUEUE_TIMEOUT * 5.0)
            pool.shutdown()


def _worker_init_error(store, sleep_time):
    try:
        raise RuntimeError("Test Startup Exception")
    except Exception as e:
        e = StartupExceptionWrapper(where="_worker_init_error")
        store.append_initial_snapshot(e)
    time.sleep(sleep_time)


def _worker_raises_after(sleep_time):
    time.sleep(sleep_time)
    raise RuntimeError(f"Thread dying {sleep_time=}")
