# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import queue
import threading
from typing import Any, Dict, Optional, Union

import torch.multiprocessing as mp

from torchdata.nodes.base_node import BaseNode

from torchdata.nodes.exception_wrapper import ExceptionWrapper, StartupExceptionWrapper
from torchdata.nodes.snapshot_store import MonotonicIndex, SnapshotStore

from .constants import QUEUE_TIMEOUT


def _populate_queue(
    source: BaseNode,
    q: Union[queue.Queue, mp.Queue],
    snapshot_store: SnapshotStore,
    snapshot_frequency: int,
    semaphore: threading.BoundedSemaphore,
    stop_event: threading.Event,
):
    """_populate_queue calls `iter(source)` to get an iterator `it`, waits for semaphore.acquire,
    and puts its outputs onto q. It never releases the sempahore. It continues to put items on the
    q as long as it can acquire the sempahore, stop_event is not set, and StopIteration has not
    been thrown by the `it`.

    This function will always put tuples of (x, idx) on the q where idx
    starts from 0 and is monotonically increasing. x may be the output of next(it), StopIteration,
    or an ExceptionWrapper.

    If there is an exception raised during the call to `iter(source)`, this function does not
    wait to acquire sempahore before putting StartupExceptionWrapper on q.

    Note: this is only intended to be used by a single thread at once. Each instance
    creates its own iter for source so if this is called with multiple threads, you may get
    duplicates if source is not sharded properly.
    """

    # Include a monotonic index starting from 0 to each item in the queue
    idx = MonotonicIndex()

    def _put(
        item,
        block: bool = True,
        snapshot: Optional[Union[Dict[str, Any], StartupExceptionWrapper]] = None,
    ):
        _idx = idx.get()
        if snapshot:
            snapshot_store.append(snapshot=snapshot, version=_idx)
        q.put((item, _idx), block=block, timeout=1.0 if block else None)

    try:
        assert (
            isinstance(snapshot_frequency, int) and snapshot_frequency >= 0
        ), f"snapshot_frequency must be non-negative integer! Got {snapshot_frequency}"
        snapshot = source.state_dict()
        if snapshot is not None:
            snapshot = copy.deepcopy(snapshot)
        snapshot_store.append_initial_snapshot(snapshot=snapshot)
    except Exception:
        e = StartupExceptionWrapper(where="in _populate_queue startup for device")
        snapshot_store.append_initial_snapshot(snapshot=e)
        return

    yielded = 0
    while not stop_event.is_set():
        if not semaphore.acquire(blocking=True, timeout=QUEUE_TIMEOUT):
            continue
        try:
            item = next(source)  # FIXME: This may hang!
            yielded += 1
            snapshot = None
            if snapshot_frequency > 0 and yielded % snapshot_frequency == 0:
                snapshot = source.state_dict()
                if snapshot is not None:
                    snapshot = copy.deepcopy(snapshot)
            _put(item, block=False, snapshot=snapshot)
        except StopIteration as e:
            _put(e, block=False)
            break
        except Exception:
            item = ExceptionWrapper(where="in _populate_queue")
            _put(item, block=False)
            break
