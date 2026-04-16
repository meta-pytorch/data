# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing.synchronize as python_mp_synchronize
import queue
import threading
from typing import Callable, Optional, Union

import torch
import torch.multiprocessing as mp
import torch.utils.data._utils.worker as _worker_module

from torch._utils import ExceptionWrapper

from .constants import QUEUE_TIMEOUT


_thread_local = threading.local()


def get_worker_info() -> Optional[object]:
    """Return a :class:`~torch.utils.data.WorkerInfo` for the current
    :class:`~torchdata.nodes.ParallelMapper` worker, or ``None`` if called
    from outside a worker context.

    Unlike :func:`torch.utils.data.get_worker_info`, this function uses
    thread-local storage and is therefore correct for both thread-based and
    process-based :class:`~torchdata.nodes.ParallelMapper` workers.

    The returned object has the following attributes:

    * ``id`` (int): the worker index (0 to num_workers - 1)
    * ``num_workers`` (int): total number of workers
    * ``seed`` (int): per-worker seed derived from the initial RNG seed
    * ``dataset``: always ``None`` for :class:`~torchdata.nodes.ParallelMapper`

    Returns:
        A ``WorkerInfo`` object, or ``None`` when called from outside a worker.
    """
    return getattr(_thread_local, "worker_info", None)


def _apply_udf(
    worker_id: int,
    in_q: Union[queue.Queue, mp.Queue],
    out_q: Union[queue.Queue, mp.Queue],
    udf: Callable,
    stop_event: Union[threading.Event, python_mp_synchronize.Event],
    num_workers: int,
):
    """_apply_udf assumes in_q emits tuples of (x, idx) where x is the
    payload, idx is the index of the result, potentially used for maintaining
    ordered outputs. For every input it pulls, a tuple (y, idx) is put on the out_q
    where the output of udf(x), an ExceptionWrapper, or StopIteration (if it pulled
    StopIteration from in_q).

    Sets up worker info before entering the processing loop so that
    :func:`torchdata.nodes.get_worker_info` returns a valid
    :class:`~torch.utils.data.WorkerInfo` from inside the UDF. For process
    workers, :func:`torch.utils.data.get_worker_info` also works because each
    process has its own memory space. For thread workers, prefer
    :func:`torchdata.nodes.get_worker_info` which uses thread-local storage.
    """
    torch.set_num_threads(1)
    seed = torch.initial_seed() + worker_id
    worker_info = _worker_module.WorkerInfo(id=worker_id, num_workers=num_workers, seed=seed, dataset=None)
    # Thread-local: always returns the correct info for this worker, regardless of
    # whether other workers (threads) have set their own worker info concurrently.
    _thread_local.worker_info = worker_info
    # Module-level global: correct for process workers (isolated memory); for thread
    # workers this may race, so callers should use torchdata.nodes.get_worker_info().
    _worker_module._worker_info = worker_info

    while True:
        if stop_event.is_set() and in_q.empty():
            break

        try:
            item, idx = in_q.get(block=True, timeout=QUEUE_TIMEOUT)
        except queue.Empty:
            continue

        if isinstance(item, ExceptionWrapper):
            out_q.put((item, idx), block=False)
        elif isinstance(item, StopIteration):
            out_q.put((item, idx), block=False)
        else:
            try:
                y = udf(item)
            except Exception:
                y = ExceptionWrapper(where="in _apply_udf")

            out_q.put((y, idx), block=False)
