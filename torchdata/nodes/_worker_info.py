# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
from typing import Any, Optional

import torch
import torch.utils.data._utils.worker as _worker_module  # type: ignore[import]

_thread_local = threading.local()


def get_worker_info() -> Optional[Any]:
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
    * ``dataset``: always ``None`` — :class:`~torchdata.nodes.ParallelMapper`
      operates on items, not datasets

    Returns:
        A ``WorkerInfo`` object, or ``None`` when called from outside a worker.
    """
    return getattr(_thread_local, "worker_info", None)


def _set_worker_info(worker_id: int, num_workers: int) -> None:
    """Set up WorkerInfo for the current thread/process so that
    :func:`get_worker_info` and :func:`torch.utils.data.get_worker_info`
    return valid info from inside a UDF.

    ``dataset`` is ``None`` because :class:`~torchdata.nodes.ParallelMapper`
    maps over arbitrary items rather than a PyTorch Dataset object.
    """
    seed = torch.initial_seed() + worker_id
    worker_info = _worker_module.WorkerInfo(id=worker_id, num_workers=num_workers, seed=seed, dataset=None)  # type: ignore[attr-defined,arg-type]
    # Thread-local storage: always returns the correct info for this worker
    # even when multiple thread-workers run concurrently in the same process.
    _thread_local.worker_info = worker_info
    # Module-level global: works correctly for process workers (each process
    # has isolated memory). For thread workers this may race between workers,
    # so thread-based callers should use torchdata.nodes.get_worker_info().
    _worker_module._worker_info = worker_info  # type: ignore[attr-defined]
