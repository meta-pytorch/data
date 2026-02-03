# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import queue
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeVar, Union

import torch

from torch._utils import ExceptionWrapper
from torch.utils.data._utils import HAS_NUMPY, MP_STATUS_CHECK_INTERVAL, signal_handling

from torch.utils.data._utils.worker import (
    _generate_state,
    _IterableDatasetStopIteration,
    _ResumeIteration,
    ManagerWatchdog,
    WorkerInfo,
)

from .incremental_state import (
    _DATASET_ITER_STATE,
    _DATASET_STATE,
    _FETCHER_ENDED,
    _FETCHER_STATE,
    _IncrementalWorkerState,
    _WORKER_ID,
)

from .stateful import Stateful


T = TypeVar("T")


def try_to_serialize(obj: Any) -> Union[dict, None]:
    if isinstance(obj, Stateful):
        obj_state = obj.state_dict()
    else:
        obj_state = None

    return obj_state


def try_to_deserialize(obj: T, state_dict: dict) -> T:
    if isinstance(obj, Stateful):
        obj.load_state_dict(state_dict)
        return obj  # type: ignore[return-value]
    return obj


@dataclass(frozen=True)
class _AckStartup:
    """Dummy class used to ack startup and return state at time 0"""

    worker_id: int
    initial_state: Optional[Union[Dict[str, Any], ExceptionWrapper]]
    is_delta: bool = False


def _worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_seed,
    worker_state,
):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        seed = base_seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
        if HAS_NUMPY:
            np_seed = _generate_state(base_seed, worker_id)
            import numpy as np

            np.random.seed(np_seed)

        from torch.utils.data import IterDataPipe
        from torch.utils.data.graph_settings import apply_random_seed

        shared_rng = torch.Generator()
        if isinstance(dataset, IterDataPipe):
            assert shared_seed is not None
            shared_rng.manual_seed(shared_seed)
            dataset = apply_random_seed(dataset, shared_rng)

        torch.utils.data._utils.worker._worker_info = WorkerInfo(
            id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset, rng=shared_rng
        )

        from torch.utils.data import _DatasetKind

        # See NOTE [ Incremental worker state ]
        incremental_worker_state: _IncrementalWorkerState
        init_exception = None
        fetcher = None
        initial_state = None
        is_delta = False
        try:
            if init_fn is not None:
                init_fn(worker_id)

            if worker_state is None:
                fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                initial_state = _make_state_dict(worker_id, dataset_kind, fetcher, dataset)
                incremental_worker_state = _IncrementalWorkerState(initial_state)
            else:
                # Always restore in this order:
                #  1. try to restore dataset state
                #  2. generate dataset iterator
                #  3. try to restore iterator state
                incremental_worker_state = _IncrementalWorkerState(worker_state)
                if worker_state[_DATASET_STATE] is not None:
                    dataset = try_to_deserialize(dataset, worker_state[_DATASET_STATE])
                fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                if worker_state[_FETCHER_STATE] is not None:
                    if dataset_kind == _DatasetKind.Iterable:
                        if worker_state[_FETCHER_STATE][_DATASET_ITER_STATE] is not None:
                            dataset_iter = try_to_deserialize(
                                fetcher.dataset_iter,
                                worker_state[_FETCHER_STATE][_DATASET_ITER_STATE],
                            )
                            if dataset_iter is not None:
                                fetcher.dataset_iter = dataset_iter
                        # We always force fetcher to request at least one batch even if
                        # we know it will lead to immediate stop iteration
                        fetcher.ended = False
                iteration_end = False
                initial_state = incremental_worker_state.generate_delta(
                    _make_state_dict(worker_id, dataset_kind, fetcher, dataset)
                )
                is_delta = True

                del worker_state
        except Exception:
            init_exception = ExceptionWrapper(where=f"in DataLoader worker process {worker_id}")

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, _AckStartup):
                # Send ack and initial state to the main process
                data_queue.put(
                    (
                        r,
                        _AckStartup(
                            worker_id=worker_id, initial_state=init_exception or initial_state, is_delta=is_delta
                        ),
                    )
                )
                del initial_state
                del is_delta
                continue
            elif isinstance(r, _ResumeIteration):
                iteration_end = False

                if isinstance(dataset, IterDataPipe):
                    assert r.seed is not None
                    shared_rng.manual_seed(r.seed)
                    dataset = apply_random_seed(dataset, shared_rng)

                try:
                    # Recreate the fetcher for worker-reuse policy
                    fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                    # see NOTE [ Incremental Worker State ]
                    initial_state = _make_state_dict(worker_id, dataset_kind, fetcher, dataset)
                    incremental_worker_state = _IncrementalWorkerState(initial_state)
                except Exception:
                    init_exception = ExceptionWrapper(where=f"in DataLoader worker process {worker_id}")

                # Acknowledge the main process
                data_queue.put((r, _AckStartup(worker_id=worker_id, initial_state=init_exception or initial_state)))
                del initial_state
                continue
            elif r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, (index, snapshot) = r
            data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
            delta_state_dict = None
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    try:
                        data = fetcher.fetch(index)  # type: ignore[union-attr]
                    except StopIteration:
                        if not dataset_kind == _DatasetKind.Iterable:
                            raise
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    if snapshot or iteration_end:
                        # Generate incremental diff from prev_state_dict and current_state_dict
                        state_dict = _make_state_dict(worker_id, dataset_kind, fetcher, dataset)
                        delta_state_dict = incremental_worker_state.generate_delta(state_dict)
                        del state_dict
                except Exception:
                    # It is important that we don't store exc_info in a variable.
                    # `ExceptionWrapper` does the correct thing.
                    # See NOTE [ Python Traceback Reference Cycle Problem ]
                    data = ExceptionWrapper(where=f"in DataLoader worker process {worker_id}")

            data_queue.put((idx, (data, worker_id, delta_state_dict)))
            del data, idx, index, r, delta_state_dict  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()


def _make_state_dict(worker_id, dataset_kind, fetcher, dataset) -> Dict[str, Any]:
    from torch.utils.data import _DatasetKind

    if dataset_kind == _DatasetKind.Iterable:
        fetcher_state = {
            _DATASET_ITER_STATE: try_to_serialize(fetcher.dataset_iter),
            _FETCHER_ENDED: fetcher.ended,
        }
        dataset_state = None
        if fetcher.dataset_iter is not fetcher.dataset:
            dataset_state = try_to_serialize(fetcher.dataset)
    else:
        fetcher_state = None
        # Pick up any user-defined dataset state
        dataset_state = try_to_serialize(dataset)

    return {
        _WORKER_ID: worker_id,
        _FETCHER_STATE: fetcher_state,
        _DATASET_STATE: dataset_state,
    }
