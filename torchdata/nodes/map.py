# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import queue
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Generic, Iterator, List, Literal, Optional, Protocol, Sequence, TypeVar, Union

import torch.multiprocessing as mp
from torchdata.nodes.base_node import BaseNode, T
from torchdata.nodes.batch import Batcher, Unbatcher
from torchdata.nodes.exception_wrapper import ExceptionWrapper, StartupExceptionWrapper
from torchdata.nodes.snapshot_store import QueueSnapshotStore, SnapshotStore

from ._apply_udf import _apply_udf

from ._populate_queue import _populate_queue

from .constants import QUEUE_TIMEOUT

ACK_TIMEOUT = 300  # Timeout after 5 minutes


# We define this protocol for type checking
class _MultiprocessContext(Protocol):
    def Process(self, *args, **kwargs):
        ...

    def Event(self, *args, **kwargs):
        ...

    def Queue(self, *args, **kwargs):
        ...


X = TypeVar("X")


def Mapper(source: BaseNode[X], map_fn: Callable[[X], T]) -> "ParallelMapper[T]":
    """Returns a :class:`ParallelMapper` node with num_workers=0, which will execute map_fn in the current process/thread.

    Args:
        source (BaseNode[X]): The source node to map over.
        map_fn (Callable[[X], T]): The function to apply to each item from the source node.
    """
    return ParallelMapper(
        source=source,
        map_fn=map_fn,
        num_workers=0,
    )


Xseq = Sequence[X]
Tseq = Sequence[T]


class MapOverBatch(Generic[X, T]):
    def __init__(self, map_fn: Callable[[X], T]):
        self.map_fn = map_fn

    def __call__(self, xlist: Sequence[X]) -> Sequence[T]:
        return [self.map_fn(x) for x in xlist]


def _sort_worker(
    in_q: Union[queue.Queue, mp.Queue],
    out_q: queue.Queue,
    stop_event: threading.Event,
):
    buffer: Dict[int, Any] = {}
    cur_idx = 0
    while not stop_event.is_set():
        try:
            item, idx = in_q.get(block=True, timeout=QUEUE_TIMEOUT)
        except queue.Empty:
            continue
        if idx == cur_idx:
            out_q.put((item, cur_idx), block=False)
            cur_idx += 1
        else:
            if idx in buffer:
                # This is the easiest way to create an exception wrapper
                try:
                    raise ValueError(f"Duplicate index {idx=}, {buffer.keys()=}, {item=}")
                except Exception:
                    item = ExceptionWrapper(where="in _sort_worker")
                out_q.put((item, idx), block=False)
                break
            buffer[idx] = item
        while cur_idx in buffer:
            out_q.put((buffer.pop(cur_idx), cur_idx), block=False)
            cur_idx += 1


class _InlineMapperIter(Iterator[T]):
    """Non-Parallel implementation of Mapper"""

    SOURCE_KEY = "source"

    def __init__(
        self,
        source: BaseNode[X],
        map_fn: Callable[[X], T],
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.map_fn = map_fn
        if initial_state is not None:
            self.source.reset(initial_state[self.SOURCE_KEY])
        else:
            self.source.reset()

    def __next__(self):
        return self.map_fn(next(self.source))

    def get_state(self) -> Dict[str, Any]:
        return {self.SOURCE_KEY: self.source.state_dict()}

    def _shutdown(self, cancel_futures=False):
        pass


class _ParallelMapperIter(Iterator[T]):
    """_ParallelMapperIter will start at least two threads, one running
    _populate_queue, and one for _apply_udf. If in_order == True, a
    third thread will be started to read from _apply_udf's result q
    and block the output_q until the appropriate in_order element is available,
    buffering outputs as needed.

    A BoundedSemaphore with initial value max_concurrent will limit the number
    of items in flight, and in all of the queues.
    """

    def __init__(
        self,
        source: BaseNode[X],
        map_fn: Callable[[X], T],
        num_workers: int,
        in_order: bool,
        method: Literal["thread", "process"],
        mp_context: _MultiprocessContext,
        max_concurrent: Optional[int],
        snapshot_frequency: int,
        initial_state: Optional[Dict[str, Any]],
    ):
        self.source = source
        self.map_fn = map_fn
        self.num_workers = num_workers
        self.in_order = in_order
        self.method = method
        self.mp_context = mp_context
        self.snapshot_frequency = snapshot_frequency

        self._in_q: Union[queue.Queue, mp.Queue] = queue.Queue() if method == "thread" else mp_context.Queue()
        self._intermed_q: Union[queue.Queue, mp.Queue] = queue.Queue() if method == "thread" else mp_context.Queue()
        self._max_tasks = 2 * self.num_workers if max_concurrent is None else max_concurrent
        self._sem = threading.BoundedSemaphore(value=self._max_tasks)

        self._done = False

        self._stop = threading.Event()
        self._mp_stop = mp_context.Event()

        # This ensures that the stop events are set before the iterator is garbage collected.
        # Theading's _register_atexit() are called before regular atexit handlers and
        # before threads are joined.
        threading._register_atexit(self._set_stop_events)  # type: ignore[attr-defined]

        self._steps_since_snapshot = 0
        fast_forward = 0
        if initial_state is not None:
            self._snapshot = initial_state["snapshot"]
            fast_forward = initial_state["steps_since_snapshot"]
            self.source.reset(self._snapshot)
        else:
            self._snapshot = None
            self.source.reset()
        self._snapshot_store = QueueSnapshotStore()

        self.pool = ThreadPoolExecutor(self.num_workers + 2)
        _read_future = self.pool.submit(
            _populate_queue,
            self.source,
            self._in_q,
            self._snapshot_store,
            self.snapshot_frequency,
            self._sem,
            self._stop,
        )

        if self.method == "thread":
            for worker_id in range(self.num_workers):
                self.pool.submit(
                    _apply_udf,
                    worker_id,
                    self._in_q,
                    self._intermed_q,
                    self.map_fn,
                    self._stop,
                )

        elif self.method == "process":
            self._workers: List[mp.Process] = []
            for worker_id in range(self.num_workers):
                _args = (
                    worker_id,
                    self._in_q,
                    self._intermed_q,
                    self.map_fn,
                    self._mp_stop,
                )
                self._workers.append(mp_context.Process(target=_apply_udf, args=_args, daemon=True))
            for t in self._workers:
                t.start()

        self._out_q = self._intermed_q
        if self.in_order:
            self._sort_q: queue.Queue = queue.Queue()
            self.pool.submit(
                _sort_worker,
                self._intermed_q,
                self._sort_q,
                self._stop,
            )
            self._out_q = self._sort_q

        time.sleep(0.01)
        self._snapshot = self._snapshot_store.get_initial_snapshot(thread=_read_future, timeout=ACK_TIMEOUT)

        for i in range(fast_forward):
            try:
                next(self)
            except StopIteration:
                raise ValueError(
                    f"Tried to fast-forward {fast_forward} items during init but "
                    f"hit StopIteration after {i} items, this is likely a bug or malformed state_dict"
                )

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        while True:
            if self._stop.is_set() or self._mp_stop.is_set():
                self._shutdown()
                raise StopIteration()
            elif self._done and self._sem._value == self._max_tasks:
                # _done is set, and semaphore is back at initial value, so we can stop
                self._stop.set()
                self._mp_stop.set()
                self._shutdown()
                raise StopIteration()
            try:
                item, idx = self._out_q.get(block=True, timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            if isinstance(item, StopIteration):
                self._done = True
                self._sem.release()
                # Make sure queues are flushed before returning early
                continue
            elif isinstance(item, ExceptionWrapper):
                if not isinstance(item, StartupExceptionWrapper):
                    self._sem.release()
                    self._shutdown()
                item.reraise()

            self._steps_since_snapshot += 1
            self._sem.release()
            self._maybe_update_snapshot(idx)
            return item

    def get_state(self) -> Dict[str, Any]:
        return {
            "snapshot": self._snapshot,
            "steps_since_snapshot": self._steps_since_snapshot,
        }

    def _maybe_update_snapshot(self, idx: int):
        if (snapshot := self._snapshot_store.pop_version(idx)) is not None:
            self._snapshot = snapshot
            self._steps_since_snapshot = 0

    def __del__(self):
        try:
            self._shutdown()
        except Exception:
            pass

    def _shutdown(self, cancel_futures=False):
        self._stop.set()
        self._mp_stop.set()
        if hasattr(self, "pool"):
            if cancel_futures:
                # Wait for all threads to finish before returning, but cancel any
                # futures that are pending. This is used when calling _shutdown()
                # from downstream shutdown()
                self.pool.shutdown(wait=True, cancel_futures=True)
            else:
                # Wait for all threads to finish before returning.
                # This is used when calling _shutdown() from __del__ or a reset()
                self.pool.shutdown(wait=True)
        if hasattr(self, "_workers"):
            for t in self._workers:
                if t.is_alive():
                    t.join(timeout=QUEUE_TIMEOUT * 5)

    def _set_stop_events(self):
        try:
            if isinstance(self._in_q, queue.Queue):
                with self._in_q.mutex:
                    self._in_q.queue.clear()
            self._stop.set()
        except Exception:
            pass


class _ParallelMapperImpl(BaseNode[T]):
    """This class implements _ParallelMapperIter and _InlineMapperIter as a BaseNode,
    allowing them to be composed with other BaseNodes.

    TODO: In the future, this class may go away once we implement reset() on
    _ParallelMapperIter and _InlineMapperIter themselves so we don't need this
    additional level of abstraction.
    """

    def __init__(
        self,
        source: BaseNode[X],
        map_fn: Callable[[X], T],
        num_workers: int,
        in_order: bool = True,
        method: Literal["thread", "process"] = "thread",
        multiprocessing_context: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        snapshot_frequency: int = 1,
    ):
        super().__init__()
        assert method in ["thread", "process"]
        self.source = source
        self.map_fn = map_fn
        self.num_workers = num_workers
        self.in_order = in_order
        self.method = method
        self.multiprocessing_context = multiprocessing_context
        self._mp_context: Any = mp
        if self.method == "process" and self.multiprocessing_context is not None:
            self._mp_context = mp.get_context(self.multiprocessing_context)

        if max_concurrent is not None and max_concurrent <= 0:
            raise ValueError(f"{max_concurrent=} should be a positive integer!")
        self.max_concurrent = max_concurrent
        self.snapshot_frequency = snapshot_frequency
        self._it: Optional[Union[_InlineMapperIter[T], _ParallelMapperIter[T]]] = None

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if self._it is not None:
            self._it._shutdown()
            del self._it

        if self.num_workers > 0:
            self._it = self._parallel_reset(initial_state)
        else:
            self._it = self._inline_reset(initial_state)

    def _inline_reset(self, initial_state: Optional[Dict[str, Any]]):
        return _InlineMapperIter(source=self.source, map_fn=self.map_fn, initial_state=initial_state)

    def _parallel_reset(self, initial_state: Optional[Dict[str, Any]]):
        return _ParallelMapperIter(
            source=self.source,
            map_fn=self.map_fn,
            num_workers=self.num_workers,
            in_order=self.in_order,
            method=self.method,
            mp_context=self._mp_context,
            max_concurrent=self.max_concurrent,
            snapshot_frequency=self.snapshot_frequency,
            initial_state=initial_state,
        )

    def next(self) -> T:
        return next(self._it)  # type: ignore[arg-type, union-attr]

    def get_state(self) -> Dict[str, Any]:
        return self._it.get_state()  # type: ignore[union-attr]

    def shutdown(self):
        if hasattr(self, "_it") and self._it is not None:
            self._it._shutdown(cancel_futures=True)

    def __del__(self):
        self.shutdown()


class ParallelMapper(BaseNode[T]):
    """ParallelMapper executes map_fn in parallel either in num_workers threads or
    processes. For processes, multiprocessing_context can be spawn, forkserver, fork,
    or None (chooses OS default). At most max_concurrent items will be either processed
    or in the iterator's output queue, to limit CPU and Memory utilization. If None
    (default) the value will be 2 * num_workers.

    At most one iter() is created from source, and at most one thread will call
    next() on it at once.

    If in_order is true, the iterator will return items in the order from which they arrive
    from source's iterator, potentially blocking even if other items are available.

    Args:
        source (BaseNode[X]): The source node to map over.
        map_fn (Callable[[X], T]): The function to apply to each item from the source node.
        num_workers (int): The number of workers to use for parallel processing.
        in_order (bool): Whether to return items in the order from which they arrive from. Default is True.
        method (Literal["thread", "process"]): The method to use for parallel processing. Default is "thread".
        multiprocessing_context (Optional[str]): The multiprocessing context to use for parallel processing. Default is None.
        max_concurrent (Optional[int]): The maximum number of items to process at once. Default is None.
        snapshot_frequency (int): The frequency at which to snapshot the state of the source node. Default is 1.
        prebatch (Optional[int]): Optionally perform pre-batching of items from source before mapping.
          For small items, this may improve throughput at the expense of peak memory.
    """

    IT_STATE_KEY = "it_state"

    def __init__(
        self,
        source: BaseNode[X],
        map_fn: Callable[[X], T],
        num_workers: int,
        in_order: bool = True,
        method: Literal["thread", "process"] = "thread",
        multiprocessing_context: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        snapshot_frequency: int = 1,
        prebatch: Optional[int] = None,
    ):
        super().__init__()
        assert method in ["thread", "process"]
        self.num_workers = num_workers
        self.in_order = in_order
        self.method = method
        self.multiprocessing_context = multiprocessing_context
        if max_concurrent is not None and max_concurrent <= 0:
            raise ValueError(f"{max_concurrent=} should be a positive integer!")
        self.max_concurrent = max_concurrent
        self.snapshot_frequency = snapshot_frequency
        self.prebatch = prebatch
        if prebatch is None:
            self.map_fn = map_fn
            self.source = source
        else:
            if prebatch <= 0:
                raise ValueError(f"{prebatch=} must be a positive integer!")
            self.map_fn = MapOverBatch(map_fn=map_fn)  # type: ignore[assignment]
            self.source = Batcher(source, batch_size=prebatch, drop_last=False)  # type: ignore[assignment]

        _it = _ParallelMapperImpl(
            source=self.source,
            map_fn=self.map_fn,
            num_workers=self.num_workers,
            in_order=self.in_order,
            method=self.method,
            multiprocessing_context=self.multiprocessing_context,
            max_concurrent=self.max_concurrent,
            snapshot_frequency=self.snapshot_frequency,
        )

        if self.prebatch is None:
            self._it = _it
        else:
            self._it = Unbatcher(_it)  # type: ignore[arg-type, assignment]

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self._it.reset(initial_state[self.IT_STATE_KEY])
        else:
            self._it.reset()

    def next(self) -> T:
        return next(self._it)  # type: ignore[arg-type, union-attr]

    def get_state(self) -> Dict[str, Any]:
        return {self.IT_STATE_KEY: self._it.state_dict()}  # type: ignore[union-attr]

    def shutdown(self):
        self._it.shutdown()
        if hasattr(self.source, "shutdown"):
            self.source.shutdown()


_WorkerType = Callable[
    [
        BaseNode,
        queue.Queue,
        SnapshotStore,
        int,
        threading.BoundedSemaphore,
        threading.Event,
    ],
    None,
]


class _SingleThreadedMapper(Iterator[T]):
    """Utility Iterator for performing mapping with a single thread.
    Because only a single thread is used, we don't need an input queue to guard
    against multiple threads reading from the same iterator. This is used for
    Prefetcher and PinMemory.

    A thread is started on __init__ and stopped on __del__/_shutdown.
    The thread runs worker, which acquires a BoundedSemaphore with initial value
    of `prefetch_factor`.

    When next() is called on this iterator, it will block until an item is available on _q.
    Next will perform the following depending on what is pulled from the q:
    - StopIteration: raise StopIteration. Any subsequent next() calls will also raise StopIteration
    - ExceptionWrapper: call reraise() on the exception wrapper
    - any other item: return the item

    A Bounded semaphore is used to limit concurrency and memory utilization.
    If N items have been pulled from the source (i.e. acquire the semaphore),
    and M items have been yielded by this iterator (i.e. release the semaphore),
    we maintain the invariant that semaphore.value + (M - N) == prefetch_factor (modulo
    non-atomicness of operations).

    worker calls semaphore.acquire. When we pull an item from the queue, we
    call semaphore.release (unless it's a StartupExceptionWrapper, because worker
    does not acquire sempahores in this case). All outstanding items are either being
    processed in worker, in the _q, or about to be returned by an in-flight next() call.
    """

    def __init__(
        self,
        source: BaseNode[T],
        prefetch_factor: int,
        worker: _WorkerType,
        snapshot_frequency: int,
        initial_state: Optional[Dict[str, Any]],
    ):
        self.source = source
        self.prefetch_factor = prefetch_factor
        self.worker = worker
        self.snapshot_frequency = snapshot_frequency

        self._q: queue.Queue = queue.Queue()
        self._sem = threading.BoundedSemaphore(value=prefetch_factor)
        self._stop_event = threading.Event()

        self._steps_since_snapshot = 0
        self._fast_forward = 0
        if initial_state is not None:
            self._snapshot = initial_state["snapshot"]
            self._fast_forward = initial_state["steps_since_snapshot"]
            self.source.reset(self._snapshot)
        else:
            self._snapshot = None
            self.source.reset()
        self._snapshot_store = QueueSnapshotStore()
        self._thread = threading.Thread(
            target=self.worker,
            args=(
                self.source,
                self._q,
                self._snapshot_store,
                self.snapshot_frequency,
                self._sem,
                self._stop_event,
            ),
            daemon=True,
            name=f"worker_thread(target={self.worker.__name__})",
        )
        self._thread.start()

        # Try and get initial snapshot
        self._snapshot = self._snapshot_store.get_initial_snapshot(thread=self._thread, timeout=ACK_TIMEOUT)

        for i in range(self._fast_forward):
            try:
                next(self)
            except StopIteration:
                raise ValueError(
                    f"Tried to fast-forward {self._fast_forward} items during init but "
                    f"hit StopIteration after {i} items, this is likely a bug or malformed state_dict"
                )
        self._fast_forward = 0

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        while True:
            if self._stop_event.is_set():
                self._shutdown()
                raise StopIteration()
            try:
                item, idx = self._q.get(block=True, timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            if isinstance(item, StopIteration):
                self._sem.release()
                self._stop_event.set()
                self._shutdown()
                raise item
            elif isinstance(item, ExceptionWrapper):
                if not isinstance(item, StartupExceptionWrapper):
                    # We don't need to release for startup exceptions
                    self._sem.release()
                    self._shutdown()
                self._stop_event.set()
                item.reraise()
            else:
                self._sem.release()
                self._steps_since_snapshot += 1
                self._maybe_update_snapshot(idx)
                return item

    def get_state(self) -> Dict[str, Any]:
        return {
            "snapshot": self._snapshot,
            "steps_since_snapshot": self._steps_since_snapshot,
        }

    def _maybe_update_snapshot(self, idx: int):
        if (snapshot := self._snapshot_store.pop_version(idx)) is not None:
            self._snapshot = snapshot
            self._steps_since_snapshot = 0

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        self._stop_event.set()
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=QUEUE_TIMEOUT * 5)
