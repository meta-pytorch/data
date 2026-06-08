# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from torchdata.nodes import BaseNode, T

from torchdata.nodes.map import _SingleThreadedMapper

from ._populate_queue import _populate_queue


class Prefetcher(BaseNode[T]):
    """Prefetches data from the source node and stores it in a queue.

    Args:
        source (BaseNode[T]): The source node to prefetch data from.
        prefetch_factor (int): The number of items to prefetch ahead of time.
        snapshot_frequency (int): The frequency at which to snapshot the state of the source node. Default is
            1, which means that the state of the source node will be snapshotted after every item. If set
            to a higher value, the state of the source node will be snapshotted after every snapshot_frequency
            items.
    """

    def __init__(self, source: BaseNode[T], prefetch_factor: int, snapshot_frequency: int = 1):
        super().__init__()
        self.source = source
        self.prefetch_factor = prefetch_factor
        self.snapshot_frequency = snapshot_frequency
        self._it: Optional[_SingleThreadedMapper[T]] = None

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if hasattr(self, "_it") and self._it is not None:
            self._it._shutdown()
            del self._it

        # This can throw, so _it may be deleted
        self._it = _SingleThreadedMapper(
            source=self.source,
            prefetch_factor=self.prefetch_factor,
            worker=_populate_queue,
            snapshot_frequency=self.snapshot_frequency,
            initial_state=initial_state,
        )

    def next(self):
        return next(self._it)  # type: ignore[arg-type]

    def get_state(self) -> Dict[str, Any]:
        return self._it.get_state()  # type: ignore[union-attr]

    def shutdown(self):
        if self._it is not None:
            self._it._shutdown()
        if hasattr(self.source, "shutdown"):
            self.source.shutdown()
