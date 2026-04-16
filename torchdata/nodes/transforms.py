# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Sequence


class Compose:
    """Composes multiple callables into a single callable that applies them sequentially.

    Useful for fusing multiple mapping functions into a single :class:`Mapper` or
    :class:`ParallelMapper` node instead of chaining multiple nodes.

    Args:
        fns (Sequence[Callable]): Sequence of callables to apply in order.
            Each callable receives the output of the previous one as its input.

    Raises:
        ValueError: If ``fns`` is empty.

    Example::

        >>> from torchdata.nodes import Mapper, Compose
        >>> def normalize(x): return x / 255.0
        >>> def to_float(x): return float(x)
        >>> node = Mapper(source, Compose([to_float, normalize]))
    """

    def __init__(self, fns: Sequence[Callable[[Any], Any]]) -> None:
        if not fns:
            raise ValueError("Compose requires at least one function, got an empty sequence.")
        self.fns = list(fns)

    def __call__(self, x: Any) -> Any:
        for fn in self.fns:
            x = fn(x)
        return x

    def __repr__(self) -> str:
        fns_str = ",\n    ".join(repr(fn) for fn in self.fns)
        return f"Compose([\n    {fns_str}\n])"
