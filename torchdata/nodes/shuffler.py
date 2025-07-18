import random
from collections import deque
from typing import Any, Deque, Dict, Optional, TypeVar

from torchdata.nodes import BaseNode


T = TypeVar("T")


class Shuffler(BaseNode[T]):
    """Node that shuffles items from source node using a buffer.
    This node maintains a buffer of items from the source node and returns them
    in a random order. The buffer is filled to capacity initially and then
    replenished as items are removed. The randomization is controlled by a
    random number generator with an optional seed for reproducibility.

    Args:
        source_node (BaseNode[T]): The source node to pull items from.
        buffer_size (int, optional): Size of the buffer used for shuffling. Must be at least 1.
            Defaults to DEFAULT_BUFFER_SIZE (1024).
        seed (Optional[int]): Optional seed for the random number generator.
        store_buffer (bool, optional): Whether to store buffer contents in state for full
            reproducibility. If True, buffer contents are serialized which ensures perfect
            state restoration but may be slower and use more memory. If False, buffer is
            not stored which is faster but may affect reproducibility. Defaults to True.
    """

    SOURCE_KEY = "source"
    BUFFER_SIZE_KEY = "buffer_size"
    RNG_STATE_KEY = "rng_state"
    BUFFER_KEY = "buffer"
    NUM_YIELDED_KEY = "num_yielded"
    STORE_BUFFER_KEY = "store_buffer"
    DEFAULT_BUFFER_SIZE = 1024

    def __init__(
        self,
        source_node: BaseNode[T],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        seed: Optional[int] = None,
        store_buffer: bool = True,
    ):
        super().__init__()
        if buffer_size < 1:
            raise ValueError("Buffer size must be at least 1")
        self.source = source_node
        self.buffer_size = buffer_size
        self.buffer: Deque[T] = deque()
        self.rng = random.Random(seed)
        self._initial_seed = seed
        self._num_yielded = 0
        self.store_buffer = store_buffer

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset the node to its initial state or to the provided state.
        Args:
            initial_state: Optional state dictionary to restore from.
        Raises:
            ValueError: If the buffer size in the state doesn't match the current buffer size.
        """
        super().reset(initial_state)
        self.buffer.clear()
        self._num_yielded = 0

        if initial_state is not None:
            # Be strict about required keys in the state
            self.source.reset(initial_state[self.SOURCE_KEY])

            # Validate buffer size matches
            if initial_state[self.BUFFER_SIZE_KEY] != self.buffer_size:
                raise ValueError(
                    f"Buffer size mismatch: state has {initial_state[self.BUFFER_SIZE_KEY]}, "
                    f"but current shuffler has {self.buffer_size}"
                )

            self.rng.setstate(initial_state[self.RNG_STATE_KEY])
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]

            # Restore buffer if it was stored
            if self.store_buffer and self.BUFFER_KEY in initial_state:
                self.buffer.extend(initial_state[self.BUFFER_KEY])

            # Buffer will be refilled on next call to next() if not stored or empty
        else:
            self.source.reset(None)
            if self._initial_seed is not None:
                self.rng = random.Random(self._initial_seed)

    def _fill_buffer(self) -> bool:
        """Fill buffer with items from source until it reaches buffer_size or source is exhausted.
        Returns:
            True if the buffer contains any items after the call, False if the buffer is empty.
        """
        if len(self.buffer) >= self.buffer_size:
            return True  # Buffer is already full

        try:
            while len(self.buffer) < self.buffer_size:
                self.buffer.append(next(self.source))
            return True  # Buffer is now full
        except StopIteration:
            # Source is exhausted, check if we have any items in the buffer
            return len(self.buffer) > 0

    def next(self) -> T:
        """Get the next item from the buffer in random order.
        Returns:
            A randomly selected item from the buffer.
        Raises:
            StopIteration: If there are no more items in the buffer and the source is exhausted.
        """
        if not self.buffer and not self._fill_buffer():
            raise StopIteration

        # Randomly select and remove an item from the buffer
        idx = self.rng.randrange(len(self.buffer))
        item = self.buffer[idx]
        self.buffer[idx] = self.buffer[-1]
        self.buffer.pop()

        # Try to refill buffer
        self._fill_buffer()
        self._num_yielded += 1
        return item

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the node.
        Returns:
            Dict[str, Any]: A dictionary containing the state of the source node,
            number of items yielded, and random generator state. If store_buffer is True,
            the buffer contents are also included for full reproducibility.
        """
        state = {
            self.SOURCE_KEY: self.source.state_dict(),
            self.BUFFER_SIZE_KEY: self.buffer_size,
            self.NUM_YIELDED_KEY: self._num_yielded,
            self.RNG_STATE_KEY: self.rng.getstate(),
            self.STORE_BUFFER_KEY: self.store_buffer,
        }

        # Include buffer contents if store_buffer is True
        if self.store_buffer:
            state[self.BUFFER_KEY] = list(self.buffer)

        return state
