import copy
from collections.abc import Sequence
from typing import Any, TypeVar

from torchdata.nodes.base_node import BaseNode

T = TypeVar("T")


class RoundRobinCyclerNode(BaseNode[T]):
    """A node that cycles through multiple datasets in a round-robin way.

    This node takes a sequence of source nodes and iterates through them.
    It yields one item from each node in order, and when a node is exhausted, it is
    immediately removed from the rotation. This continues until all nodes are exhausted.

    On complete exhaustion of all source nodes, the node will raise StopIteration.

    Args:
        source_nodes (Sequence[BaseNode[T]]): A sequence of source nodes.

    """

    SOURCE_STATES_KEY = "source_states"
    ACTIVE_NODES_KEY = "active_nodes"
    NODE_ITEM_INDICES_KEY = "node_item_indices"
    CURRENT_NODE_INDEX_KEY = "current_node_index"
    TOTAL_ITEMS_YIELDED_KEY = "total_items_yielded"

    def __init__(
        self,
        source_nodes: Sequence[BaseNode[T]],
    ) -> None:
        super().__init__()

        if not source_nodes:
            raise ValueError("At least one source node must be provided")

        self.source_nodes = source_nodes
        self._total_items_yielded = 0
        self._current_node_index = 0
        self._active_nodes = list(range(len(self.source_nodes)))

        # Initialize each node's item index to 0
        # Keeps track of how many items have been yielded from each individual source node
        self._node_item_indices = {i: 0 for i in range(len(self.source_nodes))}

    def reset(self, initial_state: dict[str, Any] | None = None):
        super().reset(initial_state)

        if initial_state is not None:
            self._total_items_yielded = initial_state[self.TOTAL_ITEMS_YIELDED_KEY]
            self._active_nodes = initial_state[self.ACTIVE_NODES_KEY]
            self._node_item_indices = initial_state[self.NODE_ITEM_INDICES_KEY]
            self._current_node_index = initial_state[self.CURRENT_NODE_INDEX_KEY]

            for i in range(len(self.source_nodes)):
                self.source_nodes[i].reset(initial_state[self.SOURCE_STATES_KEY][i])
        else:
            # Force a fresh iterator from all source nodes
            self._total_items_yielded = 0
            self._active_nodes = list(range(len(self.source_nodes)))
            self._node_item_indices = {i: 0 for i in range(len(self.source_nodes))}
            self._current_node_index = 0

            for node in self.source_nodes:
                node.reset()

    def next(self) -> T:
        if not self._active_nodes:
            raise StopIteration()

        # Make sure we don't go out of bounds with the active nodes
        if self._current_node_index >= len(self._active_nodes):
            self._current_node_index = 0

        # Get the actual source node index from active nodes
        node_index = self._active_nodes[self._current_node_index]
        node = self.source_nodes[node_index]

        try:
            item = next(node)

            # Update state
            self._total_items_yielded += 1
            self._node_item_indices[node_index] += 1

            # Move to the next node in the rotation
            self._current_node_index = (self._current_node_index + 1) % len(self._active_nodes)

            return item

        except StopIteration:
            # Remove this node from active nodes
            self._active_nodes.pop(self._current_node_index)

            # If we've removed all nodes, we're done
            if not self._active_nodes:
                raise StopIteration()

            # Make sure the index stays valid (it might now be out of bounds)
            if self._current_node_index >= len(self._active_nodes):
                self._current_node_index = 0

            # Try the next node
            return self.next()

    def get_state(self) -> dict[str, Any]:
        return {
            self.SOURCE_STATES_KEY: [node.state_dict() for node in self.source_nodes],
            self.ACTIVE_NODES_KEY: copy.deepcopy(self._active_nodes),
            self.NODE_ITEM_INDICES_KEY: copy.deepcopy(self._node_item_indices),
            self.CURRENT_NODE_INDEX_KEY: self._current_node_index,
            self.TOTAL_ITEMS_YIELDED_KEY: self._total_items_yielded,
        }
