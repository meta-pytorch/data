# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper, MapStyleWrapper
from torchdata.nodes.base_node import BaseNode
from torchdata.nodes.loader import Loader

from .utils import DummyIterableDataset, DummyMapDataset, StatefulRange


class TestLoader(TestCase):
    def _test_loader_correct_state_dict_at_midpoint(self, node: BaseNode, length: int):
        x = Loader(node)
        results = list(x)

        # Create an iterator at end of iteration
        it = iter(x)

        results_copy = []
        for _ in range(length // 2):
            results_copy.append(next(it))
        state_dict_0 = x.state_dict()

        x.load_state_dict(state_dict_0)

        # Create an iterator in the middle of iteration
        it = iter(x)

        self.assertEqual(x.state_dict(), state_dict_0)

        for _ in range(length // 2):
            results_copy.append(next(it))

        self.assertEqual(len(results), length)
        self.assertEqual(len(results_copy), length)
        self.assertEqual(results[length // 2 :], results_copy[length // 2 :])

    def test_loader_equal_state_dict_on_save_load_iterable(self) -> None:
        length = 10
        node = IterableWrapper(DummyIterableDataset(length))
        self._test_loader_correct_state_dict_at_midpoint(node, length)

    def test_loader_equal_state_dict_on_save_load_stateful(self) -> None:
        length = 10
        node = IterableWrapper(StatefulRange(length))
        self._test_loader_correct_state_dict_at_midpoint(node, length)

    def test_loader_equal_state_dict_on_save_load_map(self) -> None:
        length = 10
        node = MapStyleWrapper(DummyMapDataset(length), sampler=range(length))
        self._test_loader_correct_state_dict_at_midpoint(node, length)
