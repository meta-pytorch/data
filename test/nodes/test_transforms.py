# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase
from torchdata.nodes import Compose, Mapper, ParallelMapper
from torchdata.nodes.adapters import IterableWrapper


class TestCompose(TestCase):
    def test_basic_composition(self):
        compose = Compose([lambda x: x + 1, lambda x: x * 2])
        self.assertEqual(compose(3), 8)  # (3 + 1) * 2

    def test_single_function(self):
        compose = Compose([lambda x: x ** 2])
        self.assertEqual(compose(4), 16)

    def test_order_is_sequential(self):
        # Functions applied left to right: first divide then add, not add then divide
        compose = Compose([lambda x: x / 2, lambda x: x + 10])
        self.assertAlmostEqual(compose(4), 12.0)  # 4/2=2, 2+10=12

    def test_empty_fns_raises(self):
        with self.assertRaises(ValueError):
            Compose([])

    def test_with_mapper_node(self):
        n = 10
        source = IterableWrapper(range(n))
        node = Mapper(source, Compose([lambda x: x + 1, lambda x: x * 3]))
        for _ in range(2):
            node.reset()
            result = list(node)
            self.assertEqual(result, [(i + 1) * 3 for i in range(n)])

    def test_with_parallel_mapper_node(self):
        n = 10
        source = IterableWrapper(range(n))
        node = ParallelMapper(
            source,
            Compose([lambda x: x + 1, lambda x: x * 3]),
            num_workers=2,
            in_order=True,
        )
        for _ in range(2):
            node.reset()
            result = list(node)
            self.assertEqual(result, [(i + 1) * 3 for i in range(n)])

    def test_fns_attribute(self):
        fns = [lambda x: x, lambda x: x]
        compose = Compose(fns)
        self.assertEqual(len(compose.fns), 2)
        # fns is a copy, not the original list
        fns.append(lambda x: x)
        self.assertEqual(len(compose.fns), 2)

    def test_repr(self):
        compose = Compose([lambda x: x])
        r = repr(compose)
        self.assertIn("Compose", r)

    def test_string_transforms(self):
        compose = Compose([str.strip, str.upper])
        self.assertEqual(compose("  hello  "), "HELLO")

    def test_dict_transforms(self):
        def scale(d):
            return {k: v * 2 for k, v in d.items()}

        def shift(d):
            return {k: v + 1 for k, v in d.items()}

        compose = Compose([scale, shift])
        result = compose({"a": 3, "b": 5})
        self.assertEqual(result, {"a": 7, "b": 11})  # scale: {6, 10}, shift: {7, 11}
