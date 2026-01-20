# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torch.testing._internal.common_utils import TestCase

from torch.utils.data import Dataset, IterableDataset
from torchdata.stateful_dataloader import Stateful, StatefulDataLoader, StateParserUtil


class StatefulIterableDataset(IterableDataset, Stateful):
    def __init__(self):
        self.num_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.num_calls += 1
        return self.num_calls

    def load_state_dict(self, state_dict):
        self.num_calls = state_dict["num_calls"]

    def state_dict(self):
        return {"num_calls": self.num_calls}


def identity(x):
    return x


class TestIteratorDataset(TestCase):
    def test_increasing_worker(self):
        ds = StatefulIterableDataset()
        dl = StatefulDataLoader(ds, num_workers=2, collate_fn=identity)
        it = iter(dl)
        next(it)
        sd = dl.state_dict()
        print(sd)
        del dl

        parser = StateParserUtil(sd)
        worker_states = parser.fetch_dataset_state()
        worker_states[2] = {"num_calls": 2}
        worker_states[3] = {"num_calls": 3}
        parser.set_dataset_state(worker_states)

        # worker state doesn't equal num workers setting
        with self.assertRaises(AssertionError):
            parser.get_state_dict()
        parser.set_num_workers(4)

        # last worker yielded id is greater than num workers
        parser.set_last_worker_yielded_id(10)
        with self.assertRaises(AssertionError):
            parser.get_state_dict()
        parser.set_last_worker_yielded_id(0)

        # load the modified state
        new_sd = parser.get_state_dict()
        print(new_sd)
        dl = StatefulDataLoader(ds, num_workers=4, collate_fn=identity)
        dl.load_state_dict(new_sd)
        it = iter(dl)
        values = []
        for _ in range(4):
            values.extend(next(it))
        print(values)
        self.assertEqual(values, [1, 3, 4, 2])


if __name__ == "__main__":
    unittest.main()
