# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import tempfile

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase

from torchdata.nodes.csv_reader import CSVReader

from .utils import run_test_save_load_state


class TestCSVReader(TestCase):
    def setUp(self):
        self.test_data = [
            ["Alice", "30", "New York"],
            ["Bob", "25", "London"],
            ["Charlie", "35", "Paris"],
            ["David", "40", "Rome"],
            ["Eve", "45", "Tokyo"],
            ["Frank", "50", "Beijing"],
            ["Grace", "55", "Shanghai"],
            ["Harry", "60", "Seoul"],
            ["Iris", "65", "Buenos Aires"],
            ["Jack", "70", "Sao Paulo"],
            ["Katy", "75", "Mexico City"],
            ["Lily", "80", "Bogota"],
        ]

    def _create_temp_csv(self, delimiter=",", header=True):
        if header:
            self.test_data.insert(0, ["name", "age", "city"])
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerows(self.test_data)
        return path

    def test_basic_read_list(self):
        path = self._create_temp_csv(header=False)
        node = CSVReader(path, has_header=False)
        results = list(node)
        self.assertEqual(len(results), len(self.test_data))
        self.assertEqual(results[0], ["Alice", "30", "New York"])
        self.assertEqual(results[-1], ["Lily", "80", "Bogota"])
        node.close()

    def test_basic_read_dict(self):
        path = self._create_temp_csv()
        node = CSVReader(path, has_header=True, return_dict=True)
        results = list(node)

        self.assertEqual(len(results), len(self.test_data) - 1)
        self.assertEqual(results[0], {"name": "Alice", "age": "30", "city": "New York"})
        self.assertEqual(results[1]["city"], "London")
        self.assertEqual(results[-1]["city"], "Bogota")
        node.close()

    def test_different_delimiters(self):
        path = self._create_temp_csv(delimiter="|")
        node = CSVReader(path, has_header=True, delimiter="|", return_dict=True)
        results = list(node)

        self.assertEqual(len(results), len(self.test_data) - 1)
        self.assertEqual(results[2]["city"], "Paris")
        self.assertEqual(results[-1]["city"], "Bogota")
        node.close()

    def test_state_management(self):
        path = self._create_temp_csv()
        node = CSVReader(path, has_header=True, return_dict=True)
        print(f"initial state: {node.state_dict()}")
        for _ in range(11):
            _ = next(node)
            print(f"element = {_}, state: {node.state_dict()}")

        state = node.state_dict()

        node.reset(state)
        item = next(node)

        with self.assertRaises(StopIteration):
            next(node)

        self.assertEqual(item["name"], "Lily")
        self.assertEqual(state[CSVReader.NUM_LINES_YIELDED], 11)
        node.close()

    @parameterized.expand([3, 5, 7])
    def test_save_load_state(self, midpoint: int):
        path = self._create_temp_csv(header=True)
        node = CSVReader(path, has_header=True)
        run_test_save_load_state(self, node, midpoint)
        node.close()

    def test_load_wrong_state(self):
        path = self._create_temp_csv(header=True)
        node = CSVReader(path, has_header=True)

        state = node.state_dict()
        state[CSVReader.HEADER_KEY] = None
        with self.assertRaisesRegex(
            ValueError, "Check if has_header=True matches the state header=None"
        ):
            node.reset(state)

        node.close()

        node = CSVReader(path, has_header=False)
        state = node.state_dict()
        state[CSVReader.HEADER_KEY] = ["name", "age"]
        with self.assertRaisesRegex(
            ValueError,
            r"Check if has_header=False matches the state header=\['name', 'age'\]",
        ):
            node.reset(state)

        node.close()

    def test_empty_file(self):
        path = self._create_temp_csv()
        # Overwrite with empty file
        with open(path, "w") as _:
            pass

        node = CSVReader(path, has_header=False)
        with self.assertRaises(StopIteration):
            next(node)
        node.close()

    def test_header_validation(self):
        with self.assertRaisesRegex(
            ValueError, "return_dict=True requires has_header=True"
        ):
            CSVReader("dummy.csv", has_header=False, return_dict=True)

    def test_multi_epoch(self):
        path = self._create_temp_csv()
        node = CSVReader(path, has_header=True, return_dict=True)

        # First epoch
        epoch1 = list(node)
        self.assertEqual(len(epoch1), len(self.test_data) - 1)

        # Second epoch
        node.reset()
        epoch2 = list(node)
        self.assertEqual(epoch1, epoch2)
        node.close()

    def test_partial_read_resume(self):
        path = self._create_temp_csv(header=True)
        node = CSVReader(path, has_header=True)

        # Read partial and get state
        _ = next(node)  # Line 0
        state1 = node.state_dict()

        _ = next(node)  # Line 1
        state2 = node.state_dict()

        # Resume from first state
        node.reset(state1)
        self.assertEqual(next(node), self.test_data[2])

        # Resume from second state
        node.reset(state2)
        self.assertEqual(next(node), self.test_data[3])
        node.close()

    def test_file_closure(self):
        path = self._create_temp_csv()
        node = CSVReader(path, has_header=True)

        # Read all items
        list(node)

        # Verify file is closed
        self.assertTrue(node._file.closed)
        node.close()

    def test_state_with_header(self):
        path = self._create_temp_csv()
        node = CSVReader(path, has_header=True, return_dict=True)

        # Read one item
        _ = next(node)
        state = node.state_dict()

        # Verify header preservation
        node.reset(state)
        item = next(node)
        self.assertEqual(item["city"], "London")
        node.close()

    def tearDown(self):
        # Clean up temporary files
        for f in os.listdir(tempfile.gettempdir()):
            if f.startswith("tmp") and f.endswith(".csv"):
                os.remove(os.path.join(tempfile.gettempdir(), f))
