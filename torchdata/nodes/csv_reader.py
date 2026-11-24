# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
from itertools import islice
from typing import Any, Dict, Iterator, List, Optional, Sequence, TextIO, Union

from torchdata.nodes.base_node import BaseNode


class CSVReader(BaseNode[Union[List[str], Dict[str, str]]]):
    """Node for reading CSV files with state management and header support.
    Args:
        file_path: Path to CSV file
        has_header: Whether first row contains column headers
        delimiter: CSV field delimiter
        return_dict: Return rows as dictionaries (requires has_header=True)
    """

    NUM_LINES_YIELDED = "num_lines_yielded"
    HEADER_KEY = "header"

    def __init__(
        self,
        file_path: str,
        has_header: bool = False,
        delimiter: str = ",",
        return_dict: bool = False,
        encoding: str = "utf-8",
    ):
        super().__init__()
        self.file_path = file_path
        self.has_header = has_header
        self.delimiter = delimiter
        self.return_dict = return_dict
        if return_dict and not has_header:
            raise ValueError("return_dict=True requires has_header=True")
        self.encoding = encoding
        self._file: Optional[TextIO] = None
        self._reader: Optional[Iterator[Union[List[str], Dict[str, str]]]] = None
        self._header: Optional[Sequence[str]] = None
        self._num_lines_yielded: int = 0
        self.reset()  # Initialize reader

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset()
        self.close()

        # Reopen the file and reset counters
        self._file = open(self.file_path, encoding=self.encoding)
        self._num_lines_yielded = 0
        if initial_state is not None:
            self._handle_initial_state(initial_state)
        else:
            self._initialize_reader()

    def _handle_initial_state(self, state: Dict[str, Any]):
        """Restore reader state from checkpoint."""
        # Validate header compatibility
        if (not self.has_header and self.HEADER_KEY in state) or (
            self.has_header and state[self.HEADER_KEY] is None
        ):
            raise ValueError(
                f"Check if has_header={self.has_header} matches the state header={state[self.HEADER_KEY]}"
            )

        self._header = state.get(self.HEADER_KEY)
        target_line_num = state[self.NUM_LINES_YIELDED]
        assert self._file is not None
        # Create appropriate reader
        if self.return_dict:

            self._reader = csv.DictReader(
                self._file, delimiter=self.delimiter, fieldnames=self._header
            )
        else:
            self._reader = csv.reader(self._file, delimiter=self.delimiter)
        # Skip header if needed (applies only when file has header)

        assert isinstance(self._reader, Iterator)
        if self.has_header:
            try:
                next(self._reader)  # Skip header line
            except StopIteration:
                pass  # Empty file
        # Fast-forward to target line using efficient slicing
        consumed = sum(1 for _ in islice(self._reader, target_line_num))
        self._num_lines_yielded = consumed

    def _initialize_reader(self):
        """Create fresh reader without state."""
        assert self._file is not None
        if self.return_dict:
            self._reader = csv.DictReader(self._file, delimiter=self.delimiter)
            self._header = self._reader.fieldnames
        else:
            self._reader = csv.reader(self._file, delimiter=self.delimiter)

            if self.has_header:

                try:
                    self._header = next(self._reader)
                except StopIteration:
                    self._header = None  # Handle empty file

    def next(self) -> Union[List[str], Dict[str, str]]:
        try:
            assert isinstance(self._reader, Iterator)
            row = next(self._reader)
            self._num_lines_yielded += 1
            return row

        except StopIteration:
            self.close()
            raise

    def get_state(self) -> Dict[str, Any]:
        return {
            self.NUM_LINES_YIELDED: self._num_lines_yielded,
            self.HEADER_KEY: self._header,
        }

    def close(self):
        if self._file is not None and not self._file.closed:
            self._file.close()
