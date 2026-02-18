# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


class StateParserUtil:
    """
    Utility class that can be used to modify state returned by the dataloader
    """

    def __init__(self, state_dict: Dict[str, Any]):
        self._state_dict = state_dict
        self._is_multiprocess_state = "_snapshot" in self._state_dict

    def fetch_dataset_state(self) -> Dict[int, Any]:
        # Handle both cases of single process and multiprocess
        if not self._is_multiprocess_state:
            return self._state_dict["dataset_state"]
        return {
            state["worker_id"]: state["dataset_state"]
            for _, state in self._state_dict["_snapshot"]["_worker_snapshots"].items()
        }

    def set_last_worker_yielded_id(self, last_worker_yielded: int) -> None:
        # Ensure that this number is within the number of workers
        if not self._is_multiprocess_state:
            logger.warning("Cannot set last worker yielded id on a single process state dict")
            return
        self._state_dict["_snapshot"]["_last_yielded_worker_id"] = last_worker_yielded

    def set_num_workers(self, num_workers: int) -> None:
        if not self._is_multiprocess_state:
            logger.warning("Cannot set num_workers on a single process state dict")
            return
        self._state_dict["_snapshot"]["_main_snapshot"]["_num_workers"] = num_workers

    def set_dataset_state(self, dataset_state: Union[Dict[int, Any], Any]) -> None:
        if not self._is_multiprocess_state:
            self._state_dict["dataset_state"] = dataset_state
            return

        for id, state in dataset_state.items():
            worker_states = self._state_dict["_snapshot"]["_worker_snapshots"]
            worker_key = f"worker_{id}"
            if worker_key in worker_states:
                worker_states[worker_key]["dataset_state"] = state
            else:
                worker_states[worker_key] = {"worker_id": id, "dataset_state": state, "fetcher_state": None}

    def get_state_dict(self) -> Dict[str, Any]:
        # Perform validations
        # a) num_workers should match worker_snapshots
        # b) last yielded worker id should be within num_workers
        if not self._is_multiprocess_state:
            return self._state_dict

        last_yielded_id = self._state_dict["_snapshot"]["_last_yielded_worker_id"]
        num_workers = self._state_dict["_snapshot"]["_main_snapshot"]["_num_workers"]
        worker_ids = self._state_dict["_snapshot"]["_worker_snapshots"].keys()

        assert (
            len(worker_ids) == num_workers
        ), f"Number of worker states {len(worker_ids)} should be equal to num_workers setting {num_workers}"
        assert (
            len(set(worker_ids)) == num_workers
        ), f"Worker state for all from [0, {num_workers}) should be present. Instead found state for only {worker_ids} workers"
        assert last_yielded_id < num_workers, "Last yielded id should be strictly within the number of workers"
        return self._state_dict
