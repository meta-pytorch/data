# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pyarrow as pa
import tempfile
import unittest

import torch

from torch.testing._internal.common_utils import TestCase

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.scalable_reader import ScalableReader, PreprocessDataset, ArrowHandler

# A set of draft unit tests for the ScalableReader.
# Note that these have not been locally tested or debugged yet (fighting my local environment),
# and likely fail in horrible ways. Mostly here for discussion/reference at this stage.

# TODO: test actual save/load distributed functions via multiprocessing

def create_temp_dir(dir=None):
    # The temp dir and files within it will be released and deleted in tearDown().
    # Adding `noqa: P201` to avoid mypy's warning on not releasing the dir handle within this function.
    temp_dir = tempfile.TemporaryDirectory(dir=dir)  # noqa: P201
    return temp_dir

class TestScalableReader(TestCase):
    def setUp(self):
        super().setUp()
        data = create_temp_dir()
        datapath = data.name
        schema = pa.schema([pa.field("tokens", pa.uint32())])
        with pa.ipc.new_file(
            os.path.join(datapath, "fileshard_1.arrow"), schema
        ) as writer:
            for i in range(500):
                out = list(range(i * 100, i * 100 + 100))
                writer.write(pa.record_batch([out], schema=schema))
        os.makedirs(os.path.join(datapath, "subfolder"))
        with pa.ipc.new_file(
            os.path.join(datapath, "subfolder/fileshard_2.arrow"), schema
        ) as writer:
            for i in range(500):
                out = list(range(50000 + i * 100, 50000 + i * 100 + 100))
                writer.write(pa.record_batch([out], schema=schema))
        self.datapath = datapath
        self.data = data

    def create_scalable(
        self,
        rank = 0,
        worldsize = 1,
        delimiter = -1,
        bos = None,
        seed = 42,
        chunk = 1000,
        logicals = 10
    ):
        # Build dataloader
        data = ScalableReader(
            self.datapath, 
            rank, 
            worldsize, 
            ArrowHandler, 
            delimiter, 
            bos, 
            seed=seed, 
            max_chunksize=chunk, 
            n_logical_shards=logicals,
        )
        # Pad entries to make them batch-able
        data = PreprocessDataset(data, lambda x: x + [-1]*(chunk-len(x)))
        # Statelessly convert all outputs to tensors
        data = PreprocessDataset(data, torch.tensor)
        return data

    def test_single_epoch(self):
        for ws in [2,3,7]:
            for nw in [0,2,3]:
                loaderset = [iter(StatefulDataLoader(self.create_scalable(i, ws, logicals=555), batch_size=1, num_workers=nw)) for i in range(ws)]
                n_steps = math.ceil(1000/ws)+10
                pools = [set() for _ in loaderset]
                for _ in range(n_steps):
                    for i,l in enumerate(loaderset):
                        pools[i].add(next(l)[0,0].item())
                for i in range(len(pools)):
                    for j in range(i+1, len(pools)):
                        print(f"Checking outputs {i} and {j}")
                        overlap = len(pools[i].intersection(pools[j]))
                        self.assertEqual(overlap, 0, f"Overlapping data found in workers {i} and {j} (worldsize {ws}/{ws*max(nw,1)}): {overlap}")
                alldata = set.union(*pools)
                expected = set([x*100 for x in range(1000)])
                missing = len(expected.difference(alldata))
                self.assertEqual(missing, 0, f"Missing data from pool: {missing}")

    def test_resumption(self):
        for ws in [2,3,7]:
            for nw in [0,2,3]:
                loaderset = [StatefulDataLoader(self.create_scalable(i, ws, logicals=555), batch_size=1, num_workers=nw) for i in range(ws)]
                loaderset2 = [StatefulDataLoader(self.create_scalable(i, ws, logicals=555), batch_size=1, num_workers=nw) for i in range(ws)]
                n_steps = 2*math.ceil(1000/ws)  # Proceed well into second epoch
                iterset = [iter(l) for l in loaderset]
                for _ in range(100):
                    [next(l) for l in iterset]
                for i in range(ws):
                    loaderset2[i].load_state_dict(loaderset[i].state_dict())
                iterset2 = [iter(l) for l in loaderset2]
                for s in range(n_steps):
                    for i in range(ws):
                        expected = next(iterset[i])
                        query = next(iterset2[i])
                        self.assertEqual(expected, query, f"Mismatch at step 100+{s} rank {i}, (worldsize {ws}/{ws*max(nw,1)}): original {expected[0,:5]}..., recieved {query[0,:5]}")

    def test_rescale_epoch(self):
        nsteps = 30
        for start_ws in [1,2,6]:
            for end_ws in [3,4]:
                for logicals in [300, 555, 721]:
                    # Create checkpoint
                    avoid = []
                    data = StatefulDataLoader(self.create_scalable(logicals=logicals, chunk=40), num_workers=start_ws, batch_size=1)
                    for i, inp in enumerate(data):
                        avoid.append(inp[0,0].item())
                        if i==(nsteps-1)*start_ws:
                            sd = data.state_dict()
                            break
                    # Load checkpoint
                    # (this step likely fails without using the custom distributed save/load checkpointing fns)
                    data = StatefulDataLoader(self.create_scalable(logicals=logicals, chunk=40), num_workers=end_ws, batch_size=1)
                    data.load_state_dict(sd)
                    vals = []
                    nsteps = math.ceil(3000 - len(avoid)) + (2*math.ceil(3000/logicals)*end_ws)
                    for i, inp in enumerate(data):
                        vals.append(inp[0,0].item())
                        if i == nsteps:
                            break
                    # Invert set of seen values
                    expect = []
                    for i in range(1000):
                        for offset in [0,40,80]:
                            if i*100+offset not in avoid:
                                expect.append(i*100+offset)
                    for x in expect:
                        self.assertObjectIn(x, vals)

