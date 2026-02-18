import argparse
import math
import os
import pyarrow as pa
import time
import torch
from torch import distributed as dist

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.scalable_reader import (
    ArrowHandler,
    PreprocessDataset,
    ScalableReader,
    load_distributed_state_dict,
    save_distributed_state_dict,
)

# This example script validates the rescaling behavior of the ScalableReader.
# On first run, creates a dummy dataset and saves a distributed checkpoint at the desired location.
# On subsequent runs, loads the checkpoint (possibly on a different world size / num workers)
# and verifies that all remaining data is covered by the time the epoch finishes.

# Example usage:
# torchrun [torchrun args] examples/ibm_rescaling/rescaling_demo.py --ckpt_path=~/ckpts/rescale_test --logical_shards=48 --num_workers=6

# Do not change the number of steps between the first and second runs!

parser = argparse.ArgumentParser(description="Script to validate rescaling of dataloader checkpoints")
parser.add_argument("--ckpt_path", type=str, default="./rescale_test")
parser.add_argument(
    "--logical_shards",
    type=int,
    default=350,
    help="Total number of data partitions. Must exceed (worldsize * n_workers) but not n_docs (1000).",
)
parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers per device")
parser.add_argument("--b_size", type=int, default=2, help="Number of data points per step per device")
parser.add_argument("--n_steps", type=int, default=50, help="Number of steps to take before saving. (n_steps * b_size * worldsize) cannot exceed number of items in epoch (3000)")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()


# Setup
rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
dist.init_process_group(backend="gloo")
mesh = dist.device_mesh.init_device_mesh("cpu", [world_size])
placement = [dist.tensor.placement_types.Shard(0)]

# Check input args
assert args.logical_shards >= world_size*args.num_workers, f"Logical shards {args.logical_shards} cannot be less than total workers {world_size*args.num_workers}"
assert args.logical_shards <= 1000, f"Logical shards {args.logical_shards} cannot exceed number of documents 1000"
assert args.n_steps*args.b_size*world_size < 3000, f"Number of items drawn before saving {args.n_steps*args.b_size*world_size} cannot exceed number of document chunks 3000."

# Build dataset
datapath = os.path.join(args.ckpt_path, "dataset")
if not os.path.exists(datapath):
    if rank == 0:
        os.makedirs(datapath)
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
    else:
        # Give other ranks time for worker 0 to finish
        time.sleep(5)

# Build dataloader
data = ScalableReader(datapath, rank, world_size, ArrowHandler, -1, seed=args.seed, max_chunksize=40, n_logical_shards=args.logical_shards)
# Pad entries to make them batch-able
data = PreprocessDataset(data, lambda x: x + [-1]*(40-len(x)))
# Statelessly convert all outputs to tensors
data = PreprocessDataset(data, torch.tensor)
# Wrap in StatefulDataLoader
data = StatefulDataLoader(data, batch_size=args.b_size, num_workers=args.num_workers)

# If checkpoint does not exist, create it
ckpt_path = os.path.join(args.ckpt_path, "loader_dcp_state")
if not os.path.exists(ckpt_path) or len(os.listdir(ckpt_path)) == 0:
    os.makedirs(ckpt_path, exist_ok=True)
    # Iterate, assemble values to exclude
    if rank == 0:
        print(f"No existing checkpoint. Processing {args.n_steps} steps.")

    avoid = []
    for i, inp in enumerate(data):
        avoid.append(inp[:,0])
        if i == args.n_steps-1:
            if rank == 0:
                print("Iteration complete!")
            save_distributed_state_dict(data, ckpt_path, mesh)
            break
    avoid = torch.cat(avoid)
    # Get all vals onto each rank
    avoid = dist.tensor.DTensor.from_local(
        avoid,
        mesh,
        placement,
    ).full_tensor()

    if rank == 0:
        torch.save(avoid, os.path.join(args.ckpt_path, "avoid.pth"))
        print(
            "Generation complete! Please rerun (with different world size / workers if desired) to complete the check."
        )

# If checkpoint does exist, load and finish epoch.
# Ensure all expected values are covered once epoch concludes.
else:
    if rank == 0:
        print("Checkpoint detected!")
    load_distributed_state_dict(data, ckpt_path, mesh)
    avoid = torch.load(os.path.join(args.ckpt_path, "avoid.pth")).tolist()

    # Finish out epoch (extra 2*ceil(ceil(n_items/n_shards)/bsize) steps to account for worst-case uneven finishing times)
    vals = []
    n_steps = (
        math.ceil((3000 - len(avoid)) / (world_size * args.b_size)) 
        + 2 * math.ceil(math.ceil(3000/args.logical_shards)/args.b_size)
    )
    for i, inp in enumerate(data):
        vals.append(inp)
        if i == n_steps:
            break
    vals = torch.cat(vals)
    # Get all vals onto each rank
    vals = dist.tensor.DTensor.from_local(vals, mesh, placement).full_tensor()

    # Save final state dicts for diagnostic purposes
    os.makedirs(os.path.join(args.ckpt_path, "diag"), exist_ok=True)
    torch.save(data.state_dict(), os.path.join(args.ckpt_path, "diag", f"loader_state_{rank}.pth"))

    # Perform data coverage check on rank 0 only
    if rank == 0:
        # Invert avoid to get expected vals
        expect = []
        for i in range(1000):
            for offset in [0,40,80]:
                if i*100+offset not in avoid:
                    expect.append(i*100+offset)

        for x in expect:
            assert x in vals, x
        print("Check passed: upcoming data is covered as expected!")

dist.barrier()
dist.destroy_process_group()
