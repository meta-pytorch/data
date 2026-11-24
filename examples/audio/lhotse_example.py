#!/usr/bin/env python

"""
This is a working example of using a new PyTorch dataloading API called DataPipes
with Lhotse. It's not the final version and probably a lot of things will change.

Lhotse is a speech data representation library that brings various corpora into
a single format and interfaces well with PyTorch. See: https://github.com/lhotse-speech/lhotse

Prepare the env like this first:

   $ conda create -n torchdata python=3.8
   $ conda activate torchdata
   $ conda install numpy
   $ pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
   $ pip install --user "git+https://github.com/pytorch/data.git"

Optionally, if not in this Lhotse branch already:

   $ pip install git+https://github.com/lhotse-speech/lhotse@feature/datapipes-prototyping

"""


import warnings
from collections import deque, defaultdict
from functools import partial
from pathlib import Path
from typing import Optional

from lhotse import CutSet, load_manifest
from lhotse.utils import Seconds
from lhotse.recipes import download_librispeech, prepare_librispeech

import torchdata
import torch
import torchdata.datapipes.iter as dp  # Only use Iter-style DataPipe to illustrate the pipeline

from torch.utils.data.datapipes.iter import Multiplexer, Demultiplexer
from torch.utils.data.communication.eventloop import SpawnProcessForDataPipeline
from torch.utils.data.communication.protocol import IterDataPipeQueueProtocolClient
from torch.utils.data.communication.iter import QueueWrapper


class CutsReader(dp.IterDataPipe):
    def __init__(self, path) -> None:
        self.cuts = CutSet.from_jsonl_lazy(path)

    def __iter__(self):
        yield from self.cuts


class DurationBatcher(dp.IterDataPipe):
    def __init__(
        self,
        datapipe,
        max_frames: int = None,
        max_samples: int = None,
        max_duration: Seconds = None,
        max_cuts: Optional[int] = None,
        drop_last: bool = False,
    ):
        from lhotse.dataset.sampling.base import SamplingDiagnostics, TimeConstraint

        self.datapipe = datapipe
        self.reuse_cuts_buffer = deque()
        self.drop_last = drop_last
        self.max_cuts = max_cuts
        self.diagnostics = SamplingDiagnostics()
        self.time_constraint = TimeConstraint(
            max_duration=max_duration, max_frames=max_frames, max_samples=max_samples
        )

    def __iter__(self):
        self.datapipe = iter(self.datapipe)
        while True:
            yield self._collect_batch()

    def _collect_batch(self):
        self.time_constraint.reset()
        cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            try:
                if self.reuse_cuts_buffer:
                    next_cut = self.reuse_cuts_buffer.popleft()
                else:
                    # If this doesn't raise (typical case), it's not the end: keep processing.
                    next_cut = next(self.datapipe)
            except StopIteration:
                # No more cuts to sample from: if we have a partial batch,
                # we may output it, unless the user requested to drop it.
                # We also check if the batch is "almost there" to override drop_last.
                if cuts and (
                    not self.drop_last or self.time_constraint.close_to_exceeding()
                ):
                    # We have a partial batch and we can return it.
                    self.diagnostics.keep(cuts)
                    return CutSet.from_cuts(cuts)
                else:
                    # There is nothing more to return or it's discarded:
                    # signal the iteration code to stop.
                    self.diagnostics.discard(cuts)
                    raise StopIteration()

            # Track the duration/frames/etc. constraints.
            self.time_constraint.add(next_cut)
            next_num_cuts = len(cuts) + 1

            # Did we exceed the max_frames and max_cuts constraints?
            if not self.time_constraint.exceeded() and (
                self.max_cuts is None or next_num_cuts <= self.max_cuts
            ):
                # No - add the next cut to the batch, and keep trying.
                cuts.append(next_cut)
            else:
                # Yes. Do we have at least one cut in the batch?
                if cuts:
                    # Yes. Return the batch, but keep the currently drawn cut for later.
                    self.reuse_cuts_buffer.append(next_cut)
                    break
                else:
                    # No. We'll warn the user that the constrains might be too tight,
                    # and return the cut anyway.
                    warnings.warn(
                        "The first cut drawn in batch collection violates "
                        "the max_frames, max_cuts, or max_duration constraints - "
                        "we'll return it anyway. "
                        "Consider increasing max_frames/max_cuts/max_duration."
                    )
                    cuts.append(next_cut)

        self.diagnostics.keep(cuts)
        return CutSet.from_cuts(cuts)


class IODataPipe(dp.IterDataPipe):
    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        for cut_idx, batch_idx, batch_size, cut in self.datapipe:
            yield cut_idx, batch_idx, batch_size, cut.load_audio(), cut


class RecombineBatchAfterIO(dp.IterDataPipe):
    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        # This is a buffer that will be re-combining batches.
        # We might possibly get cuts from the same batch out-of-order.
        batches: Dict[int, List[Cut]] = defaultdict(list)
        for cut_idx, batch_idx, batch_size, audio, cut in self.datapipe:
            batches[batch_idx].append((audio, cut))
            if len(batches[batch_idx]) == batch_size:
                audios, cuts = zip(*batches[batch_idx])
                yield audios, CutSet.from_cuts(cuts)
                # Free up the buffer dict after using the batch.
                del batches[batch_idx]


class UnbatchForIO(dp.IterDataPipe):
    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        for batch_idx, cuts_batch in enumerate(self.datapipe):
            batch_size = len(cuts_batch)
            for cut_idx, cut in enumerate(cuts_batch):
                yield cut_idx, batch_idx, batch_size, cut


def classifier_fn(value, num_jobs):
    return value[0] % num_jobs


def build_pipeline(path: str, num_jobs: int = 2):

    # Open cuts manifest for lazy reading
    datapipe = CutsReader(path)

    # Custom Sampling
    datapipe = DurationBatcher(datapipe, max_duration=100)

    # Unbatch the batched to run I/O
    datapipe = UnbatchForIO(datapipe)  # Yield (batch index, dataclass)

    # The following Multiprocessing API may change, and multiprocessing should be added and handled by DataLoader
    # demux and mux would be called by DataLoader. But, I want to explicitly call them to illustrate the functionality

    # Routed prior datapipe to ``num_jobs`` datapipes in round-robin manner
    datapipes = datapipe.demux(
        num_jobs, classifier_fn=partial(classifier_fn, num_jobs=num_jobs)
    )
    for i in range(len(datapipes)):
        datapipes[i] = IODataPipe(datapipes[i])
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    for i in range(num_jobs):
        process, req_queue, res_queue = SpawnProcessForDataPipeline(ctx, datapipes[i])
        process.start()
        datapipes[i] = QueueWrapper(
            IterDataPipeQueueProtocolClient(req_queue, res_queue)
        )
    # DataLoader will also handle stop, join and cleanup subprocesses

    datapipe = dp.Multiplexer(*datapipes)

    datapipe = RecombineBatchAfterIO(datapipe)
    # datapipe = datapipe.collate().tranform(transforms=...).filter(filter_fn=...)
    return datapipe


if __name__ == "__main__":
    cuts_path = Path(
        "./workspace/LibriSpeech/manifests/mini-libri-cuts-train-clean-5.jsonl.gz"
    )

    if not cuts_path.exists():
        print("Downloading Mini LibriSpeech.")
        download_librispeech("./workspace", dataset_parts="mini_librispeech")
        print("Building Mini LibriSpeech manifests.")
        libri = prepare_librispeech(
            corpus_dir="./workspace/LibriSpeech",
            output_dir="./workspace/LibriSpeech/manifests",
        )
        print("Storing CutSet.")
        cuts = CutSet.from_manifests(**libri["train-clean-5"])
        cuts.to_file(cuts_path)

    pipeline = build_pipeline(cuts_path, num_jobs=2)
    for idx, item in enumerate(pipeline):
        print(idx, item)
