:tocdepth: 3

Stateful DataLoader
===================

.. automodule:: torchdata.stateful_dataloader

StatefulDataLoader is a drop-in replacement for `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ which offers ``state_dict`` / ``load_state_dict`` methods for handling mid-epoch checkpointing which operate on the previous/next iterator requested from the dataloader (resp.).

By default, the state includes the number of batches yielded and uses this to naively fast-forward the sampler (map-style) or the dataset (iterable-style). However if the sampler and/or dataset include ``state_dict`` / ``load_state_dict`` methods, then it will call them during its own ``state_dict`` / ``load_state_dict`` calls. Under the hood, :class:`StatefulDataLoader` handles aggregation and distribution of state across multiprocess workers (but not across ranks).

.. autoclass:: StatefulDataLoader
    :members:

Stateful sampler for distributed training
-----------------------------------------

``torchdata.stateful_dataloader.sampler.StatefulDistributedSampler`` is a subclass of
`torch.utils.data.distributed.DistributedSampler <https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>`_
that implements ``state_dict`` / ``load_state_dict``, so :class:`StatefulDataLoader` can save and restore its position
mid-epoch. It takes the same constructor arguments as ``DistributedSampler``.

* ``state_dict()`` returns the number of indices yielded so far in the current epoch (``"yielded"``), the current
  ``"epoch"`` (as set by ``set_epoch``) and a pending ``"NEXT_YIELDED"`` value.
* ``load_state_dict()`` restores the epoch and, when iteration next starts, skips the indices that were already
  yielded. It raises ``ValueError`` if the state has no ``"epoch"`` or ``"yielded"`` entry, or if ``"yielded"`` is negative.

Each rank owns its own sampler, so the state of every rank's dataloader must be saved and restored on that rank.
