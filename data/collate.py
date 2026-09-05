"""
collate.py
==========

Collate a list of dataset samples into a padded batch. The batch_sampler groups
samples of one orientation and a bounded long-side spread, so padding is
single-axis and small: pad each frame to the batch's max (H, W) with zeros on
the bottom/right, and emit a validity mask (1 = real pixel, 0 = padding).

The three frames of a sample share a size, so one mask per sample covers all of
img0/img1/img2. Zero padding is deliberate — the model is meant to read it as
image content, and the mask is passed so the loss can ignore it behind a flag.

torch imported lazily.
"""

from __future__ import annotations

from typing import List


def collate_timewarp(samples: List[dict], return_mask: bool = True) -> dict:
    import torch
    import torch.nn.functional as F

    hs = [s["img0"].shape[1] for s in samples]
    ws = [s["img0"].shape[2] for s in samples]
    Ht, Wt = max(hs), max(ws)

    def pad(t):
        c, h, w = t.shape
        if h == Ht and w == Wt:
            return t
        return F.pad(t, (0, Wt - w, 0, Ht - h))     # (left, right, top, bottom)

    batch = {
        "img0": torch.stack([pad(s["img0"]) for s in samples]),
        "img1": torch.stack([pad(s["img1"]) for s in samples]),
        "img2": torch.stack([pad(s["img2"]) for s in samples]),
        "ratio": torch.tensor([s["ratio"] for s in samples], dtype=torch.float32),
        "specs": [s["spec"] for s in samples],
    }
    if return_mask:
        mask = torch.zeros((len(samples), 1, Ht, Wt), dtype=batch["img0"].dtype)
        for i, (h, w) in enumerate(zip(hs, ws)):
            mask[i, 0, :h, :w] = 1.0
        batch["mask"] = mask
    return batch


def make_worker_init_fn(cache_items: int = 256, cache_bytes=None):
    """Give each DataLoader worker its own FrameCache and a per-worker RNG seed."""
    from .cache import FrameCache

    def _init(worker_id: int):
        import signal
        # Workers must NOT inherit the training process's SIGINT handler. On Ctrl+C
        # the whole process group is signalled; if each worker ran the checkpoint
        # saver it would torch.save the (CUDA) state dict concurrently to the same
        # file -> corrupted weights. Ignore SIGINT in workers; the main process
        # performs the single graceful save.
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass  # only settable from a process's main thread; safe to skip otherwise
        import random
        import numpy as np
        import torch
        info = torch.utils.data.get_worker_info()
        info.dataset.cache = FrameCache(max_items=cache_items, max_bytes=cache_bytes)
        base = int(info.seed) % (2 ** 31)
        random.seed(base)
        np.random.seed(base % (2 ** 32))
        torch.manual_seed(base)

    return _init


def build_dataloader(
    dataset,
    batch_sampler,
    *,
    num_workers: int = 8,
    cache_items: int = 256,
    cache_bytes=None,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    return_mask: bool = True,
):
    """Wire dataset + batch_sampler + collate + per-worker cache into a DataLoader.

    For num_workers == 0 the cache can't be set via worker_init_fn, so it's
    attached to the dataset directly here.
    """
    import torch
    from .cache import FrameCache

    if num_workers == 0:
        if dataset.cache is None:
            dataset.cache = FrameCache(max_items=cache_items, max_bytes=cache_bytes)
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=lambda s: collate_timewarp(s, return_mask=return_mask),
            num_workers=0, pin_memory=pin_memory,
        )

    return torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler,
        collate_fn=lambda s: collate_timewarp(s, return_mask=return_mask),
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
        worker_init_fn=make_worker_init_fn(cache_items, cache_bytes),
    )