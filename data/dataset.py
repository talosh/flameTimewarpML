"""
dataset.py
==========

Map-style Dataset keyed by SampleSpec. `__getitem__` resolves one training
sample: load the three frames (through the per-worker cache), apply the plan's
rotation, apply seeded shape-preserving flips, and return the triplet.

Division of labour, settled over the design:
  * Rotation (0/±90) is a *planner* decision (it drives bucketing); the dataset
    only applies it — a cheap transpose *after* the cache lookup, so cached
    frames stay rotation-invariant and one cache entry serves every rotation.
  * Flips (h/v/channel) are shape-preserving geometric aug, drawn here from a
    per-sample seed derived from (epoch_seed, spec) so they're deterministic and
    identical across the triplet — no worker RNG state, fully reproducible.
  * Photometric aug (exposure, noise) and colour-space ops stay on GPU in the
    training loop, exactly where the original does them. They are NOT here.

Returns CPU tensors at the resized (possibly transposed) size — sizes vary
within a batch's tolerance band; collate pads them. torch/OIIO are imported
lazily (here and in io.py), so `import data` works without either; only calling
into the pixel path needs them.
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Iterable

from .descriptions import Sequence
from .sampler import SampleSpec, resized_hw, _seed_for, ROT_NONE, ROT_CW, ROT_CCW
from .io import default_reader
from .cache import FrameCache


def rotate_chw(t, rotation: int):
    """Rotate a (C, H, W) tensor. +90 = clockwise, -90 = counter-clockwise.
    Both swap H<->W, matching the planner's axis swap; the sign only sets flip
    direction. torch.rot90's positive k is counter-clockwise."""
    if rotation == ROT_NONE:
        return t
    import torch
    if rotation == ROT_CCW:
        return torch.rot90(t, 1, dims=(1, 2))
    if rotation in (ROT_CW, 270):
        return torch.rot90(t, -1, dims=(1, 2))
    return t


class TimewarpDataset:
    def __init__(
        self,
        sequences: Iterable[Sequence],
        *,
        frame_size: int,
        multiple: int = 16,
        channels: int = 3,
        reader: Callable = default_reader,
        cache: Optional[FrameCache] = None,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        cflip_prob: float = 0.0,
        seed: int = 1234,
        epoch: int = 0,
    ):
        self.by_id = {s.seq_id: s for s in sequences}
        self.frame_size = frame_size
        self.multiple = multiple
        self.channels = channels
        self.reader = reader
        self.cache = cache            # per-worker; set by worker_init_fn, or here for num_workers=0
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.cflip_prob = cflip_prob
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    # -- frame loading via cache -----------------------------------------
    def _get_frame(self, path: str, out_h: int, out_w: int):
        def loader():
            return self.reader(path, out_h, out_w, channels=self.channels)
        if self.cache is not None:
            return self.cache.get_or_load(path, loader)
        return loader()

    # -- augmentation (seeded, identical across the triplet) -------------
    def _augment(self, frames, rng: random.Random):
        import torch  # noqa: F401
        do_h = rng.random() < self.hflip_prob
        do_v = rng.random() < self.vflip_prob
        do_c = rng.random() < self.cflip_prob
        if not (do_h or do_v or do_c):
            return frames
        dims = []
        if do_c:
            dims.append(0)
        if do_v:
            dims.append(1)
        if do_h:
            dims.append(2)
        return [f.flip(dims) for f in frames]

    def __getitem__(self, spec: SampleSpec) -> dict:
        seq = self.by_id[spec.seq_id]
        # canonical (pre-rotation) size: short side -> frame_size. Rotating the
        # canonical frame equals resized_hw(..., rotation) by construction.
        out_h, out_w = resized_hw(seq.height, seq.width, ROT_NONE,
                                  self.frame_size, self.multiple)

        frames = [self._get_frame(seq.path_at(i), out_h, out_w)
                  for i in (spec.start, spec.gt, spec.end)]
        frames = [rotate_chw(f, spec.rotation) for f in frames]

        rng = random.Random(_seed_for(
            self.seed, self.epoch, spec.seq_id,
            spec.start, spec.gt, spec.end, spec.rotation))
        frames = self._augment(frames, rng)

        img0, img1, img2 = frames
        return {"img0": img0, "img1": img1, "img2": img2,
                "ratio": float(spec.ratio), "spec": spec}

    def __len__(self) -> int:
        # not used with a batch_sampler, but handy for sanity
        return sum(1 for _ in self.by_id)
