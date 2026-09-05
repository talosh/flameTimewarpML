"""
sampler.py
==========

The epoch planner. Turns split sequences into a stream of size-grouped,
minimally-padded batches — the `batch_sampler` a DataLoader consumes, with the
reuse-pool sitting on top of that DataLoader.

The chain of ideas from descriptions.py pays off here:

* Bucketing is a *per-sequence* property. After an aspect-preserving resize
  (short side -> frame_size, snapped to /16), one axis is always frame_size and
  only the long axis varies. A per-epoch rotation in {0, +90, -90} swaps which
  axis that is (portrait <-> landscape). So a sequence's bucket — (orientation,
  long-side band) — is known from its native size + assigned rotation, without
  ever looking at an individual window. Rotation is decided here and *applied*
  by the dataset (a cheap transpose after the cache lookup), so cached frames
  stay rotation-invariant.

* Windows are never listed. A bucket holds sequences; a batch draws
  `batch_size` windows from a bucket's sequences via `Sequence.window_at(k)` on
  seeded random indices. Nothing scales with the ~10^8 window count — only with
  the sequence count (which we already hold) and the per-epoch step budget.

* Padding is bounded, not eliminated. Sequences are bucketed by long side into
  geometric bands of width (1 + pad_tolerance), so any two frames sharing a
  bucket differ in long side by at most that factor. Collate pads each batch to
  its members' max long side -> waste is <= pad_tolerance (default 10%), on the
  long axis only. Zero padding would mean exact-size buckets and ragged/under-
  full batches; we chose full batches + a small, bounded pad + an optional mask.

* DDP needs no data coordination. Each rank plans over its own disjoint
  sequence shard and runs the same fixed number of steps, so gradients
  all-reduce normally with no scatter, no uneven-input deadlock. The step-budget
  epoch is what makes this clean.

Determinism: everything derives from (seed, epoch, rank) through isolated RNGs.
Consuming the sampler in order reproduces exactly; the reuse-pool's optional
random serve is the only added nondeterminism, and reuse=1 + in-order recovers
this canonical order.

No torch here — pure planning over the Sequence objects from descriptions.py.
"""

from __future__ import annotations

import math
import bisect
import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from .descriptions import Sequence


LANDSCAPE, PORTRAIT = "landscape", "portrait"
ROT_NONE, ROT_CW, ROT_CCW = 0, 90, -90
ROTATIONS = (ROT_NONE, ROT_CW, ROT_CCW)

DEFAULT_FRAME_SIZE = 448
DEFAULT_MULTIPLE = 16
DEFAULT_PAD_TOLERANCE = 0.10      # <=10% wasted pixels on the padded (long) axis
DEFAULT_ROTATION_PROB = 0.5       # P(any rotation); split evenly across +/-90


# ---------------------------------------------------------------------------
# Sample spec — one training item, picklable for DataLoader workers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleSpec:
    """One sample the dataset resolves. `rotation` is applied by the dataset
    after decode; `start`/`gt`/`end` are frame indices within `seq_id`. The
    resized shape is recomputed by the dataset from the sequence's native size
    + rotation via the shared geometry helpers, so it always matches what the
    planner bucketed on."""
    seq_id: str
    start: int
    gt: int
    end: int
    ratio: float
    rotation: int


# ---------------------------------------------------------------------------
# Geometry — shared with the dataset so both agree on shapes
# ---------------------------------------------------------------------------

def snap(x: float, multiple: int = DEFAULT_MULTIPLE) -> int:
    return max(multiple, int(round(x / multiple)) * multiple)


def rotated_hw(h: int, w: int, rotation: int) -> tuple:
    """Native (h, w) after a plan-time rotation. +/-90 swap the axes."""
    if rotation in (ROT_CW, ROT_CCW, 270):
        return w, h
    return h, w


def resized_hw(h: int, w: int, rotation: int,
               frame_size: int = DEFAULT_FRAME_SIZE,
               multiple: int = DEFAULT_MULTIPLE) -> tuple:
    """Aspect-preserving target (H, W): short side -> frame_size, long side
    scaled and snapped to `multiple`. frame_size is assumed already a multiple.
    """
    H, W = rotated_hw(h, w, rotation)
    if H <= 0 or W <= 0:
        return frame_size, frame_size
    short = min(H, W)
    long = max(H, W)
    new_long = snap(long * (frame_size / short), multiple)
    if H <= W:                       # height is the short axis -> landscape
        return frame_size, new_long
    return new_long, frame_size      # width is the short axis -> portrait


def family_and_long(h: int, w: int, rotation: int,
                    frame_size: int = DEFAULT_FRAME_SIZE,
                    multiple: int = DEFAULT_MULTIPLE) -> tuple:
    """(orientation, long_side) of the resized frame — the shape signature that
    drives bucketing."""
    H, W = resized_hw(h, w, rotation, frame_size, multiple)
    family = LANDSCAPE if W >= H else PORTRAIT
    return family, max(H, W)


def _band(long_side: int, frame_size: int, tol: float) -> int:
    """Geometric long-side band index. Bands span a (1+tol) ratio, so within a
    band the long side varies by <= tol -> padding to the batch max wastes <= tol."""
    ratio = max(long_side, frame_size) / frame_size
    return int(math.log(ratio + 1e-12) / math.log(1.0 + tol))


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _seed_for(seed: int, *parts) -> int:
    """Stable (non-salted) seed from (seed, epoch, rank, tag). Python's hash()
    is salted per process, so we go through sha1 for cross-run reproducibility."""
    raw = "|".join(str(p) for p in (seed, *parts))
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12], 16)


# ---------------------------------------------------------------------------
# Rotation assignment
# ---------------------------------------------------------------------------

def normalize_rotation_weights(
    rotation_prob: float = DEFAULT_ROTATION_PROB,
    weights: Optional[dict] = None,
) -> dict:
    if weights is not None:
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError("rotation weights must sum to > 0")
        return {int(k): v / total for k, v in weights.items()}
    p = max(0.0, min(1.0, rotation_prob))
    return {ROT_NONE: 1.0 - p, ROT_CW: p / 2.0, ROT_CCW: p / 2.0}


def assign_rotations(sequences, rng: random.Random, weights: dict) -> dict:
    """Draw a rotation per sequence for this epoch. Weighted, seeded via `rng`."""
    keys = list(weights.keys())
    cum = list(itertools.accumulate(weights[k] for k in keys))
    total = cum[-1]
    out = {}
    for seq in sequences:
        r = rng.random() * total
        out[seq.seq_id] = keys[bisect.bisect_right(cum, r)]
    return out


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

class _Bucket:
    __slots__ = ("family", "band", "seqs", "rots", "longs", "wcounts", "_cum", "total")

    def __init__(self, family, band):
        self.family = family
        self.band = band
        self.seqs = []
        self.rots = []
        self.longs = []
        self.wcounts = []
        self._cum = None
        self.total = 0

    def add(self, seq, rot, long, wcount):
        self.seqs.append(seq)
        self.rots.append(rot)
        self.longs.append(long)
        self.wcounts.append(wcount)

    def finalize(self):
        self._cum = list(itertools.accumulate(self.wcounts))
        self.total = self._cum[-1] if self._cum else 0

    def _draw_one(self, rng) -> int:
        r = rng.randrange(self.total)
        return bisect.bisect_right(self._cum, r)

    def sample(self, k: int, rng) -> list:
        """Indices of `k` sequences, weighted by window count. Distinct when the
        bucket has enough sequences (max in-batch diversity, no content overlap
        from near-duplicate windows of one sequence); repeats only to fill a
        bucket smaller than the batch."""
        n = len(self.seqs)
        if n == 0:
            return []
        if n <= k:
            idxs = list(range(n))
            while len(idxs) < k:
                idxs.append(self._draw_one(rng))
            return idxs
        chosen, out, attempts = set(), [], 0
        cap = 20 * k
        while len(out) < k:
            i = self._draw_one(rng)
            if i not in chosen:
                chosen.add(i)
                out.append(i)
            attempts += 1
            if attempts > cap:                      # pathological weight skew
                remaining = [j for j in range(n) if j not in chosen]
                rng.shuffle(remaining)
                out.extend(remaining[: k - len(out)])
                break
        return out


def build_buckets(
    sequences,
    rotations: dict,
    *,
    frame_size: int = DEFAULT_FRAME_SIZE,
    multiple: int = DEFAULT_MULTIPLE,
    pad_tolerance: float = DEFAULT_PAD_TOLERANCE,
    bidirectional: bool = True,
    window_mode: str = "full",
    max_long_side: Optional[int] = None,
    warnings: Optional[list] = None,
) -> dict:
    """Group sequences into (family, long-band) buckets under their assigned
    rotations. Sequences whose resized long side exceeds `max_long_side` (would
    OOM at full frame) are dropped with a warning. Sequences with no windows or
    unknown native size are skipped."""
    warn = warnings if warnings is not None else []
    buckets: dict = {}
    n_no_size = n_too_big = n_no_window = 0

    for seq in sequences:
        if seq.height is None or seq.width is None:
            n_no_size += 1
            continue
        rot = rotations[seq.seq_id]
        family, long = family_and_long(seq.height, seq.width, rot, frame_size, multiple)
        if max_long_side is not None and long > max_long_side:
            n_too_big += 1
            continue
        wcount = seq.num_windows(bidirectional, window_mode)
        if wcount <= 0:
            n_no_window += 1
            continue
        key = (family, _band(long, frame_size, pad_tolerance))
        b = buckets.get(key)
        if b is None:
            b = buckets[key] = _Bucket(*key)
        b.add(seq, rot, long, wcount)

    for b in buckets.values():
        b.finalize()

    if n_no_size:
        warn.append(f"{n_no_size} sequence(s) skipped: native size unknown "
                    f"(build manifest with read_headers=True)")
    if n_too_big:
        warn.append(f"{n_too_big} sequence(s) skipped: resized long side > "
                    f"max_long_side={max_long_side}")
    if n_no_window:
        warn.append(f"{n_no_window} sequence(s) skipped: too short for a window")
    return buckets


# ---------------------------------------------------------------------------
# DDP sharding
# ---------------------------------------------------------------------------

def shard_sequences(sequences, rank: int, world_size: int) -> list:
    """Disjoint, balanced shard for one rank. Sequences are sorted by id (stable,
    scan-order-free) then round-robin assigned, so counts differ by at most one
    across ranks and every sequence lands on exactly one rank."""
    if world_size <= 1:
        return list(sequences)
    ordered = sorted(sequences, key=lambda s: s.seq_id)
    return ordered[rank::world_size]


# ---------------------------------------------------------------------------
# The batch sampler
# ---------------------------------------------------------------------------

class TimewarpBatchSampler:
    """A `batch_sampler` for DataLoader. Each `__iter__` yields
    `steps_per_epoch` batches, every batch a list of `batch_size` SampleSpec of
    one orientation and a bounded long-side spread.

    Buckets are (re)built every epoch because rotations are redrawn per epoch.
    Call `set_epoch(e)` before each epoch (or pass a fixed epoch) so the seed —
    hence rotations, bucket picks and window picks — advances deterministically.
    """

    def __init__(
        self,
        sequences: Iterable[Sequence],
        *,
        batch_size: int,
        frame_size: int = DEFAULT_FRAME_SIZE,
        multiple: int = DEFAULT_MULTIPLE,
        pad_tolerance: float = DEFAULT_PAD_TOLERANCE,
        bidirectional: bool = True,
        window_mode: str = "full",
        rotation_prob: float = DEFAULT_ROTATION_PROB,
        rotation_weights: Optional[dict] = None,
        max_long_side: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        seed: int = 1234,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
        verbose: bool = False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size
        self.frame_size = snap(frame_size, multiple)
        self.multiple = multiple
        self.pad_tolerance = pad_tolerance
        self.bidirectional = bidirectional
        self.window_mode = window_mode
        self.rotation_weights = normalize_rotation_weights(rotation_prob, rotation_weights)
        self.max_long_side = max_long_side
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self.verbose = verbose

        self.sequences = shard_sequences(sequences, rank, world_size)
        if not self.sequences:
            raise ValueError(f"rank {rank}/{world_size}: empty sequence shard")

        # a nominal "epoch" covers dataset-many windows (with replacement, so not
        # exact coverage) unless the caller fixes a step budget
        self._nominal_steps = self._estimate_steps()
        self.steps_per_epoch = steps_per_epoch or self._nominal_steps

    # -- coverage estimate uses a rotation-free bucketing just for the count --
    def _estimate_steps(self) -> int:
        total = sum(s.num_windows(self.bidirectional, self.window_mode)
                    for s in self.sequences
                    if s.height is not None and s.width is not None
                    and s.num_windows(self.bidirectional, self.window_mode) > 0)
        return max(1, math.ceil(total / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def build_epoch_buckets(self) -> tuple:
        """(bucket_list, warnings) for the current epoch — exposed for reporting."""
        rot_rng = random.Random(_seed_for(self.seed, self.epoch, self.rank, "rot"))
        rotations = assign_rotations(self.sequences, rot_rng, self.rotation_weights)
        warnings: list = []
        buckets = build_buckets(
            self.sequences, rotations,
            frame_size=self.frame_size, multiple=self.multiple,
            pad_tolerance=self.pad_tolerance, bidirectional=self.bidirectional,
            window_mode=self.window_mode,
            max_long_side=self.max_long_side, warnings=warnings,
        )
        return list(buckets.values()), warnings

    def __iter__(self) -> Iterator[list]:
        bucket_list, warnings = self.build_epoch_buckets()
        if not bucket_list:
            raise RuntimeError(f"rank {self.rank}: no usable buckets this epoch "
                               f"({'; '.join(warnings) or 'no sequences'})")

        if self.verbose and self.rank == 0:
            self._report(bucket_list, warnings)

        # weighted bucket choice, by total window count
        bucket_cum = list(itertools.accumulate(b.total for b in bucket_list))
        grand_total = bucket_cum[-1]
        batch_rng = random.Random(_seed_for(self.seed, self.epoch, self.rank, "batch"))

        for _ in range(self.steps_per_epoch):
            r = batch_rng.randrange(grand_total)
            b = bucket_list[bisect.bisect_right(bucket_cum, r)]
            idxs = b.sample(self.batch_size, batch_rng)
            batch = []
            for i in idxs:
                seq = b.seqs[i]
                rot = b.rots[i]
                kwin = batch_rng.randrange(b.wcounts[i])
                w = seq.window_at(kwin, self.bidirectional, self.window_mode)
                batch.append(SampleSpec(seq.seq_id, w.start, w.gt, w.end, w.ratio, rot))
            yield batch

    # -- diagnostics ------------------------------------------------------
    def _report(self, bucket_list, warnings) -> None:
        n_seq = sum(len(b.seqs) for b in bucket_list)
        worst_pad = 0.0
        for b in bucket_list:
            if b.longs:
                worst_pad = max(worst_pad, (max(b.longs) - min(b.longs)) / max(b.longs))
        fams = {}
        for b in bucket_list:
            fams.setdefault(b.family, 0)
            fams[b.family] += len(b.seqs)
        print(f"[sampler] epoch {self.epoch} rank {self.rank}/{self.world_size}: "
              f"{n_seq} seq in {len(bucket_list)} buckets "
              f"({', '.join(f'{k}:{v}' for k, v in fams.items())}), "
              f"{self.steps_per_epoch} steps x {self.batch_size}, "
              f"mode={self.window_mode}{'' if self.bidirectional else ' (uni)'}, "
              f"worst in-batch pad ~{worst_pad*100:.1f}%")
        for w in warnings[:4]:
            print(f"[sampler]   note: {w}")