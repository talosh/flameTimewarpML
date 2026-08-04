"""
splits.py
=========

Train / val / test assignment over the sequences from `descriptions.py`.

Two ways to get a test set:

1. **Fractional** — carve it out of the same tree with a seeded split. Cheap,
   reproducible, but "held out" only as far as the seed and the split unit go.

2. **External root** (preferred for a real OOS set) — point `test_root` at a
   separate tree. Those sequences are never in the pool that train/val are
   drawn from, so the hold-out is structural rather than statistical.

External roots are guarded, because the dangerous case is a *nested* test root
(`/data/clips` for train, `/data/clips/test` for test): the train scan would
pick up the test folders and quietly train on them. Overlap is detected by
resolved path (symlinks included) and, as a softer signal, by content
signature — a sequence copied to a second location has the same stem/start/
step/count even though its folder differs.

Split granularity is the *folder* by default. Frames within a sequence are
near-duplicates of their neighbours, so splitting at window or sequence level
leaks near-identical content and inflates val/test numbers. Folder-level also
keeps a shot together if a dropped frame fractured it into several sequences.

Reproducibility: split units are *sorted* before the seeded shuffle, so the
result never depends on scan or filesystem order.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .descriptions import Sequence, build_manifest


TRAIN, VAL, TEST = "train", "val", "test"

EXCLUDE, ERROR, IGNORE = "exclude", "error", "ignore"


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def sequence_signature(seq: Sequence) -> tuple:
    """Identity of a run independent of where it lives on disk. Two sequences
    with the same signature in different folders are very likely copies of one
    another (or, occasionally, a coincidence of generic naming)."""
    return (seq.left, seq.tail, seq.start, seq.step, seq.count)


def _real(path: str) -> str:
    return os.path.realpath(path)


def _at_or_under(path_real: str, ancestors: set) -> bool:
    """True if `path_real` is one of `ancestors` or nested inside one."""
    p = path_real
    while True:
        if p in ancestors:
            return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


def find_overlap(
    pool: Iterable[Sequence],
    held_out: Iterable[Sequence],
    *,
    held_out_roots: Iterable[str] = (),
):
    """Find pool sequences that overlap a held-out set.

    Returns (path_overlapping_sequences, content_warnings).

    Path overlap is authoritative: the pool sequence's folder resolves to a
    held-out folder or lives beneath a held-out folder/root — i.e. literally
    the same files. Content overlap is advisory only (same signature, different
    location) and is reported rather than acted on.
    """
    held_out = list(held_out)

    ancestors = {_real(s.folder) for s in held_out}
    ancestors |= {_real(r) for r in held_out_roots}

    # memoise per unique folder — folders are far fewer than sequences
    verdict = {}
    overlapping = []
    for s in pool:
        folder = s.folder
        if folder not in verdict:
            verdict[folder] = _at_or_under(_real(folder), ancestors)
        if verdict[folder]:
            overlapping.append(s)

    overlapping_ids = {id(s) for s in overlapping}
    held_sigs = {}
    for s in held_out:
        held_sigs.setdefault(sequence_signature(s), s.folder)

    warnings = []
    seen = set()
    for s in pool:
        if id(s) in overlapping_ids:
            continue                       # already handled as a path overlap
        sig = sequence_signature(s)
        if sig in held_sigs and sig not in seen:
            seen.add(sig)
            warnings.append(
                f"content signature match: '{s.folder}' looks like a copy of "
                f"'{held_sigs[sig]}' (stem '{s.left}*{s.tail}', "
                f"{s.count} frames from {s.start})"
            )
    return overlapping, warnings


# ---------------------------------------------------------------------------
# Split container
# ---------------------------------------------------------------------------

@dataclass
class Split:
    train: list
    val: list
    test: list
    seed: int
    unit: str
    fractions: tuple
    sources: dict = field(default_factory=dict)        # split name -> "pool" | "external"
    excluded: list = field(default_factory=list)
    overlap_warnings: list = field(default_factory=list)

    def __getitem__(self, name: str) -> list:
        return {TRAIN: self.train, VAL: self.val, TEST: self.test}[name]

    def counts(self) -> dict:
        return {
            "units": {n: self._n_units(self[n]) for n in (TRAIN, VAL, TEST)},
            "sequences": {n: len(self[n]) for n in (TRAIN, VAL, TEST)},
            "frames": {n: self._frames(self[n]) for n in (TRAIN, VAL, TEST)},
        }

    def _n_units(self, seqs):
        key = _unit_key(self.unit)
        return len({key(s) for s in seqs})

    @staticmethod
    def _frames(seqs):
        return sum(s.count for s in seqs)

    def is_externally_held_out(self, name: str) -> bool:
        return self.sources.get(name) == "external"

    def describe(self) -> str:
        c = self.counts()
        lines = []
        for n in (TRAIN, VAL, TEST):
            src = self.sources.get(n, "pool")
            tag = "external root" if src == "external" else f"seeded from pool (seed={self.seed})"
            lines.append(f"  {n:<5} {c['units'][n]:>6} {self.unit}(s)  "
                         f"{c['sequences'][n]:>7} seq  {c['frames'][n]:>10} frames   [{tag}]")
        if self.excluded:
            folders = len({s.folder for s in self.excluded})
            lines.append(f"  excluded from pool: {len(self.excluded)} sequence(s) "
                         f"in {folders} folder(s) overlapping a held-out root")
        return "\n".join(lines)


def _unit_key(unit: str):
    if unit == "folder":
        return lambda s: s.folder
    if unit == "sequence":
        return lambda s: s.seq_id
    raise ValueError(f"unit must be 'folder' or 'sequence', got {unit!r}")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_sequences(
    sequences: Iterable[Sequence],
    *,
    val_sequences: Optional[Iterable[Sequence]] = None,
    test_sequences: Optional[Iterable[Sequence]] = None,
    held_out_roots: Iterable[str] = (),
    on_overlap: str = EXCLUDE,
    fractions: tuple = (0.9, 0.05, 0.05),
    seed: int = 1234,
    unit: str = "folder",
    verbose: bool = True,
) -> Split:
    """Partition `sequences` into train/val/test.

    `val_sequences` / `test_sequences`, when given, come from a separate tree
    and are used verbatim — the corresponding fraction is ignored and the
    remaining fractions are renormalised over what's left. So the common case
    (external test, fractional val) needs no fraction bookkeeping from you.

    `on_overlap` controls what happens when a pool sequence turns out to be the
    same files as a held-out one:
        "exclude" (default) — drop it from the pool and report
        "error"             — raise, for pipelines that want a hard stop
        "ignore"            — leave it (not recommended)
    """
    pool = list(sequences)
    external_val = list(val_sequences) if val_sequences is not None else None
    external_test = list(test_sequences) if test_sequences is not None else None

    if on_overlap not in (EXCLUDE, ERROR, IGNORE):
        raise ValueError(f"on_overlap must be one of "
                         f"{EXCLUDE!r}, {ERROR!r}, {IGNORE!r}; got {on_overlap!r}")
    if min(fractions) < 0:
        raise ValueError(f"fractions must be non-negative, got {fractions}")

    key = _unit_key(unit)

    # ---- guard the pool against held-out overlap --------------------------
    held_out = []
    if external_val:
        held_out += external_val
    if external_test:
        held_out += external_test

    excluded = []
    overlap_warnings = []
    if held_out or list(held_out_roots):
        overlapping, overlap_warnings = find_overlap(
            pool, held_out, held_out_roots=held_out_roots)
        if overlapping:
            folders = sorted({s.folder for s in overlapping})
            msg = (f"{len(overlapping)} sequence(s) in {len(folders)} folder(s) of the "
                   f"training pool resolve to a held-out location "
                   f"(e.g. {folders[0]!r})")
            if on_overlap == ERROR:
                raise ValueError(
                    msg + ". The held-out root appears to be nested inside the "
                    "training root. Move it outside, or use on_overlap='exclude'.")
            if on_overlap == EXCLUDE:
                drop = {id(s) for s in overlapping}
                excluded = overlapping
                pool = [s for s in pool if id(s) not in drop]
                if verbose:
                    print(f"[split] excluding {msg}")
            elif verbose:
                print(f"[split] WARNING: {msg} — kept (on_overlap='ignore')")

    # ---- decide which splits are drawn from the pool ----------------------
    f_tr, f_va, f_te = fractions
    internal = [(TRAIN, f_tr)]
    if external_val is None:
        internal.append((VAL, f_va))
    if external_test is None:
        internal.append((TEST, f_te))

    # group pool into split units; sort for scan-order independence
    units = {}
    for s in pool:
        units.setdefault(key(s), []).append(s)
    unit_keys = sorted(units.keys())

    rng = random.Random(seed)              # isolated RNG; never touches global state
    rng.shuffle(unit_keys)

    n = len(unit_keys)
    total = sum(f for _, f in internal)
    assigned = {TRAIN: [], VAL: [], TEST: []}

    # every internal split but train is rounded from its share; train takes the
    # remainder so no unit is ever dropped
    tail_counts = []
    remaining = n
    for name, frac in internal[1:]:
        k = round(n * frac / total) if total > 0 else 0
        k = max(0, min(k, remaining))
        tail_counts.append((name, k))
        remaining -= k

    n_train = remaining
    assigned[TRAIN] = _gather(units, unit_keys[:n_train])
    cursor = n_train
    for name, k in tail_counts:
        assigned[name] = _gather(units, unit_keys[cursor:cursor + k])
        cursor += k

    if external_val is not None:
        assigned[VAL] = external_val
    if external_test is not None:
        assigned[TEST] = external_test

    split = Split(
        train=assigned[TRAIN],
        val=assigned[VAL],
        test=assigned[TEST],
        seed=seed,
        unit=unit,
        fractions=fractions,
        sources={
            TRAIN: "pool",
            VAL: "external" if external_val is not None else "pool",
            TEST: "external" if external_test is not None else "pool",
        },
        excluded=excluded,
        overlap_warnings=overlap_warnings,
    )

    if verbose:
        if n and n_train == n and len(internal) > 1:
            print(f"[split] warning: only {n} {unit}(s) in pool; "
                  f"fractional split(s) rounded to empty")
        print(f"[split] {n} pool {unit}(s)")
        print(split.describe())
        for w in overlap_warnings[:5]:
            print(f"[split] note: {w}")
        if len(overlap_warnings) > 5:
            print(f"[split] ... and {len(overlap_warnings) - 5} more content matches")

    return split


def _gather(units: dict, keys) -> list:
    out = []
    for k in keys:
        out.extend(units[k])
    return out


# ---------------------------------------------------------------------------
# Convenience: roots -> Split
# ---------------------------------------------------------------------------

def build_splits_from_roots(
    train_root: str,
    *,
    val_root: Optional[str] = None,
    test_root: Optional[str] = None,
    fractions: tuple = (0.9, 0.05, 0.05),
    seed: int = 1234,
    unit: str = "folder",
    on_overlap: str = EXCLUDE,
    verbose: bool = True,
    **manifest_kwargs,
) -> Split:
    """Build manifests for the given roots and return the resulting Split.

    Each root is scanned and cached independently (its own manifest file), so a
    read-only test mount is fine — the manifest write failure there is
    non-fatal. Any root given here is registered as held-out, so nesting is
    caught even when the nested tree contributes no sequences of its own.
    """
    train_manifest = build_manifest(train_root, verbose=verbose, **manifest_kwargs)

    val_seqs = None
    if val_root is not None:
        val_seqs = build_manifest(val_root, verbose=verbose, **manifest_kwargs).sequences

    test_seqs = None
    if test_root is not None:
        test_seqs = build_manifest(test_root, verbose=verbose, **manifest_kwargs).sequences

    held_roots = [r for r in (val_root, test_root) if r is not None]

    return split_sequences(
        train_manifest.sequences,
        val_sequences=val_seqs,
        test_sequences=test_seqs,
        held_out_roots=held_roots,
        on_overlap=on_overlap,
        fractions=fractions,
        seed=seed,
        unit=unit,
        verbose=verbose,
    )
