"""
descriptions.py
===============

Foundation layer for the timewarp dataset. Turns a directory tree of EXR
frames into a set of *sequences* and exposes windowed training samples over
them, without ever materialising the (potentially hundreds of millions of)
individual samples.

Two ideas make this scale:

1. A folder is NOT a clip. The atomic unit is a *contiguous frame run*
   (a "sequence"): frames sharing a stem whose numbers advance by a constant
   step. A single folder may hold several sequences (different shots dumped
   together, or one shot fractured by a dropped frame). Windows are built
   inside a sequence, so a window can never straddle a boundary/cut — it is a
   structural guarantee, not a post-hoc filter.

2. Windows are addressed by integer index, not stored. For a sequence of
   length L and max window W the number of windows is a closed form; the k-th
   window is decoded on demand. The planner (sampler.py, later) can therefore
   count, shuffle and sample windows with plain integers and never build a list.

No torch / tensor code lives here. Only the manifest builder touches disk
(scan + one header read per sequence), and OpenImageIO is imported lazily so
the pure logic below is testable and usable without it.
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional

# Trailing-digit frame field: the LAST run of digits in the basename (before
# the extension). This is the VFX convention and makes version tokens safe:
# "shot.v002.0001.exr" -> frame 0001, stem "shot.v002.", not the v002.
_DIGITS = re.compile(r"\d+")

MANIFEST_VERSION = 3
DEFAULT_SKIP_COMPONENTS = ("preview", "eval")
DEFAULT_MIN_LENGTH = 3          # a window needs at least 3 frames
DEFAULT_MAX_WINDOW = 12
FAST_TOKEN = "fast"             # folders tagged 'fast' are restricted to 3-frame windows


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedName:
    left: str        # everything before the frame field
    tail: str        # everything after the frame field, incl. extension (usually ".exr")
    number: int      # numeric value of the frame field
    ndigits: int     # width of the frame field as written ("0007" -> 4)

    @property
    def group_key(self) -> tuple:
        # Two files belong to the same stem group iff their non-numeric
        # surroundings match. Padding width is deliberately NOT part of the key
        # so that overflow (0999 -> 1000, or 9999 -> 10000) keeps a run whole.
        return (self.left, self.tail)


def parse_frame_name(filename: str) -> Optional[ParsedName]:
    """Parse one basename into (left, tail, number, ndigits), or None if it has
    no trailing digit field (e.g. 'background.exr')."""
    base, ext = os.path.splitext(filename)
    matches = list(_DIGITS.finditer(base))
    if not matches:
        return None
    m = matches[-1]                      # LAST digit run == frame field
    return ParsedName(
        left=base[: m.start()],
        tail=base[m.end():] + ext,
        number=int(m.group()),
        ndigits=m.end() - m.start(),
    )


# ---------------------------------------------------------------------------
# Sequence model
# ---------------------------------------------------------------------------

@dataclass
class Sequence:
    """A contiguous, constant-step run of frames.

    Frame paths are stored compactly (reconstructed on demand) whenever the
    run follows a clean `left + zero-padded-number + tail` pattern — which,
    having just detected it, it does by construction. This keeps both the
    manifest file and the in-RAM footprint tiny at millions-of-frames scale:
    a sequence is a handful of fields regardless of length. `explicit` holds a
    verbatim path list only for the rare run that fails reconstruction.
    """
    seq_id: str
    folder: str
    left: str
    tail: str
    pad: int                 # zero-pad width used for reconstruction
    step: int                # constant frame-number step (1 for on-1s, 2 for on-2s, ...)
    start: int               # first frame number
    count: int               # number of frames
    height: Optional[int] = None
    width: Optional[int] = None
    max_window: int = DEFAULT_MAX_WINDOW   # already clamped to count / fast-token
    explicit: Optional[list[str]] = None   # fallback: full basenames, or None

    def __len__(self) -> int:
        return self.count

    def number_at(self, i: int) -> int:
        return self.start + i * self.step

    def basename_at(self, i: int) -> str:
        if self.explicit is not None:
            return self.explicit[i]
        n = self.number_at(i)
        return f"{self.left}{n:0{self.pad}d}{self.tail}"

    def path_at(self, i: int) -> str:
        return os.path.join(self.folder, self.basename_at(i))

    # ---- windowing (closed-form index space) ----------------------------

    def num_windows(self, bidirectional: bool = True, mode: str = "full") -> int:
        return num_windows(self.count, self.max_window, bidirectional, mode)

    def window_at(self, k: int, bidirectional: bool = True, mode: str = "full") -> "WindowSpec":
        return window_at(self, k, bidirectional, mode)

    def iter_windows(self, bidirectional: bool = True, mode: str = "full") -> Iterator["WindowSpec"]:
        for k in range(self.num_windows(bidirectional, mode)):
            yield window_at(self, k, bidirectional, mode)

    # ---- (de)serialisation ---------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Sequence":
        return Sequence(**d)


@dataclass(frozen=True)
class WindowSpec:
    """One training sample: indices into a sequence's frames plus the temporal
    ratio. Rotation is NOT here — the epoch planner attaches it per (sequence,
    epoch). `start`/`gt`/`end` are frame indices within `seq_id`."""
    seq_id: str
    start: int
    gt: int
    end: int
    ratio: float


# ---------------------------------------------------------------------------
# Window index space
# ---------------------------------------------------------------------------
#
# Canonical enumeration for a run of length L, effective max window W = min(W, L),
# with `dir_mult` = 2 when bidirectional else 1:
#
#   for w in 3..W:                       # window covers w consecutive frames
#     for pos in 0..(L - w):             # L - w + 1 positions
#       for gt_off in 1..(w - 2):        # interior target frames
#         forward : start=pos,  end=pos+w-1, gt=pos+gt_off, ratio=gt_off/(w-1)
#         backward: start=pos+w-1, end=pos, gt=pos+gt_off, ratio=1-gt_off/(w-1)
#
# count(w) = (L - w + 1) * (w - 2) * dir_mult
#
# Backward mirrors the original bw_item: endpoints swap, gt frame is unchanged,
# ratio complements. Forward/backward are interleaved per (pos, gt_off) so the
# two views of a sample sit adjacent in the index — handy for the planner.

def _effective_window(seq_len: int, max_window: int) -> int:
    return min(max_window, seq_len)


FULL, FIXED = "full", "fixed"


def num_windows(seq_len: int, max_window: int, bidirectional: bool = True,
                mode: str = FULL) -> int:
    """Number of windows over a run of length `seq_len`.

    mode="full"  (timewarp): every window size 3..max_window at every position,
                 every interior gt frame. The original enumeration.
    mode="fixed" (stab): ONLY the max_window-sized window, slid over the run,
                 with every interior gt frame. Fewer, longer-baseline samples.
    """
    W = _effective_window(seq_len, max_window)
    if seq_len < 3 or W < 3:
        return 0
    dir_mult = 2 if bidirectional else 1
    if mode == FIXED:
        return (seq_len - W + 1) * (W - 2) * dir_mult
    total = 0
    for w in range(3, W + 1):
        total += (seq_len - w + 1) * (w - 2)
    return total * dir_mult


def window_at(seq: "Sequence", k: int, bidirectional: bool = True,
              mode: str = FULL) -> WindowSpec:
    L = seq.count
    W = _effective_window(L, seq.max_window)
    dir_mult = 2 if bidirectional else 1
    total = num_windows(L, seq.max_window, bidirectional, mode)
    if not (0 <= k < total):
        raise IndexError(f"window index {k} out of range [0, {total}) for {seq.seq_id}")

    if mode == FIXED:
        w = W                      # single band: the max window only
    else:
        # locate the window size band containing k
        for w in range(3, W + 1):
            band = (L - w + 1) * (w - 2) * dir_mult
            if k < band:
                break
            k -= band

    if bidirectional:
        pair_idx, direction = divmod(k, 2)
    else:
        pair_idx, direction = k, 0

    n_gt = w - 2
    pos, gt_off = divmod(pair_idx, n_gt)
    gt_off += 1                                   # 1..(w-2)

    start = pos
    end = pos + w - 1
    gt = pos + gt_off
    ratio = gt_off / (w - 1)

    if direction == 1:                            # backward view
        start, end = end, start
        ratio = 1.0 - ratio

    return WindowSpec(seq_id=seq.seq_id, start=start, gt=gt, end=end, ratio=ratio)


# ---------------------------------------------------------------------------
# Sequence detection (pure — no I/O)
# ---------------------------------------------------------------------------

def _make_seq_id(folder: str, left: str, tail: str, start: int, step: int) -> str:
    raw = f"{folder}|{left}|{tail}|{start}|{step}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def detect_sequences_in_folder(
    folder: str,
    filenames: list[str],
    *,
    expected_step: Optional[int] = None,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_window: int = DEFAULT_MAX_WINDOW,
    fast_token: Optional[str] = FAST_TOKEN,
    warnings: Optional[list] = None,
) -> list[Sequence]:
    """Split one folder's basenames into constant-step sequences. Pure: does no
    I/O and leaves height/width as None (filled later by a header read).

    `expected_step=None` infers the step per run (min positive delta), so a
    clean on-2s render survives as one sequence and a dropped frame in an on-1s
    render splits at the gap. Pass an int to force a specific cadence.
    """
    warn = warnings if warnings is not None else []

    # 1) group by stem signature
    groups: dict[tuple, list[tuple[int, int, str]]] = {}
    n_unparsed = 0
    for name in filenames:
        p = parse_frame_name(name)
        if p is None:
            n_unparsed += 1
            continue
        groups.setdefault(p.group_key, []).append((p.number, p.ndigits, name))
    if n_unparsed:
        warn.append(f"{folder}: {n_unparsed} file(s) without a frame number, ignored")

    fast = fast_token is not None and fast_token in folder
    sequences: list[Sequence] = []

    for (left, tail), entries in groups.items():
        # sort by number, then width, then name for deterministic de-dup
        entries.sort(key=lambda e: (e[0], e[1], e[2]))

        # 2) drop duplicate frame numbers (mixed padding collisions etc.)
        deduped: list[tuple[int, int, str]] = []
        for num, nd, name in entries:
            if deduped and deduped[-1][0] == num:
                warn.append(f"{folder}: duplicate frame number {num} for stem "
                            f"'{left}*{tail}', keeping '{deduped[-1][2]}'")
                continue
            deduped.append((num, nd, name))
        if not deduped:
            continue

        numbers = [e[0] for e in deduped]
        step = expected_step if expected_step is not None else _infer_step(numbers)

        # 3) split into constant-step runs
        for run in _split_runs(deduped, step):
            if len(run) < min_length:
                continue
            run_numbers = [e[0] for e in run]
            run_names = [e[2] for e in run]
            pad = min(e[1] for e in run)          # un-overflowed field width
            rstart = run_numbers[0]
            count = len(run)
            eff_max = 3 if fast else max_window
            eff_max = min(eff_max, count)

            seq = Sequence(
                seq_id=_make_seq_id(folder, left, tail, rstart, step),
                folder=folder,
                left=left,
                tail=tail,
                pad=pad,
                step=step,
                start=rstart,
                count=count,
                max_window=eff_max,
            )

            # 4) verify compact reconstruction; fall back to explicit if it drifts
            if not _reconstruction_matches(seq, run_names):
                seq.explicit = run_names
                warn.append(f"{folder}: irregular naming for stem '{left}*{tail}', "
                            f"storing {count} explicit path(s)")
            sequences.append(seq)

    return sequences


def _infer_step(sorted_numbers: list[int]) -> int:
    """Step = smallest positive gap present (the tightest intended cadence)."""
    if len(sorted_numbers) < 2:
        return 1
    step = min(b - a for a, b in zip(sorted_numbers, sorted_numbers[1:]) if b > a)
    return max(step, 1)


def _split_runs(entries, step):
    """Yield maximal runs whose successive numbers differ by exactly `step`."""
    run = [entries[0]]
    for prev, cur in zip(entries, entries[1:]):
        if cur[0] - prev[0] == step:
            run.append(cur)
        else:
            yield run
            run = [cur]
    yield run


def _reconstruction_matches(seq: Sequence, run_names: list[str]) -> bool:
    for i, name in enumerate(run_names):
        if seq.basename_at(i) != name:
            return False
    return True


# ---------------------------------------------------------------------------
# Manifest: scan + header reads + cache
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    version: int
    root: str
    fingerprint: str
    sequences: list[Sequence] = field(default_factory=list)
    build_warnings: list[str] = field(default_factory=list)

    # convenience -----------------------------------------------------------
    def by_id(self) -> dict[str, Sequence]:
        return {s.seq_id: s for s in self.sequences}

    def folders(self) -> set[str]:
        return {s.folder for s in self.sequences}

    def total_frames(self) -> int:
        return sum(s.count for s in self.sequences)

    def total_windows(self, bidirectional: bool = True, mode: str = "full") -> int:
        return sum(s.num_windows(bidirectional, mode) for s in self.sequences)

    # (de)serialisation -----------------------------------------------------
    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": self.version,
                    "root": self.root,
                    "fingerprint": self.fingerprint,
                    "build_warnings": self.build_warnings,
                    "sequences": [s.to_dict() for s in self.sequences],
                },
                f,
            )
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "Manifest":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Manifest(
            version=d["version"],
            root=d["root"],
            fingerprint=d["fingerprint"],
            sequences=[Sequence.from_dict(s) for s in d["sequences"]],
            build_warnings=d.get("build_warnings", []),
        )


def _iter_exr_folders(root, skip_components, follow_symlinks):
    """Yield (folder, [exr basenames]) using scandir — far cheaper than os.walk
    over millions of files. Skips any dir whose path has a skipped component."""
    skip = set(skip_components)
    stack = [root]
    while stack:
        d = stack.pop()
        exrs = []
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        is_dir = e.is_dir(follow_symlinks=follow_symlinks)
                    except OSError:
                        continue
                    if is_dir:
                        if e.name not in skip:
                            stack.append(e.path)
                    elif e.name.endswith(".exr"):
                        exrs.append(e.name)
        except (PermissionError, FileNotFoundError):
            continue
        if exrs:
            yield d, exrs


def dir_fingerprint(root, skip_components=DEFAULT_SKIP_COMPONENTS, follow_symlinks=True) -> str:
    """Cheap-to-recompute signature over *directory* mtimes only (dirs are far
    fewer than files, and a dir's mtime changes when its entries change). Used
    to auto-invalidate the manifest when the tree is modified."""
    skip = set(skip_components)
    h = hashlib.sha1()
    stack = [root]
    seen = []
    while stack:
        d = stack.pop()
        try:
            st = os.stat(d)
            seen.append((d, int(st.st_mtime)))
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=follow_symlinks) and e.name not in skip:
                            stack.append(e.path)
                    except OSError:
                        continue
        except OSError:
            continue
    for d, mt in sorted(seen):
        h.update(f"{d}:{mt}".encode("utf-8"))
    return h.hexdigest()


def read_native_size(path: str) -> tuple[Optional[int], Optional[int]]:
    """(height, width) from an EXR header. OpenImageIO imported lazily so the
    rest of this module needs neither OIIO nor torch."""
    import OpenImageIO as oiio
    inp = oiio.ImageInput.open(path)
    if inp is None:
        return None, None
    spec = inp.spec()
    inp.close()
    return spec.height, spec.width


def build_manifest(
    root: str,
    *,
    manifest_path: Optional[str] = None,
    force: bool = False,
    expected_step: Optional[int] = None,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_window: int = DEFAULT_MAX_WINDOW,
    fast_token: Optional[str] = FAST_TOKEN,
    skip_components=DEFAULT_SKIP_COMPONENTS,
    follow_symlinks: bool = True,
    header_workers: int = 8,
    read_headers: bool = True,
    verbose: bool = True,
) -> Manifest:
    """Build (or load from cache) the sequence manifest for `root`.

    Cache is reused when the directory-mtime fingerprint matches and `force` is
    False, so a normal launch is a JSON read rather than a walk over millions
    of files. One header read per *sequence* fills native size (parallelised);
    set read_headers=False to skip (e.g. when building on a box without OIIO).
    """
    manifest_path = manifest_path or os.path.join(root, ".timewarp_manifest.json")

    fp = dir_fingerprint(root, skip_components, follow_symlinks)
    if not force and os.path.isfile(manifest_path):
        try:
            cached = Manifest.load(manifest_path)
            if cached.version == MANIFEST_VERSION and cached.fingerprint == fp:
                if verbose:
                    print(f"[manifest] cache hit: {len(cached.sequences)} sequences, "
                          f"{cached.total_frames()} frames")
                return cached
            if verbose:
                print("[manifest] cache stale, rebuilding")
        except Exception as e:
            if verbose:
                print(f"[manifest] cache unreadable ({e}), rebuilding")

    warnings: list[str] = []
    sequences: list[Sequence] = []
    n_folders = 0
    for folder, exrs in _iter_exr_folders(root, skip_components, follow_symlinks):
        n_folders += 1
        if verbose and n_folders % 500 == 0:
            print(f"\r[manifest] scanned {n_folders} folders, "
                  f"{len(sequences)} sequences", end="")
        sequences.extend(detect_sequences_in_folder(
            folder, exrs,
            expected_step=expected_step,
            min_length=min_length,
            max_window=max_window,
            fast_token=fast_token,
            warnings=warnings,
        ))
    if verbose:
        print(f"\r[manifest] scanned {n_folders} folders, {len(sequences)} sequences")

    if read_headers and sequences:
        _fill_native_sizes(sequences, header_workers, verbose)

    manifest = Manifest(
        version=MANIFEST_VERSION,
        root=root,
        fingerprint=fp,
        sequences=sequences,
        build_warnings=warnings,
    )
    try:
        manifest.save(manifest_path)
    except OSError as e:
        if verbose:
            print(f"[manifest] could not write cache: {e}")

    if verbose and warnings:
        print(f"[manifest] {len(warnings)} warning(s); first few:")
        for w in warnings[:5]:
            print(f"    - {w}")
    return manifest


def _fill_native_sizes(sequences, workers, verbose):
    from concurrent.futures import ThreadPoolExecutor
    done = 0

    def work(seq):
        h, w = read_native_size(seq.path_at(0))
        seq.height, seq.width = h, w
        return seq

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for _ in ex.map(work, sequences):
            done += 1
            if verbose and done % 500 == 0:
                print(f"\r[manifest] read {done}/{len(sequences)} headers", end="")
    if verbose:
        print(f"\r[manifest] read {done}/{len(sequences)} headers")