# Dataset pipeline (`data/`)

The `data/` package turns a directory tree of OpenEXR frames into
bucketed, DDP-sharded training batches without ever materializing the full
sample list. Both trainers build their training feed from it.

## Dataset layout

A **dataset root** is any tree of folders containing EXR frames. The atomic
unit is not a folder but a *sequence*: a contiguous, constant-step run of
frames sharing the same basename stem.

```
dataset_root/
├── Scene001/
│   ├── shot_a/
│   │   ├── shot_a.0001.exr      # one sequence (stem "shot_a.", step 1)
│   │   ├── shot_a.0002.exr
│   │   └── ...
│   └── shot_b_v002/
│       ├── shot_b_v002.0001.exr # "v002" is a version token, not the frame
│       └── ...                  # field: the LAST run of digits wins
└── fast/
    └── shot_c/...               # 'fast' in the path -> 3-frame windows only
```

Rules (see `data/descriptions.py`):

- **Frame field = last run of digits** in the basename (VFX convention), so
  version tokens like `v002` are safe. Overflow keeps a run whole (`0999`
  → `1000`).
- A folder may hold several sequences (different stems, or one shot
  fractured by dropped frames); a window can never straddle a sequence
  boundary — a structural guarantee, not a filter.
- Sequences shorter than `min_length` (3) are ignored.
- Folders whose path contains the token `fast` are restricted to
  3-frame windows (fast motion → ratios near the endpoints).
- `preview/` and `eval/` folders are skipped by default
  (`DEFAULT_SKIP_COMPONENTS`) — that's where trainers write outputs.
- Only uncompressed OpenEXR is supported. Frames are read as **linear**
  (the trainers pass `tonemap=False`); color handling happens in the
  training loop.

## Manifests and windows

`build_manifest(root, max_window=…, read_headers=True)` scans the tree and
reads one EXR header per sequence (for H/W), producing a `Manifest` of
`Sequence` records:

```
Sequence(seq_id, folder, left, tail, pad, step, start, count,
         height, width, max_window, explicit=None)
```

Frame paths are reconstructed on demand (`path_at(i)`) from
`left + zero-padded number + tail`, so a million-frame sequence costs a
handful of fields. Manifests can be saved/loaded (`Manifest.save/load`) and
are fingerprinted per directory.

A **window** is one training sample: two indices around a gt frame plus the
temporal ratio.

```
WindowSpec(seq_id, start, gt, end, ratio)
# ratio = (gt - start) / (end - start) in [0, 1]
```

Windows are addressed by integer index with closed-form counts
(`num_windows`, `window_at`) — the planner counts, shuffles and samples
windows with plain integers.

Two window modes:

- **`full`** (timewarp default): every window size 3..max_window at every
  interior gt. Teaches the model arbitrary ratios densely.
- **`fixed`** (stab): only max-sized windows.

`bidirectional=True` (auto for `full`) also emits reversed windows; for
`fixed` it's off by default because the training loop already pairs both
directions.

## Splits (`splits.py`)

`split_sequences(train_sequences, val_sequences=…, test_sequences=…,
fractions=(tr, va, te), seed=…)` splits **per folder** (a folder never
straddles splits), with external `--val_folder` / `--test_folder` roots kept
wholly out of training. Overlap between the train pool and external
roots is detected and reported (`find_overlap`, `sequence_signature`).
The result is a `Split` with `.train` / `.val` / `.test` sequence lists.

## Sampler (`sampler.py`)

`TimewarpBatchSampler` plans an epoch of **bucketed steps**:

1. **Rotation plan:** each sequence gets an epoch rotation
   (0 / +90 / −90) with probability `--rotation_prob` (split evenly across
   ±90). Rotation is decided at plan time from native size, so bucket
   geometry is known without looking at frames.
2. **Buckets:** sequences are grouped by orientation + long-side band so a
   batch needs only single-axis padding bounded by `--pad_tolerance`
   (default 10%). `--max_long_side` drops oversized sequences.
3. **Sharding:** buckets are assigned to ranks (`shard_sequences`); each
   rank only ever plans its own share.
4. A step = up to `--batch` `SampleSpec`s (seq + window + rotation) from
   one bucket. `steps_per_epoch` is explicit or estimated from the window
   count.

`SampleSpec(seq_id, start, gt, end, ratio, rotation)` is what the dataset
resolves into pixels.

## Dataset + collate

`TimewarpDataset.__getitem__(spec)`:

- reads the three frames through the reader
  (`default_reader(path, out_h, out_w, channels=3, tonemap=False)` —
  EXR → CHW float32 tensor, resized),
- applies the planned rotation, then h/v/c-flips per their probabilities
  (seeded per sample so all three frames share the transform).

`collate_timewarp` pads the batch to its max (H, W) with zeros (single-axis
by construction) and, when requested, emits a validity `mask`
(1 = real pixel, 0 = padding). The trainers use `return_mask=False` — the
model reads padding as content.

`build_dataloader` wires a per-worker `FrameCache` (LRU,
`--cache_items` decoded frames) via `worker_init_fn`; with
`--num_workers 0` the cache attaches to the dataset directly. Workers
ignore SIGINT so a `Ctrl+C` can't make them fight the main process over a
checkpoint file.

## Batch pool (`pool.py`)

`BatchPool` decouples the slow DataLoader from the training step: a
background thread refills a small buffer (`--pool_size`) and each batch is
served `--reuse` times (in `--pool_order` order) before being dropped.
`reuse=1 + sequential` is a pure passthrough (no thread). The pool is
re-seeded per epoch, so `--seed` + epoch reproduce the exact batch order.

`TrainFeed` (in the trainers) wraps dataset + sampler + pool and exposes
`__len__` (steps/epoch), `next_batch()`, `set_epoch()`, `reshuffle()` and
`repeat_count=1`.

## Tests

Self-contained checks (no fixtures on disk; synthetic readers where pixels
are needed):

```bash
PYTHONPATH=. python -m data.test_window_modes   # closed-form window math
PYTHONPATH=. python -m data.test_splits_external
PYTHONPATH=. python -m data.test_sampler
PYTHONPATH=. python -m data.test_cache
PYTHONPATH=. python -m data.test_pool
PYTHONPATH=. python -m data.test_dataset        # rotation/flip/collate paths
```

(They are `test_*.py` files *inside* `data/`, un-ignored via
`!data/test_*.py` in `.gitignore`.)

## Legacy reader

`--eval_folder` in the trainers still uses the old streaming
`TimewarpMLDataset` (kept in each trainer for compatibility with existing
eval clips and the `--acescc`/`--generalize` knobs). Training itself no
longer touches it.
