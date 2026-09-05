# Training

Two trainers share the same DDP machinery and the `data/` pipeline
(see [DATASET.md](DATASET.md)):

- `flameTimewarp_train_ddp.py` — timewarp (`models/flownet*_vNNN.py`)
- `flameStabML_train_ddp.py` — stab (`models/warpnet*_vNNN.py`)

This document covers the timewarp trainer in detail; the stab trainer's
extra flags are noted at the end.

## DDP design

- `--all_gpus` spawns one process per visible GPU via
  `torch.multiprocessing.spawn`; NCCL over `localhost:12355`.
- Each rank feeds **itself** from its own `BatchPool` (sequences sharded by
  the sampler) — there is no rank-0 prefetch/scatter.
- Losses are all-reduced (average) before logging, so every rank logs
  identical numbers and the LR scheduler — driven by a rolling average of the
  reduced loss — stays in sync across ranks.
- Epoch boundary: rank 0 detects end-of-epoch, **broadcasts** it, all ranks
  reshuffle with a shared seed, then a barrier.
- Only rank 0 writes checkpoints, previews and CSVs (background threads).
- On clean completion (`--epochs N`) the writers are drained and the process
  leaves via `os._exit` — plain `sys.exit()` races the still-alive daemon
  writer threads during interpreter finalization and aborts on this
  environment. `Ctrl+C` uses the same `os._exit` path after an atomic save.
- **GPU discipline:** the script only ever touches the device it was given
  (`--device N` or the spawn rank). `--device cpu` runs fully on CPU and
  guarantees no CUDA context is created (all `torch.cuda.*` calls are gated
  on the actual device, not on `torch.cuda.is_available()`).

## Learning rate

- Base `--lr`, scaled by **sqrt(world_size)** for DDP (conservative rule for
  Adam-family optimizers).
- Scheduler (first match wins):
  - `--onecycle E` — OneCycleLR over E epochs,
  - `--cyclic N` — CyclicLR with period N steps,
  - else **ReduceLROnPlateau** on a rolling `--avg_window`-step loss average,
    stepped every `--plateau_interval` steps (default 1000), with
    `--plateau_patience` / `--plateau_factor`. With `--plateau_interval 0`
    it falls back to epoch/eval-based stepping.
- `--pulse` / `--pulse_amplitude` set the CyclicLR fallback shape.

## Flag reference (flameTimewarp_train_ddp.py)

### Required
| Flag | Meaning |
|---|---|
| `dataset_path` | Dataset root (tree of EXR sequence folders) |

### Model
| Flag | Default | Meaning |
|---|---|---|
| `--model` | newest `flownet*_v*` | Model family to load (`models/` file without `.py`) |
| `--state_file` | — | Checkpoint to resume / init from |
| `--legacy_model` | — | Load a RIFE-style state dict with `strict=False` |
| `--device` | `0` | GPU index, or `cpu` |
| `--all_gpus` | off | DDP over all visible GPUs |
| `--compile` | off | `torch.compile` (not recommended on Pascal) |
| `--lr` | `1e-6` | Learning rate (before DDP sqrt scaling) |
| `--weight_decay` | derived | AdamW weight decay (default from `--generalize`) |
| `--scales` | `8,4,2,1` | Pyramid scale list (4 entries, finest last) |
| `--iterations` | `1` | Refine iterations per pyramid level |
| `--ap0` | off | Input EXRs are AP0 (ACES2065-1) instead of AP1 |

### Data pipeline (see DATASET.md)
| Flag | Default | Meaning |
|---|---|---|
| `--batch` / `--batch_size` | `2` | Batch size per GPU |
| `--frame_size` | `448` | Training frame size (long side, px) |
| `--max_window` | `12` | Max temporal window (frames) |
| `--window_mode` | `full` | `full` = every window size 3..max at every interior gt (timewarp); `fixed` = max-sized windows only (stab) |
| `--window_bidirectional` | `auto` | `auto`: on for `full`, off for `fixed` (fixed pairs both directions anyway) |
| `--sequential` | off | No reshuffle between epochs, sequential pool |
| `--seed` | `1234` | Seed for splits/sampler/pool/augmentation |
| `--val_frac` / `--test_frac` | `0` | Split fractions (per folder) |
| `--val_folder` / `--test_folder` | — | External held-out roots (kept out of train) |
| `--num_workers` | `8` | DataLoader workers |
| `--cache_items` | `256` | Per-worker decoded-frame LRU |
| `--pool_size` | `48` | Reuse-pool size (batches) |
| `--reuse` | `4` | Serve each batch N times before refilling (1 = fresh) |
| `--pool_order` | `random` | `random` / `sequential` |
| `--pad_tolerance` | `0.10` | Max per-batch long-side padding fraction |
| `--rotation_prob` | `0.5` | Probability of ±90° rotation |
| `--max_long_side` | `0` | Drop sequences above this long side (0 = no limit) |
| `--hflip` / `--vflip` / `--cflip` | `0.5/0/0` | Flip probabilities |
| `--pin_memory` | off | Pin DataLoader memory |
| `--steps_per_epoch` | auto | Steps per epoch (default: dataset size / (batch × ranks)) |
| `--input_encodings` | `ap1,rec709,acescct` | Model-input encodings, sampled equally per step |

### Loss
| Flag | Default | Meaning |
|---|---|---|
| `--deep_sup` | `0` | Extra importance-sampled coarse-level term (see below) |
| `--lpips_alternate` | off | Compute per-level LPIPS on alternate levels only (halves its cost) |

### Checkpointing / epochs
| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | `∞` | Stop after N epochs (final eval + best save at the end) |
| `--first_epoch` | checkpoint | Start epoch |
| `--save` | `10000` | Save every N steps (also at every epoch end) |
| `--reset_stats` | off | Reset step/epoch/loss stats from the checkpoint |

### Previews
| Flag | Default | Meaning |
|---|---|---|
| `--preview` | `100` | Save a 5-frame preview triad every N steps |
| `--preview_max` / `--preview_min` | `0` | Track top/bottom-N error samples (0 = off) |
| `--preview_maxmin_steps` | `10000` | Flush max/min preview every N steps |

### Evaluation
| Flag | Default | Meaning |
|---|---|---|
| `--eval` | off (`-1`) | Evaluate every N steps |
| `--eval_first` | off | Don't skip the step-1 trigger |
| `--eval_samples` | all | Evaluate N random samples |
| `--eval_seed` | `1` | Seed for sample selection |
| `--eval_folder` | — | External eval clips (legacy reader). Default: one rng-seeded window per train sequence, same window plan as training |
| `--eval_save_imgs` | off | Write 7 EXRs per eval sample |
| `--eval_keep_all` | off | Keep every eval folder (default: keep latest) |
| `--eval_buffer` | `8` | Eval image write-buffer size |
| `--eval_half` | off | Half-precision eval |
| `--eval_samples`/`--eval_seed` | | see above |

A final evaluation always runs when `--epochs` is reached. The best model
(min of `eval_avg_l1 + 0.2 × eval_lpips`) is written as `.best.pth`.

## Loss recipe (timewarp)

Working space is **ACEScct**; EXR I/O is AP1 (or AP0) linear. Each step:

1. Sample one input encoding from `--input_encodings` (AP1 / Rec.709 /
   ACEScct). The two model inputs get exposure (50% chance ×0.25–1.4) and
   positive noise augmentation; the GT center frame stays clean.
2. The model returns **all four pyramid levels** (training forward), each
   with flow (4ch fwd/bwd), mask and confidence.
3. Per level *i* (scale *sᵢ*), with `output = warp(img0, f₀)·m + warp(img2, f₁)·(1−m)`
   and `compress()` the legacy tanh-bend:

   ```
   Lᵢ = L1(output, gt) @ (1/sᵢ) bicubic
      + LapLoss(output, gt)
      + fourier_loss_half_res(output, gt)
      + 1e-2 · variance_loss(mask, 0.1)
      + 1e-2 · L1(conf, diffmatte)
      + 1.4e-2 · (1/(i+1)) · LPIPS(output, gt)
   ```

4. Finest level additionally: `+ L1 + LapLoss + fourier + 0.1·Ternary + 0.1·Sobel + 1e-2·LPIPS`.
5. Gradients are clipped to norm 1; AdamW.

**Documented deviation from the legacy script:** the legacy
`pytorch/flameTimewarpML_train.py` converted the LPIPS terms to `float(...)`
before adding them — i.e. LPIPS was *logging-only* there. This script keeps
the same weights but lets LPIPS carry gradient (as the stab trainer does).
If you want the exact legacy behaviour, wrap the LPIPS terms in `float(...)`.

### `--deep_sup`

The per-level loop already supervises every level. `--deep_sup W` adds one
**extra** coarse level per step, importance-sampled with weights
`∝ 1/scale` (coarser levels are weighted more). The term is
`W · tot_w · (L1 + LapLoss)` on the downsampled frames/flow — unbiased in
expectation (`E[tot_w·Lᵢ] = Σ wᵢLᵢ`) and cheap. Default `0` (off). The
flownet training forward returns all levels for this; the inference forward
(`get_model()`) materializes only the finest level.

## Color pipeline

- EXR frames are read as **linear** (tonemap off).
- `--ap0` → AP0-to-AP1 matrix, then AP1→ACEScct; else AP1→ACEScct.
- Losses and previews live in ACEScct; previews/eval EXRs are written back
  through ACEScct→ACEScg (AP1) for viewing in grading tools.
- The model *inputs* are sampled per step from `--input_encodings`:
  - `ap1` — AP1 linear (ACEScg working space),
  - `rec709` — primaries + Rec.709 OETF,
  - `acescct` — ACEScct (the historical behaviour).

## Checkpoints

State dict keys: `step`, `epoch`, `start_timestamp`, `lr`, `model_info`,
`flownet_state_dict`, `optimizer_flownet_state_dict`,
`trained_model_path`. Writes are atomic (`.periodic.tmp` → rename, previous
checkpoint backed up to `.backup.pth`); `Ctrl+C` writes through
`.saving.tmp` and exits via `os._exit` (never leaves a truncated file).
Training/inference model variants have identical state dicts, so one
checkpoint serves both.

## Stab trainer extras (flameStabML_train_ddp.py)

`--smooth_mode` (blend/…), `--freeze`, `--resize`, `--acc` and the
sharpness/FiLM conditioning are stab-specific; the timewarp model has no
sharpness input, so those flags do not exist in the timewarp trainer.

## Testing without GPUs

```bash
CUDA_VISIBLE_DEVICES="" python ./flameTimewarp_train_ddp.py <small_ds> \
    --model flownet4_v001 --device cpu --batch 1 --frame_size 96 \
    --max_window 5 --num_workers 0 --pool_size 2 --reuse 1 \
    --steps_per_epoch 3 --preview 2 --save 2 --eval 2 --eval_samples 2 \
    --eval_save_imgs --epochs 1
```

`CUDA_VISIBLE_DEVICES=""` plus the `use_cuda` guard guarantees the process
cannot create a CUDA context (important on this box, where GPUs 0/1 serve
the local LLM).
