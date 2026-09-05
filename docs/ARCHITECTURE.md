# Architecture

## Repository map

| Path | What it is | Status |
|---|---|---|
| `flameTimewarp_train_ddp.py` | Timewarp trainer (flownet family) | **current** |
| `flameStabML_train_ddp.py` | Stab trainer (warpnet family) | **current** |
| `train_*.sh`, `train.sh`, `validate.sh` | Launcher wrappers | current (train.sh = legacy trainer) |
| `data/` | Shared dataset pipeline + tests | **current** |
| `models/` | Model zoo: `flownet4_v001.py`, `warpnet4_v001.py` | **current** |
| `models/archived/` | Retired flownet variants | reference |
| `pytorch/` | Legacy trainers/validators (old streaming dataset) | legacy, kept for comparison |
| `baselight/` | Node package `uk.co.andriy.mltimewarp.v1` | shipped node |
| `flameTimewarpML.py`, `flameTimewarpML_framework.py`, `bundle.py`, `pyflame_lib_flameTimewarpML.py` | Baselight node source + packaging | shipped node |
| `pybox/` | Node effects (fluidmorph …) | shipped node |
| `tools/` | One-off utilities: EXR/DPX writers, tonemap, perf tests, cut detection | utilities |
| `packages/` | Vendored conda runtime for the node bundle (hidden from Baselight scanning) | bundle |
| `hub/checkpoints/` | LPIPS alex backbone (offline training) | data |
| `weights/`, `old_trained_models/` | Local checkpoints (gitignored) | data |
| `docs/` | This documentation + handoff notes | docs |

## Model zoo conventions

Every file in `models/` defines a `Model` class:

```python
class Model:
    info = {'name': 'Flownet4_v001', 'file': 'flownet4_v001.py',
            'ratio_support': True}

    def get_info(self) / get_name(self)
    def get_model(self)           # inference: finest pyramid level only
    def get_training_model(self)  # training: returns ALL pyramid levels
    def input_channels(state_dict) / output_channels(state_dict)
```

Trainers import models dynamically (`find_and_import_model`):

- `--model flownet4_v001` → exact module,
- `--state_file` alone → `model_info['file']` from the checkpoint,
- otherwise the newest `flownet*_vNNN` (timewarp) / `warpnet*_vNNN` (stab)
  file — lexical ordering matches zero-padded version suffixes.

A new model version is therefore just a new file. `ratio_support` gates
window sizing (non-ratio models are capped to 3-frame windows).

### flownet4_v001 (timewarp)

- 4 pyramid levels (scales default `8,4,2,1`): a 16-ch encoder
  (`Head`) + four `Flownet` blocks (192/128/96/64 channels) that predict
  per-level flow updates; the final head outputs 6 channels
  = flow (4, fwd/bwd) + mask + confidence (both sigmoided).
- The temporal **ratio is per-sample** (a `(N,)` tensor broadcast to
  `(N,1,1,1)` in the blocks) — one forward pass handles a batch of
  different ratios.
- `get_training_model()` (default `return_levels=True`) materializes
  flow/mask/conf/merged at **all four levels** so the loss can supervise
  every scale (deep supervision + the `--deep_sup` sampled term).
  `get_model()` / `FlownetCasInfer` (`return_levels=False`) materializes
  only the finest level — no intermediate clones for inference.
  Both share an identical state dict, so one checkpoint serves both.

### warpnet4_v001 (stab)

2-frame stabilization net with sharpness/FiLM conditioning and
cycle-consistency training; see the stab trainer.

## Trainer structure (both `*_train_ddp.py`)

```
header            env (PCI_BUS_ID, allocator) + imports + data/ pipeline
colour modules    AP0/AP1 → ACEScct working space, ACEScct → ACEScg,
                  AP1 → Rec.709 (model-input encoding augmentation)
IO                write_exr / read_image_file (OIIO + raw EXR writer)
metrics/losses    warp, psnr, diffmatte, variance_loss, LapLoss,
                  fourier, Ternary, Sobel, compress, downscale_flow,
                  compute_lpips (LPIPS, per-rank)
legacy dataset    TimewarpMLDataset (only for --eval_folder)
model loading     find_and_import_model (dynamic models/ import)
TrainFeed         data/ wrapper (dataset+sampler+pool) per rank
build_train_feed  manifest → splits → per-rank TrainFeed
main(rank, world) argparse → DDP init → model → feed → threads →
                  training loop (loss, DDP reduce, scheduler,
                  checkpoint/preview/CSV, eval, epoch broadcast)
__main__          spawn(--all_gpus) or single process
```

The two trainers deliberately share code by **duplication** (each file is
self-contained, the repo's established pattern). A shared-module refactor
is a noted follow-up, not done yet, to keep each script independently
routable to the node bundle.

### DDP invariants

- Per-rank feeding (no rank-0 scatter) → no cross-rank data coupling.
- All-reduced loss before any logging/scheduler use → identical LR across
  ranks.
- Epoch end detected on rank 0, **broadcast**, then barrier + shared-seed
  reshuffle → sample→rank assignment stays consistent across epochs.
- Rank-0-only writers (checkpoints/preview/CSV) on daemon threads with
  atomic renames; clean completion and `Ctrl+C` drain + `os._exit`
  (finalization races with the writers on this environment).

### Checkpoints

See [TRAINING.md](TRAINING.md#checkpoints). `.pth` = latest,
`.backup.pth` = previous, `.best.pth` = best eval score. All three share
the same state-dict layout and load with the same `--state_file`.

## Relationship to the node

`flameTimewarpML.py` / `flameTimewarpML_framework.py` implement the
Baselight node; `bundle.py` packs the node + `packages/` runtime into the
`flameTimewarpML` folder that users install into Baselight's python hooks
path. The node loads the same model files (`models/*.py`) and checkpoints
(`*.pth`) this repo produces — inference uses `get_model()`.

## Legacy (`pytorch/`)

`flameTimewarpML_train.py` (timewarp, old streaming dataset),
`flameStabML_train.py` (stab, pre-DDP), `flameTimewarpML_validate.py`,
`flameTimewarpML_finetune.py`, scale-finders and one-off frame tools.
Kept for reproducing old runs and fine-tuning legacy checkpoints
(`train.sh`). The new trainers supersede them.

`pytorch/models/` holds symlinks to the current root-level model files —
the legacy scripts resolve `find_and_import_model(models_dir='models')`
relative to their own directory. When adding a new model version that the
legacy scripts should see, add the symlink there too.
