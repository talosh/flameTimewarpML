# flameTimewarpML

Machine-learning timewarp and stabilization for the **Flame** node in
**Baselight** (Baselight is a finishing color-correction system; `flame` is
its ML plugin family).

Two model families are trained and shipped from this repo:

| Family | Model | Task |
|---|---|---|
| **timewarp** | `flownet4_v001` | Retiming / temporal warp: given an *incoming* and an *outgoing* frame plus a temporal *ratio* (how far apart they are), synthesize the *target* frame at an arbitrary intermediate time. |
| **stab** | `warpnet4_v001` | Stabilization: remove shake from a 3-frame clip while preserving the operator's intentional move. |

Both are 4-level flow pyramids. Training is PyTorch with DDP support; the
dataset pipeline lives in [`data/`](data/) and is shared by both trainers.

## Repository layout

```
flameTimewarpML/
├── flameTimewarp_train_ddp.py     # timewarp trainer (new dataset pipeline, DDP)
├── flameStabML_train_ddp.py       # stab trainer (new dataset pipeline, DDP)
├── train_timewarp_ddp.sh          # convenience wrappers for the trainers
├── train_stab.sh
├── train.sh                       # legacy timewarp trainer wrapper
├── data/                          # dataset pipeline (manifest, windows, sampler, pool)
├── models/                        # model zoo (flownetN_vNNN.py, warpnetN_vNNN.py)
│   └── archived/                  # retired model versions
├── pytorch/                       # legacy training/validation scripts (old dataset)
├── baselight/                     # Baselight node package (uk.co.andriy.mltimewarp.v1)
├── bundle.py, flameTimewarpML.py  # node source + packaging
├── pybox/                         # node effects (fluidmorph etc.)
├── tools/                         # one-off utilities (EXR/DPX writers, tonemap, perf)
├── docs/                          # TRAINING, DATASET, ARCHITECTURE, handoff notes
├── weights/                       # checkpoints (gitignored, local only)
├── hub/checkpoints/               # LPIPS backbone (auto-downloaded on first run)
└── packages/                      # vendored conda runtime for the node bundle
```

## Getting started

### Environment

Training runs in a conda env (`appenv`). Activate an interactive shell with:

```bash
./activate.sh          # shell with the right env + bashrc
```

or, for non-interactive use, call the interpreter directly:

```bash
/home/flame/miniconda3/envs/appenv/bin/python
```

(If the env's bundled `libstdc++` is newer than the system one, prefix with
`LD_LIBRARY_PATH=/home/flame/miniconda3/envs/appenv/lib`.)

Dependencies: `torch`, `OpenImageIO`, `lpips` (see `requirements.txt`). The
LPIPS alex backbone (~244 MB) is not in git — on a fresh clone it is
downloaded into `hub/checkpoints/` on the first run (needs internet once).

Building an env from scratch is documented in
[docs/legacy/README_v044.md](docs/legacy/README_v044.md#installing-and-configuring-python-environment-manually).

### Training (timewarp)

```bash
./train_timewarp_ddp.sh \
    --state_file weights/Flownet4_v001.pth --model flownet4_v001 \
    --device 2 --batch 2 --frame_size 448 --max_window 12 --lr 1e-6 \
    /path/to/dataset_root/
```

Dataset root is a tree of folders containing OpenEXR frame sequences
(AP1-linear by default; `--ap0` for AP0). See [docs/DATASET.md](docs/DATASET.md).

Multi-GPU: add `--all_gpus` (DDP over all visible GPUs, NCCL).

### Training (stab)

```bash
./train_stab.sh \
    --state_file weights/Warpnet4_v001.pth --model warpnet4_v001 \
    --device 2 --batch 2 --frame_size 224 --max_window 24 \
    --window_mode fixed --lr 2e-5 \
    /path/to/dataset_root/
```

### Outputs

Per run (paths relative to `$HOME` and to the dataset root):

- `~/flameTWML_models/flameTWML_model_<stamp>.pth` — periodic checkpoint
  (atomic writes, `.backup.pth` of the previous one)
- `..._model_<stamp>.best.pth` — best evaluation score
- `<dataset>/preview/<model>/<idx>_{A_incoming,B_outgoing,C_target,D_output,E_diff}.exr`
- `<dataset>/preview/eval/<model>/Step_<n>/...` — eval results
- `<model>.csv`, `<model>.eval.csv` — per-epoch and per-eval metrics

Full flag reference, loss composition and color-space notes:
[docs/TRAINING.md](docs/TRAINING.md).

## Hardware notes

The training GPUs on this machine are Pascal (P5000/P40, sm_61):

- **No bfloat16** — train in **fp32** (mixed precision buys nothing here).
- `torch.compile` is unreliable on sm_61; not used by default.
- GPUs 0 and 1 host the local LLM service — training scripts take an explicit
  `--device N`; never run with `--all_gpus` unless those GPUs are free.

## Documentation

- [docs/TRAINING.md](docs/TRAINING.md) — trainer flags, loss recipe, DDP design, checkpoints
- [docs/DATASET.md](docs/DATASET.md) — the `data/` pipeline, dataset layout, tests
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — repo map, model zoo conventions
- [docs/legacy/README_v044.md](docs/legacy/README_v044.md) — original v0.4.4 readme
