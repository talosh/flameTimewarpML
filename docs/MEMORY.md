# Project memory (flameTimewarpML)

Durable facts for the next session. Keep this file small; session detail
goes to `PROGRESS.md`.

## Environment

- Training env: conda `appenv` at `/home/flame/miniconda3/envs/appenv`
  (Python 3.11.15). `./activate.sh` opens an interactive shell in it.
- Non-interactive: call the interpreter by absolute path; if it fails with
  `GLIBCXX_3.4.29 not found`, prefix
  `LD_LIBRARY_PATH=/home/flame/miniconda3/envs/appenv/lib`.
- LPIPS alex backbone (~244 MB) is **not** in git (too big for GitHub);
  a local copy is kept on disk at `hub/checkpoints/`. Trainers set
  `TORCH_HOME=<repo root>` before `import lpips` — on a fresh clone the
  backbone auto-downloads there on first run (internet needed once).
- Pascal GPUs (sm_61): **fp32 only**, no `torch.compile`, no bfloat16.

## Machine / GPU discipline

- GPU 0/1 = llama-server (the LLM serving this agent) — never touch.
  GPU 2 = free test card (a training job usually sits on it).
- CPU smoke tests of the trainers: `CUDA_VISIBLE_DEVICES="" ... --device cpu`.
  The trainers gate all `torch.cuda.*` calls on the *actual device*, never
  on `torch.cuda.is_available()`, so a CPU run cannot create a CUDA context.
- Verified 2026-09-05: CPU end-to-end smoke of `flameTimewarp_train_ddp.py`
  (train + eval + previews + checkpoints, `--epochs 1`) exits 0 and leaves
  complete `.pth` / `.best.pth` / CSVs.

## Gotchas

- Interpreter finalization (`sys.exit`) on this environment aborts with
  `terminate called without an active exception` when the daemon writer
  threads are alive; both trainers therefore drain writers and
  `os._exit(0)` on clean completion (and on SIGINT).
- `step % N == 1` checkpoint/preview triggers: with `--save 1` /
  `--preview 1` the per-step check never fires (epoch-end save still runs).
- Data-package tests run as modules:
  `PYTHONPATH=. python -m data.test_<name>` (6 suites, ~5.7k checks).
- `.gitignore` keeps `*.pth`, `*.csv`, `*.exr`, `test_*` out of git, with
  exceptions for `data/test_*.py` (the LPIPS backbone stays local-only).

## Live training (as of 2026-09-05)

- Stab training running on cuda:2:
  `python ./flameStabML_train_ddp.py --state_file weights/Warpnet4_v001.pth
  --model warpnet4_v001 --batch 2 --device 2 --max_window 24
  --window_mode fixed --lr 2e-5 --frame_size 224 --smooth_mode blend
  --deep_sup 0.2 /mnt/StorageMedia/dataset_image/pexels_exr/`
- Do not stop it; if GPU 2 is needed, prefer CPU checks first.
