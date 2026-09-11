# Progress / handoff log

One entry per session, newest last. Format:

```
## YYYY-MM-DD — <short title>
- Done: ...
- Next: ...
- Verify: <commands that must pass>
```

## 2026-09-05 — New timewarp trainer + repo reorg

- Done:
  - `models/flownet4_v001.py`: training forward returns all 4 pyramid
    levels (deep supervision); `FlownetCasInfer` returns finest level only;
    identical state dicts; per-sample ratio guard (`(N,)` → `(N,1,1,1)`).
  - `flameTimewarp_train_ddp.py`: new timewarp trainer mirroring the DDP
    stab trainer — new `data/` pipeline, timewarp loss recipe (per-level
    L1+Lap+fourier+mask+conf+LPIPS, full-res +Ternary+Sobel+LPIPS),
    `--deep_sup` sampled coarse term, `--window_mode full|fixed`,
    per-step input-encoding sampling (ap1/rec709/acescct), eval path that
    works without `--eval_folder` (one rng-seeded window per train
    sequence, converted to ACEScct), best-model `.best.pth` save.
    LPIPS kept as gradient (legacy detached it — documented in
    docs/TRAINING.md). Clean completion drains writers + `os._exit`
    (sys.exit aborts on this env). `--device cpu` fully supported.
  - Repo reorg: tools → `tools/`, README_v044 → `docs/legacy/`, junk
    removed, `models/__init__.py`, train scripts fixed
    (`train_stab.sh` path, new `train_timewarp_ddp.sh`), `.gitignore`,
    README rewritten, docs/ created (TRAINING, DATASET, ARCHITECTURE,
    MEMORY, PROGRESS), `pytorch/models/` = symlinks to root models so the
    legacy `find_and_import_model(models_dir='models')` still resolves
    (verified: `import models.flownet4_v001` from `pytorch/` works).
  - LPIPS backbone (244 MB) kept on disk but **out of git** — GitHub
    rejects single files >100 MB on push; fresh clones auto-download it
    on first run (trainers set `TORCH_HOME` to the repo root).
  - Committed in three logical commits on `main`: layout reorg
    (`834d4bbb`), trainer + models + data + hub (`d4a23855`), and
    README/docs/launchers (`6bc9eba4`). Working tree clean afterwards.
  - Pushed to origin (be2c37a0..6bc9eba4) under group inet (outgoing
    net is blocked except for that group; see machine memory
    [inet-net-access]).
- Next:
  - Optional follow-up: extract the duplicated trainer sections into a
    shared module (see docs/ARCHITECTURE.md).
  - Optional: real-GPU validation of `--deep_sup` convergence vs legacy.
- Verify:
  - `env LD_LIBRARY_PATH=/home/flame/miniconda3/envs/appenv/lib CUDA_VISIBLE_DEVICES="" /home/flame/miniconda3/envs/appenv/bin/python -m py_compile flameTimewarp_train_ddp.py flameStabML_train_ddp.py`
  - CPU smoke: see "Live training" note in docs/MEMORY.md for the pattern;
    exit code must be 0 with `.pth`/`.best.pth`/CSVs present.
  - `PYTHONPATH=. /home/flame/miniconda3/envs/appenv/bin/python -m data.test_dataset` (all 6 suites, 0 failed)

## 2026-09-07 — Registered as a qwen daemon workspace

- Done:
  - This project is now its own registered qwen daemon workspace
    (`POST /workspaces {persist:true}`, id `d76e48690e843420`). Sessions
    for it run with cwd = this repo root; agentctl supervision works via
    `webctl supervise --cwd /mnt/StorageMedia/dev/flameTimewarpML`.
  - Background: the 2026-09-06 workspace reorg (primary dev → agentctl)
    silently lost ALL sibling registrations — the store file is keyed by
    the primary's sha256 and is only created by the first persisted
    `POST /workspaces`, not at daemon start (see agentctl machine memory
    [workspace-ftml-registered]).
  - Added `QWEN.md` (project pointers; machine rules stay in the
    user-global file) and `.qwen/settings.json` (allow: appenv python).
- Next:
  - ts_test will be merged into this project (user decision 2026-09-07);
    its workspace registration was intentionally NOT re-created.
- Verify:
  - `curl -s -H "Authorization: Bearer $QWEN_SERVER_TOKEN"
    http://127.0.0.1:4170/workspace-registrations` (token lives in
    /etc/qwen-code/qwen-serve.env, root 0600 — read via a sudo-run script)
    → entries contains cwd `/mnt/StorageMedia/dev/flameTimewarpML`,
    persisted=true.
  - Store file `~/.qwen/daemon/workspaces/3c7b3de56be98e06e9f6e3494f7793f9d2e5f79777ad3ec302098280ccb5cdf2.json`
    lists the workspace (survives daemon restart).

## 2026-09-09 — warpnet4_v002: AE latent FiLM integration (refwarp)

- Done:
  - `models/warpnet4_v002.py`: v001 backbone unchanged + embedded ts_test
    `LatentEncoder` (frozen, 110 tensors, `latent_version 1`) + zero-init
    `film_z` Linear(256→2c) per ResConv (Style B, applied after the
    existing sharpness film). AE input contract (canonical in the model
    docstring): clean AP1-linear pair [B,3,H,W] + encoding name →
    `nan_to_num(nan=0, posinf=1.1)` → user `soft_limit(0, 1.1, knee
    0.002)` → short-side-64 aspect-preserving resize IN LINEAR space →
    vendored encode (bit-identical to ts_test) → frozen encoder at fixed
    ratio 0.5 → z_global. `ae_in=None` → z=0 → exact v001 function.
  - `tools/merge_warpnet_v002.py` (5 gates, ALL PASSED): base 261 keys in,
    exactly 174 new (64 film_z + 110 ae_encoder); AE baked strict;
    PASSTHROUGH v001(base) vs v002(merged) max|Δ| == 0.0 over all 4 flow
    levels; embedded AE bit-identical to standalone ts_test encoder;
    strict round-trip. Output `weights/Warpnet4_v002.pth` (58 MB, 14.47 M
    total / 8.42 M trainable, NO optimizer state — trainer falls back to
    fresh AdamW). Merge base snapshot: `weights/Warpnet4_v001_mergebase.pth`
    (live run writes atomically, read-safe; user can flag if they want the
    Aug-31 `.copy.pth` base instead).
  - `flameStabML_train_ddp.py`: `has_ae` detection after model build;
    forward passes clean (un-augmented) linear pair + encoding as
    `ae_in`/`encoding` (no-op kwargs for v001-style models); REPLACED the
    stale `--freeze` section (it referenced nonexistent `blockN.film` →
    AttributeError if ever used) with name-pattern freeze: ae_encoder +
    encode always frozen, blocks keep scaler + film paths, `film` only
    unless new `--freeze_film` (phase 1: latent path alone); `--device cpu`
    now supported (str arg; per-step synchronize guarded by device type).
  - `tools/check_v002_ae_seam.py` — colour-seam audit on real EXRs (4
    clips × 3 frames × 4 encodings; production two-step resize + model
    sanitise vs the ts_test one-step training path): no NaN/inf, all
    pexels frames ≤ 1.0 linear (soft-limit upper knee never engages);
    colour seam (A' vs B) 73.6 % bit-identical, max Δ 0.001 (ap0/ap1) to
    0.015 (cct) — entirely the (0, 0.002) shadow roll-off + pure black
    (0 → ~0.001, open-interval rail); resize staging max Δ 0.12–0.36 on
    size-matched frames; 3 portrait frames land on a 2 px grid offset
    (sampler snap-to-16 changes aspect at the 64 side, 123 vs 121) with
    max Δ 1.37; z_global L2 mean 0.20 (ap0/ap1) / 0.59 (rec709/cct), max
    ≤ 2.05. Verdict: film_z trains on production z from step 0 — the
    shift is moderate by design, no action needed.
  - CPU smokes (both exited cleanly via SIGINT: "Interrupted - saving
    current state ... Saved. Exiting.", no orphans): (1) full v002 train —
    52 epochs / 207 steps, finite losses (PSNR 41–50, LPIPS 0.004–0.006),
    `.pth` + `.backup.pth` + CSV + preview EXRs written, post-run
    checkpoint strict round-trip OK; (2) `--freeze --freeze_film` —
    trainable set EXACTLY 72 params (32× film_z w+b + 4× scaler w+b), 47
    epochs finite losses.
  - Smoke artifacts (`Warpnet4_v002_smoke.*`) deleted from `weights/`
    afterwards; dataset scratch left in `.qwen/tmp/v002_smoke/`.
- Next:
  - Start v002 training on GPU 2 once the live v001 run ends (or user
    approves a slot): phase 1 `--model warpnet4_v002
    --state_file weights/Warpnet4_v002.pth --freeze --freeze_film
    --batch 2 --device 2 --max_window 24 --window_mode fixed --lr 2e-5
    --frame_size 224 --smooth_mode blend --deep_sup 0.2
    /mnt/StorageMedia/dataset_image/pexels_exr/`, then phase 2 without
    `--freeze_film`.
  - Baselight node wiring (feed clean linear pair + encoding name into
    the model forward) — documented follow-up, out of scope here.
  - The no-`--eval_folder` eval path is dormant and incompatible with
    warpnet (pre-existing; never fires, defaults -1/-1) — fix if eval
    becomes needed.
- Notes:
  - `frame_size 64` CPU smokes CRASH in the dilated conv stack at pyramid
    scale 16 (64/16 = 4 px, pre-existing v001 behaviour, not a v002
    regression) — smoke at `--frame_size 224` like production.
  - Current parser defines `--batch_size`; the live v001 process was
    launched with `--batch` (the file changed after that process started).
- Verify:
  - `PYTHONPATH=. env LD_LIBRARY_PATH=/home/flame/miniconda3/envs/appenv/lib CUDA_VISIBLE_DEVICES="" /home/flame/miniconda3/envs/appenv/bin/python tools/merge_warpnet_v002.py --base weights/Warpnet4_v001_mergebase.pth --ae /mnt/StorageMedia/dev/ts_test/runs/m6_full/ckpt_final.pt --out weights/Warpnet4_v002.pth` → `GATE-1..GATE-5 OK`, `GATE-3 PASSTHROUGH OK: max|Δ| over 4 flow levels = 0.0`
  - `PYTHONPATH=. env LD_LIBRARY_PATH=/home/flame/miniconda3/envs/appenv/lib CUDA_VISIBLE_DEVICES="" /home/flame/miniconda3/envs/appenv/bin/python tools/check_v002_ae_seam.py --clips 4 --frames-per-clip 3 --frame-size 224` → `[seam] DONE`
  - CPU smoke pattern (background, SIGINT the python pid after a couple of
    epoch CSV rows): `env LD_LIBRARY_PATH=/home/flame/miniconda3/envs/appenv/lib CUDA_VISIBLE_DEVICES="" /home/flame/miniconda3/envs/appenv/bin/python ./flameStabML_train_ddp.py --model warpnet4_v002 --state_file <copy-of-weights/Warpnet4_v002.pth> --device cpu --batch_size 2 --frame_size 224 --max_window 4 --window_mode fixed --lr 2e-5 --epochs -1 --steps_per_epoch 5 --smooth_mode blend --deep_sup 0.2 <small-exr-dir>` → log ends `Saved. Exiting.`, `.pth`/CSV present, `kill -0` clean.
