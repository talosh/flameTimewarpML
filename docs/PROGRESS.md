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
