# Project: flameTimewarpML — ML timewarp / stabilization (Baselight Flame node)

ML timewarp/stabilization for the Baselight Flame node. This project is its
own registered qwen daemon workspace (2026-09-07): sessions for it run with
cwd = /mnt/StorageMedia/dev/flameTimewarpML. Agentctl-style milestone
supervision also works from the primary: webctl supervise
--cwd /mnt/StorageMedia/dev/flameTimewarpML.

- Machine rules: the user-global ~/.qwen/QWEN.md — loaded into every session
  of every workspace. Do NOT duplicate machine rules here.
- Project memory: docs/MEMORY.md (read at session start; append durable
  facts at the end of a milestone; keep it small and consolidated).
- Progress / handoff log: docs/PROGRESS.md (one entry per session, newest
  last; each entry ends with its Verify commands).
- Docs: docs/ARCHITECTURE.md, docs/TRAINING.md, docs/DATASET.md; legacy
  docs under docs/legacy/.
- Environment: conda env `appenv` at /home/flame/miniconda3/envs/appenv
  (Python 3.11). Call the interpreter by absolute path; the GLIBCXX note
  lives in docs/MEMORY.md. Record any env change where the next session
  will find it.
- GPU discipline: GPU 0/1 = llama-server — never touch. GPU 2 = the free
  test card; a training job may be sitting on it (see the "Live training"
  note in docs/MEMORY.md) — check with nvidia-smi before using it, and do
  NOT stop a live training job. Long / multi-GPU / >16GB training goes
  through the GPU handoff (machine memory [gpu-handoff-threshold]).
- Write model: the project tree is the default write area; writes outside
  it only to paths named in the task's SCOPE; reading is unrestricted.
- pybox/ and baselight/ are shipped node code — place/organize around
  them, but don't refactor them casually.
- Gates are truth: every milestone ends with a verification command; paste
  its real output. Never report a check as passing without showing it.
