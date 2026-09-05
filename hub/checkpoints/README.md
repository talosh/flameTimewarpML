# LPIPS backbone

The trainers set `TORCH_HOME` to the repo root before `import lpips`, so the
LPIPS alex backbone (`alexnet-owt-7be5be79.pth`, ~244 MB) is fetched into
`hub/checkpoints/` on first use and reused afterwards.

The weights file is intentionally **not** in git (too large for a normal
repo). On a fresh clone it is downloaded automatically from
`download.pytorch.org` the first time a trainer runs (needs internet once);
this machine keeps a local copy on disk for offline runs.
