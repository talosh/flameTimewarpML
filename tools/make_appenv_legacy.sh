#!/usr/bin/env bash
# Create a backwards-compatible pybox appenv: the oldest-driver torch build
# for a given CUDA generation.
#
#   ./make_appenv_legacy.sh <cu118|cu126|cu128|cu130|cpu> [appenv_path]
#
# torch 2.7.1 covers cu118/cu126/cu128 (last release before cu118 was dropped);
# CUDA 13 needs torch 2.9 since cu130 postdates 2.7. Python 3.10 works for all.
# Verified against the PyTorch matrix Sept 2026 — re-check before trusting.
set -euo pipefail

TARGET="${1:-}"
APPENV="${2:-packages/miniconda/appenv}"
PY=3.10

# cu102 (torch 1.12) is a different shape: it predates the isolated wheel index
# (needs --extra-index-url) and requires numpy<2. Handled as a special case.
case "$TARGET" in
    cu102) TORCH="torch==1.12.1+cu102"; IDX="cu102"; EXTRA=1; NP="numpy<2" ;;
    cu118) TORCH="torch==2.7.1"; IDX="cu118"; EXTRA=0; NP="numpy" ;;
    cu126) TORCH="torch==2.7.1"; IDX="cu126"; EXTRA=0; NP="numpy" ;;
    cu128) TORCH="torch==2.7.1"; IDX="cu128"; EXTRA=0; NP="numpy" ;;
    cu130) TORCH="torch";        IDX="cu130"; EXTRA=0; NP="numpy" ;;   # latest, 2.9+
    cpu)   TORCH="torch==2.7.1"; IDX="cpu";   EXTRA=0; NP="numpy" ;;
    "")    echo "usage: $0 <cu102|cu118|cu126|cu128|cu130|cpu> [appenv_path]" >&2; exit 1 ;;
    *)     echo "unknown target: $TARGET (see APPENV_LEGACY.md / APPENV_CUDA10.md)" >&2; exit 1 ;;
esac

echo ">> creating $APPENV for $TARGET (python $PY, $TORCH from $IDX)"
conda create -y -p "$APPENV" -c conda-forge "python=$PY" "$NP" py-openimageio
if [ "$EXTRA" = "1" ]; then
    # torch 1.12: fall back to PyPI for deps, and re-pin numpy<2 in case
    # the wheel pulled a newer one.
    "$APPENV/bin/python" -m pip install "$NP" "$TORCH" \
        --extra-index-url "https://download.pytorch.org/whl/$IDX"
else
    "$APPENV/bin/python" -m pip install "$TORCH" \
        --index-url "https://download.pytorch.org/whl/$IDX"
fi

echo ">> verifying"
"$APPENV/bin/python" - <<'EOF'
import sys, numpy, torch, OpenImageIO as oiio
print('python     ', sys.version.split()[0])
print('numpy      ', numpy.__version__)
print('torch      ', torch.__version__, '| cuda build:', torch.version.cuda,
      '| available:', torch.cuda.is_available())
print('OpenImageIO', getattr(oiio, '__version__', oiio.VERSION_STRING))
s = oiio.ImageSpec(64, 32, 3, oiio.TypeDesc(oiio.FLOAT))
assert hasattr(s, 'erase_attribute')
s.nchannels = 2; assert s.nchannels == 2
assert oiio.ImageSpec(s) is not s
print('OIIO API checks passed')
EOF
