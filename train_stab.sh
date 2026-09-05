#!/bin/bash
# Train the stabilization model (warpnet).
#
# Usage:  ./train_stab.sh --device 2 --batch 2 --frame_size 224 \
#            --max_window 24 --window_mode fixed --lr 2e-5 \
#            --state_file weights/Warpnet4_v001.pth --model warpnet4_v001 \
#            /path/to/dataset_root/
#
# Run from an activated environment (./activate.sh) or override PYTHON_CMD.
# Multi-GPU: pass --all_gpus to spawn one process per visible GPU (DDP).

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

PYTHON_CMD="${PYTHON_CMD:-python}"
exec $PYTHON_CMD ./flameStabML_train_ddp.py "$@"
