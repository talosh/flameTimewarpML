#!/bin/bash
# Train the timewarp model (flownet) with the new dataset pipeline.
#
# Usage:  ./train_timewarp_ddp.sh --device 2 --batch 2 --frame_size 448 \
#            --max_window 12 --lr 1e-6 /path/to/dataset_root/
#
# Run from an activated environment (./activate.sh) or override PYTHON_CMD.
# Multi-GPU: pass --all_gpus to spawn one process per visible GPU (DDP).

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

PYTHON_CMD="${PYTHON_CMD:-python}"
exec $PYTHON_CMD ./flameTimewarp_train_ddp.py "$@"
