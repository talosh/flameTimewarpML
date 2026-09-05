#!/bin/bash
# Train the timewarp model (legacy streaming-dataset script, kept for
# comparison / fine-tuning old checkpoints). Prefer train_timewarp_ddp.sh.
#
# Run from an activated environment (./activate.sh) or override PYTHON_CMD.

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

PYTHON_CMD="${PYTHON_CMD:-python}"
exec $PYTHON_CMD ./pytorch/flameTimewarpML_train.py "$@"
