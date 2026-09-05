#!/bin/bash

# Exit on error
set -e

# Resolve the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Change to the script directory
cd "$SCRIPT_DIR"

# Use the Python from the currently activated conda environment
PYTHON_CMD="python"
# PYTHON_SCRIPT="./private/lllm/scripts/run.py"
# PYTHON_SCRIPT="./private/test.py"
CUSTOM_RC="$SCRIPT_DIR/timewarp.bashrc"

# Run under group 'inet' with local .bashrc and correct working directory
sg inet -c "cd \"$SCRIPT_DIR\" && bash --rcfile \"$CUSTOM_RC\" -i"