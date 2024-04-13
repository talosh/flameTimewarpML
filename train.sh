#!/bin/bash

# Resolves the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Change to the script directory
cd "$SCRIPT_DIR"

# Define the path to the Python executable and the Python script
PYTHON_CMD="./packages/.miniconda/appenv/bin/python"
PYTHON_SCRIPT="./pytorch/train.py"

# Run the Python script with all arguments passed to this shell script
$PYTHON_CMD $PYTHON_SCRIPT "$@"
