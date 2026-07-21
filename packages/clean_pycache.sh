#!/bin/bash

# Define the target directory. Default is the current directory.
TARGET_DIR="${1:-.}"

# Find all __pycache__ directories and remove them
find "$TARGET_DIR" -type d -name "__pycache__" -exec rm -r {} +

echo "All __pycache__ directories have been removed from $TARGET_DIR"