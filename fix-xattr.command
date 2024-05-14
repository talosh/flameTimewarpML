#!/bin/bash

# Define the target directory relative to the script location
TARGET_DIR="$(dirname "$0")/packages/.miniconda"

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Directory $TARGET_DIR does not exist."
  exit 1
fi

# Find all files in the target directory and clear extended attributes
find "$TARGET_DIR" -type f -exec xattr -c {} \;

echo "Extended attributes cleared from all files in $TARGET_DIR"