import os
import sys
import random
import struct
import ctypes
import argparse
import importlib
import queue
import threading
import time
import platform

import torch
import torch.nn as nn

import cv2

from pprint import pprint

def find_folders_with_png(path):
    """
    Find all folders under the given path that contain .exr files.

    Parameters:
    path (str): The root directory to start the search from.

    Returns:
    list: A list of directories containing .exr files.
    """
    directories_with_png = set()

    # Walk through all directories and files in the given path
    for root, dirs, files in os.walk(path):
        if root.endswith('preview'):
            continue
        print (root)
        for file in files:
            if file.endswith('.png'):
                directories_with_png.add(root)
                break  # No need to check other files in the same directory

    return directories_with_png    

def main():
    parser = argparse.ArgumentParser(description='Convert vimeo 90k pngs to uncompreesed exrs.')
    # Required argument
    parser.add_argument('src_path', type=str, help='Path to the source tree')
    parser.add_argument('dst_path', type=str, help='Path to the destination folder')
    args = parser.parse_args()

    folders_with_png = find_folders_with_png(args.src_path)

    pprint (folders_with_png)


if __name__ == "__main__":
    main()
