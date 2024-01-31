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

    if not os.path.isdir(args.dst_path):
        os.makedirs(args.dst_path)

    for folder_index, folder_path in enumerate(sorted(folders_with_png)):
        # print (f'\rScanning folder {folder_index + 1} of {len(folders_with_png)}', end='')
        png_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.exr')]
        png_files.sort()
        for png_file_path in png_files:
            image_bgr = cv2.imread(png_file_path, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            print (f'\nimage_rgb shape: {image_rgb.shape}')

    print ('')


if __name__ == "__main__":
    main()
