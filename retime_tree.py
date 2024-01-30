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

from pprint import pprint

import flameSimpleML_framework
importlib.reload(flameSimpleML_framework)
from flameSimpleML_framework import flameAppFramework

settings = {
    'menu_group_name': 'Simple ML',
    'debug': False,
    'app_name': 'flameSimpleML',
    'prefs_folder': os.getenv('FLAMESMPLMsL_PREFS'),
    'bundle_folder': os.getenv('FLAMESMPLML_BUNDLE'),
    'packages_folder': os.getenv('FLAMESMPLML_PACKAGES'),
    'temp_folder': os.getenv('FLAMESMPLML_TEMP'),
    'requirements': [
        'numpy>=1.16',
        'torch>=1.12.0'
    ],
    'version': 'v0.0.3',
}

fw = flameAppFramework(settings = settings)
importlib.invalidate_caches()
try:
    import numpy as np
    import torch
except:
    if fw.site_packages_folder in sys.path:

        print ('unable to import numpy and pytorch')
        sys.exit()
    else:
        sys.path.append(fw.site_packages_folder)
        try:
            import numpy as np
            import torch
        except:
            print ('unable to import numpy and pytorch')
            sys.exit()
        if fw.site_packages_folder in sys.path:
            sys.path.remove(fw.site_packages_folder)

import math

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from models.flownet import FlownetCas
from models.multires_v001 import Model as Model_01

def find_folders_with_exr(path):
    """
    Find all folders under the given path that contain .exr files.

    Parameters:
    path (str): The root directory to start the search from.

    Returns:
    list: A list of directories containing .exr files.
    """
    directories_with_exr = set()

    # Walk through all directories and files in the given path
    for root, dirs, files in os.walk(path):
        if root.endswith('preview'):
            continue
        for file in files:
            if file.endswith('.exr'):
                directories_with_exr.add(root)
                break  # No need to check other files in the same directory

    return directories_with_exr

def compose_frames_map_speed(folder_path, common_path, dst_path, speed):
    exr_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.exr')]
    exr_files.sort()

    duration = len(exr_files)
    relative_start_frame = 0
    max_frame_value = relative_start_frame + duration - 1

    speed_multiplier = speed / 100
    new_duration = int(duration / speed_multiplier)

    frames_map = {}
    frame_value = relative_start_frame
    for frame in range(relative_start_frame, new_duration + 1):
        frames_map[frame] = {
            'ratio': frame_value - int(frame_value),
            'incoming': exr_files[int(frame_value) if int(frame_value) < max_frame_value else max_frame_value],
            'outgoing': exr_files[int(frame_value) + 1 if int(frame_value) + 1 < max_frame_value else max_frame_value],
            'destination': os.path.join(dst_path, os.path.relpath(folder_path, common_path), f'{frame:08d}.exr')
        }
        frame_value = frame_value + speed_multiplier
    return frames_map


def main():
    parser = argparse.ArgumentParser(description='Retime script.')
    # Required argument
    parser.add_argument('src_path', type=str, help='Path to the source tree')
    parser.add_argument('dst_path', type=str, help='Path to the destination folder')
    parser.add_argument('--speed', type=float, required=True, help='Speed factor for retime in percents')
    # Optional arguments
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model (optional)')

    args = parser.parse_args()

    folders_with_exr = find_folders_with_exr(args.src_path)
    common_path = os.path.commonpath(folders_with_exr)
    for folder_path in folders_with_exr:
        folder_frames_map = compose_frames_map_speed(folder_path, common_path, args.dst_path, args.speed)
        pprint (folder_frames_map)


if __name__ == "__main__":
    main()
