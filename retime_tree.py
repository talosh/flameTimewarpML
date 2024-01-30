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

def read_frames(all_frame_descriptions, frames_queue):
    timeout = 1e-8
    while True:
        for index in range(len(all_frame_descriptions)):
            description = all_frame_descriptions[index]
            try:
                description['incoming_data'] = fw.read_openexr_file(description['incoming'])['image_data']
                description['outgoing_data'] = fw.read_openexr_file(description['outgoing'])['image_data']
                frames_queue.put(description)
            except Exception as e:
                print (e)                
        time.sleep(timeout)

def normalize(image_array) :
    def custom_bend(x):
        linear_part = x
        exp_bend = torch.sign(x) * torch.pow(torch.abs(x), 1 / 4 )
        return torch.where(x > 1, exp_bend, torch.where(x < -1, exp_bend, linear_part))

    # transfer (0.0 - 1.0) onto (-1.0 - 1.0) for tanh
    image_array = (image_array * 2) - 1
    # bend values below -1.0 and above 1.0 exponentially so they are not larger then (-4.0 - 4.0)
    image_array = custom_bend(image_array)
    # bend everything to fit -1.0 - 1.0 with hyperbolic tanhent
    image_array = torch.tanh(image_array)
    # move it to 0.0 - 1.0 range
    image_array = (image_array + 1) / 2

    return image_array

def restore_normalized_values(image_array):
    def custom_de_bend(x):
        linear_part = x
        exp_deband = torch.sign(x) * torch.pow(torch.abs(x), 4 )
        return torch.where(x > 1, exp_deband, torch.where(x < -1, exp_deband, linear_part))

    epsilon = torch.tensor(4e-8, dtype=torch.float32).to(image_array.device)
    # clamp image befor arctanh
    image_array = torch.clamp((image_array * 2) - 1, -1.0 + epsilon, 1.0 - epsilon)
    # restore values from tanh  s-curve
    image_array = torch.arctanh(image_array)
    # restore custom bended values
    image_array = custom_de_bend(image_array)
    # move it to 0.0 - 1.0 range
    image_array = ( image_array + 1.0) / 2.0

    return image_array

def main():
    parser = argparse.ArgumentParser(description='Retime script.')
    # Required argument
    parser.add_argument('src_path', type=str, help='Path to the source tree')
    parser.add_argument('dst_path', type=str, help='Path to the destination folder')
    parser.add_argument('--speed', type=float, required=True, help='Speed factor for retime in percents')
    # Optional arguments
    default_model_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'models_data',
        'flownet_v412.pkl'
    )
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to the pre-trained model (optional)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')

    args = parser.parse_args()

    folders_with_exr = find_folders_with_exr(args.src_path)
    common_path = os.path.commonpath(folders_with_exr)
    
    all_frame_descriptions = []

    for folder_index, folder_path in enumerate(folders_with_exr):
        print (f'\rScanning folder {folder_index + 1} of {len(folders_with_exr)}', end='')
        folder_frames_map = compose_frames_map_speed(folder_path, common_path, args.dst_path, args.speed)
        for key in sorted(folder_frames_map.keys()):
            all_frame_descriptions.append(folder_frames_map[key])
    print ('')

    print ('starting frame read therad...')
    frames_queue = queue.Queue(maxsize=4)
    frame_read_thread = threading.Thread(target=read_frames, args=(all_frame_descriptions, frames_queue))
    frame_read_thread.daemon = True
    frame_read_thread.start()

    print ('loading model...')
    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')
    model = FlownetCas().to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    def convert(param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }
    model.load_state_dict(convert(state_dict))
    model.half()
    model.eval()

    for frame_idx in range(len(all_frame_descriptions)):
        print (f'\rProcessing frame {frame_idx + 1} of {len(all_frame_descriptions)}', end='')

        frame_data = frames_queue.get()
        img0 = torch.from_numpy(frame_data['incoming_data'].copy())
        img0 = img0.to(device = device, dtype = torch.float16, non_blocking = True)
        img0 = img0.permute(2, 0, 1).unsqueeze(0)
        img0 = normalize(img0)

        img1 = torch.from_numpy(frame_data['outgoing_data'].copy())
        img1 = img1.to(device = device, dtype = torch.float16, non_blocking = True)
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
        img1 = normalize(img1)

        flow_list, mask, merged, teacher_res, loss_cons = model(torch.cat((img0, img1), dim=1), timestep = frame_data['ratio'])    

        del img0, img1, frame_data


if __name__ == "__main__":
    main()
