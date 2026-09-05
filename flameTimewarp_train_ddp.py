import os

# Enumerate GPUs in PCI bus order so --device / cuda:N match `nvidia-smi` indices.
# CUDA's default is FASTEST_FIRST, which reorders by capability -- on a mixed box
# (e.g. two P5000 + a P40) cuda:0 then is NOT nvidia-smi's GPU 0. Must be set
# before torch initialises CUDA, hence before the torch import below.
# setdefault: an explicit CUDA_DEVICE_ORDER in the environment still wins.
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
# Expandable segments keep the allocator from fragmenting on long runs.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
import random
import shutil
import struct
import ctypes
import argparse
import importlib
import queue
import threading
import time
import platform
import heapq
import json
from copy import deepcopy
import traceback

import tracemalloc

from pprint import pprint

import numpy as np
import OpenImageIO as oiio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import functools

from data import (
    build_manifest,
    split_sequences,
    TimewarpDataset,
    TimewarpBatchSampler,
    build_dataloader,
    BatchPool,
    default_reader,
)

exit_event = threading.Event()  # For threads
process_exit_event = torch.multiprocessing.Event()  # For processes
write_shutdown = threading.Event()  # asked the background write threads to stop (clean-completion drain)

def write_exr(image_data, filename, half_float = False, pixelAspectRatio = 1.0):
    import struct
    import numpy as np

    if image_data.dtype == np.float16:
        half_float = True

    height, width, depth = image_data.shape
    red = image_data[:, :, 0]
    green = image_data[:, :, 1]
    blue = image_data[:, :, 2]
    if depth > 3:
        alpha = image_data[:, :, 3]
    else:
        alpha = np.array([])

    channels_list = ['B', 'G', 'R'] if not alpha.size else ['A', 'B', 'G', 'R']

    MAGIC = 20000630
    VERSION = 2
    UINT = 0
    HALF = 1
    FLOAT = 2

    def write_attr(f, name, type, value):
        f.write(name.encode('utf-8') + b'\x00')
        f.write(type.encode('utf-8') + b'\x00')
        f.write(struct.pack('<I', len(value)))
        f.write(value)

    def get_channels_attr(channels_list):
        channel_list = b''
        for channel_name in channels_list:
            name_padded = channel_name[:254] + '\x00'
            bit_depth = 1 if half_float else 2
            pLinear = 0
            reserved = (0, 0, 0)  # replace with your values if needed
            xSampling = 1  # replace with your value
            ySampling = 1  # replace with your value
            channel_list += struct.pack(
                f"<{len(name_padded)}s i B 3B 2i",
                name_padded.encode(), 
                bit_depth, 
                pLinear, 
                *reserved, 
                xSampling, 
                ySampling
                )
        channel_list += struct.pack('c', b'\x00')

            # channel_list += (f'{i}\x00').encode('utf-8')
            # channel_list += struct.pack("<i4B", HALF, 1, 1, 0, 0)
        return channel_list
    
    def get_box2i_attr(x_min, y_min, x_max, y_max):
        return struct.pack('<iiii', x_min, y_min, x_max, y_max)

    with open(filename, 'wb') as f:
        # Magic number and version field
        f.write(struct.pack('I', 20000630))  # Magic number
        f.write(struct.pack('H', 2))  # Version field
        f.write(struct.pack('H', 0))  # Version field
        write_attr(f, 'channels', 'chlist', get_channels_attr(channels_list))
        write_attr(f, 'compression', 'compression', b'\x00')  # no compression
        write_attr(f, 'dataWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
        write_attr(f, 'displayWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
        write_attr(f, 'lineOrder', 'lineOrder', b'\x00')  # increasing Y
        write_attr(f, 'pixelAspectRatio', 'float', struct.pack('<f', pixelAspectRatio))
        write_attr(f, 'screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
        write_attr(f, 'screenWindowWidth', 'float', struct.pack('<f', 1.0))
        f.write(b'\x00')  # end of header

        # Scan line offset table size and position
        line_offset_pos = f.tell()
        pixel_data_start = line_offset_pos + 8 * height
        bytes_per_channel = 2 if half_float else 4
        # each scan line starts with 4 bytes for y coord and 4 bytes for pixel data size
        bytes_per_scan_line = width * len(channels_list) * bytes_per_channel + 8 

        for y in range(height):
            f.write(struct.pack('<Q', pixel_data_start + y * bytes_per_scan_line))

        channel_data = {'R': red, 'G': green, 'B': blue, 'A': alpha}

        # Pixel data
        for y in range(height):
            f.write(struct.pack('I', y))  # Line number
            f.write(struct.pack('I', bytes_per_channel * len(channels_list) * width))  # Pixel data size
            for channel in sorted(channels_list):
                f.write(channel_data[channel][y].tobytes())
        f.close

    del image_data, red, green, blue

def read_image_file(file_path, header_only = False):
    result = {'spec': None, 'image_data': None}
    inp = oiio.ImageInput.open(file_path)
    if inp:
        spec = inp.spec()
        result['spec'] = spec
        if not header_only:
            channels = spec.nchannels
            result['image_data'] = inp.read_image(0, 0, 0, channels)
            # img_data = inp.read_image(0, 0, 0, channels) #.transpose(1, 0, 2)
            # result['image_data'] = np.ascontiguousarray(img_data)
        inp.close()
        # del inp
    return result

def ap0_to_ap1(x):
    M = torch.tensor([
        [1.45143932, -0.23651075, -0.21492857],
        [-0.07655377, 1.17622970, -0.09967593],
        [0.00831615, -0.00603245, 0.99771630]
    ]).to(x.device, x.dtype)
    return torch.einsum('ij,bjhw->bjhw', M, x)

class AP0toACESCCT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("const_cond1", torch.tensor(0.0078125))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = ap0_to_ap1(image)
        condition = image <= self.const_cond1
        value_if_true = image * 10.5402377416545 + 0.0729055341958155 
        ACEScct = torch.where(condition, value_if_true, image)
        
        condition = image > self.const_cond1
        value_if_true = (torch.log2(image) + 9.72) / 17.52
        ACEScct = torch.where(condition, value_if_true, ACEScct)

        return ACEScct

class AP1toACESCCT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("const_cond1", torch.tensor(0.0078125))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        condition = image <= self.const_cond1
        value_if_true = image * 10.5402377416545 + 0.0729055341958155 
        ACEScct = torch.where(condition, value_if_true, image)
        
        condition = image > self.const_cond1
        value_if_true = (torch.log2(image) + 9.72) / 17.52
        ACEScct = torch.where(condition, value_if_true, ACEScct)

        return ACEScct

class ACESCCTtoACESCG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("const_cond1", torch.tensor(0.155251141552511))
        self.register_buffer("const_cond2", (torch.log2(torch.tensor(65504.0)) + 9.72) / 17.52)
        self.register_buffer("const_cond3", torch.tensor(65504.0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        condition = image < self.const_cond1
        value_if_true = (image - 0.0729055341958155) / 10.5402377416545
        ACEScg = torch.where(condition, value_if_true, image)

        condition = (image >= self.const_cond1) & (image < self.const_cond2)
        value_if_true = torch.exp2(image * 17.52 - 9.72)
        ACEScg = torch.where(condition, value_if_true, ACEScg)

        ACEScg = torch.clamp(ACEScg, max=self.const_cond3)

        return ACEScg

def ap1_to_rec709(x):
    # ACEScg (AP1) linear -> linear Rec.709 (sRGB) primaries, D60->D65 adapted
    M = torch.tensor([
        [ 1.70505, -0.62179, -0.08326],
        [-0.13026,  1.14080, -0.01055],
        [-0.02400, -0.12897,  1.15297]
    ]).to(x.device, x.dtype)
    return torch.einsum('ij,bjhw->bihw', M, x)

class AP1toRec709(torch.nn.Module):
    """AP1 (ACEScg) linear -> Rec.709 primaries + Rec.709 OETF. One of the input
    encodings sampled for augmentation, alongside AP1-linear and ACEScct."""
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rgb = ap1_to_rec709(image)
        rgb = torch.clamp(rgb, min=0.0)               # clip out-of-gamut negatives
        base = torch.clamp(rgb, min=1e-8)             # safe base for fractional power
        return torch.where(rgb < 0.018, 4.5 * rgb, 1.099 * torch.pow(base, 0.45) - 0.099)

class TimewarpMLDataset(torch.utils.data.Dataset):
    def __init__(   
            self, 
            data_root, 
            batch_size = 4, 
            device = None, 
            frame_size=448, 
            max_window=9,
            acescc_rate = 0,
            generalize = 80,
            repeat = 1,
            sequential = False,
            start_reader=True
            ):
        
        self.data_root = data_root
        self.batch_size = batch_size
        self.max_window = max_window
        self.acescc_rate = acescc_rate
        self.generalize = generalize
        self.sequential = sequential

        print (f'scanning for exr files in {self.data_root}...')
        self.folders_with_exr = self.find_folders_with_exr(data_root)
        print (f'found {len(self.folders_with_exr)} clip folders.')
        
        self.train_descriptions = []

        for folder_index, folder_path in enumerate(sorted(self.folders_with_exr)):
            print (f'\rReading headers and building training data from clip {folder_index + 1} of {len(self.folders_with_exr)}', end='')
            self.train_descriptions.extend(self.create_dataset_descriptions(folder_path, max_window=self.max_window))

        self.initial_train_descriptions = list(self.train_descriptions)

        if not self.sequential:
            print ('\nReshuffling training data indices...')
            self.reshuffle()
        else:
            print (f'\nSequential: {self.sequential}')

        self.h = frame_size
        self.w = frame_size

        if start_reader:
            self.frames_queue = queue.Queue(maxsize=4)
            self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
            self.frame_read_thread.daemon = True
            self.frame_read_thread.start()

            print('reading first block of training data...')
            self.last_train_data_size = 24
            self.last_train_data = [self.frames_queue.get()] * self.last_train_data_size
            self.train_data_index = 0

            def new_sample_fetch(frames_queue, new_sample_queue):
                while not exit_event.is_set():
                    try:
                        new_sample = frames_queue.get_nowait()
                        new_sample_queue.put(new_sample)
                    except queue.Empty:
                        time.sleep(1e-8)

            self.new_sample_queue = queue.Queue(maxsize=1)
            self.new_sample_thread = threading.Thread(
                target=new_sample_fetch, args=(self.frames_queue, self.new_sample_queue))
            self.new_sample_thread.daemon = True
            self.new_sample_thread.start()
        else:
            # stub — no reader, no cache; data will be scattered from rank 0
            self.frames_queue = None
            self.last_train_data = []
            self.last_train_data_size = 0
            self.train_data_index = 0
            self.new_sample_queue = None

        self.repeat_count = repeat
        self.repeat_counter = 1

        # self.last_shuffled_index = -1
        # self.last_source_image_data = None
        # self.last_target_image_data = None

        if device is None:
            self.device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda')
        else:
            self.device = device

        print (f'ACEScc rate: {self.acescc_rate}%')

    def reshuffle(self):
        random.shuffle(self.train_descriptions)

    def find_folders_with_exr(self, path):
        """
        Find all folders under the given path that contain .exr files.

        Parameters:
        path (str): The root directory to start the search from.

        Returns:
        list: A list of directories containing .exr files.
        """
        directories_with_exr = set()

        # Walk through all directories and files in the given path
        for root, dirs, files in os.walk(path, followlinks=True):
            if 'preview' in root:
                continue
            if 'eval' in root:
                continue
            for file in files:
                if file.endswith('.exr'):
                    directories_with_exr.add(root)
                    break  # No need to check other files in the same directory

        return directories_with_exr

    def create_dataset_descriptions(self, folder_path, max_window=9):

        def sliding_window(lst, n):
            for i in range(len(lst) - n + 1):
                yield lst[i:i + n]

        exr_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.exr')]
        exr_files.sort()

        descriptions = []

        if len(exr_files) < max_window:
            max_window = len(exr_files)
        if max_window < 3:
            print(f'\nWarning: minimum clip length is 3 frames, {folder_path} has {len(exr_files)} frame(s) only')
            return descriptions

        if 'fast' in folder_path:
            max_window = 3
        else:
            max_window = max_window

        try:
            first_exr_file_header = read_image_file(exr_files[0], header_only = True)
            h = first_exr_file_header['spec'].height
            w = first_exr_file_header['spec'].width

            for window_size in range(3, max_window + 1):
                for window in sliding_window(exr_files, window_size):
                    start_frame       = window[0]
                    start_frame_index = exr_files.index(window[0])
                    end_frame         = window[-1]
                    end_frame_index   = exr_files.index(window[-1])

                    gt_frame_index = (len(window) - 1) // 2 - 1  # index into window[1:-1]
                    gt_frame       = window[gt_frame_index + 1]   # index into window

                    fw_item = {
                        'h': h, 'w': w,
                        'start': start_frame,
                        'gt':    gt_frame,
                        'end':   end_frame,
                        'ratio': 1 / (len(window) - 1) * (gt_frame_index + 1)
                    }
                    descriptions.append(fw_item)

                    if not self.sequential:
                        bw_item = {
                            'h': h, 'w': w,
                            'start': end_frame,
                            'gt':    gt_frame,
                            'end':   start_frame,
                            'ratio': 1 - (1 / (len(window) - 1) * (gt_frame_index + 1))
                        }
                        descriptions.append(bw_item)

            '''
            for window_size in range(3, max_window + 1):
                for window in sliding_window(exr_files, window_size):
                    start_frame = window[0]
                    start_frame_index = exr_files.index(window[0])
                    end_frame = window[-1]
                    end_frame_index = exr_files.index(window[-1])
                    for gt_frame_index, gt_frame in enumerate(window[1:-1]):
                        fw_item = {
                            'h': h,
                            'w': w,
                            'start': start_frame,
                            'gt': gt_frame,
                            'end': end_frame,
                            'ratio': 1 / (len(window) - 1) * (gt_frame_index + 1)
                        }
                        descriptions.append(fw_item)

                        if not self.sequential:
                            bw_item = {
                                'h': h,
                                'w': w,
                                # 'pre_start': exr_files[min(end_frame_index + 1, len(exr_files) - 1)],
                                'start': end_frame,
                                'gt': gt_frame,
                                'end': start_frame,
                                # 'after_end': exr_files[max(start_frame_index - 1, 0)],
                                'ratio': 1 - (1 / (len(window) - 1) * (gt_frame_index + 1))
                            }
                            descriptions.append(bw_item)
                '''

        except Exception as e:
            print (f'\nError scanning {folder_path}: {e}')

        return descriptions

    def read_frames_thread(self):
        while not exit_event.is_set():
            # Thread instead of Process — avoids CUDA context reinit under spawn
            t = threading.Thread(
                target=self.read_frames,
                args=(
                    self.frames_queue,
                    list(self.train_descriptions),
                    self.generalize,
                    self.h,
                    self.w
                ),
                daemon=True
            )
            t.start()
            t.join()
            if not self.sequential:
                self.reshuffle()

    @staticmethod
    def read_frames(frames_queue, train_descriptions, generalize, self_h, self_w):
        from PIL import Image
        while not exit_event.is_set():
            for index in range(len(train_descriptions)):
                description = train_descriptions[index]
                train_data = {}
                train_data['description'] = description

                try:
                    img0 = read_image_file(description['start'])['image_data']
                    img1 = read_image_file(description['gt'])['image_data']
                    img2 = read_image_file(description['end'])['image_data']

                    img0 = np.arcsinh(img0 * 2) / 2
                    img1 = np.arcsinh(img1 * 2) / 2
                    img2 = np.arcsinh(img2 * 2) / 2

                    img0 = np.maximum(img0, 1e-6)
                    img1 = np.maximum(img1, 1e-6)
                    img2 = np.maximum(img2, 1e-6)

                    # get rid of negative values before scale
                    img0[img0 < 0] = 0.
                    img1[img1 < 0] = 0.
                    img2[img2 < 0] = 0.

                    '''
                    img0 = torch.from_numpy(img0['image_data']).to(dtype = torch.float32)
                    img1 = torch.from_numpy(img1['image_data']).to(dtype = torch.float32)
                    img2 = torch.from_numpy(img2['image_data']).to(dtype = torch.float32)

                    img0 = img0.permute(2, 0, 1)
                    img1 = img1.permute(2, 0, 1)
                    img2 = img2.permute(2, 0, 1)
                    '''

                    if generalize == 0:
                        h_scaled = self_h
                    else:
                        q = random.uniform(0, 1)
                        if q < 0.25:
                            h_scaled = self_h
                        elif q < 0.5:
                            h_scaled = int(self_h * (1 + 1/4))
                        elif q < 0.75:
                            h_scaled = int(self_h * (1 + 1/3))
                        else:
                            h_scaled = int(self_h * (1 + 1/2))

                    h, w = img0.shape[0], img0.shape[1]
                    if h > w:
                        new_w = h_scaled
                        new_h = int(h_scaled * h / w)
                    else:
                        new_h = h_scaled
                        new_w = int(h_scaled * w / h)

                    channels = [Image.fromarray(img0[:, :, i], mode='F') for i in range(3)]
                    resized_channels = [channel.resize((new_w, new_h), resample=Image.LANCZOS) for channel in channels]
                    resized_arrays = [np.array(channel) for channel in resized_channels]
                    img0 = np.stack(resized_arrays, axis=-1)

                    channels = [Image.fromarray(img1[:, :, i], mode='F') for i in range(3)]
                    resized_channels = [channel.resize((new_w, new_h), resample=Image.LANCZOS) for channel in channels]
                    resized_arrays = [np.array(channel) for channel in resized_channels]
                    img1 = np.stack(resized_arrays, axis=-1)

                    channels = [Image.fromarray(img2[:, :, i], mode='F') for i in range(3)]
                    resized_channels = [channel.resize((new_w, new_h), resample=Image.LANCZOS) for channel in channels]
                    resized_arrays = [np.array(channel) for channel in resized_channels]
                    img2 = np.stack(resized_arrays, axis=-1)

                    # resize_transform = torchvision.transforms.Resize((new_w, new_h), interpolation=torchvision.transforms.InterpolationMode.LANCZOS)

                    '''
                    img0 = torchvision.transforms.functional.resize(img0, (new_h, new_w))
                    img1 = torchvision.transforms.functional.resize(img1, (new_h, new_w))
                    img2 = torchvision.transforms.functional.resize(img2, (new_h, new_w))

                    img0 = img0.squeeze(0).permute(1, 2, 0)
                    img1 = img1.squeeze(0).permute(1, 2, 0)
                    img2 = img2.squeeze(0).permute(1, 2, 0)
                    '''

                    train_data['start'] = img0
                    train_data['gt'] = img1
                    train_data['end'] = img2
                    train_data['ratio'] = description['ratio']
                    train_data['h'] = description['h']
                    train_data['w'] = description['w']
                    train_data['description'] = description
                    train_data['index'] = index
                    frames_queue.put(train_data)

                    # del img0, img1, img2, train_data
                
                except Exception as e:
                    del train_data
                    print (f'\n\nError reading file: {e}')
                    print (f'{description}\n\n')

            # time.sleep(timeout)

    def __len__(self):
        return len(self.train_descriptions)
    
    def crop(self, img0, img1, img2, h, w):
        np.random.seed(None)
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        img2 = img2[x:x+h, y:y+w, :]
        # img3 = img3[x:x+h, y:y+w, :]
        # img4 = img4[x:x+h, y:y+w, :]
        return img0, img1, img2 #, img3, img4

    def resize_image(self, tensor, x):
        """
        Resize the tensor of shape [h, w, c] so that the smallest dimension becomes x,
        while retaining aspect ratio.

        Parameters:
        tensor (torch.Tensor): The input tensor with shape [h, w, c].
        x (int): The target size for the smallest dimension.

        Returns:
        torch.Tensor: The resized tensor.
        """
        # Adjust tensor shape to [n, c, h, w]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        # Calculate new size
        h, w = tensor.shape[2], tensor.shape[3]
        if h > w:
            new_w = x
            new_h = int(x * h / w)
        else:
            new_h = x
            new_w = int(x * w / h)

        # Resize
        resized_tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Adjust tensor shape back to [h, w, c]
        resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)

        del tensor

        return resized_tensor

    def getimg(self, index):        
        if self.repeat_count == 0:
            new_data = self.frames_queue.get()
            self.train_data_index = new_data['index']
            return new_data
    
        if self.repeat_counter >= self.repeat_count:
            self.repeat_counter = 1
            try:
                new_data = self.new_sample_queue.get_nowait()
                self.last_train_data[random.randint(0, len(self.last_train_data) - 1)] = new_data
                self.train_data_index = new_data['index']
                return new_data
            except queue.Empty:
                return random.choice(self.last_train_data)
        else:
            self.repeat_counter += 1
            return random.choice(self.last_train_data)

    def srgb_to_linear(self, srgb_image):
        # Apply the inverse sRGB gamma curve
        mask = srgb_image <= 0.04045
        srgb_image[mask] = srgb_image[mask] / 12.92
        srgb_image[~mask] = ((srgb_image[~mask] + 0.055) / 1.055) ** 2.4

        return srgb_image

    def apply_acescc(self, linear_image):
        const_neg16 = torch.tensor(2**-16, dtype=linear_image.dtype, device=linear_image.device)
        const_neg15 = torch.tensor(2**-15, dtype=linear_image.dtype, device=linear_image.device)
        const_972 = torch.tensor(9.72, dtype=linear_image.dtype, device=linear_image.device)
        const_1752 = torch.tensor(17.52, dtype=linear_image.dtype, device=linear_image.device)
        
        condition = linear_image < 0
        value_if_true = (torch.log2(const_neg16) + const_972) / const_1752
        value_if_false = (torch.log2(const_neg16 + linear_image * 0.5) + const_972) / const_1752
        ACEScc = torch.where(condition, value_if_true, value_if_false)

        condition = linear_image >= const_neg15
        value_if_true = (torch.log2(linear_image) + const_972) / const_1752
        ACEScc = torch.where(condition, value_if_true, ACEScc)
        
        del value_if_true
        del value_if_false

        return ACEScc

    def __getitem__(self, index):
        train_data = self.getimg(index)

        src_img0 = train_data['start']
        src_img1 = train_data['gt']
        src_img2 = train_data['end']
        imgh = train_data['h']
        imgw = train_data['w']
        ratio = train_data['ratio']
        description = train_data['description']
        images_idx = self.train_data_index

        batch_img0 = []
        batch_img1 = []
        batch_img2 = []

        for batch_index in range(self.batch_size):

            img0, img1, img2 = self.crop(src_img0, src_img1, src_img2, self.h, self.w)

            img0 = torch.from_numpy(img0).to(device = self.device, dtype = torch.float32)
            img1 = torch.from_numpy(img1).to(device = self.device, dtype = torch.float32)
            img2 = torch.from_numpy(img2).to(device = self.device, dtype = torch.float32)

            img0 = img0.permute(2, 0, 1)
            img1 = img1.permute(2, 0, 1)
            img2 = img2.permute(2, 0, 1)

            if self.generalize == 0:
                # No augmentaton
                pass
            elif self.generalize == 1:
                if random.uniform(0, 1) < 0.5:
                    img0 = img0.flip(-1)
                    img1 = img1.flip(-1)
                    img2 = img2.flip(-1)
            else:
                # Horizontal flip (reverse width)
                if random.uniform(0, 1) < 0.5:
                    img0 = img0.flip(-1)
                    img1 = img1.flip(-1)
                    img2 = img2.flip(-1)

                # Rotation
                if random.uniform(0, 1) < (self.generalize / 100):
                    p = random.uniform(0, 1)
                    if p < 0.25:
                        img0 = torch.flip(img0.transpose(1, 2), [2])
                        img1 = torch.flip(img1.transpose(1, 2), [2])
                        img2 = torch.flip(img2.transpose(1, 2), [2])
                    elif p < 0.5:
                        img0 = torch.flip(img0, [1, 2])
                        img1 = torch.flip(img1, [1, 2])
                        img2 = torch.flip(img2, [1, 2])
                    elif p < 0.75:
                        img0 = torch.flip(img0.transpose(1, 2), [1])
                        img1 = torch.flip(img1.transpose(1, 2), [1])
                        img2 = torch.flip(img2.transpose(1, 2), [1])

                if random.uniform(0, 1) < (self.generalize / 100):
                    # Vertical flip (reverse height)
                    if random.uniform(0, 1) < 0.5:
                        img0 = img0.flip(-2)
                        img1 = img1.flip(-2)
                        img2 = img2.flip(-2)

                if random.uniform(0, 1) < (self.generalize / 100):
                    # Depth-wise flip (reverse channels)
                    if random.uniform(0, 1) < 0.28:
                        img0 = img0.flip(0)
                        img1 = img1.flip(0)
                        img2 = img2.flip(0)

                '''
                if random.uniform(0, 1) < (self.generalize / 100):
                    # Exposure augmentation
                    exp = random.uniform(1 / 8, 2)
                    if random.uniform(0, 1) < 0.4:
                        img0 = img0 * exp
                        img1 = img1 * exp
                        img2 = img2 * exp

                if random.uniform(0, 1) < (self.generalize / 100):
                    # add colour banace shift
                    delta = random.uniform(0, 0.49)
                    r = random.uniform(1-delta, 1+delta)
                    g = random.uniform(1-delta, 1+delta)
                    b = random.uniform(1-delta, 1+delta)
                    multipliers = torch.tensor([r, g, b]).view(3, 1, 1).to(device = device, dtype = torch.float32)
                    img0 = img0 * multipliers
                    img1 = img1 * multipliers
                    img2 = img2 * multipliers
                    del multipliers
                
                def gamma_up(img, gamma = 1.18):
                    return torch.sign(img) * torch.pow(torch.abs(img), 1 / gamma )
                
                if random.uniform(0, 1) < (self.generalize / 100):
                    if random.uniform(0, 1) < 0.44:
                        gamma = random.uniform(0.9, 1.9)
                        img0 = gamma_up(img0, gamma=gamma)
                        img1 = gamma_up(img1, gamma=gamma)
                        img2 = gamma_up(img2, gamma=gamma)
                '''

            '''
            # Convert to ACEScc
            if random.uniform(0, 1) < (self.acescc_rate / 100):
                img0 = self.apply_acescc(torch.clamp(img0, min=0.01))
                img1 = self.apply_acescc(torch.clamp(img1, min=0.01))
                img2 = self.apply_acescc(torch.clamp(img2, min=0.01))
            '''
            
            batch_img0.append(img0)
            batch_img1.append(img1)
            batch_img2.append(img2)

        # del train_data, src_img0, src_img1, src_img2

        return torch.stack(batch_img0), torch.stack(batch_img1), torch.stack(batch_img2), ratio, images_idx, description

def get_dataset(
        data_root, 
        batch_size = 8, 
        device = None, 
        frame_size=448, 
        max_window=24,
        acescc_rate = 0,
        generalize = 80,
        repeat = 1,
        sequential = False,
        start_reader=True
        ):

    return TimewarpMLDataset(
        data_root, 
        batch_size=batch_size, 
        device=device, 
        frame_size=frame_size, 
        max_window=max_window,
        acescc_rate=acescc_rate,
        generalize=generalize,
        repeat=repeat,
        sequential = sequential,
        start_reader=start_reader
        )

def clear_lines(n=2):
    """Clears a specified number of lines in the terminal."""
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)

def warp(tenInput, tenFlow):
    backwarp_tenGrid = {}
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device = tenInput.device, dtype = tenInput.dtype)
    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

def psnr_torch(imageA, imageB, max_pixel=1.0):
    mse = torch.mean((imageA.cpu().detach().data - imageB.cpu().detach().data) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def create_timestamp_uid():
    import random
    import uuid
    from datetime import datetime

    def number_to_letter(number):
        # Map each digit to a letter
        mapping = {
            '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E',
            '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'
        }
        return ''.join(mapping.get(char, char) for char in number)

    uid = ((str(uuid.uuid4()).replace('-', '')).upper())
    uid = ''.join(random.sample(number_to_letter(uid), 4))
    timestamp = (datetime.now()).strftime('%Y%b%d_%H%M').upper()
    return f'{timestamp}_{uid}'

def create_csv_file(file_name, fieldnames):
    import csv
    """
    Creates a CSV file with the specified field names as headers.
    """
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def append_row_to_csv(file_name, row):
    import csv
    """
    Appends a single row to an existing CSV file.
    """
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        writer.writerow(row)

class MaxNValues:
    def __init__(self, n):
        """
        Initializes the MaxNValues object.

        Parameters:
        - n (int): The maximum number of top values to keep.
        """
        self.n = n  # Maximum number of values to keep
        self.heap = []  # Min-heap to store the top n values (as tuples of (value, data))

    def add(self, value, data):
        """
        Adds a new value and its associated dictionary to the collection.
        Keeps only the top n values.

        Parameters:
        - value (float): The float value to add.
        - data (dict): The dictionary associated with the value.
        """
        if len(self.heap) < self.n:
            # If the heap is not full, push the new item
            heapq.heappush(self.heap, (value, data))
        else:
            # If the new value is greater than the smallest in the heap, replace it
            if value > self.heap[0][0]:
                heapq.heapreplace(self.heap, (value, data))

        self.heap = heapq.nlargest(self.n, self.heap)
        heapq.heapify(self.heap)

    def get_values(self):
        """
        Returns the list of top n values and their associated data,
        sorted in descending order.

        Returns:
        - List[Tuple[float, dict]]: A list of tuples containing the values and their data.
        """
        # Sort the heap in descending order based on the values
        return sorted(self.heap, key=lambda x: x[0], reverse=True)

    def reset(self):
        """
        Clears the heap, removing all stored values.
        """
        self.heap = []

    def set_n(self, new_n):
        """
        Sets a new value for n and adjusts the heap accordingly.

        Parameters:
        - new_n (int): The new maximum number of top values to keep.
        """
        self.n = new_n
        if len(self.heap) > new_n:
            # Keep only the top new_n values
            self.heap = heapq.nlargest(new_n, self.heap)
            heapq.heapify(self.heap)

    def __len__(self):
        """
        Returns the current number of values stored.

        Returns:
        - int: The number of values in the heap.
        """
        return len(self.heap)

class MinNValues:
    def __init__(self, n):
        """
        Initializes the MinNValues object.

        Parameters:
        - n (int): The maximum number of minimum values to keep.
        """
        self.n = n  # Maximum number of values to keep
        self.heap = []  # Max-heap to store the top n minimum values (as tuples of (-value, data))

    def add(self, value, data):
        """
        Adds a new value and its associated dictionary to the collection.
        Keeps only the top n minimum values.

        Parameters:
        - value (float): The float value to add.
        - data (dict): The dictionary associated with the value.
        """
        # Invert the value to simulate a max-heap
        heap_item = (-value, data)
        if len(self.heap) < self.n:
            # If the heap is not full, push the new item
            heapq.heappush(self.heap, heap_item)
        else:
            # If the new value is smaller than the largest in the heap
            if -value > self.heap[0][0]:
                # Replace the largest value with the new value
                heapq.heapreplace(self.heap, heap_item)

        self.heap = heapq.nsmallest(self.n, self.heap)
        heapq.heapify(self.heap)

    def get_values(self):
        """
        Returns the list of top n minimum values and their associated data,
        sorted in ascending order.

        Returns:
        - List[Tuple[float, dict]]: A list of tuples containing the values and their data.
        """
        # Convert inverted values back to positive and sort in ascending order
        sorted_heap = sorted([(-item[0], item[1]) for item in self.heap], key=lambda x: x[0])
        return sorted_heap

    def reset(self):
        """
        Clears the heap, removing all stored values.
        """
        self.heap = []

    def set_n(self, new_n):
        """
        Sets a new value for n and adjusts the heap accordingly.

        Parameters:
        - new_n (int): The new maximum number of top values to keep.
        """
        self.n = new_n
        if len(self.heap) > new_n:
            # Keep only the top new_n values
            self.heap = heapq.nsmallest(new_n, self.heap)
            heapq.heapify(self.heap)

    def __len__(self):
        """
        Returns the current number of values stored.

        Returns:
        - int: The number of values in the heap.
        """
        return len(self.heap)

def diffmatte(tensor1, tensor2):
    """
    Computes the difference matte between two tensors.

    Parameters:
    - tensor1 (torch.Tensor): First tensor of shape (n, c, h, w)
    - tensor2 (torch.Tensor): Second tensor of shape (n, c, h, w)
    - threshold (float): Threshold value to binarize the difference matte

    Returns:
    - difference_matte (torch.Tensor): Tensor of shape (n, 1, h, w) with values between 0 and 1
    """
    # Ensure the tensors are of the same shape
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"

    # Compute the per-pixel L2 norm difference across the channel dimension
    difference = torch.norm(tensor1 - tensor2, p=2, dim=1, keepdim=True)  # Shape: (n, 1, h, w)

    # Normalize the difference to range [0, 1]
    max_val = difference.view(difference.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    difference_normalized = difference / (max_val + 1e-8)  # Add epsilon to prevent division by zero

    return difference_normalized

def variance_loss(tensor, threshold):
    mean = tensor.mean()
    variance = tensor.std()
    mean_deviation = abs(mean - 0.5)
    variance_loss = torch.relu(threshold - variance).item() # / (threshold + 1e-11)
    return variance_loss + 0.1 * mean_deviation

def sinusoidal_scale_fn(x):
    import math
    # x is a fraction of the cycle's progress (0 to 1)
    return 0.5 * (1 + math.sin(math.pi * (x - 0.5)))

class LapLoss(torch.nn.Module):

    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.register_buffer('gk', self.gauss_kernel(channels=channels))

    def gauss_kernel(self, size=5, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1.],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        return torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])

    def downsample(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=0.5, mode='bilinear', align_corners=False)

    def upsample(self, x, size):
        # bilinear upsample to exact target size, then Gaussian smooth.
        # No 4x factor needed since we're not inserting zeros (unlike classical pyrUp).
        x = torch.nn.functional.interpolate(
            x, size=size, mode='bilinear', align_corners=False)
        return self.conv_gauss(x, self.gk)

    def laplacian_pyramid(self, img, max_levels):
        current = img
        pyr = []
        for _ in range(max_levels):
            h, w = current.shape[2], current.shape[3]
            filtered = self.conv_gauss(current, self.gk)
            down = self.downsample(filtered)
            up = self.upsample(down, size=(h, w))
            pyr.append(current - up)
            current = down
        return pyr

    def forward(self, input, target):
        pyr_input  = self.laplacian_pyramid(input,  self.max_levels)
        pyr_target = self.laplacian_pyramid(target, self.max_levels)
        return sum(
            torch.nn.functional.l1_loss(a, b)
            for a, b in zip(pyr_input, pyr_target)
        )

def compress(x):
    src_dtype = x.dtype
    x = x.float()
    scale = torch.tanh(torch.tensor(1.0))
    x = torch.where(
        (x >= -1) & (x <= 1), scale * x,
        torch.tanh(x)
    )
    x = (x + 1) / 2
    x = x.to(dtype = src_dtype)
    return x

def downscale_flow(flow, sh, sw):
    """Resize a pixel-unit flow field to (sh, sw), rescaling the vectors to match.
    warp() reads flow in pixels, so the values must scale with the resolution."""
    _, _, h, w = flow.shape
    f = torch.nn.functional.interpolate(flow, size=(sh, sw), mode='bilinear', align_corners=False)
    return torch.cat([f[:, 0:1] * (sw / w), f[:, 1:2] * (sh / h)], dim=1)

def compute_lpips(loss_fn, pred, gt, max_size=256):
    _, _, h, w = pred.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        size = (round(h * scale), round(w * scale))
        pred = torch.nn.functional.interpolate(pred, size=size, mode='bilinear', align_corners=False)
        gt   = torch.nn.functional.interpolate(gt,   size=size, mode='bilinear', align_corners=False)
    return loss_fn(pred * 2 - 1, gt * 2 - 1).mean()

def fourier_loss_half_res(img1, img2):
    # Downscale to half resolution using bicubic interpolation
    img1_down = torch.nn.functional.interpolate(img1, scale_factor=0.5, mode='bicubic', align_corners=False, antialias=True)
    img2_down = torch.nn.functional.interpolate(img2, scale_factor=0.5, mode='bicubic', align_corners=False, antialias=True)

    # Apply real 2D FFT
    fft1 = torch.fft.rfft2(img1_down, norm='ortho')
    fft2 = torch.fft.rfft2(img2_down, norm='ortho')

    # Compute magnitude difference
    mag1 = torch.abs(fft1)
    mag2 = torch.abs(fft2)

    # Use L1 or L2 loss in Fourier domain
    return torch.nn.functional.l1_loss(mag1, mag2)

class Ternary(torch.nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        w = np.transpose(w, (3, 2, 0, 1))
        self.register_buffer("w", torch.tensor(w).float())

    def transform(self, img):
        patches = torch.nn.functional.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), dim=1, keepdim=True)
        return dist_norm  # shape (N, 1, H, W)

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = torch.nn.functional.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        loss_map = self.hamming(img0, img1)
        mask = self.valid_mask(img0, 1)
        masked_loss = loss_map * mask
        scalar_loss = masked_loss.sum() / mask.sum()
        return scalar_loss, masked_loss  # (scalar, (N,1,H,W))

class Sobel(torch.nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()

        kernel = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()

        # Register Sobel X and Y as buffers
        self.register_buffer('kernelX', kernel.unsqueeze(0).unsqueeze(0))           # shape [1, 1, 3, 3]
        self.register_buffer('kernelY', kernel.T.contiguous().unsqueeze(0).unsqueeze(0))  # shape [1, 1, 3, 3]

    def forward(self, pred, gt):
        N, C, H, W = pred.shape
        img_stack = torch.cat(
            [pred.reshape(N * C, 1, H, W), gt.reshape(N * C, 1, H, W)], dim=0)

        sobel_stack_x = torch.nn.functional.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = torch.nn.functional.conv2d(img_stack, self.kernelY, padding=1)

        pred_X, gt_X = sobel_stack_x[:N * C], sobel_stack_x[N * C:]
        pred_Y, gt_Y = sobel_stack_y[:N * C], sobel_stack_y[N * C:]

        L1X = torch.abs(pred_X - gt_X)
        L1Y = torch.abs(pred_Y - gt_Y)

        loss = L1X + L1Y  # shape: (N*C, 1, H, W)
        return loss.mean()

class TrainFeed:
    """Drop-in replacement for the old training dataset object. Exposes __len__
    (steps/epoch), .repeat_count (==1; reuse lives in the pool),
    .reshuffle()/.set_epoch(), and .next_batch() yielding {img0,img1,img2,
    ratio,specs}. Frames are AP1 *linear* (tonemap off) so the loop owns colour."""
    def __init__(self, sequences, *, batch_size, frame_size, seed, rank, world_size,
                 num_workers, cache_items, pool_size, reuse, pool_order,
                 pad_tolerance, rotation_prob, max_long_side, hflip, vflip, cflip,
                 pin_memory, steps_per_epoch, window_mode='full', bidirectional=True,
                 epoch=0):
        self.dataset = TimewarpDataset(
            sequences, frame_size=frame_size, seed=seed,
            hflip_prob=hflip, vflip_prob=vflip, cflip_prob=cflip, epoch=epoch,
            reader=functools.partial(default_reader, tonemap=False))
        self.sampler = TimewarpBatchSampler(
            sequences, batch_size=batch_size, frame_size=frame_size,
            pad_tolerance=pad_tolerance, rotation_prob=rotation_prob,
            max_long_side=max_long_side, steps_per_epoch=steps_per_epoch,
            window_mode=window_mode, bidirectional=bidirectional,
            seed=seed, rank=rank, world_size=world_size, epoch=epoch)
        self.loader = build_dataloader(
            self.dataset, self.sampler, num_workers=num_workers,
            cache_items=cache_items, pin_memory=pin_memory, return_mask=False)
        self.pool = BatchPool(
            self.loader, steps_per_epoch=steps_per_epoch, size=pool_size,
            reuse=reuse, order=pool_order, seed=seed, epoch=epoch)
        self.repeat_count = 1
        self._steps = steps_per_epoch
        self.epoch = epoch
        self._it = None

    def __len__(self):
        return self._steps

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.sampler.set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.pool.set_epoch(epoch)
        if self._it is not None:
            self._it.close()
            self._it = None

    def reshuffle(self):
        self.set_epoch(self.epoch + 1)

    def next_batch(self):
        if self._it is None:
            self._it = iter(self.pool)
        try:
            return next(self._it)
        except StopIteration:
            self._it.close()
            self._it = iter(self.pool)
            return next(self._it)

def find_and_import_model(models_dir='models', base_name=None, model_name=None, model_file=None):
    """
    Dynamically imports a model from the models/ package and returns its
    `Model` class.

    Resolution order:
    - model_file:  exact file name ('flownet4_v001.py'), imported verbatim.
    - model_name:  exact module name ('flownet4_v001').
    - base_name:   newest file matching '<base_name>_v*' (e.g. 'flownet'),
                   falling back to the newest .py in the directory.
    - none of the above: newest .py in the directory.

    Version ordering is lexical on the file name, which matches the
    flownetN_vNNN naming convention (zero-padded version suffix).
    """
    import os
    import re
    import importlib

    if model_file:
        module_name = model_file[:-3]  # Remove '.py' from filename to get module name
        module_path = f"models.{module_name}"
        module = importlib.import_module(module_path)
        model_object = getattr(module, 'Model')
        return model_object

    # Resolve the absolute path of the models directory
    models_abs_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            models_dir
        )
    )

    # List all files in the models directory
    try:
        files = os.listdir(models_abs_path)
    except FileNotFoundError:
        print(f"Directory not found: {models_abs_path}")
        return None

    py_files = [f for f in files if f.endswith('.py') and f != '__init__.py']

    # Filter files based on model_name or base_name
    if model_name:
        # Look for a specific model version
        filtered_files = [f for f in py_files if f == f"{model_name}.py"]
    elif base_name:
        # Newest file of the '<base_name>_v*' family, else newest of all
        family = [f for f in py_files if re.match(rf"^{re.escape(base_name)}_v", f)]
        filtered_files = sorted(family, reverse=True)[:1] if family else sorted(py_files, reverse=True)[:1]
    else:
        filtered_files = sorted(py_files, reverse=True)[:1]

    # Import the module and return the Model object
    if filtered_files:
        module_name = filtered_files[0][:-3]  # Remove '.py' from filename to get module name
        module_path = f"models.{module_name}"
        module = importlib.import_module(module_path)
        model_object = getattr(module, 'Model')
        return model_object
    else:
        print(f"Model not found: {base_name or model_name}")
        return None

def build_train_feed(args, rank, world_size, is_master, max_dataset_window, window_mode, bidirectional):
    """Build the per-rank training feed from the data/ pipeline.

    window_mode/bidirectional are computed in main() (they are shared with the
    no-folder eval description synthesis) and passed in here.
    Returns (feed, train_manifest).
    """
    import math

    def _manifest(root):
        return build_manifest(root, max_window=max_dataset_window,
                              read_headers=True, verbose=is_master)

    def _sequences(root):
        if world_size > 1:
            m = _manifest(root) if rank == 0 else None
            torch.distributed.barrier()
            if rank != 0:
                m = _manifest(root)
        else:
            m = _manifest(root)
        return m

    train_m = _sequences(args.dataset_path)
    val_seqs = _sequences(args.val_folder).sequences if args.val_folder else None
    test_seqs = _sequences(args.test_folder).sequences if args.test_folder else None

    fractions = (max(0.0, 1.0 - args.val_frac - args.test_frac),
                 args.val_frac, args.test_frac)
    split = split_sequences(
        train_m.sequences, val_sequences=val_seqs, test_sequences=test_seqs,
        fractions=fractions, seed=args.seed, verbose=is_master)
    train_seqs = split.train

    if args.steps_per_epoch and args.steps_per_epoch > 0:
        steps = args.steps_per_epoch
    else:
        total = sum(x.num_windows(bidirectional, window_mode) for x in train_seqs
                    if x.height is not None and x.num_windows(bidirectional, window_mode) > 0)
        steps = max(1, math.ceil(total / (args.batch_size * max(1, world_size))))

    pool_order = "sequential" if args.sequential else args.pool_order
    reuse = 1 if args.sequential else args.reuse

    feed = TrainFeed(
        train_seqs, batch_size=args.batch_size, frame_size=args.frame_size, seed=args.seed,
        rank=rank, world_size=world_size, num_workers=args.num_workers,
        cache_items=args.cache_items, pool_size=args.pool_size, reuse=reuse,
        pool_order=pool_order, pad_tolerance=args.pad_tolerance,
        rotation_prob=args.rotation_prob,
        max_long_side=(args.max_long_side if args.max_long_side > 0 else None),
        hflip=args.hflip, vflip=args.vflip, cflip=args.cflip,
        pin_memory=args.pin_memory, steps_per_epoch=steps,
        window_mode=window_mode, bidirectional=bidirectional)

    if is_master:
        print(f"[train feed] {len(train_seqs)} train sequences, {steps} steps/epoch x "
              f"{args.batch_size} (world_size {world_size}), pool_size {args.pool_size} "
              f"reuse {reuse} order {pool_order}, {args.num_workers} workers, "
              f"pin_memory {args.pin_memory}, window_mode {window_mode}"
              f"{'' if bidirectional else ' (uni)'}")
    return feed, train_m

current_state_dict = {}

def main(rank, world_size):
    global current_state_dict
    parser = argparse.ArgumentParser(description='Timewarp model training (DDP, new dataset pipeline).')

    def check_range_percent(value):
        ivalue = int(value)
        if ivalue < 0 or ivalue > 100:
            raise argparse.ArgumentTypeError(f"Percent must be between 0 and 100, got value={ivalue}")
        return ivalue

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset root (folders of OpenEXR sequences)')

    # Optimizer / learning rate
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
    parser.add_argument('--pulse', type=float, default=10000, help='Period in steps to pulse learning rate (default: 10K)')
    parser.add_argument('--pulse_amplitude', type=float, default=1e-1, help='Learning rate pulse amplitude (fraction) (default: 0.1)')
    parser.add_argument('--onecycle', type=int, default=-1, help='Train one cycle for N epochs (default: off)')
    parser.add_argument('--cyclic', type=int, default=-1, help='Use cyclic LR scheduler with period N steps (default: off)')
    parser.add_argument('--weight_decay', type=float, default=-1, help='AdamW weight decay (default: derived from --generalize)')

    # Model
    parser.add_argument('--state_file', type=str, default=None, help='Pre-trained model state dict: resume point or init weights (optional)')
    parser.add_argument('--model', type=str, default=None, help='Model name (file in models/ without .py); default: newest flownet*')
    parser.add_argument('--legacy_model', type=str, default=None, help='Load a RIFE-style state dict into the model with strict=False (optional)')
    parser.add_argument('--device', type=str, default='0', help="GPU index, or 'cpu' for CPU-only runs (default: 0)")
    parser.add_argument('--all_gpus', action='store_true', dest='all_gpus', default=False, help='Train with DistributedDataParallel across all visible GPUs')
    parser.add_argument('--compile', action='store_true', dest='compile', default=False, help='Compile the model with torch.compile')

    # Data / training pipeline
    parser.add_argument('--batch_size', '--batch', dest='batch_size', type=int, default=2, help='Batch size per GPU (default: 2)')
    parser.add_argument('--frame_size', type=int, default=448, help='Training frame size in pixels (default: 448)')
    parser.add_argument('--max_window', type=int, default=12, help='Max temporal window in frames (default: 12)')
    parser.add_argument('--window_mode', type=str, default='full', choices=['full', 'fixed'],
                        help="'full' = every window size 3..max_window at every interior gt (default, timewarp); 'fixed' = max_window-sized windows only (stab)")
    parser.add_argument('--window_bidirectional', type=str, default='auto', choices=['auto', 'yes', 'no'],
                        help='Also emit reversed windows; auto = yes for full mode, no for fixed (the loop already builds both pairs)')
    parser.add_argument('--sequential', action='store_true', dest='sequential', default=False, help='Keep window order, do not reshuffle between epochs')
    parser.add_argument('--scales', type=str, default='8,4,2,1', help='Pyramid scale list for the flownet blocks (default: 8,4,2,1)')
    parser.add_argument('--iterations', type=int, default=1, help='Refine each pyramid level N times (default: 1)')
    parser.add_argument('--generalize', type=check_range_percent, default=85, help='Legacy eval-dataset augmentation rate 0-100 (default: 85)')
    parser.add_argument('--repeat', type=int, default=1, help='Legacy eval-dataset: repeat each triad N times (default: 1)')
    parser.add_argument('--ap0', action='store_true', dest='ap0', default=False, help='Input EXRs are AP0 linear (default: AP1)')

    # Dataset pipeline (data/ package)
    parser.add_argument('--seed', type=int, default=1234, help='Global seed for splits/sampler/pool/augmentation (default: 1234)')
    parser.add_argument('--val_folder', type=str, default=None, help='External held-out validation root')
    parser.add_argument('--test_folder', type=str, default=None, help='External held-out test root (kept out of the train pool)')
    parser.add_argument('--val_frac', type=float, default=0.0, help='Validation split fraction when --val_folder is not set (default: 0)')
    parser.add_argument('--test_frac', type=float, default=0.0, help='Test split fraction when --test_folder is not set (default: 0)')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader worker processes (default: 8)')
    parser.add_argument('--cache_items', type=int, default=256, help='Per-worker decoded-frame LRU size (default: 256)')
    parser.add_argument('--pool_size', type=int, default=48, help='Reuse-pool size in batches (default: 48)')
    parser.add_argument('--reuse', type=int, default=4, help='Reuse-pool serve factor; 1 = fresh batch every step (default: 4)')
    parser.add_argument('--pool_order', type=str, default='random', choices=['random', 'sequential'], help='Pool serve order (default: random)')
    parser.add_argument('--pad_tolerance', type=float, default=0.10, help='Max per-batch long-side padding fraction (default: 0.10)')
    parser.add_argument('--rotation_prob', type=float, default=0.5, help='Probability of +/-90 rotation augmentation (default: 0.5)')
    parser.add_argument('--max_long_side', type=int, default=0, help='Drop sequences whose resized long side exceeds this (default: 0 = no limit)')
    parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip probability (default: 0.5)')
    parser.add_argument('--vflip', type=float, default=0.0, help='Vertical flip probability (default: 0)')
    parser.add_argument('--cflip', type=float, default=0.0, help='Channel flip probability (default: 0)')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Pin DataLoader memory (default: off)')
    parser.add_argument('--steps_per_epoch', type=int, default=-1, help='Steps per epoch; -1 = auto from dataset size (default: -1)')
    parser.add_argument('--input_encodings', type=str, default='ap1,rec709,acescct',
                        help='Model-input encodings sampled equally per step (comma subset of ap1,rec709,acescct)')

    # Loss
    parser.add_argument('--deep_sup', type=float, default=0.0,
                        help='Extra deep-supervision weight: one coarse pyramid level sampled per step '
                             '(the per-level loop already supervises every level; this adds a sampled coarse term) (default: 0 = off)')
    parser.add_argument('--lpips_alternate', action='store_true', default=False,
                        help='Compute the per-level LPIPS term on alternate levels only (halves its cost) (default: off)')

    # Epochs / checkpoint
    parser.add_argument('--first_epoch', type=int, default=-1, help='Start epoch (default: from checkpoint)')
    parser.add_argument('--epochs', type=int, default=-1, help='Stop after N epochs (default: unlimited)')
    parser.add_argument('--reset_stats', action='store_true', dest='reset_stats', default=False, help='Reset saved step, epoch and loss stats')
    parser.add_argument('--save', type=int, default=10000, help='Save model state dict every N steps (default: 10000)')

    # Previews
    parser.add_argument('--preview', type=int, default=100, help='Save a preview triad every N steps (default: 100)')
    parser.add_argument('--preview_max', type=int, default=0, help='Save separate preview for N highest-error samples (default: 0 = off)')
    parser.add_argument('--preview_min', type=int, default=0, help='Save separate preview for N lowest-error samples (default: 0 = off)')
    parser.add_argument('--preview_maxmin_steps', type=int, default=10000, help='Save max or min preview every N steps (default: 10000)')

    # Evaluation
    parser.add_argument('--eval', type=int, dest='eval', default=-1, help='Evaluate every N steps (default: off)')
    parser.add_argument('--eval_first', action='store_true', dest='eval_first', default=False, help='Do not skip the evaluation triggered at step 1')
    parser.add_argument('--eval_samples', type=int, dest='eval_samples', default=-1, help='Evaluate N random samples instead of the full eval set')
    parser.add_argument('--eval_seed', type=int, dest='eval_seed', default=1, help='Random seed to select samples if --eval_samples set')
    parser.add_argument('--eval_buffer', type=int, dest='eval_buffer', default=8, help='Write buffer size for evaluated images (default: 8)')
    parser.add_argument('--eval_save_imgs', action='store_true', dest='eval_save_imgs', default=False, help='Save eval result images')
    parser.add_argument('--eval_keep_all', action='store_true', dest='eval_keep_all', default=False, help='Keep eval result folders for each eval step')
    parser.add_argument('--eval_folder', type=str, default=None, help='Folder with clips for evaluation (default: one window per train sequence)')
    parser.add_argument('--eval_half', action='store_true', dest='eval_half', default=False, help='Evaluate in half precision')
    parser.add_argument('--acescc', type=check_range_percent, default=0, help='Legacy eval dataset: percent of frames converted to ACEScc (default: 0)')

    # LR plateau on a rolling average
    parser.add_argument('--avg_window', type=int, default=100000, help='Rolling window (in steps) for the running loss average (default: 100000)')
    parser.add_argument('--plateau_interval', type=int, default=1000, help='Step ReduceLROnPlateau every N steps using the rolling average; 0 = epoch/eval-based behaviour (default: 1000)')
    parser.add_argument('--plateau_patience', type=int, default=10, help='ReduceLROnPlateau patience, counted in plateau_interval units (default: 10)')
    parser.add_argument('--plateau_factor', type=float, default=0.1, help='ReduceLROnPlateau LR reduction factor (default: 0.1)')

    args = parser.parse_args()
    training_scale = [int(s.strip()) for s in args.scales.split(',') if s.strip()]

    # -------------------------------------------------------------------------
    # DDP init
    # -------------------------------------------------------------------------
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        device = torch.device(f'cuda:{rank}') if not platform.system() == 'Darwin' else torch.device('mps')
        if not platform.system() == 'Darwin':
            torch.cuda.set_device(device)
    else:
        if str(args.device).lower() == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{args.device}')
            if not platform.system() == 'Darwin':
                torch.cuda.set_device(device)

    # True only if THIS process actually runs on CUDA. torch.cuda.is_available()
    # is not enough: it is True on any box with a driver, and the first CUDA API
    # call (sync/empty_cache) would lazily init a context on cuda:0 — on this
    # machine that is the LLM GPU and must never be touched.
    use_cuda = (device.type == 'cuda')

    is_master = (rank == 0)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    Flownet = None
    checkpoint = None

    if args.model:
        model_name = args.model
        Flownet = find_and_import_model(base_name='flownet', model_name=model_name)
        if args.state_file:
            try:
                checkpoint = torch.load(args.state_file, map_location=device, weights_only=False)
                if is_master:
                    print('loaded previously saved model checkpoint')
            except Exception as e:
                if is_master:
                    print(f'unable to load saved model checkpoint: {e}')
    elif args.state_file and os.path.isfile(args.state_file):
        try:
            checkpoint = torch.load(args.state_file, map_location=device, weights_only=False)
            if is_master:
                print('loaded previously saved model checkpoint')
        except Exception as e:
            if is_master:
                print(f'unable to load saved model checkpoint: {e}')
                sys.exit()

        model_info = checkpoint.get('model_info')
        model_file = model_info.get('file')
        Flownet = find_and_import_model(model_file=model_file)
    else:
        if not args.state_file:
            if is_master:
                print('Please specify either model name or model state file')
            return
        if not os.path.isfile(args.state_file):
            if is_master:
                print(f'Model state file {args.state_file} does not exist and "--model" flag is not set to start from scratch')
            return

    if Flownet is None:
        if is_master:
            print(f'Unable to load model {args.model}')
        return

    model_info = Flownet.get_info()
    if is_master:
        print('Model info:')
        pprint(model_info)

    max_dataset_window = args.max_window
    if not model_info.get('ratio_support'):
        max_dataset_window = 3

    if args.compile:
        flownet_uncompiled = Flownet().get_training_model()().to(torch.float32).to(device)
        flownet = torch.compile(flownet_uncompiled, mode='reduce-overhead')
    else:
        flownet = Flownet().get_training_model()().to(device)

    # -------------------------------------------------------------------------
    # DDP wrapping
    # -------------------------------------------------------------------------
    if world_size > 1:
        flownet = torch.nn.parallel.DistributedDataParallel(flownet, device_ids=[rank])
        if is_master:
            print(f'Using DistributedDataParallel across {world_size} GPUs')

    if is_master:
        if not os.path.isdir(os.path.join(args.dataset_path, 'preview')):
            os.makedirs(os.path.join(args.dataset_path, 'preview'))

    frame_size = args.frame_size

    # Window planning (single source of truth for feed and eval)
    window_mode = args.window_mode
    if args.window_bidirectional == 'auto':
        # 'fixed' pairs both directions inside the loop already (img0/img2 -> img1),
        # so reversed windows would be exact duplicates
        bidirectional = (window_mode == 'full')
    else:
        bidirectional = (args.window_bidirectional == 'yes')

    # ---- new dataset pipeline replaces the old streaming reader ----
    dataset, train_m = build_train_feed(args, rank, world_size, is_master, max_dataset_window,
                                        window_mode, bidirectional)

    if args.eval_folder:
        if is_master:
            print(f'Scanning data for evaluation:')
        eval_dataset = get_dataset(
            args.eval_folder,
            batch_size=args.batch_size,
            device=device,
            frame_size=frame_size,
            max_window=max_dataset_window,
            acescc_rate=args.acescc,
            generalize=args.generalize,
            repeat=args.repeat,
            sequential=True,
            start_reader=is_master
        )
    else:
        # no external eval folder: evaluate one window per train sequence
        eval_dataset = None

    # -------------------------------------------------------------------------
    # Background write threads — rank 0 only
    # -------------------------------------------------------------------------
    if is_master:
        def write_images(write_image_queue):
            while True:
                if write_shutdown.is_set():
                    return
                try:
                    write_data = write_image_queue.get_nowait()
                    preview_index = write_data.get('preview_index', 0)
                    preview_folder = write_data["preview_folder"]
                    if not os.path.isdir(preview_folder):
                        os.makedirs(preview_folder)
                    write_exr(write_data['sample_source1'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_A_incoming.exr'), half_float=True)
                    write_exr(write_data['sample_source2'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_B_outgoing.exr'), half_float=True)
                    write_exr(write_data['sample_target'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_C_target.exr'), half_float=True)
                    write_exr(write_data['sample_output'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_D_output.exr'), half_float=True)
                    write_exr(write_data['sample_output_diff'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_E_diff.exr'), half_float=True)
                    del write_data
                except:
                    time.sleep(1e-2)

        def write_eval_images(write_eval_image_queue):
            while True:
                if write_shutdown.is_set():
                    return
                try:
                    write_data = write_eval_image_queue.get_nowait()
                    write_exr(write_data['sample_source1'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_source1_name']), half_float=True)
                    write_exr(write_data['sample_source2'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_source2_name']), half_float=True)
                    write_exr(write_data['sample_target'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_target_name']), half_float=True)
                    write_exr(write_data['sample_output'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_output_name']), half_float=True)
                    write_exr(write_data['sample_output_diff'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_output_diff_name']), half_float=True)
                    write_exr(write_data['sample_output_conf'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_output_conf_name']), half_float=True)
                    write_exr(write_data['sample_output_mask'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_output_mask_name']), half_float=True)
                    del write_data
                except:
                    time.sleep(1e-2)

        def write_model_state(write_model_state_queue):
            while True:
                if write_shutdown.is_set():
                    return
                try:
                    current_state_dict = write_model_state_queue.get_nowait()
                except queue.Empty:
                    time.sleep(1e-2)
                    continue
                except Exception:
                    time.sleep(1e-2)
                    continue
                try:
                    trained_model_path = current_state_dict['trained_model_path']
                    # atomic: write to a temp file, back up the previous checkpoint,
                    # then rename over it. A kill mid-save can no longer truncate the
                    # live weights. Distinct tmp name from graceful_exit's so the two
                    # writers can never collide.
                    tmp_path = trained_model_path + '.periodic.tmp'
                    torch.save(current_state_dict, tmp_path)
                    if os.path.isfile(trained_model_path):
                        backup_file = trained_model_path.replace('.pth', '.backup.pth')
                        shutil.copy(trained_model_path, backup_file)
                    os.replace(tmp_path, trained_model_path)
                except Exception as e:
                    sys.stderr.write(f'\nperiodic checkpoint save failed: '
                                     f'{type(e).__name__}: {e}\n')
                    sys.stderr.flush()

        write_image_queue = queue.Queue(maxsize=16)
        write_thread = threading.Thread(target=write_images, args=(write_image_queue,))
        write_thread.daemon = True
        write_thread.start()

        write_eval_image_queue = queue.Queue(maxsize=args.eval_buffer)
        write_eval_thread = threading.Thread(target=write_eval_images, args=(write_eval_image_queue,))
        write_eval_thread.daemon = True
        write_eval_thread.start()

        write_model_state_queue = queue.Queue(maxsize=2)
        write_model_state_thread = threading.Thread(target=write_model_state, args=(write_model_state_queue,))
        write_model_state_thread.daemon = True
        write_model_state_thread.start()

    pulse_dive = args.pulse_amplitude
    pulse_period = args.pulse

    # -------------------------------------------------------------------------
    # LR: square root scaling for DDP — adaptive optimizers benefit from
    # conservative scaling vs linear rule
    # -------------------------------------------------------------------------
    lr = args.lr * (world_size ** 0.5) if world_size > 1 else args.lr
    if is_master and world_size > 1:
        print(f'Scaling LR by sqrt({world_size}): {args.lr:.2e} -> {lr:.2e}')

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    criterion_lap = LapLoss().to(device)
    criterion_huber = torch.nn.HuberLoss(delta=0.001)

    weight_decay = 10 ** (-2 - 0.02 * (args.generalize - 1)) if args.generalize > 1 else 1e-4
    if args.weight_decay != -1:
        weight_decay = args.weight_decay
    optimizer_flownet = torch.optim.AdamW(flownet.parameters(), lr=lr, weight_decay=weight_decay)

    if is_master:
        if args.generalize == 0:
            print(f'Disabling augmentation and setting weight decay to {weight_decay:.2e}')
        elif args.generalize == 1:
            print(f'Setting augmentation to horizontal flip and scale only and weight decay to {weight_decay:.2e}')
        else:
            print(f'Setting augmentation rate to {args.generalize}% and weight decay to {weight_decay:.2e}')

    step = 0
    loaded_step = 0
    current_epoch = 0
    preview_index = 0

    if args.state_file:
        trained_model_path = args.state_file
        if checkpoint is None:
            try:
                checkpoint = torch.load(trained_model_path, map_location=device, weights_only=False)
                if is_master:
                    print('loaded previously saved model checkpoint')
            except Exception as e:
                if is_master:
                    print(f'unable to load saved model: {e}')

        try:
            # DDP wraps model under .module
            target_model = flownet.module if world_size > 1 else flownet
            missing_keys, unexpected_keys = target_model.load_state_dict(checkpoint['flownet_state_dict'], strict=False)
            if is_master:
                print('loaded previously saved net state')
                if missing_keys:
                    print(f'\nMissing keys:\n{missing_keys}\n')
                if unexpected_keys:
                    print(f'\nUnexpected keys:\n{unexpected_keys}\n')
        except Exception as e:
            if is_master:
                print(f'unable to load net state: {e}')

        try:
            optimizer_flownet.load_state_dict(checkpoint['optimizer_flownet_state_dict'])
            for pg in optimizer_flownet.param_groups:
                pg['lr'] = lr
                pg['weight_decay'] = weight_decay
            if is_master:
                print('loaded previously saved optimizer state')
        except Exception as e:
            if is_master:
                print(f'unable to load optimizer state: {e}')

        try:
            loaded_step = checkpoint['step']
            current_epoch = checkpoint['epoch']
            if is_master:
                print(f'loaded step: {loaded_step}')
                print(f'epoch: {current_epoch + 1}')
        except Exception as e:
            if is_master:
                print(f'unable to set step and epoch: {e}')

    else:
        if is_master:
            traned_model_name = 'flameTWML_model_' + create_timestamp_uid() + '.pth'
            if platform.system() == 'Darwin':
                trained_model_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'flameTWML_models')
            else:
                trained_model_dir = os.path.join(os.path.expanduser('~'), 'flameTWML_models')
            if not os.path.isdir(trained_model_dir):
                os.makedirs(trained_model_dir)
            trained_model_path = os.path.join(trained_model_dir, traned_model_name)

    if world_size > 1:
        path_list = [trained_model_path]
        torch.distributed.broadcast_object_list(path_list, src=0)
        trained_model_path = path_list[0]

    if args.legacy_model:
        rife_state_dict = torch.load(args.legacy_model)
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        target_model = flownet.module if world_size > 1 else flownet
        missing_keys, unexpected_keys = target_model.load_state_dict(convert(rife_state_dict), strict=False)
        if is_master:
            print(f'\nMissing keys:\n{missing_keys}\n')
            print(f'\nUnexpected keys:\n{unexpected_keys}\n')

    if args.reset_stats:
        step = 0
        loaded_step = 0
        current_epoch = 0
        preview_index = 0

    if args.onecycle != -1:
        try:
            scheduler_flownet = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_flownet,
                max_lr=lr,
                div_factor=4,
                final_div_factor=1,
                steps_per_epoch=len(dataset) * dataset.repeat_count,
                epochs=args.onecycle,
                last_epoch=-1 if loaded_step == 0 else loaded_step
            )
        except:
            scheduler_flownet = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_flownet,
                max_lr=lr,
                div_factor=4,
                final_div_factor=1,
                steps_per_epoch=len(dataset) * dataset.repeat_count,
                epochs=args.onecycle,
                last_epoch=-1
            )
        if is_master:
            print(f'setting OneCycleLR scheduler with max_lr={lr:.2e}, steps_per_epoch={len(dataset)*dataset.repeat_count}, epochs={args.onecycle}, last: {-1 if loaded_step == 0 else loaded_step}')
        args.epochs = args.onecycle
    elif args.cyclic != -1:
        if is_master:
            print(f'setting CyclicLR scheduler with max_lr={lr:.2e}, base_lr={lr * pulse_dive:.2e}, step_size_up={args.cyclic}')
        scheduler_flownet = torch.optim.lr_scheduler.CyclicLR(
            optimizer_flownet,
            base_lr=lr * pulse_dive,
            max_lr=lr,
            step_size_up=args.cyclic,
            mode='exp_range',
            cycle_momentum=False,
            scale_fn=sinusoidal_scale_fn,
            scale_mode='cycle'
        )
    else:
        if is_master:
            src = (f'rolling avg of {args.avg_window} steps, every {args.plateau_interval} steps'
                   if args.plateau_interval > 0 else 'epoch/eval average')
            print(f'setting ReduceLROnPlateau scheduler with factor={args.plateau_factor}, '
                  f'patience={args.plateau_patience} ({src})')
        scheduler_flownet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_flownet, 'min', factor=args.plateau_factor, patience=args.plateau_patience)

    # -------------------------------------------------------------------------
    # LPIPS — each rank loads its own copy on its own device
    # -------------------------------------------------------------------------
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    import lpips
    # Keep the torch.hub cache inside the repo so the LPIPS backbone resolves
    # offline (first run downloads it into hub/checkpoints/).
    os.environ['TORCH_HOME'] = os.path.abspath(os.path.dirname(__file__))
    loss_fn_alex = lpips.LPIPS(net='alex', spatial=False)
    loss_fn_alex.to(device)

    loss_fn_alex.eval()
    loss_fn_alex.requires_grad_(False)

    warnings.resetwarnings()

    ternary_loss = Ternary().to(device)
    sobel_loss = Sobel().to(device)

    start_timestamp = time.time()
    time_stamp = time.time()
    epoch = current_epoch if args.first_epoch == -1 else args.first_epoch
    step = loaded_step if args.first_epoch == -1 else step
    batch_idx = 0

    if is_master:
        print('\n\n')
        # initial state dict snapshot
        current_state_dict['step'] = int(step)
        current_state_dict['epoch'] = int(epoch)
        current_state_dict['start_timestamp'] = start_timestamp
        current_state_dict['lr'] = optimizer_flownet.param_groups[0]['lr']
        current_state_dict['model_info'] = model_info
        target_model = flownet.module if world_size > 1 else flownet
        current_state_dict['flownet_state_dict'] = target_model.state_dict()
        current_state_dict['optimizer_flownet_state_dict'] = optimizer_flownet.state_dict()
        current_state_dict['trained_model_path'] = trained_model_path

        if not os.path.isfile(f'{os.path.splitext(trained_model_path)[0]}.csv'):
            create_csv_file(
                f'{os.path.splitext(trained_model_path)[0]}.csv',
                ['Epoch', 'Step', 'Min', 'Avg', 'Max', 'PSNR', 'LPIPS']
            )
        if not os.path.isfile(f'{os.path.splitext(trained_model_path)[0]}.eval.csv'):
            create_csv_file(
                f'{os.path.splitext(trained_model_path)[0]}.eval.csv',
                ['Epoch', 'Step', 'Min', 'Avg', 'Max', 'PSNR', 'LPIPS']
            )

    if is_master:
        import signal
        def create_graceful_exit(current_state_dict):
            def graceful_exit(signum, frame):
                # Re-entrant guard: a second Ctrl+C during the save would otherwise
                # start a second concurrent writer on the same file.
                if getattr(graceful_exit, '_saving', False):
                    return
                graceful_exit._saving = True

                path = current_state_dict['trained_model_path']
                # stderr + flush: the loop's clear_lines(2) can erase stdout lines,
                # and these must survive regardless of terminal cursor games.
                sys.stderr.write(f'\n\nInterrupted - saving current state to {path}...\n')
                sys.stderr.write(f'Epoch: {current_state_dict["epoch"] + 1}, '
                                 f'Step: {current_state_dict["step"]}\n')
                sys.stderr.flush()

                ok = False
                try:
                    # atomic write: never leave a half-written checkpoint at `path`
                    tmp = path + '.saving.tmp'
                    torch.save(current_state_dict, tmp)
                    if os.path.isfile(path):
                        shutil.copy(path, path.replace('.pth', '.backup.pth'))
                    os.replace(tmp, path)
                    ok = True
                    sys.stderr.write(f'Saved. Exiting.\n')
                except Exception as e:
                    sys.stderr.write(f'SAVE FAILED: {type(e).__name__}: {e}\n')
                    traceback.print_exc()
                sys.stderr.flush()

                exit_event.set()
                process_exit_event.set()
                # os._exit bypasses SystemExit, which a bare `except:` in the training
                # loop would otherwise swallow (leaving the process running after the
                # save message was already erased). DataLoader workers are daemonic
                # and are reaped with the parent.
                os._exit(0 if ok else 1)
            return graceful_exit
        signal.signal(signal.SIGINT, create_graceful_exit(current_state_dict))

    def exeption_handler(exctype, value, tb):
        exit_event.set()
        process_exit_event.set()
        sys.__excepthook__(exctype, value, tb)
    sys.excepthook = exeption_handler

    min_l1 = float(sys.float_info.max)
    avg_l1 = 0
    max_l1 = 0
    avg_pnsr = 0
    avg_lpips = 0
    avg_loss = 0

    cur_size = max(1, args.avg_window)
    cur_mask = np.full(cur_size, True)
    # Scheduler metric buffer. Kept on EVERY rank (loss_val is already all-reduced,
    # so each rank computes an identical mean) — the display buffers below are
    # rank-0 only, and driving the LR from those would desync ranks.
    sched_buf = None
    sched_count = 0
    cur_l1 = None
    cur_comb = None
    cur_lpips = None

    repeat_count = dataset.repeat_count if dataset.repeat_count > 0 else 1
    preview_maxmin_steps = args.preview_maxmin_steps if args.preview_maxmin_steps < len(dataset) * repeat_count else len(dataset) * repeat_count
    max_values = MaxNValues(n=args.preview_max if args.preview_max else 10)
    min_values = MinNValues(n=args.preview_min if args.preview_min else 10)

    best_eval_loss = float(sys.float_info.max)

    data_time = 0
    data_time1 = 0
    data_time2 = 0
    train_time = 0
    model_time = 0

    ap02cct = AP0toACESCCT().to(device)
    ap12cct = AP1toACESCCT().to(device)
    cct2cg = ACESCCTtoACESCG().to(device)
    ap1torec709 = AP1toRec709().to(device)

    # model-input encodings sampled equally per step (colour-space augmentation)
    _valid_enc = {'ap1', 'rec709', 'acescct'}
    input_encodings = [e.strip().lower() for e in args.input_encodings.split(',') if e.strip()]
    input_encodings = [e for e in input_encodings if e in _valid_enc] or ['acescct']
    if is_master:
        print(f'model-input encodings (sampled equally per step): {input_encodings}')

    def encode_input(x_linear, mode):
        # x_linear is AP1 (ACEScg) linear -> the chosen model-input encoding
        if mode == 'ap1':
            return x_linear
        if mode == 'rec709':
            return ap1torec709(x_linear)
        return ap12cct(x_linear)          # 'acescct'  (current behaviour)

    # new dataset is fed per-rank from its own pool; no rank-0 prefetch/scatter
    dataset.set_epoch(epoch)

    loss_LPIPS = torch.zeros(1, device=device, requires_grad=True)

    while True:
        time_stamp = time.time()

        # --- new dataset: each rank pulls its own bucketed batch from its pool ---
        batch = dataset.next_batch()
        img0 = batch['img0']
        img1 = batch['img1']
        img2 = batch['img2']
        ratio = batch['ratio']
        current_desc = {'seq_ids': [x.seq_id for x in batch['specs'][:4]],
                        'rotations': [x.rotation for x in batch['specs'][:4]]}
        idx = batch_idx
        sample_idx = batch_idx

        img0 = img0.to(device, non_blocking=True)
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        ratio = ratio.to(device, non_blocking=True)

        # working / loss / preview space is ACEScct; EXRs are AP1 linear unless --ap0
        if args.ap0:
            img0 = ap02cct(img0)          # AP0 linear -> ACEScct
            img1 = ap02cct(img1)
            img2 = ap02cct(img2)
        else:
            img0 = ap12cct(img0)          # AP1 linear -> ACEScct
            img1 = ap12cct(img1)
            img2 = ap12cct(img2)

        img0_orig = img0.detach().clone()
        img1_orig = img1.detach().clone()
        img2_orig = img2.detach().clone()

        current_lr_str = str(f'{optimizer_flownet.param_groups[0]["lr"]:.2e}')

        data_time = time.time() - time_stamp
        time_stamp = time.time()

        optimizer_flownet.zero_grad()
        flownet.train()

        # Exposure + noise augmentation on the two model inputs only;
        # the GT center frame (img1) stays clean.
        exp1 = random.uniform(1 / 4, 1.4) if random.uniform(0, 1) < 0.5 else 1
        exp2 = random.uniform(1 / 4, 1.4) if random.uniform(0, 1) < 0.5 else 1

        noise1 = min(random.uniform(0, 0.25), random.uniform(0, 0.25)) * torch.clamp(torch.randn_like(img0), 0)
        noise2 = min(random.uniform(0, 0.25), random.uniform(0, 0.25)) * torch.clamp(torch.randn_like(img2), 0)

        # sample one input encoding for this step (AP1 / Rec.709 / ACEScct, equal by default)
        enc = random.choice(input_encodings)
        result = flownet(
            encode_input(cct2cg(img0) * exp1 + noise1, enc),
            encode_input(cct2cg(img2) * exp2 + noise2, enc),
            ratio,
            scale=training_scale,
            iterations=args.iterations
        )

        flow_list = result['flow_list']
        mask_list = result['mask_list']
        conf_list = result['conf_list']
        scales = result.get('scale', training_scale)

        model_time = time.time() - time_stamp
        time_stamp = time.time()

        # ---------------------------------------------------------------------
        # Timewarp loss: every pyramid level is supervised (the model returns
        # all levels in training mode); the finest level additionally gets the
        # full-res structure terms. Weights are the legacy timewarp recipe.
        # ---------------------------------------------------------------------
        loss = torch.zeros(1, device=device, requires_grad=True)
        img1_compr = torch.clamp(compress(img1_orig), 0, 1)

        for i in range(len(flow_list)):
            if flow_list[i] is None:
                continue
            scale = scales[i]
            flow0 = flow_list[i][:, :2]
            flow1 = flow_list[i][:, 2:4]
            mask = mask_list[i]
            conf = conf_list[i]
            output_clean = warp(img0_orig, flow0) * mask + warp(img2_orig, flow1) * (1 - mask)

            output_compr = torch.clamp(compress(output_clean), 0, 1)

            loss_mask = variance_loss(mask, 0.1)
            loss_conf = criterion_l1(conf, diffmatte(output_compr, img1_compr))
            loss_l1 = criterion_l1(
                torch.nn.functional.interpolate(output_compr, scale_factor=1. / scale, mode='bicubic', align_corners=True, antialias=True),
                torch.nn.functional.interpolate(img1_compr, scale_factor=1. / scale, mode='bicubic', align_corners=True, antialias=True)
            )
            loss_lap = criterion_lap(output_compr, img1_compr)
            loss_fourier = fourier_loss_half_res(output_compr, img1_compr)
            if args.lpips_alternate and (i % 2 == 1):
                loss_LPIPS = torch.zeros((), device=device)
            else:
                loss_LPIPS = compute_lpips(loss_fn_alex, output_compr, img1_compr)

            loss = loss + loss_l1 + loss_lap + loss_fourier + 1e-2 * loss_mask + 1e-2 * loss_conf + 1.4e-2 * (1 / (i + 1)) * loss_LPIPS

        # full-res structure terms on the finest level
        loss_ternary, _ = ternary_loss(output_compr, img1_compr)
        loss_sobel = sobel_loss(output_compr, img1_compr)
        loss_LPIPS = compute_lpips(loss_fn_alex, output_compr, img1_compr)
        loss = loss + loss_l1 + loss_lap + loss_fourier + 0.1 * loss_ternary + 0.1 * loss_sobel + 1e-2 * loss_LPIPS

        diff_matte = diffmatte(output_compr, img1_compr)

        # ---------------------------------------------------------------------
        # Deep supervision (extra): one coarse pyramid level per step,
        # importance-sampled. The per-level loop above already supervises every
        # level; this term adds extra pressure on a coarse level (unbiased:
        # E[tot_w * L_i] == sum_i w_i * L_i).
        # ---------------------------------------------------------------------
        loss_deep = torch.zeros((), device=device)
        if args.deep_sup > 0 and len(flow_list) > 1:
            n_coarse = len(flow_list) - 1
            lvl_w = [1.0 / max(1.0, float(scales[i])) for i in range(n_coarse)]
            tot_w = float(sum(lvl_w))
            if tot_w > 0:
                r = random.uniform(0, tot_w)
                acc, lvl = 0.0, 0
                for i, wgt in enumerate(lvl_w):
                    acc += wgt
                    if r <= acc:
                        lvl = i
                        break
                _, _, _h, _w = img0.shape
                _s = max(1.0, float(scales[lvl]))
                _sh = max(64, int(round((_h / _s) / 16) * 16))
                _sw = max(64, int(round((_w / _s) / 16) * 16))
                d_flow0 = downscale_flow(flow_list[lvl][:, :2], _sh, _sw)
                d_flow1 = downscale_flow(flow_list[lvl][:, 2:4], _sh, _sw)
                d_mask = torch.nn.functional.interpolate(mask_list[lvl], size=(_sh, _sw), mode='bilinear', align_corners=False)
                d_img0 = torch.nn.functional.interpolate(img0_orig, size=(_sh, _sw), mode='bilinear', antialias=True)
                d_img1 = torch.nn.functional.interpolate(img1_orig, size=(_sh, _sw), mode='bilinear', antialias=True)
                d_img2 = torch.nn.functional.interpolate(img2_orig, size=(_sh, _sw), mode='bilinear', antialias=True)
                d_out = warp(d_img0, d_flow0) * d_mask + warp(d_img2, d_flow1) * (1 - d_mask)
                # cheap losses only -- perceptual detail is meaningless at coarse scales
                d_term = criterion_l1(d_out, d_img1) + criterion_lap(d_out, d_img1)
                loss_deep = args.deep_sup * tot_w * d_term

        loss = loss + loss_deep

        # non-compressed values for logging (L1 / PSNR / LPIPS)
        loss_l1_log = criterion_l1(output_clean, img1_orig)
        loss_LPIPS_log = compute_lpips(loss_fn_alex, output_clean, img1_orig)

        # ---------------------------------------------------------------------
        # Loss tracking — reduce across ranks so all ranks log the same values
        # ---------------------------------------------------------------------
        if world_size > 1:
            loss_reduced = loss.detach().clone()
            torch.distributed.all_reduce(loss_reduced, op=torch.distributed.ReduceOp.AVG)
            loss_val = float(loss_reduced.item())
            l1_reduced = loss_l1_log.detach().clone()
            torch.distributed.all_reduce(l1_reduced, op=torch.distributed.ReduceOp.AVG)
            l1_val = float(l1_reduced.item())
            lpips_reduced = loss_LPIPS_log.detach().clone()
            torch.distributed.all_reduce(lpips_reduced, op=torch.distributed.ReduceOp.AVG)
            lpips_val = float(lpips_reduced.item())
        else:
            loss_val = float(loss.item())
            l1_val = float(loss_l1_log.item())
            lpips_val = float(torch.mean(loss_LPIPS_log).item())

        # rolling scheduler metric — all ranks, identical values
        if sched_buf is None:
            sched_buf = np.full(cur_size, loss_val, dtype=np.float64)
        sched_buf[step % cur_size] = loss_val
        sched_count += 1

        if is_master:
            if cur_comb is None:
                cur_comb = np.full(cur_size, loss_val)
            if cur_l1 is None:
                cur_l1 = np.full(cur_size, l1_val)
            if cur_lpips is None:
                cur_lpips = np.full(cur_size, lpips_val)

            cur_idx = step % cur_size
            cur_mask[cur_idx] = False
            cur_comb[cur_idx] = loss_val
            cur_l1[cur_idx] = l1_val
            cur_lpips[cur_idx] = lpips_val

            min_l1 = min(min_l1, l1_val)
            max_l1 = max(max_l1, l1_val)
            avg_loss = loss_val if batch_idx == 0 else (avg_loss * (batch_idx - 1) + loss_val) / batch_idx
            avg_l1 = l1_val if batch_idx == 0 else (avg_l1 * (batch_idx - 1) + l1_val) / batch_idx
            avg_lpips = lpips_val if batch_idx == 0 else (avg_lpips * (batch_idx - 1) + lpips_val) / batch_idx
            avg_pnsr = float(psnr_torch(output_clean, img1_orig)) if batch_idx == 0 else (avg_pnsr * (batch_idx - 1) + float(psnr_torch(output_clean, img1_orig))) / batch_idx

            cur_comb[cur_mask] = avg_loss
            cur_l1[cur_mask] = avg_l1
            cur_lpips[cur_mask] = avg_lpips

        loss.backward()
        torch.nn.utils.clip_grad_norm_(flownet.parameters(), 1)
        optimizer_flownet.step()

        if isinstance(scheduler_flownet, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if args.plateau_interval > 0 and step % args.plateau_interval == 0:
                window = min(sched_count, cur_size)
                metric = float(np.mean(sched_buf[:window])) if window < cur_size else float(np.mean(sched_buf))
                prev_lr = optimizer_flownet.param_groups[0]['lr']
                scheduler_flownet.step(metric)
                new_lr = optimizer_flownet.param_groups[0]['lr']
                if is_master and new_lr != prev_lr:
                    sys.stderr.write(f'\nReduceLROnPlateau: rolling avg {metric:.6f} '
                                     f'-> lr {prev_lr:.2e} to {new_lr:.2e}\n')
                    sys.stderr.flush()
        else:
            try:
                scheduler_flownet.step()
            except Exception as e:
                current_lr = float(optimizer_flownet.param_groups[0]["lr"])
                if is_master:
                    print(f'switching to CyclicLR scheduler with base {current_lr * pulse_dive} and max {current_lr}\n\n')
                scheduler_flownet = torch.optim.lr_scheduler.CyclicLR(
                    optimizer_flownet,
                    base_lr=current_lr * pulse_dive,
                    max_lr=current_lr,
                    step_size_up=pulse_period,
                    cycle_momentum=False,
                    mode='exp_range',
                    scale_fn=sinusoidal_scale_fn,
                    scale_mode='cycle'
                )
            if args.cyclic != -1 and step % args.cyclic == 1:
                scheduler_flownet = torch.optim.lr_scheduler.CyclicLR(
                    optimizer_flownet,
                    base_lr=lr * pulse_dive,
                    max_lr=lr,
                    step_size_up=args.cyclic,
                    mode='exp_range',
                    cycle_momentum=False,
                    scale_fn=sinusoidal_scale_fn,
                    scale_mode='cycle'
                )

        train_time = time.time() - time_stamp
        time_stamp = time.time()

        # ---------------------------------------------------------------------
        # Checkpoint, preview, logging — rank 0 only
        # ---------------------------------------------------------------------
        if is_master:

            target_model = flownet.module if world_size > 1 else flownet
            current_state_dict['step'] = int(step)
            current_state_dict['epoch'] = int(epoch)
            current_state_dict['start_timestamp'] = start_timestamp
            current_state_dict['lr'] = optimizer_flownet.param_groups[0]['lr']
            current_state_dict['model_info'] = model_info
            current_state_dict['flownet_state_dict'] = target_model.state_dict()
            current_state_dict['optimizer_flownet_state_dict'] = optimizer_flownet.state_dict()
            current_state_dict['trained_model_path'] = trained_model_path

            if step % args.save == 1:
                write_model_state_queue.put(deepcopy(current_state_dict))

            if step % args.preview == 1:
                rgb_source1 = cct2cg(img0_orig)
                rgb_source2 = cct2cg(img2_orig)
                rgb_target = cct2cg(img1_orig)
                rgb_output = cct2cg(output_clean)
                rgb_output_diff = cct2cg(diff_matte.repeat_interleave(3, dim=1))

                preview_index += 1
                preview_index = preview_index if preview_index < 10 else 0

                write_image_queue.put({
                    'preview_folder': os.path.join(args.dataset_path, 'preview', os.path.splitext(os.path.basename(trained_model_path))[0]),
                    'preview_index': int(preview_index),
                    'sample_source1': rgb_source1[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_source2': rgb_source2[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_target': rgb_target[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output': rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output_diff': rgb_output_diff[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                })
                del rgb_source1, rgb_source2, rgb_target, rgb_output, rgb_output_diff

            current_desc['loss'] = loss_val
            current_desc['loss_l1'] = l1_val
            current_desc['lpips'] = lpips_val

            min_max_item = {
                'description': current_desc,
                'img0_orig': img0_orig.numpy(force=True).copy(),
                'img1_orig': img1_orig.numpy(force=True).copy(),
                'img2_orig': img2_orig.numpy(force=True).copy(),
                'output': output_clean.numpy(force=True).copy(),
            }

            try:
                max_values.add(loss_val, min_max_item)
                min_values.add(loss_val, min_max_item)
            except:
                pass

            if (args.preview_max > 0) and ((step + 1 % preview_maxmin_steps) == 1 or (sample_idx + 1) == len(dataset)):
                max_preview_folder = os.path.join(args.dataset_path, 'preview', os.path.splitext(os.path.basename(trained_model_path))[0], 'max')
                if not os.path.isdir(max_preview_folder):
                    os.makedirs(max_preview_folder)
                max_loss_values = max_values.get_values()
                index = 0
                item = None
                for index, item in enumerate(max_loss_values):
                    item_data = item[1]
                    n, c, h, w = item_data['img0_orig'].shape
                    for b_indx in range(n):
                        write_eval_image_queue.put({
                            'preview_folder': max_preview_folder,
                            'sample_source1': item_data['img0_orig'][b_indx].transpose(1, 2, 0),
                            'sample_source1_name': f'{index:04}_{b_indx:02}_A_incomng.exr',
                            'sample_source2': item_data['img2_orig'][b_indx].transpose(1, 2, 0),
                            'sample_source2_name': f'{index:04}_{b_indx:02}_B_outgoing.exr',
                            'sample_target': item_data['img1_orig'][b_indx].transpose(1, 2, 0),
                            'sample_target_name': f'{index:04}_{b_indx:02}_C_target.exr',
                            'sample_output': item_data['output'][b_indx].transpose(1, 2, 0),
                            'sample_output_name': f'{index:04}_{b_indx:02}_D_output.exr',
                        })
                        json_filename = os.path.join(max_preview_folder, f'{index:04}_{b_indx:02}.json')
                        with open(json_filename, 'w', encoding='utf-8') as json_file:
                            json.dump(item_data['description'], json_file, indent=4, ensure_ascii=False)
                del index, item

            if (args.preview_min > 0) and ((step + 1 % preview_maxmin_steps) == 1 or (sample_idx + 1) == len(dataset)):
                min_preview_folder = os.path.join(args.dataset_path, 'preview', os.path.splitext(os.path.basename(trained_model_path))[0], 'min')
                if not os.path.isdir(min_preview_folder):
                    os.makedirs(min_preview_folder)
                min_loss_values = min_values.get_values()
                index = 0
                item = None
                for index, item in enumerate(min_loss_values):
                    item_data = item[1]
                    n, c, h, w = item_data['img0_orig'].shape
                    for b_indx in range(n):
                        write_eval_image_queue.put({
                            'preview_folder': min_preview_folder,
                            'sample_source1': item_data['img0_orig'][b_indx].transpose(1, 2, 0),
                            'sample_source1_name': f'{index:04}_{b_indx:02}_A_incomng.exr',
                            'sample_source2': item_data['img2_orig'][b_indx].transpose(1, 2, 0),
                            'sample_source2_name': f'{index:04}_{b_indx:02}_B_outgoing.exr',
                            'sample_target': item_data['img1_orig'][b_indx].transpose(1, 2, 0),
                            'sample_target_name': f'{index:04}_{b_indx:02}_C_target.exr',
                            'sample_output': item_data['output'][b_indx].transpose(1, 2, 0),
                            'sample_output_name': f'{index:04}_{b_indx:02}_D_output.exr',
                        })
                        json_filename = os.path.join(min_preview_folder, f'{index:04}_{b_indx:02}.json')
                        with open(json_filename, 'w', encoding='utf-8') as json_file:
                            json.dump(item_data['description'], json_file, indent=4, ensure_ascii=False)
                del index, item

            data_time_str = str(f'{data_time:.2f}')
            model_time_str = str(f'{model_time:.2f}')
            train_time_str = str(f'{train_time:.2f}')
            data_time2_str = str(f'{data_time2:.2f}')

            epoch_time = time.time() - start_timestamp
            days = int(epoch_time // (24 * 3600))
            hours = int((epoch_time % (24 * 3600)) // 3600)
            minutes = int((epoch_time % (3600)) // 60)

            clear_lines(2)
            print(f'\r[Epoch {(epoch + 1):04} Step {step} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time_str}+{model_time_str}+{train_time_str}+{data_time2_str}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}')
            if len(dataset) > cur_size:
                print(f'\r[{cur_size//1000}K Average] L1: {np.mean(cur_l1):.6f} LPIPS: {np.mean(cur_lpips):.4f} Combined: {np.mean(cur_comb):.6f}')
                if (step + 1) % cur_size == 1:
                    csv_file_name = f'{os.path.splitext(trained_model_path)[0]}_train_loss_10K.csv'
                    if not os.path.isfile(csv_file_name):
                        create_csv_file(csv_file_name, ['Epoch', 'Step', 'L1', 'LPIPS', 'Combined'])
                    else:
                        for row in [{'Epoch': epoch, 'Step': step, 'L1': np.mean(cur_l1), 'LPIPS': np.mean(cur_lpips), 'Combined': np.mean(cur_comb)}]:
                            append_row_to_csv(csv_file_name, row)
                    clear_lines(2)
                    print(f'\r[Step {step + 1}] Avg L1: {np.mean(cur_l1):.6f} LPIPS: {np.mean(cur_lpips):.4f} Combined: {np.mean(cur_comb):.6f}')
                    print('\n')
            else:
                print(f'\r[Epoch] Min L1: {min_l1:.6f} Avg L1: {avg_l1:.6f} Max L1: {max_l1:.6f} LPIPS: {avg_lpips:.4f} Combined: {avg_loss:.6f}')

        # ---------------------------------------------------------------------
        # Epoch boundary — detect on rank 0, broadcast to all ranks
        # ---------------------------------------------------------------------
        if world_size > 1:
            epoch_done = torch.tensor(1 if (sample_idx + 1) >= len(dataset) else 0, device=device)
            torch.distributed.broadcast(epoch_done, src=0)
            epoch_done = epoch_done.item()
        else:
            epoch_done = 1 if (sample_idx + 1) >= len(dataset) else 0

        if epoch_done:
            if is_master:
                write_model_state_queue.put(deepcopy(current_state_dict))

                epoch_time = time.time() - start_timestamp
                days = int(epoch_time // (24 * 3600))
                hours = int((epoch_time % (24 * 3600)) // 3600)
                minutes = int((epoch_time % (3600)) // 60)

                clear_lines(2)
                print(f'\rEpoch [{epoch + 1} (Step {step:11} - {days:02}d {hours:02}:{minutes:02}], Min L1: {min_l1:.6f} Avg L1: {avg_l1:.6f} Max L1: {max_l1:.6f} Avg LPIPS: {avg_lpips:.4f} Combined: {avg_loss:.6f}')
                print('\n')

                for row in [{'Epoch': epoch, 'Step': step, 'Min': min_l1, 'Avg': avg_l1, 'Max': max_l1, 'PSNR': avg_pnsr, 'LPIPS': avg_lpips}]:
                    append_row_to_csv(f'{os.path.splitext(trained_model_path)[0]}.csv', row)

                if args.eval == 0 and args.plateau_interval <= 0:
                    if isinstance(scheduler_flownet, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler_flownet.step(avg_loss)

                min_l1 = float(sys.float_info.max)
                max_l1 = 0
                avg_l1 = 0
                avg_pnsr = 0
                avg_lpips = 0
                avg_loss = 0
                max_values.reset()
                min_values.reset()

            epoch = epoch + 1
            batch_idx = 0

            # All ranks reshuffle with same seed so sample->rank assignment stays consistent
            if not args.sequential:
                random.seed(epoch)
                dataset.reshuffle()
                random.seed()

            if world_size > 1:
                torch.distributed.barrier()  # sync before next epoch starts

        # ---------------------------------------------------------------------
        # Evaluation block — rank 0 only
        # ---------------------------------------------------------------------
        if is_master and (((args.eval > 0) and (step % args.eval) == 1) or (epoch == args.epochs)):
            if not args.eval_first:
                if step == 1:
                    batch_idx = batch_idx + 1
                    step = step + 1
                    continue

            preview_folder = os.path.join(args.dataset_path, 'preview')

            try:
                prev_eval_folder
            except:
                prev_eval_folder = None

            eval_folder = os.path.join(
                preview_folder, 'eval',
                os.path.splitext(os.path.basename(trained_model_path))[0],
                f'Step_{step:09}'
            )
            if not os.path.isdir(eval_folder):
                os.makedirs(eval_folder)

            if eval_dataset is not None:
                descriptions = list(eval_dataset.initial_train_descriptions)
            else:
                # one random window per train sequence (same window plan as the feed)
                rng = random.Random(args.eval_seed)
                eval_seqs = [s for s in train_m.sequences
                             if s.num_windows(bidirectional, window_mode) > 0 and s.height is not None]
                descriptions = []
                for s in eval_seqs:
                    k = rng.randrange(s.num_windows(bidirectional, window_mode))
                    w = s.window_at(k, bidirectional, window_mode)
                    descriptions.append({
                        'h': s.height, 'w': s.width,
                        'start': s.path_at(w.start),
                        'gt': s.path_at(w.gt),
                        'end': s.path_at(w.end),
                        'ratio': w.ratio,
                    })

            if args.eval_samples > 0:
                rng = random.Random(args.eval_seed)
                descriptions = rng.sample(descriptions, min(args.eval_samples, len(descriptions)))

            def read_eval_images(read_eval_image_queue, descriptions):
                for ev_item_index, description in enumerate(descriptions):
                    try:
                        desc_data = dict(description)
                        desc_data['eval_img0'] = read_image_file(description['start'])['image_data']
                        desc_data['eval_img1'] = read_image_file(description['gt'])['image_data']
                        desc_data['eval_img2'] = read_image_file(description['end'])['image_data']
                        desc_data['ev_item_index'] = ev_item_index
                        read_eval_image_queue.put(desc_data)
                        del desc_data
                    except Exception as e:
                        pprint(f'\nerror while reading eval images: {e}\n{description}\n\n')
                read_eval_image_queue.put(None)

            read_eval_image_queue = queue.Queue(maxsize=4)
            read_eval_thread = threading.Thread(target=read_eval_images, args=(read_eval_image_queue, descriptions))
            read_eval_thread.daemon = True
            read_eval_thread.start()

            eval_loss = []
            eval_psnr = []
            eval_lpips = []

            # eval uses flownet.module to avoid DDP overhead during inference
            original_state_dict = deepcopy((flownet.module if world_size > 1 else flownet).state_dict())

            if use_cuda:
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()

            flownet.cpu()

            if use_cuda:
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            evalnet = Flownet().get_model()().to(device)
            evalnet.load_state_dict(original_state_dict)
            for param in evalnet.parameters():
                param.requires_grad = False

            if args.eval_half:
                evalnet.half()

            evalnet.eval()
            with torch.no_grad():
                description = read_eval_image_queue.get()
                while description is not None:
                    ev_item_index = description['ev_item_index']

                    if eval_loss:
                        eval_loss_min = min(eval_loss)
                        eval_loss_max = max(eval_loss)
                        eval_loss_avg = float(np.array(eval_loss).mean())
                    else:
                        eval_loss_min = eval_loss_max = eval_loss_avg = -1
                    eval_psnr_mean = float(np.array(eval_psnr).mean()) if eval_psnr else -1
                    eval_lpips_mean = float(np.array(eval_lpips).mean()) if eval_lpips else -1

                    epoch_time = time.time() - start_timestamp
                    days = int(epoch_time // (24 * 3600))
                    hours = int((epoch_time % (24 * 3600)) // 3600)
                    minutes = int((epoch_time % (3600)) // 60)

                    clear_lines(1)
                    print(f'\rEvaluating {ev_item_index} of {len(descriptions)}: Min: {eval_loss_min:.6f} Avg: {eval_loss_avg:.6f}, Max: {eval_loss_max:.6f} LPIPS: {eval_lpips_mean:.4f} PSNR: {eval_psnr_mean:.4f}')

                    try:
                        eval_img0 = torch.from_numpy(description['eval_img0']).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                        eval_img1 = torch.from_numpy(description['eval_img1']).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                        eval_img2 = torch.from_numpy(description['eval_img2']).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                        eval_ratio = description['ratio']

                        eval_img0_orig = eval_img0.clone()
                        eval_img2_orig = eval_img2.clone()

                        # working space is ACEScct, same as training
                        if args.ap0:
                            eval_img0 = ap02cct(eval_img0)
                            eval_img1 = ap02cct(eval_img1)
                            eval_img2 = ap02cct(eval_img2)
                        else:
                            eval_img0 = ap12cct(eval_img0)
                            eval_img1 = ap12cct(eval_img1)
                            eval_img2 = ap12cct(eval_img2)

                        if args.eval_half:
                            eval_img0 = eval_img0.half()
                            eval_img2 = eval_img2.half()

                        result = evalnet(eval_img0, eval_img2, eval_ratio, iterations=args.iterations)

                        eval_flow_list = result['flow_list']
                        eval_mask_list = result['mask_list']
                        eval_conf_list = result['conf_list']

                        if args.eval_half:
                            eval_flow_list[-1] = eval_flow_list[-1].float()
                            eval_mask_list[-1] = eval_mask_list[-1].float()

                        eval_result = (
                            warp(eval_img0_orig, eval_flow_list[-1][:, :2, :, :]) * eval_mask_list[-1] +
                            warp(eval_img2_orig, eval_flow_list[-1][:, 2:4, :, :]) * (1 - eval_mask_list[-1])
                        )

                        if torch.isnan(eval_img0_orig).any() or torch.isnan(eval_img2_orig).any() or torch.isnan(eval_result).any() or torch.isnan(eval_img1).any():
                            print(f'eval: NaN detected: {description["start"]}\n\n')
                            description = read_eval_image_queue.get()
                            continue

                        eval_loss_l1 = criterion_l1(eval_result, eval_img1)
                        eval_loss.append(float(eval_loss_l1.item()))
                        eval_psnr.append(float(psnr_torch(eval_result, eval_img1)))
                        eval_loss_LPIPS = loss_fn_alex(eval_result * 2 - 1, eval_img1 * 2 - 1)
                        eval_lpips.append(float(torch.mean(eval_loss_LPIPS).item()))

                        eval_rgb_output_mask = eval_mask_list[-1].repeat_interleave(3, dim=1)
                        eval_rgb_conf = eval_conf_list[-1].repeat_interleave(3, dim=1)
                        eval_rgb_diff = diffmatte(eval_result, eval_img1).repeat_interleave(3, dim=1)

                        if args.eval_save_imgs:
                            write_eval_image_queue.put({
                                'preview_folder': eval_folder,
                                'sample_source1': cct2cg(eval_img0_orig)[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_source1_name': f'{ev_item_index:08}_A_incomng.exr',
                                'sample_source2': cct2cg(eval_img2_orig)[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_source2_name': f'{ev_item_index:08}_B_outgoing.exr',
                                'sample_target': cct2cg(eval_img1)[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_target_name': f'{ev_item_index:08}_C_target.exr',
                                'sample_output': cct2cg(eval_result)[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_output_name': f'{ev_item_index:08}_D_output.exr',
                                'sample_output_diff': cct2cg(eval_rgb_diff)[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_output_diff_name': f'{ev_item_index:08}_E_diff.exr',
                                'sample_output_conf': cct2cg(eval_rgb_conf)[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_output_conf_name': f'{ev_item_index:08}_F_conf.exr',
                                'sample_output_mask': cct2cg(eval_rgb_output_mask)[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_output_mask_name': f'{ev_item_index:08}_G_mask.exr'
                            })

                        if use_cuda:
                            torch.cuda.synchronize()
                        elif torch.backends.mps.is_available():
                            torch.mps.synchronize()

                        del eval_img0, eval_img1, eval_img2, eval_img0_orig, eval_img2_orig
                        del eval_flow_list, eval_mask_list, eval_conf_list
                        del result, eval_result, eval_rgb_output_mask, eval_rgb_diff, eval_rgb_conf
                        del description['eval_img0'], description['eval_img1'], description['eval_img2']

                        if use_cuda:
                            torch.cuda.empty_cache()
                        elif torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                    except Exception as e:
                        try:
                            del description['eval_img0']
                            del description['eval_img1']
                            del description['eval_img2']
                        except Exception:
                            pass
                        print(f'\n\nerror while evaluating: {e}\n{description}\n{traceback.format_exc()}\n\n')
                    description = read_eval_image_queue.get()

            # the loop above only refreshes these stats at the TOP of the next
            # iteration, so after the last sample they are one step stale (-1 for
            # the first/only sample) — recompute before logging / best-model check
            if eval_loss:
                eval_loss_min = min(eval_loss)
                eval_loss_max = max(eval_loss)
                eval_loss_avg = float(np.array(eval_loss).mean())
                eval_psnr_mean = float(np.array(eval_psnr).mean())
                eval_lpips_mean = float(np.array(eval_lpips).mean())

            for eval_row in [{'Epoch': epoch, 'Step': step, 'Min': eval_loss_min, 'Avg': eval_loss_avg, 'Max': eval_loss_max, 'PSNR': eval_psnr_mean, 'LPIPS': eval_lpips_mean}]:
                append_row_to_csv(f'{os.path.splitext(trained_model_path)[0]}.eval.csv', eval_row)

            clear_lines(2)
            print(f'\r[Epoch {(epoch + 1):04} Step {step:08} - {days:02}d {hours:02}:{minutes:02}], Eval Min: {eval_loss_min:.6f} Avg: {eval_loss_avg:.6f}, Max: {eval_loss_max:.6f}, [PSNR] {eval_psnr_mean:.4f}, [LPIPS] {eval_lpips_mean:.4f}')
            print('\n')

            # keep the best-scoring checkpoint alongside the periodic one
            if eval_loss:
                eval_loss_combined = float(eval_loss_avg + 2e-1 * eval_lpips_mean)
                if eval_loss_combined < best_eval_loss:
                    best_eval_loss = eval_loss_combined
                    best_state_dict = deepcopy(current_state_dict)
                    broot, bext = os.path.splitext(trained_model_path)
                    best_state_dict['trained_model_path'] = f"{broot}.best{bext}"
                    write_model_state_queue.put(best_state_dict)

            if not args.eval_keep_all:
                if prev_eval_folder and os.path.isdir(prev_eval_folder):
                    threading.Thread(target=lambda: os.system(f'rm -rf {os.path.abspath(prev_eval_folder)}')).start()
            prev_eval_folder = eval_folder

            del evalnet
            if use_cuda:
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            flownet.to(device)
            flownet.train()

            read_eval_thread.join()
            del read_eval_image_queue

            if args.plateau_interval <= 0 and isinstance(
                    scheduler_flownet, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_flownet.step(eval_loss_avg)

        # End of evaluation block

        batch_idx = batch_idx + 1
        step = step + 1

        # (per-rank pool feeds the next batch; nothing to prefetch here)

        data_time2 = time.time() - time_stamp

        if epoch == args.epochs:
            if is_master:
                # Clean completion: let the background writers finish their
                # pending saves, then os._exit. Plain sys.exit() runs the
                # interpreter finalization, which in this environment races the
                # still-alive daemon threads (torch.save / EXR writes) and
                # aborts with std::terminate — losing the final checkpoints.
                # (The SIGINT path already uses os._exit for the same reason.)
                _deadline = time.time() + 300
                while time.time() < _deadline and not (
                        write_model_state_queue.empty()
                        and write_image_queue.empty()
                        and write_eval_image_queue.empty()):
                    time.sleep(0.2)
                write_shutdown.set()
                for _t in (write_model_state_thread, write_thread, write_eval_thread):
                    _t.join(timeout=120)
                sys.stdout.flush()
                sys.stderr.flush()
            if world_size > 1:
                torch.distributed.destroy_process_group()
            os._exit(0)


if __name__ == "__main__":
    if '--all_gpus' in sys.argv:
        world_size = torch.cuda.device_count()
        try:
            torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
        except KeyboardInterrupt:
            pass
    else:
        main(0, 1)
