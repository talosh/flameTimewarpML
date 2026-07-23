import os
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

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
except:
    python_executable_path = sys.executable
    if '.miniconda' in python_executable_path:
        print ('Unable to import PyTorch')
        print (f'Using "{python_executable_path}" as python interpreter')
        sys.exit()
    else:
        # make Flame happy on hooks scan
        class torch(object):
            class nn(object):
                class Module(object):
                    pass
                class Conv2d(object):
                    pass

try:
    import numpy as np
except:
    python_executable_path = sys.executable
    if '.miniconda' in python_executable_path:
        print (f'Using "{python_executable_path}" as python interpreter')
        print ('Unable to import Numpy')
        sys.exit()

try:
    import OpenImageIO as oiio
except:
    python_executable_path = sys.executable
    if '.miniconda' in python_executable_path:
        print ('Unable to import OpenImageIO')
        print (f'Using "{python_executable_path}" as python interpreter')
        sys.exit()

exit_event = threading.Event()  # For threads
process_exit_event = torch.multiprocessing.Event()  # For processes

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

            '''
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

            # '''
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
            # '''

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

def normalize(x):
    x = x * 2 - 1
    scale = torch.tanh(torch.tensor(1.0))
    x = torch.where(
        (x >= -1) & (x <= 1), scale * x,
        torch.tanh(x)
    )
    x = (x + 1) / 2
    return x

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

def normalize_numpy(image_array_torch):
    import numpy as np

    image_array = image_array_torch.clone().cpu().detach().numpy()

    def custom_bend(x):
        linear_part = x
        exp_bend = np.sign(x) * np.power(np.abs(x), 1 / 4)
        return np.where(x > 1, exp_bend, np.where(x < -1, exp_bend, linear_part))

    # Transfer (0.0 - 1.0) onto (-1.0 - 1.0) for tanh
    image_array = (image_array * 2) - 1
    # Bend values below -1.0 and above 1.0 exponentially so they are not larger than (-4.0 - 4.0)
    image_array = custom_bend(image_array)
    # Bend everything to fit -1.0 - 1.0 with hyperbolic tangent
    image_array = np.tanh(image_array)
    # Move it to 0.0 - 1.0 range
    image_array = (image_array + 1) / 2

    return torch.from_numpy(image_array.copy())

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

def restore_normalized_values_numpy(image_array_torch):
    import numpy as np

    image_array = image_array_torch.clone().cpu().detach().numpy()

    def custom_de_bend(x):
        linear_part = x
        de_bend = np.sign(x) * np.power(np.abs(x), 4)
        return np.where(x > 1, de_bend, np.where(x < -1, de_bend, linear_part))

    epsilon = 4e-8
    # Clamp image before arctanh
    image_array = np.clip((image_array * 2) - 1, -1.0 + epsilon, 1.0 - epsilon)
    # Restore values from tanh s-curve
    image_array = np.arctanh(image_array)
    # Restore custom bended values
    image_array = custom_de_bend(image_array)
    # Move it to 0.0 - 1.0 range
    image_array = (image_array + 1.0) / 2.0

    return torch.from_numpy(image_array.copy())

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

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

def tenflow(tenFlow):
    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenFlow.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenFlow.shape[2] - 1.0) / 2.0) ], 1)
    return tenFlow

def warp_tenflow(tenInput, tenFlow):
    backwarp_tenGrid = {}
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device = tenInput.device, dtype = tenInput.dtype)
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bicubic', padding_mode='border', align_corners=True)

def id_flow(tenInput):
    tenHorizontal = torch.linspace(-1.0, 1.0, tenInput.shape[3]).view(1, 1, 1, tenInput.shape[3]).expand(tenInput.shape[0], -1, tenInput.shape[2], -1)
    tenVertical = torch.linspace(-1.0, 1.0, tenInput.shape[2]).view(1, 1, tenInput.shape[2], 1).expand(tenInput.shape[0], -1, -1, tenInput.shape[3])
    return torch.cat([ tenHorizontal, tenVertical ], 1).to(device = tenInput.device, dtype = tenInput.dtype)

def split_to_yuv(rgb_tensor):
    r_tensor, g_tensor, b_tensor = rgb_tensor[:, 0:1, :, :], rgb_tensor[:, 1:2, :, :], rgb_tensor[:, 2:3, :, :]
    y_tensor = 0.299 * r_tensor + 0.587 * g_tensor + 0.114 * b_tensor
    u_tensor = -0.147 * r_tensor - 0.289 * g_tensor + 0.436 * b_tensor
    v_tensor = 0.615 * r_tensor - 0.515 * g_tensor - 0.100 * b_tensor

    return y_tensor, u_tensor, v_tensor

def gamma_up(img, gamma = 1.18):
    return torch.sign(img) * torch.pow(torch.abs(img), 1 / gamma )

def blurit(img, interations = 16):
    def gaussian_kernel(size, sigma):
        """Creates a 2D Gaussian kernel using the specified size and sigma."""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return torch.tensor(kernel / np.sum(kernel))

    class GaussianBlur(torch.nn.Module):
        def __init__(self, kernel_size, sigma):
            super(GaussianBlur, self).__init__()
            # Create a Gaussian kernel
            self.kernel = gaussian_kernel(kernel_size, sigma)
            self.kernel = self.kernel.view(1, 1, kernel_size, kernel_size)
            
            # Set up the convolutional layer, without changing the number of channels
            self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2, groups=1, bias=False, padding_mode='reflect')
            
            # Initialize the convolutional layer with the Gaussian kernel
            self.conv.weight.data = self.kernel
            self.conv.weight.requires_grad = False  # Freeze the weights

        def forward(self, x):
            self.kernel.to(device = x.device, dtype = x.dtype)
            return self.conv(x)
    
    gaussian_blur = GaussianBlur(9, 1.0).to(device = img.device, dtype = img.dtype)
    blurred_img = img
    n, c, h, w = img.shape
    for _ in range(interations):
        channel_tensors = [gaussian_blur(blurred_img[:, i:i+1, :, :]) for i in range(c)]
        blurred_img = torch.cat(channel_tensors, dim=1)
        '''
        r_tensor, g_tensor, b_tensor = blurred_img[:, 0:1, :, :], blurred_img[:, 1:2, :, :], blurred_img[:, 2:3, :, :]
        r_blurred = gaussian_blur(r_tensor)
        g_blurred = gaussian_blur(g_tensor)
        b_blurred = gaussian_blur(b_tensor)
        blurred_img = torch.cat((r_blurred, g_blurred, b_blurred), dim=1)
        '''
    return blurred_img

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

def find_and_import_model(models_dir='models', base_name=None, model_name=None, model_file=None):
    """
    Dynamically imports the latest version of a model based on the base name,
    or a specific model if the model name/version is given, and returns the Model
    object named after the base model name.

    :param models_dir: Relative path to the models directory.
    :param base_name: Base name of the model to search for.
    :param model_name: Specific name/version of the model (optional).
    :return: Imported Model object or None if not found.
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

    # Filter files based on base_name or model_name
    if model_name:
        # Look for a specific model version
        filtered_files = [f for f in files if f == f"{model_name}.py"]
    else:
        # Find all versions of the model and select the latest one
        # regex_pattern = fr"{base_name}_v(\d+)\.py"
        # versions = [(f, int(m.group(1))) for f in files if (m := re.match(regex_pattern, f))]
        versions = [f for f in files if f.endswith('.py')]
        if versions:
            # Sort by version number (second item in tuple) and select the latest one
            # latest_version_file = sorted(versions, key=lambda x: x[1], reverse=True)[0][0]
            latest_version_file = sorted(versions, reverse=True)[0]
            filtered_files = [latest_version_file]

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

def closest_divisible(x, padding = 64):
    """
    Find the closest integer divisible by 64 to the given number x.

    Args:
    x (int or float): The number to find the closest divisible value for.

    Returns:
    int: Closest number divisible by 64.
    """
    # Round down to the nearest multiple of 64
    lower = (x // padding) * padding
    
    # Round up to the nearest multiple of 64
    upper = lower + padding

    # Check which one is closer to x
    if x - lower > upper - x:
        return upper
    else:
        return lower

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

def gaussian(window_size, sigma):
    from math import exp
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, stride=None):
    mu1 = torch.nn.functional.conv2d(img1, window, padding = (window_size-1)//2, groups = channel, stride=stride)
    mu2 = torch.nn.functional.conv2d(img2, window, padding = (window_size-1)//2, groups = channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = torch.nn.functional.conv2d(img1*img1, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 3, size_average = True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)

class MeanShift(torch.nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class VGGPerceptualLoss(torch.nn.Module):

    def __init__(self):
        from torchvision import models
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:                
                loss += weights[k] * (X - Y.detach()).abs().mean()
                k += 1
        return loss

def convert_to_data_parallel(param):
    return {
        f'module.{k}': v
        for k, v in param.items()
    }

def convert_from_data_parallel(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }

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

class LossStats:
    def __init__(self):
        self.epoch_l1_loss = []
        self.psnr_list = []
        self.lpips_list = []
        self.l1 = 0
        self.l1_last10k = 0
        self.l1_min = 0
        self.l1_min_last10k = 0
        self.l1_max = 0
        self.l1_max_last10k = 0
        self.psnr = 0
        self.psnr_last10k = 0
        self.lpips = 0
        self.lpips_last10k = 0

        stats_thread = threading.Thread(target=self.calclulate_stats, args=())
        stats_thread.daemon = True
        stats_thread.start()

    def calclulate_stats(self):
        while True:
            try:
                if len(self.epoch_l1_loss) < 9999:
                    self.l1_last10k = float(np.mean(moving_average(self.epoch_l1_loss, 9)))
                    self.l1_min_last10k = min(self.epoch_l1_loss)
                    self.l1_max_last10k = max(self.epoch_l1_loss)
                    self.lpips_last10k = float(np.array(self.lpips_list).mean())
                else:
                    self.l1_last10k = np.mean(moving_average(self.epoch_l1_loss[-9999:], 9))
                    self.l1_min_last10k = min(self.epoch_l1_loss[-9999:])
                    self.l1_max_last10k = max(self.epoch_l1_loss[-9999:])
                    self.lpips_last10k = float(np.array(self.lpips_list[-9999:]).mean())
                self.l1 = float(np.mean(moving_average(self.epoch_l1_loss, 9)))
                self.l1_min = min(self.epoch_l1_loss)
                self.l1_max = max(self.epoch_l1_loss)
                self.lpips = float(np.array(self.lpips_list).mean())
            except:
                pass
            time.sleep(0.1)
            
    def add_l1(self, val):
        self.epoch_l1_loss.append(val)

    def add_pnsr(self, val):
        self.psnr_list.append(val)

    def add_lpips(self, val):
        self.lpips_list.append(val)

    def reset(self):
        self.epoch_l1_loss = []
        self.psnr_list = []
        self.lpips_list = []

    def __len__(self):
        return len(self.epoch_l1_loss)

def sinusoidal_scale_fn(x):
    import math
    # x is a fraction of the cycle's progress (0 to 1)
    return 0.5 * (1 + math.sin(math.pi * (x - 0.5)))

def centered_highpass_filter(rgb_image, gamma=1.8):
    """
    Apply a centered high-pass filter to an RGB image tensor.
    
    Args:
        rgb_image (torch.Tensor): Input tensor of shape (n, 3, h, w).
        cutoff_ratio (float): Proportion of low frequencies to block. Typical values are between 0.05 and 0.2.
    
    Returns:
        torch.Tensor: High-pass filtered image of the same shape as the input.
    """
    '''
    n, c, h, w = rgb_image.shape
    
    # Step 1: Apply Fourier Transform along spatial dimensions
    freq_image = torch.fft.fft2(rgb_image, dim=(-2, -1))
    freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))  # Shift the zero-frequency component to the center

    # Step 2: Create a high-pass filter mask
    center_x, center_y = h // 2, w // 2
    x = torch.arange(h).view(-1, 1).repeat(1, w)
    y = torch.arange(w).repeat(h, 1)
    distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2).sqrt()
    radius = min(h, w) * cutoff_ratio
    highpass_mask = (distance_from_center > radius).float()
    
    # Step 3: Apply the mask to each color channel in the frequency domain
    highpass_mask = highpass_mask.to(freq_image.device)  # Ensure mask is on the same device
    freq_image_filtered = freq_image * highpass_mask.unsqueeze(0).unsqueeze(1)

    # Step 4: Inverse Fourier Transform to return to spatial domain
    freq_image_filtered = torch.fft.ifftshift(freq_image_filtered, dim=(-2, -1))
    highpass_image = torch.fft.ifft2(freq_image_filtered, dim=(-2, -1)).real  # Take the real part only
    
    return highpass_image
    '''

    """
    Apply a centered high-pass filter to an RGB image tensor with a fixed cutoff distance.
    
    Args:
        rgb_image (torch.Tensor): Input tensor of shape (n, 3, h, w).
        cutoff_distance (int): Absolute distance in pixels for low frequencies to be blocked.
    
    Returns:
        torch.Tensor: High-pass filtered image of the same shape as the input.
    """
    '''
    n, c, h, w = rgb_image.shape
    
    # Step 1: Apply Fourier Transform along spatial dimensions
    freq_image = torch.fft.fft2(rgb_image, dim=(-2, -1))
    freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))  # Shift the zero-frequency component to the center

    # Step 2: Create a high-pass filter mask
    center_x, center_y = h // 2, w // 2
    x = torch.arange(h).view(-1, 1).repeat(1, w)
    y = torch.arange(w).repeat(h, 1)
    distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2).sqrt()
    highpass_mask = (distance_from_center > cutoff_distance).float()
    
    # Step 3: Apply the mask to each color channel in the frequency domain
    highpass_mask = highpass_mask.to(freq_image.device)  # Ensure mask is on the same device
    freq_image_filtered = freq_image * highpass_mask.unsqueeze(0).unsqueeze(1)

    # Step 4: Inverse Fourier Transform to return to spatial domain
    freq_image_filtered = torch.fft.ifftshift(freq_image_filtered, dim=(-2, -1))
    highpass_image = torch.fft.ifft2(freq_image_filtered, dim=(-2, -1)).real  # Take the real part only
    
    return highpass_image
    '''
    """
    Apply an uncentered high-pass filter to an RGB image tensor.
    
    Args:
        rgb_image (torch.Tensor): Input tensor of shape (n, 3, h, w).
        cutoff_distance (int): Distance in pixels for low frequencies to be blocked.
    
    Returns:
        torch.Tensor: High-pass filtered image of the same shape as the input.
    """
    '''
    n, c, h, w = rgb_image.shape

    # Step 1: Apply Fourier Transform along spatial dimensions without shifting
    freq_image = torch.fft.fft2(rgb_image, dim=(-2, -1))

    # Step 2: Create a high-pass filter mask without centering
    # Calculate distance from (0, 0) for each pixel in the frequency domain
    x = torch.arange(h).view(-1, 1).repeat(1, w)
    y = torch.arange(w).repeat(h, 1)
    distance_from_corner = (x ** 2 + y ** 2).sqrt()
    highpass_mask = (distance_from_corner > cutoff_distance).float()

    # Step 3: Apply the mask to each color channel in the frequency domain
    highpass_mask = highpass_mask.to(freq_image.device)  # Ensure mask is on the same device
    freq_image_filtered = freq_image * highpass_mask.unsqueeze(0).unsqueeze(1)

    # Step 4: Inverse Fourier Transform to return to the spatial domain
    highpass_image = torch.fft.ifft2(freq_image_filtered, dim=(-2, -1)).real  # Take the real part only

    return highpass_image
    '''


    """
    Apply a scaling to each frequency component in an RGB image tensor,
    where the lowest frequency is scaled to 0 and the highest frequency is scaled to 1.
    
    Args:
        rgb_image (torch.Tensor): Input tensor of shape (n, 3, h, w).
    
    Returns:
        torch.Tensor: Frequency-scaled image of the same shape as the input.
    """
    
    padding = 32

    rgb_image = torch.nn.functional.pad(rgb_image, (padding, padding, padding, padding), mode='reflect')
    n, c, h, w = rgb_image.shape

    # Step 1: Apply Fourier Transform along spatial dimensions
    freq_image = torch.fft.fft2(rgb_image, dim=(-2, -1))
    freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))  # Shift the zero-frequency component to the center

    # Step 2: Calculate the distance of each frequency component from the center
    center_x, center_y = h // 2, w // 2
    x = torch.arange(h).view(-1, 1).repeat(1, w)
    y = torch.arange(w).repeat(h, 1)
    distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2).sqrt()
    
    # Normalize distance to the range [0, 1]
    max_distance = distance_from_center.max()
    distance_weight = distance_from_center / max_distance  # Now scaled from 0 (low freq) to 1 (high freq)
    distance_weight = distance_weight.to(freq_image.device)  # Ensure the weight is on the same device as the image
    distance_weight = distance_weight ** (1 / gamma)
    
    k = 11  # Controls the steepness of the curve
    x0 = 0.5  # Midpoint where the function crosses 0.5

    # Compute the S-like function using a sigmoid
    distance_weight = 1 / (1 + torch.exp(-k * (distance_weight - x0)))

    start=0.96
    end=1.0
    steepness=20
    mask = (distance_weight >= start) & (distance_weight <= end)
    distance_weight[mask] = 1 / (1 + torch.exp(steepness * (distance_weight[mask] - start) / (end - start)))
    # Step 3: Apply the distance weight to both real and imaginary parts of the frequency components
    freq_image_scaled = freq_image * distance_weight.unsqueeze(0).unsqueeze(1)

    # Step 4: Inverse Fourier Transform to return to spatial domain
    freq_image_scaled = torch.fft.ifftshift(freq_image_scaled, dim=(-2, -1))
    scaled_image = torch.fft.ifft2(freq_image_scaled, dim=(-2, -1)).real  # Take the real part only
    scaled_image = torch.max(scaled_image, dim=1, keepdim=True).values
    # scaled_image = scaled_image ** (1 / 1.8)

    return scaled_image[:, :, padding:-padding, padding:-padding]

def highpass(img):  
    def gauss_kernel(size=5, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel
    
    def conv_gauss(img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def normalize(tensor, min_val, max_val):
        return (tensor - min_val) / (max_val - min_val)

    gkernel = gauss_kernel()
    gkernel = gkernel.to(device=img.device, dtype=img.dtype)
    hp = img - conv_gauss(img, gkernel) + 0.5
    hp = torch.clamp(hp, 0.49, 0.51)
    hp = normalize(hp, hp.min(), hp.max())
    hp = torch.max(hp, dim=1, keepdim=True).values
    return hp

def overlay(base, blend):
    return torch.where(
        base < 0.5,
        2 * base * blend,
        1 - 2 * (1 - base) * (1 - blend)
    )

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernels — better noise rejection than simple differences
        kx = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]]) / 8.
        ky = kx.T
        # shape: (1, 1, 3, 3) — applied depthwise per channel
        self.register_buffer('kx', kx.unsqueeze(0).unsqueeze(0))
        self.register_buffer('ky', ky.unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        # process each channel independently
        loss = 0.0
        for c in range(pred.shape[1]):
            p = pred[:, c:c+1, :, :]
            t = target[:, c:c+1, :, :]

            gx_pred = F.conv2d(p, self.kx, padding=1)
            gy_pred = F.conv2d(p, self.ky, padding=1)
            gx_tgt  = F.conv2d(t, self.kx, padding=1)
            gy_tgt  = F.conv2d(t, self.ky, padding=1)

            loss += F.l1_loss(gx_pred, gx_tgt)
            loss += F.l1_loss(gy_pred, gy_tgt)

        return loss / pred.shape[1]

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

class LapLossNCC(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3, window=7,
                 ncc_levels=None, eps=1e-3):
        super(LapLossNCC, self).__init__()
        self.max_levels = max_levels
        self.window = window
        self.eps = eps
        # which pyramid levels get the invariant treatment.
        # coarser levels (higher index here, since pyr[0] is finest)
        # carry more of the low-frequency, illumination-gradient content —
        # default to invariant loss there, plain L1 on the finest levels
        # where sharpness/detail matters most and lighting rarely varies
        # at pixel scale.
        self.ncc_levels = ncc_levels if ncc_levels is not None else set(range(2, max_levels))
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

    def _local_stats(self, x, window, pad):
        mean = torch.nn.functional.avg_pool2d(x, window, stride=1, padding=pad,
                                            count_include_pad=False)
        var = torch.nn.functional.avg_pool2d(x * x, window, stride=1, padding=pad,
                                            count_include_pad=False) - mean * mean
        return mean, var.clamp(min=0)

    def invariant_band_loss(self, a, b):
        pad = self.window // 2
        _, var_a = self._local_stats(a, self.window, pad)
        _, var_b = self._local_stats(b, self.window, pad)

        # stability epsilon INSIDE sqrt -- this is the fix, not an optional extra.
        # var.clamp(min=0) only protects the forward value; sqrt's gradient at 0
        # is 1/(2*sqrt(x)) = inf, and Laplacian bands hit exact/near-zero variance
        # constantly in flat image regions.
        sqrt_eps = 1e-6
        std_a = torch.sqrt(var_a + sqrt_eps)
        std_b = torch.sqrt(var_b + sqrt_eps)

        cov = torch.nn.functional.avg_pool2d(a * b, self.window, stride=1, padding=pad,
                                            count_include_pad=False)

        band_scale = (std_a.mean() + std_b.mean()).detach() * 0.5 + 1e-6
        eps = self.eps * band_scale

        structure = cov / (std_a * std_b + eps)
        structure_loss = 1 - structure.mean()

        magnitude_loss = ((std_a - std_b).abs() / (std_a + std_b + eps)).mean()

        return structure_loss + magnitude_loss

    def forward(self, input, target):
        pyr_input = self.laplacian_pyramid(input, self.max_levels)
        pyr_target = self.laplacian_pyramid(target, self.max_levels)

        total = 0.0
        for i, (a, b) in enumerate(zip(pyr_input, pyr_target)):
            if i in self.ncc_levels:
                total = total + self.invariant_band_loss(a, b)
            else:
                total = total + torch.nn.functional.l1_loss(a, b)
        return total

'''
def hpass(img):
    def gauss_kernel(size=5, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel
    
    def conv_gauss(img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    gkernel = gauss_kernel()
    gkernel = gkernel.to(device=img.device, dtype=img.dtype)
    hp = img - conv_gauss(img, gkernel)
    return hp
'''

def hpass(img):
    def gauss_kernel(channels):
        kernel = torch.tensor([[1., 4., 6., 4., 1.],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        return kernel.repeat(channels, 1, 1, 1)

    def conv_gauss(img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        return torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])

    gkernel = gauss_kernel(img.shape[1])
    gkernel = gkernel.to(device=img.device, dtype=img.dtype)
    hp = img - conv_gauss(img, gkernel)

    # hp is zero-mean — normalize by per-image max abs deviation, then shift to [0, 1]
    scale = hp.abs().flatten(1).max(dim=1).values + 1e-6   # (B,)
    scale = scale.view(-1, 1, 1, 1)                         # broadcast over C, H, W
    hp = hp / scale * 0.5 + 0.5

    return hp

def blur(img):  
    def gauss_kernel(size=5, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel
    
    def conv_gauss(img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    gkernel = gauss_kernel()
    gkernel = gkernel.to(device=img.device, dtype=img.dtype)
    return conv_gauss(img, gkernel)

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

def normalize_min_max(tensor, min_val, max_val):
    src_dtype = tensor.dtype
    tensor = tensor.float()
    t_min = tensor.min()
    t_max = tensor.max()

    if t_min == t_max:
        return torch.full_like(tensor, (min_val + max_val) / 2.0)
    
    tensor = ((tensor - t_min) / (t_max - t_min)) * (max_val - min_val) + min_val
    tensor = tensor.to(dtype = src_dtype)
    return tensor

def to_freq(x):
    n, c, h, w = x.shape
    src_dtype = x.dtype
    x = x.float()
    x = torch.fft.fft2(x, dim=(-2, -1))
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.cat([x.real.unsqueeze(2), x.imag.unsqueeze(2)], dim=2).view(n, c * 2, h, w)
    x = x.to(dtype = src_dtype)
    return x

def to_freq_mph(x):
    n, c, h, w = x.shape
    src_dtype = x.dtype
    x = x.float()
    x = torch.fft.fft2(x, dim=(-2, -1))  # Perform 2D FFT
    magnitude = torch.abs(x)  # Compute magnitude
    phase = torch.angle(x)  # Compute phase
    x = torch.cat([magnitude.unsqueeze(2), phase.unsqueeze(2)], dim=2).view(n, c * 2, h, w)
    x = x.to(dtype=src_dtype)
    return x

def to_spat_mph(x):
    n, c, h, w = x.shape
    src_dtype = x.dtype
    x = x.float()
    x = x.view(n, c // 2, 2, h, w)
    magnitude = x[:, :, 0, :, :]
    phase = x[:, :, 1, :, :]
    phase = torch.clamp(phase, -torch.pi, torch.pi)
    x = torch.polar(magnitude, phase)  # Convert magnitude and phase back to complex
    x = torch.fft.ifft2(x, dim=(-2, -1)).real  # Perform inverse FFT
    x = x.to(dtype=src_dtype)
    return x

class MultiScaleLogMagSpectralLoss(torch.nn.Module):
    def __init__(
        self, 
        scales=(1, 2, 4), 
        eps: float = 1e-6, 
        reduction: str = "mean",
        high_freq_weight: float = 2.5,      # Emphasize high frequencies
        use_phase: bool = False,              # Phase carries edge/texture info
        phase_weight: float = 0.1,
        use_gradient: bool = False,           # Gradient domain loss for textures
        gradient_weight: float = 0.5,
        patch_size: int = 64,                # Local patches for local textures
        use_patches: bool = False,
    ):
        super().__init__()
        self.scales = scales
        self.eps = eps
        self.reduction = reduction
        self.high_freq_weight = high_freq_weight
        self.use_phase = use_phase
        self.phase_weight = phase_weight
        self.use_gradient = use_gradient
        self.gradient_weight = gradient_weight
        self.patch_size = patch_size
        self.use_patches = use_patches
        
        # Cache for frequency masks
        self._freq_mask_cache = {}

    def _get_high_freq_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create mask emphasizing high frequencies (fine details)."""
        key = (h, w, device)
        if key not in self._freq_mask_cache:
            # Frequency coordinates normalized to [-0.5, 0.5]
            fy = torch.fft.fftfreq(h, device=device)[:, None]
            fx = torch.fft.rfftfreq(w, device=device)[None, :]
            
            # Radial frequency normalized to [0, 1]
            freq_radius = torch.sqrt(fy ** 2 + fx ** 2)
            freq_radius = freq_radius / (freq_radius.max() + self.eps)
            
            # Smooth ramp: low freq -> 1.0, high freq -> high_freq_weight
            # Using squared ramp to emphasize highest frequencies more
            weight = 1.0 + (self.high_freq_weight - 1.0) * (freq_radius ** 1.5)
            
            self._freq_mask_cache[key] = weight
        return self._freq_mask_cache[key]

    def _single_scale_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h, w = pred.shape[-2:]
        
        F_pred = torch.fft.rfftn(pred, dim=(-2, -1))
        F_tgt = torch.fft.rfftn(target, dim=(-2, -1))
        
        mag_pred = torch.abs(F_pred)
        mag_tgt = torch.abs(F_tgt)
        
        log_mag_pred = torch.log(mag_pred + self.eps)
        log_mag_tgt = torch.log(mag_tgt + self.eps)
        
        # High-frequency weighted magnitude loss
        hf_mask = self._get_high_freq_mask(h, w, pred.device)
        mag_loss = torch.abs(log_mag_pred - log_mag_tgt) * hf_mask
        
        loss = mag_loss.mean(dim=(-2, -1))
        
        # Phase loss (important for edges and texture structure)
        if self.use_phase:
            phase_pred = torch.angle(F_pred)
            phase_tgt = torch.angle(F_tgt)
            
            # Circular phase difference
            phase_diff = phase_pred - phase_tgt
            phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
            
            # Weight by normalized magnitude (phase matters where signal exists)
            # and by frequency (high-freq phase = texture details)
            mag_weight = mag_tgt / (mag_tgt.amax(dim=(-2, -1), keepdim=True) + self.eps)
            phase_loss = (torch.abs(phase_diff) * mag_weight * hf_mask).mean(dim=(-2, -1))
            
            loss = loss + self.phase_weight * phase_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-order gradient loss for textures and edges."""
        loss = 0.0
        
        # First-order gradients
        dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
        dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
        dx_tgt = target[..., :, 1:] - target[..., :, :-1]
        dy_tgt = target[..., 1:, :] - target[..., :-1, :]
        
        loss += torch.abs(dx_pred - dx_tgt).mean()
        loss += torch.abs(dy_pred - dy_tgt).mean()
        
        # Second-order gradients (Laplacian-like, captures fine texture)
        dxx_pred = dx_pred[..., :, 1:] - dx_pred[..., :, :-1]
        dyy_pred = dy_pred[..., 1:, :] - dy_pred[..., :-1, :]
        dxx_tgt = dx_tgt[..., :, 1:] - dx_tgt[..., :, :-1]
        dyy_tgt = dy_tgt[..., 1:, :] - dy_tgt[..., :-1, :]
        
        loss += 0.5 * torch.abs(dxx_pred - dxx_tgt).mean()
        loss += 0.5 * torch.abs(dyy_pred - dyy_tgt).mean()
        
        return loss

    def _patch_spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss on local patches for local texture fidelity."""
        B, C, H, W = pred.shape
        ps = self.patch_size
        
        # Skip if image is too small
        if H < ps or W < ps:
            return torch.tensor(0.0, device=pred.device)
        
        # Unfold into patches
        pred_patches = pred.unfold(2, ps, ps // 2).unfold(3, ps, ps // 2)
        tgt_patches = target.unfold(2, ps, ps // 2).unfold(3, ps, ps // 2)
        
        # Reshape: (B, C, nH, nW, ps, ps) -> (B * nH * nW, C, ps, ps)
        nH, nW = pred_patches.shape[2], pred_patches.shape[3]
        pred_patches = pred_patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ps, ps)
        tgt_patches = tgt_patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ps, ps)
        
        # Compute spectral loss on patches (no reduction, then average)
        F_pred = torch.fft.rfftn(pred_patches, dim=(-2, -1))
        F_tgt = torch.fft.rfftn(tgt_patches, dim=(-2, -1))
        
        hf_mask = self._get_high_freq_mask(ps, ps, pred.device)
        
        log_mag_pred = torch.log(torch.abs(F_pred) + self.eps)
        log_mag_tgt = torch.log(torch.abs(F_tgt) + self.eps)
        
        loss = (torch.abs(log_mag_pred - log_mag_tgt) * hf_mask).mean()
        
        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        num_terms = 0
        
        # Multi-scale spectral loss
        for s in self.scales:
            if s > 1:
                pred_s = F.interpolate(
                    pred, scale_factor=1/s, mode='bilinear', 
                    align_corners=False, antialias=True
                )
                tgt_s = F.interpolate(
                    target, scale_factor=1/s, mode='bilinear', 
                    align_corners=False, antialias=True
                )
            else:
                pred_s, tgt_s = pred, target
            total += self._single_scale_loss(pred_s, tgt_s)
            num_terms += 1
        
        # Local patch spectral loss (captures local texture patterns)
        if self.use_patches:
            total += self._patch_spectral_loss(pred, target)
            num_terms += 1
        
        total = total / num_terms
        
        # Gradient domain loss (edges and micro-textures)
        if self.use_gradient:
            total = total + self.gradient_weight * self._gradient_loss(pred, target)
        
        return total


def radial_profile(log_mag):
    B, C, H, W = log_mag.shape
    device = log_mag.device

    y, x, = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    center_y, center_x = H // 2, W // 2
    r = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.long()
    max_r = r.max() + 1

    radial_mean = []

    for i in range(max_r):
        mask = (r == i)
        if mask.sum() > 0:
            radial_mean.append(log_mag[..., mask].mean(dim=-1))
        else:
            radial_mean.append(torch.zeros(B, C, device=device))

    return torch.stack(radial_mean, dim=-1)

def log_fft_magn_loss(pred, target, eps=1e-6):
    pred_fft = torch.fft.fft2(pred, norm='ortho')
    targ_fft = torch.fft.fft2(target, norm='ortho')

    pred_logmag = torch.log(torch.abs(pred_fft) + eps)
    targ_logmag = torch.log(torch.abs(targ_fft) + eps)

    pred_radial = radial_profile(pred_logmag)
    targ_radial = radial_profile(targ_logmag)

    return torch.mean(torch.abs(pred_radial - targ_radial))

def smoothness_loss(flow):
    b, _, h, w = flow.shape

    # Mirror warp()'s normalization: pixel → [-1, 1] space
    # warp uses: flow_x / ((W-1)/2),  flow_y / ((H-1)/2)
    flow_norm = torch.cat([
        flow[:, 0:1] / ((w - 1) / 2.0),
        flow[:, 1:2] / ((h - 1) / 2.0),
    ], dim=1)

    # Spatial gradients in normalized space — now resolution-independent
    dx = (flow_norm[:, :, :, 1:] - flow_norm[:, :, :, :-1]).abs()  # (B, 2, H, W-1)
    dy = (flow_norm[:, :, 1:, :] - flow_norm[:, :, :-1, :]).abs()  # (B, 2, H-1, W)

    # Local magnitude normalization — also on normalized flow
    mag_x = flow_norm[:, :, :, 1:].norm(dim=1, keepdim=True) + 1e-6
    mag_y = flow_norm[:, :, 1:, :].norm(dim=1, keepdim=True) + 1e-6

    return (dx / mag_x).mean() + (dy / mag_y).mean()

class ContrastStructureLoss(nn.Module):
    """
    SSIM with luminance term dropped — sensitive to edges and local contrast
    but invariant to brightness offsets.
    
    CS = (2σxσy + C2)/(σx² + σy² + C2)  ← contrast
       * (σxy + C3)/(σxσy + C3)           ← structure
    """
    def __init__(self, window_size=11, C2=0.03**2, C3=None):
        super().__init__()
        self.window_size = window_size
        self.C2 = C2
        self.C3 = C3 if C3 is not None else C2 / 2

        # 1D Gaussian → outer product → 2D, shape (1, 1, ws, ws)
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g /= g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window)

    def _gauss(self, x):
        # depthwise Gaussian blur, one channel at a time
        C = x.shape[1]
        w = self.window.expand(C, 1, -1, -1)
        pad = self.window_size // 2
        return F.conv2d(x, w, padding=pad, groups=C)

    def forward(self, pred, target):
        eps = 1e-6

        mu_p = self._gauss(pred)
        mu_t = self._gauss(target)

        sigma_p2 = self._gauss(pred   * pred)   - mu_p * mu_p
        sigma_t2 = self._gauss(target * target) - mu_t * mu_t
        sigma_pt = self._gauss(pred   * target) - mu_p * mu_t

        # clamp with eps before sqrt so backward stays finite
        sigma_p = sigma_p2.clamp(min=eps).sqrt()
        sigma_t = sigma_t2.clamp(min=eps).sqrt()
        sigma_pt_s = sigma_p * sigma_t

        contrast  = (2 * sigma_pt_s       + self.C2) / (sigma_p2 + sigma_t2 + self.C2)
        structure = (sigma_pt              + self.C3) / (sigma_pt_s           + self.C3)

        # clamp structure to [-1, 1] — negative covariance is valid but extreme values cause instability
        structure = structure.clamp(-1, 1)

        cs = contrast * structure
        return 1.0 - cs.mean()


def compute_lpips(loss_fn, pred, gt, max_size=256):
    _, _, h, w = pred.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        size = (round(h * scale), round(w * scale))
        pred = torch.nn.functional.interpolate(pred, size=size, mode='bilinear', align_corners=False)
        gt   = torch.nn.functional.interpolate(gt,   size=size, mode='bilinear', align_corners=False)
    return loss_fn(pred * 2 - 1, gt * 2 - 1).mean()


def prefetch_batch(dataset, batch_idx, world_size, batch_size, device):
    scatter_img0, scatter_img1, scatter_img2 = [], [], []
    ratio, idx, sample_idx, current_desc = None, 0, 0, {}
    for r in range(world_size):
        r_img0, r_img1, r_img2 = [], [], []
        for i in range(batch_size):
            sample_idx = (batch_idx * world_size * batch_size) + (r * batch_size) + i
            sample_idx = sample_idx % len(dataset)
            s0, s1, s2, ratio, idx, current_desc = dataset[sample_idx]
            r_img0.append(s0)
            r_img1.append(s1)
            r_img2.append(s2)
            if (sample_idx + 1) >= len(dataset):
                break
        while len(r_img0) < batch_size:
            r_img0.append(r_img0[-1])
            r_img1.append(r_img1[-1])
            r_img2.append(r_img2[-1])
        scatter_img0.append(torch.cat(r_img0[:batch_size]).to(device))
        scatter_img1.append(torch.cat(r_img1[:batch_size]).to(device))
        scatter_img2.append(torch.cat(r_img2[:batch_size]).to(device))
    return scatter_img0, scatter_img1, scatter_img2, ratio, idx, sample_idx, current_desc

def to_grey(x):
    weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)
def to_grey(x):
    weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)

current_state_dict = {}

def main(rank, world_size):
    global current_state_dict
    parser = argparse.ArgumentParser(description='Training script.')

    def check_range_percent(value):
        ivalue = int(value)
        if ivalue < 0 or ivalue > 100:
            raise argparse.ArgumentTypeError(f"Percent must be between 0 and 100, got value={ivalue}")
        return ivalue

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    # Optional arguments
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
    parser.add_argument('--type', type=int, default=1, help='Model type (int): 1 - MultiresNet, 2 - MultiresNet 4 (default: 1)')
    parser.add_argument('--pulse', type=float, default=10000, help='Period in steps to pulse learning rate (float) (default: 10K)')
    parser.add_argument('--pulse_amplitude', type=float, default=1e-1, help='Learning rate pulse amplitude (percentage) (default: 1e-1)')
    parser.add_argument('--onecycle', type=int, default=-1, help='Train one cycle for N epochs (default: None)')
    parser.add_argument('--cyclic', type=int, default=-1, help='Use cyclic LR scheduler')
    parser.add_argument('--state_file', type=str, default=None, help='Path to the pre-trained model state dict file (optional)')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--legacy_model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (int) (default: 8)')
    parser.add_argument('--first_epoch', type=int, default=-1, help='Epoch (int) (default: Saved)')
    parser.add_argument('--epochs', type=int, default=-1, help='Number of epoch to run (int) (default: Unlimited)')
    parser.add_argument('--reset_stats', action='store_true', dest='reset_stats', default=False, help='Reset saved step, epoch and loss stats')

    parser.add_argument('--eval', type=int, dest='eval', default=-1, help='Evaluate after N steps')
    parser.add_argument('--eval_first', action='store_true', dest='eval_first', default=False, help='Reset saved step, epoch and loss stats')
    parser.add_argument('--eval_samples', type=int, dest='eval_samples', default=-1, help='Evaluate N random training samples')
    parser.add_argument('--eval_seed', type=int, dest='eval_seed', default=1, help='Random seed to select samples if --eval_samples set')
    parser.add_argument('--eval_buffer', type=int, dest='eval_buffer', default=8, help='Write buffer size for evaluated images')
    parser.add_argument('--eval_save_imgs', action='store_true', dest='eval_save_imgs', default=False, help='Save eval result images')
    parser.add_argument('--eval_keep_all', action='store_true', dest='eval_keep_all', default=False, help='Keep eval results for each eval step')
    parser.add_argument('--eval_folder', type=str, default=None, help='Folder with clips for evaluation')
    parser.add_argument('--eval_half', action='store_true', dest='eval_half', default=False, help='Evaluate in half-precision')

    parser.add_argument('--frame_size', type=int, default=448, help='Frame size in pixels (default: 448)')
    parser.add_argument('--max_window', type=int, default=12, help='Temporal window size in frames (default: 12)')
    parser.add_argument('--all_gpus', action='store_true', dest='all_gpus', default=False, help='Use DistributedDataParallel')
    parser.add_argument('--freeze', action='store_true', dest='freeze', default=False, help='Freeze custom hardcoded parameters')
    parser.add_argument('--acescc', type=check_range_percent, default=0, help='Percentage of ACEScc encoded frames (default: 49))')
    parser.add_argument('--ap0', action='store_true', dest='ap0', default=False, help='Images are in ap0')
    parser.add_argument('--generalize', type=check_range_percent, default=85, help='Generalization level (0 - 100) (default: 85)')
    parser.add_argument('--weight_decay', type=float, default=-1, help='AdamW weight decay (default: calculated from --generalize value)')
    parser.add_argument('--preview', type=int, default=100, help='Save preview each N steps (default: 100)')
    parser.add_argument('--preview_max', type=int, default=0, help='Save separate preview for N highest error samples (default: 0)')
    parser.add_argument('--preview_min', type=int, default=0, help='Save separate preview for N lowest error samples (default: 0)')
    parser.add_argument('--preview_maxmin_steps', type=int, default=10000, help='Save max or min preview each N steps (default: 10000)')
    parser.add_argument('--save', type=int, default=10000, help='Save model state dict each N steps (default: 10000)')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat each triade N times with augmentation (default: 1)')
    parser.add_argument('--iterations', type=int, default=1, help='Process each flow refinement N times (default: 1)')
    parser.add_argument('--compile', action='store_true', dest='compile', default=False, help='Compile with torch.compile')
    parser.add_argument('--sequential', action='store_true', dest='sequential', default=False, help='Keep sequences, do not reshuffle')

    args = parser.parse_args()

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
        device = torch.device(f'cuda:{args.device}') if not platform.system() == 'Darwin' else torch.device('mps')
        if not platform.system() == 'Darwin':
            torch.cuda.set_device(device)

    is_master = (rank == 0)

    Flownet = None

    if args.model:
        model_name = args.model
        Flownet = find_and_import_model(base_name='stabnet', model_name=model_name)
    else:
        if args.state_file and os.path.isfile(args.state_file):
            trained_model_path = args.state_file
            try:
                checkpoint = torch.load(trained_model_path, map_location=device)
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
    # DDP wrapping — replaces DataParallel
    # -------------------------------------------------------------------------
    if world_size > 1:
        flownet = torch.nn.parallel.DistributedDataParallel(flownet, device_ids=[rank])
        if is_master:
            print(f'Using DistributedDataParallel across {world_size} GPUs')

    if is_master:
        if not os.path.isdir(os.path.join(args.dataset_path, 'preview')):
            os.makedirs(os.path.join(args.dataset_path, 'preview'))

    frame_size = args.frame_size

    dataset = get_dataset(
        args.dataset_path,
        batch_size=1,
        device=device,
        frame_size=frame_size,
        max_window=max_dataset_window,
        acescc_rate=args.acescc,
        generalize=args.generalize,
        repeat=args.repeat,
        sequential=args.sequential,
        start_reader=is_master
    )

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
        eval_dataset = dataset

    # -------------------------------------------------------------------------
    # Background write threads — rank 0 only
    # -------------------------------------------------------------------------
    if is_master:
        def write_images(write_image_queue):
            while True:
                try:
                    write_data = write_image_queue.get_nowait()
                    preview_index = write_data.get('preview_index', 0)
                    preview_folder = write_data["preview_folder"]
                    if not os.path.isdir(preview_folder):
                        os.makedirs(preview_folder)
                    write_exr(write_data['sample_target'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_A_target.exr'), half_float=True)
                    write_exr(write_data['sample_output'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_B_output.exr'), half_float=True)
                    write_exr(write_data['sample_source'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_C_src.exr'), half_float=True)
                    write_exr(write_data['sample_rev'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_D_rev.exr'), half_float=True)
                    del write_data
                except:
                    time.sleep(1e-2)

        def write_eval_images(write_eval_image_queue):
            while True:
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
                try:
                    current_state_dict = write_model_state_queue.get_nowait()
                    trained_model_path = current_state_dict['trained_model_path']
                    if os.path.isfile(trained_model_path):
                        backup_file = trained_model_path.replace('.pth', '.backup.pth')
                        shutil.copy(trained_model_path, backup_file)
                    torch.save(current_state_dict, current_state_dict['trained_model_path'])
                except:
                    time.sleep(1e-2)

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
    criterion_lap = LapLossNCC().to(device)
    criterion_grad = GradientLoss().to(device)
    criterion_contrast_struct = ContrastStructureLoss().to(device)
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
                last_epoch=-1 if loaded_sranktep == 0 else loaded_step
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
            print(f'setting ReduceLROnPlateau scheduler with factor={0.1}, patience={10}')
        scheduler_flownet = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_flownet, 'min', factor=0.1, patience=10)

    # '''
    # -------------------------------------------------------------------------
    # LPIPS — each rank loads its own copy on its own device
    # -------------------------------------------------------------------------
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    import lpips
    os.environ['TORCH_HOME'] = os.path.abspath(os.path.dirname(__file__))
    loss_fn_alex = lpips.LPIPS(net='alex', spatial=False)
    loss_fn_alex.to(device)
    for param in loss_fn_alex.parameters():
        param.requires_grad = False

    warnings.resetwarnings()
    # '''

    start_timestamp = time.time()
    time_stamp = time.time()
    epoch = current_epoch if args.first_epoch == -1 else args.first_epoch
    step = loaded_step if args.first_epoch == -1 else step
    batch_idx = 0

    if args.freeze:
        if is_master:
            print('\nFreezing parameters')
        target_model = flownet.module if world_size > 1 else flownet
        for param in target_model.encode.parameters():
            param.requires_grad = False
        for param in target_model.block0.parameters():
            param.requires_grad = False
        if is_master:
            for name, param in flownet.named_parameters():
                if not param.requires_grad:
                    print(name, param.requires_grad)
            print('\nUn-freezing parameters:')
            for name, param in flownet.named_parameters():
                if param.requires_grad:
                    print(name, param.requires_grad)

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
                print(f'\nSaving current state to {current_state_dict["trained_model_path"]}...')
                print(f'Epoch: {current_state_dict["epoch"] + 1}, Step: {current_state_dict["step"]:11}')
                torch.save(current_state_dict, current_state_dict['trained_model_path'])
                exit_event.set()
                process_exit_event.set()
                exit(0)
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

    cur_size = 100000
    cur_mask = np.full(cur_size, True)
    cur_l1 = None
    cur_rev_l1 = None
    cur_comb = None
    cur_lpips = None

    repeat_count = dataset.repeat_count if dataset.repeat_count > 0 else 1
    preview_maxmin_steps = args.preview_maxmin_steps if args.preview_maxmin_steps < len(dataset) * repeat_count else len(dataset) * repeat_count
    max_values = MaxNValues(n=args.preview_max if args.preview_max else 10)
    min_values = MinNValues(n=args.preview_min if args.preview_min else 10)

    data_time = 0
    data_time1 = 0
    data_time2 = 0
    train_time = 0

    log_spec_loss = MultiScaleLogMagSpectralLoss(scales=(1, 2, 4))

    ap02cct = AP0toACESCCT().to(device)
    ap12cct = AP1toACESCCT().to(device)
    cct2cg = ACESCCTtoACESCG().to(device)

    # prefetch executor for DDP batch preparation
    import concurrent.futures
    _prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if (is_master and world_size > 1) else None
    _next_batch_future = None

    loss_LPIPS = torch.zeros(1, device=device, requires_grad=True)

    while True:
        time_stamp = time.time()

        img0_list, img1_list, img2_list, ratio_list = [], [], [], []

        # rank 0 collects a full batch for every rank, then scatters
        if world_size > 1:
            if is_master:
                # retrieve prefetched batch from previous step, or fetch now on first step
                if _next_batch_future is not None:
                    scatter_img0, scatter_img1, scatter_img2, ratio, idx, sample_idx, current_desc = _next_batch_future.result()
                else:
                    scatter_img0, scatter_img1, scatter_img2, ratio, idx, sample_idx, current_desc = prefetch_batch(
                        dataset, batch_idx, world_size, args.batch_size, device)
            else:
                scatter_img0 = scatter_img1 = scatter_img2 = None
                ratio = None
                idx = 0
                sample_idx = 0
                current_desc = {}

            # receive buffer — shape must match what rank 0 prepared per rank
            img0 = torch.zeros(args.batch_size, 3, frame_size, frame_size, device=device)
            img1 = torch.zeros_like(img0)
            img2 = torch.zeros_like(img0)

            torch.distributed.scatter(img0, scatter_img0, src=0)
            torch.distributed.scatter(img1, scatter_img1, src=0)
            torch.distributed.scatter(img2, scatter_img2, src=0)

            # broadcast scalar metadata from rank 0 (used only in is_master blocks anyway)
            meta = [ratio, idx, sample_idx]
            torch.distributed.broadcast_object_list(meta, src=0)
            ratio, idx, sample_idx = meta

        else:
            # single GPU — original path
            img0_list, img1_list, img2_list = [], [], []
            for i in range(args.batch_size):
                sample_idx = batch_idx * args.batch_size + i
                sample_idx = sample_idx % len(dataset)
                s0, s1, s2, ratio, idx, current_desc = dataset[sample_idx]
                img0_list.append(s0)
                img1_list.append(s1)
                img2_list.append(s2)
                if (sample_idx + 1) >= len(dataset):
                    break
            img0 = torch.cat(img0_list)
            img1 = torch.cat(img1_list)
            img2 = torch.cat(img2_list)

        img0 = img0.to(device, non_blocking=True)
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        if args.ap0:
            img0 = ap02cct(img0)
            img1 = ap02cct(img1)
            img2 = ap02cct(img2)

        '''
        if random.uniform(0, 1) > 0.2:
            scale_augm = random.uniform(1, 2)        
            nn, nc, nh, nw = img0.shape
            sh, sw = round(nh * (1 / scale_augm)), round(nw * (1 / scale_augm))
            sh = max(sh, 96)
            sw = max(sw, 96)
            sh += 4 - (sh % 4)
            sw += 4 - (sw % 4)
            img0 = torch.nn.functional.interpolate(img0, size=(sh, sw), mode="bicubic", align_corners=False)
            img1 = torch.nn.functional.interpolate(img1, size=(sh, sw), mode="bicubic", align_corners=False)
            img2 = torch.nn.functional.interpolate(img2, size=(sh, sw), mode="bicubic", align_corners=False)
        '''

        img0 = torch.cat([img0, img0], dim=0)
        img1 = torch.cat([img1, img2], dim=0)

        img0_orig = img0.detach().clone()
        img1_orig = img1.detach().clone()
        img2_orig = img2.detach().clone()

        current_lr_str = str(f'{optimizer_flownet.param_groups[0]["lr"]:.2e}')
        
        # '''
        if random.uniform(0, 1) < 0.69:
            scale = torch.linspace(random.randint(4, 16), 1, steps=4).tolist()
            training_scale = [round(v) for v in scale]
        else:
            training_scale = [8, 4, 2, 1]
        # '''

        training_scale = [8, 4, 2, 1]

        data_time = time.time() - time_stamp
        time_stamp = time.time()

        optimizer_flownet.zero_grad()
        flownet.train()

        # Exposure augmentation
        exp1 = random.uniform(1 / 4, 1.4) if random.uniform(0, 1) < 0.44 else 1
        exp2 = random.uniform(1 / 4, 1.4) if random.uniform(0, 1) < 0.44 else 1

        noise1 = min(random.uniform(0, 0.12), random.uniform(0, 0.12)) * torch.clamp(torch.randn_like(img0), 0)
        noise2 = min(random.uniform(0, 0.12), random.uniform(0, 0.12)) * torch.clamp(torch.randn_like(img0), 0)

        result = flownet(
            ap12cct(cct2cg(img0) * exp1 + noise1),
            ap12cct(cct2cg(img1) * exp2 + noise2),
            scale=training_scale,
        )

        flow_list = result['flow_list']

        flow_fwd, flow_bkw = torch.split(flow_list[-1], 2, dim=1)

        output_fwd = warp(img1, flow_fwd)
        output_bkw = warp(img0, flow_bkw)
        output_rev = warp(output_fwd, flow_bkw)

        model_time = time.time() - time_stamp
        time_stamp = time.time()

        int_loss = torch.zeros(1, device=device, requires_grad=True)

        '''
        out_int_fwd = warp(img1, int_flow_fwd)

        # --- Fidelity role: trains flow_bkw only ---
        # stop-grad on the forward output so this loss never touches flow_fwd,
        # regardless of which experiment arm is active
        out_int_rev_fidelity = warp(out_int_fwd.detach(), int_flow_bkw)
        loss_rev_fidelity_l1  = criterion_l1(out_int_rev_fidelity, gt_int_bkw)
        loss_rev_fidelity_lap = criterion_lap(out_int_rev_fidelity, gt_int_bkw)

        # --- Cycle-consistency role: optionally lets flow_fwd feel reverse-cycle pressure ---
        # stop-grad on flow_bkw here so this term never also retunes flow_bkw —
        # it should purely probe "is flow_fwd's output invertible", not re-teach flow_bkw
        if cycle_consistency_mode == "coupled":
            out_int_rev_cycle = warp(out_int_fwd, int_flow_bkw.detach())
            loss_cycle_l1 = criterion_l1(out_int_rev_cycle, gt_int_bkw)
        else:  # "detached" arm
            loss_cycle_l1 = torch.zeros((), device=out_int_fwd.device)

        loss_rev = (loss_weights["l1"]  * loss_rev_fidelity_l1
                    + loss_weights["lap"] * loss_rev_fidelity_lap
                    + cycle_weight * loss_cycle_l1)   # cycle_weight = 0 in "detached" arm
        '''

        '''
        g_align = torch.autograd.grad(loss_contr_fwd + loss_grad_fwd, int_flow_fwd,
                                    retain_graph=True)[0].norm()
        g_cycle = torch.autograd.grad(loss_cycle_l1, int_flow_fwd,
                                    retain_graph=True)[0].norm()
        print(g_cycle / g_align)
        '''

        loss_weights = {
            "l1": 1.0,
            "lap": 0.1,
            "grad": 0.5,
            "contr": 0.5,
            "smooth_fwd": 1e-2,
            "smooth_bkw": 1e-3,
        }
        cycle_weight = 0.05  # single knob if reverse is meant to matter less overall

        # '''
        for flow_idx, int_flow in enumerate(flow_list):
            _, _, h, w = img1.shape
            scale = training_scale[flow_idx]
            sh, sw = round( (h * (1 / scale)) / 16 ) * 16, round( (w * (1 / scale)) / 16 ) * 16
            sh = max(sh, 64)
            sw = max(sw, 64)

            int_flow_fwd, int_flow_bkw = torch.split(int_flow, 2, dim=1)

            out_int_fwd = warp(img1, int_flow_fwd)
            out_int_rev_fidelity = warp(out_int_fwd.detach(), int_flow_bkw)
            out_int_rev_cycle = warp(out_int_fwd, int_flow_bkw.detach())

            out_int_fwd = torch.nn.functional.interpolate(out_int_fwd, size=(sh, sw), mode="bilinear",  antialias=True)
            out_int_rev_fidelity = torch.nn.functional.interpolate(out_int_rev_fidelity, size=(sh, sw), mode="bilinear", antialias=True)
            out_int_rev_cycle = torch.nn.functional.interpolate(out_int_rev_cycle, size=(sh, sw), mode="bilinear", antialias=True)
            gt_int_fwd = torch.nn.functional.interpolate(img0, size=(sh, sw), mode="bilinear", antialias=True)
            gt_int_bkw = torch.nn.functional.interpolate(img1, size=(sh, sw), mode="bilinear", antialias=True)

            # --- Fidelity role: trains flow_bkw only ---
            # stop-grad on the forward output so this loss never touches flow_fwd,
            # regardless of which experiment arm is active
            loss_rev_fidelity_l1  = criterion_l1(out_int_rev_fidelity, gt_int_bkw)
            loss_rev_fidelity_lap = criterion_lap(out_int_rev_fidelity, gt_int_bkw)

            # --- Cycle-consistency role: optionally lets flow_fwd feel reverse-cycle pressure ---
            # stop-grad on flow_bkw here so this term never also retunes flow_bkw —
            # it should purely probe "is flow_fwd's output invertible", not re-teach flow_bkw
            loss_cycle_l1 = criterion_l1(out_int_rev_cycle, gt_int_bkw)

            int_loss_rev = (loss_weights["l1"]  * loss_rev_fidelity_l1
                        + loss_weights["lap"] * loss_rev_fidelity_lap
                        + cycle_weight * loss_cycle_l1)   # cycle_weight = 0 in "detached" arm

            # scale down and back to reduce detail level according to scale factor to mimc what model see
            '''
            out_fwd_scaled = torch.nn.functional.interpolate(out_int_fwd, size=(sh, sw), mode="bilinear", align_corners=True)
            out_bkw_scaled = torch.nn.functional.interpolate(out_int_bkw, size=(sh, sw), mode="bilinear", align_corners=True)
            gt_fwd_scaled = torch.nn.functional.interpolate(img0, size=(sh, sw), mode="bilinear", align_corners=True)
            gt_bkw_scaled = torch.nn.functional.interpolate(img1, size=(sh, sw), mode="bilinear", align_corners=True)

            out_int_fwd = torch.nn.functional.interpolate(out_fwd_scaled, size=(h, w), mode="bilinear", align_corners=True)
            out_int_bkw = torch.nn.functional.interpolate(out_bkw_scaled, size=(h, w), mode="bilinear", align_corners=True)
            gt_int_fwd = torch.nn.functional.interpolate(gt_fwd_scaled, size=(h, w), mode="bilinear", align_corners=True)
            gt_int_bkw = torch.nn.functional.interpolate(gt_bkw_scaled, size=(h, w), mode="bilinear", align_corners=True)
            '''

            '''
            # if flow_idx == 0:
            int_loss_fwd_LPIPS = compute_lpips(loss_fn_alex, out_int_fwd, gt_int_fwd)
            int_loss_bkw_LPIPS = compute_lpips(loss_fn_alex, out_int_bkw, gt_int_bkw)
            int_loss = int_loss + 0.1 * int_loss_fwd_LPIPS + 0.1 * int_loss_bkw_LPIPS
            '''
            
            '''
            int_loss_l1_fwd = criterion_l1(out_int_fwd, gt_int_fwd)
            int_loss_l1_bkw = criterion_l1(out_int_bkw, gt_int_bkw)
            int_loss_l1_fwd_hp = criterion_l1(hpass(out_int_fwd), hpass(gt_int_fwd))
            int_loss_l1_bkw_hp = criterion_l1(hpass(out_int_bkw), hpass(gt_int_bkw))
            '''

            int_loss_fwd_l1 = criterion_l1(out_int_fwd, gt_int_fwd)

            int_loss_fwd_lap = criterion_lap(out_int_fwd, gt_int_fwd)
            # int_loss_bkw_lap = criterion_lap(out_int_bkw, gt_int_bkw)
            # int_loss_rev_lap = criterion_lap(out_int_rev, gt_int_bkw)

            int_loss_fwd_grad = criterion_grad(out_int_fwd, gt_int_fwd)
            # int_loss_bkw_grad = criterion_grad(out_int_bkw, gt_int_bkw)

            int_loss_fwd_contr = criterion_contrast_struct(out_int_fwd, gt_int_fwd)
            # int_loss_bkw_contr = criterion_contrast_struct(out_int_bkw, gt_int_bkw)

            int_loss_fwd_smooth = smoothness_loss(int_flow_fwd)
            int_loss_bkw_smooth = smoothness_loss(int_flow_bkw)

            int_loss_fwd = (
                + loss_weights["grad"] * int_loss_fwd_grad
                + loss_weights["contr"] * int_loss_fwd_contr
            )

            int_loss_smooth = (
                loss_weights['smooth_fwd'] * int_loss_fwd_smooth
                + loss_weights['smooth_fwd'] * int_loss_bkw_smooth
            )


            int_loss = int_loss + int_loss_fwd + int_loss_rev + int_loss_smooth

        '''
        g_align = torch.autograd.grad(int_loss_fwd, int_flow_fwd,
                                    retain_graph=True)[0].norm()
        g_cycle = torch.autograd.grad(cycle_weight * loss_cycle_l1, int_flow_fwd,
                                    retain_graph=True)[0].norm()
        print(g_cycle / g_align)
        print ('\n\n')
        '''

        # '''

        loss_l1 = criterion_l1(output_fwd, img0)
        loss_rev = criterion_l1(output_rev, img1)
        # loss_smooth = smoothness_loss(flow_fwd) + smoothness_loss(flow_bkw)
        # loss_contr = criterion_contrast_struct(output_fwd, img0) # + criterion_contrast_struct(output_bkw, img1)
        # loss_grad_hp = criterion_grad(hpass(output_fwd), hpass(img0)) + criterion_grad(hpass(output_bkw), hpass(img1))
        # loss_grad = criterion_grad(output_fwd, img0) # + criterion_grad(output_bkw, img1)
        loss_lap_fwd = criterion_lap(output_fwd, img0) # + criterion_lap(output_bkw, img1)
        # loss_lap_rev = criterion_lap(output_rev, img1)
        # loss_lap_hp = criterion_lap(hpass(output_fwd), hpass(img0)) + criterion_lap(hpass(output_bkw), hpass(img1))
        
        # loss_l1_hp = criterion_l1(hpass(output_fwd), hpass(img0)) + criterion_l1(hpass(output_bkw), hpass(img1))
        # loss_lap = criterion_lap(output_fwd, img0) + criterion_lap(output_bkw, img1)
        # 
        # loss_contr = criterion_contrast_struct(output_fwd, img0) + criterion_contrast_struct(output_bkw, img1)
        if step % 100 == 1:
            loss_LPIPS = compute_lpips(loss_fn_alex, output_fwd, img0) # + compute_lpips(loss_fn_alex, output_bkw, img1)
        # loss_spectral = log_spec_loss(output_fwd, img0)
        # loss_LPIPS = loss_smooth # torch.zeros(1, device=device, requires_grad=False)

        # loss = int_loss + 0.05 * loss_spectral + 0.2 * loss_LPIPS + 0.1 * loss_grad + 0.4 * loss_contr + 0.1 * loss_lap + 0.4 * loss_rev + 1e-2 * loss_smooth # + 0.1 * loss_lap_hp + 0.1 * loss_contr + 0.1 * loss_grad + 
        
        loss = int_loss + loss_lap_fwd

        '''
            int_loss +
            loss_contr +
            loss_LPIPS +
            0.05 * loss_lap +
            0.1 * loss_l1_hp +
            0.1 * loss_lap_hp
        )
        '''

        # ---------------------------------------------------------------------
        # Loss tracking — reduce across ranks so all ranks log same value
        # ---------------------------------------------------------------------
        if world_size > 1:
            loss_reduced = loss.detach().clone()
            torch.distributed.all_reduce(loss_reduced, op=torch.distributed.ReduceOp.AVG)
            loss_val = float(loss_reduced.item())
            l1_reduced = loss_l1.detach().clone()
            torch.distributed.all_reduce(l1_reduced, op=torch.distributed.ReduceOp.AVG)
            l1_val = float(l1_reduced.item())
            lpips_reduced = loss_LPIPS.detach().clone()
            torch.distributed.all_reduce(lpips_reduced, op=torch.distributed.ReduceOp.AVG)
            lpips_val = float(lpips_reduced.item())
            rev_l1_reduced = loss_rev.detach().clone()
            torch.distributed.all_reduce(rev_l1_reduced, op=torch.distributed.ReduceOp.AVG)
            rev_l1_val = float(rev_l1_reduced.item())
        else:
            loss_val = float(loss.item())
            l1_val = float(loss_l1.item())
            lpips_val = float(torch.mean(loss_LPIPS).item())
            rev_l1_val = float(loss_rev.item())

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
            if cur_rev_l1 is None:
                cur_rev_l1 = np.full(cur_size, rev_l1_val)
            cur_rev_l1[cur_idx] = rev_l1_val

            min_l1 = min(min_l1, l1_val)
            max_l1 = max(max_l1, l1_val)
            avg_loss = loss_val if batch_idx == 0 else (avg_loss * (batch_idx - 1) + loss_val) / batch_idx
            avg_l1 = l1_val if batch_idx == 0 else (avg_l1 * (batch_idx - 1) + l1_val) / batch_idx
            avg_lpips = lpips_val if batch_idx == 0 else (avg_lpips * (batch_idx - 1) + lpips_val) / batch_idx
            avg_pnsr = float(psnr_torch(output_fwd, img0_orig)) if batch_idx == 0 else (avg_pnsr * (batch_idx - 1) + float(psnr_torch(output_fwd, img0_orig))) / batch_idx

            cur_comb[cur_mask] = avg_loss
            cur_l1[cur_mask] = avg_l1
            cur_lpips[cur_mask] = avg_lpips

        loss.backward()
        torch.nn.utils.clip_grad_norm_(flownet.parameters(), 1)
        optimizer_flownet.step()

        if isinstance(scheduler_flownet, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass
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
        
        # '''
        if platform.system() == 'Darwin':
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize(device=device)
        # '''

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
                rgb_target = cct2cg(img0_orig)
                rgb_source = cct2cg(img1_orig)
                rgb_output = cct2cg(output_fwd)
                rgb_output2 = cct2cg(output_rev)

                preview_index += 1
                preview_index = preview_index if preview_index < 10 else 0

                write_image_queue.put({
                    'preview_folder': os.path.join(args.dataset_path, 'preview', os.path.splitext(os.path.basename(trained_model_path))[0]),
                    'preview_index': int(preview_index),
                    'sample_target': rgb_target[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_source': rgb_source[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output': rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_rev': rgb_output2[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                })
                del rgb_source, rgb_target, rgb_output

            current_desc['loss'] = loss_val
            current_desc['loss_l1'] = l1_val
            current_desc['lpips'] = lpips_val

            min_max_item = {
                'description': current_desc,
                'img0_orig': img0_orig.numpy(force=True).copy(),
                'img1_orig': img1_orig.numpy(force=True).copy(),
                'img2_orig': img2_orig.numpy(force=True).copy(),
                'output': output_fwd.numpy(force=True).copy(),
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
            minutes = int((epoch_time % 3600) // 60)

            clear_lines(2)
            print(f'\r[Epoch {(epoch + 1):04} Step {step} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time_str}+{model_time_str}+{train_time_str}+{data_time2_str}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}')
            if len(dataset) > cur_size:
                print(f'\r[100K Average] L1: {np.mean(cur_l1):.6f} RevL1: {np.mean(cur_rev_l1):.6f} LPIPS: {np.mean(cur_lpips):.4f} Combined: {np.mean(cur_comb):.6f}')
                if (step + 1) % cur_size == 1:
                    csv_file_name = f'{os.path.splitext(trained_model_path)[0]}_train_loss_10K.csv'
                    if not os.path.isfile(csv_file_name):
                        create_csv_file(csv_file_name, ['Epoch', 'Step', 'L1', 'LPIPS', 'Combined'])
                    else:
                        for row in [{'Epoch': epoch, 'Step': step, 'L1': np.mean(cur_l1), 'LPIPS': np.mean(cur_lpips), 'Combined': np.mean(cur_comb)}]:
                            append_row_to_csv(csv_file_name, row)
                    clear_lines(2)
                    print(f'\r[Step {step + 1}] Avg L1: {avg_l1:.6f} RevL1: {np.mean(cur_rev_l1):.6f} LPIPS: {avg_lpips:.4f} Combined: {avg_loss:.6f}')
                    print('\n')
            else:
                print(f'\r[Epoch] Min L1: {min_l1:.6f} Avg L1: {avg_l1:.6f} Max L1: {max_l1:.6f} RevL1: {np.mean(cur_rev_l1):.6f} LPIPS: {avg_lpips:.4f} Combined: {avg_loss:.6f}')

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
                minutes = int((epoch_time % 3600) // 60)

                clear_lines(2)
                print(f'\rEpoch [{epoch + 1} (Step {step:11} - {days:02}d {hours:02}:{minutes:02}], Min L1: {min_l1:.6f} Avg L1: {avg_l1:.6f} Max L1: {max_l1:.6f} Avg LPIPS: {avg_lpips:.4f} Combined: {avg_loss:.6f}')
                print('\n')

                for row in [{'Epoch': epoch, 'Step': step, 'Min': min_l1, 'Avg': avg_l1, 'Max': max_l1, 'PSNR': avg_pnsr, 'LPIPS': avg_lpips}]:
                    append_row_to_csv(f'{os.path.splitext(trained_model_path)[0]}.csv', row)

                if args.eval == 0:
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

            descriptions = list(eval_dataset.initial_train_descriptions)
            if args.eval_samples > 0:
                rng = random.Random(args.eval_seed)
                descriptions = rng.sample(descriptions, args.eval_samples)

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

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()

            flownet.cpu()

            if torch.cuda.is_available():
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
                    minutes = int((epoch_time % 3600) // 60)

                    clear_lines(1)
                    print(f'\rEvaluating {ev_item_index} of {len(descriptions)}: Min: {eval_loss_min:.6f} Avg: {eval_loss_avg:.6f}, Max: {eval_loss_max:.6f} LPIPS: {eval_lpips_mean:.4f} PSNR: {eval_psnr_mean:4f}')

                    try:
                        eval_img0 = torch.from_numpy(description['eval_img0']).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                        eval_img1 = torch.from_numpy(description['eval_img1']).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                        eval_img2 = torch.from_numpy(description['eval_img2']).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                        eval_ratio = description['ratio']

                        eval_img0_orig = eval_img0.clone()
                        eval_img2_orig = eval_img2.clone()

                        if args.eval_half:
                            eval_img0 = eval_img0.half()
                            eval_img2 = eval_img2.half()

                        result = evalnet(eval_img0, eval_img2, eval_ratio, iterations=args.iterations)

                        eval_flow_list = result['flow_list']
                        eval_mask_list = result['mask_list']
                        eval_conf_list = result['conf_list']
                        eval_merged = result['merged']

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
                                'sample_source1': eval_img0_orig[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_source1_name': f'{ev_item_index:08}_A_incomng.exr',
                                'sample_source2': eval_img2_orig[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_source2_name': f'{ev_item_index:08}_B_outgoing.exr',
                                'sample_target': eval_img1[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_target_name': f'{ev_item_index:08}_C_target.exr',
                                'sample_output': eval_result[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_output_name': f'{ev_item_index:08}_D_output.exr',
                                'sample_output_diff': eval_rgb_diff[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_output_diff_name': f'{ev_item_index:08}_E_diff.exr',
                                'sample_output_conf': eval_rgb_conf[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_output_conf_name': f'{ev_item_index:08}_F_conf.exr',
                                'sample_output_mask': eval_rgb_output_mask[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                'sample_output_mask_name': f'{ev_item_index:08}_G_mask.exr'
                            })

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        elif torch.backends.mps.is_available():
                            torch.mps.synchronize()

                        del eval_img0, eval_img1, eval_img2, eval_img0_orig, eval_img2_orig
                        del eval_flow_list, eval_mask_list, eval_conf_list, eval_merged
                        del result, eval_result, eval_rgb_output_mask, eval_rgb_diff, eval_rgb_conf
                        del description['eval_img0'], description['eval_img1'], description['eval_img2']

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                    except Exception as e:
                        del description['eval_img0']
                        del description['eval_img1']
                        del description['eval_img2']
                        print(f'\n\nerror while evaluating: {e}\n{description}\n{traceback.format_exc()}\n\n')
                    description = read_eval_image_queue.get()

            for eval_row in [{'Epoch': epoch, 'Step': step, 'Min': eval_loss_min, 'Avg': eval_loss_avg, 'Max': eval_loss_max, 'PSNR': eval_psnr_mean, 'LPIPS': eval_lpips_mean}]:
                append_row_to_csv(f'{os.path.splitext(trained_model_path)[0]}.eval.csv', eval_row)

            clear_lines(2)
            print(f'\r[Epoch {(epoch + 1):04} Step {step:08} - {days:02}d {hours:02}:{minutes:02}], Eval Min: {eval_loss_min:.6f} Avg: {eval_loss_avg:.6f}, Max: {eval_loss_max:.6f}, [PSNR] {eval_psnr_mean:.4f}, [LPIPS] {eval_lpips_mean:.4f}')
            print('\n')

            if not args.eval_keep_all:
                if prev_eval_folder and os.path.isdir(prev_eval_folder):
                    threading.Thread(target=lambda: os.system(f'rm -rf {os.path.abspath(prev_eval_folder)}')).start()
            prev_eval_folder = eval_folder

            del evalnet
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            flownet.to(device)
            flownet.train()

            read_eval_thread.join()
            del read_eval_image_queue

            if isinstance(scheduler_flownet, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_flownet.step(eval_loss_avg)

        # End of evaluation block

        batch_idx = batch_idx + 1
        step = step + 1

        # kick off next batch prefetch overlapping with next forward pass
        if is_master and world_size > 1:
            _next_batch_future = _prefetch_executor.submit(
                prefetch_batch, dataset, batch_idx, world_size, args.batch_size, device)

        data_time2 = time.time() - time_stamp

        if epoch == args.epochs:
            if world_size > 1:
                torch.distributed.destroy_process_group()
            sys.exit()

if __name__ == "__main__":
    if '--all_gpus' in sys.argv:
        world_size = torch.cuda.device_count()
        try:
            torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
        except KeyboardInterrupt:
            pass
    else:
        main(0, 1)

