import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def get_dataset(
        data_root, 
        batch_size = 8, 
        device = None, 
        frame_size=448, 
        max_window=9,
        acescc_rate = 0,
        generalize = 80,
        repeat = 1,
        sequential = False
        ):
    class TimewarpMLDataset(torch.utils.data.Dataset):
        def __init__(   
                self, 
                data_root, 
                batch_size = 4, 
                device = None, 
                frame_size=448, 
                max_window=9,
                acescc_rate = 40,
                generalize = 80,
                repeat = 1,
                sequential = False
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
            # self.frame_multiplier = (self.src_w // self.w) * (self.src_h // self.h) * 4

            self.frame_read_process = None
            self.frames_queue = torch.multiprocessing.Queue(maxsize=4)
            self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
            self.frame_read_thread.daemon = True
            self.frame_read_thread.start()

            print ('reading first block of training data...')
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
            self.new_sample_thread = threading.Thread(target=new_sample_fetch, args=(self.frames_queue, self.new_sample_queue))
            self.new_sample_thread.daemon = True
            self.new_sample_thread.start()

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
            elif 'slow' in folder_path:
                max_window = 7
            elif 'slower' in folder_path:
                max_window = max_window
            else:
                if max_window > 5:
                    max_window = 5

            try:
                first_exr_file_header = read_image_file(exr_files[0], header_only = True)
                h = first_exr_file_header['spec'].height
                w = first_exr_file_header['spec'].width

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

            except Exception as e:
                print (f'\nError scanning {folder_path}: {e}')

            return descriptions

        def read_frames_thread(self):
            while not exit_event.is_set():
                self.frame_read_process = torch.multiprocessing.Process(
                    target=self.read_frames,
                    args=(
                        self.frames_queue,
                        list(self.train_descriptions),
                        batch_size,
                        self.h,
                        self.w
                        ),
                    daemon = True
                )
                self.frame_read_process.daemon = True
                self.frame_read_process.start()
                self.frame_read_process.join()
                if not self.sequential:
                    self.reshuffle()

        @staticmethod
        def read_frames(frames_queue, train_descriptions, generalize, self_h, self_w):
            from PIL import Image
            while not process_exit_event.is_set():
                for index in range(len(train_descriptions)):
                    description = train_descriptions[index]
                    train_data = {}
                    train_data['description'] = description

                    try:
                        img0 = read_image_file(description['start'])['image_data']
                        img1 = read_image_file(description['gt'])['image_data']
                        img2 = read_image_file(description['end'])['image_data']

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
                                h_scaled = int(self_h * (1 + 1/8))
                            elif q < 0.75:
                                h_scaled = int(self_h * (1 + 1/7))
                            else:
                                h_scaled = int(self_h * (1 + 1/6))

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
            resized_tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bicubic', align_corners=True, antialias=True)

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

                img0 = torch.from_numpy(img0).to(device = device, dtype = torch.float32)
                img1 = torch.from_numpy(img1).to(device = device, dtype = torch.float32)
                img2 = torch.from_numpy(img2).to(device = device, dtype = torch.float32)

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

                # Convert to ACEScc
                # if random.uniform(0, 1) < (self.acescc_rate / 100):
                #    img0 = self.apply_acescc(torch.clamp(img0, min=0.01))
                #    img1 = self.apply_acescc(torch.clamp(img1, min=0.01))
                #    img2 = self.apply_acescc(torch.clamp(img2, min=0.01))
                
                batch_img0.append(img0)
                batch_img1.append(img1)
                batch_img2.append(img2)

            # del train_data, src_img0, src_img1, src_img2

            return torch.stack(batch_img0), torch.stack(batch_img1), torch.stack(batch_img2), ratio, images_idx, description

    return TimewarpMLDataset(
        data_root, 
        batch_size=batch_size, 
        device=device, 
        frame_size=frame_size, 
        max_window=max_window,
        acescc_rate=acescc_rate,
        generalize=generalize,
        repeat=repeat,
        sequential = sequential
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
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bicubic', padding_mode='border', align_corners=True)

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

class LapLoss(torch.nn.Module):
    def gauss_kernel(self, size=5, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        # kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return torch.nn.functional.interpolate(x, scale_factor= 1. / 2, mode="bilinear", align_corners=False)
        # return x[:, :, ::2, ::2]

    def upsample(self, x):
        return torch.nn.functional.interpolate(x, scale_factor= 2, mode="bilinear", align_corners=False)
        device = x.device
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
        cc = cc.permute(0,1,3,2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
        x_up = cc.permute(0,1,3,2)
        gauss_kernel = self.gauss_kernel(channels=x.shape[1])
        gauss_kernel = gauss_kernel.to(device)
        return self.conv_gauss(x_up, 4*gauss_kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def laplacian_pyramid(self, img, kernel, max_levels=3):
        current = img
        pyr = []
        for level in range(max_levels):
            filtered = self.conv_gauss(current, kernel)
            n, c, h, w = filtered.shape
            sh, sw = round(h * (1 / 2)), round(w * (1 / 2))
            down = torch.nn.functional.interpolate(filtered, size=(sh, sw), mode="bilinear", align_corners=False)
            up = torch.nn.functional.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
            diff = current-up
            pyr.append(diff)
            current = down
        return pyr

    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.maxdepth = 4 * max_levels
        self.gk = self.gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        '''
        n, c, sh, sw = input.shape
        ph = self.maxdepth - (sh % self.maxdepth)
        pw = self.maxdepth - (sw % self.maxdepth)
        padding = (0, pw, 0, ph)
        input = torch.nn.functional.pad(input, padding)
        target = torch.nn.functional.pad(target, padding)
        '''
        
        self.gk = self.gk.to(device = input.device)
        pyr_input  = self.laplacian_pyramid(img=input, kernel=self.gk, max_levels=self.max_levels)
        pyr_target = self.laplacian_pyramid(img=target, kernel=self.gk, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

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

def to_freq(x):
    n, c, h, w = x.shape
    src_dtype = x.dtype
    x = x.float()
    x = torch.fft.fft2(x, dim=(-2, -1))
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.cat([x.real.unsqueeze(2), x.imag.unsqueeze(2)], dim=2).view(n, c * 2, h, w)
    x = x.to(dtype = src_dtype)
    return x

def ap0_to_ap1(image):
    """
    Convert from AP0 (ACES2065-1) to AP1 (ACEScg) for input in [N, 3, H, W] format.
    """
    M = torch.tensor([
        [ 1.45143932, -0.23651075, -0.21492857],
        [-0.07655377,  1.17622970, -0.09967593],
        [ 0.00831615, -0.00603245,  0.99771630]
    ], dtype=image.dtype, device=image.device)

    # Reshape from [N, 3, H, W] -> [N, H, W, 3]
    image = image.permute(0, 2, 3, 1)
    # Apply matrix multiplication
    image = torch.matmul(image, M.T)
    # Reshape back to [N, 3, H, W]
    return image.permute(0, 3, 1, 2)

current_state_dict = {}

'''
def generate_scales(scale=24):
    results = []

    for a in range(scale, 1 - 1, -1):       # First number from 32 down to 2
        for b in range(a, 1 - 1, -1):     # Second number ≤ a
            for c in range(b, 1 - 1, -1): # Third number ≤ b
                for d in range(c, 1 - 1, -1): # Fourth number ≤ c
                    for e in range(d, 1 - 1, -1): # Fifth number ≤ d
                        f = 1  # Sixth number always 1
                        results.append((a, b, c, d, e, f))

    return results

def generate_scales4(scale=24):
    results = []

    for a in range(scale, 1 - 1, -1):       # First number from 32 down to 2
        for b in range(a, 1 - 1, -1):     # Second number ≤ a
            for c in range(b, 1 - 1, -1): # Third number ≤ b
                d = 1
                results.append((a, b, c, d))

    return results
'''

def linear_values(start, count):
    return list(np.linspace(start, 1.0, count))

def optimize_scales(loss_fn, n, max_initial=32.0, search_steps=32):
    current_scales = [1.0] * n
    max_range = max_initial

    for i in range(n - 1):  # Do not optimize last (fixed at 1.0)
        candidates = np.linspace(1.0, max_range, search_steps)
        best_val = None
        best_loss = float('inf')

        for val in candidates:
            tail = linear_values(val, n - i)
            candidate = current_scales[:i] + tail
            loss = loss_fn(candidate)

            if loss < best_loss:
                best_loss = loss
                best_val = val

        max_range = best_val  # shrink max range
        current_scales = current_scales[:i] + linear_values(best_val, n - i)

    return current_scales

'''
def generate_scales(n, scale=24):
    starts = np.linspace(1, scale, n)
    return [linear_values(start, n) for start in starts]
'''

def main():
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
    parser.add_argument('--state_file', type=str, default=None, help='Path to the pre-trained model state dict file (optional)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')
    parser.add_argument('--max', type=int, default=32, help='Starting scale (default: 32)')
    parser.add_argument('--eval_half', action='store_true', dest='eval_half', default=False, help='Evaluate in half-precision')
    parser.add_argument('--eval_trained', action='store_true', dest='eval_trained', default=False, help='Evaluate in half-precision')
    parser.add_argument('--ap0', action='store_true', dest='ap0', default=False, help='input exrs are in ap0')

    args = parser.parse_args()

    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')

    Flownet = None

    # Find and initialize model
    if args.state_file and os.path.isfile(args.state_file):
        trained_model_path = args.state_file
        try:
            checkpoint = torch.load(trained_model_path, map_location=device)
            print('loaded previously saved model checkpoint')
        except Exception as e:
            print (f'unable to load saved model checkpoint: {e}')
            sys.exit()

        model_info = checkpoint.get('model_info')
        model_file = model_info.get('file')
        Flownet = find_and_import_model(model_file=model_file)
    else:
        if not args.state_file:
            print ('Please specify model state file')
            return
        if not os.path.isfile(args.state_file):
            print (f'Model state file {args.state_file} does not exist')
            return

    if Flownet is None:
        print (f'Unable to load model {args.model}')
        return
    
    model_info = Flownet.get_info()
    print ('Model info:')
    pprint (model_info)
    max_dataset_window = 11
    if not model_info.get('ratio_support'):
        max_dataset_window = 3

    if args.eval_trained:
        flownet = Flownet().get_training_model()().to(device)
    else:
        flownet = Flownet().get_model()().to(device)

    print (f'Scanning data for evaluation:')
    eval_dataset = get_dataset(
    args.dataset_path, 
    batch_size=1,
    device=device, 
    frame_size=128,
    max_window=max_dataset_window,
    acescc_rate = 0, # args.acescc,
    generalize=0,
    repeat=1,
    sequential = True
    )

    import signal
    def create_graceful_exit(current_state_dict):
        def graceful_exit(signum, frame):
            # print(f'\nSaving current state to {current_state_dict["trained_model_path"]}...')
            # print (f'Epoch: {current_state_dict["epoch"] + 1}, Step: {current_state_dict["step"]:11}')
            # torch.save(current_state_dict, current_state_dict['trained_model_path'])
            exit_event.set()  # Signal threads to stop
            process_exit_event.set()  # Signal processes to stop
            exit(0)
            # signal.signal(signum, signal.SIG_DFL)
            # os.kill(os.getpid(), signal.SIGINT)
        return graceful_exit
    signal.signal(signal.SIGINT, create_graceful_exit(current_state_dict))

    def exeption_handler(exctype, value, tb):
        exit_event.set()
        process_exit_event.set()
        sys.__excepthook__(exctype, value, tb)
    sys.excepthook = exeption_handler

    # LPIPS Init

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    import lpips
    os.environ['TORCH_HOME'] = os.path.abspath(os.path.dirname(__file__))
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex.to(device)
    # loss_fn_lpips = lpips.LPIPS(net='vgg', lpips=False, spatial=True)
    # loss_fn_lpips.to(device)

    # loss_fn_ssim = SSIM()
    # loss_fn_vgg = VGGPerceptualLoss().to(device)

    warnings.resetwarnings()

    trained_model_path = args.state_file
    try:
        checkpoint = torch.load(trained_model_path, map_location=device)
        missing_keys, unexpected_keys = flownet.load_state_dict(checkpoint['flownet_state_dict'], strict=False)
        print (f'loaded saved model state dict')
    except Exception as e:
        print (f'unable to load saved model: {e}')

    '''
    if args.eval_trained:
        scales_list = generate_scales(4, scale=args.max)
    else:
        scales_list = generate_scales(6, scale=args.max)

    print(f"Generated {len(scales_list)} scale sequences.")

    scales_list.reverse()
    '''

    dataset_dirname = os.path.basename(os.path.abspath(args.dataset_path))

    csv_filename = f'{os.path.splitext(trained_model_path)[0]}.scale.{dataset_dirname}.csv'

    if not os.path.isfile(csv_filename):
        create_csv_file(
            csv_filename,
            [
                'Loss',
                'LPIPS',
                'Combined',
                'Scale',
            ]
        )

    criterion_l1 = torch.nn.L1Loss()

    print('\n\n')

    evalnet = flownet
    for param in evalnet.parameters():
        param.requires_grad = False
    if args.eval_half:
        evalnet.half()
    evalnet.eval()

    # for idx, scale in enumerate(scales_list):
    def example_loss(scales_list):
        scales_list = [round(v, 1) for v in scales_list]
        descriptions = list(eval_dataset.initial_train_descriptions)

        def read_eval_images(read_eval_image_queue, descriptions):
            for ev_item_index, description in enumerate(descriptions):
                try:
                    desc_data = dict(description)
                    eval_img0 = read_image_file(description['start'])['image_data']
                    eval_img1 = read_image_file(description['gt'])['image_data']
                    eval_img2 = read_image_file(description['end'])['image_data']

                    desc_data['eval_img0'] = eval_img0
                    desc_data['eval_img1'] = eval_img1
                    desc_data['eval_img2'] = eval_img2

                    desc_data['ev_item_index'] = ev_item_index
                    read_eval_image_queue.put(desc_data)
                    del desc_data
                
                except Exception as e:
                    pprint (f'\nerror while reading eval images: {e}\n{description}\n\n')
            read_eval_image_queue.put(None)

        read_eval_image_queue = queue.Queue(maxsize=4)
        read_eval_thread = threading.Thread(target=read_eval_images, args=(read_eval_image_queue, descriptions))
        read_eval_thread.daemon = True
        read_eval_thread.start()
        eval_loss = []
        eval_lpips = []

        try:
            with torch.no_grad():
                description = read_eval_image_queue.get()
                while description is not None:
                    ev_item_index = description['ev_item_index']
                    if eval_loss:
                        eval_loss_avg = float(np.array(eval_loss).mean())
                    else:
                        eval_loss_avg = -1
                    if eval_lpips:
                        eval_lpips_mean = float(np.array(eval_lpips).mean())
                    else:
                        eval_lpips_mean = -1

                    clear_lines(1)
                    print (f'\rScale: {scales_list}, Evaluating {ev_item_index+1} of {len(descriptions)}: Avg L1: {eval_loss_avg:.6f}, LPIPS: {eval_lpips_mean:.4f}')

                    eval_img0 = description['eval_img0']
                    eval_img1 = description['eval_img1']
                    eval_img2 = description['eval_img2']
                    eval_ratio = description['ratio']
                    eval_img0 = torch.from_numpy(eval_img0)
                    eval_img1 = torch.from_numpy(eval_img1)
                    eval_img2 = torch.from_numpy(eval_img2)
                    eval_img0 = eval_img0.to(device = device, dtype = torch.float32, non_blocking = True)
                    eval_img1 = eval_img1.to(device = device, dtype = torch.float32, non_blocking = True)
                    eval_img2 = eval_img2.to(device = device, dtype = torch.float32, non_blocking = True)
                    eval_img0 = eval_img0.permute(2, 0, 1).unsqueeze(0)
                    eval_img1 = eval_img1.permute(2, 0, 1).unsqueeze(0)
                    eval_img2 = eval_img2.permute(2, 0, 1).unsqueeze(0)
                    eval_img0_orig = eval_img0.clone()
                    eval_img2_orig = eval_img2.clone()

                    if args.eval_half:
                        eval_img0 = eval_img0.half()
                        eval_img2 = eval_img2.half()

                    result = evalnet(
                        eval_img0, 
                        eval_img2,
                        eval_ratio,
                        scale = scales_list,
                        iterations = 1
                        )

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elif torch.backends.mps.is_available():
                        torch.mps.synchronize()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()            
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()


                    eval_flow_list = result['flow_list']
                    eval_mask_list = result['mask_list']
                    eval_conf_list = result['conf_list']
                    eval_merged = result['merged']

                    if args.eval_half:
                        eval_flow_list[-1] = eval_flow_list[-1].float()
                        eval_mask_list[-1] = eval_mask_list[-1].float()

                    eval_result = warp(eval_img0_orig, eval_flow_list[-1][:, :2, :, :]) * eval_mask_list[-1][:, :, :, :] + warp(eval_img2_orig, eval_flow_list[-1][:, 2:4, :, :]) * (1 - eval_mask_list[-1][:, :, :, :])
                    eval_loss_l1 = criterion_l1(eval_result, eval_img1)
                    eval_loss.append(float(eval_loss_l1.item()))
                    eval_loss_LPIPS = loss_fn_alex(eval_result * 2 - 1, eval_img1 * 2 - 1)
                    eval_lpips.append(float(torch.mean(eval_loss_LPIPS).item()))

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elif torch.backends.mps.is_available():
                        torch.mps.synchronize()

                    del eval_img0, eval_img1, eval_img2, eval_img0_orig, eval_img2_orig
                    del eval_flow_list, eval_mask_list, eval_conf_list, eval_merged
                    del result, eval_result,
                    del description['eval_img0'], description['eval_img1'], description['eval_img2']

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()            
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                    description = read_eval_image_queue.get()

            eval_loss_avg = float(np.array(eval_loss).mean())
            eval_lpips_mean = float(np.array(eval_lpips).mean())

            eval_rows_to_append = [
                {
                    'Loss': eval_loss_avg,
                    'LPIPS': eval_lpips_mean,
                    'Combined': (eval_loss_avg + 2e-1 * eval_lpips_mean),
                    'Scale': scales_list, 
                }
            ]

            for eval_row in eval_rows_to_append:
                append_row_to_csv(csv_filename, eval_row)

            clear_lines(2)
            print(f'\rScale {scales_list}\t\tAvg L1: {eval_loss_avg:.6f}, LPIPS: {eval_lpips_mean:.4f}')
            print ('\n')

        except Exception as e:
            clear_lines(2)
            print(f'\r[Scale {scales_list} Error: {e}')
            print ('\n')
            while description is not None:
                description = read_eval_image_queue.get()
            read_eval_thread.join()
            return float('inf')

        read_eval_thread.join()
        return (eval_loss_avg + 2e-1 * eval_lpips_mean)

    optimized = optimize_scales(example_loss, n=4, max_initial=args.max, search_steps=args.max)
    optimized = [round(v, 1) for v in optimized]
    print("Optimized scales:", optimized)

if __name__ == "__main__":
    main()

