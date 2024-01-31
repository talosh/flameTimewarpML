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

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

import math

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from models.flownet import FlownetCas
from models.multires_v001 import Model as Model_01


class Yogi(Optimizer):
    r"""Implements Yogi Optimizer Algorithm.
    It has been proposed in `Adaptive methods for Nonconvex Optimization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-2)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 0.001)
        initial_accumulator: initial values for first and
            second moments (default: 1e-6)
        weight_decay: weight decay (L2 penalty) (default: 0)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Yogi(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization  # noqa

    Note:
        Reference code: https://github.com/4rtemi5/Yogi-Optimizer_Keras
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-2,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-3,
        initial_accumulator: float = 1e-6,
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            initial_accumulator=initial_accumulator,
            weight_decay=weight_decay,
        )
        super(Yogi, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Yogi does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                # Followed from official implementation in tensorflow addons:
                # https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/yogi.py#L118 # noqa
                # For more details refer to the discussion:
                # https://github.com/jettify/pytorch-optimizer/issues/77
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = nn.init.constant_(
                        torch.empty_like(
                            p.data, memory_format=torch.preserve_format
                        ),
                        group["initial_accumulator"],
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = nn.init.constant_(
                        torch.empty_like(
                            p.data, memory_format=torch.preserve_format
                        ),
                        group["initial_accumulator"],
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_squared = grad.mul(grad)

                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_squared),
                    grad_squared,
                    value=-(1 - beta2),
                )

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class TimewarpMLDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, batch_size = 8):
        self.fw = flameAppFramework()
        self.data_root = data_root
        self.batch_size = batch_size
        
        print (f'scanning for exr files in {self.data_root}...')
        self.folders_with_exr = self.find_folders_with_exr(data_root)
        print (f'found {len(self.folders_with_exr)} clip folders.')
        
        '''
        folders_with_descriptions = set()
        folders_to_scan = set()
        if not rescan:
            print (f'scanning dataset description files...')
            folders_with_descriptions, folders_to_scan = self.scan_dataset_descriptions(
                self.folders_with_exr,
                file_name='dataset_folder.json'
                )
            print (f'found {len(folders_with_descriptions)} pre-processed folders, {len(folders_to_scan)} folders to scan.')
        else:
            folders_to_scan = self.folders_with_exr
        '''

        self.train_descriptions = []

        for folder_index, folder_path in enumerate(sorted(self.folders_with_exr)):
            print (f'\rBuilding train data from clip {folder_index + 1} of {len(self.folders_with_exr)}', end='')
            self.train_descriptions.extend(self.create_dataset_descriptions(folder_path))
        print ('\nReshuffling train data...')

        random.shuffle(self.train_descriptions)

        self.h = 256
        self.w = 256
        # self.frame_multiplier = (self.src_w // self.w) * (self.src_h // self.h) * 4

        self.frames_queue = queue.Queue(maxsize=4)
        self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
        self.frame_read_thread.daemon = True
        self.frame_read_thread.start()

        # self.last_shuffled_index = -1
        # self.last_source_image_data = None
        # self.last_target_image_data = None

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
        for root, dirs, files in os.walk(path):
            if root.endswith('preview'):
                continue
            for file in files:
                if file.endswith('.exr'):
                    directories_with_exr.add(root)
                    break  # No need to check other files in the same directory

        return directories_with_exr

    def scan_dataset_descriptions(self, folders, file_name='dataset_folder.json'):
        """
        Scan folders for the presence of a specific file and categorize them.

        Parameters:
        folders (set): A set of folder paths to check.
        file_name (str, optional): The name of the file to look for in each folder. Defaults to 'twml_dataset_folder.json'.

        Returns:
        tuple of (set, set): Two sets, the first contains folders where the file exists, the second contains folders where it does not.
        """
        folders_with_file = set()
        folders_without_file = set()

        for folder in folders:
            # Construct the full path to the file
            file_path = os.path.join(folder, file_name)

            # Check if the file exists in the folder
            if os.path.exists(file_path):
                folders_with_file.add(folder)
            else:
                folders_without_file.add(folder)

        return folders_with_file, folders_without_file

    def create_dataset_descriptions(self, folder_path, max_window=4):

        def sliding_window(lst, n):
            for i in range(len(lst) - n + 1):
                yield lst[i:i + n]

        exr_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.exr')]
        exr_files.sort()

        if len(exr_files) < max_window:
            max_window = len(exr_files)
        if max_window < 3:
            print(f'\nminimum clip length is 3 frames, {folder_path} has {len(exr_files)}')
            return
        
        descriptions = []

        if 'fast' in os.path.basename(folder_path):
            max_window = max_window if max_window < 4 else 4
        
        try:
            first_exr_file_header = self.fw.read_openexr_file(exr_files[0], header_only = True)
            h = first_exr_file_header['shape'][0]
            w = first_exr_file_header['shape'][1]

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
                            'pre_start': exr_files[max(start_frame_index - 1, 0)],
                            'start': start_frame,
                            'gt': gt_frame,
                            'end': end_frame,
                            'after_end': exr_files[min(end_frame_index + 1, len(exr_files) - 1)],
                            'ratio': 1 / (len(window) - 1) * (gt_frame_index + 1)
                        }

                        bw_item = {
                            'h': h,
                            'w': w,
                            'pre_start': exr_files[min(end_frame_index + 1, len(exr_files) - 1)],
                            'start': end_frame,
                            'gt': gt_frame,
                            'end': start_frame,
                            'after_end': exr_files[max(start_frame_index - 1, 0)],
                            'ratio': 1 - (1 / (len(window) - 1) * (gt_frame_index + 1))
                        }

                        descriptions.append(fw_item)
                        descriptions.append(bw_item)

        except Exception as e:
            print (f'\nError scanning {folder_path}: {e}')

        return descriptions
        
    def read_frames_thread(self):
        timeout = 1e-8
        while True:
            for index in range(len(self.train_descriptions)):
                description = self.train_descriptions[index]
                try:
                    train_data = {}
                    train_data['pre_start'] = self.fw.read_openexr_file(description['pre_start'])['image_data']
                    train_data['start'] = self.fw.read_openexr_file(description['start'])['image_data']
                    train_data['gt'] = self.fw.read_openexr_file(description['gt'])['image_data']
                    train_data['end'] = self.fw.read_openexr_file(description['end'])['image_data']
                    train_data['after_end'] = self.fw.read_openexr_file(description['after_end'])['image_data']
                    train_data['ratio'] = description['ratio']
                    train_data['h'] = description['h']
                    train_data['w'] = description['w']
                    train_data['description'] = description
                    self.frames_queue.put(train_data)
                except Exception as e:
                    print (e)                
            time.sleep(timeout)

    def __len__(self):
        return len(self.train_descriptions)
    
    def crop(self, img0, img1, img2, img3, img4, h, w):
        np.random.seed(None)
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        img2 = img2[x:x+h, y:y+w, :]
        img3 = img3[x:x+h, y:y+w, :]
        img4 = img4[x:x+h, y:y+w, :]
        return img0, img1, img2, img3, img4

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

        return resized_tensor

    def getimg(self, index):
        '''
        shuffled_index = self.indices[index // self.frame_multiplier]
        if shuffled_index != self.last_shuffled_index:
            self.last_source_image_data, self.last_target_image_data = self.frames_queue.get()
            self.last_shuffled_index = shuffled_index
        
        return self.last_source_image_data, self.last_target_image_data
        '''
        return self.frames_queue.get()

    def __getitem__(self, index):
        train_data = self.getimg(index)
        src_img0 = train_data['pre_start']
        src_img1 = train_data['start']
        src_img2 = train_data['gt']
        src_img3 = train_data['end']
        src_img4 = train_data['after_end']
        imgh = train_data['h']
        imgw = train_data['w']
        ratio = train_data['ratio']

        device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda')

        src_img0 = torch.from_numpy(src_img0.copy())
        src_img1 = torch.from_numpy(src_img1.copy())
        src_img2 = torch.from_numpy(src_img2.copy())
        src_img3 = torch.from_numpy(src_img3.copy())
        src_img4 = torch.from_numpy(src_img4.copy())
        src_img0 = src_img0.to(device = device, dtype = torch.float32)
        src_img1 = src_img1.to(device = device, dtype = torch.float32)
        src_img2 = src_img2.to(device = device, dtype = torch.float32)
        src_img3 = src_img3.to(device = device, dtype = torch.float32)
        src_img4 = src_img4.to(device = device, dtype = torch.float32)

        batch_img0 = []
        batch_img1 = []
        batch_img2 = []
        batch_img3 = []
        batch_img4 = []

        for index in range(self.batch_size):
            q = random.uniform(0, 1)
            if q < 0.5:
                rsz_img0 = self.resize_image(src_img0, self.h)
                rsz_img1 = self.resize_image(src_img1, self.h)
                rsz_img2 = self.resize_image(src_img2, self.h)
                rsz_img3 = self.resize_image(src_img3, self.h)
                rsz_img4 = self.resize_image(src_img4, self.h)
                img0, img1, img2, img3, img4 = self.crop(rsz_img0, rsz_img1, rsz_img2, rsz_img3, rsz_img4, self.h, self.w)
                img0 = img0.permute(2, 0, 1)
                img1 = img1.permute(2, 0, 1)
                img2 = img2.permute(2, 0, 1)
                img3 = img3.permute(2, 0, 1)
                img4 = img4.permute(2, 0, 1)
            elif q < 0.75:
                rsz_img0 = self.resize_image(src_img0, self.h * 2)
                rsz_img1 = self.resize_image(src_img1, self.h * 2)
                rsz_img2 = self.resize_image(src_img2, self.h * 2)
                rsz_img3 = self.resize_image(src_img3, self.h * 2)
                rsz_img4 = self.resize_image(src_img4, self.h * 2)
                img0, img1, img2, img3, img4 = self.crop(rsz_img0, rsz_img1, rsz_img2, rsz_img3, rsz_img4, self.h, self.w)
                img0 = img0.permute(2, 0, 1)
                img1 = img1.permute(2, 0, 1)
                img2 = img2.permute(2, 0, 1)
                img3 = img3.permute(2, 0, 1)
                img4 = img4.permute(2, 0, 1)
            else:
                rsz_img0 = self.resize_image(src_img0, self.h * 3)
                rsz_img1 = self.resize_image(src_img1, self.h * 3)
                rsz_img2 = self.resize_image(src_img2, self.h * 3)
                rsz_img3 = self.resize_image(src_img3, self.h * 3)
                rsz_img4 = self.resize_image(src_img4, self.h * 3)
                img0, img1, img2, img3, img4 = self.crop(rsz_img0, rsz_img1, rsz_img2, rsz_img3, rsz_img4, self.h, self.w)
                img0 = img0.permute(2, 0, 1)
                img1 = img1.permute(2, 0, 1)
                img2 = img2.permute(2, 0, 1)
                img3 = img3.permute(2, 0, 1)
                img4 = img4.permute(2, 0, 1)

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = torch.flip(img0.transpose(1, 2), [2])
                img1 = torch.flip(img1.transpose(1, 2), [2])
                img2 = torch.flip(img2.transpose(1, 2), [2])
                img3 = torch.flip(img3.transpose(1, 2), [2])
                img4 = torch.flip(img4.transpose(1, 2), [2])
            elif p < 0.5:
                img0 = torch.flip(img0, [1, 2])
                img1 = torch.flip(img1, [1, 2])
                img2 = torch.flip(img2, [1, 2])
                img3 = torch.flip(img3, [1, 2])
                img4 = torch.flip(img4, [1, 2])
            elif p < 0.75:
                img0 = torch.flip(img0.transpose(1, 2), [1])
                img1 = torch.flip(img1.transpose(1, 2), [1])
                img2 = torch.flip(img2.transpose(1, 2), [1])
                img3 = torch.flip(img3.transpose(1, 2), [1])
                img4 = torch.flip(img4.transpose(1, 2), [1])

            # Horizontal flip (reverse width)
            if random.uniform(0, 1) < 0.5:
                img0 = img0.flip(-1)
                img1 = img1.flip(-1)
                img2 = img2.flip(-1)
                img3 = img3.flip(-1)
                img4 = img4.flip(-1)

            # Vertical flip (reverse height)
            if random.uniform(0, 1) < 0.5:
                img0 = img0.flip(-2)
                img1 = img1.flip(-2)
                img2 = img2.flip(-2)
                img3 = img3.flip(-2)
                img4 = img4.flip(-2)

            # Depth-wise flip (reverse channels)
            if random.uniform(0, 1) < 0.4:
                img0 = img0.flip(0)
                img1 = img1.flip(0)
                img2 = img2.flip(0)
                img3 = img3.flip(0)
                img4 = img4.flip(0)

            # img0, img1 = self.crop(img0, img1, self.h, self.w)
            # img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            # img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            batch_img0.append(img0)
            batch_img1.append(img1)
            batch_img2.append(img2)
            batch_img3.append(img3)
            batch_img4.append(img4)

        return torch.stack(batch_img0), torch.stack(batch_img1), torch.stack(batch_img2), torch.stack(batch_img3), torch.stack(batch_img4), ratio

    def get_input_channels_number(self, source_frames_paths_list):
        total_num_channels = 0
        for src_path in source_frames_paths_list:
            file_header = self.fw.read_openexr_file(src_path, header_only=True)
            total_num_channels += file_header['shape'][2]
        return total_num_channels

def write_exr(image_data, filename, half_float = False, pixelAspectRatio = 1.0):
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

def main():
    parser = argparse.ArgumentParser(description='Training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')

    # Optional arguments
    parser.add_argument('--lr', type=float, default=0.00009, help='Learning rate (default: 9e-5)')
    parser.add_argument('--type', type=int, default=1, help='Model type (int): 1 - MultiresNet, 2 - MultiresNet 4 (default: 1)')
    parser.add_argument('--warmup', type=float, default=0.001, help='Warmup epochs (float) (default: 1)')
    parser.add_argument('--pulse', type=float, default=9, help='Period in number of epochs to pulse learning rate (float) (default: 9)')
    parser.add_argument('--pulse_amplitude', type=float, default=10, help='Learning rate pulse amplitude (percentage) (default: 10)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model (optional)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (int) (default: 8)')

    # parser.add_argument('--rescan', action='store_true', help='Rescan the dataset (default: False)')

    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.dataset_path, 'preview')):
        os.makedirs(os.path.join(args.dataset_path, 'preview'))

    read_image_queue = queue.Queue(maxsize=12)
    dataset = TimewarpMLDataset(args.dataset_path, batch_size=args.batch_size)

    def read_images(read_image_queue, dataset):
        while True:
            for batch_idx in range(len(dataset)):
                img0, img1, img2, img3, img4, ratio = dataset[batch_idx]
                read_image_queue.put((img0, img1, img2, img3, img4, ratio))

    read_thread = threading.Thread(target=read_images, args=(read_image_queue, dataset))
    read_thread.daemon = True
    read_thread.start()

    steps_per_epoch = len(dataset)
    
    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')
    '''
    device = torch.device(f'cuda:{args.device}')
    device = torch.device("mps")
    '''
    
    '''
    if args.type == 1:
        model_name = Model_01.get_name()
        model = Model_01().get_training_model()(dataset.in_channles, dataset.out_channels).to(device)
    elif args.type == 2:
        model_name = Model_02.get_name()
        model = Model_02().get_training_model()(dataset.in_channles, dataset.out_channels).to(device)
    else:
        print (f'Model type {args.type} is not yet implemented')
        sys.exit()
    '''


    model = FlownetCas().to(device)
    model_name = FlownetCas

    fusion_model_name = Model_01.get_name()
    fusion_model = Model_01().get_training_model()(15, 3).to(device)

    warmup_epochs = args.warmup
    pulse_dive = args.pulse_amplitude
    pulse_period = args.pulse
    lr = args.lr
    number_warmup_steps = steps_per_epoch * warmup_epochs

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()

    criterion_mse_fusion = torch.nn.MSELoss()
    criterion_l1_fusion = torch.nn.L1Loss()

    # optimizer_sgd = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = Yogi(model.parameters(), lr=lr)
    optimizer_fusion = Yogi(model.parameters(), lr=lr)

    def warmup(current_step, lr = 4e-3, number_warmup_steps = 999):
        mul_lin = current_step / number_warmup_steps
        # print (f'\n number_warmup_steps {number_warmup_steps} lr {lr} mul {mul_lin} res {(lr * mul_lin):.4e}')
        return lr * mul_lin

    # remove annoying message in pytorch 1.12.1 when using CosineAnnealingLR
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * pulse_period, eta_min = lr - (( lr / 100 ) * pulse_dive) )
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warmup(step, lr=lr, number_warmup_steps=( steps_per_epoch * warmup_epochs )))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [steps_per_epoch * warmup_epochs])

    # Rest of your training script...

    step = 0
    current_epoch = 0
    preview_index = 0

    steps_loss = []
    epoch_loss = []

    if args.model_path:
        trained_model_path = args.model_path
        try:
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print('loaded previously saved model')
        except Exception as e:
            print (f'unable to load saved model: {e}')

        try:
            fusion_model.load_state_dict(checkpoint['fusion_state_dict'])
            print('loaded previously saved fusion model')
        except Exception as e:
            print (f'unable to load saved fusion model: {e}')

        try:
            step = checkpoint['step']
            print (f'step: {step}')
            current_epoch = checkpoint['epoch']
            print (f'epoch: {current_epoch + 1}')
        except Exception as e:
            print (f'unable to set step and epoch: {e}')
        try:
            steps_loss = checkpoint['steps_loss']
            print (f'loaded loss statistics for step: {step}')
            epoch_loss = checkpoint['epoch_loss']
            print (f'loaded loss statistics for epoch: {current_epoch + 1}')
        except Exception as e:
            print (f'unable to load step and epoch loss statistics: {e}')
    else:
        traned_model_name = 'flameTWML_model_' + fw.create_timestamp_uid() + '.pth'
        if platform.system() == 'Darwin':
            trained_model_dir = os.path.join(
                os.path.expanduser('~'),
                'Documents',
                'flameTWML_models')
        else:
            trained_model_dir = os.path.join(
                os.path.expanduser('~'),
                'flameTWML_models')
        if not os.path.isdir(trained_model_dir):
            os.makedirs(trained_model_dir)
        trained_model_path = os.path.join(trained_model_dir, traned_model_name)

    try:
        start_timestamp = checkpoint.get('start_timestamp')
    except:
        start_timestamp = time.time()

    time_stamp = time.time()
    epoch = current_epoch
    
    '''
    state_dict = torch.load('/home/flame/Documents/dev/flameTimewarpML/models_data/flownet_v412.pkl')
    def convert(param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }
    model.load_state_dict(convert(state_dict))
    '''

    while True:
        for batch_idx in range(len(dataset)):
            data_time = time.time() - time_stamp
            time_stamp = time.time()

            img0, img1, img2, img3, img4, ratio = read_image_queue.get()

            if platform.system() == 'Darwin':
                img0 = normalize_numpy(img0)
                img1 = normalize_numpy(img1)
                img2 = normalize_numpy(img2)
                img3 = normalize_numpy(img3)
                img4 = normalize_numpy(img4)
                img0 = img0.to(device, non_blocking = True)
                img1 = img1.to(device, non_blocking = True)
                img2 = img2.to(device, non_blocking = True)
                img3 = img3.to(device, non_blocking = True)
                img4 = img4.to(device, non_blocking = True)
            else:
                img0 = img0.to(device, non_blocking = True)
                img1 = img1.to(device, non_blocking = True)
                img2 = img2.to(device, non_blocking = True)
                img3 = img3.to(device, non_blocking = True)
                img4 = img4.to(device, non_blocking = True)
                img0 = normalize(img0)
                img1 = normalize(img1)
                img2 = normalize(img2)
                img3 = normalize(img3)
                img4 = normalize(img4)

            if step < number_warmup_steps:
                current_lr = warmup(step, lr=lr, number_warmup_steps=number_warmup_steps)
            else:
                current_lr = scheduler.get_last_lr()[0]
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            current_lr_str = str(f'{optimizer.param_groups[0]["lr"]:.4e}')

            optimizer.zero_grad(set_to_none=True)

            x = torch.cat((img1, img3, img2), dim=1)
            flow_list, mask, merged, teacher_res, loss_cons = model(x, timestep = ratio)
            output = merged[3]

            timestep = (img1[:, :1].clone() * 0 + 1) * ratio
            flow = flow_list[3]
            fusion_input = torch.cat((img1*2 - 1, flow[:, :2], mask*2 - 1, timestep, output*2 - 1, flow[:, 2:4], img3*2 - 1), dim=1)
            output = fusion_model(fusion_input)  
            output = ( output + 1 ) / 2
            
            loss = criterion_mse(output, img2)
            loss_l1 = criterion_l1(output, img2)
            loss_l1_str = str(f'{loss_l1.item():.6f}')

            epoch_loss.append(float(loss_l1))
            steps_loss.append(float(loss_l1))

            loss.backward()
            optimizer.step()

            '''
            optimizer_fusion.zero_grad(set_to_none=True)
            
            flow = flow_list[3].detach().clone()
            img1_f = img1.detach().clone()
            output_f = output.detach().clone()
            mask_f = mask.detach().clone()
            img2_f = img2.detach().clone()
            img3_f = img3.detach().clone()
            timestep = (img1_f[:, :1].clone() * 0 + 1) * ratio

            fusion_input = torch.cat((img1_f*2 - 1, flow[:, :2], mask_f*2 - 1, timestep, output_f*2 - 1, flow[:, 2:4], img3_f*2 - 1), dim=1)
            output_fusion = fusion_model(fusion_input)
            output_fusion = ( output_fusion + 1 ) / 2
            loss_fusion = criterion_mse_fusion(output_fusion, img2_f)
            loss_l1_fusion = criterion_l1_fusion(output_fusion, img2_f)
            loss_l1_str_fusion = str(f'{loss_l1_fusion.item():.6f}')
            loss_fusion.backward()
            optimizer_fusion.step()
            '''

            scheduler.step()


            train_time = time.time() - time_stamp
            time_stamp = time.time()

            if step % 40 == 1:
                if platform.system() == 'Darwin':
                    rgb_source1 = restore_normalized_values_numpy(img1)
                    rgb_source2 = restore_normalized_values_numpy(img3)
                    rgb_target = restore_normalized_values_numpy(img2)
                    rgb_output = restore_normalized_values_numpy(output)
                    rgb_output_fusion = restore_normalized_values_numpy(output_fusion)
                else:
                    rgb_source1 = restore_normalized_values(img1)
                    rgb_source2 = restore_normalized_values(img3)
                    rgb_target = restore_normalized_values(img2)
                    rgb_output = restore_normalized_values(output)
                    rgb_output_fusion = restore_normalized_values(output_fusion)

                preview_folder = os.path.join(args.dataset_path, 'preview')
                sample_source1 = rgb_source1[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                sample_source2 = rgb_source2[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                sample_target = rgb_target[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                sample_output = rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                sample_output_fusion = rgb_output_fusion[0].clone().cpu().detach().numpy().transpose(1, 2, 0)

                write_exr(sample_source1, os.path.join(preview_folder, f'{preview_index:02}_incomng.exr'))
                write_exr(sample_source2, os.path.join(preview_folder, f'{preview_index:02}_outgoing.exr'))
                write_exr(sample_target, os.path.join(preview_folder, f'{preview_index:02}_target.exr'))
                write_exr(sample_output, os.path.join(preview_folder, f'{preview_index:02}_output.exr'))
                write_exr(sample_output_fusion, os.path.join(preview_folder, f'{preview_index:02}_output_fusion.exr'))

                preview_index = preview_index + 1 if preview_index < 9 else 0

                # sample_current = rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0)

            if step % 100 == 1:
                torch.save({
                    'step': step,
                    'steps_loss': steps_loss,
                    'epoch': epoch,
                    'epoch_loss': epoch_loss,
                    'start_timestamp': start_timestamp,
                    # 'batch_idx': batch_idx,
                    'lr': optimizer.param_groups[0]['lr'],
                    'model_state_dict': model.state_dict(),
                    'fusion_state_dict': fusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_fusion_state_dict': optimizer_fusion.state_dict(),
                    'model_name': model_name,
                    'fusion_model_name': fusion_model_name,
                }, trained_model_path)

            data_time += time.time() - time_stamp
            data_time_str = str(f'{data_time:.2f}')
            train_time_str = str(f'{train_time:.2f}')
            # current_lr_str = str(f'{optimizer.param_groups[0]["lr"]:.4e}')
            # current_lr_str = str(f'{scheduler.get_last_lr()[0]:.4e}')

            epoch_time = time.time() - start_timestamp
            days = int(epoch_time // (24 * 3600))
            hours = int((epoch_time % (24 * 3600)) // 3600)
            minutes = int((epoch_time % 3600) // 60)

            smoothed_loss = np.mean(moving_average(epoch_loss, 9))

            print (f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Time:{data_time_str} + {train_time_str}, Batch [{batch_idx + 1} / {len(dataset)}], Lr: {current_lr_str}, Loss L1: {loss_l1_str}', end='')
            step = step + 1

        torch.save({
            'step': step,
            'steps_loss': steps_loss,
            'epoch': epoch,
            'epoch_loss': epoch_loss,
            'start_timestamp': start_timestamp,
            # 'batch_idx': batch_idx,
            'lr': optimizer.param_groups[0]['lr'],
            'model_state_dict': model.state_dict(),
            'fusion_state_dict': fusion_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_fusion_state_dict': optimizer_fusion.state_dict(),
            'model_name': model_name,
            'fusion_model_name': fusion_model_name,
        }, trained_model_path)
        
        smoothed_loss = np.mean(moving_average(epoch_loss, 9))
        epoch_time = time.time() - start_timestamp
        days = int(epoch_time // (24 * 3600))
        hours = int((epoch_time % (24 * 3600)) // 3600)
        minutes = int((epoch_time % 3600) // 60)
        print (f'\r {" "*120}', end='')
        print(f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f}')
        steps_loss = []
        epoch_loss = []
        epoch = epoch + 1

if __name__ == "__main__":
    main()

