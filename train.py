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
    def __init__(self, data_root):
        self.fw = flameAppFramework()
        self.data_root = data_root

        print (data_root)

        folders_with_exr = self.find_folders_with_exr(data_root)

        pprint (folders_with_exr)

        sys.exit()

        def exr_files_in_folder(folder_path):
            """
            Check if there are any .exr files in the specified folder.

            Parameters:
            folder_path (str): Path to the folder to be checked.

            Returns:
            bool: True if there are .exr files in the folder, False otherwise.
            """
            for file in os.listdir(folder_path):
                if file.endswith('.exr'):
                    return True
            return False


        self.source_root = os.path.join(self.data_root, 'source')
        self.target_root = os.path.join(self.data_root, 'target')

        if exr_files_in_folder(self.source_root):
            self.source_files = [[os.path.join(self.source_root, file)] for file in sorted(os.listdir(self.source_root))]
        else:
            self.source_files = []
            source_files_map = self.create_source_files_map(self.source_root)
            for key in sorted(source_files_map.keys()):
                self.source_files.append(source_files_map[key])

        self.target_files = [os.path.join(self.target_root, file) for file in sorted(os.listdir(self.target_root))]
        self.indices = list(range(len(self.source_files)))

        try:
            src_header = self.fw.read_openexr_file(self.source_files[0][0], header_only=True)
        except Exception as e:
            print (f'Unable to read {self.source_files[0]}: {e}')
            sys.exit()
        
        self.src_h = src_header['shape'][0]
        self.src_w = src_header['shape'][1]
        self.in_channles = self.get_input_channels_number(self.source_files[0])
        print (f'source channels: {self.in_channles}')

        try:
            target_header = self.fw.read_openexr_file(self.target_files[0], header_only=True)
        except Exception as e:
            print (f'Unable to read {self.source_files[0]}: {e}')
            sys.exit()

        self.out_channels = target_header['shape'][2]
        print (f'target channels: {self.out_channels}')

        self.h = 256
        self.w = 256
        self.frame_multiplier = (self.src_w // self.w) * (self.src_h // self.h) * 4

        self.frames_queue = queue.Queue(maxsize=4)
        self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
        self.frame_read_thread.daemon = True
        self.frame_read_thread.start()

        self.last_shuffled_index = -1
        self.last_source_image_data = None
        self.last_target_image_data = None

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
            for file in files:
                if file.endswith('.exr'):
                    directories_with_exr.add(root)
                    break  # No need to check other files in the same directory

        return directories_with_exr

    def read_frames_thread(self):
        timeout = 1e-8
        while True:
            for index in range(len(self.source_files)):
                source_file_paths_list = self.source_files[index]
                target_file_path = self.target_files[index]
                source_image_data = None
                target_image_data = None

                try:
                    tensors = []
                    for src_path in source_file_paths_list:
                        src_image_dict = self.fw.read_openexr_file(src_path)
                        tensors.append(src_image_dict.get('image_data').astype(np.float32))
                    source_image_data = np.concatenate(tensors, axis=2)
                except Exception as e:
                    print (e)

                try:
                    target_image_dict = self.fw.read_openexr_file(target_file_path)
                    target_image_data = target_image_dict['image_data'].astype(np.float32)
                except Exception as e:
                    print (e)

                if source_image_data is None or target_image_data is None:
                    time.sleep(timeout)
                    continue
                
                self.frames_queue.put([
                    source_image_data,
                    target_image_data
                ])

            time.sleep(timeout)

    def __len__(self):
        return len(self.source_files) * self.frame_multiplier
    
    def crop(self, img0, img1, h, w):
        np.random.seed(None)
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        return img0, img1

    def getimg(self, index):
        shuffled_index = self.indices[index // self.frame_multiplier]
        
        if shuffled_index != self.last_shuffled_index:
            self.last_source_image_data, self.last_target_image_data = self.frames_queue.get()
            self.last_shuffled_index = shuffled_index
        
        return self.last_source_image_data, self.last_target_image_data

    def __getitem__(self, index):
        img0, img1 = self.getimg(index)

        device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda')

        q = random.uniform(0, 1)
        if q < 0.5:
            img0, img1 = self.crop(img0, img1, self.h, self.w)
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            img0 = img0.to(device)
            img1 = img1.to(device)
        elif q < 0.75:
            img0, img1 = self.crop(img0, img1, self.h // 2, self.w // 2)
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            img0 = img0.to(device)
            img1 = img1.to(device)
            img0 = torch.nn.functional.interpolate(img0.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)[0]
            img1 = torch.nn.functional.interpolate(img1.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)[0]
        else:
            img0, img1 = self.crop(img0, img1, int(self.h * 2), int(self.w * 2))
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            img0 = img0.to(device)
            img1 = img1.to(device)
            img0 = torch.nn.functional.interpolate(img0.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)[0]
            img1 = torch.nn.functional.interpolate(img1.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)[0]

        p = random.uniform(0, 1)
        if p < 0.25:
            img0 = torch.flip(img0.transpose(1, 2), [2])
            img1 = torch.flip(img1.transpose(1, 2), [2])
        elif p < 0.5:
            img0 = torch.flip(img0, [1, 2])
            img1 = torch.flip(img1, [1, 2])
        elif p < 0.75:
            img0 = torch.flip(img0.transpose(1, 2), [1])
            img1 = torch.flip(img1.transpose(1, 2), [1])

        # img0, img1 = self.crop(img0, img1, self.h, self.w)
        # img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        # img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)

        return img0, img1

    def get_input_channels_number(self, source_frames_paths_list):
        total_num_channels = 0
        for src_path in source_frames_paths_list:
            file_header = self.fw.read_openexr_file(src_path, header_only=True)
            total_num_channels += file_header['shape'][2]
        return total_num_channels

    def create_source_files_map(self, folder_path):
        '''
        Creates a dictionary of .exr files from sorted subfolders of a given folder.

        Each key in the dictionary corresponds to an index starting from 1, representing the .exr file's 
        index in the first subfolder. The value is a list containing paths to .exr files from each subfolder,
        where the file's index matches the key. If a subfolder has fewer .exr files than the first one, 
        the last file path in that subfolder is repeated to match the count of the first subfolder.

        Parameters:
        folder_path (str): The path to the main folder containing subfolders.

        Returns:
        dict: A dictionary where each key is an integer starting from 1, and the value is a list of file paths.
            Returns a message string if the folder does not exist or if no subfolders are found.

        Example:
            {1: ['/preview/render1_ML_2024JAN20_1819_HIDH/src/01/render1.00000000.exr',
                '/preview/render1_ML_2024JAN20_1819_HIDH/src/02/render2.00000000.exr',
                '/preview/render1_ML_2024JAN20_1819_HIDH/src/03/004_Subclip_001-RSZ_Result.00100853.exr'],
            2: ['/preview/render1_ML_2024JAN20_1819_HIDH/src/01/render1.00000001.exr',
                '/preview/render1_ML_2024JAN20_1819_HIDH/src/02/render2.00000001.exr',
                '/preview/render1_ML_2024JAN20_1819_HIDH/src/03/004_Subclip_001-RSZ_Result.00100854.exr']}
        '''

        exr_dict = {}

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            message_string = f'Folder {folder_path} does not exist'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )

        # List and sort all subfolders
        subfolders = sorted([f.path for f in os.scandir(folder_path) if f.is_dir()])

        # Check if there are any subfolders
        if not subfolders:
            message_string = f'No clip folders found in {folder_path}'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )

        # Process the first subfolder separately
        first_subfolder = subfolders[0]
        first_subfolder_files = sorted([f for f in os.listdir(first_subfolder) if f.endswith('.exr')])

        # Initialize the dictionary with files from the first subfolder
        for i, file in enumerate(first_subfolder_files, start=1):
            exr_dict[i] = [os.path.join(first_subfolder, file)]

        # Process the remaining subfolders
        for subfolder in subfolders[1:]:
            subfolder_files = sorted([f for f in os.listdir(subfolder) if f.endswith('.exr')])
            for i, file in enumerate(subfolder_files, start=1):
                if i <= len(first_subfolder_files):
                    exr_dict[i].append(os.path.join(subfolder, file))
                else:
                    exr_dict[i].append(os.path.join(subfolder, subfolder_files[-1]))

        return exr_dict

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
    parser.add_argument('--lr', type=float, default=0.0034, help='Learning rate (default: 0.0034)')
    parser.add_argument('--type', type=int, default=1, help='Model type (int): 1 - MultiresNet, 2 - MultiresNet 4 (default: 1)')
    parser.add_argument('--warmup', type=float, default=9, help='Warmup epochs (float) (default: 1)')
    parser.add_argument('--pulse', type=float, default=9, help='Period in number of epochs to pulse learning rate (float) (default: 9)')
    parser.add_argument('--pulse_amplitude', type=float, default=10, help='Learning rate pulse amplitude (percentage) (default: 10)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model (optional)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')

    args = parser.parse_args()

    read_image_queue = queue.Queue(maxsize=12)
    dataset = TimewarpMLDataset(args.dataset_path)

    sys.exit()

    if not os.path.isdir(os.path.join(args.dataset_path, 'source')):
        print (f'dataset {args.dataset_path} must have "source" and "target" folders')
        sys.exit()
    if not os.path.isdir(os.path.join(args.dataset_path, 'target')):
        print (f'dataset {args.dataset_path} must have "source" and "target" folders')
        sys.exit()
    if not os.path.isdir(os.path.join(args.dataset_path, 'preview')):
        os.makedirs(os.path.join(args.dataset_path, 'preview'))


    def read_images(read_image_queue, dataset):
        while True:
            for batch_idx in range(len(dataset)):
                before, after = dataset[batch_idx]
                read_image_queue.put([before, after])

    read_thread = threading.Thread(target=read_images, args=(read_image_queue, dataset))
    read_thread.daemon = True
    read_thread.start()

    steps_per_epoch = len(dataset)
    
    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')
    '''
    device = torch.device(f'cuda:{args.device}')
    device = torch.device("mps")
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

    warmup_epochs = args.warmup
    pulse_dive = args.pulse_amplitude
    pulse_period = args.pulse
    lr = args.lr
    number_warmup_steps = steps_per_epoch * warmup_epochs
    batch_size = 1

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = Yogi(model.parameters(), lr=lr)

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
            step = checkpoint['step']
            print (f'step: {step}')
            current_epoch = checkpoint['epoch']
            print (f'epoch: {current_epoch + 1}')
            # saved_batch_idx = checkpoint['batch_idx']
            # print (f'saved batch index: {saved_batch_idx}')
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
        traned_model_name = 'flameSimpleML_model_' + fw.create_timestamp_uid() + '.pth'
        if platform.system() == 'Darwin':
            trained_model_dir = os.path.join(
                os.path.expanduser('~'),
                'Documents',
                'flameSimpleML_models')
        else:
            trained_model_dir = os.path.join(
                os.path.expanduser('~'),
                'flameSimpleML_models')
        if not os.path.isdir(trained_model_dir):
            os.makedirs(trained_model_dir)
        trained_model_path = os.path.join(trained_model_dir, traned_model_name)

    try:
        start_timestamp = checkpoint.get('start_timestamp')
    except:
        start_timestamp = time.time()

    time_stamp = time.time()
    epoch = current_epoch

    while True:
        for batch_idx in range(len(dataset)):
            data_time = time.time() - time_stamp
            time_stamp = time.time()

            source, target = read_image_queue.get()

            if platform.system() == 'Darwin':
                source = normalize_numpy(source).unsqueeze(0)
                target = normalize_numpy(target).unsqueeze(0)
                source = source.to(device, non_blocking = True)
                target = target.to(device, non_blocking = True)
            else:
                source = source.to(device, non_blocking = True)
                target = target.to(device, non_blocking = True)
                source = normalize(source).unsqueeze(0)
                target = normalize(target).unsqueeze(0)


            if step < number_warmup_steps:
                current_lr = warmup(step, lr=lr, number_warmup_steps=number_warmup_steps)
            else:
                current_lr = scheduler.get_last_lr()[0]
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            current_lr_str = str(f'{optimizer.param_groups[0]["lr"]:.4e}')

            optimizer.zero_grad(set_to_none=True)
            output = model(source * 2 - 1)
            output = ( output + 1 ) / 2

            loss = criterion_mse(output, target)
            loss_l1 = criterion_l1(output, target)
            loss_l1_str = str(f'{loss_l1.item():.6f}')

            epoch_loss.append(float(loss_l1))
            steps_loss.append(float(loss_l1))

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_time = time.time() - time_stamp
            time_stamp = time.time()

            if step % 40 == 1:
                if platform.system() == 'Darwin':
                    rgb_source = restore_normalized_values_numpy(source)
                    rgb_target = restore_normalized_values_numpy(target)
                    rgb_output = restore_normalized_values_numpy(output)
                else:
                    rgb_source = restore_normalized_values(source)
                    rgb_target = restore_normalized_values(target)
                    rgb_output = restore_normalized_values(output)

                preview_folder = os.path.join(args.dataset_path, 'preview')
                sample_source = rgb_source[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                sample_target = rgb_target[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                sample_output = rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                write_exr(sample_source, os.path.join(preview_folder, f'{preview_index:02}_source.exr'))
                write_exr(sample_target, os.path.join(preview_folder, f'{preview_index:02}_target.exr'))
                write_exr(sample_output, os.path.join(preview_folder, f'{preview_index:02}_output.exr'))
                preview_index = preview_index + 1 if preview_index < 9 else 0

                # sample_current = rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0)

            if step % 1000 == 1:
                torch.save({
                    'step': step,
                    'steps_loss': steps_loss,
                    'epoch': epoch,
                    'epoch_loss': epoch_loss,
                    'start_timestamp': start_timestamp,
                    # 'batch_idx': batch_idx,
                    'lr': optimizer.param_groups[0]['lr'],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_name': model_name,
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
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': model_name,
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

