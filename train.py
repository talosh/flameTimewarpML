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

from models.encoder_v001 import Model as Encoder
from models.flownet_v001 import Model as Flownet

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
            print (f'\rBuilding training data from clip {folder_index + 1} of {len(self.folders_with_exr)}', end='')
            self.train_descriptions.extend(self.create_dataset_descriptions(folder_path))
        print ('\nReshuffling training data indices...')

        self.reshuffle()

        self.h = 448
        self.w = 448
        # self.frame_multiplier = (self.src_w // self.w) * (self.src_h // self.h) * 4

        self.frames_queue = queue.Queue(maxsize=12)
        self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
        self.frame_read_thread.daemon = True
        self.frame_read_thread.start()

        print ('reading first block of training data...')
        self.last_train_data = self.frames_queue.get()

        self.repeat_count = 1
        self.repeat_counter = 0

        # self.last_shuffled_index = -1
        # self.last_source_image_data = None
        # self.last_target_image_data = None

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
        for root, dirs, files in os.walk(path):
            if root.endswith('preview'):
                continue
            if root.endswith('eval'):
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
        if 'medium' in folder_path:
            max_window = 5
        
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
                            # 'pre_start': exr_files[max(start_frame_index - 1, 0)],
                            'start': start_frame,
                            'gt': gt_frame,
                            'end': end_frame,
                            # 'after_end': exr_files[min(end_frame_index + 1, len(exr_files) - 1)],
                            'ratio': 1 / (len(window) - 1) * (gt_frame_index + 1)
                        }

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
                    # train_data['pre_start'] = self.fw.read_openexr_file(description['pre_start'])['image_data']
                    train_data['start'] = self.fw.read_openexr_file(description['start'])['image_data']
                    train_data['gt'] = self.fw.read_openexr_file(description['gt'])['image_data']
                    train_data['end'] = self.fw.read_openexr_file(description['end'])['image_data']
                    # train_data['after_end'] = self.fw.read_openexr_file(description['after_end'])['image_data']
                    train_data['ratio'] = description['ratio']
                    train_data['h'] = description['h']
                    train_data['w'] = description['w']
                    train_data['description'] = description
                    train_data['index'] = index
                    self.frames_queue.put(train_data)
                except Exception as e:
                    print (e)                
            time.sleep(timeout)

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

        return resized_tensor

    def getimg(self, index):
        if not self.last_train_data:
            self.last_train_data = self.frames_queue.get()
        '''
        shuffled_index = self.indices[index // self.frame_multiplier]
        if shuffled_index != self.last_shuffled_index:
            self.last_source_image_data, self.last_target_image_data = self.frames_queue.get()
            self.last_shuffled_index = shuffled_index
        
        return self.last_source_image_data, self.last_target_image_data
        '''
        if self.repeat_counter >= self.repeat_count:
            try:
                self.last_train_data = self.frames_queue.get_nowait()
                self.repeat_counter = 0
            except queue.Empty:
                pass

        self.repeat_counter += 1
        return self.last_train_data
        # return self.frames_queue.get()

    def srgb_to_linear(self, srgb_image):
        # Apply the inverse sRGB gamma curve
        mask = srgb_image <= 0.04045
        srgb_image[mask] = srgb_image[mask] / 12.92
        srgb_image[~mask] = ((srgb_image[~mask] + 0.055) / 1.055) ** 2.4

        return srgb_image

    def apply_aces_logc(self, linear_image, middle_grey=0.18, min_exposure=-6.5, max_exposure=6.5):
        """
        Apply the ACES LogC curve to a linear image.

        Parameters:
        linear_image (torch.Tensor): The linear image tensor.
        middle_grey (float): The middle grey value. Default is 0.18.
        min_exposure (float): The minimum exposure value. Default is -6.5.
        max_exposure (float): The maximum exposure value. Default is 6.5.

        Returns:
        torch.Tensor: The image with the ACES LogC curve applied.
        """
        # Constants for the ACES LogC curve
        A = (max_exposure - min_exposure) * 0.18 / middle_grey
        B = min_exposure
        C = math.log2(middle_grey) / 0.18

        # Apply the ACES LogC curve
        logc_image = (torch.log2(linear_image * A + B) + C) / (max_exposure - min_exposure)

        return logc_image

    def __getitem__(self, index):
        train_data = self.getimg(index)
        # src_img0 = train_data['pre_start']
        src_img0 = train_data['start']
        src_img1 = train_data['gt']
        src_img2 = train_data['end']
        # src_img4 = train_data['after_end']
        imgh = train_data['h']
        imgw = train_data['w']
        ratio = train_data['ratio']
        images_idx = train_data['index']

        device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda')

        src_img0 = torch.from_numpy(src_img0.copy())
        src_img1 = torch.from_numpy(src_img1.copy())
        src_img2 = torch.from_numpy(src_img2.copy())
        #src_img3 = torch.from_numpy(src_img3.copy())
        # src_img4 = torch.from_numpy(src_img4.copy())
        src_img0 = src_img0.to(device = device, dtype = torch.float32)
        src_img1 = src_img1.to(device = device, dtype = torch.float32)
        src_img2 = src_img2.to(device = device, dtype = torch.float32)
        # src_img3 = src_img3.to(device = device, dtype = torch.float32)
        # src_img4 = src_img4.to(device = device, dtype = torch.float32)

        rsz1_img0 = self.resize_image(src_img0, self.h)
        rsz1_img1 = self.resize_image(src_img1, self.h)
        rsz1_img2 = self.resize_image(src_img2, self.h)
        # rsz1_img3 = self.resize_image(src_img3, self.h)
        # rsz1_img4 = self.resize_image(src_img4, self.h)

        rsz2_img0 = self.resize_image(src_img0, int(self.h * (1 + 1/4)))
        rsz2_img1 = self.resize_image(src_img1, int(self.h * (1 + 1/4)))
        rsz2_img2 = self.resize_image(src_img2, int(self.h * (1 + 1/4)))
        # rsz2_img3 = self.resize_image(src_img3, int(self.h * (1 + 1/4)))
        # rsz2_img4 = self.resize_image(src_img4, int(self.h * (1 + 1/4)))

        rsz3_img0 = self.resize_image(src_img0, int(self.h * (1 + 1/3)))
        rsz3_img1 = self.resize_image(src_img1, int(self.h * (1 + 1/3)))
        rsz3_img2 = self.resize_image(src_img2, int(self.h * (1 + 1/3)))
        # rsz3_img3 = self.resize_image(src_img3, int(self.h * (1 + 1/3)))
        # rsz3_img4 = self.resize_image(src_img4, int(self.h * (1 + 1/3)))

        rsz4_img0 = self.resize_image(src_img0, int(self.h * (1 + 1/2)))
        rsz4_img1 = self.resize_image(src_img1, int(self.h * (1 + 1/2)))
        rsz4_img2 = self.resize_image(src_img2, int(self.h * (1 + 1/2)))

        batch_img0 = []
        batch_img1 = []
        batch_img2 = []

        for index in range(self.batch_size):
            q = random.uniform(0, 1)
            if q < 0.25:
                img0, img1, img2 = self.crop(rsz1_img0, rsz1_img1, rsz1_img2, self.h, self.w)
            elif q < 0.5:
                img0, img1, img2 = self.crop(rsz2_img0, rsz2_img1, rsz2_img2, self.h, self.w)
            elif q < 0.75:
                img0, img1, img2 = self.crop(rsz3_img0, rsz3_img1, rsz3_img2, self.h, self.w)
            else:
                img0, img1, img2 = self.crop(rsz4_img0, rsz4_img1, rsz4_img2, self.h, self.w)

            img0 = img0.permute(2, 0, 1)
            img1 = img1.permute(2, 0, 1)
            img2 = img2.permute(2, 0, 1)

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

            # Horizontal flip (reverse width)
            if random.uniform(0, 1) < 0.5:
                img0 = img0.flip(-1)
                img1 = img1.flip(-1)
                img2 = img2.flip(-1)

            # Vertical flip (reverse height)
            if random.uniform(0, 1) < 0.5:
                img0 = img0.flip(-2)
                img1 = img1.flip(-2)
                img2 = img2.flip(-2)

            # Depth-wise flip (reverse channels)
            if random.uniform(0, 1) < 0.28:
                img0 = img0.flip(0)
                img1 = img1.flip(0)
                img2 = img2.flip(0)

            # Exposure agumentation
            exp = random.uniform(1 / 8, 2)
            if random.uniform(0, 1) < 0.4:
                img0 = img0 * exp
                img1 = img1 * exp
                img2 = img2 * exp
            
            '''
            # slight colour shift between imcoming and outgoing images
            delta = random.uniform(0, 0.05)
            r = random.uniform(1-delta, 1+delta)
            g = random.uniform(1-delta, 1+delta)
            b = random.uniform(1-delta, 1+delta)
            multipliers = torch.tensor([r, g, b]).view(3, 1, 1).to(device)
            img0 = img0 * multipliers

            r = random.uniform(1-delta, 1+delta)
            g = random.uniform(1-delta, 1+delta)
            b = random.uniform(1-delta, 1+delta)
            multipliers = torch.tensor([r, g, b]).view(3, 1, 1).to(device)
            img2 = img2 * multipliers
            '''

            batch_img0.append(img0)
            batch_img1.append(img1)
            batch_img2.append(img2)

        return torch.stack(batch_img0), torch.stack(batch_img1), torch.stack(batch_img2), ratio, images_idx

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
        # end
    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

def warp_tenflow(tenInput, tenFlow):
    g = (tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

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

def blur(img, interations = 16):
    def gaussian_kernel(size, sigma):
        """Creates a 2D Gaussian kernel using the specified size and sigma."""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return torch.tensor(kernel / np.sum(kernel))

    class GaussianBlur(nn.Module):
        def __init__(self, kernel_size, sigma):
            super(GaussianBlur, self).__init__()
            # Create a Gaussian kernel
            self.kernel = gaussian_kernel(kernel_size, sigma)
            self.kernel = self.kernel.view(1, 1, kernel_size, kernel_size)
            
            # Set up the convolutional layer, without changing the number of channels
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2, groups=1, bias=False, padding_mode='reflect')
            
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

def main():
    parser = argparse.ArgumentParser(description='Training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')

    # Optional arguments
    parser.add_argument('--lr', type=float, default=0.00009, help='Learning rate (default: 9e-5)')
    parser.add_argument('--type', type=int, default=1, help='Model type (int): 1 - MultiresNet, 2 - MultiresNet 4 (default: 1)')
    parser.add_argument('--warmup', type=float, default=0.001, help='Warmup epochs (float) (default: 1)')
    parser.add_argument('--pulse', type=float, default=999, help='Period in steps to pulse learning rate (float) (default: 999)')
    parser.add_argument('--pulse_amplitude', type=float, default=25, help='Learning rate pulse amplitude (percentage) (default: 25)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model (optional)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (int) (default: 8)')
    parser.add_argument('--first_epoch', type=int, default=-1, help='Epoch (int) (default: Saved)')
    parser.add_argument('--epochs', type=int, default=-1, help='Epoch (int) (default: Saved)')
    parser.add_argument('--no_eval', action='store_false', dest='eval', default=True, help='Disable evaluation mode')

    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.dataset_path, 'preview')):
        os.makedirs(os.path.join(args.dataset_path, 'preview'))

    read_image_queue = queue.Queue(maxsize=12)
    dataset = TimewarpMLDataset(args.dataset_path, batch_size=args.batch_size)

    def read_images(read_image_queue, dataset):
        while True:
            for batch_idx in range(len(dataset)):
                img0, img1, img2, ratio, idx = dataset[batch_idx]
                read_image_queue.put((img0, img1, img2, ratio, idx))

    def write_images(write_image_queue):
        while True:
            try:
                write_data = write_image_queue.get_nowait()
                write_exr(write_data['sample_source1'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_incomng.exr'))
                write_exr(write_data['sample_source2'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_outgoing.exr'))
                write_exr(write_data['sample_target'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_target.exr'))
                write_exr(write_data['sample_output'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_output.exr'))
                write_exr(write_data['sample_output_mask'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_output_mask.exr'))
            except:
            # except queue.Empty:
                time.sleep(0.1)

    read_thread = threading.Thread(target=read_images, args=(read_image_queue, dataset))
    read_thread.daemon = True
    read_thread.start()

    write_image_queue = queue.Queue(maxsize=96)
    write_thread = threading.Thread(target=write_images, args=(write_image_queue, ))
    write_thread.daemon = True
    write_thread.start()
    
    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')

    encoder = Encoder().get_training_model()().to(device)
    flownet = Flownet().get_training_model()().to(device)
    
    pulse_dive = args.pulse_amplitude
    pulse_period = args.pulse
    lr = args.lr

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()

    # optimizer_encoder = Yogi(encoder.parameters(), lr=lr)
    # optimizer_flownet = Yogi(flownet.parameters(), lr=lr)
    optimizer_flownet = torch.optim.AdamW(flownet.parameters(), lr=lr, weight_decay=1e-2)
    '''
    def warmup(current_step, lr = 4e-3, number_warmup_steps = 999):
        mul_lin = current_step / number_warmup_steps
        # print (f'\n number_warmup_steps {number_warmup_steps} lr {lr} mul {mul_lin} res {(lr * mul_lin):.4e}')
        return lr * mul_lin
    '''

    # remove annoying message in pytorch 1.12.1 when using CosineAnnealingLR
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    # train_scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=pulse_period, eta_min = lr - (( lr / 100 ) * pulse_dive) )
    # train_scheduler_flownet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_flownet, T_max=pulse_period, eta_min = lr - (( lr / 100 ) * pulse_dive) )
    
    lambda_function = lambda epoch: 1
    train_scheduler_flownet = torch.optim.lr_scheduler.LambdaLR(optimizer_flownet, lr_lambda=lambda_function)

    # scheduler_encoder = train_scheduler_encoder
    scheduler_flownet = train_scheduler_flownet

    step = 0
    loaded_step = 0
    current_epoch = 0
    preview_index = 0

    steps_loss = []
    epoch_loss = []
    psnr_list = []

    if args.model_path:
        trained_model_path = args.model_path

        try:
            checkpoint = torch.load(trained_model_path)
            print('loaded previously saved model checkpoint')
        except Exception as e:
            print (f'unable to load saved model: {e}')

        try:
            encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
            print('loaded previously saved Encoder state')
        except Exception as e:
            print (f'unable to load Encoder state: {e}')

        try:
            flownet.load_state_dict(checkpoint['flownet_state_dict'], strict=False)
            print('loaded previously saved Flownet state')
        except Exception as e:
            print (f'unable to load Flownet state: {e}')

        try:
            loaded_step = checkpoint['step']
            print (f'loaded step: {loaded_step}')
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

    '''
    default_rife_model_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'models_data',
        'flownet_v412.pkl'
    )
    rife_state_dict = torch.load(default_rife_model_path)
    def convert(param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }
    model_rife.load_state_dict(convert(rife_state_dict))
    '''

    start_timestamp = time.time()
    time_stamp = time.time()
    epoch = current_epoch if args.first_epoch == -1 else args.first_epoch
    step = loaded_step if args.first_epoch == -1 else step
    print('\n\n')
    batch_idx = 0

    while True:
        data_time = time.time() - time_stamp
        time_stamp = time.time()

        img0, img1, img2, ratio, idx = read_image_queue.get()

        if platform.system() == 'Darwin':
            img0 = normalize_numpy(img0)
            img1 = normalize_numpy(img1)
            img2 = normalize_numpy(img2)
            img0 = img0.to(device, non_blocking = True)
            img1 = img1.to(device, non_blocking = True)
            img2 = img2.to(device, non_blocking = True)
        else:
            img0 = img0.to(device, non_blocking = True)
            img1 = img1.to(device, non_blocking = True)
            img2 = img2.to(device, non_blocking = True)
            img0 = normalize(img0)
            img1 = normalize(img1)
            img2 = normalize(img2)

        current_lr = scheduler_flownet.get_last_lr()[0]
        # for param_group_encoder in optimizer_encoder.param_groups:
        #    param_group_encoder['lr'] = current_lr
        for param_group_flownet in optimizer_flownet.param_groups:
            param_group_flownet['lr'] = current_lr

        current_lr_str = str(f'{optimizer_flownet.param_groups[0]["lr"]:.4e}')

        # optimizer_encoder.zero_grad(set_to_none=True)
        optimizer_flownet.zero_grad(set_to_none=True)

        # f0 = encoder(img0)
        # f1 = encoder(img2)

        flownet.train()
        flow_list, mask_list, merged = flownet(img0, img1, img2, None, None, ratio)

        output = merged[3]
        mask = mask_list[3]

        loss_x8 = criterion_mse(
            torch.nn.functional.interpolate(merged[0], scale_factor= 1. / 8, mode="bilinear", align_corners=False),
            torch.nn.functional.interpolate(img1, scale_factor= 1. / 8, mode="bilinear", align_corners=False)
        )

        '''
        loss_x4 = criterion_mse(
            torch.nn.functional.interpolate(merged[1], scale_factor= 1. / 4, mode="bilinear", align_corners=False),
            torch.nn.functional.interpolate(img1, scale_factor= 1. / 4, mode="bilinear", align_corners=False)
        )
        

        loss_x2 = criterion_mse(
            torch.nn.functional.interpolate(merged[1], scale_factor= 1. / 2, mode="bilinear", align_corners=False),
            torch.nn.functional.interpolate(img1, scale_factor= 1. / 2, mode="bilinear", align_corners=False)
        )
        '''

        loss_x1 = criterion_mse(merged[3], img1)

        # loss_enc = criterion_mse(encoder(output), encoder(img1))

        loss = 0.4 * loss_x8 + 0.6 * loss_x1

        # loss = 0.4 * loss_x8 + 0.3 * loss_x4 + 0.2 * loss_x2 + 0.1 * loss_x1 + 0.01 * loss_enc

        loss_l1 = criterion_l1(merged[3], img1)
        loss_l1_str = str(f'{loss_l1.item():.6f}')

        epoch_loss.append(float(loss_l1.item()))
        steps_loss.append(float(loss_l1.item()))
        psnr_list.append(psnr_torch(output, img1))

        if len(epoch_loss) < 999:
            smoothed_window_loss = np.mean(moving_average(epoch_loss, 9))
            window_min = min(epoch_loss)
            window_max = max(epoch_loss)
        else:
            smoothed_window_loss = np.mean(moving_average(epoch_loss[-999:], 9))
            window_min = min(epoch_loss[-999:])
            window_max = max(epoch_loss[-999:])
        smoothed_loss = np.mean(moving_average(epoch_loss, 9))

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(flownet.parameters(), 1.0)

        # optimizer_encoder.step()
        optimizer_flownet.step()
        # scheduler_encoder.step()
        scheduler_flownet.step()

        train_time = time.time() - time_stamp
        time_stamp = time.time()

        if step % 99 == 1:
            '''
            def warp(tenInput, tenFlow):
                backwarp_tenGrid = {}
                k = (str(tenFlow.device), str(tenFlow.size()))
                if k not in backwarp_tenGrid:
                    tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                    tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                    backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device = tenInput.device, dtype = tenInput.dtype)
                    # end
                tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

                g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
                return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
            
            output = ( warp(img1, flow0) + warp(img3, flow1) ) / 2
            '''

            if platform.system() == 'Darwin':
                rgb_source1 = restore_normalized_values_numpy(img0)
                rgb_source2 = restore_normalized_values_numpy(img2)
                rgb_target = restore_normalized_values_numpy(img1)
                rgb_output = restore_normalized_values_numpy(output)
                rgb_output_mask = mask.repeat_interleave(3, dim=1)
            else:
                rgb_source1 = restore_normalized_values(img0)
                rgb_source2 = restore_normalized_values(img2)
                rgb_target = restore_normalized_values(img1)
                rgb_output = restore_normalized_values(output)
                rgb_output_mask = mask.repeat_interleave(3, dim=1)

            write_image_queue.put(
                {
                    'preview_folder': os.path.join(args.dataset_path, 'preview'),
                    'sample_source1': rgb_source1[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_source2': rgb_source2[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_target': rgb_target[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output': rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output_mask': rgb_output_mask[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                }
            )

            preview_index = preview_index + 1 if preview_index < 9 else 0

            # sample_current = rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0)

        if step % 1000 == 1:
            torch.save({
                'step': step,
                'steps_loss': steps_loss,
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                'start_timestamp': start_timestamp,
                'lr': optimizer_flownet.param_groups[0]['lr'],
                'encoder_state_dict': encoder.state_dict(),
                'flownet_state_dict': flownet.state_dict(),
                # 'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                'optimizer_flownet_state_dict': optimizer_flownet.state_dict(),
            }, trained_model_path)
            
            # model.load_state_dict(convert(rife_state_dict))

        data_time += time.time() - time_stamp
        data_time_str = str(f'{data_time:.2f}')
        train_time_str = str(f'{train_time:.2f}')
        # current_lr_str = str(f'{optimizer.param_groups[0]["lr"]:.4e}')
        # current_lr_str = str(f'{scheduler.get_last_lr()[0]:.4e}')

        epoch_time = time.time() - start_timestamp
        days = int(epoch_time // (24 * 3600))
        hours = int((epoch_time % (24 * 3600)) // 3600)
        minutes = int((epoch_time % 3600) // 60)

        clear_lines(2)
        # print (f'\r {" "*180}', end='')
        # print ('\n')
        print (f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Time:{data_time_str} + {train_time_str}, Batch [{batch_idx+1}, {idx+1} / {len(dataset)}], Lr: {current_lr_str}, Loss L1: {loss_l1_str}')
        print(f'\r[Last 1K steps] Min: {window_min:.6f} Avg: {smoothed_window_loss:.6f}, Max: {window_max:.6f} [Epoch] Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f}')

        if ( idx + 1 ) == len(dataset):
            torch.save({
                'step': step,
                'steps_loss': steps_loss,
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                'start_timestamp': start_timestamp,
                'lr': optimizer_flownet.param_groups[0]['lr'],
                'encoder_state_dict': encoder.state_dict(),
                'flownet_state_dict': flownet.state_dict(),
                # 'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                'optimizer_flownet_state_dict': optimizer_flownet.state_dict(),
            }, trained_model_path)

            psnr = 0

            if args.eval:
                psnr_list = []

                try:
                    for ev_item_index in range(111):

                        clear_lines(2)
                        print (f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Time:{data_time_str} + {train_time_str}, Batch [{batch_idx+1}, {idx+1} / {len(dataset)}], Lr: {current_lr_str}, Loss L1: {loss_l1_str}')
                        print (f'\rCalcualting PSNR on full-scale image {ev_item_index} of 111...')

                        ev_item = dataset.frames_queue.get()
                        ev_img0 = ev_item['start']
                        ev_img1 = ev_item['gt']
                        ev_img2 = ev_item['end']
                        ev_ratio = ev_item['ratio']

                        ev_img0 = torch.from_numpy(ev_img0.copy())
                        ev_img1 = torch.from_numpy(ev_img1.copy())
                        ev_img2 = torch.from_numpy(ev_img2.copy())
                        ev_img0 = ev_img0.to(device = device, dtype = torch.float32)
                        ev_img1 = ev_img1.to(device = device, dtype = torch.float32)
                        ev_img2 = ev_img2.to(device = device, dtype = torch.float32)
                        ev_img0 = ev_img0.permute(2, 0, 1).unsqueeze(0)
                        ev_img1 = ev_img1.permute(2, 0, 1).unsqueeze(0)
                        ev_img2 = ev_img2.permute(2, 0, 1).unsqueeze(0)
                        evn_img0 = normalize(ev_img0)
                        evn_img1 = normalize(ev_img1)
                        evn_img2 = normalize(ev_img2)

                        n, c, h, w = evn_img1.shape
                        
                        ph = ((h - 1) // 64 + 1) * 64
                        pw = ((w - 1) // 64 + 1) * 64
                        padding = (0, pw - w, 0, ph - h)
                        evp_img0 = torch.nn.functional.pad(evn_img0, padding)
                        evp_img1 = torch.nn.functional.pad(evn_img1, padding)
                        evp_img2 = torch.nn.functional.pad(evn_img2, padding)

                        with torch.no_grad():
                            f0 = encoder(evp_img0)
                            f1 = encoder(evp_img2)

                            flownet.eval()
                            _, _, merged = flownet(evp_img0, evp_img2, f0, f1, timestep = ev_ratio)
                            evp_output = merged[3]
                            psnr_list.append(psnr_torch(evp_output, evp_img1))

                            # ev_gt = ev_gt[0].permute(1, 2, 0)[:h, :w]                        
                            # evp_timestep = (evp_img1[:, :1].clone() * 0 + 1) * ev_ratio

                            '''
                            evp_id_flow = id_flow(evp_img1)
                            ev_in_flow0, ev_in_flow1, ev_in_mask, ev_in_deep = model(torch.cat((evp_img1, evp_id_flow, evp_timestep, evp_img3), dim=1))
                            ev_in_flow0 = ev_in_flow0 + evp_id_flow
                            ev_in_flow1 = ev_in_flow1 + evp_id_flow

                            ev_output_inflow = warp_tenflow(evp_img1, ev_in_flow0) * ev_in_mask + warp_tenflow(evp_img3, ev_in_flow1) * (1 - ev_in_mask)
                            ev_output_inflow = restore_normalized_values(ev_output_inflow)
                            psnr_list.append(psnr_torch(ev_output_inflow, evp_img2))
                            ev_output_inflow = ev_output_inflow[0].permute(1, 2, 0)[:h, :w]
                            '''
                            # ev_output_inflow, ev_in_deep = model(torch.cat((evp_img1*2-1, evp_img3*2-1, evp_timestep*2-1, ), dim=1))
                            # psnr_list.append(psnr_torch(ev_output_rife, evp_img2))
                            # ev_output_inflow = restore_normalized_values(ev_output_inflow)
                            # ev_output_inflow = ev_output_inflow[0].permute(1, 2, 0)[:h, :w]

                        preview_folder = os.path.join(args.dataset_path, 'preview')
                        eval_folder = os.path.join(preview_folder, 'eval')
                        if not os.path.isdir(eval_folder):
                            try:
                                os.makedirs(eval_folder)
                            except Exception as e:
                                print (e)

                        evp_output = restore_normalized_values(evp_output)
                        ev_output = evp_output[0].permute(1, 2, 0)[:h, :w]

                        if ev_item_index  % 9 == 1:
                            try:
                                write_exr(ev_img0[0].permute(1, 2, 0)[:h, :w].clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_incomng.exr'))
                                write_exr(ev_img2[0].permute(1, 2, 0)[:h, :w].clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_outgoing.exr'))
                                write_exr(ev_img1[0].permute(1, 2, 0)[:h, :w].clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_target.exr'))
                                # write_exr(ev_output_inflow.clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_output.exr'))
                                write_exr(ev_output.clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_output.exr'))
                            except Exception as e:
                                print (f'{e}\n\n')      

                except Exception as e:
                    print (f'{e}\n\n')
   
            psnr = np.array(psnr_list).mean()

            epoch_time = time.time() - start_timestamp
            days = int(epoch_time // (24 * 3600))
            hours = int((epoch_time % (24 * 3600)) // 3600)
            minutes = int((epoch_time % 3600) // 60)

            # rife_psnr = np.array(rife_psnr_list).mean()

            clear_lines(2)
            # print (f'\r {" "*240}', end='')
            print(f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f}, [PNSR] {psnr:.4f}')
            print ('\n')

            steps_loss = []
            epoch_loss = []
            psnr_list = []
            epoch = epoch + 1
            batch_idx = 0

            while  ( idx + 1 ) == len(dataset):
                img0, img1, img2, ratio, idx = read_image_queue.get()
            dataset.reshuffle()

        batch_idx = batch_idx + 1
        step = step + 1
        if epoch == args.epochs:
            sys.exit()

if __name__ == "__main__":
    main()

