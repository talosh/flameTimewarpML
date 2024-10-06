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
from copy import deepcopy

from pprint import pprint

try:
    import torch
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


def read_image_file(file_path, header_only = False):
    result = {'spec': None, 'image_data': None}
    inp = oiio.ImageInput.open(file_path)
    if inp :
        spec = inp.spec()
        result['spec'] = spec
        if not header_only:
            height = spec.height
            width = spec.width
            channels = spec.nchannels
            result['image_data'] = inp.read_image(0, 0, 0, channels)
        inp.close()
    return result

class TimewarpMLDataset(torch.utils.data.Dataset):
    def __init__(   
            self, 
            data_root, 
            batch_size = 4, 
            frame_size=448, 
            max_window=9,
            repeat = 1,
            scale_list = [1.0, 1.12, 1.25]
            ):
        
        self.data_root = data_root
        self.batch_size = batch_size
        self.max_window = max_window
        self.h = frame_size
        self.w = frame_size
        self.scale_list = scale_list
        self.repeat_count = repeat
        self.repeat_counter = 0

        print (f'scanning for exr files in {self.data_root}...')
        self.folders_with_exr = self.find_folders_with_images(data_root)
        print (f'found {len(self.folders_with_exr)} clip folders.')
        
        self.initial_train_descriptions = []    
        for folder_index, folder_path in enumerate(sorted(self.folders_with_exr)):
            print (f'\rReading headers and building training data from clip {folder_index + 1} of {len(self.folders_with_exr)}', end='')
            self.initial_train_descriptions.extend(self.create_dataset_descriptions(folder_path, max_window=self.max_window))

        self.train_descriptions = list(self.initial_train_descriptions)

        print ('\nReshuffling training data indices...')
        self.reshuffle()

        self.mp_frames_queue = torch.multiprocessing.Queue(maxsize=4)
        self.frames_queue = queue.Queue(maxsize=32)

        def read_frames_thread(train_descriptions, frames_queue, batch_size, scale_list, h, w):
            while True:
                frame_read_process = torch.multiprocessing.Process(
                    target=self.read_frames,
                    args=(
                        frames_queue,
                        list(train_descriptions),
                        batch_size,
                        scale_list,
                        h,
                        w
                        ),
                    daemon = True
                )
                frame_read_process.start()
                frame_read_process.join()

        self.frame_read_thread = threading.Thread(
            target=read_frames_thread, args=(
                self.train_descriptions, 
                self.mp_frames_queue, 
                self.batch_size, 
                self.scale_list, 
                self.h,
                self.w
                )
            )
        self.frame_read_thread.daemon = True
        self.frame_read_thread.start()

        # to reduce overhead when getting directly from mp queue
        def transfer_frames_thread(mp_frames_queue, frames_queue):
            while True:
                frames_queue.put(mp_frames_queue.get())

        self.frame_read_thread = threading.Thread(target=transfer_frames_thread, args=(self.mp_frames_queue, self.frames_queue))
        self.frame_read_thread.daemon = True
        self.frame_read_thread.start()

        print ('Starting frame reader process...', end='')
        # reading first block of training data here
        self.last_train_data = [self.frames_queue.get()]
        self.last_train_data_size = 4 # size of the buffer that stores training data
        self.new_sample_shown = False
        self.train_data_index = 0
        print (' Done.')

    def reshuffle(self):
        random.shuffle(self.train_descriptions)

    def find_folders_with_images(self, path, ext_list=['.exr']):
        directories_with_imgs = set()

        # Walk through all directories and files in the given path
        for root, dirs, files in os.walk(path):
            if 'dead_pixel_scan_' in root:
                continue
            for file in files:
                if any(file.lower().endswith(ext) for ext in ext_list):
                    directories_with_imgs.add(root)
                    break  # No need to check other files in the same directory

        return directories_with_imgs

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
            max_window = max_window
        else:
            if max_window > 5:
                max_window = 5

        try:
            first_exr_file = read_image_file(exr_files[0], header_only = True)
            h = first_exr_file['spec'].height
            w = first_exr_file['spec'].width

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
    
    @staticmethod
    def read_frames(frames_queue, train_descriptions, batch_size, scale_list, h, w):

        def resize_image(tensor, x):
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
            # tensor = tensor.permute(2, 0, 1).unsqueeze(0)

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
            # resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)

            return resized_tensor

        def crop_images(img0, img1, img2, h, w):
            np.random.seed(None)
            _, ih, iw = img0.shape
            x = np.random.randint(0, ih - h + 1)
            y = np.random.randint(0, iw - w + 1)
            img0 = img0[:, x:x+h, y:y+w]
            img1 = img1[:, x:x+h, y:y+w]
            img2 = img2[:, x:x+h, y:y+w]
            return img0, img1, img2

        num_items = len(train_descriptions)
        for start_idx in range(0, num_items, batch_size):
            end_idx = min(start_idx + batch_size, num_items)
            batch = train_descriptions[start_idx:end_idx]
            
            batch_img0 = []
            batch_img1 = []
            batch_img2 = []
            batch_ratio = []

            for item in batch:
                scale = scale_list[random.randint(0, len(scale_list) - 1)]

                try:
                    img0 = read_image_file(item['start'])['image_data']
                    img1 = read_image_file(item['gt'])['image_data']
                    img2 = read_image_file(item['end'])['image_data']

                    img0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0)
                    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
                    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)

                    img0 = img0.to(device = 'cpu', dtype = torch.float32, non_blocking=True)
                    img1 = img0.to(device = 'cpu', dtype = torch.float32, non_blocking=True)
                    img2 = img0.to(device = 'cpu', dtype = torch.float32, non_blocking=True)

                    img0 = resize_image(img0, int(h * scale))
                    img1 = resize_image(img0, int(h * scale))
                    img2 = resize_image(img0, int(h * scale))

                    img0, img1, img2 = crop_images(img0[0], img1[0], img2[0], h, w)

                    batch_img0.append(img0)
                    batch_img1.append(img1)
                    batch_img2.append(img2)
                    batch_ratio.append(torch.full((1, w, h), item['ratio']))
                except Exception as e:
                    print (f'\n{e}\n')

            training_data = {}
            training_data['start'] = torch.stack(batch_img0)
            training_data['gt'] = torch.stack(batch_img1)
            training_data['end'] = torch.stack(batch_img2)
            training_data['ratio'] = torch.stack(batch_ratio)
            training_data['index'] = end_idx
            frames_queue.put(training_data)

    def getimg(self, index):
        # '''
        if not self.last_train_data:
            new_data = self.frames_queue.get_nowait()
            self.last_train_data = [new_data]
            del new_data
        if self.repeat_counter >= self.repeat_count:
            try:
                if len(self.last_train_data) == self.last_train_data_size:
                    self.last_train_data.pop(0)
                elif len(self.last_train_data) == self.last_train_data_size:
                    self.last_train_data = self.last_train_data[:-(self.last_train_data_size - 1)]
                new_data = self.frames_queue.get_nowait()
                self.train_data_index = new_data['index']
                self.last_train_data.append(new_data)
                self.new_sample_shown = False
                del new_data
                self.repeat_counter = 0
            except queue.Empty:
                pass

        self.repeat_counter += 1
        if not self.new_sample_shown:
            self.new_sample_shown = True
            return self.last_train_data[-1]
        # if random.uniform(0, 1) < 0.44:
        #    return self.last_train_data[-1]
        else:
            return self.last_train_data[random.randint(0, len(self.last_train_data) - 1)]
        # '''
        # return self.frames_queue.get()

    def __len__(self):
        return len(self.train_descriptions)

    def __getitem__(self, index):
        training_data =  self.getimg(index)
        images_idx = self.train_data_index
        return training_data['start'], training_data['gt'], training_data['end'], training_data['ratio'], images_idx

def get_dataset(
        data_root, 
        batch_size = 8, 
        frame_size=448, 
        max_window=9,
        repeat = 1
        ):

    return TimewarpMLDataset(
        data_root, 
        batch_size=batch_size, 
        frame_size=frame_size, 
        max_window=max_window,
        repeat=repeat
        )

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

def srgb_to_linear(self, srgb_image):
    # Apply the inverse sRGB gamma curve
    mask = srgb_image <= 0.04045
    srgb_image[mask] = srgb_image[mask] / 12.92
    srgb_image[~mask] = ((srgb_image[~mask] + 0.055) / 1.055) ** 2.4

    return srgb_image

def apply_acescc(linear_image):
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

    return ACEScc

def augment_images(img0, img1, img2, generalize, acescc_rate):
    if generalize == 0:
        # No augmentaton
        return img0, img1, img2
    elif generalize == 1:
        # Augment only with horizontal flipping
        if random.uniform(0, 1) < 0.5:
            img0 = img0.flip(-1)
            img1 = img1.flip(-1)
            img2 = img2.flip(-1)
        return img0, img1, img2
    else:
        # Horizontal flip (reverse width)
        if random.uniform(0, 1) < 0.5:
            img0 = img0.flip(-1)
            img1 = img1.flip(-1)
            img2 = img2.flip(-1)

        # Rotation
        if random.uniform(0, 1) < (generalize / 100):
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

        if random.uniform(0, 1) < (generalize / 100):
            # Vertical flip (reverse height)
            if random.uniform(0, 1) < 0.5:
                img0 = img0.flip(-2)
                img1 = img1.flip(-2)
                img2 = img2.flip(-2)

        if random.uniform(0, 1) < (generalize / 100):
            # Depth-wise flip (reverse channels)
            if random.uniform(0, 1) < 0.28:
                img0 = img0.flip(0)
                img1 = img1.flip(0)
                img2 = img2.flip(0)

        if random.uniform(0, 1) < (generalize / 100):
            # Exposure augmentation
            exp = random.uniform(1 / 8, 2)
            if random.uniform(0, 1) < 0.4:
                img0 = img0 * exp
                img1 = img1 * exp
                img2 = img2 * exp

        if random.uniform(0, 1) < (generalize / 100):
            # add colour banace shift
            delta = random.uniform(0, 0.28)
            r = random.uniform(1-delta, 1+delta)
            g = random.uniform(1-delta, 1+delta)
            b = random.uniform(1-delta, 1+delta)
            multipliers = torch.tensor([r, g, b]).view(3, 1, 1).to(img0.device)
            img0 = img0 * multipliers
            img1 = img1 * multipliers
            img2 = img2 * multipliers

        def gamma_up(img, gamma = 1.18):
            return torch.sign(img) * torch.pow(torch.abs(img), 1 / gamma )

        if random.uniform(0, 1) < (generalize / 100):
            if random.uniform(0, 1) < 0.44:
                gamma = random.uniform(0.9, 1.9)
                img0 = gamma_up(img0, gamma=gamma)
                img1 = gamma_up(img1, gamma=gamma)
                img2 = gamma_up(img2, gamma=gamma)

        # Convert to ACEScc
        if random.uniform(0, 1) < (acescc_rate / 100):
            img0 = apply_acescc(img0)
            img1 = apply_acescc(img1)
            img2 = apply_acescc(img2)

        return img0, img1, img2

current_state_dict = {}

def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
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
    parser.add_argument('--pulse', type=float, default=9999, help='Period in steps to pulse learning rate (float) (default: 10K)')
    parser.add_argument('--pulse_amplitude', type=float, default=25, help='Learning rate pulse amplitude (percentage) (default: 25)')
    parser.add_argument('--onecycle', type=int, default=-1, help='Train one cycle for N epochs (default: None)')
    parser.add_argument('--state_file', type=str, default=None, help='Path to the pre-trained model state dict file (optional)')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--legacy_model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (int) (default: 2)')
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
    parser.add_argument('--all_gpus', action='store_true', dest='all_gpus', default=False, help='Use nn.DataParallel')
    parser.add_argument('--freeze', action='store_true', dest='freeze', default=False, help='Freeze RIFE parameters')
    parser.add_argument('--acescc', type=check_range_percent, default=40, help='Percentage of ACEScc encoded frames (default: 40))')
    parser.add_argument('--generalize', type=check_range_percent, default=85, help='Generalization level (0 - 100) (default: 85)')
    parser.add_argument('--weight_decay', type=float, default=-1, help='AdamW weight decay (default: calculated from --generalize value)')
    parser.add_argument('--preview', type=int, default=1000, help='Save preview each N steps (default: 1000)')
    parser.add_argument('--save', type=int, default=1000, help='Save model state dict each N steps (default: 1000)')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat each triade N times with augmentation (default: 1)')
    parser.add_argument('--iterations', type=int, default=1, help='Process each flow refinement N times (default: 1)')
    parser.add_argument('--compile', action='store_true', dest='compile', default=False, help='Compile with torch.compile')

    args = parser.parse_args()

    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')
    if args.all_gpus:
        device = 'cuda'

    Flownet = None

    if args.model:
        model_name = args.model
        Flownet = find_and_import_model(base_name='flownet', model_name=model_name)            
    else:
        # Find and initialize model
        if args.state_file and os.path.isfile(args.state_file):
            trained_model_path = args.state_file
            try:
                checkpoint = torch.load(trained_model_path, map_location=device, weights_only=False)
                print('loaded previously saved model checkpoint')
            except Exception as e:
                print (f'unable to load saved model checkpoint: {e}')
                sys.exit()

            model_info = checkpoint.get('model_info')
            model_file = model_info.get('file')
            Flownet = find_and_import_model(model_file=model_file)
        else:
            if not args.state_file:
                print ('Please specify either model name or model state file')
                return
            if not os.path.isfile(args.state_file):
                print (f'Model state file {args.state_file} does not exist and "--model" flag is not set to start from scratch')
                return

    if Flownet is None:
        print (f'Unable to load model {args.model}')
        return
    
    model_info = Flownet.get_info()
    print ('Model info:')
    pprint (model_info)
    max_dataset_window = 9
    if not model_info.get('ratio_support'):
        max_dataset_window = 3
    
    if args.compile:
        flownet_uncompiled = Flownet().get_training_model()().to(torch.float32).cuda()
        flownet = torch.compile(flownet_uncompiled,mode='reduce-overhead')
    else:
        flownet = Flownet().get_training_model()().to(device)
    
    if args.all_gpus:
        print ('Using nn.DataParallel')
        flownet = torch.nn.DataParallel(flownet)
        flownet.to(device)

    if not os.path.isdir(os.path.join(args.dataset_path, 'preview')):
        os.makedirs(os.path.join(args.dataset_path, 'preview'))

    # frame_size = closest_divisible(abs(int(args.frame_size)), model_info.get('padding', 64))
    # if frame_size != args.frame_size:
    #     print (f'Frame size should be divisible by 64 for training. Using {frame_size}')

    frame_size = args.frame_size

    dataset = get_dataset(
        args.dataset_path, 
        batch_size=args.batch_size, 
        frame_size=frame_size,
        max_window=max_dataset_window,
        repeat=args.repeat
        )
    
    if args.eval_folder:
        print (f'Scanning data for evaluation:')
        eval_dataset = get_dataset(
        args.eval_folder, 
        batch_size=args.batch_size,
        frame_size=frame_size,
        max_window=max_dataset_window,
        repeat=args.repeat
        )
    else:
        eval_dataset = dataset
    
    def write_images(write_image_queue):
        while True:
            try:
                write_data = write_image_queue.get_nowait()
                preview_index = write_data.get('preview_index', 0)
                write_exr(write_data['sample_source1'].astype(np.float16), os.path.join(write_data['preview_folder'], f'{preview_index:02}_incomng.exr'), half_float = True)
                write_exr(write_data['sample_source2'].astype(np.float16), os.path.join(write_data['preview_folder'], f'{preview_index:02}_outgoing.exr'), half_float = True)
                write_exr(write_data['sample_target'].astype(np.float16), os.path.join(write_data['preview_folder'], f'{preview_index:02}_target.exr'), half_float = True)
                write_exr(write_data['sample_output'].astype(np.float16), os.path.join(write_data['preview_folder'], f'{preview_index:02}_output.exr'), half_float = True)
                write_exr(write_data['sample_output_mask'].astype(np.float16), os.path.join(write_data['preview_folder'], f'{preview_index:02}_output_mask.exr'), half_float = True)
                del write_data
            except:
            # except queue.Empty:
                time.sleep(1e-2)

    def write_eval_images(write_eval_image_queue):
        while True:
            try:
                write_data = write_eval_image_queue.get_nowait()
                write_exr(write_data['sample_source1'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_source1_name']), half_float = True)
                write_exr(write_data['sample_source2'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_source2_name']), half_float = True)
                write_exr(write_data['sample_target'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_target_name']), half_float = True)
                write_exr(write_data['sample_output'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_output_name']), half_float = True)
                write_exr(write_data['sample_output_mask'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_output_mask_name']), half_float = True)
                del write_data
            except:
            # except queue.Empty:
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

    '''
    write_image_queue = queue.Queue(maxsize=16)
    write_thread = threading.Thread(target=write_images, args=(write_image_queue, ))
    write_thread.daemon = True
    write_thread.start()
    
    write_eval_image_queue = queue.Queue(maxsize=args.eval_buffer)
    write_eval_thread = threading.Thread(target=write_eval_images, args=(write_eval_image_queue, ))
    write_eval_thread.daemon = True
    write_eval_thread.start()
    '''

    write_model_state_queue = queue.Queue(maxsize=2)
    write_model_state_thread = threading.Thread(target=write_model_state, args=(write_model_state_queue, ))
    write_model_state_thread.daemon = True
    write_model_state_thread.start()

    pulse_dive = args.pulse_amplitude
    pulse_period = args.pulse
    lr = args.lr

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    criterion_huber = torch.nn.HuberLoss(delta=0.001)

    weight_decay = 10 ** (-2 - 0.02 * (args.generalize - 1)) if args.generalize > 1 else 1e-4

    if args.weight_decay != -1:
        weight_decay = args.weight_decay

    if args.generalize == 0:
        print (f'Disabling augmentation and setting weight decay to {weight_decay:.2e}')
    elif args.generalize == 1:
        print (f'Setting augmentation to horizontal flip and scale only and weight decay to {weight_decay:.2e}')
    else:
        print (f'Setting augmentation rate to {args.generalize}% and weight decay to {weight_decay:.2e}')

    optimizer_flownet = torch.optim.AdamW(flownet.parameters(), lr=lr, weight_decay=weight_decay)
    train_scheduler_flownet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_flownet, T_max=pulse_period, eta_min = lr - (( lr / 100 ) * pulse_dive) )

    if args.onecycle != -1:
        train_scheduler_flownet = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_flownet,
            max_lr=args.lr,
            div_factor = 11,
            steps_per_epoch=len(dataset)*dataset.repeat_count, 
            epochs=args.onecycle
            )
        print (f'setting OneCycleLR with max_lr={args.lr}, steps_per_epoch={len(dataset)*dataset.repeat_count}, epochs={args.onecycle}')
        args.epochs = args.onecycle

    scheduler_flownet = train_scheduler_flownet

    step = 0
    loaded_step = 0
    current_epoch = 0
    preview_index = 0
    first_pass = True

    steps_loss = []
    epoch_loss = []
    psnr_list = []
    lpips_list = []

    if args.state_file:
        trained_model_path = args.state_file

        try:
            checkpoint = torch.load(trained_model_path, map_location=device, weights_only=False)
            print('loaded previously saved model checkpoint')
        except Exception as e:
            print (f'unable to load saved model: {e}')

        try:
            if args.all_gpus:
                missing_keys, unexpected_keys = flownet.load_state_dict(convert_to_data_parallel(checkpoint['flownet_state_dict']), strict=False, weights_only=False)
            else:
                missing_keys, unexpected_keys = flownet.load_state_dict(checkpoint['flownet_state_dict'], strict=False, weights_only=False)
            print('loaded previously saved Flownet state')
            if missing_keys:
                print (f'\nMissing keys:\n{missing_keys}\n')
            if unexpected_keys:
                print (f'\nUnexpected keys:\n{unexpected_keys}\n')
        except Exception as e:
            print (f'unable to load Flownet state: {e}')

        try:
            loaded_step = checkpoint['step']
            print (f'loaded step: {loaded_step}')
            current_epoch = checkpoint['epoch']
            print (f'epoch: {current_epoch + 1}')
        except Exception as e:
            print (f'unable to set step and epoch: {e}')

    else:
        traned_model_name = 'flameTWML_model_' + create_timestamp_uid() + '.pth'
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

    if args.legacy_model:
        rife_state_dict = torch.load(args.legacy_model, weights_only=False)
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        missing_keys, unexpected_keys = flownet.load_state_dict(convert(rife_state_dict), strict=False, weights_only=False)
        print (f'\nMissing keys:\n{missing_keys}\n')
        print (f'\nUnexpected keys:\n{unexpected_keys}\n')

    if args.reset_stats:
        step = 0
        loaded_step = 0
        current_epoch = 0
        preview_index = 0
        steps_loss = []
        epoch_loss = []
        psnr_list = []
        lpips_list = []

    # LPIPS Init

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    import lpips
    os.environ['TORCH_HOME'] = os.path.abspath(os.path.dirname(__file__))
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex.to(device)

    warnings.resetwarnings()

    start_timestamp = time.time()
    time_stamp = time.time()
    epoch = current_epoch if args.first_epoch == -1 else args.first_epoch
    step = loaded_step if args.first_epoch == -1 else step
    batch_idx = 0

    if args.freeze:
        print ('\nFreezing parameters')
        # '''
        # Freeze predictors - its custom hard-coded depending on a task

        # for param in flownet.encode.parameters():
        #    param.requires_grad = False

        '''
        for param in flownet.block0.encode01.parameters():
            param.requires_grad = False
        for param in flownet.block1.encode01.parameters():
            param.requires_grad = False
        for param in flownet.block2.encode01.parameters():
            param.requires_grad = False
        for param in flownet.block3.encode01.parameters():
            param.requires_grad = False
        '''

        for param in flownet.encode.parameters():
            param.requires_grad = False

        # '''
        for param in flownet.block0.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block1.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block2.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block3.conv0.parameters():
            param.requires_grad = False
        # '''

        for param in flownet.block0.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block1.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block2.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block3.convblock.parameters():
            param.requires_grad = False

        '''
        for param in flownet.block0.convblock[0].parameters():
            param.requires_grad = True
        for param in flownet.block1.convblock[0].parameters():
            param.requires_grad = True
        for param in flownet.block2.convblock[0].parameters():
            param.requires_grad = True
        for param in flownet.block3.convblock[0].parameters():
            param.requires_grad = True
        '''

        '''
        for param in flownet.block0.convblock[-1].parameters():
            param.requires_grad = True
        for param in flownet.block1.convblock[-1].parameters():
            param.requires_grad = True
        for param in flownet.block2.convblock[-1].parameters():
            param.requires_grad = True
        for param in flownet.block3.convblock[-1].parameters():
            param.requires_grad = True
        '''

        '''
        for param in flownet.block0.convblock[-2].parameters():
            param.requires_grad = True
        for param in flownet.block1.convblock[-2].parameters():
            param.requires_grad = True
        for param in flownet.block2.convblock[-2].parameters():
            param.requires_grad = True
        for param in flownet.block3.convblock[-2].parameters():
            param.requires_grad = True
        '''

        '''
        for param in flownet.block0.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block0.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block0.encode01.parameters():
            param.requires_grad = False

        for param in flownet.block1.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block1.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block1.encode01.parameters():
            param.requires_grad = False

        for param in flownet.block2.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block2.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block2.encode01.parameters():
            param.requires_grad = False

        for param in flownet.block3.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block3.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block3.encode01.parameters():
            param.requires_grad = False

        '''

        '''
        for param in flownet.block0.convblock[-1].parameters():
            param.requires_grad = True

        for param in flownet.block1.convblock[-1].parameters():
            param.requires_grad = True

        for param in flownet.block2.convblock[-1].parameters():
            param.requires_grad = True

        for param in flownet.block3.convblock[-1].parameters():
            param.requires_grad = True
        '''

        # '''
        
        for name, param in flownet.named_parameters():
            if not param.requires_grad:
                print(name, param.requires_grad)

        print ('\nUn-freezing parameters:')
        for name, param in flownet.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)

    print('\n\n')

    current_state_dict['step'] = int(step)
    current_state_dict['steps_loss'] = list(steps_loss)
    current_state_dict['epoch'] = int(epoch)
    current_state_dict['epoch_loss'] = list(epoch_loss)
    current_state_dict['start_timestamp'] = start_timestamp
    current_state_dict['lr'] = optimizer_flownet.param_groups[0]['lr']
    current_state_dict['model_info'] = model_info
    if args.all_gpus:
        current_state_dict['flownet_state_dict'] = convert_from_data_parallel(flownet.state_dict())
    else:
        current_state_dict['flownet_state_dict'] = flownet.state_dict()
    current_state_dict['optimizer_flownet_state_dict'] = optimizer_flownet.state_dict()
    current_state_dict['trained_model_path'] = trained_model_path

    if not os.path.isfile(f'{os.path.splitext(trained_model_path)[0]}.csv'):
        create_csv_file(
            f'{os.path.splitext(trained_model_path)[0]}.csv',
            [
                'Epoch',
                'Step',
                'Min',
                'Avg',
                'Max',
                'PSNR',
                'LPIPS'
            ]
        )

    if not os.path.isfile(f'{os.path.splitext(trained_model_path)[0]}.eval.csv'):
        create_csv_file(
            f'{os.path.splitext(trained_model_path)[0]}.eval.csv',
            [
                'Epoch',
                'Step',
                'Min',
                'Avg',
                'Max',
                'PSNR',
                'LPIPS'
            ]
        )

    import signal
    def create_graceful_exit(current_state_dict):
        def graceful_exit(signum, frame):
            print(f'\nSaving current state to {current_state_dict["trained_model_path"]}...')
            print (f'Epoch: {current_state_dict["epoch"]}, Step: {current_state_dict["step"]:11}')
            torch.save(current_state_dict, current_state_dict['trained_model_path'])
            exit(0)
        return graceful_exit
    signal.signal(signal.SIGINT, create_graceful_exit(current_state_dict))

    while True:
        data_time = time.time() - time_stamp
        time_stamp = time.time()

        # data block starts here
        img0, img1, img2, ratio, idx = dataset[0]

        img0 = img0.to(device = device, dtype = torch.float32)
        img1 = img1.to(device = device, dtype = torch.float32)
        img2 = img2.to(device = device, dtype = torch.float32)
        ratio = ratio.to(device = device, dtype = torch.float32)

        for i in range(img0.shape[0]):
            img0[i], img1[i], img2[i] = augment_images(img0[i], img1[i], img2[i], args.generalize, args.acescc)

        img0_orig = img0.detach().clone()
        img1_orig = img1.detach().clone()
        img2_orig = img2.detach().clone()

        img0 = normalize(img0)
        img1 = normalize(img1)
        img2 = normalize(img2)

        data_time += time.time() - time_stamp
        data_time_str = str(f'{data_time:.2f}')
        time_stamp = time.time()

        # trainimg block starts here
        current_lr_str = str(f'{optimizer_flownet.param_groups[0]["lr"]:.2e}')
        optimizer_flownet.zero_grad()

        # scale list augmentation
        random_scales = [
            [8, 4, 2, 1],
            [4, 4, 2, 1],
            [4, 2, 2, 1],
            [4, 2, 1, 1],
            [2, 2, 2, 1],
            [2, 2, 1, 1],
            [2, 1, 1, 1],
            [1, 1, 1, 1],
        ]

        if random.uniform(0, 1) < 0.44:
            training_scale = random_scales[random.randint(0, len(random_scales) - 1)]
        else:
            training_scale = [8, 4, 2, 1]

        flownet.train()

        # n_iterations = random.randint(1, 4) if args.iterations == -1 else args.iterations
        
        flow_list, mask_list, merged = flownet(
            img0, 
            img2, 
            ratio, 
            scale=training_scale,
            iterations = args.iterations
            )
        
        flow0 = flow_list[3][:, :2]
        flow1 = flow_list[3][:, 2:4]
        mask = mask_list[3]
        output = warp(img0_orig, flow0) * mask + warp(img2_orig, flow1) * (1 - mask)

        loss_LPIPS_ = loss_fn_alex(output * 2 - 1, img1_orig * 2 - 1)
        lpips_weight = 0.5

        lpips_weight = 0.5
        loss = (1 - lpips_weight ) * criterion_l1(output, img1_orig) + lpips_weight * 0.2 * float(torch.mean(loss_LPIPS_).item())

        loss_l1 = criterion_l1(output, img1_orig)
        loss_l1_str = str(f'{loss_l1.item():.6f}')

        epoch_loss.append(float(loss_l1.item()))
        steps_loss.append(float(loss_l1.item()))
        lpips_list.append(float(torch.mean(loss_LPIPS_).item()))
        psnr_list.append(float(psnr_torch(output, img1)))

        if len(epoch_loss) < 9999:
            smoothed_window_loss = np.mean(moving_average(epoch_loss, 9))
            window_min = min(epoch_loss)
            window_max = max(epoch_loss)
            lpips_window_val = float(np.array(lpips_list).mean())
        else:
            smoothed_window_loss = np.mean(moving_average(epoch_loss[-9999:], 9))
            window_min = min(epoch_loss[-9999:])
            window_max = max(epoch_loss[-9999:])
            lpips_window_val = float(np.array(lpips_list[-9999:]).mean())
        smoothed_loss = float(np.mean(moving_average(epoch_loss, 9)))
        lpips_val = float(np.array(lpips_list).mean())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(flownet.parameters(), 1)

        optimizer_flownet.step()

        try:
            scheduler_flownet.step()
        except Exception as e:
            # if Onecycle is over due to variable number of steps per epoch
            # fall back to Cosine

            print (f'switching to CosineAnnealingLR scheduler:')
            print (f'{e}\n\n')

            current_lr = float(optimizer_flownet.param_groups[0]["lr"])
            scheduler_flownet = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_flownet, 
                T_max=pulse_period, 
                eta_min = current_lr - (( current_lr / 100 ) * pulse_dive)
                )

        train_time = time.time() - time_stamp
        train_time_str = str(f'{train_time:.2f}')
        time_stamp = time.time()

        epoch_time = time.time() - start_timestamp
        days = int(epoch_time // (24 * 3600))
        hours = int((epoch_time % (24 * 3600)) // 3600)
        minutes = int((epoch_time % 3600) // 60)

        clear_lines(1)
        print (f'\r[Epoch {(epoch + 1):04} Step {step:08} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time_str}+{train_time_str}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}, Loss L1: {loss_l1_str}')
        # if len(epoch_loss) < 9999:
        #    print(f'\r[Epoch] Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f} LPIPS: {lpips_val:.4f}')
        #else:
        #    print(f'\r[Last 10K] Min: {window_min:.6f} Avg: {smoothed_window_loss:.6f}, Max: {window_max:.6f} LPIPS: {lpips_window_val:.4f} [Epoch] Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f} LPIPS: {lpips_val:.4f}')
        
        if ( idx + 1 ) == len(dataset):
            steps_loss = []
            epoch_loss = []
            psnr_list = []
            lpips_list = []
            epoch = epoch + 1
            batch_idx = 0

            while  ( idx + 1 ) == len(dataset):
                img0, img1, img2, ratio, idx = dataset[idx]
            dataset.reshuffle()

        batch_idx = batch_idx + 1
        step = step + 1

        continue
        
        
        # flow0 = flow_list[3][:, :2]
        # flow1 = flow_list[3][:, 2:4]
        mask = mask_list[3]
        # output = warp(img0_orig, flow0) * mask + warp(img2_orig, flow1) * (1 - mask)
        
        output = merged[3]
        # warped_img0 = warp(img0, flow_list[3][:, :2])
        # warped_img2 = warp(img2, flow_list[3][:, 2:4])
        # output = warped_img0 * mask_list[3] + warped_img2 * (1 - mask_list[3])

        # self.vgg(merged[3], gt).mean() - self.ss(merged[3], gt) * 0.1

        '''
        x8_output = torch.nn.functional.interpolate(merged[0], scale_factor= 1. / training_scale[0], mode="bilinear", align_corners=False)
        x8_orig = torch.nn.functional.interpolate(img1, scale_factor= 1. / training_scale[0], mode="bilinear", align_corners=False)
        # x8_lpips = torch.mean(loss_fn_lpips.forward(x8_output * 2 - 1, x8_orig * 2 - 1))
        x8_lpips = torch.mean(loss_fn_vgg.forward(x8_output, x8_orig)) - loss_fn_ssim(x8_output, x8_orig) * 0.1
        loss_x8 = pm_weight * criterion_huber(x8_output, x8_orig) + lpips_weight * x8_lpips

        x4_output = torch.nn.functional.interpolate(merged[1], scale_factor= 1. / training_scale[1], mode="bilinear", align_corners=False)
        x4_orig = torch.nn.functional.interpolate(img1, scale_factor= 1. / training_scale[1], mode="bilinear", align_corners=False)
        # x4_lpips = torch.mean(loss_fn_lpips.forward(x4_output * 2 - 1, x4_orig * 2 - 1))
        x4_lpips = torch.mean(loss_fn_vgg.forward(x4_output, x4_orig)) - loss_fn_ssim(x4_output, x4_orig) * 0.1
        loss_x4 = pm_weight * criterion_huber(x4_output, x4_orig) + lpips_weight * x4_lpips

        x2_output = torch.nn.functional.interpolate(merged[2], scale_factor= 1. / training_scale[2], mode="bilinear", align_corners=False)
        x2_orig = torch.nn.functional.interpolate(img1, scale_factor= 1. / training_scale[2], mode="bilinear", align_corners=False)
        # x2_lpips = torch.mean(loss_fn_lpips.forward(x2_output * 2 - 1, x2_orig * 2 - 1))
        x2_lpips = torch.mean(loss_fn_vgg.forward(x2_output, x2_orig)) - loss_fn_ssim(x2_output, x2_orig) * 0.1
        loss_x2 = pm_weight * criterion_huber(x2_output, x2_orig) + lpips_weight * x2_lpips

        x1_output = merged[3]
        x1_orig = img1
        # x1_lpips = torch.mean(loss_fn_lpips.forward(x1_output * 2 - 1, x1_orig * 2 - 1))
        x1_lpips = torch.mean(loss_fn_vgg.forward(x1_output, x1_orig)) - loss_fn_ssim(x1_output, x1_orig) * 0.1
        loss_x1 = pm_weight * criterion_huber(x1_output, x1_orig) + lpips_weight * x1_lpips

        loss = 0.24 * loss_x8 + 0.24 * loss_x4 + 0.24 * loss_x2 + 0.28 * loss_x1
        '''


        x1_output = merged[3]
        x1_orig = img1

        # vgg_loss = torch.mean(loss_fn_vgg.forward(x1_output, x1_orig)) - loss_fn_ssim(x1_output, x1_orig) * 0.1

        # print (f'Out: {output.shape}, Orig: {img1_orig.shape}')

        loss_LPIPS_ = loss_fn_alex(restore_normalized_values(output) * 2 - 1, img1_orig * 2 - 1)
        # loss = (criterion_l1(x1_output, x1_orig)  + 0.1 * (float(torch.mean(loss_LPIPS_).item()) ** 1.1)) / 2

        lpips_weight = 0.5
        loss = (1 - lpips_weight ) * criterion_l1(x1_output, x1_orig) + lpips_weight * 0.2 * float(torch.mean(loss_LPIPS_).item())

        loss_l1 = criterion_l1(restore_normalized_values(output), img1_orig)
        loss_l1_str = str(f'{loss_l1.item():.6f}')

        epoch_loss.append(float(loss_l1.item()))
        steps_loss.append(float(loss_l1.item()))
        lpips_list.append(float(torch.mean(loss_LPIPS_).item()))
        # lpips_list.append(1.)
        psnr_list.append(float(psnr_torch(output, img1)))

        if len(epoch_loss) < 9999:
            smoothed_window_loss = np.mean(moving_average(epoch_loss, 9))
            window_min = min(epoch_loss)
            window_max = max(epoch_loss)
            lpips_window_val = float(np.array(lpips_list).mean())
        else:
            smoothed_window_loss = np.mean(moving_average(epoch_loss[-9999:], 9))
            window_min = min(epoch_loss[-9999:])
            window_max = max(epoch_loss[-9999:])
            lpips_window_val = float(np.array(lpips_list[-9999:]).mean())
        smoothed_loss = float(np.mean(moving_average(epoch_loss, 9)))
        lpips_val = float(np.array(lpips_list).mean())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(flownet.parameters(), 1)

        optimizer_flownet.step()

        try:
            scheduler_flownet.step()
        except Exception as e:
            # if Onecycle is over due to variable number of steps per epoch
            # fall back to Cosine

            print (f'switching to CosineAnnealingLR scheduler:')
            print (f'{e}\n\n')

            current_lr = float(optimizer_flownet.param_groups[0]["lr"])
            scheduler_flownet = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_flownet, 
                T_max=pulse_period, 
                eta_min = current_lr - (( current_lr / 100 ) * pulse_dive)
                )

        train_time = time.time() - time_stamp
        time_stamp = time.time()

        current_state_dict['step'] = int(step)
        current_state_dict['steps_loss'] = list(steps_loss)
        current_state_dict['epoch'] = int(epoch)
        current_state_dict['epoch_loss'] = list(epoch_loss)
        current_state_dict['start_timestamp'] = start_timestamp
        current_state_dict['lr'] = optimizer_flownet.param_groups[0]['lr']
        current_state_dict['model_info'] = model_info
        if args.all_gpus:
            current_state_dict['flownet_state_dict'] = convert_from_data_parallel(flownet.state_dict())
        else:
            current_state_dict['flownet_state_dict'] = flownet.state_dict()
        current_state_dict['optimizer_flownet_state_dict'] = optimizer_flownet.state_dict()
        current_state_dict['trained_model_path'] = trained_model_path

        if step % args.preview == 1:
            rgb_source1 = img0_orig
            rgb_source2 = img2_orig
            rgb_target = img1_orig
            rgb_output = restore_normalized_values(output)
            rgb_output_mask = mask.repeat_interleave(3, dim=1)
            # rgb_refine = refine_list[0] + refine_list[1] + refine_list[2] + refine_list[3]
            # rgb_refine = (rgb_refine + 1) / 2
            # sample_refine = rgb_refine[0].clone().cpu().detach().numpy().transpose(1, 2, 0)

            preview_index = preview_index + 1 if preview_index < 9 else 0

            write_image_queue.put(
                {
                    'preview_folder': os.path.join(args.dataset_path, 'preview'),
                    'preview_index': int(preview_index),
                    'sample_source1': rgb_source1[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_source2': rgb_source2[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_target': rgb_target[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output': rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output_mask': rgb_output_mask[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                }
            )

            del rgb_source1, rgb_source2, rgb_target, rgb_output, rgb_output_mask

        if step % args.save == 1:
            write_model_state_queue.put(deepcopy(current_state_dict))

        data_time += time.time() - time_stamp
        data_time_str = str(f'{data_time:.2f}')
        train_time_str = str(f'{train_time:.2f}')

        epoch_time = time.time() - start_timestamp
        days = int(epoch_time // (24 * 3600))
        hours = int((epoch_time % (24 * 3600)) // 3600)
        minutes = int((epoch_time % 3600) // 60)

        clear_lines(2)
        print (f'\r[Epoch {(epoch + 1):04} Step {step:08} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time_str}+{train_time_str}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}, Loss L1: {loss_l1_str}')
        if len(epoch_loss) < 9999:
            print(f'\r[Epoch] Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f} LPIPS: {lpips_val:.4f}')
        else:
            print(f'\r[Last 10K] Min: {window_min:.6f} Avg: {smoothed_window_loss:.6f}, Max: {window_max:.6f} LPIPS: {lpips_window_val:.4f} [Epoch] Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f} LPIPS: {lpips_val:.4f}')

        if ( idx + 1 ) == len(dataset):
            write_model_state_queue.put(deepcopy(current_state_dict))

            psnr = float(np.array(psnr_list).mean())
            lpips_val = float(np.array(lpips_list).mean())

            epoch_time = time.time() - start_timestamp
            days = int(epoch_time // (24 * 3600))
            hours = int((epoch_time % (24 * 3600)) // 3600)
            minutes = int((epoch_time % 3600) // 60)

            # clear_lines(2)
            # print(f'\rEpoch [{epoch + 1} (Step {step:11} - {days:02}d {hours:02}:{minutes:02}], Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f}, [PSNR] {psnr:.4f}, [LPIPS] {lpips_val:.4f}')
            # print ('\n')

            rows_to_append = [
                {
                    'Epoch': epoch,
                    'Step': step, 
                    'Min': min(epoch_loss),
                    'Avg': smoothed_loss,
                    'Max': max(epoch_loss),
                    'PSNR': psnr,
                    'LPIPS': lpips_val
                 }
            ]
            for row in rows_to_append:
                append_row_to_csv(f'{os.path.splitext(trained_model_path)[0]}.csv', row)

            psnr = 0

            '''
            if args.onecycle != -1:
                if first_pass:
                    first_pass = False
                    optimizer_state_dict = optimizer_flownet.state_dict()
                    scheduler_flownet = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer_flownet,
                        max_lr=args.lr, 
                        total_steps= step * args.onecycle, 
                        )
                    optimizer_flownet.load_state_dict(optimizer_state_dict)
                print (f'setting OneCycleLR after first cycle with max_lr={args.lr}, steps={step}\n\n')
            '''

            steps_loss = []
            epoch_loss = []
            psnr_list = []
            lpips_list = []
            epoch = epoch + 1
            batch_idx = 0

            while  ( idx + 1 ) == len(dataset):
                img0, img1, img2, ratio, idx = read_image_queue.get()
            dataset.reshuffle()

        if ((args.eval > 0) and (step % args.eval) == 1) or (epoch == args.epochs):
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
                preview_folder,
                'eval',
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
                        eval_img0 = read_openexr_file(description['start'])['image_data']
                        eval_img1 = read_openexr_file(description['gt'])['image_data']
                        eval_img2 = read_openexr_file(description['end'])['image_data']

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
            eval_psnr = []
            eval_lpips = []

            if args.eval_half:
                original_float_state_dict = deepcopy(flownet.state_dict())
                flownet.half()
            
            flownet.eval()
            with torch.no_grad():
                # for ev_item_index, description in enumerate(descriptions):
                description = read_eval_image_queue.get()
                while description is not None:
                    ev_item_index = description['ev_item_index']
                    
                    if eval_loss:
                        eval_loss_min = min(eval_loss)
                        eval_loss_max = max(eval_loss)
                        eval_loss_avg = float(np.array(eval_loss).mean())
                    else:
                        eval_loss_min = -1
                        eval_loss_max = -1
                        eval_loss_avg = -1
                    if eval_psnr:
                        eval_psnr_mean = float(np.array(eval_psnr).mean())
                    else:
                        eval_psnr_mean = -1
                    if eval_lpips:
                        eval_lpips_mean = float(np.array(eval_lpips).mean())
                    else:
                        eval_lpips_mean = -1

                    epoch_time = time.time() - start_timestamp
                    days = int(epoch_time // (24 * 3600))
                    hours = int((epoch_time % (24 * 3600)) // 3600)
                    minutes = int((epoch_time % 3600) // 60)

                    clear_lines(2)
                    print (f'\r[Epoch {(epoch + 1):04} Step {step:08} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time_str}+{train_time_str}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}, Loss L1: {loss_l1_str}')
                    print (f'\rEvaluating {ev_item_index} of {len(descriptions)}: Min: {eval_loss_min:.6f} Avg: {eval_loss_avg:.6f}, Max: {eval_loss_max:.6f} LPIPS: {eval_lpips_mean:.4f} PSNR: {eval_psnr_mean:4f}')

                    try:
                        # eval_img0 = read_openexr_file(description['start'])['image_data']
                        # eval_img1 = read_openexr_file(description['gt'])['image_data']
                        # eval_img2 = read_openexr_file(description['end'])['image_data']
                        
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
                        eval_img0 = normalize(eval_img0)
                        eval_img2 = normalize(eval_img2)

                        pvalue = model_info.get('padding', 64)
                        n, c, eh, ew = eval_img0.shape
                        ph = ((eh - 1) // pvalue + 1) * pvalue
                        pw = ((ew - 1) // pvalue + 1) * pvalue
                        padding = (0, pw - ew, 0, ph - eh)
                        
                        eval_img0 = torch.nn.functional.pad(eval_img0, padding)
                        eval_img2 = torch.nn.functional.pad(eval_img2, padding)

                        if args.eval_half:
                            eval_img0 = eval_img0.half()
                            eval_img2 = eval_img2.half()

                        eval_flow_list, eval_mask_list, eval_merged = flownet(
                            eval_img0, 
                            eval_img2,
                            eval_ratio, 
                            iterations = args.iterations
                            )
                        
                        if args.eval_half:
                            eval_flow_list[3] = eval_flow_list[3].float()
                            eval_mask_list[3] = eval_mask_list[3].float()
                        
                        eval_result = warp(eval_img0_orig, eval_flow_list[3][:, :2, :eh, :ew]) * eval_mask_list[3][:, :, :eh, :ew] + warp(eval_img2_orig, eval_flow_list[3][:, 2:4, :eh, :ew]) * (1 - eval_mask_list[3][:, :, :eh, :ew])

                        eval_loss_l1 = criterion_l1(eval_result, eval_img1)
                        eval_loss.append(float(eval_loss_l1.item()))
                        eval_psnr.append(float(psnr_torch(eval_result, eval_img1)))
                        eval_loss_LPIPS_ = loss_fn_alex(eval_result * 2 - 1, eval_img1 * 2 - 1)
                        eval_lpips.append(float(torch.mean(eval_loss_LPIPS_).item()))

                        eval_rgb_output_mask = eval_mask_list[3][:, :, :eh, :ew].repeat_interleave(3, dim=1)

                        if args.eval_save_imgs:
                            write_eval_image_queue.put(
                                {
                                    'preview_folder': eval_folder,
                                    'sample_source1': eval_img0_orig[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                    'sample_source1_name': f'{ev_item_index:08}_incomng.exr',
                                    'sample_source2': eval_img2_orig[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                    'sample_source2_name': f'{ev_item_index:08}_outgoing.exr',
                                    'sample_target': eval_img1[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                    'sample_target_name': f'{ev_item_index:08}_target.exr',
                                    'sample_output': eval_result[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                    'sample_output_name': f'{ev_item_index:08}_output.exr',
                                    'sample_output_mask': eval_rgb_output_mask[0].permute(1, 2, 0).clone().cpu().detach().numpy(),
                                    'sample_output_mask_name': f'{ev_item_index:08}_output_mask.exr'
                                }
                            )

                    except Exception as e:
                        del description['eval_img0']
                        del description['eval_img1']
                        del description['eval_img2']
                        print (f'\nerror while evaluating: {e}\n{description}\n\n')
                    description = read_eval_image_queue.get()

            if args.eval_half:
                flownet.float()
                flownet.load_state_dict(original_float_state_dict, weights_only=False)

            flownet.train()

            eval_rows_to_append = [
                {
                    'Epoch': epoch,
                    'Step': step, 
                    'Min': eval_loss_min,
                    'Avg': eval_loss_avg,
                    'Max': eval_loss_max,
                    'PSNR': eval_psnr_mean,
                    'LPIPS': eval_lpips_mean
                 }
            ]

            for eval_row in eval_rows_to_append:
                append_row_to_csv(f'{os.path.splitext(trained_model_path)[0]}.eval.csv', eval_row)

            clear_lines(2)
            print(f'\r[Epoch {(epoch + 1):04} Step {step:08} - {days:02}d {hours:02}:{minutes:02}], Eval Min: {eval_loss_min:.6f} Avg: {eval_loss_avg:.6f}, Max: {eval_loss_max:.6f}, [PSNR] {eval_psnr_mean:.4f}, [LPIPS] {eval_lpips_mean:.4f}')
            print ('\n')


            if not args.eval_keep_all:
            # print (f'prev folder: {prev_eval_folder}\n\n')
                if prev_eval_folder:
                    # print (f'exec "rm -rf {os.path.abspath(prev_eval_folder)}"\n\n')
                    if os.path.isdir(prev_eval_folder):
                        clean_thread = threading.Thread(target=lambda: os.system(f'rm -rf {os.path.abspath(prev_eval_folder)}')).start()
                    # os.system(f'rm -rf {os.path.abspath(prev_eval_folder)}')
            prev_eval_folder = eval_folder

        batch_idx = batch_idx + 1
        step = step + 1

        del img0, img1, img2, img0_orig, img1_orig, img2_orig, flow_list, mask_list, merged, mask, output

        if epoch == args.epochs:
            sys.exit()

if __name__ == "__main__":
    main()

