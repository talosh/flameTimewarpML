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

class MinExrReader:
    '''Minimal, standalone OpenEXR reader for single-part, uncompressed scan line files.

    This OpenEXR reader makes a couple of assumptions
    - single-part files with arbitrary number of channels,
    - no pixel data compression, and
    - equal channel types (HALF, FLOAT, UINT).

    These assumptions allow us to efficiently parse and read the `.exr` file. In particular
    we gain constant offsets between scan lines which allows us to read the entire image
    in (H,C,W) format without copying.

    Use `MinimalEXR.select` to select a subset of channels in the given order. `MinimalEXR.select`
    tries to be smart when copying is required and when views are ok.
    
    Based on the file format presented in
    https://www.openexr.com/documentation/openexrfilelayout.pdf

    Attributes
    ----------
    shape: tuple
        Shape of image in (H,C,W) order
    image: numpy.array
        Uncompressed image data.
    attrs: dict
        OpenEXR header attributes.
    '''

    class BufferReader:
        '''A lightweight io.BytesIO object with convenience functions.
        
        Params
        ------
        data : bytes-like
            Bytes for which random access is required.
        
        '''

        def __init__(self, data):
            self.data = data
            self.len = len(data)
            self.off = 0

        def read(self, n):
            '''Read next `n` bytes.'''
            v = self.data[self.off:self.off+n]
            self.off += n
            return v

        def read_null_string(self):
            import ctypes
            '''Read a null-terminated string.'''
            s = ctypes.create_string_buffer(self.data[self.off:]).value
            if s != None:
                s = s.decode('utf-8')
                self.off += len(s) + 1
            return s

        def peek(self):
            '''Peek next byte.'''
            return self.data[self.off]

        def advance(self, n):
            '''Advance offset by `n` bytes.'''
            self.off += n

        def nleft(self):
            '''Returns the number of bytes left to read.'''
            return self.len - self.off - 1

    def __init__(self, fp, header_only = False):
        self.fp = fp
        self.image = None
        self.shape = None

        self._read_header()
        if not header_only:
            self._read_image()

    def select(self, channels, channels_last=True):
        import numpy as np
        '''Returns an image composed only of the given channels.
        
        Attempts to be smart about memory views vs. memory copies.

        Params
        ------
        channels: list-like
            Names of channels to be extracted. Appearance in list
            also defines the order of the channels. 
        channels_last: bool, optional
            When true return image in (H,W,C) format.

        Returns
        -------
        image: HxWxC or HxCxW array
            Selected image data.
        '''
        H,C,W = self.shape
        ids = [self.channel_map[c] for c in channels]                
        if len(ids) == 0:
            img = np.empty((H,0,W), dtype=self.image.dtype)
        else:
            diff = np.diff(ids)
            sH = slice(0, H)
            sW = slice(0, W)
            if len(diff) == 0:
                # single channel select, return view
                sC = slice(ids[0],ids[0]+1)
                img = self.image[sH,sC,sW]
            elif len(set(diff)) == 1:
                # mutliple channels, constant offset between, return view
                # Careful here with negative steps, ie. diff[0] < 0:
                start = ids[0]
                step = diff[0]
                end = ids[-1]+diff[0]
                end = None if end < 0 else end                
                sC = slice(start,end,step)
                img = self.image[sH,sC,sW]
            else:
                # multiple channels not slicable -> copy mem
                chdata = [self.image[sH,i:i+1,sW] for i in ids]
                img = np.concatenate(chdata, 1)
        
        if channels_last:
            img = img.transpose(0,2,1)
        return img

    def _read_header(self):
        import numpy as np
        import struct

        self.fp.seek(0)        
        buf = self.BufferReader(self.fp.read(10000))

        # Magic and version and info bits
        magic, version, b2, b3, b4 = struct.unpack('<iB3B', buf.read(8))
        assert magic == 20000630, 'Not an OpenEXR file.'
        assert b2 in (0, 4), 'Not a single-part scan line file.'
        assert b3 == b4 == 0, 'Unused flags in version field are not zero.'

        # Header attributes
        self.attrs = self._read_header_attrs(buf)

        # Parse channels and datawindow
        self.compr = self._parse_compression(self.attrs)        
        self.channel_names, self.channel_types = self._parse_channels(self.attrs)
        self.channel_map = {cn:i for i,cn in enumerate(self.channel_names)}
        H, W = self._parse_data_window(self.attrs)
        self.shape = (H,len(self.channel_names),W)
        self.first_offset = self._read_first_offset(buf)
        
        # Assert our assumptions
        assert self.compr == 0x00, 'Compression not supported.'
        assert len(set(self.channel_types)) <= 1, 'All channel types must be equal.'

    def _read_image(self):
        import numpy as np
        # Here is a shortcut: We assume all channels of the same type and thus constant offsets between
        # scanlines (SOFF). Note, each scanline has a header (y-coordinate (int4), data size DS (int4)) and data in scanlines
        # is stored consecutively for channels (in order of appearance in header). Thus we can interpret the content
        # as HxCxW image with strides: (SOFF,DS*W,DS)
        H,C,W = self.shape

        if np.prod(self.shape) == 0:
            return np.empty(self.shape, dtype=np.float32)

        dtype  = self.channel_types[0]
        DS = np.dtype(dtype).itemsize
        SOFF = 8+DS*W*C        
        strides = (SOFF, DS*W, DS)
        nbytes = SOFF*H

        self.fp.seek(self.first_offset, 0)
        image = np.frombuffer(self.fp.read(nbytes), dtype=dtype, count=-1, offset=8)
        self.image = np.lib.stride_tricks.as_strided(image, (H,C,W), strides)

    def _read_header_attrs(self, buf):
        attrs = {}
        while buf.nleft() > 0:
            attr = self._read_header_attr(buf)
            if attr is None:
                break
            attrs[attr[0]] = attr
        return attrs

    def _read_header_attr(self, buf):
        import struct
        if buf.peek() == 0x00:
            buf.advance(1)
            return None
        aname = buf.read_null_string()
        atype = buf.read_null_string()
        asize = struct.unpack('<i', buf.read(4))[0]
        data = buf.read(asize)
        return (aname, atype, asize, data)

    def _parse_channels(self, attrs):
        import struct
        import numpy as np

        attr = attrs['channels']
        assert attr[1] == 'chlist', 'Unexcepted type for channels attribute.'
        buf = self.BufferReader(attr[-1])
        channel_names, channel_types = [], []
        PT_LOOKUP = [np.uint32, np.float16, np.float32]
        while buf.nleft() > 0 and buf.peek() != 0x00:            
            channel_names.append(buf.read_null_string())
            pt = struct.unpack('<i', buf.read(4))[0]
            channel_types.append(PT_LOOKUP[pt])
            buf.advance(12) # skip remaining entries
        if buf.nleft() > 0:
            buf.advance(1) # account for zero byte
        return channel_names, channel_types

    def _parse_data_window(self, attrs):
        import struct
        attr = attrs['dataWindow']
        assert attr[1] == 'box2i', 'Unexcepted type for dataWindow attribute.'
        xmin, ymin, xmax, ymax = struct.unpack('<iiii', attr[-1])
        return (ymax-ymin+1, xmax-xmin+1)

    def _parse_compression(self, attrs):
        return attrs['compression'][-1][0]

    def _read_offsets(self, buf):
        import struct
        offsets = []
        while buf.nleft() > 0 and buf.peek() != 0x00:
            o = struct.unpack('<Q', buf.read(8))[0]
            offsets.append(o)
        if buf.nleft() > 0:
            buf.advance(1) # account for zero byte
        return offsets

    def _read_first_offset(self, buf):
        import struct
        assert buf.nleft() > 0 and buf.peek() != 0x00, 'Failed to read offset.'
        return struct.unpack('<Q', buf.read(8))[0]

def read_openexr_file(file_path, header_only = False):
    """
    Reads data from an OpenEXR file specified by the file path.

    This function opens an OpenEXR file and reads its contents, either the header information only or the full data, including image data. It utilizes the MinExrReader to process the file.

    Parameters:
    - file_path (str): Path to the OpenEXR file to be read.
    - header_only (bool, optional): If True, only header information is read. Defaults to False.

    Returns:
    - dict: A dictionary containing the OpenEXR file's metadata and image data (if header_only is False). The dictionary includes the following keys:
        - 'attrs': Attributes of the OpenEXR file.
        - 'compr': Compression type used in the OpenEXR file.
        - 'channel_names': Names of the channels in the OpenEXR file.
        - 'channel_types': Data types of the channels in the OpenEXR file.
        - 'shape': The shape of the image data, rearranged as (height, width, channels).
        - 'image_data': Numpy array of the image data if header_only is False. The data is transposed to match the shape (height, width, channels).

    Note:
    - The function uses a context manager to ensure the file is properly closed after reading.
    - It assumes the existence of a class method `MinExrReader` for reading the OpenEXR file.
    """

    import numpy as np
    with open(file_path, 'rb') as sfp:
        source_reader = MinExrReader(sfp, header_only)
        result = {
            'attrs': source_reader.attrs,
            'compr': source_reader.compr,
            'channel_names': source_reader.channel_names,
            'channel_types': source_reader.channel_types,
            'shape': (source_reader.shape[0], source_reader.shape[2], source_reader.shape[1]),
        }
        if not header_only:
            result['image_data'] = source_reader.image.transpose(0, 2, 1)[:, :, ::-1].copy()
        del source_reader
    return result

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
        acescc_rate = 40,
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
            self.last_train_data_size = 11
            self.last_train_data = [self.frames_queue.get()] * self.last_train_data_size
            self.new_sample_shown = False
            self.train_data_index = 0

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

                            '''
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
            resized_tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

            # Adjust tensor shape back to [h, w, c]
            resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)

            del tensor

            return resized_tensor

        def getimg(self, index):
            '''
            if not self.last_train_data:
                new_data = self.frames_queue.get_nowait()
                self.last_train_data = [new_data]
                del new_data
            if self.repeat_counter >= self.repeat_count:
                try:
                    if len(self.last_train_data) == self.last_train_data_size:
                        data_to_delete = self.last_train_data.pop(0)
                        del data_to_delete
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
            else:
                return self.last_train_data[random.randint(0, len(self.last_train_data) - 1)]
            '''

            '''
            if self.repeat_counter >= self.repeat_count:
                try:
                    self.last_train_data = [self.frames_queue.get_nowait()]
                except queue.Empty:
                    pass
                self.repeat_counter = 0
                self.train_data_index = self.last_train_data[0]['index']
            else:
                self.repeat_counter += 1
            
            return self.last_train_data[0]
            '''

            if self.repeat_counter >= self.repeat_count:
                self.repeat_counter = 1
                try:
                    new_data = self.frames_queue.get_nowait()
                    self.last_train_data[random.randint(0, len(self.last_train_data) - 1)] = new_data
                    self.train_data_index = new_data['index']
                    return new_data
                except queue.Empty:
                    return random.choice(self.last_train_data)
            else:
                self.repeat_counter += 1
                return random.choice(self.last_train_data)

            # return self.frames_queue.get()

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
                    
                    if random.uniform(0, 1) < (self.generalize / 100):
                        # add noise
                        if random.uniform(0, 1) < 0.2:
                            delta = random.uniform(0, 1e-2)
                            img0 += torch.rand_like(img0) * delta
                            img1 += torch.rand_like(img1) * delta
                            img2 += torch.rand_like(img1) * delta

                    def gamma_up(img, gamma = 1.18):
                        return torch.sign(img) * torch.pow(torch.abs(img), 1 / gamma )
                    
                    if random.uniform(0, 1) < (self.generalize / 100):
                        if random.uniform(0, 1) < 0.44:
                            gamma = random.uniform(0.9, 1.9)
                            img0 = gamma_up(img0, gamma=gamma)
                            img1 = gamma_up(img1, gamma=gamma)
                            img2 = gamma_up(img2, gamma=gamma)

                # Convert to ACEScc
                if random.uniform(0, 1) < (self.acescc_rate / 100):
                    img0 = self.apply_acescc(torch.clamp(img0, min=0.01))
                    img1 = self.apply_acescc(torch.clamp(img1, min=0.01))
                    img2 = self.apply_acescc(torch.clamp(img2, min=0.01))
                
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

current_state_dict = {}

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
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
    parser.add_argument('--type', type=int, default=1, help='Model type (int): 1 - MultiresNet, 2 - MultiresNet 4 (default: 1)')
    parser.add_argument('--pulse', type=float, default=10000, help='Period in steps to pulse learning rate (float) (default: 10K)')
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
    parser.add_argument('--freeze', action='store_true', dest='freeze', default=False, help='Freeze custom hardcoded parameters')
    parser.add_argument('--acescc', type=check_range_percent, default=40, help='Percentage of ACEScc encoded frames (default: 40))')
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
    max_dataset_window = 11
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

    '''
    frame_size = closest_divisible(abs(int(args.frame_size)), model_info.get('padding', 64))
    if frame_size != args.frame_size:
        print (f'Frame size should be divisible by 64 for training. Using {frame_size}')
    '''

    frame_size = args.frame_size

    dataset = get_dataset(
        args.dataset_path, 
        batch_size=args.batch_size, 
        device=device, 
        frame_size=frame_size,
        max_window=max_dataset_window,
        acescc_rate=args.acescc,
        generalize=args.generalize,
        repeat=args.repeat,
        sequential = args.sequential
        )
    
    if args.eval_folder:
        print (f'Scanning data for evaluation:')
        eval_dataset = get_dataset(
        args.eval_folder, 
        batch_size=args.batch_size,
        device=device, 
        frame_size=frame_size,
        max_window=max_dataset_window,
        acescc_rate=args.acescc,
        generalize=args.generalize,
        repeat=args.repeat,
        sequential = args.sequential
        )
    else:
        eval_dataset = dataset

    def write_images(write_image_queue):
        while True:
            try:
                write_data = write_image_queue.get_nowait()
                preview_index = write_data.get('preview_index', 0)
                preview_folder = write_data["preview_folder"]
                if not os.path.isdir(preview_folder):
                    os.makedirs(preview_folder)
                write_exr(write_data['sample_source1'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_A_incomng.exr'), half_float = True)
                write_exr(write_data['sample_source2'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_B_outgoing.exr'), half_float = True)
                write_exr(write_data['sample_target'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_C_target.exr'), half_float = True)
                write_exr(write_data['sample_output'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_D_output.exr'), half_float = True)
                write_exr(write_data['sample_output_diff'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_E_output_diff.exr'), half_float = True)
                write_exr(write_data['sample_output_conf'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_F_output_conf.exr'), half_float = True)
                write_exr(write_data['sample_output_mask'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_G_output_mask.exr'), half_float = True)
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
                write_exr(write_data['sample_output_diff'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_output_diff_name']), half_float = True)
                write_exr(write_data['sample_output_conf'].astype(np.float16), os.path.join(write_data['preview_folder'], write_data['sample_output_conf_name']), half_float = True)
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

    write_image_queue = queue.Queue(maxsize=16)
    write_thread = threading.Thread(target=write_images, args=(write_image_queue, ))
    write_thread.daemon = True
    write_thread.start()
    
    write_eval_image_queue = queue.Queue(maxsize=args.eval_buffer)
    write_eval_thread = threading.Thread(target=write_eval_images, args=(write_eval_image_queue, ))
    write_eval_thread.daemon = True
    write_eval_thread.start()

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
    optimizer_flownet = torch.optim.AdamW(flownet.parameters(), lr=lr, weight_decay=weight_decay)

    if args.weight_decay != -1:
        weight_decay = args.weight_decay

    if args.generalize == 0:
        print (f'Disabling augmentation and setting weight decay to {weight_decay:.2e}')
    elif args.generalize == 1:
        print (f'Setting augmentation to horizontal flip and scale only and weight decay to {weight_decay:.2e}')
    else:
        print (f'Setting augmentation rate to {args.generalize}% and weight decay to {weight_decay:.2e}')

    step = 0
    loaded_step = 0
    current_epoch = 0
    preview_index = 0

    if args.state_file:
        trained_model_path = args.state_file

        try:
            checkpoint = torch.load(trained_model_path, map_location=device)
            print('loaded previously saved model checkpoint')
        except Exception as e:
            print (f'unable to load saved model: {e}')

        try:
            if args.all_gpus:
                missing_keys, unexpected_keys = flownet.load_state_dict(convert_to_data_parallel(checkpoint['flownet_state_dict']), strict=False)
            else:
                missing_keys, unexpected_keys = flownet.load_state_dict(checkpoint['flownet_state_dict'], strict=False)
            print('loaded previously saved Flownet state')
            if missing_keys:
                print (f'\nMissing keys:\n{missing_keys}\n')
            if unexpected_keys:
                print (f'\nUnexpected keys:\n{unexpected_keys}\n')
        except Exception as e:
            print (f'unable to load Flownet state: {e}')

        try:
            optimizer_flownet.load_state_dict(checkpoint['optimizer_flownet_state_dict'])
            print('loaded previously saved optimizer state')
        except Exception as e:
            print (f'unable to load optimizer state: {e}')

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
        rife_state_dict = torch.load(args.legacy_model)
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        missing_keys, unexpected_keys = flownet.load_state_dict(convert(rife_state_dict), strict=False)
        print (f'\nMissing keys:\n{missing_keys}\n')
        print (f'\nUnexpected keys:\n{unexpected_keys}\n')

    if args.reset_stats:
        step = 0
        loaded_step = 0
        current_epoch = 0
        preview_index = 0
    
    if args.onecycle != -1:
        try:
            scheduler_flownet = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_flownet,
                max_lr=args.lr,
                div_factor = 4,
                final_div_factor = 1,
                steps_per_epoch=len(dataset)*dataset.repeat_count, 
                epochs=args.onecycle,
                last_epoch = -1 if loaded_step == 0 else loaded_step
                )
        except:
            scheduler_flownet = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_flownet,
                max_lr=args.lr,
                div_factor = 4,
                final_div_factor = 1,
                steps_per_epoch=len(dataset)*dataset.repeat_count, 
                epochs=args.onecycle,
                last_epoch = -1
                )
        print (f'setting OneCycleLR with max_lr={args.lr}, steps_per_epoch={len(dataset)*dataset.repeat_count}, epochs={args.onecycle}, last: {-1 if loaded_step == 0 else loaded_step}')
        args.epochs = args.onecycle
    else:
        scheduler_flownet = torch.optim.lr_scheduler.CyclicLR(
                        optimizer_flownet,
                        base_lr=lr - (( lr / 100 ) * pulse_dive),  # Lower boundary of the learning rate cycle
                        max_lr=lr,    # Upper boundary of the learning rate cycle
                        step_size_up=pulse_period,  # Number of iterations for the increasing part of the cycle
                        mode='exp_range',  # Use exp_range to enable scale_fn
                        cycle_momentum=False,
                        scale_fn=sinusoidal_scale_fn,  # Custom sinusoidal function
                        scale_mode='cycle'  # Apply scaling once per cycle
                    )

    # LPIPS Init

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    import lpips
    os.environ['TORCH_HOME'] = os.path.abspath(os.path.dirname(__file__))
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex.to(device)
    # loss_fn_lpips = lpips.LPIPS(net='vgg', lpips=False, spatial=True)
    # loss_fn_lpips.to(device)

    # loss_fn_ssim = SSIM()
    # loss_fn_vgg = VGGPerceptualLoss().to(device)

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

        # for param in flownet.encode.parameters():
        #    param.requires_grad = False

        '''
        for param in flownet.block0.conv0.parameters():
            param.requires_grad = False
        '''

        # for param in flownet.module.encode.parameters():
        #    param.requires_grad = False
        # for param in flownet.module.block0.parameters():
        #    param.requires_grad = False     
        for param in flownet.module.block1.parameters():
            param.requires_grad = False
        for param in flownet.module.block2.parameters():
            param.requires_grad = False
        for param in flownet.module.block3.parameters():
            param.requires_grad = False
        for param in flownet.module.block4.parameters():
            param.requires_grad = False

        '''
        for param in flownet.block0.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block1.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block2.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block3.convblock.parameters():
            param.requires_grad = False
        '''

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

    try:
        print ()
        print (f'flownet.module.encoder.attn.channel_scale {flownet.module.encode.attn.channel_scale.data}')
        print (f'flownet.module.encoder.attn.channel_offset {flownet.module.encode.attn.channel_offset.data}')
        print (f'flownet.module.encoder.attn.spatial_scale {flownet.module.encode.attn.spatial_scale.data}')
        print (f'flownet.module.encoder.attn.spatial_offset {flownet.module.encode.attn.spatial_offset.data}')

        print ()
        print (f'flownet.module.block0.attn.channel_scale {flownet.module.block0.attn.channel_scale.data}')
        print (f'flownet.module.block0.attn.channel_offset {flownet.module.block0.attn.channel_offset.data}')
        print (f'flownet.module.block0.attn.spatial_scale {flownet.module.block0.attn.spatial_scale.data}')
        print (f'flownet.module.block0.attn.spatial_offset {flownet.module.block0.attn.spatial_offset.data}')
        print (f'flownet.module.block0.noise_level {flownet.module.block0.noise_level.data}')
        # print (f'flownet.module.block0.mix.noise_level {flownet.module.block0.mix.theta.data}')

        print ()
        print (f'flownet.module.block1.attn.channel_scale {flownet.module.block1.attn.channel_scale.data}')
        print (f'flownet.module.block1.attn.channel_offset {flownet.module.block1.attn.channel_offset.data}')
        print (f'flownet.module.block1.attn.spatial_scale {flownet.module.block1.attn.spatial_scale.data}')
        print (f'flownet.module.block1.attn.spatial_offset {flownet.module.block1.attn.spatial_offset.data}')
        print (f'flownet.module.block1.noise_level {flownet.module.block1.noise_level.data}')

        print ()
        print (f'flownet.module.block2.attn.channel_scale {flownet.module.block2.attn.channel_scale.data}')
        print (f'flownet.module.block2.attn.channel_offset {flownet.module.block2.attn.channel_offset.data}')
        print (f'flownet.module.block2.attn.spatial_scale {flownet.module.block2.attn.spatial_scale.data}')
        print (f'flownet.module.block2.attn.spatial_offset {flownet.module.block2.attn.spatial_offset.data}')
    except:
        pass

    print('\n\n')

    # print("\n"*12)

    current_state_dict['step'] = int(step)
    current_state_dict['epoch'] = int(epoch)
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
            exit_event.set()  # Signal threads to stop
            process_exit_event.set()  # Signal processes to stop
            exit(0)
            # signal.signal(signum, signal.SIG_DFL)
            # os.kill(os.getpid(), signal.SIGINT)
        return graceful_exit
    signal.signal(signal.SIGINT, create_graceful_exit(current_state_dict))

    min_l1 = float(sys.float_info.max)
    avg_l1 = 0
    max_l1 = 0
    avg_pnsr = 0
    avg_lpips = 0

    preview_maxmin_steps = args.preview_maxmin_steps if args.preview_maxmin_steps < len(dataset)*dataset.repeat_count else len(dataset)*dataset.repeat_count
    max_values = MaxNValues(n=args.preview_max if args.preview_max else 10)
    min_values = MinNValues(n=args.preview_min if args.preview_min else 10)

    data_time = 0
    data_time1 = 0
    data_time2 = 0
    train_time = 0

    while True:
        # tracemalloc.start()
        # data_time = time.time() - time_stamp
        time_stamp = time.time()

        img0, img1, img2, ratio, idx, current_desc = dataset[batch_idx]

        data_time = time.time() - time_stamp
        time_stamp = time.time()

        img0 = img0.to(device, non_blocking = True)
        img1 = img1.to(device, non_blocking = True)
        img2 = img2.to(device, non_blocking = True)
        img0_orig = img0.detach().clone()
        img1_orig = img1.detach().clone()
        img2_orig = img2.detach().clone()
        img0 = normalize(img0)
        img1 = normalize(img1)
        img2 = normalize(img2)

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

        # del img0, img1, img2, img0_orig, img1_orig, img2_orig
        # continue

        data_time1 = time.time() - time_stamp
        time_stamp = time.time()

        flownet.train()
        
        flow_list, mask_list, conf_list, merged = flownet(
            img0,
            img2,
            ratio,
            scale=training_scale,
            iterations = args.iterations
            )

        # continue

        flow0 = flow_list[-1][:, :2]
        flow1 = flow_list[-1][:, 2:4]
        mask = mask_list[-1]
        conf = conf_list[-1]

        output = merged[-1]
        output_clean = warp(img0_orig, flow0) * mask + warp(img2_orig, flow1) * (1 - mask)

        diff_matte = diffmatte(output_clean, img1_orig)
        loss_conf = criterion_l1(conf, diff_matte)

        lpips_weight = 0.5
        loss_LPIPS = loss_fn_alex(output_clean * 2 - 1, img1_orig * 2 - 1)
        loss = (1 - lpips_weight ) * criterion_l1(output, img1) + lpips_weight * 0.2 * float(torch.mean(loss_LPIPS).item()) + 0.01 * loss_conf
        loss_l1 = criterion_l1(output_clean, img1_orig)

        # del img0, img1, img2, img0_orig, img1_orig, img2_orig, flow_list, mask_list, conf_list, merged, flow0, flow1, output, output_clean, diff_matte, loss_LPIPS
        # continue

        min_l1 = min(min_l1, float(loss_l1.item()))
        max_l1 = max(max_l1, float(loss_l1.item()))
        avg_l1 = float(loss_l1.item()) if batch_idx == 0 else (avg_l1 * (batch_idx - 1) + float(loss_l1.item())) / batch_idx 
        avg_lpips = float(torch.mean(loss_LPIPS).item()) if batch_idx == 0 else (avg_lpips * (batch_idx - 1) + float(torch.mean(loss_LPIPS).item())) / batch_idx
        avg_pnsr = float(psnr_torch(output, img1)) if batch_idx == 0 else (avg_pnsr * (batch_idx - 1) + float(psnr_torch(output, img1))) / batch_idx

        loss.backward()
        torch.nn.utils.clip_grad_norm_(flownet.parameters(), 1)

        if platform.system() == 'Darwin':
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize(device=device)

        optimizer_flownet.step()

        try:
            scheduler_flownet.step()
        except Exception as e:
            # if Onecycle is over due to variable number of steps per epoch
            # fall back to Cosine

            current_lr = float(optimizer_flownet.param_groups[0]["lr"])
            max_lr = current_lr + (( current_lr / 100 ) * pulse_dive)
            print (f'switching to CyclicLR scheduler with base {current_lr} and max {max_lr}')
            print (f'\n\n')

            scheduler_flownet = torch.optim.lr_scheduler.CyclicLR(
                            optimizer_flownet,
                            base_lr=current_lr,  # Lower boundary of the learning rate cycle
                            max_lr=max_lr,    # Upper boundary of the learning rate cycle
                            step_size_up=pulse_period,  # Number of iterations for the increasing part of the cycle
                            cycle_momentum=False,
                            mode='exp_range',  # Use exp_range to enable scale_fn
                            scale_fn=sinusoidal_scale_fn,  # Custom sinusoidal function
                            scale_mode='cycle'  # Apply scaling once per cycle
                        )
            
            # current_lr = float(optimizer_flownet.param_groups[0]["lr"])
            
            '''
            scheduler_flownet = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_flownet,
                T_max=pulse_period, 
                eta_min = lr/10 - (( lr / 100 ) * pulse_dive)
                )
            '''

        train_time = time.time() - time_stamp
        time_stamp = time.time()

        # del img0, img1, img2, img0_orig, img1_orig, img2_orig, flow0, flow1, flow_list, mask, mask_list, conf, conf_list, merged, output, output_clean, diff_matte
        # continue

        current_state_dict['step'] = int(step)
        current_state_dict['epoch'] = int(epoch)
        current_state_dict['start_timestamp'] = start_timestamp
        current_state_dict['lr'] = optimizer_flownet.param_groups[0]['lr']
        current_state_dict['model_info'] = model_info
        if args.all_gpus:
            current_state_dict['flownet_state_dict'] = convert_from_data_parallel(flownet.state_dict())
        else:
            current_state_dict['flownet_state_dict'] = flownet.state_dict()
        current_state_dict['optimizer_flownet_state_dict'] = optimizer_flownet.state_dict()
        current_state_dict['trained_model_path'] = trained_model_path

        if step % args.save == 1:
            write_model_state_queue.put(deepcopy(current_state_dict))

        if step % args.preview == 1:
            rgb_source1 = img0_orig
            rgb_source2 = img2_orig
            rgb_target = img1_orig
            rgb_output = output_clean
            rgb_output_mask = mask.repeat_interleave(3, dim=1)
            rgb_output_conf = conf.repeat_interleave(3, dim=1)
            rgb_output_diff = diff_matte.repeat_interleave(3, dim=1)
            # rgb_refine = refine_list[0] + refine_list[1] + refine_list[2] + refine_list[3]
            # rgb_refine = (rgb_refine + 1) / 2
            # sample_refine = rgb_refine[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
            
            preview_index += 1
            preview_index = preview_index if preview_index < 10 else 0

            write_image_queue.put(
                {
                    'preview_folder': os.path.join(args.dataset_path, 'preview', os.path.splitext(os.path.basename(trained_model_path))[0]),
                    'preview_index': int(preview_index),
                    'sample_source1': rgb_source1[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_source2': rgb_source2[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_target': rgb_target[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output': rgb_output[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output_mask': rgb_output_mask[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output_conf': rgb_output_conf[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_output_diff': rgb_output_diff[0].clone().cpu().detach().numpy().transpose(1, 2, 0)
                }
            )

            del rgb_source1, rgb_source2, rgb_target, rgb_output, rgb_output_mask

        current_desc['loss'] = float(loss.item())
        current_desc['loss_l1'] = float(loss_l1.item())
        current_desc['lpips'] = float(torch.mean(loss_LPIPS).item())

        min_max_item = {
                'description': current_desc,
                'img0_orig': img0_orig.numpy(force=True).copy(),
                'img1_orig': img1_orig.numpy(force=True).copy(),
                'img2_orig': img2_orig.numpy(force=True).copy(),
                'diff': diff_matte.repeat_interleave(3, dim=1).numpy(force=True).copy(),
                'conf': conf.repeat_interleave(3, dim=1).numpy(force=True).copy(),
                'mask': mask.repeat_interleave(3, dim=1).numpy(force=True).copy(),
                'output': output_clean.numpy(force=True).copy(),
        }

        max_values.add(loss.item(), min_max_item)
        min_values.add(loss.item(), min_max_item)

        if (args.preview_max > 0) and ((step % preview_maxmin_steps) == 1 or ( idx + 1 ) == len(dataset)):
            max_preview_folder = os.path.join(
                args.dataset_path,
                'preview',
                os.path.splitext(os.path.basename(trained_model_path))[0],
                'max')
            if not os.path.isdir(max_preview_folder):
                os.makedirs(max_preview_folder)
            max_loss_values = max_values.get_values()
            index = 0
            item = None
            for index, item in enumerate(max_loss_values):
                item_data = item[1]
                n, c, h, w = item_data['img0_orig'].shape
                for b_indx in range(n):
                    write_eval_image_queue.put(
                    {
                        'preview_folder': max_preview_folder,
                        'sample_source1': item_data['img0_orig'][b_indx].transpose(1, 2, 0),
                        'sample_source1_name': f'{index:04}_{b_indx:02}_A_incomng.exr',
                        'sample_source2': item_data['img2_orig'][b_indx].transpose(1, 2, 0),
                        'sample_source2_name': f'{index:04}_{b_indx:02}_B_outgoing.exr',
                        'sample_target': item_data['img1_orig'][b_indx].transpose(1, 2, 0),
                        'sample_target_name': f'{index:04}_{b_indx:02}_C_target.exr',
                        'sample_output': item_data['output'][b_indx].transpose(1, 2, 0),
                        'sample_output_name': f'{index:04}_{b_indx:02}_D_output.exr',
                        'sample_output_diff': item_data['diff'][b_indx].transpose(1, 2, 0),
                        'sample_output_diff_name': f'{index:04}_{b_indx:02}_E_diff.exr',
                        'sample_output_conf': item_data['conf'][b_indx].transpose(1, 2, 0),
                        'sample_output_conf_name': f'{index:04}_{b_indx:02}_F_conf.exr',
                        'sample_output_mask': item_data['mask'][b_indx].transpose(1, 2, 0),
                        'sample_output_mask_name': f'{index:04}_{b_indx:02}_G_mask.exr',
                    }
                    )
                    json_filename = os.path.join(
                        max_preview_folder,
                        f'{index:04}_{b_indx:02}.json'
                    )
                    with open(json_filename, 'w', encoding='utf-8') as json_file:
                        json.dump(item_data['description'], json_file, indent=4, ensure_ascii=False)
            del index, item

        if (args.preview_min > 0) and ((step % preview_maxmin_steps) == 1 or ( idx + 1 ) == len(dataset)):
            min_preview_folder = os.path.join(
                args.dataset_path,
                'preview',
                os.path.splitext(os.path.basename(trained_model_path))[0],
                'min')
            if not os.path.isdir(min_preview_folder):
                os.makedirs(min_preview_folder)
            min_loss_values = min_values.get_values()
            index = 0
            item = None
            for index, item in enumerate(min_loss_values):
                item_data = item[1]
                n, c, h, w = item_data['img0_orig'].shape
                for b_indx in range(n):
                    write_eval_image_queue.put(
                    {
                        'preview_folder': min_preview_folder,
                        'sample_source1': item_data['img0_orig'][b_indx].transpose(1, 2, 0),
                        'sample_source1_name': f'{index:04}_{b_indx:02}_A_incomng.exr',
                        'sample_source2': item_data['img2_orig'][b_indx].transpose(1, 2, 0),
                        'sample_source2_name': f'{index:04}_{b_indx:02}_B_outgoing.exr',
                        'sample_target': item_data['img1_orig'][b_indx].transpose(1, 2, 0),
                        'sample_target_name': f'{index:04}_{b_indx:02}_C_target.exr',
                        'sample_output': item_data['output'][b_indx].transpose(1, 2, 0),
                        'sample_output_name': f'{index:04}_{b_indx:02}_D_output.exr',
                        'sample_output_diff': item_data['diff'][b_indx].transpose(1, 2, 0),
                        'sample_output_diff_name': f'{index:04}_{b_indx:02}_E_diff.exr',
                        'sample_output_conf': item_data['conf'][b_indx].transpose(1, 2, 0),
                        'sample_output_conf_name': f'{index:04}_{b_indx:02}_F_conf.exr',
                        'sample_output_mask': item_data['mask'][b_indx].transpose(1, 2, 0),
                        'sample_output_mask_name': f'{index:04}_{b_indx:02}_G_mask.exr',
                    }
                    )
                    json_filename = os.path.join(
                        min_preview_folder,
                        f'{index:04}_{b_indx:02}.json'
                    )
                    with open(json_filename, 'w', encoding='utf-8') as json_file:
                        json.dump(item_data['description'], json_file, indent=4, ensure_ascii=False)
            del index, item

        data_time_str = str(f'{data_time:.2f}')
        data_time1_str = str(f'{data_time1:.2f}')
        data_time2_str = str(f'{data_time2:.2f}')
        train_time_str = str(f'{train_time:.2f}')

        epoch_time = time.time() - start_timestamp
        days = int(epoch_time // (24 * 3600))
        hours = int((epoch_time % (24 * 3600)) // 3600)
        minutes = int((epoch_time % 3600) // 60)

        clear_lines(2)
        print (f'\r[Epoch {(epoch + 1):04} Step {step} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time_str}+{data_time1_str}+{train_time_str}+{data_time2_str}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}')
        print(f'\r[Epoch] Min: {min_l1:.6f} Avg: {avg_l1:.6f}, Max: {max_l1:.6f} LPIPS: {avg_lpips:.4f}')

        '''
        if len(stats) < 9999:
            print(f'\r[Epoch] Min: {stats.l1_min:.6f} Avg: {stats.l1:.6f}, Max: {stats.l1_max:.6f} LPIPS: {stats.lpips:.4f}')
        else:
            print(f'\r[Last 10K] Min: {stats.l1_min_last10k:.6f} Avg: {stats.l1_min_last10k:.6f}, Max: {stats.l1_max:.6f} LPIPS: {stats.lpips_last10k:.4f} [Epoch] Min: {stats.l1_min:.6f} Avg: {stats.l1:.6f}, Max: {stats.l1_max:.6f} LPIPS: {stats.lpips:.4f}')
        '''

        if ( idx + 1 ) == len(dataset):
            write_model_state_queue.put(deepcopy(current_state_dict))

            '''
            psnr = float(np.array(psnr_list).mean())
            lpips_val = float(np.array(lpips_list).mean())
            '''

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
                    'Min': min_l1,
                    'Avg': avg_l1,
                    'Max': max_l1,
                    'PSNR': avg_pnsr,
                    'LPIPS': avg_lpips
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

            min_l1 = float(sys.float_info.max)
            max_l1 = 0
            avg_l1 = 0
            avg_pnsr = 0
            avg_lpips = 0
            epoch = epoch + 1
            batch_idx = 0
            
            while  ( idx + 1 ) == len(dataset):
                img0, img1, img2, ratio, idx, current_desc = dataset[batch_idx]

            if not args.sequential:
                dataset.reshuffle()
            max_values.reset()
            min_values.reset()

        # Evaluation block
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
            eval_psnr = []
            eval_lpips = []

            original_state_dict = deepcopy(flownet.state_dict())
            if args.all_gpus:
                original_state_dict = convert_from_data_parallel(original_state_dict)

            flownet.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()            
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            evalnet = Flownet().get_model()().to(device)
            evalnet.load_state_dict(original_state_dict)
                
            if args.eval_half:
                evalnet.half()

            evalnet.eval()
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
                    print (f'\r[Epoch {(epoch + 1):04} Step {step:08} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time_str}+{train_time_str}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}')
                    print (f'\rEvaluating {ev_item_index} of {len(descriptions)}: Min: {eval_loss_min:.6f} Avg: {eval_loss_avg:.6f}, Max: {eval_loss_max:.6f} LPIPS: {eval_lpips_mean:.4f} PSNR: {eval_psnr_mean:4f}')

                    try:                        
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

                        '''
                        if args.all_gpus:
                            eval_img0 = torch.cat([eval_img0, eval_img0], dim=0)
                            eval_img1 = torch.cat([eval_img1, eval_img1], dim=0)
                            eval_img2 = torch.cat([eval_img2, eval_img2], dim=0)
                        '''

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

                        eval_flow_list, eval_mask_list, eval_conf_list, eval_merged = evalnet(
                            eval_img0, 
                            eval_img2,
                            eval_ratio, 
                            iterations = args.iterations
                            )
                        
                        if args.eval_half:
                            eval_flow_list[-1] = eval_flow_list[-1].float()
                            eval_mask_list[-1] = eval_mask_list[-1].float()
                        
                        eval_result = warp(eval_img0_orig, eval_flow_list[-1][:, :2, :eh, :ew]) * eval_mask_list[-1][:, :, :eh, :ew] + warp(eval_img2_orig, eval_flow_list[-1][:, 2:4, :eh, :ew]) * (1 - eval_mask_list[-1][:, :, :eh, :ew])

                        eval_loss_l1 = criterion_l1(eval_result, eval_img1)
                        eval_loss.append(float(eval_loss_l1.item()))
                        eval_psnr.append(float(psnr_torch(eval_result, eval_img1)))
                        eval_loss_LPIPS = loss_fn_alex(eval_result * 2 - 1, eval_img1 * 2 - 1)
                        eval_lpips.append(float(torch.mean(eval_loss_LPIPS).item()))

                        eval_rgb_output_mask = eval_mask_list[-1][:, :, :eh, :ew].repeat_interleave(3, dim=1)
                        eval_rgb_conf = eval_conf_list[-1][:, :, :eh, :ew].repeat_interleave(3, dim=1)
                        eval_rgb_diff = diffmatte(eval_result, eval_img1)[:, :, :eh, :ew].repeat_interleave(3, dim=1)

                        if args.eval_save_imgs:
                            write_eval_image_queue.put(
                                {
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
                                }
                            )

                    except Exception as e:
                        del description['eval_img0']
                        del description['eval_img1']
                        del description['eval_img2']
                        print (f'\nerror while evaluating: {e}\n{description}\n\n')
                    description = read_eval_image_queue.get()

            del evalnet
            if torch.cuda.is_available():
                torch.cuda.empty_cache()            
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            flownet.to(device)
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

        # snapshot = tracemalloc.take_snapshot()
        
        '''
        clear_lines(12)
        print (f'\r[Epoch {(epoch + 1):04} Step {step} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time_str}+{data_time1_str}+{train_time_str}+{data_time2_str}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}')
        print(f'\r[Epoch] Min: {min_l1:.6f} Avg: {avg_l1:.6f}, Max: {max_l1:.6f} LPIPS: {avg_lpips:.4f}')
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            print(stat)
        '''

        # del img0, img1, img2, img0_orig, img1_orig, img2_orig, flow_list, mask_list, conf_list, merged, flow0, flow1, output, output_clean, diff_matte, loss_LPIPS
        data_time2 = time.time() - time_stamp

        if epoch == args.epochs:
            sys.exit()

if __name__ == "__main__":
    main()

