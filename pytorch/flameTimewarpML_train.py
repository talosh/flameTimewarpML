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

from pprint import pprint

try:
    import numpy as np
    import torch
except:
    python_executable_path = sys.executable
    if '.miniconda' in python_executable_path:
        print ('Unable to import Numpy and PyTorch libraries')
        print (f'Using {python_executable_path} python interpreter')
        sys.exit()

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

def get_dataset(
        data_root, 
        batch_size = 8, 
        device = None, 
        frame_size=448, 
        max_window=5,
        acescc_rate = 40,
        generalize = 80,
        repeat = 1
        ):
    class TimewarpMLDataset(torch.utils.data.Dataset):
        def __init__(   
                self, 
                data_root, 
                batch_size = 8, 
                device = None, 
                frame_size=448, 
                max_window=5,
                acescc_rate = 40,
                generalize = 80,
                repeat = 1
                ):
            
            self.data_root = data_root
            self.batch_size = batch_size
            self.max_window = max_window
            self.acescc_rate = acescc_rate
            self.generalize = generalize

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
                print (f'\rReading headers and building training data from clip {folder_index + 1} of {len(self.folders_with_exr)}', end='')
                self.train_descriptions.extend(self.create_dataset_descriptions(folder_path, max_window=self.max_window))
            print ('\nReshuffling training data indices...')

            self.reshuffle()

            self.h = frame_size
            self.w = frame_size
            # self.frame_multiplier = (self.src_w // self.w) * (self.src_h // self.h) * 4

            self.frames_queue = queue.Queue(maxsize=8)
            self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
            self.frame_read_thread.daemon = True
            self.frame_read_thread.start()

            print ('reading first block of training data...')
            self.last_train_data = self.frames_queue.get()

            self.repeat_count = repeat
            self.repeat_counter = 0

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
            elif 'slow' in folder_path:
                max_window = max_window
            else:
                if max_window > 5:
                    max_window = 5

            try:
                first_exr_file_header = read_openexr_file(exr_files[0], header_only = True)
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
                        # train_data['pre_start'] = read_openexr_file(description['pre_start'])['image_data']
                        train_data['start'] = read_openexr_file(description['start'])['image_data']
                        train_data['gt'] = read_openexr_file(description['gt'])['image_data']
                        train_data['end'] = read_openexr_file(description['end'])['image_data']
                        # train_data['after_end'] = read_openexr_file(description['after_end'])['image_data']
                        train_data['ratio'] = description['ratio']
                        train_data['h'] = description['h']
                        train_data['w'] = description['w']
                        train_data['description'] = description
                        train_data['index'] = index
                        self.frames_queue.put(train_data)
                    except Exception as e:
                        del train_data
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
            # '''
            if not self.last_train_data:
                new_data = self.frames_queue.get_nowait()
                self.last_train_data = new_data
                del new_data
            if self.repeat_counter >= self.repeat_count:
                try:
                    new_data = self.frames_queue.get_nowait()
                    self.last_train_data = new_data
                    del new_data
                    self.repeat_counter = 0
                except queue.Empty:
                    pass

            self.repeat_counter += 1
            return self.last_train_data
            # '''
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

            return ACEScc

        def __getitem__(self, index):
            train_data = self.getimg(index)
            # src_img0 = train_data['pre_start']
            np_img0 = train_data['start']
            np_img1 = train_data['gt']
            np_img2 = train_data['end']
            # src_img4 = train_data['after_end']
            imgh = train_data['h']
            imgw = train_data['w']
            ratio = train_data['ratio']
            images_idx = train_data['index']

            device = self.device

            src_img0 = torch.from_numpy(np_img0.copy())
            src_img1 = torch.from_numpy(np_img1.copy())
            src_img2 = torch.from_numpy(np_img2.copy())

            del train_data, np_img0, np_img1, np_img2

            src_img0 = src_img0.to(device = device, dtype = torch.float32)
            src_img1 = src_img1.to(device = device, dtype = torch.float32)
            src_img2 = src_img2.to(device = device, dtype = torch.float32)

            rsz1_img0 = self.resize_image(src_img0, self.h)
            rsz1_img1 = self.resize_image(src_img1, self.h)
            rsz1_img2 = self.resize_image(src_img2, self.h)

            rsz2_img0 = self.resize_image(src_img0, int(self.h * (1 + 1/6)))
            rsz2_img1 = self.resize_image(src_img1, int(self.h * (1 + 1/6)))
            rsz2_img2 = self.resize_image(src_img2, int(self.h * (1 + 1/6)))

            rsz3_img0 = self.resize_image(src_img0, int(self.h * (1 + 1/5)))
            rsz3_img1 = self.resize_image(src_img1, int(self.h * (1 + 1/5)))
            rsz3_img2 = self.resize_image(src_img2, int(self.h * (1 + 1/5)))

            rsz4_img0 = self.resize_image(src_img0, int(self.h * (1 + 1/4)))
            rsz4_img1 = self.resize_image(src_img1, int(self.h * (1 + 1/4)))
            rsz4_img2 = self.resize_image(src_img2, int(self.h * (1 + 1/4)))

            batch_img0 = []
            batch_img1 = []
            batch_img2 = []

            for index in range(self.batch_size):

                if self.generalize == 0:
                    # No agumentaton
                    img0, img1, img2 = self.crop(rsz1_img0, rsz1_img1, rsz1_img2, self.h, self.w)
                    img0 = img0.permute(2, 0, 1)
                    img1 = img1.permute(2, 0, 1)
                    img2 = img2.permute(2, 0, 1)
                elif self.generalize == 1:
                    # Agument only scale and horizontal flip
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
                    if random.uniform(0, 1) < 0.5:
                        img0 = img0.flip(-1)
                        img1 = img1.flip(-1)
                        img2 = img2.flip(-1)
                else:
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
                        # Exposure agumentation
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
                        multipliers = torch.tensor([r, g, b]).view(3, 1, 1).to(device)
                        img0 = img0 * multipliers
                        img1 = img1 * multipliers
                        img2 = img2 * multipliers

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
                    img0 = self.apply_acescc(img0)
                    img1 = self.apply_acescc(img1)
                    img2 = self.apply_acescc(img2)

                batch_img0.append(img0)
                batch_img1.append(img1)
                batch_img2.append(img2)

            return torch.stack(batch_img0), torch.stack(batch_img1), torch.stack(batch_img2), ratio, images_idx

        def get_input_channels_number(self, source_frames_paths_list):
            total_num_channels = 0
            for src_path in source_frames_paths_list:
                file_header = read_openexr_file(src_path, header_only=True)
                total_num_channels += file_header['shape'][2]
            return total_num_channels

    return TimewarpMLDataset(
        data_root, 
        batch_size=batch_size, 
        device=device, 
        frame_size=frame_size, 
        max_window=max_window,
        acescc_rate=acescc_rate,
        generalize=generalize,
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

def create_timestamp_uid(self):
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

def find_and_import_model(models_dir='models', base_name=None, model_name=None):
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
        regex_pattern = fr"{base_name}_v(\d+)\.py"
        versions = [(f, int(m.group(1))) for f in files if (m := re.match(regex_pattern, f))]
        if versions:
            # Sort by version number (second item in tuple) and select the latest one
            latest_version_file = sorted(versions, key=lambda x: x[1], reverse=True)[0][0]
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

def closest_divisible_by_64(x):
    """
    Find the closest integer divisible by 64 to the given number x.

    Args:
    x (int or float): The number to find the closest divisible value for.

    Returns:
    int: Closest number divisible by 64.
    """
    # Round down to the nearest multiple of 64
    lower = (x // 64) * 64
    
    # Round up to the nearest multiple of 64
    upper = lower + 64

    # Check which one is closer to x
    if x - lower > upper - x:
        return upper
    else:
        return lower

def main():
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
    parser.add_argument('--onecyclelr', action='store_true', default=False, help='Use OneCycleLR strategy. If number of epochs is not set cycle is set to single epoch.')
    parser.add_argument('--onecycle', type=int, default=-1, help='Train one cycle for N epochs (default: None)')
    parser.add_argument('--state_file', type=str, default=None, help='Path to the pre-trained model state dict file (optional)')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--legacy_model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (int) (default: 8)')
    parser.add_argument('--first_epoch', type=int, default=-1, help='Epoch (int) (default: Saved)')
    parser.add_argument('--epochs', type=int, default=-1, help='Number of epoch to run (int) (default: Unlimited)')
    parser.add_argument('--eval', type=int, dest='eval', default=-1, help='Evaluate after each epoch for N samples')
    parser.add_argument('--frame_size', type=int, default=448, help='Frame size in pixels (default: 448)')
    parser.add_argument('--all_gpus', action='store_true', dest='all_gpus', default=False, help='Use nn.DataParallel')
    parser.add_argument('--freeze', action='store_true', dest='freeze', default=False, help='Freeze RIFE parameters')
    parser.add_argument('--acescc', type=check_range_percent, default=40, help='Percentage of ACEScc encoded frames (default: 40))')
    parser.add_argument('--generalize', type=check_range_percent, default=85, help='Generalization level (0 - 100) (default: 85)')
    parser.add_argument('--preview', type=int, default=1000, help='Save preview each N steps (default: 1000)')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat each triade N times with agumentation (default: 1)')
    parser.add_argument('--iterations', type=int, default=1, help='Process each flow refinement N times (default: 1)')

    args = parser.parse_args()

    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')
    if args.all_gpus:
        device = 'cuda'
    
    if args.state_file:
        trained_model_path = args.state_file
        try:
            checkpoint = torch.load(trained_model_path, map_location=device)
            print('loaded previously saved model checkpoint')
        except Exception as e:
            print (f'unable to load saved model checkpoint: {e}')
            sys.exit()

        model_info = checkpoint.get('model_info')
        model_name = model_info.get('name')
    else:
        model_name = args.model

    # Find and initialize model
    Flownet = find_and_import_model(base_name='flownet', model_name=model_name)
    if Flownet is None:
        print (f'Unable to load model {args.model}')
        return
    model_info = Flownet.get_info()
    print ('Model info:')
    pprint (model_info)
    max_dataset_window = 9
    if not model_info.get('ratio_support'):
        max_dataset_window = 3
    flownet = Flownet().get_training_model()().to(device)
    
    if args.all_gpus:
        print ('Using nn.DataParallel')
        flownet = torch.nn.DataParallel(flownet)
        flownet.to(device)

    if not os.path.isdir(os.path.join(args.dataset_path, 'preview')):
        os.makedirs(os.path.join(args.dataset_path, 'preview'))

    frame_size = closest_divisible_by_64(abs(int(args.frame_size)))
    if frame_size != args.frame_size:
        print (f'Frame size should be divisible by 64 for training. Using {frame_size}')

    read_image_queue = queue.Queue(maxsize=8)
    dataset = get_dataset(
        args.dataset_path, 
        batch_size=args.batch_size, 
        device=device, 
        frame_size=abs(int(args.frame_size)),
        max_window=max_dataset_window,
        acescc_rate=args.acescc,
        generalize=args.generalize,
        repeat=args.repeat
        )

    def read_images(read_image_queue, dataset):
        while True:
            for batch_idx in range(len(dataset)):
                img0, img1, img2, ratio, idx = dataset[batch_idx]
                read_image_queue.put((img0, img1, img2, ratio, idx))
                del img0, img1, img2, ratio, idx

    def write_images(write_image_queue):
        while True:
            try:
                write_data = write_image_queue.get_nowait()
                write_exr(write_data['sample_source1'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_incomng.exr'))
                write_exr(write_data['sample_source2'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_outgoing.exr'))
                write_exr(write_data['sample_target'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_target.exr'))
                write_exr(write_data['sample_output'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_output.exr'))
                write_exr(write_data['sample_output_mask'], os.path.join(write_data['preview_folder'], f'{preview_index:02}_output_mask.exr'))
                del write_data
            except:
            # except queue.Empty:
                time.sleep(0.1)

    read_thread = threading.Thread(target=read_images, args=(read_image_queue, dataset))
    read_thread.daemon = True
    read_thread.start()

    write_image_queue = queue.Queue(maxsize=8)
    write_thread = threading.Thread(target=write_images, args=(write_image_queue, ))
    write_thread.daemon = True
    write_thread.start()
    
    pulse_dive = args.pulse_amplitude
    pulse_period = args.pulse
    lr = args.lr

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    criterion_huber = torch.nn.HuberLoss(delta=0.01)

    weight_decay = 10 ** (0.07 * args.generalize - 9) if args.generalize > 1 else 1e-9

    if args.generalize == 0:
        print (f'Disabling agumentation and setting weight decay to {weight_decay:.2e}')
    elif args.generalize == 1:
        print (f'Setting agumentation to horizontal flip and scale only and weight decay to {weight_decay:.2e}')
    else:
        print (f'Setting agumentation rate to {args.generalize}% and weight decay to {weight_decay:.2e}')

    optimizer_flownet = torch.optim.AdamW(flownet.parameters(), lr=lr, weight_decay=9e-4)
    # optimizer_dt = torch.optim.Adam(model_D.parameters(), lr=lr)

    train_scheduler_flownet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_flownet, T_max=pulse_period, eta_min = lr - (( lr / 100 ) * pulse_dive) )

    if args.onecyclelr:
        if args.epochs == -1:
            train_scheduler_flownet = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_flownet, 
                max_lr=args.lr, 
                total_steps=len(dataset)*dataset.repeat_count
                )
            print (f'setting repeating OneCycleLR with max_lr={args.lr}, total_steps={len(dataset)*dataset.repeat_count}')
        else:
            train_scheduler_flownet = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_flownet, 
                max_lr=args.lr, 
                steps_per_epoch=len(dataset)*dataset.repeat_count, 
                epochs=args.epochs
                )
            print (f'setting repeating OneCycleLR with max_lr={args.lr}, steps_per_epoch={len(dataset)*dataset.repeat_count}, epochs={args.epochs}')

    if args.onecycle != -1:
        train_scheduler_flownet = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_flownet,
            max_lr=args.lr, 
            steps_per_epoch=len(dataset)*dataset.repeat_count, 
            epochs=args.onecycle
            )
        print (f'setting OneCycleLR with max_lr={args.lr}, steps_per_epoch={len(dataset)*dataset.repeat_count}, epochs={args.onecycle}')

    # train_scheduler_flownet = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_flownet, mode='min', factor=0.1, patience=2)
    # lambda_function = lambda epoch: 1
    # train_scheduler_flownet = torch.optim.lr_scheduler.LambdaLR(optimizer_flownet, lr_lambda=lambda_function)

    scheduler_flownet = train_scheduler_flownet

    step = 0
    loaded_step = 0
    current_epoch = 0
    preview_index = 0

    steps_loss = []
    epoch_loss = []
    psnr_list = []
    lpips_list = []

    if args.state_file:
        trained_model_path = args.state_file

        try:
            checkpoint = torch.load(trained_model_path, map_location=device)
            print('loaded previously saved model checkpoint')
        except Exception as e:
            print (f'unable to load saved model: {e}')

        try:
            missing_keys, unexpected_keys = flownet.load_state_dict(checkpoint['flownet_state_dict'], strict=False)
            print('loaded previously saved Flownet state')
            if missing_keys:
                print (f'\nMissing keys:\n{missing_keys}\n')
            if unexpected_keys:
                print (f'\nUnexpected keys:\n{unexpected_keys}\n')
        except Exception as e:
            print (f'unable to load Flownet state: {e}')

        '''
        try:
            model_D.load_state_dict(checkpoint['model_d_state_dict'], strict=False)
            print('loaded previously saved Determinator state')
        except Exception as e:
            print (f'unable to load Determinator state: {e}')
        '''

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
        '''
        try:
            Flownet().load_model(args.legacy_model, flownet)
            print (f'loaded legacy model state: {args.legacy_model}')
        except Exception as e:
            print (f'unable to load legacy model: {e}')
        '''

        '''
        default_rife_model_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'models_data',
            'flownet_v412.pkl'
        )
        '''
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

        # '''
        # copy encoder weights for flownet4_v004 to separate encoders on each pass
        if model_info.get('name') == 'Flownet4_v004':
            flownet.block0.encode01[0].load_state_dict(flownet.encode.cnn0.state_dict())
            flownet.block0.encode01[2].load_state_dict(flownet.encode.cnn1.state_dict())
            flownet.block0.encode01[4].load_state_dict(flownet.encode.cnn2.state_dict())
            flownet.block0.encode01[6].load_state_dict(flownet.encode.cnn3.state_dict())

            flownet.block1.encode01[0].load_state_dict(flownet.encode.cnn0.state_dict())
            flownet.block1.encode01[2].load_state_dict(flownet.encode.cnn1.state_dict())
            flownet.block1.encode01[4].load_state_dict(flownet.encode.cnn2.state_dict())
            flownet.block1.encode01[6].load_state_dict(flownet.encode.cnn3.state_dict())

            flownet.block2.encode01[0].load_state_dict(flownet.encode.cnn0.state_dict())
            flownet.block2.encode01[2].load_state_dict(flownet.encode.cnn1.state_dict())
            flownet.block2.encode01[4].load_state_dict(flownet.encode.cnn2.state_dict())
            flownet.block2.encode01[6].load_state_dict(flownet.encode.cnn3.state_dict())

            flownet.block3.encode01[0].load_state_dict(flownet.encode.cnn0.state_dict())
            flownet.block3.encode01[2].load_state_dict(flownet.encode.cnn1.state_dict())
            flownet.block3.encode01[4].load_state_dict(flownet.encode.cnn2.state_dict())
            flownet.block3.encode01[6].load_state_dict(flownet.encode.cnn3.state_dict())
        # '''

    # LPIPS Init

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

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

        for param in flownet.encode.parameters():
            param.requires_grad = False

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

        for param in flownet.block0.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block1.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block2.conv0.parameters():
            param.requires_grad = False
        for param in flownet.block3.conv0.parameters():
            param.requires_grad = False

        for param in flownet.block0.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block1.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block2.convblock.parameters():
            param.requires_grad = False
        for param in flownet.block3.convblock.parameters():
            param.requires_grad = False

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

        for param in flownet.encode.parameters():
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

    while True:
        data_time = time.time() - time_stamp
        time_stamp = time.time()

        img0, img1, img2, ratio, idx = read_image_queue.get()

        img0 = img0.to(device, non_blocking = True)
        img1 = img1.to(device, non_blocking = True)
        img2 = img2.to(device, non_blocking = True)
        img0_orig = img0.detach().clone()
        img1_orig = img1.detach().clone()
        img2_orig = img2.detach().clone()
        img0 = normalize(img0)
        img1 = normalize(img1)
        img2 = normalize(img2)

        current_lr_str = str(f'{optimizer_flownet.param_groups[0]["lr"]:.4e}')

        optimizer_flownet.zero_grad()

        # scale list agumentation
        random_scales = [
            [4, 2, 1, 1],
            [2, 2, 1, 1],
            [2, 1, 1, 1],
            [1, 1, 1, 1],
        ]

        if random.uniform(0, 1) < 0.44:
            training_scale = random_scales[random.randint(0, len(random_scales) - 1)]
        else:
            training_scale = [8, 4, 2, 1]

        flownet.train()
        
        n_iterations = random.randint(1, 4) if args.iterations == -1 else args.iterations

        flow_list, mask_list, merged = flownet(
            img0, 
            img2, 
            ratio, 
            scale=training_scale,
            iterations = n_iterations
            )
        # flow0 = flow_list[3][:, :2]
        # flow1 = flow_list[3][:, 2:4]
        mask = mask_list[3]
        # output = warp(img0_orig, flow0) * mask + warp(img2_orig, flow1) * (1 - mask)
        
        output = merged[3]
        # warped_img0 = warp(img0, flow_list[3][:, :2])
        # warped_img2 = warp(img2, flow_list[3][:, 2:4])
        # output = warped_img0 * mask_list[3] + warped_img2 * (1 - mask_list[3])

        loss_x8 = criterion_huber(
            torch.nn.functional.interpolate(restore_normalized_values(merged[0]), scale_factor= 1. / training_scale[0], mode="bilinear", align_corners=False),
            torch.nn.functional.interpolate(img1_orig, scale_factor= 1. / training_scale[0], mode="bilinear", align_corners=False)
        )

        loss_x4 = criterion_huber(
            torch.nn.functional.interpolate(restore_normalized_values(merged[1]), scale_factor= 1. / training_scale[1], mode="bilinear", align_corners=False),
            torch.nn.functional.interpolate(img1_orig, scale_factor= 1. / training_scale[1], mode="bilinear", align_corners=False)
        )
        
        loss_x2 = criterion_huber(
            torch.nn.functional.interpolate(restore_normalized_values(merged[2]), scale_factor= 1. / training_scale[2], mode="bilinear", align_corners=False),
            torch.nn.functional.interpolate(img1_orig, scale_factor= 1. / training_scale[2], mode="bilinear", align_corners=False)
        )

        loss_x1 = criterion_huber(restore_normalized_values(output), img1_orig)
        # loss_LPIPS_ = loss_fn_alex(restore_normalized_values(output) * 2 - 1, img1_orig * 2 - 1)

        # loss_x1 = criterion_huber(output, img1)
        loss_LPIPS_ = loss_fn_alex(output * 2 - 1, img1 * 2 - 1)
        loss_LPIPS = torch.mean(loss_LPIPS_)

        loss_deep = 0.2 * loss_x8 + 0.2 * loss_x4 + 0.2 * loss_x2 + 0.4 * loss_x1
        loss = loss_deep + 1e-6 * loss_LPIPS # + loss_FM + loss_Adv

        loss_l1 = criterion_l1(restore_normalized_values(output), img1_orig)
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

        torch.nn.utils.clip_grad_norm_(flownet.parameters(), 0.9)

        optimizer_flownet.step()
        scheduler_flownet.step()

        train_time = time.time() - time_stamp
        time_stamp = time.time()

        if step % args.preview == 1:
            rgb_source1 = img0_orig
            rgb_source2 = img2_orig
            rgb_target = img1_orig
            rgb_output = restore_normalized_values(output)
            rgb_output_mask = mask.repeat_interleave(3, dim=1)
            # rgb_refine = refine_list[0] + refine_list[1] + refine_list[2] + refine_list[3]
            # rgb_refine = (rgb_refine + 1) / 2
            # sample_refine = rgb_refine[0].clone().cpu().detach().numpy().transpose(1, 2, 0)


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

            del rgb_source1, rgb_source2, rgb_target, rgb_output, rgb_output_mask
            preview_index = preview_index + 1 if preview_index < 9 else 0

        if step % 1000 == 1:
            if os.path.isfile(trained_model_path):
                backup_file = trained_model_path.replace('.pth', '.backup.pth')
                shutil.copy(trained_model_path, backup_file)
            torch.save({
                'step': step,
                'steps_loss': steps_loss,
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                'start_timestamp': start_timestamp,
                'lr': optimizer_flownet.param_groups[0]['lr'],
                'model_info': model_info,
                'flownet_state_dict': flownet.state_dict(),
                # 'model_d_state_dict': model_D.state_dict(),
                'optimizer_flownet_state_dict': optimizer_flownet.state_dict(),
            }, trained_model_path)

        data_time += time.time() - time_stamp
        data_time_str = str(f'{data_time:.2f}')
        train_time_str = str(f'{train_time:.2f}')

        epoch_time = time.time() - start_timestamp
        days = int(epoch_time // (24 * 3600))
        hours = int((epoch_time % (24 * 3600)) // 3600)
        minutes = int((epoch_time % 3600) // 60)

        clear_lines(2)
        print (f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Time:{data_time_str} + {train_time_str}, Batch [Step: {batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}, Loss L1: {loss_l1_str}')
        print(f'\r[Last 10K steps] Min: {window_min:.6f} Avg: {smoothed_window_loss:.6f}, Max: {window_max:.6f} LPIPS: {lpips_window_val:.4f} [Epoch] Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f} LPIPS: {lpips_val:.4f}')

        if ( idx + 1 ) == len(dataset):
            if os.path.isfile(trained_model_path):
                backup_file = trained_model_path.replace('.pth', '.backup.pth')
                shutil.copy(trained_model_path, backup_file)
            torch.save({
                'step': step,
                'steps_loss': steps_loss,
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                'start_timestamp': start_timestamp,
                'lr': optimizer_flownet.param_groups[0]['lr'],
                'model_info': model_info,
                'flownet_state_dict': flownet.state_dict(),
                # 'model_d_state_dict': model_D.state_dict(),
                'optimizer_flownet_state_dict': optimizer_flownet.state_dict(),
            }, trained_model_path)

            psnr = 0

            if args.eval != -1:
                psnr_list = []
                lpips_list = []

                try:
                    for ev_item_index in range(args.eval):

                        clear_lines(2)
                        print (f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Time:{data_time_str} + {train_time_str}, Batch [Step: {batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}, Loss L1: {loss_l1_str}')
                        print (f'\rCalcualting PSNR on full-scale image {ev_item_index} of {args.eval}...')

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
                        evn_img0_orig = ev_img0
                        evn_img1_orig = ev_img1
                        evn_img2_orig = ev_img2
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
                            flownet.eval()
                            _, _, merged = flownet(evp_img0, evp_img1, evp_img2, None, None, ev_ratio)
                            evp_output = merged[3]
                            psnr_list.append(psnr_torch(restore_normalized_values(evp_output)[:h, :w], evn_img1_orig))
                            lpips_list.append(float(loss_fn_alex(restore_normalized_values(evp_output)[:h, :w] * 2 - 1, evn_img1_orig * 2 - 1).item()))

                        preview_folder = os.path.join(args.dataset_path, 'preview')
                        eval_folder = os.path.join(preview_folder, 'eval')
                        if not os.path.isdir(eval_folder):
                            try:
                                os.makedirs(eval_folder)
                            except Exception as e:
                                print (e)

                        evp_output = restore_normalized_values(evp_output)
                        ev_output = evp_output[0].permute(1, 2, 0)[:h, :w]

                        try:
                            write_exr(evn_img0_orig[0].permute(1, 2, 0).clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_incomng.exr'))
                            write_exr(evn_img2_orig[0].permute(1, 2, 0).clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_outgoing.exr'))
                            write_exr(evn_img1_orig[0].permute(1, 2, 0).clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_target.exr'))
                            # write_exr(ev_output_inflow.clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_output.exr'))
                            write_exr(ev_output.clone().cpu().detach().numpy(), os.path.join(eval_folder, f'{ev_item_index:04}_output.exr'))
                        except Exception as e:
                            print (f'{e}\n\n')      

                except Exception as e:
                    print (f'{e}\n\n')
   
            psnr = np.array(psnr_list).mean()
            lpips_val = np.array(lpips_list).mean()

            epoch_time = time.time() - start_timestamp
            days = int(epoch_time // (24 * 3600))
            hours = int((epoch_time % (24 * 3600)) // 3600)
            minutes = int((epoch_time % 3600) // 60)

            clear_lines(2)
            print(f'\rEpoch [{epoch + 1} - {days:02}d {hours:02}:{minutes:02}], Min: {min(epoch_loss):.6f} Avg: {smoothed_loss:.6f}, Max: {max(epoch_loss):.6f}, [PNSR] {psnr:.4f}, [LPIPS] {lpips_val:.4f}')
            print ('\n')

            steps_loss = []
            epoch_loss = []
            psnr_list = []
            lpips_list = []
            epoch = epoch + 1
            batch_idx = 0

            while  ( idx + 1 ) == len(dataset):
                img0, img1, img2, ratio, idx = read_image_queue.get()
            dataset.reshuffle()

        batch_idx = batch_idx + 1
        step = step + 1

        del img0, img1, img2, img0_orig, img1_orig, img2_orig, flow_list, mask_list, merged, mask, output
        # import gc
        # gc.collect()
        # continue

        if epoch == args.epochs:
            sys.exit()



if __name__ == "__main__":
    main()

