try:
    import os
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget
    from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
    from PyQt5.QtGui import QTextCursor, QFont, QFontDatabase, QFontInfo

    import time
    from tqdm import tqdm
    import queue
    import threading
    import platform

except Exception as e:
    python_executable_path = sys.executable
    if '.miniconda' in python_executable_path:
        print (f'Exception: {e}')
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

class DynamicAttributes:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self):
        # Optional: Provide a string representation for the object
        return f"DynamicAttributes({self.data})"

def main():
    # Custom stream object to capture output
    class Stream(QObject):
        newText = pyqtSignal(str)

        def write(self, text):
            self.newText.emit(str(text))

        def flush(self):
            pass

    # A thread that does some work and produces output
    class Worker(QThread):

        result = pyqtSignal(bool, str)

        def __init__(self, argv, parent=None, window = None):
            super(Worker, self).__init__(parent)
            self.argv = argv
            self.lockfile = self.argv[1]
            self.window = window
            self.running = True

        def run(self):
            import os
            import sys
            import random
            import shutil
            import queue
            import threading
            import time
            import platform
            from pprint import pprint

            def clear_lines(n=2):
                """Clears a specified number of lines in the terminal."""
                CURSOR_UP_ONE = '\x1b[1A'
                ERASE_LINE = '\x1b[2K'
                for _ in range(n):
                    sys.stdout.write(CURSOR_UP_ONE)
                    sys.stdout.write(ERASE_LINE)

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

            def closest_divisible(x):
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
                            batch_size = 4, 
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

                        self.initial_train_descriptions = list(self.train_descriptions)

                        print ('\nReshuffling training data indices...')

                        self.reshuffle()

                        self.h = frame_size
                        self.w = frame_size
                        # self.frame_multiplier = (self.src_w // self.w) * (self.src_h // self.h) * 4

                        self.frames_queue = queue.Queue(maxsize=32)
                        self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
                        self.frame_read_thread.daemon = True
                        self.frame_read_thread.start()

                        print ('reading first block of training data...')
                        self.last_train_data = [self.frames_queue.get()]
                        self.last_train_data_size = 11
                        self.new_sample_shown = False
                        self.train_data_index = 0

                        self.current_batch_data = []

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
                            if 'preview' in root:
                                continue
                            if 'eval' in root:
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
                        images_idx = self.train_data_index

                        device = self.device

                        src_img0 = torch.from_numpy(np_img0.copy())
                        src_img1 = torch.from_numpy(np_img1.copy())
                        src_img2 = torch.from_numpy(np_img2.copy())

                        del train_data, np_img0, np_img1, np_img2

                        src_img0 = src_img0.to(device = device, dtype = torch.float32)
                        src_img1 = src_img1.to(device = device, dtype = torch.float32)
                        src_img2 = src_img2.to(device = device, dtype = torch.float32)

                        '''
                        train_sample_data = {}

                        train_sample_data['rsz1_img0'] = self.resize_image(src_img0, self.h)
                        train_sample_data['rsz1_img1'] = self.resize_image(src_img1, self.h)
                        train_sample_data['rsz1_img2'] = self.resize_image(src_img2, self.h)

                        train_sample_data['rsz2_img0'] = self.resize_image(src_img0, int(self.h * (1 + 1/6)))
                        train_sample_data['rsz2_img1'] = self.resize_image(src_img1, int(self.h * (1 + 1/6)))
                        train_sample_data['rsz2_img2'] = self.resize_image(src_img2, int(self.h * (1 + 1/6)))

                        train_sample_data['rsz3_img0'] = self.resize_image(src_img0, int(self.h * (1 + 1/5)))
                        train_sample_data['rsz3_img1'] = self.resize_image(src_img1, int(self.h * (1 + 1/5)))
                        train_sample_data['rsz3_img2'] = self.resize_image(src_img2, int(self.h * (1 + 1/5)))

                        train_sample_data['rsz4_img0'] = self.resize_image(src_img0, int(self.h * (1 + 1/4)))
                        train_sample_data['rsz4_img1'] = self.resize_image(src_img1, int(self.h * (1 + 1/4)))
                        train_sample_data['rsz4_img2'] = self.resize_image(src_img2, int(self.h * (1 + 1/4)))

                        if len(self.current_batch_data) < self.batch_size:
                            self.current_batch_data = [train_sample_data] * self.batch_size
                        else:
                            old_data = self.current_batch_data.pop(0)
                            del old_data
                            self.current_batch_data.append(train_sample_data)
                        '''

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
                            '''
                            rsz1_img0 = self.current_batch_data[index]['rsz1_img0']
                            rsz1_img1 = self.current_batch_data[index]['rsz1_img1']
                            rsz1_img2 = self.current_batch_data[index]['rsz1_img2']

                            rsz2_img0 = self.current_batch_data[index]['rsz2_img0']
                            rsz2_img1 = self.current_batch_data[index]['rsz2_img1']
                            rsz2_img2 = self.current_batch_data[index]['rsz2_img2']

                            rsz3_img0 = self.current_batch_data[index]['rsz3_img0']
                            rsz3_img1 = self.current_batch_data[index]['rsz3_img1']
                            rsz3_img2 = self.current_batch_data[index]['rsz3_img2']

                            rsz4_img0 = self.current_batch_data[index]['rsz4_img0']
                            rsz4_img1 = self.current_batch_data[index]['rsz4_img1']
                            rsz4_img2 = self.current_batch_data[index]['rsz4_img2']
                            '''

                            if self.generalize == 0:
                                # No augmentaton
                                img0, img1, img2 = self.crop(rsz1_img0, rsz1_img1, rsz1_img2, self.h, self.w)
                                img0 = img0.permute(2, 0, 1)
                                img1 = img1.permute(2, 0, 1)
                                img2 = img2.permute(2, 0, 1)
                            elif self.generalize == 1:
                                # Augment only scale and horizontal flip
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

            def create_csv_file(file_name, fieldnames):
                import csv
                """
                Creates a CSV file with the specified field names as headers.
                """
                with open(file_name, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

            # ----------------------

            if len(self.argv) < 2:
                message = f'Missing input arguments:\n{self.argv}'
                print (message)
                self.result.emit(False, message)
                return

            try:
                import json
                with open(self.argv[1], 'r') as json_file:
                    json_info = json.load(json_file)
            except Exception as e:
                message = f'Unable to load input data from {self.argv[1]}: {e}'
                print (message)
                self.result.emit(False, message)
                return
            
            args = DynamicAttributes(json_info)
            self.window.setWindowTitle(f'TimewarpML Finetune {args.state_file}')

            print (f'Initializing PyTorch...')

            import numpy as np
            import torch

            clear_lines(1)

            print (f'Loading model...')

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
                        print ('prase specify model or model state file')
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

            frame_size = closest_divisible(abs(int(args.frame_size)))
            if frame_size != args.frame_size:
                print (f'Frame size should be divisible by 64 for training. Using {frame_size}')

            read_image_queue = queue.Queue(maxsize=16)

            dataset = get_dataset(
                args.dataset_path, 
                batch_size=args.batch_size, 
                device=device, 
                frame_size=frame_size,
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

            read_thread = threading.Thread(target=read_images, args=(read_image_queue, dataset))
            read_thread.daemon = True
            read_thread.start()

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
                steps_loss = []
                epoch_loss = []
                psnr_list = []
                lpips_list = []

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

            print('\n\n')

            current_state_dict = {
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
                'trained_model_path': trained_model_path
            }

            self.current_state_dict = current_state_dict

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

            while True:
                time.sleep(0.1)



            '''
            print ('Initializing PyTorch...')
            import torch
            if torch.backends.mps.is_available():
                mps_device = torch.device("mps")
                x = torch.randn(4, device=mps_device)
                print (x)
            else:
                print ("MPS device not found.")
            '''

            '''
            tw = Timewarp(json_info)
            result = tw.process()
            if os.path.isfile(self.lockfile):
                os.remove(self.lockfile)
            self.result.emit(result, '')
            '''

            '''
            for i in tqdm(range(100),
                        file=sys.stdout,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', 
                        ascii=f' {chr(0x2588)}',
                        # ascii=False,
                        ncols=50):
                time.sleep(0.1)  # Simulate work
            self.result.emit(True, '')
            '''

        def graceful_exit(self):
            self.running = False
            import torch
            print(f'\nSaving current state to {self.current_state_dict["trained_model_path"]}...')
            torch.save(self.current_state_dict, self.current_state_dict['trained_model_path'])

    # Main window class
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.initUI()
            
            # Redirect sys.stdout and sys.stderr
            sys.stdout = Stream(newText=self.onUpdateText)
            sys.stderr = Stream(newText=self.onUpdateText)

            self.worker_status = False

            self.worker_thread = QThread()
            self.worker = Worker(sys.argv, window=self)
            self.worker.moveToThread(self.worker_thread)

            self.worker_thread.started.connect(self.worker.run)
            self.worker.result.connect(self.handleWorkerResult)
            self.worker.finished.connect(self.onWorkerFinished)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            self.worker.start()

            self.last_progress_line = None  # Keep track of the last progress line

        def loadMonospaceFont(self):
            DejaVuSansMono = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'fonts',
                'DejaVuSansMono.ttf'
            )

            font_id = QFontDatabase.addApplicationFont(DejaVuSansMono)
            if font_id == -1:
                all_fonts = QFontDatabase.families()
                monospaced_fonts = []
                for font_family in all_fonts:
                    font = QFont(font_family)
                    font_info = QFontInfo(font)
                    if font_info.fixedPitch():
                        monospaced_fonts.append(font_family)
                font = QFont(monospaced_fonts[0], 11)  # Generic monospace font
                return font
            else:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                if font_families:
                    font_family = font_families[0]  # Get the first family name
                    font = QFont(font_family, 11)  # Set the desired size
                    return font
                else:
                    return None

        def initUI(self):
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)

            self.text_edit = QTextEdit()
            self.text_edit.setReadOnly(True)
            self.text_edit.setStyleSheet("""
                QTextEdit {
                    color: rgb(188, 188, 188); 
                    background-color: #223438;
                    border: 1px solid #474747;
                }
            """)
            font = self.loadMonospaceFont()
            if font:
                self.text_edit.setFont(font)

            layout.addWidget(self.text_edit)
            
            central_widget = QWidget()
            central_widget.setLayout(layout)
            self.setCentralWidget(central_widget)
            
            self.setGeometry(300, 300, 1100, 600)
            self.setWindowTitle('TimewarpML Finetune')
            self.show()

        def onUpdateText(self, text):
            import re
            if '\r' in text:
                text.replace('\n', '')
                text.replace('\r', '')
                if self.last_progress_line is not None:
                    # Remove the last progress update
                    self.text_edit.moveCursor(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                    self.text_edit.textCursor().removeSelectedText()
                    self.text_edit.moveCursor(QTextCursor.End)
                    self.text_edit.textCursor().deletePreviousChar()  # Remove newline left after text removal
                self.last_progress_line = text
            elif '\x1b[1A' in text:
                self.text_edit.moveCursor(QTextCursor.Up)
                self.text_edit.moveCursor(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                self.text_edit.textCursor().removeSelectedText()
                self.text_edit.moveCursor(QTextCursor.End)
                self.text_edit.textCursor().deletePreviousChar()
                text = ''
            elif '\x1b[2K' in text:
                self.text_edit.moveCursor(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                self.text_edit.textCursor().removeSelectedText()
                self.text_edit.moveCursor(QTextCursor.End)
                self.text_edit.textCursor().deletePreviousChar()
                text = ''
            else:
                pass

            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(text)  # Insert the text at the end
            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()

        def keyPressEvent(self, event):
            sys.stdout.write('keypress:')
            if event.key() == Qt.Key_Control:
                self.ctrl_pressed = True
            super().keyPressEvent(event)
            '''
            elif event.key() == Qt.Key_C and self.ctrl_pressed:
                sys.stdout.write('ctrl+c\n')
                # self.worker.graceful_exit()
                # self.close()
            else:
                super().keyPressEvent(event)
            '''

        def keyReleaseEvent(self, event):
            sys.stdout.write('keyrelease:')
            if event.key() == Qt.Key_Control:
                self.ctrl_pressed = False
            elif event.key() == Qt.Key_C and self.ctrl_pressed:
                sys.stdout.write('ctrl+c\n')
            super().keyReleaseEvent(event)

        def handleWorkerResult(self, status, message):
            self.worker_status = status
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            if self.worker_status:
                self.close()

        def onWorkerFinished(self):
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            if self.worker_status:
                self.close()

    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()