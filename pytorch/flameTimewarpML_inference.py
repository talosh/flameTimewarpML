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

class Timewarp():
    def __init__(self, json_info):
        self.json_info = json_info
        self.source_folder = self.json_info.get('input')
        self.target_folder = self.json_info.get('output')
        self.clip_name = self.json_info.get('clip_name')
        self.settings = self.json_info.get('settings')
        print('Initializing TimewarpML from Flame setup...')
        import torch
        self.device = torch.device("mps") if platform.system() == 'Darwin' else torch.device('cuda')
        self.model_path = self.json_info.get('model_path')
        self.model = self.find_and_import_model(self.model_path)
        if not self.model:
            print (f'Unable to import model from file {self.model_path}')

    def find_and_import_model(self, model_file_path):
        import importlib
        import torch

        checkpoint = torch.load(model_file_path, map_location=self.device)
        model_info = checkpoint.get('model_info')
        model_file = model_info.get('file')
        module_name = model_file[:-3]  # Remove '.py' from filename to get module name
        module_path = f"models.{module_name}"

        try:
            module = importlib.import_module(module_path)
            model_object = getattr(module, 'Model')
            model = model_object().get_model()().to(self.device)
            model.load_state_dict(checkpoint['flownet_state_dict'])
            model.eval()
            return model
        except Exception as e:
            print ({e})
            return None

    def process(self):
        if not self.model:
            print (f'Unable to import model from file {self.model_path}')
            return False

        tw_setup_string = self.json_info.get('setup')
        '''
        for k in self.json_info.keys():
            print (k)
        print (f'{self.json_info}')
        '''

        src_files_list = [file for file in os.listdir(self.source_folder) if file.endswith('.exr')]
        input_duration = len(src_files_list)
        if not input_duration:
            print(f'no input frames found in: "{self.source_folder}"')
            return False
        self.record_in = self.json_info.get('record_in', 1)
        self.record_out = self.json_info.get('record_out', input_duration)
        frame_value_map = self.bake_flame_tw_setup(tw_setup_string)

        start_frame = 1
        src_files_list.sort()
        src_files = {x:os.path.join(self.source_folder, file_path) for x, file_path in enumerate(src_files_list, start=start_frame)}

        frame_info_dict = {}
        output_frame_number = 1

        # print (f'{frame_value_map}')
        for frame_number in range(self.record_in, self.record_out + 1):
            frame_info = {}
            incoming_frame_number = int(frame_value_map[frame_number])

            if incoming_frame_number < 1:
                frame_info['incoming'] = src_files.get(1)
                frame_info['outgoing'] = None
                frame_info['ratio'] = 0
                frame_info['output'] = os.path.join(self.target_folder, f'{self.clip_name}.{output_frame_number:08}.exr')
                frame_info_dict[output_frame_number] = frame_info
                output_frame_number += 1
                continue

            if incoming_frame_number >= input_duration:
                frame_info['incoming'] = src_files.get(input_duration)
                frame_info['outgoing'] = None
                frame_info['ratio'] = 0
                frame_info['output'] = os.path.join(self.target_folder, f'{self.clip_name}.{output_frame_number:08}.exr')
                frame_info_dict[output_frame_number] = frame_info
                output_frame_number += 1
                continue

            frame_info['incoming'] = src_files.get(incoming_frame_number)
            frame_info['outgoing'] = src_files.get(incoming_frame_number + 1)
            frame_info['ratio'] = frame_value_map[frame_number] - int(frame_value_map[frame_number])
            frame_info['output'] = os.path.join(self.target_folder, f'{self.clip_name}.{output_frame_number:08}.exr')
            frame_info_dict[output_frame_number] = frame_info

            # print (f'fr: {output_frame_number}, inc:{incoming_frame_number}, out: {incoming_frame_number + 1}, r: {frame_info["ratio"]}')

            output_frame_number += 1

        def read_images(read_image_queue, frame_info_dict):
            for out_frame_number in sorted(frame_info_dict.keys()):
                frame_info = frame_info_dict[out_frame_number]
                frame_info['incoming_image_data'] = read_openexr_file(frame_info['incoming'])
                frame_info['outgoing_image_data'] = read_openexr_file(frame_info['outgoing'])
                read_image_queue.put(frame_info)

        read_image_queue = queue.Queue(maxsize=9)
        read_thread = threading.Thread(target=read_images, args=(read_image_queue, frame_info_dict))
        read_thread.daemon = True
        read_thread.start()

        print(f'rendering {len(frame_info_dict.keys())} frames to {self.target_folder}')

        '''
        self.pbar = tqdm(total=len(frame_info_dict.keys()), 
                         unit='frame',
                         file=sys.stdout,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                         ascii=f' {chr(0x2588)}',
                         ncols=80
                         )
        '''

        def write_images(write_image_queue):
            while True:
                image_path = ''
                try:
                    write_data = write_image_queue.get_nowait()
                    image_data = write_data['image_data']
                    image_path = write_data['image_path']
                    if image_data is None:
                        print ('finishing write thread')
                        break
                    write_exr(image_data, image_path)
                    # self.pbar.update(1)
                except queue.Empty:
                    time.sleep(1e-4)
                except Exception as e:
                    print (f'error writing file: {image_path}: {e}')

        write_image_queue = queue.Queue(maxsize=9)
        write_thread = threading.Thread(target=write_images, args=(write_image_queue, ))
        write_thread.daemon = True
        write_thread.start()

        for idx in range(len(frame_info_dict.keys())):
            frame_info = read_image_queue.get()
            # print (f'frame {idx + 1} of {len(frame_info_list)}')
            img0 = frame_info['incoming_image_data']['image_data']
            img1 = frame_info['outgoing_image_data']['image_data']
            ratio = frame_info['ratio']
            try:
                result = self.predict(img0, img1, ratio = ratio, iterations = 1)
            except Exception as e:
                print (f'{e}')
            image_path = frame_info['output']
            write_image_queue.put({'image_data': result, 'image_path': image_path})

        write_image_queue.put({'image_data': None, 'image_path': None})
        write_thread.join()
        # self.pbar.close()
        return True

    def predict(self, incoming_data, outgoing_data, ratio = 0.5, iterations = 1):
        import numpy as np
        import torch

        print (ratio)

        device = self.device

        def normalize(image_array) :
            def custom_bend(x):
                linear_part = x
                exp_bend = torch.sign(x) * torch.pow(torch.abs(x), 1 / 4 )
                return torch.where(x > 1, exp_bend, torch.where(x < -1, exp_bend, linear_part))
            image_array = (image_array * 2) - 1
            input_device = image_array.device
            if 'mps' in str(input_device):
                image_array = custom_bend(image_array.detach().to(device=torch.device('cpu'))).to(device=input_device)
            else:
                image_array = custom_bend(image_array)
            image_array = torch.tanh(image_array)
            image_array = (image_array + 1) / 2
            return image_array

        def warp(tenInput, tenFlow):
            input_device = tenInput.device
            input_dtype = tenInput.dtype
            if 'mps' in str(input_device):
                tenInput = tenInput.detach().to(device=torch.device('cpu'), dtype=torch.float32)
                tenFlow = tenFlow.detach().to(device=torch.device('cpu'), dtype=torch.float32)

            backwarp_tenGrid = {}
            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device, dtype=tenInput.dtype)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            result = torch.nn.functional.grid_sample(
                input=tenInput, 
                grid=g, 
                mode='bilinear', 
                padding_mode='border', 
                align_corners=True
                )

            return result.detach().to(device=input_device, dtype=input_dtype)

        with torch.no_grad():
            if ratio == 0:
                return incoming_data
            elif ratio == 1:
                return outgoing_data
            else:            
                img0 = torch.from_numpy(incoming_data.copy())
                if 'mps' not in str(device):
                    img0 = img0.to(device = device, dtype = torch.float16, non_blocking = True)
                else:
                    img0 = img0.to(device = device, dtype = torch.float32, non_blocking = True)
                img0 = img0.permute(2, 0, 1).unsqueeze(0)

                img1 = torch.from_numpy(outgoing_data.copy())
                if 'mps' not in str(device):
                    img1 = img1.to(device = device, dtype = torch.float16, non_blocking = True)
                else:
                    img1 = img1.to(device = device, dtype = torch.float32, non_blocking = True)
                img1 = img1.permute(2, 0, 1).unsqueeze(0)

                img0_ref = normalize(img0)
                img1_ref = normalize(img1)

                n, c, h, w = img0.shape
                ph = ((h - 1) // 64 + 1) * 64
                pw = ((w - 1) // 64 + 1) * 64
                padding = (0, pw - w, 0, ph - h)
                
                img0_ref = torch.nn.functional.pad(img0_ref, padding)
                img1_ref = torch.nn.functional.pad(img1_ref, padding)

                flow_list, mask_list, merged = self.model(
                    img0_ref, 
                    img1_ref, 
                    ratio, 
                    iterations = iterations
                    )

                result = warp(img0, flow_list[3][:, :2, :h, :w]) * mask_list[3][:, :, :h, :w] + warp(img1, flow_list[3][:, 2:4, :h, :w]) * (1 - mask_list[3][:, :, :h, :w])
                result = result[0].clone().cpu().detach().numpy().transpose(1, 2, 0).astype(np.float16)
                del img0, img1, img0_ref, img1_ref, flow_list, mask_list, merged, incoming_data, outgoing_data
            return result

    def bake_flame_tw_setup(self, tw_setup_string):
        # parses tw setup from flame and returns dictionary
        # with baked frame - value pairs
        
        def dictify(r, root=True):
            def string_to_value(s):
                if (s.find('-') <= 0) and s.replace('-', '', 1).isdigit():
                    return int(s)
                elif (s.find('-') <= 0) and (s.count('.') < 2) and \
                        (s.replace('-', '', 1).replace('.', '', 1).isdigit()):
                    return float(s)
                elif s == 'True':
                    return True
                elif s == 'False':
                    return False
                else:
                    return s

            from copy import copy

            if root:
                return {r.tag: dictify(r, False)}

            d = copy(r.attrib)
            if r.text:
                # d["_text"] = r.text
                d = r.text
            for x in r.findall('./*'):
                if x.tag not in d:
                    v = dictify(x, False)
                    if not isinstance (d, dict):
                        d = {}
                    if isinstance (v, str):
                        d[x.tag] = string_to_value(v)
                    else:
                        d[x.tag] = []
                if isinstance(d[x.tag], list):
                    d[x.tag].append(dictify(x, False))
            return d

        class FlameChannellInterpolator:
            # An attempt of a python rewrite of Julit Tarkhanov's original
            # Flame Channel Parsr written in Ruby.

            class ConstantSegment:
                def __init__(self, from_frame, to_frame, value):
                    self._mode = 'constant'
                    self.start_frame = from_frame
                    self.end_frame = to_frame
                    self.v1 = value

                def mode(self):
                    return self._mode

                def defines(self, frame):
                    return (frame < self.end_frame) and (frame >= self.start_frame)

                def value_at(self, frame):
                    return self.v1

            class LinearSegment(ConstantSegment):
                def __init__(self, from_frame, to_frame, value1, value2):
                    self.vint = (value2 - value1)
                    super().__init__(from_frame, to_frame, value1)
                    self._mode = 'linear'

                def value_at(self, frame):
                    on_t_interval = (frame - self.start_frame) / (self.end_frame - self.start_frame)
                    return self.v1 + (on_t_interval * self.vint)
                
                '''
                self.HERMATRIX = np.array([
                    [2,  -2,  1,  1],
                    [-3, 3,   -2, -1],
                    [0,   0,  1,  0],
                    [1,   0,  0,  0]
                ])
                '''

            class HermiteSegment(LinearSegment):
                def __init__(self, from_frame, to_frame, value1, value2, tangent1, tangent2):
                    self.start_frame, self.end_frame = from_frame, to_frame
                    frame_interval = (self.end_frame - self.start_frame)
                    self._mode = 'hermite'

                    self.HERMATRIX = np.array([
                        [0,  0,  0,  1],
                        [1,  1,  1,  1],
                        [0,  0,  1,  0],
                        [3,  2,  1,  0]
                    ])
                    self.HERMATRIX = np.linalg.inv(self.HERMATRIX)

                    # Default tangents in flame are 0, so when we do None.to_f this is what we will get
                    # CC = {P1, P2, T1, T2}
                    p1, p2, t1, t2 = value1, value2, tangent1 * frame_interval, tangent2 * frame_interval
                    self.hermite = np.array([p1, p2, t1, t2])
                    self.basis = np.dot(self.HERMATRIX, self.hermite)

                def value_at(self, frame):
                    if frame == self.start_frame:
                        return self.hermite[0]

                    # Get the 0 < T < 1 interval we will interpolate on
                    # Q[frame_] = P[ ( frame - 149 ) / (time_to - time_from)]
                    t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

                    # S[s_] = {s^3, s^2, s^1, s^0}
                    multipliers_vec = np.array([t ** 3, t ** 2, t ** 1, t ** 0])

                    # P[s_] = S[s].h.CC
                    interpolated_scalar = np.dot(self.basis, multipliers_vec)
                    return interpolated_scalar

            class BezierSegment(LinearSegment):
                class Pt:
                    def __init__(self, x, y, tanx, tany):
                        self.x = x
                        self.y = y
                        self.tanx = tanx
                        self.tany = tany
                
                def __init__(self, x1, x2, y1, y2, t1x, t1y, t2x, t2y):
                    super().__init__(x1, x2, y1, y2)
                    self.a = self.Pt(x1, y1, t1x, t1y)
                    self.b = self.Pt(x2, y2, t2x, t2y)
                    self._mode = 'bezier'

                def value_at(self, frame):
                    if frame == self.start_frame:
                        return self.a.y
                    
                    t = self.approximate_t(frame, self.a.x, self.a.tanx, self.b.tanx, self.b.x)
                    vy = self.bezier(t, self.a.y, self.a.tany, self.b.tany, self.b.y)
                    return vy
                
                def bezier(self, t, a, b, c, d):
                    return a + (a*(-3) + b*3)*(t) + (a*3 - b*6 + c*3)*(t**2) + (-a + b*3 - c*3 + d)*(t**3)
                
                def clamp(self, value):
                    if value < 0:
                        return 0.0
                    elif value > 1:
                        return 1.0
                    else:
                        return value
                
                APPROXIMATION_EPSILON = 1.0e-09
                VERYSMALL = 1.0e-20
                MAXIMUM_ITERATIONS = 100
                
                def approximate_t(self, atX, p0x, c0x, c1x, p1x):
                    if atX - p0x < self.VERYSMALL:
                        return 0.0
                    elif p1x - atX < self.VERYSMALL:
                        return 1.0

                    u, v = 0.0, 1.0
                    
                    for i in range(self.MAXIMUM_ITERATIONS):
                        a = (p0x + c0x) / 2.0
                        b = (c0x + c1x) / 2.0
                        c = (c1x + p1x) / 2.0
                        d = (a + b) / 2.0
                        e = (b + c) / 2.0
                        f = (d + e) / 2.0
                        
                        if abs(f - atX) < self.APPROXIMATION_EPSILON:
                            return self.clamp((u + v) * 0.5)
                        
                        if f < atX:
                            p0x = f
                            c0x = e
                            c1x = c
                            u = (u + v) / 2.0
                        else:
                            c0x = a
                            c1x = d
                            p1x = f
                            v = (u + v) / 2.0
                    
                    return self.clamp((u + v) / 2.0)

            class ConstantPrepolate(ConstantSegment):
                def __init__(self, to_frame, base_value):
                    super().__init__(float('-inf'), to_frame, base_value)
                    self._mode = 'ConstantPrepolate'

                def value_at(self, frame):
                    return self.v1

            class ConstantExtrapolate(ConstantSegment):
                def __init__(self, from_frame, base_value):
                    super().__init__(from_frame, float('inf'), base_value)
                    self._mode = 'ConstantExtrapolate'

                def value_at(self, frame):
                    return self.v1
                
            class LinearPrepolate(ConstantPrepolate):
                def __init__(self, to_frame, base_value, tangent):
                    self.tangent = float(tangent)
                    super().__init__(to_frame, base_value)
                    self._mode = 'LinearPrepolate'

                def value_at(self, frame):
                    frame_diff = (self.end_frame - frame)
                    return self.v1 + (self.tangent * frame_diff)
                
            class LinearExtrapolate(ConstantExtrapolate):
                def __init__(self, from_frame, base_value, tangent):
                    self.tangent = float(tangent)
                    super().__init__(from_frame, base_value)
                    self._mode = 'LinearExtrapolate'

                def value_at(self, frame):
                    frame_diff = (frame - self.start_frame)
                    return self.v1 + (self.tangent * frame_diff)

            class ConstantFunction(ConstantSegment):
                def __init__(self, value):
                    super().__init__(float('-inf'), float('inf'), value)
                    self._mode = 'ConstantFunction'

                def defines(self, frame):
                    return True

                def value_at(self, frame):
                    return self.v1

            def __init__(self, channel):
                self.segments = []
                self.extrap = channel.get('Extrap', 'constant')

                if channel.get('Size', 0) == 0:
                    self.segments = [FlameChannellInterpolator.ConstantFunction(channel.get('Value', 0))]
                elif channel.get('Size') == 1 and self.extrap == 'constant':
                    self.segments = [FlameChannellInterpolator.ConstantFunction(channel.get('Value', 0))]
                elif channel.get('Size') == 1 and self.extrap == 'linear':
                    kframes = channel.get('KFrames')
                    frame = list(kframes.keys())[0]
                    base_value = kframes[frame].get('Value')
                    left_tangent = kframes[frame].get('LHandle_dY') / kframes[frame].get('LHandle_dX') * -1
                    right_tangent = kframes[frame].get('RHandle_dY') / kframes[frame].get('RHandle_dX')
                    self.segments = [
                        FlameChannellInterpolator.LinearPrepolate(frame, base_value, left_tangent),
                        FlameChannellInterpolator.LinearExtrapolate(frame, base_value, right_tangent)
                    ]
                else:
                    self.segments = self.create_segments_from_channel(channel)

            def sample_at(self, frame):
                if self.extrap == 'cycle':
                    return self.sample_from_segments(self.frame_number_in_cycle(frame))
                elif self.extrap == 'revcycle':
                    return self.sample_from_segments(self.frame_number_in_revcycle(frame))
                else:
                    return self.sample_from_segments(frame)

            def first_defined_frame(self):
                first_f = self.segments[0].end_frame
                if first_f == float('-inf'):
                    return 1
                return first_f

            def last_defined_frame(self):
                last_f = self.segments[-1].start_frame
                if last_f == float('inf'):
                    return 100
                return last_f

            def frame_number_in_revcycle(self, frame):
                animated_across = self.last_defined_frame() - self.first_defined_frame()
                offset = abs(frame - self.first_defined_frame())
                absolute_unit = offset % animated_across
                cycles = offset // animated_across
                if cycles % 2 == 0:
                    return self.first_defined_frame() + absolute_unit
                else:
                    return self.last_defined_frame() - absolute_unit

            def frame_number_in_cycle(self, frame):
                animated_across = self.last_defined_frame() - self.first_defined_frame()
                offset = frame - self.first_defined_frame()
                modulo = offset % animated_across
                return self.first_defined_frame() + modulo

            def create_segments_from_channel(self, channel):
                kframes = channel.get('KFrames')
                index_frames = list(kframes.keys())
                # First the prepolating segment
                segments = [self.pick_prepolation(channel.get('Extrap', 'constant'), kframes[index_frames[0]], kframes[index_frames[1]])]

                # Then all the intermediate segments, one segment between each pair of keys
                for index, key in enumerate(index_frames[:-1]):
                    segments.append(self.key_pair_to_segment(kframes[key], kframes[index_frames[index + 1]]))

                # and the extrapolator
                segments.append(self.pick_extrapolation(channel.get('Extrap', 'constant'), kframes[index_frames[-2]], kframes[index_frames[-1]]))
                return segments

            def sample_from_segments(self, at_frame):
                for segment in self.segments:
                    if segment.defines(at_frame):
                        return segment.value_at(at_frame)
                raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')
            
            def segment_mode(self, at_frame):
                for segment in self.segments:
                    if segment.defines(at_frame):
                        return segment.mode()
                raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')
            
            def get_segment(self, at_frame):
                for segment in self.segments:
                    if segment.defines(at_frame):
                        return segment
                raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')

            def pick_prepolation(self, extrap_symbol, first_key, second_key):
                if extrap_symbol == 'linear' and second_key:
                    if first_key.get('CurveMode') != 'linear':
                        first_key_left_slope = first_key.get('LHandle_dY') / first_key.get('LHandle_dX') * -1
                        return FlameChannellInterpolator.LinearPrepolate(
                            first_key.get('Frame'), 
                            first_key.get('Value'), 
                            first_key_left_slope)
                    else:
                        # For linear keys the tangent actually does not do anything, so we need to look a frame
                        # ahead and compute the increment
                        increment = (second_key.get('Value') - first_key.get('Value')) / (second_key.get('Frame') - first_key.get('Frame'))
                        return FlameChannellInterpolator.LinearPrepolate(first_key.get('Frame'), first_key.get('Value'), increment)
                else:
                    return FlameChannellInterpolator.ConstantPrepolate(first_key.get('Frame'), first_key.get('Value'))
            
            def pick_extrapolation(self, extrap_symbol, previous_key, last_key):
                if extrap_symbol != 'constant':
                    if previous_key and (last_key.get('CurveMode')  == 'linear' or last_key.get('CurveOrder')  == 'linear'):
                        # For linear keys the tangent actually does not do anything, so we need to look a frame
                        # ahead and compute the increment
                        increment = (last_key.get('Value') - previous_key.get('Value')) / (last_key.get('Frame') - previous_key.get('Frame'))
                        return FlameChannellInterpolator.LinearExtrapolate(last_key.get('Frame'), last_key.get('Value'), increment)
                    else:
                        last_key_right_slope = last_key.get('LHandle_dY') / last_key.get('LHandle_dX')
                        return FlameChannellInterpolator.LinearExtrapolate(last_key.get('Frame'), last_key.get('Value'), last_key_right_slope)
                else:
                    return FlameChannellInterpolator.ConstantExtrapolate(last_key.get('Frame'), last_key.get('Value'))

            def key_pair_to_segment(self, key, next_key):
                key_left_tangent = key.get('LHandle_dY') / key.get('LHandle_dX') * -1
                key_right_tangent = key.get('RHandle_dY') / key.get('RHandle_dX')
                next_key_left_tangent = next_key.get('LHandle_dY') / next_key.get('LHandle_dX') * -1
                next_key_right_tangent = next_key.get('RHandle_dY') / next_key.get('RHandle_dX')

                if key.get('CurveMode') == 'bezier':
                    return FlameChannellInterpolator.BezierSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'),
                        key.get('Value'), 
                        next_key.get('Value'),
                        float(key.get('Frame')) + float(key.get('RHandle_dX')), 
                        float(key.get('Value')) + float(key.get('RHandle_dY')),
                        float(next_key.get('Frame')) + float(next_key.get('LHandle_dX')),
                        float(next_key.get('Value')) + float(next_key.get('LHandle_dY'))
                        )
                
                elif (key.get('CurveMode') in ['natural', 'hermite']) and (key.get('CurveOrder') == 'cubic'):
                    return FlameChannellInterpolator.HermiteSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'), 
                        key.get('Value'), 
                        next_key.get('Value'),
                        key_right_tangent, 
                        next_key_left_tangent
                        )
                elif (key.get('CurveMode') in ['natural', 'hermite']) and (key.get('CurveOrder') == 'quartic'):
                    return FlameChannellInterpolator.HermiteSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'), 
                        key.get('Value'), 
                        next_key.get('Value'),
                        key_right_tangent, 
                        next_key_left_tangent
                        )
                elif key.get('CurveMode') == 'constant':
                    return FlameChannellInterpolator.ConstantSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'), 
                        key.get('Value')
                        )
                else:  # Linear and safe
                    return FlameChannellInterpolator.LinearSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'), 
                        key.get('Value'), 
                        next_key.get('Value')
                        )

        def approximate_speed_curve(tw_setup_string, start, end, tw_channel):
            from xml.dom import minidom
            xml = minidom.parseString(tw_setup_string)  
            tw_speed_timing = {}
            TW_SpeedTiming = xml.getElementsByTagName('TW_SpeedTiming')
            keys = TW_SpeedTiming[0].getElementsByTagName('Key')
            for key in keys:
                index = key.getAttribute('Index') 
                frame = key.getElementsByTagName('Frame')
                if frame:
                    frame = (frame[0].firstChild.nodeValue)
                value = key.getElementsByTagName('Value')
                if value:
                    value = (value[0].firstChild.nodeValue)
                tw_speed_timing[int(index)] = {'frame': int(frame), 'value': float(value)}

            if tw_speed_timing[0]['frame'] > start:
                # we need to extrapolate backwards from the first 
                # keyframe in SpeedTiming channel

                anchor_frame_value = tw_speed_timing[0]['value']
                for frame_number in range(tw_speed_timing[0]['frame'] - 1, start - 1, -1):
                    if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                        step_back = tw_channel[min(list(tw_channel.keys()))] / 100
                    else:
                        step_back = (tw_channel[frame_number + 1] + tw_channel[frame_number]) / 200
                    frame_value_map[frame_number] = anchor_frame_value - step_back
                    anchor_frame_value = frame_value_map[frame_number]

            # build up frame values between keyframes of SpeedTiming channel
            for key_frame_index in range(0, len(tw_speed_timing.keys()) - 1):
                # The value from my gess algo is close to the one in flame but not exact
                # and error is accumulated. SO quick and dirty way is to do forward
                # and backward pass and mix them rationally

                range_start = tw_speed_timing[key_frame_index]['frame']
                range_end = tw_speed_timing[key_frame_index + 1]['frame']
                
                if range_end == range_start + 1:
                # keyframes on next frames, no need to interpolate
                    frame_value_map[range_start] = tw_speed_timing[key_frame_index]['value']
                    frame_value_map[range_end] = tw_speed_timing[key_frame_index + 1]['value']
                    continue

                forward_pass = {}
                anchor_frame_value = tw_speed_timing[key_frame_index]['value']
                forward_pass[range_start] = anchor_frame_value

                for frame_number in range(range_start + 1, range_end):
                    if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                        step = tw_channel[max(list(tw_channel.keys()))] / 100
                    else:
                        step = (tw_channel[frame_number] + tw_channel[frame_number + 1]) / 200
                    forward_pass[frame_number] = anchor_frame_value + step
                    anchor_frame_value = forward_pass[frame_number]
                forward_pass[range_end] = tw_speed_timing[key_frame_index + 1]['value']
                
                backward_pass = {}
                anchor_frame_value = tw_speed_timing[key_frame_index + 1]['value']
                backward_pass[range_end] = anchor_frame_value
                
                for frame_number in range(range_end - 1, range_start -1, -1):
                    if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                        step_back = tw_channel[min(list(tw_channel.keys()))] / 100
                    else:
                        step_back = (tw_channel[frame_number + 1] + tw_channel[frame_number]) / 200
                    backward_pass[frame_number] = anchor_frame_value - step_back
                    anchor_frame_value = backward_pass[frame_number]
                
                backward_pass[range_start] = tw_speed_timing[key_frame_index]['value']

                def hermite_curve(t):
                    P0, P1 = 0, 1
                    T0, T1 = 8, 1.8 # this values are made by hand to get approximation closer to flame
                    h00 = 2*t**3 - 3*t**2 + 1  # Compute basis function 1
                    h10 = t**3 - 2*t**2 + t    # Compute basis function 2
                    h01 = -2*t**3 + 3*t**2     # Compute basis function 3
                    h11 = t**3 - t**2          # Compute basis function 4

                    return h00 * P0 + h10 * T0 + h01 * P1 + h11 * T1


                work_range = list(forward_pass.keys())
                ratio = 0
                rstep = 1 / len(work_range)
                for frame_number in sorted(work_range):
                    frame_value_map[frame_number] = forward_pass[frame_number] * (1 - hermite_curve(ratio)) + backward_pass[frame_number] * hermite_curve(ratio)
                    ratio += rstep

            last_key_index = list(sorted(tw_speed_timing.keys()))[-1]
            if tw_speed_timing[last_key_index]['frame'] < end:
                # we need to extrapolate further on from the 
                # last keyframe in SpeedTiming channel
                anchor_frame_value = tw_speed_timing[last_key_index]['value']
                frame_value_map[tw_speed_timing[last_key_index]['frame']] = anchor_frame_value

                for frame_number in range(tw_speed_timing[last_key_index]['frame'] + 1, end + 1):
                    if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                        step = tw_channel[max(list(tw_channel.keys()))] / 100
                    else:
                        step = (tw_channel[frame_number] + tw_channel[frame_number + 1]) / 200
                    frame_value_map[frame_number] = anchor_frame_value + step
                    anchor_frame_value = frame_value_map[frame_number]

            return frame_value_map

        import numpy as np
        import xml.etree.ElementTree as ET

        tw_setup_xml = ET.fromstring(tw_setup_string)
        tw_setup = dictify(tw_setup_xml)

        start_frame = int(tw_setup['Setup']['Base'][0]['Range'][0]['Start'])
        end_frame = int(tw_setup['Setup']['Base'][0]['Range'][0]['End'])
        # TW_Timing_size = int(tw_setup['Setup']['State'][0]['TW_Timing'][0]['Channel'][0]['Size'][0]['_text'])

        # TW_SpeedTiming_size = tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size']
        TW_RetimerMode = tw_setup['Setup']['State'][0]['TW_RetimerMode']

        frame_value_map = {}

        if TW_RetimerMode == 1:
            # 'Timing' channel is enough
            tw_channel = 'TW_Timing'
            channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
            if 'KFrames' in channel.keys():
                channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Value'])}
            interpolator = FlameChannellInterpolator(channel)
            for frame_number in range (start_frame, end_frame+1):
                frame_value_map[frame_number] = round(interpolator.sample_at(frame_number), 4)
            return frame_value_map

        else:
            # speed - based timewarp seem to
            # work in a different way
            # depending on a segment mode

            tw_channel = 'TW_Speed'
            channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
            if 'KFrames' in channel.keys():
                channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Value'])}
            speed_channel = dict(channel)
            tw_channel = 'TW_SpeedTiming'
            channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
            if 'KFrames' in channel.keys():
                channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Value'])}
            speed_timing_channel = dict(channel)

            if 'quartic' in tw_setup_string:
                speed_interpolator = FlameChannellInterpolator(speed_channel)
                interpolated_speed_channel = {}
                for frame_number in range (start_frame, end_frame+1):
                    interpolated_speed_channel[frame_number] = round(speed_interpolator.sample_at(frame_number), 4)
                return approximate_speed_curve(tw_setup_string, self.record_in, self.record_out, interpolated_speed_channel)

            timing_interpolator = FlameChannellInterpolator(speed_timing_channel)

            for frame_number in range (start_frame, end_frame+1):
                frame_value_map[frame_number] = round(timing_interpolator.sample_at(frame_number), 4)
                    
        return frame_value_map

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

        def __init__(self, argv, parent=None):
            super(Worker, self).__init__(parent)
            self.argv = argv

        def run(self):
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
            
            # print(f'{json_info}')

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

            tw = Timewarp(json_info)
            result = tw.process()
            # self.result.emit(result, '')

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

    # Main window class
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.initUI()
            
            # Redirect sys.stdout and sys.stderr
            sys.stdout = Stream(newText=self.onUpdateText)
            sys.stderr = Stream(newText=self.onUpdateText)

            self.worker_status = False
            self.worker = Worker(sys.argv)
            self.worker.result.connect(self.handleWorkerResult)
            self.worker.finished.connect(self.onWorkerFinished)
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
                    background-color: #292929;
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
            
            self.setGeometry(300, 300, 600, 400)
            self.setWindowTitle('Capture tqdm Output')
            self.show()

        def onUpdateText(self, text):
            # text = text.rstrip('\n')
            # Check for carriage return indicating a progress update
            if '\r' in text:
                text.replace('\n', '')
                text.replace('\r', '')
                # text = text.rstrip('\n')
                # text = text.rstrip('\r')
                if self.last_progress_line is not None:
                    # Remove the last progress update
                    self.text_edit.moveCursor(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                    self.text_edit.textCursor().removeSelectedText()
                    self.text_edit.moveCursor(QTextCursor.End)
                    self.text_edit.textCursor().deletePreviousChar()  # Remove newline left after text removal
                self.last_progress_line = text
            else:
                pass
                # text = text + '\n'
                # Not a progress line, so append normally and reset progress tracking
                # self.last_progress_line = None
                # text = text + '\n'  # Add newline for regular prints

            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(text)  # Insert the text at the end
            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()

        def keyPressEvent(self, event):        
            # Check if Ctrl+C was pressed
            if event.key() == Qt.Key_C and event.modifiers():
                self.close()
            else:
                super().keyPressEvent(event)

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