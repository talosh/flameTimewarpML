import os
import sys
import argparse
import importlib
import platform

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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

def find_folders_with_exr(path):
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

def clear_lines(n=2):
    """Clears a specified number of lines in the terminal."""
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)

def write_exr(image_data, filename, half_float = False, pixelAspectRatio = 1.0):
    import struct

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

def resize_image(tensor, new_h, new_w):
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

    # Resize
    resized_tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)

    # Adjust tensor shape back to [h, w, c]
    resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)

    return resized_tensor

def halve(exr_file_path):
    exr_data = read_openexr_file(exr_file_path)
    h = exr_data['shape'][0]
    w = exr_data['shape'][1]
    img0 = exr_data['image_data']
    img0 = torch.from_numpy(img0)
    img0 = img0.to(device = device, dtype = torch.float32)
    img0 = resize_image(img0, h//2, w//2)
    img0 = img0.to(dtype=torch.half)
    write_exr(img0.cpu().detach().numpy(), exr_file_path, half_float = True)
    del img0

def main():
    parser = argparse.ArgumentParser(description='In-place resize exrs to half-res (WARNING - DESTRUCTIVE!)')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to folders with exrs')
    args = parser.parse_args()

    folders_with_exr = find_folders_with_exr(args.dataset_path)

    exr_files = []

    for folder_path in sorted(folders_with_exr):
        folder_exr_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.exr')]
        folder_exr_files.sort()
        exr_files.extend(folder_exr_files)

    idx = 0
    for exr_file_path in exr_files:
        clear_lines(1)
        print (f'\rFile [{idx+1} / {len(exr_files)}], {os.path.basename(exr_file_path)}')
        try:
            halve(exr_file_path)
        except Exception as e:
            print (f'\n\nError halving {exr_file_path}: {e}')
        idx += 1
    print ('')

if __name__ == "__main__":
    main()
