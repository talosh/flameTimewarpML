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
        print('Initializing TimewarpML from Flame setup...')
        import torch

        if self.json_info.get('cpu'):
            print('Processing on CPU (Slow)')
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("mps") if platform.system() == 'Darwin' else torch.device('cuda')

        self.model_path = self.json_info.get('model_path')
        self.model = self.find_and_import_model(self.model_path)
        # if not self.model:
        #    print (f'Unable to import model from file {self.model_path}')

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

        def run(self):
            import os
            import sys
            import random
            import shutil
            import queue
            import threading
            import time
            import platform

            def clear_lines(n=2):
                """Clears a specified number of lines in the terminal."""
                CURSOR_UP_ONE = '\x1b[1A'
                ERASE_LINE = '\x1b[2K'
                for _ in range(n):
                    sys.stdout.write(CURSOR_UP_ONE)
                    sys.stdout.write(ERASE_LINE)

            print (f'Initializing PyTorch...')

            import numpy as np
            import torch

            clear_lines()

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

            device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')
            if args.all_gpus:
                device = 'cuda'

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

    # Main window class
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.initUI()
            
            # Redirect sys.stdout and sys.stderr
            sys.stdout = Stream(newText=self.onUpdateText)
            sys.stderr = Stream(newText=self.onUpdateText)

            self.worker_status = False
            self.worker = Worker(sys.argv, window=self)
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
            
            self.setGeometry(300, 300, 900, 400)
            self.setWindowTitle('TimewarpML Finetune')
            self.show()

        def onUpdateText(self, text):
            import re
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()

            clear_line_pattern = re.compile(r'\x1b\[2K')
            cursor_up_pattern = re.compile(r'\x1b\[1A')

            parts = re.split('(\x1b\[2K|\x1b\[1A|\r)', text)

            for part in parts:
                if part == '\x1b[2K':
                    cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()
                    cursor.deletePreviousChar()  # Remove newline left after text removal
                elif part == '\x1b[1A':
                    cursor.movePosition(QTextCursor.Up)
                elif part == '\r':
                    cursor.movePosition(QTextCursor.StartOfLine)
                    cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()
                else:
                    clean_part = clear_line_pattern.sub('', part)
                    clean_part = cursor_up_pattern.sub('', clean_part)
                    cursor.insertText(clean_part)
                    cursor.insertText('\n')

            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()

        '''
        def onUpdateText(self, text):
            import re
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
        '''

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