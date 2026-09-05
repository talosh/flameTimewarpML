import os
import sys

from adsk.libwiretapPythonClientAPI import (
    WireTapClient,
    WireTapClientUninit,
    WireTapServerId,
    WireTapServerHandle,
    WireTapNodeHandle,
    WireTapClipFormat,
    WireTapInt,
    WireTapStr,
)

import ctypes
import flame

# assuming you have numpy unpacked in /var/tmp/numpy
# one can just unzip wheel

sys.path.insert(0, '/var/tmp/numpy')
import numpy as np
del sys.path[0]

python_api = ctypes.CDLL(sys.executable)
python_api.PyUnicode_FromKindAndData.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_ssize_t]
python_api.PyUnicode_FromKindAndData.restype = ctypes.py_object

class WireTapException(Exception):
    def __init__(self, msg):
        flame.messages.show_in_dialog(
            title = 'flameTimewrarpML',
            message = msg,
            type = 'error',
            buttons = ['Ok']
        )

server_handle = WireTapServerHandle('localhost')

num_frames = 4

library = flame.projects.current_project.create_shared_library('twml')
parent_node_handle = WireTapNodeHandle(server_handle, flame.PyClip.get_wiretap_node_id(library))

new_clip_node_handle = WireTapNodeHandle()
clip_format = WireTapClipFormat(
    24,
    24,  # width, height
    3 * 8,  # bits per pixel
    3,  # number of channels
    24,  # frame rate
    1,  # pixel ratio
    WireTapClipFormat.ScanFormat.SCAN_FORMAT_PROGRESSIVE,
    WireTapClipFormat.FORMAT_RGB(),
)

if not parent_node_handle.createClipNode(
    "MyNewClip",  # display name
    clip_format,  # clip format
    "CLIP",  # extended (server-specific) type
    new_clip_node_handle,  # created node returned here
):
    raise WireTapException(
        "Unable to create clip node: %s." % parent_node_handle.lastError()
    )

if not new_clip_node_handle.setNumFrames(int(num_frames)):
    raise WireTapException(
        "Unable to set the number of frames: %s." % new_clip_node_handle.lastError()
    )

new_fmt = WireTapClipFormat()
if not new_clip_node_handle.getClipFormat(new_fmt):
    raise WireTapException(
        "Unable to obtain clip format: %s." % new_clip_node_handle.lastError()
    )

# array of red
pattern = np.array([0xff, 0x00, 0x00], dtype=np.uint8)
repeated_array = np.tile(pattern, new_fmt.frameBufferSize() // len(pattern))
remainder = new_fmt.frameBufferSize() % len(pattern)
if remainder > 0:
    arr = np.concatenate([repeated_array,
                            np.zeros(remainder, dtype=np.uint8)])


for frame_number in range(0, num_frames):

    # method 1
    # buff_out = str(arr.tobytes(), 'latin-1')
    # method 2
    # buff_out = python_api.PyUnicode_FromKindAndData(ctypes.c_int(1), arr.tobytes(), arr.nbytes)
    # method 3

    temp_file_name = '/var/tmp/test.raw'
    with open(temp_file_name, 'wb') as f:
        arr.tofile(f, sep='')
        buff_out = f.read()

    if not new_clip_node_handle.writeFrame(frame_number, buff_out, new_fmt.frameBufferSize()):
        raise WireTapException(
            "Unable to obtain write frame %i: %s."
            % (frame_number, new_clip_node_handle.lastError())
        )
    print("Successfully wrote frame %i." % frame_number)
