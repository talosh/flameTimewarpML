import sys
import numpy as np
import struct
from pprint import pprint

height, width, depth = 10, 10, 3
bit_depth = 8  # Can be 8, 10, or 16

# Dummy data
# Define image size
width = 48
height = 24

# Initialize arrays
R = np.ones((height, width), dtype=np.uint8)
G = np.full((height, width), 0.5, dtype=np.uint8)
B = np.zeros((height, width), dtype=np.uint8)
A = np.full((height, width), 0.8, dtype=np.uint8)

arr = np.zeros((height, width, depth), dtype=np.uint8)
arr[:,:,0] = R
arr[:,:,0] = G
arr[:,:,0] = B

offset_to_image_data = 1664
total_file_size = offset_to_image_data + arr.size * bit_depth // 8

file_info_header = struct.pack(
    "<4sL4s2s2s4s4sLHHLLLLLLLLLL32s100s24s",
    b'SDPX', # Magic number
    offset_to_image_data, # Offset to image data
    b'V1.0', # Version number
    b'01', # File was written from a "right to left" machine
    b'01', # Image information is formatted for a "right to left" machine
    b'4321', # This field is only used when there's encryption (there's none here)
    b'1234', # This field is only used when there's encryption (there's none here)
    0, # Offset to generic section header (not used)
    0, # Offset to industry specific header (not used)
    0, # Offset to user-defined header (not used)
    total_file_size, # Size of total file
    total_file_size, # Ditto (DPX spec requires this redundancy)
    0, # Ditto
    0, # Ditto
    0, # Ditto
    0, # Ditto
    0, # Ditto
    0, # Ditto
    b'', # Reserved for future use (must be set to 0)
    b'', # Internal filename for the image
    b'' # Timestamp of file creation
)

# Define the image information header
image_info_header = struct.pack(
    "<HhH2B2H4s4s4B4x4sHHHHLL4sH2B2H2L2H6x48s",
    1, # Number of image elements
    width, # Pixels per line
    height, # Number of lines
    bit_depth, # Data sign
    0, # Reference low data code value
    0, # Reference low quantity represented
    (2**bit_depth)-1, # Reference high data code value
    1, # Reference high quantity represented
    bit_depth, # Bits per pixel
    0 if bit_depth != 10 else 1, # Packing: 0 for filled to byte boundary, 1 for 10-bit
    1, # Encoding: 1 for no compression
    offset_to_image_data, # Data offset
    b'', # Reserved for future use (must be set to 0)
    0, # End of line padding
    0, # End of image padding
    b'RGB ', # Description of image element
    0, # Transfer characteristics for the image element
    0, # Colorimetric characteristics for the image element
    0, # Bit depth of image element
    0, # Reserved for future use (must be set to 0)
    0, # Reserved for future use (must be set to 0)
    b'' # Reserved for future use (must be set to 0)
)

# Define the image orientation header
image_orientation_header = struct.pack(
    "<2L2H4L4H2L32s32s32s32s",
    0, # X offset of the lower left corner of the image
    0, # Y offset of the lower left corner of the image
    0, # X center of the image
    0, # Y center of the image
    width, # X original size of the image
    height, # Y original size of the image
    width, # X source image size
    height, # Y source image size
    0, # Source image filename
    b'', # Source image date and time
    b'', # Input device
    b'', # Input device model number
    b'', # Input device serial number
    b'' # Border validity data for the image
)

# Combine the headers and the image data
dpx_data = file_info_header + image_info_header + image_orientation_header + arr.tobytes()

# Write the data to a file
with open('test.dpx', 'wb') as f:
    f.write(dpx_data)

# Write to file
# write_exr('test.exr', width, height, R, G, B, alpha=A, half_float = False)
