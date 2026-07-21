import OpenEXR
import Imath
import numpy as np

from pprint import pprint, pformat

# Define image size
width = 48
height = 24

# Initialize arrays
R = np.ones((height, width), dtype=np.float16)
G = np.full((height, width), 0.5, dtype=np.float16)
B = np.zeros((height, width), dtype=np.float16)
A = np.full((height, width), 0.8, dtype=np.float16)

# Convert to string format for OpenEXR
R_str = R.tobytes()
G_str = G.tobytes()
B_str = B.tobytes()
A_str = A.tobytes()

# Create the OpenEXR header
header = OpenEXR.Header(width, height)
half_chan = Imath.Channel(Imath.PixelType(OpenEXR.HALF))
header['channels'] = dict([(c, half_chan) for c in "RGBA"])
pprint (header)
header['compression'] = Imath.Compression(Imath.Compression.NO_COMPRESSION) # Specify no compression

# Open the file for writing
exr_file = OpenEXR.OutputFile('output.exr', header)

# Write to the file
exr_file.writePixels({'R': R_str, 'G': G_str, 'B': B_str, 'A': A_str})
