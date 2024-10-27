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

def write_image_file(file_path, image_data, image_spec):
    out = oiio.ImageOptput.create(file_path)
    if out:
        out.open(file_path, image_spec)
        out.write_image(image_data)
        out.close()

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

def halve(exr_file_path, new_h):
    device = torch.device('cuda')
    exr_data = read_openexr_file(exr_file_path)
    h = exr_data['shape'][0]
    w = exr_data['shape'][1]

    aspect_ratio = w / h
    new_w = int(new_h * aspect_ratio)

    img0 = exr_data['image_data']
    img0 = torch.from_numpy(img0)
    img0 = img0.to(device = device, dtype = torch.float32)
    img0 = resize_image(img0, new_h, new_w)
    img0 = img0.to(dtype=torch.half)
    write_exr(img0.cpu().detach().numpy(), exr_file_path, half_float = True)
    del img0

def main():
    parser = argparse.ArgumentParser(description='In-place resize exrs to half-res (WARNING - DESTRUCTIVE!)')

    # Required argument
    parser.add_argument('h', type=int, help='New height')
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
            halve(exr_file_path, args.h)
        except Exception as e:
            print (f'\n\nError halving {exr_file_path}: {e}')
        idx += 1
    print ('')

if __name__ == "__main__":
    main()