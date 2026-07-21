import os
import sys
import argparse
import importlib
import platform

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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

def read_image_file(file_path, header_only = False):
    result = {'spec': None, 'image_data': None}
    inp = oiio.ImageInput.open(file_path)
    if inp :
        spec = inp.spec()
        result['spec'] = spec
        if not header_only:
            channels = spec.nchannels
            result['image_data'] = inp.read_image(0, 0, 0, channels).transpose(1, 0, 2)
        inp.close()
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

def main():
    parser = argparse.ArgumentParser(description='In-place recompress exrs to half PIZ (WARNING - DESTRUCTIVE!)')

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
        # clear_lines(1)
        print (f'\r{" "*120}', end='')
        print (f'\rFile [{idx+1} / {len(exr_files)}], {os.path.basename(exr_file_path)}', end='')
        try:
            result = read_image_file(exr_file_path)
            image_data = result['image_data']
            spec = result['spec']

            if image_data is None or spec is None:
                raise RuntimeError(f"Could not read image or header from {exr_file_path}")

            # Transpose back to (W, H, C) for OIIO
            image_data_for_write = image_data.transpose(1, 0, 2)

            # Clone the spec and modify compression only
            new_spec = oiio.ImageSpec(spec)
            new_spec.set_format(oiio.TypeDesc.TypeHalf)  # ensure half-precision
            new_spec.attribute("compression", "piz")

            out = oiio.ImageOutput.create(exr_file_path)
            if not out:
                raise RuntimeError(f"Could not create output for {exr_file_path}")

            if not out.open(exr_file_path, new_spec):
                raise RuntimeError(f"Failed to open output file {exr_file_path}")
            
            if not out.write_image(image_data_for_write):
                raise RuntimeError(f"Failed to write image to {exr_file_path}")

            out.close()

        except Exception as e:
            print (f'\nError reading {exr_file_path}: {e}')
        idx += 1
    print ('')

if __name__ == "__main__":
    main()
