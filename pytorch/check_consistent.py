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
    parser = argparse.ArgumentParser(description='In-place resize exrs to half-res (WARNING - DESTRUCTIVE!)')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to folders with exrs')
    args = parser.parse_args()

    folders_with_exr = find_folders_with_exr(args.dataset_path)

    exr_files = []

    for folder_idx, folder_path in enumerate(sorted(folders_with_exr)):
        print (f'\rFolder [{folder_idx+1} / {len(folders_with_exr)}], {folder_path}', end='')
        folder_exr_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.exr')]
        folder_exr_files.sort()
        
        '''
        frame_numbers = []
        for filename in folder_exr_files:
            parts = filename.split('.')
            try:
                frame_number = int(parts[-2])
                frame_numbers.append(frame_number)
            except ValueError:
                print (f'\nFormat error in {folder_path}: {filename}')
        
        frame_numbers.sort()
        for i in range(1, len(frame_numbers)):
            if frame_numbers[i] != frame_numbers[i - 1] + 1:
                print(f'\nMissing or non-consecutive frame between {frame_numbers[i-1]} and {frame_numbers[i]}')
        '''

    print ('')

if __name__ == "__main__":
    main()
