import os
import sys
import argparse

from pprint import pprint

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
    parser = argparse.ArgumentParser(description='Training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
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
        idx += 1
    print ('')

if __name__ == "__main__":
    main()
