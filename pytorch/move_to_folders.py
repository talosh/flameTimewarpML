import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Move exrs to corresponding folders')

    # Required argument
    parser.add_argument('path', type=str, help='Path to folder with exrs')
    args = parser.parse_args()

    files_and_dirs = os.listdir(args.path)
    exr_files = [file for file in files_and_dirs if file.endswith('.exr')]

    exr_file_names = set()
    for exr_file_name in exr_files:
        exr_file_names.add(exr_file_name.split('.')[0])

    print (exr_file_names)

if __name__ == "__main__":
    main()