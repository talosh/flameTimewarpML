import os
import sys
import argparse

try:
    import numpy as np
    import torch
except:
    python_executable_path = sys.executable
    if '.miniconda' in python_executable_path:
        print ('Unable to import Numpy and PyTorch libraries')
        print (f'Using {python_executable_path} python interpreter')
        sys.exit()

def main():
    parser = argparse.ArgumentParser(description='Move exrs to corresponding folders')

    # Required argument
    parser.add_argument('source', type=str, help='Path to source state dict')
    parser.add_argument('dest', type=str, help='Path to dest state dict')
    args = parser.parse_args()

    src_checkpoint = torch.load(args.source)
    src_state_dict = src_checkpoint.get('flownet_state_dict')

    for key in src_state_dict.keys():
        print (key)

if __name__ == "__main__":
    main()