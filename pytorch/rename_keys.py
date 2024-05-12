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

key_mapping_01 = {
    'block0.encode01.0.weight': 'block0.encode01.downconv01.conv1.weight',
    'block0.encode01.0.bias':   'block0.encode01.downconv01.conv1.bias',
    'block0.encode01.2.weight': 'block0.encode01.conv01.conv1.weight',
    'block0.encode01.2.bias':   'block0.encode01.conv01.conv1.bias',
    'block0.encode01.4.weight': 'block0.encode01.conv02.conv1.weight',
    'block0.encode01.4.bias':   'block0.encode01.conv02.conv1.bias',
    'block0.encode01.6.weight': 'block0.encode01.upsample01.conv1.weight',
    'block0.encode01.6.bias':   'block0.encode01.upsample01.conv1.bias',

    'block1.encode01.0.weight': 'block1.encode01.downconv01.conv1.weight',
    'block1.encode01.0.bias':   'block1.encode01.downconv01.conv1.bias',
    'block1.encode01.2.weight': 'block1.encode01.conv01.conv1.weight',
    'block1.encode01.2.bias':   'block1.encode01.conv01.conv1.bias',
    'block1.encode01.4.weight': 'block1.encode01.conv02.conv1.weight',
    'block1.encode01.4.bias':   'block1.encode01.conv02.conv1.bias',
    'block1.encode01.6.weight': 'block1.encode01.upsample01.conv1.weight',
    'block1.encode01.6.bias':   'block1.encode01.upsample01.conv1.bias',

    'block2.encode01.0.weight': 'block2.encode01.downconv01.conv1.weight',
    'block2.encode01.0.bias':   'block2.encode01.downconv01.conv1.bias',
    'block2.encode01.2.weight': 'block2.encode01.conv01.conv1.weight',
    'block2.encode01.2.bias':   'block2.encode01.conv01.conv1.bias',
    'block2.encode01.4.weight': 'block2.encode01.conv02.conv1.weight',
    'block2.encode01.4.bias':   'block2.encode01.conv02.conv1.bias',
    'block2.encode01.6.weight': 'block2.encode01.upsample01.conv1.weight',
    'block2.encode01.6.bias':   'block2.encode01.upsample01.conv1.bias',

    'block3.encode01.0.weight': 'block3.encode01.downconv01.conv1.weight',
    'block3.encode01.0.bias':   'block3.encode01.downconv01.conv1.bias',
    'block3.encode01.2.weight': 'block3.encode01.conv01.conv1.weight',
    'block3.encode01.2.bias':   'block3.encode01.conv01.conv1.bias',
    'block3.encode01.4.weight': 'block3.encode01.conv02.conv1.weight',
    'block3.encode01.4.bias':   'block3.encode01.conv02.conv1.bias',
    'block3.encode01.6.weight': 'block3.encode01.upsample01.conv1.weight',
    'block3.encode01.6.bias':   'block3.encode01.upsample01.conv1.bias',
}

def main():
    parser = argparse.ArgumentParser(description='Move exrs to corresponding folders')

    # Required argument
    parser.add_argument('source', type=str, help='Path to source state dict')
    parser.add_argument('dest', type=str, help='Path to dest state dict')
    args = parser.parse_args()

    checkpoint = torch.load(args.source)
    src_state_dict = checkpoint.get('flownet_state_dict')
    
    dest_state_dict = {}    
    for key in src_state_dict:
        new_key = key_mapping_01.get(key, key)
        dest_state_dict[new_key] = src_state_dict[key]

    checkpoint['flownet_state_dict'] = dest_state_dict
    torch.save(checkpoint, args.dest)

    '''
    for key in src_state_dict.keys():
        print (key)
    '''

if __name__ == "__main__":
    main()