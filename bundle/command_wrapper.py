import os
import sys
import pickle
import time

from pprint import pprint


if __name__ == '__main__':
    try:
        cmd_args_file = open(sys.argv[1], 'rb')
        cmd_args = pickle.load(cmd_args_file)
        cmd_args_file.close()
    except Exception as e:
        print('unable to load command arguments from %s' % sys.argv[1])
        print(e)
        sys.exit(1)

    excluded_keys = [
        'cmd_name',
        'cpu',
        'input',
        'output',
        'model'
    ]

    cmd = 'python3 ' + cmd_args.get('cmd_name')
    if cmd_args.get('cpu'):
        cmd = 'export OMP_NUM_THREADS=1; python3 ' + cmd_args.get('cmd_name') + ' --cpu'
    cmd += ' --input "' + cmd_args.get('input') + '"'
    cmd += ' --output "' + cmd_args.get('output') + '"'
    cmd += ' --model "' + cmd_args.get('model') +'"'
 
    for arg_key in cmd_args.keys():
        if arg_key in excluded_keys:
            continue        
        cmd += ' --' + arg_key + ' ' + str(cmd_args.get(arg_key))

    os.system(cmd)
    
    try:
        os.remove(sys.argv[1])
    except Exception as e:
        print('unable to remove lock file %s' % sys.argv[1])
        print(e)

    sys.exit()