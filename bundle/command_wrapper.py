import os
import sys
import pickle
import time

from pprint import pprint


if __name__ == '__main__':
    try:
        cmd_package_file = open(sys.argv[1], 'rb')
        cmd_package = pickle.load(cmd_package_file)
        cmd_package_file.close()
    except Exception as e:
        print('unable to load command arguments from %s' % sys.argv[1])
        print(e)
        sys.exit(1)
    
    try:
        cmd = 'python3 ' + cmd_package.get('cmd_name')
        if cmd_package.get('cpu'):
            cmd = 'export OMP_NUM_THREADS=1; python3 ' + cmd_package.get('cmd_name') + ' --cpu'

        cmd_quoted_args = cmd_package.get('quoted_args')
        cmd_args = cmd_package.get('args')

        for quoted_arg_key in cmd_quoted_args.keys():
            cmd += ' --' + quoted_arg_key + ' "' + str(cmd_quoted_args.get(quoted_arg_key)) + '"'

        for arg_key in cmd_args.keys():
            cmd += ' --' + arg_key + ' ' + str(cmd_args.get(arg_key))
    except Exception as e:
        print ('error parsing command arguments:')
        print (e)
        pprint (cmd_package)

    os.system(cmd)
    
    try:
        os.remove(sys.argv[1])
    except Exception as e:
        print('unable to remove lock file %s' % sys.argv[1])
        print(e)

    sys.exit()