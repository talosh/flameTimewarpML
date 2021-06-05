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

    pprint (cmd_args)
    time.sleep(2)
    
    try:
        os.remove(sys.argv[1])
    except Exception as e:
        print('unable to remove lock file %s' % sys.argv[1])
        print(e)

    sys.exit()