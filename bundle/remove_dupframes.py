import os
import sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import threading
import skvideo.io
from queue import Queue, Empty
warnings.filterwarnings("ignore")

from pprint import pprint, pformat
import time
import psutil

import multiprocessing as mp

ThreadsFlag = True
IOProcesses = []

# Exception handler
def exeption_handler(exctype, value, tb):
    import traceback

    locks = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'locks')
    cmd = 'rm -f ' + locks + '/*'
    os.system(cmd)

    pprint ('%s in %s' % (value, exctype))
    pprint(traceback.format_exception(exctype, value, tb))
    sys.__excepthook__(exctype, value, tb)
    input("Press Enter to continue...")
sys.excepthook = exeption_handler

# ctrl+c handler
import signal
def signal_handler(sig, frame):
    global ThreadsFlag
    ThreadsFlag = False
    time.sleep(0.1)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def clear_write_buffer(args, write_buffer, input_duration):
    global ThreadsFlag
    global IOProcesses

    cnt = 0
    while ThreadsFlag:
        item = write_buffer.get()
            
        if item is None:
            break

        if cnt < input_duration:
            path = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(cnt))
            p = mp.Process(target=cv2.imwrite, args=(path, item[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF], ))
            p.start()
            IOProcesses.append(p)
        cnt += 1

def build_read_buffer(user_args, read_buffer, videogen):
    global ThreadsFlag

    for frame in videogen:
        frame_data = cv2.imread(os.path.join(user_args.input, frame), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        read_buffer.put(frame_data)
    read_buffer.put(None)

if __name__ == '__main__':
    start = time.time()
    print('scanning for duplicate frames...')

    parser = argparse.ArgumentParser(description='Remove duplicate frames')
    parser.add_argument('--input', dest='input', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)

    args = parser.parse_args()
    if (args.output is None or args.input is None):
         parser.print_help()
         sys.exit()

    img_formats = ['.exr',]
    files_list = []
    for f in os.listdir(args.input):
        name, ext = os.path.splitext(f)
        if ext in img_formats:
            files_list.append(f)

    input_duration = len(files_list)
    if input_duration < 3:
        print('not enough input frames: %s given' % input_duration)
        input("Press Enter to continue...")
        sys.exit()

    files_list.sort()
    write_buffer = Queue(maxsize=mp.cpu_count() - 3)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, files_list))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer, input_duration))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    pbar = tqdm(total=input_duration, desc='Total frames', unit='frame')
    pbar_dup = tqdm(total=input_duration, desc='Duplicates', bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}')

    IP = None
    for file in files_list:
        current_frame = read_buffer.get()
        pbar.update(1) # type: ignore

        IC = torch.from_numpy(np.transpose(current_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
        if type(IP) is not type(None):
            diff = (F.interpolate(IC,scale_factor=0.5, mode='bicubic', align_corners=False)
                    - F.interpolate(IP, scale_factor=0.5, mode='bicubic', align_corners=False)).abs()
            if diff.max() < 2e-3:
                pbar_dup.update(1)
                continue

        write_buffer.put(current_frame)
        IP = IC
    
    write_buffer.put(None)
    ThreadsFlag = False
    pbar.close() # type: ignore
    pbar_dup.close()

    for p in IOProcesses:
        p.join()
