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

        path = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(cnt))
        p = mp.Process(target=cv2.imwrite, args=(path, item[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF], ))
        # p.daemon = True
        p.start()
        IOProcesses.append(p)
        cnt += 1

    
def build_read_buffer(user_args, read_buffer, videogen):
    global ThreadsFlag

    for frame in videogen:
        frame_data = cv2.imread(os.path.join(user_args.input, frame), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        read_buffer.put(frame_data)
    read_buffer.put(None)

def make_inference_rational(model, I0, I1, ratio, rthreshold=0.02, maxcycles = 8, UHD=False):
    I0_ratio = 0.0
    I1_ratio = 1.0
    
    if ratio <= I0_ratio:
        return I0
    if ratio >= I1_ratio:
        return I1
    
    for inference_cycle in range(0, maxcycles):
        middle = model.inference(I0, I1, UHD)
        middle_ratio = ( I0_ratio + I1_ratio ) / 2

        if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
            return middle

        if ratio > middle_ratio:
            I0 = middle
            I0_ratio = middle_ratio
        else:
            I1 = middle
            I1_ratio = middle_ratio
    
    return middle

if __name__ == '__main__':
    start = time.time()

    msg = 'Fill / Remove duplicate frames\n'
    msg += 'detect duplicate frames and fill it with interpolated frames instead\n'
    msg += 'or just cut them out of resulting sequence'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('--input', dest='input', type=str, default=None, help='folder with input sequence')
    parser.add_argument('--output', dest='output', type=str, default=None, help='folder to output sequence to')
    parser.add_argument('--model', dest='model', type=str, default='./trained_models/default/v1.8.model')
    parser.add_argument('--remove', dest='remove', action='store_true', help='remove duplicate frames')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='do not use GPU at all, process only on CPU')

    args = parser.parse_args()
    if (args.output is None or args.input is None):
         parser.print_help()
         sys.exit()

    if args.remove:
        print('Initializing duplicate frames removal...')
    else:
        print('Initializing duplicate frames removal...')

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

    output_folder = os.path.abspath(args.output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files_list.sort()
    write_buffer = Queue(maxsize=mp.cpu_count() - 3)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, files_list))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer, input_duration))

    if not args.remove:
        if torch.cuda.is_available() and not args.cpu:
            from model.RIFE_HD import Model     # type: ignore
            model = Model()
            model.load_model(args.model, -1)
            model.eval()
            model.device()
        else:
            from model_cpu.RIFE_HD import Model     # type: ignore
            model = Model()
            model.load_model(args.model, -1)
            model.eval()
            model.device()

        print ('AI model loaded: %s' % args.model)
    
        first_image = cv2.imread(os.path.join(args.input, files_list[0]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        h, w, _ = first_image.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
    
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    print('scanning for duplicate frames...')

    pbar = tqdm(total=input_duration, desc='Total frames', unit='frame')
    pbar_dup = tqdm(total=input_duration, desc='Duplicates', bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}')

    IPrevious = None
    dframes = 0
    for file in files_list:
        current_frame = read_buffer.get()
        pbar.update(1) # type: ignore

        ICurrent = torch.from_numpy(np.transpose(current_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
        if not args.remove:
            ICurrent = F.pad(ICurrent, padding)

        if type(IPrevious) is not type(None):
            
            diff = (F.interpolate(ICurrent,scale_factor=0.5, mode='bicubic', align_corners=False)
                    - F.interpolate(IPrevious, scale_factor=0.5, mode='bicubic', align_corners=False)).abs()

            if diff.max() < 2e-3:
                pbar_dup.update(1)
                dframes += 1
                continue
        
        if dframes and not args.remove:
            # start = time.time()
            rstep = 1 / ( dframes + 1 )
            ratio = rstep
            for dframe in range(0, dframes):
                mid = make_inference_rational(model, IPrevious, ICurrent, ratio)
                if sys.platform == 'darwin' or args.cpu:
                    mid = (((mid[0]).cpu().detach().numpy().transpose(1, 2, 0)))
                else:
                    mid = (((mid[0]).cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(mid[:h, :w])
                # pprint ('ratio: %s, %s' % (ratio, time.time() - start))
                ratio += rstep

        write_buffer.put(current_frame)
        IPrevious = ICurrent
        dframes = 0
    
    write_buffer.put(None)
    ThreadsFlag = False
    pbar.close() # type: ignore
    pbar_dup.close()

    for p in IOProcesses:
        p.join(timeout=8)

    for p in IOProcesses:
        p.terminate()
        p.join(timeout=0)

    import hashlib
    lockfile = os.path.join('locks', hashlib.sha1(output_folder.encode()).hexdigest().upper() + '.lock')
    if os.path.isfile(lockfile):
        os.remove(lockfile)