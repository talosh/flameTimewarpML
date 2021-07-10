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

import inference_common

IOThreadsFlag = True
IOProcesses = []
cv2.setNumThreads(1)

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
    global IOThreadsFlag
    IOThreadsFlag = False
    time.sleep(0.1)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def clear_write_buffer(args, write_buffer, input_duration, pbar=None):
    global IOThreadsFlag
    global IOProcesses

    cv2_flags = []
    if args.bit_depth != 32:
        cv2_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]

    number_of_write_threads = 4

    while IOThreadsFlag:

        alive_processes = []
        for process in IOProcesses:
            if process.is_alive():
                alive_processes.append(process)
            else:
                process.join(timeout=0)
        IOProcesses = list(alive_processes)

        item = write_buffer.get()
                    
        frame_number, image_data = item
        if frame_number == -1:
            IOThreadsFlag = False
            break

        path = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(frame_number))
        if len(IOProcesses) < number_of_write_threads:
            try:
                p = mp.Process(target=cv2.imwrite, args=(path, image_data[:, :, ::-1], cv2_flags, ))
                p.start()
                IOProcesses.append(p)
            except:
                try:
                    cv2.imwrite(path, image_data[:, :, ::-1], cv2_flags)
                except Exception as e:
                    print ('Error wtiring %s: %s' % (path, e))
        else:
            try:
                cv2.imwrite(path, image_data[:, :, ::-1], cv2_flags)
            except Exception as e:
                print ('Error wtiring %s: %s' % (path, e))

        if pbar:
            pbar.update(1)


def build_read_buffer(user_args, read_buffer, videogen):
    global IOThreadsFlag

    for frame in videogen:
        frame_data = cv2.imread(os.path.join(user_args.input, frame), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        read_buffer.put(frame_data)
    read_buffer.put(None)

def make_inference_rational(model, I0, I1, ratio, rthreshold=0.02, maxcycles = 8, scale=1.0, always_interp=False):
    I0_ratio = 0.0
    I1_ratio = 1.0
    rational_m = torch.mean(I0) * ratio + torch.mean(I1) * (1 - ratio)
    
    if not always_interp:
        if ratio <= I0_ratio + rthreshold / 2:
            return I0
        if ratio >= I1_ratio - rthreshold / 2:
            return I1
    
    for inference_cycle in range(0, maxcycles):
        middle = model.inference(I0, I1, scale)
        middle_ratio = ( I0_ratio + I1_ratio ) / 2

        if not always_interp:
            if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                return middle #+ (rational_m - torch.mean(middle)).expand_as(middle)

        if ratio > middle_ratio:
            I0 = middle
            I0_ratio = middle_ratio
        else:
            I1 = middle
            I1_ratio = middle_ratio
    
    return middle #+ (rational_m - torch.mean(middle)).expand_as(middle)

def make_inference_rational_cpu(model, I0, I1, ratio, frame_num, w, h, write_buffer, rthreshold=0.02, maxcycles = 8, scale=1.0, always_interp=False):
    device = torch.device("cpu")   
    torch.set_grad_enabled(False) 
    
    I0_ratio = 0.0
    I1_ratio = 1.0
    # rational_m = torch.mean(I0) * ratio + torch.mean(I1) * (1 - ratio)

    if not always_interp:
        if ratio <= I0_ratio + rthreshold / 2:
            I0 = (((I0[0]).cpu().detach().numpy().transpose(1, 2, 0)))
            write_buffer.put((frame_num, I0[:h, :w]))
            return
        if ratio >= I1_ratio - rthreshold / 2:
            I1 = (((I1[0]).cpu().detach().numpy().transpose(1, 2, 0)))
            write_buffer.put((frame_num, I1[:h, :w]))
            return
    
    for inference_cycle in range(0, maxcycles):
        middle = model.inference(I0, I1, scale)
        middle_ratio = ( I0_ratio + I1_ratio ) / 2

        if not always_interp:
            if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                # middle = middle + (rational_m - torch.mean(middle)).expand_as(middle)
                middle = (((middle[0]).cpu().detach().numpy().transpose(1, 2, 0)))
                write_buffer.put((frame_num, middle[:h, :w]))
                return

        if ratio > middle_ratio:
            middle = middle.detach()
            I0 = middle.to(device, non_blocking=True)
            I0_ratio = middle_ratio
        else:
            middle = middle.detach()
            I1 = middle.to(device, non_blocking=True)
            I1_ratio = middle_ratio
    
    # middle = middle + (rational_m - torch.mean(middle)).expand_as(middle)
    middle = (((middle[0]).cpu().detach().numpy().transpose(1, 2, 0)))
    write_buffer.put((frame_num, middle[:h, :w]))
    return

if __name__ == '__main__':
    start = time.time()

    msg = 'Fill / Remove duplicate frames\n'
    msg += 'detect duplicate frames and fill it with interpolated frames instead\n'
    msg += 'or just cut them out of resulting sequence'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('--input', dest='input', type=str, default=None, help='folder with input sequence')
    parser.add_argument('--output', dest='output', type=str, default=None, help='folder to output sequence to')
    parser.add_argument('--model', dest='model', type=str, default='./trained_models/default/v2.0.model')
    parser.add_argument('--remove', dest='remove', action='store_true', help='remove duplicate frames')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='do not use GPU at all, process only on CPU')
    parser.add_argument('--flow_scale', dest='flow_scale', type=float, help='motion analysis resolution scale')
    parser.add_argument('--bit_depth', dest='bit_depth', type=int, default=16)

    args = parser.parse_args()
    if (args.output is None or args.input is None):
         parser.print_help()
         sys.exit()

    if args.remove:
        print('Initializing duplicate frames removal...')
    else:
        print('Initializing duplicate frames interpolation...')

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

    read_buffer = Queue(maxsize=444)
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, files_list))

    if args.remove:
        write_buffer = Queue(maxsize=mp.cpu_count() - 3)
        _thread.start_new_thread(clear_write_buffer, (args, write_buffer, input_duration))

        if torch.cuda.is_available() and not args.cpu:
            device = torch.device("cuda")
            torch.set_grad_enabled(False)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            torch.set_grad_enabled(False)

        print('scanning for duplicate frames...')
        pbar = tqdm(total=input_duration, desc='Total frames  ', unit='frame')
        pbar_dup = tqdm(total=input_duration, desc='Removed    ', bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}')

        IPrevious = None
        output_frame_num = 1
        for file in files_list:
            current_frame = read_buffer.get()
            pbar.update(1) # type: ignore
            ICurrent = torch.from_numpy(np.transpose(current_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            if type(IPrevious) is not type(None):
                diff = (F.interpolate(ICurrent,scale_factor=0.5, mode='bicubic', align_corners=False)
                    - F.interpolate(IPrevious, scale_factor=0.5, mode='bicubic', align_corners=False)).abs()
                if diff.max() < 2e-3:
                    pbar_dup.update(1)
                    continue
            write_buffer.put((output_frame_num, current_frame))
            IPrevious = ICurrent
            output_frame_num += 1

        write_buffer.put((-1, -1))

        while(IOThreadsFlag):
            time.sleep(0.01)

        pbar.close() # type: ignore
        pbar_dup.close()

    elif torch.cuda.is_available() and not args.cpu:
        # Process on GPU

        model = inference_common.load_model(args.model)
        model.eval()
        model.device()
        print ('Trained model loaded: %s' % args.model)
    
        first_image = cv2.imread(os.path.join(args.input, files_list[0]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        h, w, _ = first_image.shape
        pv = max(32, int(32 / args.flow_scale))
        ph = ((h - 1) // pv + 1) * pv
        pw = ((w - 1) // pv + 1) * pv
        padding = (0, pw - w, 0, ph - h)
    
        device = torch.device("cuda")
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        pbar = tqdm(total=input_duration, desc='Total frames', unit='frame')
        pbar_dup = tqdm(total=input_duration, desc='Interpolating', bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}')

        write_buffer = Queue(maxsize=mp.cpu_count() - 3)
        _thread.start_new_thread(clear_write_buffer, (args, write_buffer, input_duration, pbar))

        IPrevious = None
        dframes = 0
        output_frame_num = 1

        for file in files_list:
            current_frame = read_buffer.get()

            ICurrent = torch.from_numpy(np.transpose(current_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            if not args.remove:
                ICurrent = F.pad(ICurrent, padding)

            if type(IPrevious) is not type(None):
                
                diff = (F.interpolate(ICurrent,scale_factor=0.5, mode='bicubic', align_corners=False)
                        - F.interpolate(IPrevious, scale_factor=0.5, mode='bicubic', align_corners=False)).abs()

                if diff.max() < 2e-3:
                    dframes += 1
                    continue
            
            if dframes and not args.remove:
                rstep = 1 / ( dframes + 1 )
                ratio = rstep
                for dframe in range(0, dframes):
                    mid = make_inference_rational(model, IPrevious, ICurrent, ratio, scale = args.flow_scale)
                    mid = (((mid[0]).cpu().numpy().transpose(1, 2, 0)))
                    write_buffer.put((output_frame_num, mid[:h, :w]))
                    # pbar.update(1) # type: ignore
                    pbar_dup.update(1)
                    output_frame_num += 1
                    ratio += rstep

            write_buffer.put((output_frame_num, current_frame))
            # pbar.update(1) # type: ignore
            IPrevious = ICurrent
            output_frame_num += 1
            dframes = 0

        # send write loop exit code
        write_buffer.put((-1, -1))

        # it should put IOThreadsFlag to False it return
        while(IOThreadsFlag):
            time.sleep(0.01)

        # pbar.update(1)
        pbar.close() # type: ignore
        pbar_dup.close()
    
    else:
        # process on CPU

        model = inference_common.load_model(args.model, cpu=True)
        model.eval()
        model.device()
        print ('Trained model loaded: %s' % args.model)

        first_image = cv2.imread(os.path.join(args.input, files_list[0]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        h, w, _ = first_image.shape
        pv = max(32, int(32 / args.flow_scale))
        ph = ((h - 1) // pv + 1) * pv
        pw = ((w - 1) // pv + 1) * pv
        padding = (0, pw - w, 0, ph - h)

        device = torch.device("cpu")
        torch.set_grad_enabled(False)

        sim_workers, thread_ram = inference_common.safe_threads_number(h, w)

        pbar = tqdm(total=input_duration, desc='Total frames', unit='frame')
        pbar_dup = tqdm(total=input_duration, desc='Interpolating', bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}')

        write_buffer = mp.Queue(maxsize=mp.cpu_count() - 3)
        _thread.start_new_thread(clear_write_buffer, (args, write_buffer, input_duration, pbar))

        IPrevious = None
        dframes = 0
        output_frame_num = 1
        active_workers = []

        for file in files_list:
            current_frame = read_buffer.get()
            # pbar.update(1) # type: ignore

            ICurrent = torch.from_numpy(np.transpose(current_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            ICurrent = F.pad(ICurrent, padding)

            if type(IPrevious) is not type(None):
                
                diff = (F.interpolate(ICurrent,scale_factor=0.5, mode='bicubic', align_corners=False)
                        - F.interpolate(IPrevious, scale_factor=0.5, mode='bicubic', align_corners=False)).abs()

                if diff.max() < 2e-3:
                    dframes += 1
                    continue
            
            if dframes:
                rstep = 1 / ( dframes + 1 )
                ratio = rstep
                last_thread_time = time.time()
                for dframe in range(dframes):
                    p = mp.Process(target=make_inference_rational_cpu, args=(model, IPrevious, ICurrent, ratio, output_frame_num, w, h, write_buffer), kwargs = {'scale': args.flow_scale})
                    p.start()
                    active_workers.append(p)

                    if (time.time() - last_thread_time) < (thread_ram / 8):
                        if sim_workers > 1:
                            time.sleep(thread_ram/8)

                    while len(active_workers) >= sim_workers:
                        finished_workers = []
                        alive_workers = []
                        for worker in active_workers:
                            if not worker.is_alive():
                                finished_workers.append(worker)
                            else:
                                alive_workers.append(worker)
                        active_workers = list(alive_workers)
                        time.sleep(0.01)
                    last_thread_time = time.time()

                    # mid = (((ICurrent[0]).cpu().detach().numpy().transpose(1, 2, 0)))
                    # write_buffer.put((output_frame_num, mid[:h, :w]))
                    pbar_dup.update(1)
                    output_frame_num += 1
                    ratio += rstep

            write_buffer.put((output_frame_num, current_frame))

            IPrevious = ICurrent
            output_frame_num += 1
            dframes = 0
        
        # wait for all active worker threads left to finish                
        for p in active_workers:
            p.join()

        # send write loop exit code
        write_buffer.put((-1, -1))

        # it should put IOThreadsFlag to False it return
        while(IOThreadsFlag):
            time.sleep(0.01)

        pbar.close() # type: ignore
        pbar_dup.close()
        
    for p in IOProcesses:
        p.join(timeout=8)

    for p in IOProcesses:
        p.terminate()
        p.join(timeout=0)

    '''
    import hashlib
    lockfile = os.path.join('locks', hashlib.sha1(output_folder.encode()).hexdigest().upper() + '.lock')
    if os.path.isfile(lockfile):
        os.remove(lockfile)
    '''

    # input("Press Enter to continue...")
    sys.exit(0)