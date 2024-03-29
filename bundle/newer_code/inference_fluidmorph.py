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

def clear_write_buffer(args, write_buffer, tot_frame):
    global IOThreadsFlag
    global IOProcesses

    folder = args.output
    cv2_flags = []
    if args.bit_depth != 32:
        cv2_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]

    cnt = 0
    print ('rendering %s frames to %s' % (tot_frame, folder))
    pbar = tqdm(total=tot_frame, unit='frame')

    while IOThreadsFlag:
        item = write_buffer.get()
                
        frame_number, image_data = item
        if frame_number == -1:
            pbar.close() # type: ignore
            IOThreadsFlag = False
            break
        
        # print ('recieved %s' % frame_number)
        path = os.path.join(os.path.abspath(folder), '{:0>7d}.exr'.format(frame_number))
        p = mp.Process(target=cv2.imwrite, args=(path, image_data[:, :, ::-1], cv2_flags, ))
        p.start()
        IOProcesses.append(p)

        pbar.update(1) # type: ignore
        cnt += 1

def build_read_buffer(folder, read_buffer, file_list):
    global IOThreadsFlag
    for frame in file_list:
        path = os.path.join(folder, frame)
        if not os.path.isfile(path):
            print ('Unable to find file: %s' % path)
        frame_data = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        read_buffer.put(frame_data)
    read_buffer.put(None)

def make_inference_rational(model, I0, I1, ratio, rthreshold = 0.02, maxcycles = 49, scale=1.0, always_interp=False):
    I0_ratio = 0.0
    I1_ratio = 1.0
    rational_m = torch.mean(I0) * ratio + torch.mean(I1) * (1 - ratio)

    if not always_interp:
        if ratio <= I0_ratio + rthreshold / 2:
            return I0
        if ratio >= I1_ratio - rthreshold / 2:
            return I1
    # print ('target ratio: %s' % ratio)
    for inference_cycle in range(0, maxcycles):
        middle = model.inference(I0, I1, scale)
        middle_ratio = ( I0_ratio + I1_ratio ) / 2
        if not always_interp:
            if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                return middle # + (rational_m - torch.mean(middle)).expand_as(middle)

        if ratio > middle_ratio:
            I0 = middle
            I0_ratio = middle_ratio
        else:
            I1 = middle
            I1_ratio = middle_ratio

    return middle # + (rational_m - torch.mean(middle)).expand_as(middle)


def three_of_a_perfect_pair(incoming_frame, outgoing_frame, frame_num, ratio, device, padding, model, args, h, w, write_buffer, rthreshold):
    # print ('target ratio %s' % ratio)
    maxcycles = 49
    I0_ratio = 0.0
    I1_ratio = 1.0
    
    if ratio <= I0_ratio + rthreshold / 2:
        write_buffer.put((frame_num, incoming_frame))
        return
    if ratio >= I1_ratio - rthreshold / 2:
        write_buffer.put((frame_num, outgoing_frame))
        return

    for inference_cycle in range(0, maxcycles):
        start = time.time()

        I0 = torch.from_numpy(np.transpose(incoming_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
        I0 = F.pad(I0, padding)
        I1 = torch.from_numpy(np.transpose(outgoing_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
        I1 = F.pad(I1, padding)

        # rational_m = torch.mean(I0) * ratio + torch.mean(I1) * (1 - ratio)

        middle = model.inference(I0, I1, args.flow_scale)
        # middle = middle + (rational_m - torch.mean(middle)).expand_as(middle)
        middle = (((middle[0]).cpu().detach().numpy().transpose(1, 2, 0)))
        middle_ratio = ( I0_ratio + I1_ratio ) / 2
        
        if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
            write_buffer.put((frame_num, middle[:h, :w]))
            return

        if ratio > middle_ratio:
            incoming_frame = middle[:h, :w]
            I0_ratio = middle_ratio
        else:
            outgoing_frame = middle[:h, :w]
            I1_ratio = middle_ratio

    write_buffer.put((frame_num, middle[:h, :w]))
    return

if __name__ == '__main__':
    start = time.time()

    msg = 'FluidMorph\n'
    msg += 'Attempts to replicate Avid fluid morph transition\n'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('--incoming', dest='incoming', type=str, default=None)
    parser.add_argument('--outgoing', dest='outgoing', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--model', dest='model', type=str, default='./trained_models/default/v2.0.model')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='process only on CPU(s)')
    parser.add_argument('--curve', dest='curve', type=int, default=1, help='1 - linear, 2 - smooth')
    parser.add_argument('--flow_scale', dest='flow_scale', type=float, help='motion analysis resolution scale')
    parser.add_argument('--bit_depth', dest='bit_depth', type=int, default=16)

    args = parser.parse_args()
    if (args.incoming is None or args.outgoing is None or args.output is None):
         parser.print_help()
         sys.exit()

    img_formats = ['.exr',]
    incoming_files_list = []
    for f in os.listdir(args.incoming):
        name, ext = os.path.splitext(f)
        if ext in img_formats:
            incoming_files_list.append(f)

    outgoing_files_list = []
    for f in os.listdir(args.outgoing):
        name, ext = os.path.splitext(f)
        if ext in img_formats:
            outgoing_files_list.append(f)

    incoming_input_duration = len(incoming_files_list)
    outgoing_input_duration = len(outgoing_files_list)

    if incoming_input_duration < 3:
        print('not enough frames in incoming sequence: %s given' % incoming_input_duration)
        input("Press Enter to continue...")
        sys.exit()

    if outgoing_input_duration < 3:
        print('not enough frames in outgoing sequence: %s given' % outgoing_input_duration)
        input("Press Enter to continue...")
        sys.exit()
    if incoming_input_duration != outgoing_input_duration:
        print('incoming sequence duration should be equal to outgoing')
        print('incoming sequence %s frames long\noutgoing sequencce %s framse long' % (incoming_input_duration, outgoing_input_duration))
        input("Press Enter to continue...")
        sys.exit()

    input_duration = incoming_input_duration

    print('initializing FluidMorph ML...')

    incoming_first_image = cv2.imread(os.path.join(args.incoming, incoming_files_list[0]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
    h_inc, w_inc, _ = incoming_first_image.shape
    outgoing_first_image = cv2.imread(os.path.join(args.outgoing, outgoing_files_list[0]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
    h_outg, w_outg, _ = outgoing_first_image.shape

    if ( h_inc != h_outg ) or ( w_inc != w_outg ):
        print('incoming and outgoing images dimentions are not equal')
        print('incoming sequence %s x %s\noutgoing sequencce %s x %s' % (w_inc, h_inc, w_outg, h_outg))
        input("Press Enter to continue...")
        sys.exit()
    
    h = h_inc
    w = w_inc
    
    pv = max(32, int(32 / args.flow_scale))
    ph = ((h - 1) // pv + 1) * pv
    pw = ((w - 1) // pv + 1) * pv
    padding = (0, pw - w, 0, ph - h)

    output_folder = os.path.abspath(args.output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    if torch.cuda.is_available() and not args.cpu:
        # process on GPU

        incoming_files_list.sort()
        incoming_files_list.append(incoming_files_list[-1])
        outgoing_files_list.sort()
        outgoing_files_list.append(outgoing_files_list[-1])

        write_buffer = Queue(maxsize=mp.cpu_count() - 3)
        incoming_read_buffer = Queue(maxsize=500)
        outgoing_read_buffer = Queue(maxsize=500)

        _thread.start_new_thread(build_read_buffer, (args.incoming, incoming_read_buffer, incoming_files_list))
        _thread.start_new_thread(build_read_buffer, (args.outgoing, outgoing_read_buffer, outgoing_files_list))

        model = inference_common.load_model(args.model)
        model.eval()
        model.device()
        print ('Trained model loaded: %s' % args.model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_grad_enabled(False)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        _thread.start_new_thread(clear_write_buffer, (args, write_buffer, input_duration))

        rstep = 1 / ( input_duration + 1 )
        ratio = rstep

        for frame in range(1, input_duration + 1):
            incoming_frame = incoming_read_buffer.get()
            outgoing_frame = outgoing_read_buffer.get()

            I0 = torch.from_numpy(np.transpose(incoming_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I0 = F.pad(I0, padding)
            I1 = torch.from_numpy(np.transpose(outgoing_frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I1 = F.pad(I1, padding)

            mid = make_inference_rational(model, I0, I1, ratio, rthreshold = rstep / 2, scale = args.flow_scale)
            mid = (((mid[0]).cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put((frame, mid[:h, :w]))
            
            ratio += rstep

        # send write loop exit code
        write_buffer.put((-1, -1))

        # it should put IOThreadsFlag to False it return
        while(IOThreadsFlag):
            time.sleep(0.01)

    else:
        # process on CPU(s)
        
        manager = mp.Manager()
        frames_written = manager.dict()

        incoming_files_list.sort()
        incoming_files_list.append(incoming_files_list[-1])
        outgoing_files_list.sort()
        outgoing_files_list.append(outgoing_files_list[-1])

        write_buffer = mp.Queue(maxsize=mp.cpu_count() - 3)
        incoming_read_buffer = Queue(maxsize=500)
        outgoing_read_buffer = Queue(maxsize=500)

        _thread.start_new_thread(build_read_buffer, (args.incoming, incoming_read_buffer, incoming_files_list))
        _thread.start_new_thread(build_read_buffer, (args.outgoing, outgoing_read_buffer, outgoing_files_list))

        model = inference_common.load_model(args.model, cpu=True)
        model.eval()
        model.device()
        print ('Trained model loaded: %s' % args.model)

        device = torch.device('cpu')
        torch.set_grad_enabled(False)
        
        sim_workers, thread_ram = inference_common.safe_threads_number(h, w)

        '''
        max_cpu_workers = mp.cpu_count() - 2
        available_ram = psutil.virtual_memory()[1]/( 1024 ** 3 )
        megapixels = ( h * w ) / ( 10 ** 6 )
        thread_ram = megapixels * 2.4
        sim_workers = round( available_ram / thread_ram )
        if sim_workers < 1:
            sim_workers = 1
        elif sim_workers > max_cpu_workers:
            sim_workers = max_cpu_workers
        
        # sim_workers = 1
        
        print ('---\nFree RAM: %s Gb available' % '{0:.1f}'.format(available_ram))
        print ('Image size: %s x %s' % ( w, h,))
        print ('Peak memory usage estimation: %s Gb per CPU thread ' % '{0:.1f}'.format(thread_ram))
        print ('Using %s CPU worker thread%s (of %s available)\n---' % (sim_workers, '' if sim_workers == 1 else 's', mp.cpu_count()))
        if thread_ram > available_ram:
            print ('Warning: estimated peak memory usage is greater then RAM avaliable')
        '''
        
        # print ('rendering %s frames to %s/' % (last_frame_number, args.output))
        _thread.start_new_thread(clear_write_buffer, (args, write_buffer, input_duration))


        active_workers = []

        # cpu_progress_updater = threading.Thread(target=cpu_progress_updater, args=(frames_written, last_frame_number, ))
        # cpu_progress_updater.daemon = True
        # cpu_progress_updater.start()

        rstep = 1 / ( input_duration + 1 )
        ratio = rstep

        last_thread_time = time.time()
        for frame in range(1, input_duration + 1):
            incoming_frame = incoming_read_buffer.get()
            outgoing_frame = outgoing_read_buffer.get()

            
            p = mp.Process(target=three_of_a_perfect_pair, args=(incoming_frame, outgoing_frame, frame, ratio, device, padding, model, args, h, w, write_buffer, rstep / 2, ))
            p.start()
            active_workers.append(p)

            ratio += rstep
            
            # try to shift threads in time to avoid memory congestion
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

        # wait for all active worker threads left to finish                
        for p in active_workers:
            p.join()

        # send write loop exit code
        write_buffer.put((-1, -1))

        # it should put IOThreadsFlag to False it return
        while(IOThreadsFlag):
            time.sleep(0.01)

    for p in IOProcesses:
        p.join(timeout=8)

    for p in IOProcesses:
        p.terminate()
        p.join(timeout=0)

    import hashlib
    lockfile = os.path.join('locks', hashlib.sha1(output_folder.encode()).hexdigest().upper() + '.lock')
    if os.path.isfile(lockfile):
        os.remove(lockfile)
    
    # input("Press Enter to continue...")
    sys.exit(0)


