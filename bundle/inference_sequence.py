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


ThreadsFlag = True
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
    global ThreadsFlag
    ThreadsFlag = False
    time.sleep(0.1)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def clear_write_buffer(user_args, write_buffer, tot_frame):
    global ThreadsFlag
    global IOProcesses

    new_frames_number = ((tot_frame - 1) * ((2 ** args.exp) -1)) + tot_frame
    cnt = 0
    while ThreadsFlag:
        item = write_buffer.get()
        
        if cnt == 0:
            print ('rendering %s frames to %s' % (new_frames_number, args.output))
            pbar = tqdm(total=new_frames_number, unit='frame')
        
        if item is None:
            pbar.close() # type: ignore
            break
        if cnt < new_frames_number:
            path = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(cnt))
            
            try:
                p = mp.Process(target=cv2.imwrite, args=(path, item[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF], ))
                p.start()
                IOProcesses.append(p)
            except:
                try:
                    cv2.imwrite(path, item[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                except Exception as e:
                    print ('Error wtiring %s: %s' % (path, e))
        
        pbar.update(1) # type: ignore
        cnt += 1

def build_read_buffer(user_args, read_buffer, videogen):
    global ThreadsFlag

    for frame in videogen:
        frame_data = cv2.imread(os.path.join(user_args.input, frame), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        read_buffer.put(frame_data)
    read_buffer.put(None)

def make_inference(model, I0, I1, exp, UHD):
    middle = model.inference(I0, I1, UHD)
    if exp == 1:
        return [middle]
    first_half = make_inference(model, I0, middle, exp=exp - 1, UHD=UHD)
    second_half = make_inference(model, middle, I1, exp=exp - 1, UHD=UHD)
    return [*first_half, middle, *second_half]

def find_middle_frame(frames, frames_taken):
    for start_frame in range(1, len(frames.keys()) + 1):
        for frame_number in range (start_frame, len(frames.keys()) + 1):
            if frames.get(frame_number) and (not frames.get(frame_number + 1, True)):
                start_frame = frame_number
                break
        
        end_frame = start_frame + 1
        for frame_number in range(start_frame + 1, len(frames.keys()) + 1):
            if frames.get(frame_number):
                end_frame = frame_number
                break
            end_frame = frame_number
        
        middle_frame = start_frame + int((end_frame - start_frame) / 2)

        if frames.get(start_frame) and not frames.get(middle_frame):
            if middle_frame in frames_taken.keys():
                # this frame is taken by another worker
                continue
            else:
                # mark frame as taken
                frames_taken[middle_frame] = 'taken between %s and %s' % (start_frame, end_frame)

                #print ('s: %s m: %s e: %s' % (start_frame, middle_frame, end_frame))
                #print ('%s: %s' % ( start_frame, frames.get(start_frame) ))
                #print ('%s: %s' % ( middle_frame, frames.get(middle_frame) ))
                #print ('%s: %s' % ( end_frame, frames.get(end_frame) ))

                return (start_frame, middle_frame, end_frame)
    return False

def three_of_a_perfect_pair(frames, device, padding, model, args, h, w, frames_written, frames_taken):
    perfect_pair = find_middle_frame(frames, frames_taken)

    if not perfect_pair:
        # print ('no more frames left')
        return False

    start_frame = perfect_pair[0]
    middle_frame = perfect_pair[1]
    end_frame = perfect_pair[2]

    frame0 = cv2.imread(frames[start_frame], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
    frame1 = cv2.imread(frames[end_frame], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
    
    I0 = torch.from_numpy(np.transpose(frame0, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
    I1 = torch.from_numpy(np.transpose(frame1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)

    I0 = F.pad(I0, padding)
    I1 = F.pad(I1, padding)

    diff = (F.interpolate(I0, (16, 16), mode='bilinear', align_corners=False)
        - F.interpolate(I1, (16, 16), mode='bilinear', align_corners=False)).abs()
    
    mid = model.inference(I0, I1, args.UHD)
    mid = (((mid[0]).cpu().detach().numpy().transpose(1, 2, 0)))
    midframe = mid[:h, :w]
    cv2.imwrite(os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(middle_frame)), midframe[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])

    start_frame_out_file_name = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(start_frame))
    if not os.path.isfile(start_frame_out_file_name):
        cv2.imwrite(start_frame_out_file_name, frame0[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        frames_written[ start_frame ] = start_frame_out_file_name

    end_frame_out_file_name = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(end_frame))
    if not os.path.isfile(end_frame_out_file_name):
        cv2.imwrite(end_frame_out_file_name, frame1[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        frames_written[ end_frame ] = end_frame_out_file_name

    frames[ middle_frame ] = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(middle_frame))
    frames_written[ middle_frame ] = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(middle_frame))

    return True

def cpu_progress_updater(frames_written, last_frame_number):
    global ThreadsFlag

    pbar = tqdm(total=last_frame_number, unit='frame')
    lastframe = 0
    while ThreadsFlag:
        pbar.n = len(frames_written.keys())
        pbar.last_print_n = len(frames_written.keys())
        if lastframe != len(frames_written.keys()):
            pbar.refresh()
            lastframe = len(frames_written.keys())
        time.sleep(0.01)
    pbar.close()

if __name__ == '__main__':
    start = time.time()
    print('initializing Timewarp ML...')

    parser = argparse.ArgumentParser(description='Interpolation for a sequence of exr images')
    parser.add_argument('--input', dest='input', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--model', dest='model', type=str, default='./trained_models/default/v2.0.model')
    parser.add_argument('--UHD', dest='UHD', action='store_true', help='flow size 1/4')
    parser.add_argument('--exp', dest='exp', type=int, default=1)
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='process only on CPU(s)')


    args = parser.parse_args()
    assert (not args.output is None or not args.input is None)

    manager = mp.Manager()
    frames = manager.dict()
    frames_written = manager.dict()
    frames_taken = manager.dict()

    img_formats = ['.exr',]
    files_list = []
    for f in os.listdir(args.input):
        name, ext = os.path.splitext(f)
        if ext in img_formats:
            files_list.append(f)

    input_duration = len(files_list)
    if input_duration < 2:
        print('not enough frames to perform slow motion: %s given' % input_duration)
        input("Press Enter to continue...")
        sys.exit()

    first_frame_number = 1
    step = (2 ** args.exp) -1
    last_frame_number = (input_duration - 1) * step + input_duration

    frame_number = first_frame_number
    
    for file_name in sorted(files_list):
        frames[frame_number] = os.path.join(args.input, file_name)
        frame_number += step + 1

    for frame_number in range(first_frame_number, last_frame_number):
        frames[frame_number] = frames.get(frame_number, '')

    first_image = cv2.imread(frames.get(first_frame_number), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
    h, w, _ = first_image.shape

    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)

    output_folder = os.path.abspath(args.output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if torch.cuda.is_available() and not args.cpu:
        # process on GPU

        files_list.sort()
        files_list.append(files_list[-1])

        write_buffer = Queue(maxsize=mp.cpu_count() - 3)
        read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(build_read_buffer, (args, read_buffer, files_list))
        _thread.start_new_thread(clear_write_buffer, (args, write_buffer, input_duration))

        if 'v1.8.model' in args.model:
            from model.RIFE_HD import Model     # type: ignore
        else:
            from model.RIFE_HDv2 import Model     # type: ignore
        model = Model()
        model.load_model(args.model, -1)
        model.eval()
        model.device()
        print ('Trained model loaded: %s' % args.model)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_grad_enabled(False)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # print ('Loading initial frames...')
        lastframe = first_image
        I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
        I1 = F.pad(I1, padding)
        frame = read_buffer.get()

        for nn in range(1, input_duration+1):

            frame = read_buffer.get()
            if frame is None:
                break
        
            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I1 = F.pad(I1, padding)

            output = make_inference(model, I0, I1, args.exp, args.UHD)
            write_buffer.put(lastframe)
            for mid in output:
                if sys.platform == 'darwin':
                    mid = (((mid[0]).cpu().detach().numpy().transpose(1, 2, 0)))
                else:
                    mid = (((mid[0]).cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(mid[:h, :w])
            lastframe = frame
        
        write_buffer.put(lastframe)
        while(not write_buffer.empty()):
            time.sleep(0.1)

    else:
        # process on CPU(s)
        if 'v1.8.model' in args.model:
            from model_cpu.RIFE_HD import Model     # type: ignore
        else:
            from model_cpu.RIFE_HDv2 import Model     # type: ignore
        model = Model()
        model.load_model(args.model, -1)
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

        print ('---\nFree RAM: %s Gb available' % '{0:.1f}'.format(available_ram))
        print ('Image size: %s x %s' % ( w, h,))
        print ('Peak memory usage estimation: %s Gb per CPU thread ' % '{0:.1f}'.format(thread_ram))
        print ('Using %s CPU worker thread%s (of %s available)\n---' % (sim_workers, '' if sim_workers == 1 else 's', mp.cpu_count()))
        if thread_ram > available_ram:
            print ('Warning: estimated peak memory usage is greater then RAM avaliable')
        '''
        
        print ('rendering %s frames to %s/' % (last_frame_number, args.output))

        active_workers = []

        cpu_progress_updater = threading.Thread(target=cpu_progress_updater, args=(frames_written, last_frame_number, ))
        cpu_progress_updater.daemon = True
        cpu_progress_updater.start()

        last_thread_time = time.time()
        while len(frames_written.keys()) != last_frame_number:
            p = mp.Process(target=three_of_a_perfect_pair, args=(frames, device, padding, model, args, h, w, frames_written, frames_taken, ))
            p.start()
            active_workers.append(p)
            
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

        while len(active_workers):
            finished_workers = []
            alive_workers = []
            for worker in active_workers:
                if not worker.is_alive():
                    finished_workers.append(worker)
                else:
                    alive_workers.append(worker)
            active_workers = list(alive_workers)
            time.sleep(0.01)
        
        ThreadsFlag = False
        cpu_progress_updater.join()
    
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
    sys.exit()


