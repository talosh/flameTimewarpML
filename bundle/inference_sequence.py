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
import skvideo.io
from queue import Queue, Empty
warnings.filterwarnings("ignore")

from pprint import pprint
import time
import psutil

import multiprocessing as mp

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

def find_middle_frame(frames, frames_taken):
    for start_frame in range(1, len(frames.keys()) + 1):
        for frame_number in range (start_frame, len(frames.keys()) + 1):
            if frames.get(frame_number) and (not frames.get(frame_number + 1, True)):
                start_frame = frame_number
                break

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
        cv2.imwrite(os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(start_frame)), frame0[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        frames_written[ start_frame ] = start_frame_out_file_name

    end_frame_out_file_name = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(end_frame))
    if not os.path.isfile(end_frame_out_file_name):
        cv2.imwrite(os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(end_frame)), frame1[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        frames_written[ end_frame ] = end_frame_out_file_name

    frames[ middle_frame ] = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(middle_frame))
    frames_written[ middle_frame ] = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(middle_frame))

    return True

def progress_updater(frames_written, last_frame_number, ThreadsFlag):
    pbar = tqdm(total=last_frame_number, unit='frame')
    lastframe = 0
    while ThreadsFlag:
        while len(frames_written.keys()) < last_frame_number:
            pbar.n = len(frames_written.keys())
            pbar.last_print_n = len(frames_written.keys())
            if lastframe != len(frames_written.keys()):
                pbar.refresh()
                lastframe = len(frames_written.keys())
            time.sleep(0.01)
    
    pbar.n = last_frame_number
    pbar.last_print_n = last_frame_number
    pbar.refresh()
    pbar.close()

def read_cache(cache, ThreadsFlag):
    pass

def write_cache(cache, frames_written, ThreadsFlag):
    pass

if __name__ == '__main__':
    cpus = None
    ThreadsFlag = True
    print('initializing Timewarp ML...')

    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--input', dest='input', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--UHD', dest='UHD', action='store_true', help='flow size 1/4')
    parser.add_argument('--exp', dest='exp', type=int, default=1)
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='process only on CPU(s)')

    args = parser.parse_args()
    assert (not args.output is None or not args.input is None)

    if torch.cuda.is_available() and not args.cpu:
        frames = dict()
        frames_written = dict()
        frames_taken = dict()
    else:
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
    first_frame_number = 1
    step = (2 ** args.exp) -1
    last_frame_number = (input_duration - 1) * step + input_duration

    frame_number = first_frame_number
    for file_name in sorted(files_list):
        frames[frame_number] = os.path.join(args.input, file_name)
        frame_number += step + 1

    for frame_number in range(first_frame_number, last_frame_number):
        frames[frame_number] = frames.get(frame_number, '')

    first_frame = cv2.imread(frames.get(first_frame_number), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
    h, w, _ = first_frame.shape

    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)

    output_folder = os.path.abspath(args.output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    _thread.start_new_thread(progress_updater, (frames_written, last_frame_number, ThreadsFlag))

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')

        from model.RIFE_HD import Model
        model = Model(device = device)
        model.load_model('./train_log', -1)
        model.eval()

        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        cache = dict()
        for frame_number in sorted(frames.keys()):
            cache[frame_number] = {
                'path' : frames.get(frame_number),
                'frame_data' : None,
                'is_root_frame' : True if frames.get(frame_number) else False,
                'is_written' : False
            }
        
        pprint (cache)
        sys.exit()

        # chto za huinja?
        # test3

        _thread.start_new_thread(io_cache, (cache, frames, frames_written, last_frame_number, ThreadsFlag))
        _thread.start_new_thread(io_cache, (cache, frames, frames_written, last_frame_number, ThreadsFlag))

        while len(frames_written.keys()) != last_frame_number:
            three_of_a_perfect_pair(frames, device, padding, model, args, h, w, frames_written, frames_taken)

    else:
        device = torch.device('cpu')
        
        from model.RIFE_HD import Model
        model = Model(device = device)
        model.load_model('./train_log', -1)
        model.eval()

        max_cpu_workers = mp.cpu_count() - 2
        available_ram = psutil.virtual_memory()[1]/( 1024 ** 3 )
        megapixels = ( h * w ) / ( 10 ** 6 )
        thread_ram = megapixels * 1.98
        sim_workers = round( available_ram / thread_ram )
        if sim_workers < 1:
            sim_workers = 1
        elif sim_workers > max_cpu_workers:
            sim_workers = max_cpu_workers

        print ('---\nFree RAM: %s Gb avaliable' % '{0:.1f}'.format(available_ram))
        print ('Image size: %s x %s' % ( w, h,))
        print ('Peak memory usage estimation: %s Gb per CPU thread ' % '{0:.1f}'.format(thread_ram))
        print ('Using %s CPU worker thread%s (of %s avaliable)\n---' % (sim_workers, '' if sim_workers == 1 else 's', mp.cpu_count()))
        if thread_ram > available_ram:
            print ('Warning: estimated peak memory usage is greater then RAM avaliable')
    
        active_workers = []

        while len(frames_written.keys()) != last_frame_number:
            p = mp.Process(target=three_of_a_perfect_pair, args=(frames, device, padding, model, args, h, w, frames_written, frames_taken, ))
            p.start()
            active_workers.append(p)
            
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

    '''
    else:

        I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
        I1 = F.pad(I1, padding)
        frame = read_buffer.get()

        for nn in range(1, tot_frame+1):

            frame = read_buffer.get()
            if frame is None:
                break

            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I1 = F.pad(I1, padding)

            diff = (F.interpolate(I0, (16, 16), mode='bilinear', align_corners=False)
                - F.interpolate(I1, (16, 16), mode='bilinear', align_corners=False)).abs()
            
            if diff.mean() > 0.2:
                output = []
                for i in range((2 ** args.exp) - 1):
                    output.append(I0)
            else:
                output = make_inference(model, I0, I1, args.exp, args.UHD)
                
            write_buffer.put(lastframe)
            for mid in output:
                if sys.platform == 'darwin':
                    mid = (((mid[0]).cpu().detach().numpy().transpose(1, 2, 0)))
                else:
                    mid = (((mid[0]).cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(mid[:h, :w])

            # pbar.update(1)
            lastframe = frame

    write_buffer.put(lastframe)

    while(not write_buffer.empty()):
        time.sleep(0.1)

    # pbar.close()
    if not vid_out is None:
        vid_out.release()
    '''
    
    ThreadsFlag = False
    time.sleep(0.1)

    import hashlib
    lockfile = os.path.join('locks', hashlib.sha1(output_folder.encode()).hexdigest().upper() + '.lock')
    if os.path.isfile(lockfile):
        os.remove(lockfile)

    input("Press Enter to continue...")


