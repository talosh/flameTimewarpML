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

from pprint import pprint, pformat
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
        pbar.n = len(frames_written.keys())
        pbar.last_print_n = len(frames_written.keys())
        if lastframe != len(frames_written.keys()):
            pbar.refresh()
            lastframe = len(frames_written.keys())
        time.sleep(0.1)
    pbar.close()
    
def read_cache(cache, read_ahead, frames_written, ThreadsFlag):
    root_frames = []
    for frame_number in sorted(cache.keys()):
        frame_data = cache.get(frame_number)
        if frame_data.get('is_root_frame'):
            root_frames.append(frame_number)

    current_rf_index = 0
    while ThreadsFlag:
        for index in range(current_rf_index, current_rf_index + read_ahead):
            frame_data = cache.get(root_frames[ index ])
            if frame_data:
                if type(frame_data.get('image')) == type(None):
                    frame_data['image'] = cv2.imread(frame_data.get('path'), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
                    frame_data['ready'] = True
            cache[root_frames[ index ]] = frame_data
        
        written_frames = []
        for frame_number in range(root_frames[current_rf_index], root_frames[current_rf_index + 1]):
            frame_data = cache.get(frame_number)
            if frame_data:
                written_frames.append(frame_data.get('is_written'))
                #if frame_data.get('is_written') and frame_number != root_frames[current_rf_index + 1]:
                #    frames_written[frame_number] = True

        if len(set(written_frames)) == 1 and list(set(written_frames))[0]:
            current_rf_index += 1
            if current_rf_index > len(root_frames) - read_ahead:
                current_rf_index = len(root_frames) - read_ahead

def write_cache(cache, frames_written, args, ThreadsFlag):
    root_frames = []
    for frame_number in sorted(cache.keys()):
        frame_data = cache.get(frame_number)
        if frame_data.get('is_root_frame'):
            root_frames.append(frame_number)

    while ThreadsFlag:
        for frame_number in sorted(cache.keys()):
            frame_data = cache.get(frame_number)
            if frame_data:
                if frame_data.get('ready') and not frame_data.get('is_written'):
                    path = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(frame_number))
                    image = frame_data.get('image')
                    cv2.imwrite(path, image[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                    frame_data['path'] = path
                    frame_data['is_written'] = True
                    cache[frame_number] = frame_data
                    if not frame_data.get('is_root_frame'):
                        frames_written[frame_number] = path
                        for x in sorted(root_frames):
                            lower_root_frame = x
                            if frame_number > x:
                                break
                        for y in sorted(root_frames):
                            if frame_number < y:
                                upper_root_frame = y
                                break
                        if cache.get(lower_root_frame).get('is_written'):
                            frames_written[lower_root_frame] = cache.get(lower_root_frame).get('path')
                        if cache.get(lower_root_frame).get('is_written'):
                            frames_written[upper_root_frame] = cache.get(upper_root_frame).get('path')
                        
        for index in range(0, len(root_frames) - 1):
            written_frames = []
            for frame_number in range (root_frames[ index ], root_frames[ index +1 ]):
                frame_data = cache.get(frame_number)
                if frame_data:
                    written_frames.append(frame_data.get('is_written'))
            
            if len(set(written_frames)) == 1 and list(set(written_frames))[0]:
                for frame_number in range (root_frames[ index ], root_frames[ index +1 ]):
                    frame_data = cache.get(frame_number)
                    frame_data['image'] = None
                    cache[frame_number] = frame_data

def print_cache_size(cache, ThreadsFlag):
    while ThreadsFlag:
        cache_size = 0
        img_count = 0
        for key in cache.keys():
            if type(cache.get(key).get('image')) != type(None):
                img_count += 1
                img_size = sys.getsizeof(cache.get(key).get('image'))
        print ('size: %s, count: %s' % ((img_size*img_count) / (1024 ** 2), img_count))
        time.sleep(1)

if __name__ == '__main__':
    start = time.time()
    cpus = None
    ThreadsFlag = True
    print('initializing Timewarp ML...')

    from model.RIFE_HD import Model
    model = Model()
    model.load_model('./train_log', -1)
    model.eval()
    model.device()

    pprint (time.time() - start)


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
        # Process on GPU

        cache = dict()
        for frame_number in sorted(frames.keys()):
            cache[frame_number] = {
                'path' : frames.get(frame_number),
                'image' : None,
                'is_root_frame' : True if frames.get(frame_number) else False,
                'in_progres' : False,
                'ready' : False,
                'is_written' : False
            }

        cache[1]['image'] = first_image
        cache[1]['ready'] = True
        
        read_ahead = 4 # number of root frames to read and keep in memory, inbetween frames to be added
        if read_ahead > input_duration:
            read_ahead = input_duration
        
        # _thread.start_new_thread(print_cache_size, (cache, ThreadsFlag))
        _thread.start_new_thread(read_cache, (cache, read_ahead, frames_written, ThreadsFlag))
        _thread.start_new_thread(write_cache, (cache, frames_written, args, ThreadsFlag))

        device = torch.device('cuda')
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # print (str(time.time() - start)) 
        # pprint (len(frames_written.keys()))
        print ('Rendering frames:')
        #_thread.start_new_thread(progress_updater, (frames_written, last_frame_number, ThreadsFlag))

        pprint (time.time()-start)

        while len(frames_written.keys()) != last_frame_number:
            perfect_pair = find_middle_frame(frames, frames_taken)
            if not perfect_pair:
                continue

            start_frame_data = cache.get(perfect_pair[0])
            start_image = start_frame_data.get('image')
            if type(start_image) == type(None):
                print('reading frame %s in main thread as start frame' % perfect_pair[0])
                start_image = cv2.imread(start_frame_data.get('path'), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()

            end_frame_data = cache.get(perfect_pair[2])
            end_image = end_frame_data.get('image')
            if type(end_image) == type(None):
                print('reading frame %s in main thread as end frame' % perfect_pair[2])
                end_image = cv2.imread(end_frame_data.get('path'), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()

            I0 = torch.from_numpy(np.transpose(start_image, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I1 = torch.from_numpy(np.transpose(end_image, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I0 = F.pad(I0, padding)
            I1 = F.pad(I1, padding)

            mid_frame_data = cache.get(perfect_pair[1])
            start = time.time()
            mid = model.inference(I0, I1, args.UHD)
            pprint (time.time() - start)
            mid = (((mid[0]).cpu().detach().numpy().transpose(1, 2, 0)))
            mid_frame_data['image'] = mid[:h, :w]
            mid_frame_data['ready'] = True
            cache[perfect_pair[1]] = mid_frame_data
            frames[perfect_pair[1]] = True

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

    pprint (time.time()-start)

    # input("Press Enter to continue...")


