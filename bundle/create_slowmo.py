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

def worker(num):
    print ('Worker:', num)
    return

def clear_write_buffer(user_args, write_buffer, tot_frame):
    new_frames_number = ((tot_frame - 1) * ((2 ** args.exp) -1)) + tot_frame
    print ('rendering %s frames to %s/' % (new_frames_number, args.output))
    pbar = tqdm(total=new_frames_number, unit='frame')
    cnt = 0
    while ThreadsFlag:
        item = write_buffer.get()

        if item is None:
            pbar.close()
            break
        
        if cnt < new_frames_number:
            cv2.imwrite(os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(cnt)), item[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        pbar.update(1)
        cnt += 1

def build_read_buffer(user_args, read_buffer, videogen):
    for frame in videogen:
        if not user_args.img is None:
            if frame.endswith('.exr'):
                frame_data = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        read_buffer.put(frame_data)
    read_buffer.put(None)

def make_inference(model, I0, I1, exp, UHD):
    middle = model.inference(I0, I1, UHD)
    if exp == 1:
        return [middle]
    first_half = make_inference(model, I0, middle, exp=exp - 1, UHD=UHD)
    second_half = make_inference(model, middle, I1, exp=exp - 1, UHD=UHD)
    return [*first_half, middle, *second_half]

def three_of_a_perfect_pair(frame0, frame1, index, mp_output, device, padding, model, args, h, w):
    local_output = []

    I0 = torch.from_numpy(np.transpose(frame0, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
    I1 = torch.from_numpy(np.transpose(frame1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
    I0 = F.pad(I0, padding)
    I1 = F.pad(I1, padding)

    diff = (F.interpolate(I0, (16, 16), mode='bilinear', align_corners=False)
        - F.interpolate(I1, (16, 16), mode='bilinear', align_corners=False)).abs()

    mp_output[index] = frame0
    # cut detection - we probably don't need it here
    # shall try to interpolate anyway
    if diff.mean() > 0.2:
        for i in range((2 ** args.exp) - 1):
            mp_output[index + 1 + i] = I0
    else:
        local_output = make_inference(model, I0, I1, args.exp, args.UHD)
        i = 0
        for mid in local_output:
            mid = (((mid[0]).cpu().detach().numpy().transpose(1, 2, 0)))
            mp_output[index + 1 + i] = mid[:h, :w]
            i += 1
    return

if __name__ == '__main__':
    cpus = None
    ThreadsFlag = True
    print('initializing Timewarp ML...')

    from model.RIFE_HD import Model
    model = Model()
    model.load_model('./train_log', -1)
    model.eval()
    model.device()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        cpus = mp.cpu_count()
        cpus = int(cpus/2)
        print ('no cuda is available, using %s cpu workers instead' % cpus)

    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--img', dest='img', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
    parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
    parser.add_argument('--fps', dest='fps', type=int, default=None)
    parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
    parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
    parser.add_argument('--exp', dest='exp', type=int, default=1)

    args = parser.parse_args()
    assert (not args.video is None or not args.img is None)
    if not args.img is None:
        args.png = True

    videogen = []
    exrs = []
    for f in os.listdir(args.img):
        if f.endswith('.png'):
            videogen.append(f)
        if f.endswith('.exr'):
            exrs.append(f)
    
    # prefer exr sequences if detected
    if exrs:
        videogen = exrs
    tot_frame = len(videogen)
    videogen.sort()
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
    videogen.append(videogen[-1])
    h, w, _ = lastframe.shape
    vid_out = None
    if args.png:
        output_folder = os.path.abspath(args.output)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)

    skip_frame = 1

    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer, tot_frame))

    if cpus:
        manager = mp.Manager()
        mp_output = manager.dict()

        lastframe = read_buffer.get()
        for nn in range(1, tot_frame+1, cpus):
            frames = []
            frames.append(lastframe)
            for frame_count in range(cpus):
                frame = read_buffer.get()
                if frame is None:
                    break
                frames.append(frame)
            lastframe = frame

            workers = []
            index = 0
            for frame_index in range(len(frames)-1):
                p = mp.Process(target=three_of_a_perfect_pair, args=(frames[frame_index], frames[frame_index+1], index, mp_output, device, padding, model, args, h, w, ))
                p.start()
                workers.append(p)
                index += (2 ** args.exp) + 1
            
            for worker in workers:
                worker.join()

            for local_frame_index in sorted(mp_output.keys()):
                write_buffer.put(mp_output[local_frame_index])

        write_buffer.put(frames[-1])

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
            
            # if diff.max() < 2e-3 and args.skip:
            #    if skip_frame % 100 == 0:
            #        print("Warning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
            #    skip_frame += 1
            #    pbar.update(1)
            #    continue

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

    import hashlib
    lockfile = os.path.join('locks', hashlib.sha1(output_folder.encode()).hexdigest().upper() + '.lock')
    if os.path.isfile(lockfile):
        os.remove(lockfile)

    # input("Press Enter to continue...")


