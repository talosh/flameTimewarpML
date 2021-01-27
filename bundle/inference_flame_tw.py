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

from pprint import pprint, pformat
import time
import psutil
import signal

import multiprocessing as mp

warnings.filterwarnings("ignore")

IOThreadsFlag = True
IOProcesses = []
cv2.setNumThreads(1)


# Exception handler
def exeption_handler(exctype, value, tb):
    import traceback

    locks = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'locks')
    cmd = 'rm -f ' + locks + '/*'
    os.system(cmd)

    pprint('%s in %s' % (value, exctype))
    pprint(traceback.format_exception(exctype, value, tb))
    sys.__excepthook__(exctype, value, tb)
    input("Press Enter to continue...")
sys.excepthook = exeption_handler


# ctrl+c handler
def signal_handler(sig, frame):
    global IOThreadsFlag
    IOThreadsFlag = False
    time.sleep(0.1)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def clear_write_buffer(args, write_buffer, output_duration):
    global IOThreadsFlag
    global IOProcesses

    print('rendering %s frames to %s' % (output_duration, args.output))
    pbar = tqdm(total=output_duration, unit='frame')

    while IOThreadsFlag:
        item = write_buffer.get()

        frame_number, image_data = item
        if frame_number == -1:
            pbar.close()    # type: ignore
            IOThreadsFlag = False
            break

        path = os.path.join(os.path.abspath(args.output), '{:0>7d}.exr'.format(frame_number))
        pprint ('recieved %s and sending to write' % output_frame_number)
        p = mp.Process(target=cv2.imwrite, args=(path, image_data[:, :, ::-1], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF], ))
        p.start()
        IOProcesses.append(p)
        pbar.update(1)


def build_read_buffer(user_args, read_buffer, videogen):
    global IOThreadsFlag

    for frame in videogen:
        frame_data = cv2.imread(os.path.join(user_args.input, frame), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        read_buffer.put(frame_data)
    read_buffer.put(None)


def make_inference_rational(model, I0, I1, ratio, rthreshold=0.02, maxcycles=8, UHD=False):
    I0_ratio = 0.0
    I1_ratio = 1.0

    if ratio <= I0_ratio + rthreshold / 2:
        return I0
    if ratio >= I1_ratio - rthreshold / 2:
        return I1

    for inference_cycle in range(0, maxcycles):
        middle = model.inference(I0, I1, UHD)
        middle_ratio = (I0_ratio + I1_ratio) / 2

        if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
            return middle

        if ratio > middle_ratio:
            I0 = middle
            I0_ratio = middle_ratio
        else:
            I1 = middle
            I1_ratio = middle_ratio

    return middle


def make_inference_rational_cpu(model, I0, I1, ratio, frame_num, w, h, write_buffer, rthreshold=0.02, maxcycles=8, UHD=False):
    device = torch.device("cpu")

    I0_ratio = 0.0
    I1_ratio = 1.0

    if ratio <= I0_ratio + rthreshold / 2:
        I0 = (((I0[0]).cpu().detach().numpy().transpose(1, 2, 0)))
        write_buffer.put((frame_num, I0[:h, :w]))
        return
    if ratio >= I1_ratio - rthreshold / 2:
        I1 = (((I1[0]).cpu().detach().numpy().transpose(1, 2, 0)))
        write_buffer.put((frame_num, I1[:h, :w]))
        return

    for inference_cycle in range(0, maxcycles):
        middle = model.inference(I0, I1, UHD)
        middle_ratio = (I0_ratio + I1_ratio) / 2

        if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
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

    middle = (((middle[0]).cpu().detach().numpy().transpose(1, 2, 0)))
    write_buffer.put((frame_num, middle[:h, :w]))
    return


def dictify(r, root=True):
    from copy import copy

    if root:
        return {r.tag: dictify(r, False)}

    d = copy(r.attrib)
    if r.text:
        d["_text"] = r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag] = []
        d[x.tag].append(dictify(x, False))
    return d


def bake_flame_tw_setup(tw_setup_path, start, end):
    # parses tw setup from flame and returns dictionary
    # with baked frame - value pairs
    
    def extrapolate_linear(xa, ya, xb, yb, xc):
        m = (ya - yb) / (xa - xb)
        yc = (xc - xb) * m + yb
        return yc

    import xml.etree.ElementTree as ET

    frame_value_map = {}

    with open(args.setup, 'r') as tw_setup_file:
        tw_setup_string = tw_setup_file.read()
        tw_setup_file.close()
        tw_setup_xml = ET.fromstring(tw_setup_string)
        tw_setup = dictify(tw_setup_xml)

    # start = int(tw_setup['Setup']['Base'][0]['Range'][0]['Start'])
    # end = int(tw_setup['Setup']['Base'][0]['Range'][0]['End'])
    # TW_Timing_size = int(tw_setup['Setup']['State'][0]['TW_Timing'][0]['Channel'][0]['Size'][0]['_text'])
    TW_SpeedTiming_size = int(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size'][0]['_text'])

    TW_RetimerMode = int(tw_setup['Setup']['State'][0]['TW_RetimerMode'][0]['_text'])
    parsed_and_baked_path = os.path.join(os.path.dirname(args.setup), 'parsed_and_baked.txt')
    parser_and_baker = os.path.join(os.path.dirname(__file__), 'flame_channel_parser', 'bin', 'bake_flame_channel')

    if TW_SpeedTiming_size == 1 and TW_RetimerMode == 0:
        # just constant speed change with no keyframes set       
        x = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['Frame'][0]['_text'])
        y = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['Value'][0]['_text'])
        ldx = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['LHandle_dX'][0]['_text'])
        ldy = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['LHandle_dY'][0]['_text'])
        rdx = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['RHandle_dX'][0]['_text'])
        rdy = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['RHandle_dY'][0]['_text'])

        for frame_number in range(start, end+1):
            frame_value_map[frame_number] = extrapolate_linear(x + ldx, y + ldy, x + rdx, y + rdy, frame_number)
    
        return frame_value_map
    
    elif TW_RetimerMode == 0:
        # Speed with multiple keyframes
        tw_channel_name = 'TW_SpeedTiming'

    else:
        # Timing with multiple keyframes
        tw_channel_name = 'TW_Timing'

    # tw_channel_name = 'TW_Timing' if TW_Timing_size > TW_SpeedTiming_size else 'TW_SpeedTiming'

    tw_keyframes = tw_setup['Setup']['State'][0][tw_channel_name][0]['Channel'][0]['KFrames'][0]['Key']
    for tw_keyframe in tw_keyframes:
        frame = int(tw_keyframe['Frame'][0]['_text'])
        value = float(tw_keyframe['Value'][0]['_text'])
        frame_value_map[frame] = value

    return frame_value_map

if __name__ == '__main__':
    start = time.time()

    msg = 'Timewarp using FX setup from Flame\n'
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('--input', dest='input', type=str, default=None, help='folder with input sequence')
    parser.add_argument('--output', dest='output', type=str, default=None, help='folder to output sequence to')
    parser.add_argument('--setup', dest='setup', type=str, default=None, help='flame tw setup to use')
    parser.add_argument('--record_in', dest='record_in', type=int, default=1, help='record in point relative to tw setup')
    parser.add_argument('--record_out', dest='record_out', type=int, default=0, help='record out point relative to tw setup')
    parser.add_argument('--model', dest='model', type=str, default='./trained_models/default/v1.8.model')
    parser.add_argument('--UHD', dest='UHD', action='store_true', help='flow size 1/4')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='do not use GPU at all, process only on CPU')

    args = parser.parse_args()
    if (args.output is None or args.input is None or args.setup is None):
         parser.print_help()
         sys.exit()

    print('Initializing TimewarpML from Flame setup...')

    img_formats = ['.exr',]
    src_files_list = []
    for f in os.listdir(args.input):
        name, ext = os.path.splitext(f)
        if ext in img_formats:
            src_files_list.append(f)

    input_duration = len(src_files_list)
    if not input_duration:
        print('not enough input frames: %s given' % input_duration)
        input("Press Enter to continue...")
        sys.exit()
    if not args.record_out:
        args.record_out = input_duration
    
    frame_value_map = bake_flame_tw_setup(args.setup, args.record_in, args.record_out)
        
    start_frame = 1
    src_files_list.sort()
    src_files = {x:os.path.join(args.input, file_path) for x, file_path in enumerate(src_files_list, start=start_frame)}

    output_folder = os.path.abspath(args.output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_duration = (args.record_out - args.record_in) + 1

    if torch.cuda.is_available() and not args.cpu:
        # Process on GPU

        from model.RIFE_HD import Model     # type: ignore
        model = Model()
        model.load_model(args.model, -1)
        model.eval()
        model.device()
        print ('Trained model loaded: %s' % args.model)

        write_buffer = Queue(maxsize=mp.cpu_count() - 3)
        _thread.start_new_thread(clear_write_buffer, (args, write_buffer, output_duration))
    
        src_start_frame = cv2.imread(src_files.get(start_frame), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        h, w, _ = src_start_frame.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
    
        device = torch.device("cuda")
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_frame_number = 1
        for frame_number in range(args.record_in, args.record_out +1):

            I0_frame_number = int(frame_value_map[frame_number])
            if I0_frame_number < 1:
                I0_image = cv2.imread(src_files.get(1), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
                write_buffer.put((output_frame_number, I0_image))

                output_frame_number += 1
                continue
            if I0_frame_number >= input_duration:
                I0_image = cv2.imread(src_files.get(input_duration), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
                write_buffer.put((output_frame_number, I0_image))
                output_frame_number += 1
                continue

            I1_frame_number = I0_frame_number + 1
            ratio = frame_value_map[frame_number] - int(frame_value_map[frame_number]) 

            # pprint ('frame_number: %s, value: %s' % (frame_number, frame_value_map[frame_number]))
            # pprint ('I0_frame_number: %s, I1_frame_number: %s, ratio: %s' % (I0_frame_number, I1_frame_number, ratio))
    
            I0_image = cv2.imread(src_files.get(I0_frame_number), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
            I1_image = cv2.imread(src_files.get(I1_frame_number), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        
            I0 = torch.from_numpy(np.transpose(I0_image, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I1 = torch.from_numpy(np.transpose(I1_image, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I0 = F.pad(I0, padding)
            I1 = F.pad(I1, padding)
            
            mid = make_inference_rational(model, I0, I1, ratio, UHD = args.UHD)
            mid = (((mid[0]).cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put((output_frame_number, mid[:h, :w]))

            output_frame_number += 1

        # send write loop exit code
        write_buffer.put((-1, -1))

        # it should put IOThreadsFlag to False it return
        while(IOThreadsFlag):
            time.sleep(0.01)
    
    else:
        # process on GPU

        from model_cpu.RIFE_HD import Model     # type: ignore
        model = Model()
        model.load_model(args.model, -1)
        model.eval()
        model.device()
        print ('Trained model loaded: %s' % args.model)

        src_start_frame = cv2.imread(src_files.get(start_frame), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        h, w, _ = src_start_frame.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)

        device = torch.device("cpu")

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

        write_buffer = mp.Queue(maxsize=mp.cpu_count() - 3)
        _thread.start_new_thread(clear_write_buffer, (args, write_buffer, input_duration))

        active_workers = []
        output_frame_number = 1
        last_thread_time = time.time()

        for frame_number in range(args.record_in, args.record_out +1):

            I0_frame_number = int(frame_value_map[frame_number])
            if I0_frame_number < 1:
                I0_image = cv2.imread(src_files.get(1), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
                write_buffer.put((output_frame_number, I0_image))

                output_frame_number += 1
                continue
            if I0_frame_number >= input_duration:
                I0_image = cv2.imread(src_files.get(input_duration), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
                write_buffer.put((output_frame_number, I0_image))
                output_frame_number += 1
                continue

            I1_frame_number = I0_frame_number + 1
            ratio = frame_value_map[frame_number] - int(frame_value_map[frame_number]) 

            # pprint ('frame_number: %s, value: %s' % (frame_number, frame_value_map[frame_number]))
            # pprint ('I0_frame_number: %s, I1_frame_number: %s, ratio: %s' % (I0_frame_number, I1_frame_number, ratio))
    
            I0_image = cv2.imread(src_files.get(I0_frame_number), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
            I1_image = cv2.imread(src_files.get(I1_frame_number), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
        
            I0 = torch.from_numpy(np.transpose(I0_image, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I1 = torch.from_numpy(np.transpose(I1_image, (2,0,1))).to(device, non_blocking=True).unsqueeze(0)
            I0 = F.pad(I0, padding)
            I1 = F.pad(I1, padding)
            
            pprint ('sending output number %s' % output_frame_number)
            p = mp.Process(target=make_inference_rational_cpu, args=(model, I0, I1, ratio, output_frame_number, w, h, write_buffer), kwargs = {'UHD': args.UHD})
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

            output_frame_number += 1

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