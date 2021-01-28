from ntpath import basename
import os
import sys
from turtle import backward, forward
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
    # os.system(cmd)

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

def make_inference_rational(model, I0, I1, ratio, rthreshold=0.02, maxcycles=5, UHD=False, always_interp=False):
    I0_ratio = 0.0
    I1_ratio = 1.0
    rational_m = torch.mean(I0) * ratio + torch.mean(I1) * (1 - ratio)

    if not always_interp:
        if ratio <= I0_ratio + rthreshold / 2:
            return I0
        if ratio >= I1_ratio - rthreshold / 2:
            return I1

    for inference_cycle in range(0, maxcycles):
        middle = model.inference(I0, I1, UHD)
        middle_ratio = (I0_ratio + I1_ratio) / 2
        
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


def make_inference_rational_cpu(model, I0, I1, ratio, frame_num, w, h, write_buffer, rthreshold=0.02, maxcycles=8, UHD=False, always_interp=False):
    device = torch.device("cpu")

    I0_ratio = 0.0
    I1_ratio = 1.0
    rational_m = torch.mean(I0) * ratio + torch.mean(I1) * (1 - ratio)

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
        middle = model.inference(I0, I1, UHD)
        middle_ratio = (I0_ratio + I1_ratio) / 2
        
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

    # middle + (rational_m - torch.mean(middle)).expand_as(middle)
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

    with open(tw_setup_path, 'r') as tw_setup_file:
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
    
    # add point tangents from vecrors to match older version of setup
    # used by Julik's parser

    from xml.dom import minidom
    xml = minidom.parse(tw_setup_path)
    keys = xml.getElementsByTagName('Key')
    for key in keys:        
        frame = key.getElementsByTagName('Frame')
        if frame:
            frame = (frame[0].firstChild.nodeValue)
        value = key.getElementsByTagName('Value')
        if value:
            value = (value[0].firstChild.nodeValue)
        rdx = key.getElementsByTagName('RHandle_dX')
        if rdx:
            rdx = (rdx[0].firstChild.nodeValue)
        rdy = key.getElementsByTagName('RHandle_dY')
        if rdy:
            rdy = (rdy[0].firstChild.nodeValue)
        ldx = key.getElementsByTagName('LHandle_dX')
        if ldx:
            ldx = (ldx[0].firstChild.nodeValue)
        ldy = key.getElementsByTagName('LHandle_dY')
        if ldy:
            ldy = (ldy[0].firstChild.nodeValue)

        lx = xml.createElement('LHandleX')
        lx.appendChild(xml.createTextNode('{:.6f}'.format(float(frame) + float(ldx))))
        key.appendChild(lx)
        ly = xml.createElement('LHandleY')
        ly.appendChild(xml.createTextNode('{:.6f}'.format(float(value) + float(ldy))))
        key.appendChild(ly)
        rx = xml.createElement('RHandleX')
        rx.appendChild(xml.createTextNode('{:.6f}'.format(float(frame) + float(rdx))))
        key.appendChild(rx)
        ry = xml.createElement('RHandleY')
        ry.appendChild(xml.createTextNode('{:.6f}'.format(float(value) + float(rdy))))
        key.appendChild(ry)

    xml_string = xml.toxml()
    dirname, name = os.path.dirname(tw_setup_path), os.path.basename(tw_setup_path)
    xml_path = os.path.join(dirname, 'fix_' + name)
    with open(xml_path, 'a') as xml_file:
        xml_file.write(xml_string)
        xml_file.close()

    intp_start = start
    intp_end = end

    if TW_RetimerMode == 0:
        tw_speed = {}
        tw_speed_frames = []
        TW_Speed = xml.getElementsByTagName('TW_Speed')
        keys = TW_Speed[0].getElementsByTagName('Key')
        for key in keys:
            index = key.getAttribute('Index') 
            frame = key.getElementsByTagName('Frame')
            if frame:
                frame = (frame[0].firstChild.nodeValue)
            value = key.getElementsByTagName('Value')
            if value:
                value = (value[0].firstChild.nodeValue)
            tw_speed[int(index)] = {'frame': int(frame), 'value': float(value)}
            tw_speed_frames.append(int(frame))

            intp_start = min(start, min(tw_speed_frames))
            intp_end = max(end, max(tw_speed_frames))
    else:
        tw_timing = {}
        tw_timing_frames = []
        TW_Timing = xml.getElementsByTagName('TW_Timing')
        keys = TW_Timing[0].getElementsByTagName('Key')
        for key in keys:
            index = key.getAttribute('Index') 
            frame = key.getElementsByTagName('Frame')
            if frame:
                frame = (frame[0].firstChild.nodeValue)
            value = key.getElementsByTagName('Value')
            if value:
                value = (value[0].firstChild.nodeValue)
            tw_timing[int(index)] = {'frame': int(frame), 'value': float(value)}
            tw_timing_frames.append(int(frame))

            intp_start = min(start, min(tw_timing_frames))
            intp_end = max(end, max(tw_timing_frames))

    tw_channel_name = 'Speed' if TW_RetimerMode == 0 else 'Timing'

    cmd = parser_and_baker + ' -c ' + tw_channel_name
    cmd += ' -s ' + str(intp_start) + ' -e ' + str(intp_end)
    cmd += ' --to-file ' + parsed_and_baked_path + ' ' + xml_path
    os.system(cmd)

    if not os.path.isfile(parsed_and_baked_path):
        print ('can not find parsed channel file %s' % parsed_and_baked_path)
        input("Press Enter to continue...")
        sys.exit(1)

    tw_channel = {}
    with open(parsed_and_baked_path, 'r') as parsed_and_baked:
        import re
        
        # taken from Julik's parser

        CORRELATION_RECORD = re.compile(
        r"""
        ^([-]?\d+)            # -42 or 42
        \t                    # tab
        (
            [-]?(\d+(\.\d*)?) # "-1" or "1" or "1.0" or "1."
            |                 # or:
            \.\d+             # ".2"
        )
        ([eE][+-]?[0-9]+)?    # "1.2e3", "1.2e-3" or "1.2e+3"
        $
        """, re.VERBOSE)
    
        lines = parsed_and_baked.readlines()
        for i, line in enumerate(lines):
            line = line.rstrip()
            m = CORRELATION_RECORD.match(line)
            if m is not None:
                frame_number = int(m.group(1))
                value = float(m.group(2))
                tw_channel[frame_number] = value

    if TW_RetimerMode == 1:
        # job's done for 'Timing' channel
        return tw_channel

    else:
        # speed - based timewaro needs a bit more love
        # to solve frame values against speed channel
        # with the help of anchor frames in SpeedTiming channel

        tw_speed_timing = {}
        TW_SpeedTiming = xml.getElementsByTagName('TW_SpeedTiming')
        keys = TW_SpeedTiming[0].getElementsByTagName('Key')
        for key in keys:
            index = key.getAttribute('Index') 
            frame = key.getElementsByTagName('Frame')
            if frame:
                frame = (frame[0].firstChild.nodeValue)
            value = key.getElementsByTagName('Value')
            if value:
                value = (value[0].firstChild.nodeValue)
            tw_speed_timing[int(index)] = {'frame': int(frame), 'value': float(value)}
        
        if tw_speed_timing[0]['frame'] > start:
            # we need to extrapolate backwards from the first 
            # keyframe in SpeedTiming channel

            anchor_frame_value = tw_speed_timing[0]['value']
            for frame_number in range(tw_speed_timing[0]['frame'] - 1, start - 1, -1):
                if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                    step_back = tw_channel[min(list(tw_channel.keys()))] / 100
                else:
                    step_back = (tw_channel[frame_number + 1] + tw_channel[frame_number]) / 200
                frame_value_map[frame_number] = anchor_frame_value - step_back
                anchor_frame_value = frame_value_map[frame_number]

        # build up frame values between keyframes of SpeedTiming channel
        for key_frame_index in range(0, len(tw_speed_timing.keys()) - 1):
            # The value from my gess algo is close to the one in flame but not exact
            # and error is accumulated. SO quick and dirty way is to do forward
            # and backward pass and mix them rationally

            range_start = tw_speed_timing[key_frame_index]['frame']
            range_end = tw_speed_timing[key_frame_index + 1]['frame']
            
            if range_end == range_start + 1:
            # keyframes on next frames, no need to interpolate
                frame_value_map[range_start] = tw_speed_timing[key_frame_index]['value']
                frame_value_map[range_end] = tw_speed_timing[key_frame_index + 1]['value']
                continue

            forward_pass = {}
            anchor_frame_value = tw_speed_timing[key_frame_index]['value']
            forward_pass[range_start] = anchor_frame_value

            for frame_number in range(range_start + 1, range_end):
                if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                    step = tw_channel[max(list(tw_channel.keys()))] / 100
                else:
                    step = (tw_channel[frame_number] + tw_channel[frame_number + 1]) / 200
                forward_pass[frame_number] = anchor_frame_value + step
                anchor_frame_value = forward_pass[frame_number]
            forward_pass[range_end] = tw_speed_timing[key_frame_index + 1]['value']
            
            backward_pass = {}
            anchor_frame_value = tw_speed_timing[key_frame_index + 1]['value']
            backward_pass[range_end] = anchor_frame_value
            
            for frame_number in range(range_end - 1, range_start -1, -1):
                if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                    step_back = tw_channel[min(list(tw_channel.keys()))] / 100
                else:
                    step_back = (tw_channel[frame_number + 1] + tw_channel[frame_number]) / 200
                backward_pass[frame_number] = anchor_frame_value - step_back
                anchor_frame_value = backward_pass[frame_number]
            
            backward_pass[range_start] = tw_speed_timing[key_frame_index]['value']

            # create easy in and out soft mixing curve
            import numpy as np
            from scipy import interpolate
            
            ctr =np.array( [(0 , 0), (0.1, 0), (0.9, 1),  (1, 1)])
            x=ctr[:,0]
            y=ctr[:,1]
            interp = interpolate.CubicSpline(x, y)

            work_range = list(forward_pass.keys())
            ratio = 0
            rstep = 1 / len(work_range)
            for frame_number in sorted(work_range):
                frame_value_map[frame_number] = forward_pass[frame_number] * (1 - interp(ratio)) + backward_pass[frame_number] * interp(ratio)
                ratio += rstep
        
        last_key_index = list(sorted(tw_speed_timing.keys()))[-1]
        if tw_speed_timing[last_key_index]['frame'] < end:
            # we need to extrapolate further on from the 
            # last keyframe in SpeedTiming channel
            anchor_frame_value = tw_speed_timing[last_key_index]['value']
            frame_value_map[tw_speed_timing[last_key_index]['frame']] = anchor_frame_value

            for frame_number in range(tw_speed_timing[last_key_index]['frame'] + 1, end + 1):
                if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                    step = tw_channel[max(list(tw_channel.keys()))] / 100
                else:
                    step = (tw_channel[frame_number] + tw_channel[frame_number + 1]) / 200
                frame_value_map[frame_number] = anchor_frame_value + step
                anchor_frame_value = frame_value_map[frame_number]

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

    # input("Press Enter to continue...")
    # sys.exit(0)

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

        device = torch.device('cpu')
        torch.set_grad_enabled(False)

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