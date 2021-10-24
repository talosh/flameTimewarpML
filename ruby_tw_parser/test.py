from io import TextIOWrapper
import os
import sys
import numpy as np
from pprint import pprint

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

def decode_tw_setup(temp_setup_path):
    import xml.etree.ElementTree as ET
    import math
    
    # return value is a dict of int frames as keys
    frame_value_map = {}

    with open(temp_setup_path, 'r') as tw_setup_file:
        tw_setup_string = tw_setup_file.read()
        tw_setup_file.close()

    tw_setup_xml = ET.fromstring(tw_setup_string)
    tw_setup_dict = dictify(tw_setup_xml)

    start_frame = math.floor(float(tw_setup_dict['Setup']['Base'][0]['Range'][0]['Start']))
    end_frame = math.ceil(float(tw_setup_dict['Setup']['Base'][0]['Range'][0]['End']))
    TW_SpeedTiming_size = int(tw_setup_dict['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size'][0]['_text'])
    TW_RetimerMode = int(tw_setup_dict['Setup']['State'][0]['TW_RetimerMode'][0]['_text'])


    pprint ((start_frame, end_frame))
    return frame_value_map


temp_setup_path = sys.argv[1]
if not temp_setup_path:
    print ('no file to parse')
    sys.exit()

keys = decode_tw_setup(temp_setup_path)



HERMATRIX = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
print (HERMATRIX)
hermite_vector = np.array([9.0, 12.0, 18.78345865418239, -12.138369540337518])
print (hermite_vector)
hermite_basis = HERMATRIX.dot(hermite_vector)
print (hermite_basis)

start_frame = 1
end_frame = 8

for x in range(start_frame, end_frame+1):
    t = float(x - start_frame) / (end_frame - start_frame)
    print ("frame = %s" % x)
    print ("t = %s" % t)
    multipliers_vec = np.array([t ** 3,  t ** 2, t ** 1, t ** 0])
    print ("multipliers vec: %s" % multipliers_vec)
    sum = 0.0
    for i in range (0, 4):
        sum += hermite_basis[i] * multipliers_vec[i]
    print (sum)