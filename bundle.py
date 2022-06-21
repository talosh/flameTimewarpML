#!/usr/bin/python
'''
Andriy's bunlde packer
'''

import os
import sys
import base64
import argparse

parser = argparse.ArgumentParser(description='Interpolation for a sequence of exr images')
parser.add_argument('--copy', dest='copy', action='store_true', help='copy to /opt/Autodesk/shared/python')
parser.add_argument('--run', dest='run', action='store_true', help='run flame')
args = parser.parse_args()

flame_cmd = '/opt/Autodesk/flame_2022.1/bin/startApplication'
plugin_dirname = os.path.dirname(os.path.abspath(__file__))
plugin_file_name = os.path.basename(plugin_dirname) + '.py'
python_source = os.path.join(plugin_dirname, plugin_file_name)
bundle_folder = 'bundle'
bundle_code = 'flameTimewarpML.py'
dest_dir = os.path.join(plugin_dirname, 'flameTimewarpML.package')
dest_file = os.path.join(dest_dir, bundle_code)

if not os.path.isdir(bundle_folder):
    print('no folder named %s' % bundle_folder)
    sys.exit()
    
tarfile_name = os.path.splitext(bundle_code)[0] + '_bundle.tar'
tarfile_path = os.path.join(dest_dir,tarfile_name)

print ('creating %s' % tarfile_path + '.tar\n---')
cmd = 'tar cvf ' + tarfile_path + ' ' + bundle_folder + '/'
print ('executing: %s\n---' % cmd)
os.system(cmd)
# cmd = bundle_folder + '/bin/pbzip2 -v ' + bundle_folder + '.tar'
# os.system(cmd)

'''

print ('---\nadding data to python script %s' % python_source)

f = open(bundle_folder + '.tar', 'rb')
bundle_data = f.read()
f.close()

encoded_bundle_data = base64.b64encode(bundle_data)
del bundle_data

f = open(bundle_code, 'r')
bundle_code = f.read()
f.close()

bundled_code = bundle_code.replace('BUNDLE_PAYLOAD', encoded_bundle_data)
del encoded_bundle_data

if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

with open(dest_file, 'w') as tempfile:
    with open(python_source, 'r') as src:
#       tempfile.write(src.read())
        tempfile.write(bundled_code)

del bundled_code

# cmd = 'mv tmp.py /opt/Autodesk/shared/python/' + plugin_file_name
# os.system(cmd)


print ('---\nremoving %s' % bundle_folder + '.tar')
os.remove(bundle_folder + '.tar')

readme = os.path.join(plugin_dirname, 'README.md')
cmd = 'cp ' + readme + ' ' + dest_dir
os.system(cmd)

if args.copy:
    cmd = 'cp flameTimewarpML.package/flameTimewarpML.py /opt/Autodesk/shared/python/'
    print ('copying flameTimewarpML.package/flameTimewarpML.py to /opt/Autodesk/shared/python/')
    os.system(cmd)

if args.run:
    os.system(flame_cmd)
'''