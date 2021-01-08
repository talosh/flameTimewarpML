#!/usr/bin/python
'''
Andriy's bunlde packer
'''

import os
import sys
import base64

pligin_dirname = os.path.dirname(os.path.abspath(__file__))
plugin_file_name = os.path.basename(pligin_dirname) + '.py'
python_source = os.path.join(pligin_dirname, plugin_file_name)
bundle_folder = 'bundle'
bundle_code = 'flameTimewarpML.py'

if not os.path.isdir(bundle_folder):
    print('no folder named %s' % bundle_folder)
    sys.exit()
    
print ('creating %s' % bundle_folder + '.tar\n---')
cmd = 'tar cvf ' + bundle_folder + '.tar ' + bundle_folder
os.system(cmd)
# cmd = bundle_folder + '/bin/pbzip2 -v ' + bundle_folder + '.tar'
# os.system(cmd)

f = open(bundle_folder + '.tar', 'rb')
bunle_data = f.read()
f.close()

encoded_bundle_data = base64.b64encode(bunle_data)

f = open(bundle_code, 'r')
bundle_code = f.read()
f.close()

bundled_code = bundle_code.replace('BUNDLE_PAYLOAD', encoded_bundle_data)

with open('flameTimewarpML.package/flameTimewarpML.py', 'w') as tempfile:
    with open(python_source, 'r') as src:
#       tempfile.write(src.read())
        tempfile.write(bundled_code)

# cmd = 'mv tmp.py /opt/Autodesk/shared/python/' + plugin_file_name
# os.system(cmd)

print ('---\nremoving %s' % bundle_folder + '.tar')
os.remove(bundle_folder + '.tar')