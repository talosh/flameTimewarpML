#!/usr/bin/python3
'''
Andriy's bunlde packer
'''

import os
import sys
import base64
import argparse

def find_value(filename, variable_name):
    with open(filename, 'r') as file:
        for line in file.readlines():
            if line.startswith(variable_name + " = "):
                # split the line by " = ", then strip whitespace, quotes...
                value = line.split(" = ")[1].strip().strip("'\"")
                return value
    return "Variable not found"

parser = argparse.ArgumentParser(description='Interpolation for a sequence of exr images')
parser.add_argument('--platform', dest='platform', action='store', default='all', help='bundle for specific platform')
parser.add_argument('--copy', dest='copy', action='store_true', help='copy to /opt/Autodesk/shared/python')
parser.add_argument('--run', dest='run', action='store_true', help='run flame')
args = parser.parse_args()

flame_cmd = '/opt/Autodesk/flame_2023.2/bin/startApplication'
plugin_dirname = os.path.dirname(os.path.abspath(__file__))
plugin_file_name = os.path.basename(plugin_dirname) + '.py'
python_source = os.path.join(plugin_dirname, plugin_file_name)
bundle_folder_name = 'bundle'
bundle_code = 'flameTimewarpML.py'
bundle_folder = os.path.join(plugin_dirname, bundle_folder_name)
platform_folder = os.path.join(plugin_dirname, bundle_folder_name, 'site-packages', 'platform')
packages_folder = os.path.join(plugin_dirname, 'packages')

# dest_dir = os.path.join(plugin_dirname, 'flameTimewarpML.package')
# dest_file = os.path.join(dest_dir, bundle_code)

if not os.path.isdir(bundle_folder):
    print('no folder named %s' % bundle_folder)
    sys.exit()

platform_folders = [d for d in os.listdir(platform_folder) if os.path.isdir(os.path.join(platform_folder, d))]
if not platform_folders:
    print (f'no platformms defined in {platform_folder}')
    print ('nothing to bundle')
    sys.exit()

if args.platform == 'all':
    platforms = platform_folders
elif args.platform in platform_folders:
    platforms = [args.platform]
else:
    print ('unknown platform to create package: "%s"' % args.platform)
    print ('platforms: %s' % platform_folders)
    sys.exit()

version = find_value(
    os.path.join(plugin_dirname, bundle_code),
    '__version__'
)

for platform in platforms:
    package_folder = os.path.join(
        packages_folder,
        version,
        os.path.splitext(bundle_code)[0] + '.' + platform + '.package'
    )
    package_bundle_folder = os.path.join(
        packages_folder,
        version,
        os.path.splitext(bundle_code)[0] + '.' + platform + '.bundle'
    )

    os.makedirs(package_folder, exist_ok=True)
    #!/usr/bin/python3
'''
Andriy's bunlde packer
'''

import os
import sys
import base64
import argparse

def find_value(filename, variable_name):
    with open(filename, 'r') as file:
        for line in file.readlines():
            if line.startswith(variable_name + " = "):
                # split the line by " = ", then strip whitespace, quotes...
                value = line.split(" = ")[1].strip().strip("'\"")
                return value
    return "Variable not found"

parser = argparse.ArgumentParser(description='Interpolation for a sequence of exr images')
parser.add_argument('--platform', dest='platform', action='store', default='all', help='bundle for specific platform')
parser.add_argument('--c', dest='cleanup', action='store_true', help='clean up bundle folder')
parser.add_argument('--copy', dest='copy', action='store_true', help='copy to /opt/Autodesk/shared/python')
parser.add_argument('--run', dest='run', action='store_true', help='run flame')
args = parser.parse_args()

flame_cmd = '/opt/Autodesk/flame_2023.3/bin/startApplication'
plugin_dirname = os.path.dirname(os.path.abspath(__file__))
plugin_file_name = os.path.basename(plugin_dirname) + '.py'
python_source = os.path.join(plugin_dirname, plugin_file_name)
bundle_folder_name = 'bundle'
bundle_code_file_name = 'flameTimewarpML.py'
bundle_folder = os.path.join(plugin_dirname, bundle_folder_name)
platform_folder = os.path.join(plugin_dirname, bundle_folder_name, 'site-packages', 'platform')
packages_folder = os.path.join(plugin_dirname, 'packages')

# dest_dir = os.path.join(plugin_dirname, 'flameTimewarpML.package')
# dest_file = os.path.join(dest_dir, bundle_code)

if not os.path.isdir(bundle_folder):
    print('no folder named %s' % bundle_folder)
    sys.exit()

platform_folders = [d for d in os.listdir(platform_folder) if os.path.isdir(os.path.join(platform_folder, d))]
if args.platform == 'all':
    platforms = platform_folders
elif args.platform in platform_folders:
    platforms = [args.platform]
else:
    print ('unknown platform to create package: "%s"' % args.platform)
    print ('platforms: %s' % platform_folders)
    sys.exit()

version = find_value(
    os.path.join(plugin_dirname, bundle_code_file_name),
    '__version__'
)

for platform in platforms:
    print ('processing packages for %s' % platform)
    version_folder = os.path.join(
        packages_folder,
        version,
    )
    print ('version folder: %s' % version_folder)
    os.makedirs(version_folder, exist_ok=True)

    package_folder = os.path.join(
        version_folder,
        os.path.splitext(bundle_code_file_name)[0] + '.' + platform + '.package'
    )
    print ('package_folder: %s' % package_folder)
    os.makedirs(package_folder, exist_ok=True)

    package_bundle_folder = os.path.join(
        version_folder,
        os.path.splitext(bundle_code_file_name)[0] + '.' + platform + '.bundle'
    )

    if not os.path.isdir(package_bundle_folder):
        os.makedirs(package_bundle_folder, exist_ok=True)

        cmd = 'rsync -avh --exclude="site-packages/*" ' + bundle_folder + os.path.sep + ' ' + package_bundle_folder + os.path.sep
        os.system (cmd)
        cmd = 'rsync -avh ' + os.path.join(platform_folder, platform) + os.path.sep \
            + ' ' + os.path.join(package_bundle_folder, 'site-packages' + os.path.sep)
        os.system(cmd)

    tar_file_name = os.path.splitext(bundle_code_file_name)[0] + '.' + platform + '.bundle.tar.gz'
    tar_file_path = os.path.join(version_folder, tar_file_name)
    if not os.path.isfile(tar_file_path):
        print ('creating %s' % tar_file_path + '\n---')
        cmd = 'tar czvf ' + tar_file_path + ' -C ' + package_bundle_folder + '/ .'
        print ('executing: %s\n---' % cmd)
        os.system(cmd)

    dest_python_file = os.path.join(package_folder, bundle_code_file_name)

    print ('---\nadding data to python script %s' % dest_python_file)
    if os.path.isfile(dest_python_file):
        os.remove(dest_python_file)

    f = open(tar_file_path, 'rb')
    bundle_data = f.read()
    f.close()
    encoded_bundle_data = base64.b64encode(bundle_data).decode()
    del bundle_data
    f = open(bundle_code_file_name, 'r')
    bundle_code = f.read()
    f.close()
    bundled_code = bundle_code.replace('BUNDLE_PAYLOAD', encoded_bundle_data)
    del encoded_bundle_data
    with open(dest_python_file, 'w') as tempfile:
        tempfile.write(bundled_code)
        tempfile.close()
    del bundled_code

    readme = os.path.join(plugin_dirname, 'README.md')
    cmd = 'cp ' + readme + ' ' + package_bundle_folder + os.path.sep
    os.system(cmd)

    if args.cleanup:
        print ('cleaning up %s' % tar_file_path)
        cmd = 'rm -rf ' + tar_file_path
        os.system(cmd)

        print ('cleaning up %s' % package_bundle_folder)
        cmd = 'rm -rf ' + package_bundle_folder
        os.system(cmd)

if args.copy:
    cmd = 'cp ' + dest_python_file + ' /opt/Autodesk/shared/python/'
    print ('copying %s to /opt/Autodesk/shared/python/' % dest_python_file)
    os.system(cmd)

if args.run:
    os.system(flame_cmd)

sys.exit()