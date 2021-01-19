# type: ignore
'''
flameTimewarpML
Flame 2020 and higher
written by Andrii Toloshnyy
andriy.toloshnyy@gmail.com
'''

import os
import sys
import time
import threading
import atexit
from pprint import pprint
from pprint import pformat

menu_group_name = 'Timewarp ML'
DEBUG = False

__version__ = 'v0.2.0.beta.019'


class flameAppFramework(object):
    # flameAppFramework class takes care of preferences and bundle unpack/install routines

    class prefs_dict(dict):
        # subclass of a dict() in order to directly link it 
        # to main framework prefs dictionaries
        # when accessed directly it will operate on a dictionary under a 'name'
        # key in master dictionary.
        # master = {}
        # p = prefs(master, 'app_name')
        # p['key'] = 'value'
        # master - {'app_name': {'key', 'value'}}
            
        def __init__(self, master, name, **kwargs):
            self.name = name
            self.master = master
            if not self.master.get(self.name):
                self.master[self.name] = {}
            self.master[self.name].__init__()

        def __getitem__(self, k):
            return self.master[self.name].__getitem__(k)
        
        def __setitem__(self, k, v):
            return self.master[self.name].__setitem__(k, v)

        def __delitem__(self, k):
            return self.master[self.name].__delitem__(k)
        
        def get(self, k, default=None):
            return self.master[self.name].get(k, default)
        
        def setdefault(self, k, default=None):
            return self.master[self.name].setdefault(k, default)

        def pop(self, k, v=object()):
            if v is object():
                return self.master[self.name].pop(k)
            return self.master[self.name].pop(k, v)
        
        def update(self, mapping=(), **kwargs):
            self.master[self.name].update(mapping, **kwargs)
        
        def __contains__(self, k):
            return self.master[self.name].__contains__(k)

        def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
            return type(self)(self)
        
        def keys(self):
            return self.master[self.name].keys()

        @classmethod
        def fromkeys(cls, keys, v=None):
            return self.master[self.name].fromkeys(keys, v)
        
        def __repr__(self):
            return '{0}({1})'.format(type(self).__name__, self.master[self.name].__repr__())

        def master_keys(self):
            return self.master.keys()

    def __init__(self):
        self.name = self.__class__.__name__
        self.bundle_name = 'flameTimewarpML'

        # self.prefs scope is limited to flame project and user
        self.prefs = {}
        self.prefs_user = {}
        self.prefs_global = {}
        self.debug = DEBUG
        
        try:
            import flame
            self.flame = flame
            self.flame_project_name = self.flame.project.current_project.name
            self.flame_user_name = flame.users.current_user.name
        except:
            self.flame = None
            self.flame_project_name = None
            self.flame_user_name = None
        
        import socket
        self.hostname = socket.gethostname()
        
        if sys.platform == 'darwin':
            self.prefs_folder = os.path.join(
                os.path.expanduser('~'),
                 'Library',
                 'Caches',
                 self.bundle_name)
        elif sys.platform.startswith('linux'):
            self.prefs_folder = os.path.join(
                os.path.expanduser('~'),
                '.config',
                self.bundle_name)

        self.prefs_folder = os.path.join(
            self.prefs_folder,
            self.hostname,
        )

        self.log('[%s] waking up' % self.__class__.__name__)
        self.load_prefs()

        # preferences defaults

        # if not self.prefs_global.get('bundle_location'):
        if sys.platform == 'darwin':
            self.bundle_location = os.path.join(
                os.path.expanduser('~'),
                'Documents',
                self.bundle_name)
        else:
            self.bundle_location = os.path.join(
                os.path.expanduser('~'),
                self.bundle_name)
        # else:
        #    self.bundle_location = self.prefs_global.get('bundle_location')

        #    self.prefs_global['menu_auto_refresh'] = {
        #        'media_panel': True,
        #        'batch': True,
        #        'main_menu': True
        #    }
        
        self.apps = []

        import hashlib
        self.bundle_id = hashlib.sha1(__version__.encode()).hexdigest()

        bundle_path = os.path.join(self.bundle_location, 'bundle')
        if (os.path.isdir(bundle_path) and os.path.isfile(os.path.join(bundle_path, 'bundle_id'))):
            self.log('checking existing bundle id %s' % os.path.join(bundle_path, 'bundle_id'))
            with open(os.path.join(bundle_path, 'bundle_id'), 'r') as bundle_id_file:
                if bundle_id_file.read() == self.bundle_id:
                    self.log('env bundle already exists with id matching current version')
                    bundle_id_file.close()
                    return
                else:
                    self.log('existing env bundle id does not match current one')

        if self.show_unpack_message(bundle_path):
            # unpack bundle sequence
            self.unpacking_thread = threading.Thread(target=self.unpack_bundle, args=(bundle_path, ))
            self.unpacking_thread.daemon = True
            self.unpacking_thread.start()
        else:
            self.log('user cancelled bundle unpack')

    def log(self, message, logfile = None):
        msg = '[%s] %s' % (self.bundle_name, message)
        print (msg)
        if logfile:
            try:
                logfile.write(msg + '\n')
                logfile.flush()
            except:
                pass

    def log_debug(self, message):
        if self.debug:
            print ('[DEBUG %s] %s' % (self.bundle_name, message))

    def load_prefs(self):
        import pickle
        
        prefix = self.prefs_folder + os.path.sep + self.bundle_name
        prefs_file_path = prefix + '.' + self.flame_user_name + '.' + self.flame_project_name + '.prefs'
        prefs_user_file_path = prefix + '.' + self.flame_user_name  + '.prefs'
        prefs_global_file_path = prefix + '.prefs'

        try:
            prefs_file = open(prefs_file_path, 'r')
            self.prefs = pickle.load(prefs_file)
            prefs_file.close()
            self.log('preferences loaded from %s' % prefs_file_path)
            self.log_debug('preferences contents:\n' + pformat(self.prefs))
        except:
            self.log('unable to load preferences from %s' % prefs_file_path)

        try:
            prefs_file = open(prefs_user_file_path, 'r')
            self.prefs_user = pickle.load(prefs_file)
            prefs_file.close()
            self.log('preferences loaded from %s' % prefs_user_file_path)
            self.log_debug('preferences contents:\n' + pformat(self.prefs_user))
        except:
            self.log('unable to load preferences from %s' % prefs_user_file_path)

        try:
            prefs_file = open(prefs_global_file_path, 'r')
            self.prefs_global = pickle.load(prefs_file)
            prefs_file.close()
            self.log('preferences loaded from %s' % prefs_global_file_path)
            self.log_debug('preferences contents:\n' + pformat(self.prefs_global))

        except:
            self.log('unable to load preferences from %s' % prefs_global_file_path)

        return True

    def save_prefs(self):
        import pickle

        if not os.path.isdir(self.prefs_folder):
            try:
                os.makedirs(self.prefs_folder)
            except:
                self.log('unable to create folder %s' % prefs_folder)
                return False

        prefix = self.prefs_folder + os.path.sep + self.bundle_name
        prefs_file_path = prefix + '.' + self.flame_user_name + '.' + self.flame_project_name + '.prefs'
        prefs_user_file_path = prefix + '.' + self.flame_user_name  + '.prefs'
        prefs_global_file_path = prefix + '.prefs'

        try:
            prefs_file = open(prefs_file_path, 'w')
            pickle.dump(self.prefs, prefs_file)
            prefs_file.close()
            self.log('preferences saved to %s' % prefs_file_path)
            self.log('preferences contents:\n' + pformat(self.prefs))
        except:
            self.log('unable to save preferences to %s' % prefs_file_path)

        try:
            prefs_file = open(prefs_user_file_path, 'w')
            pickle.dump(self.prefs_user, prefs_file)
            prefs_file.close()
            self.log('preferences saved to %s' % prefs_user_file_path)
            self.log('preferences contents:\n' + pformat(self.prefs_user))
        except:
            self.log('unable to save preferences to %s' % prefs_user_file_path)

        try:
            prefs_file = open(prefs_global_file_path, 'w')
            pickle.dump(self.prefs_global, prefs_file)
            prefs_file.close()
            self.log('preferences saved to %s' % prefs_global_file_path)
            self.log('preferences contents:\n' + pformat(self.prefs_global))
        except:
            self.log('unable to save preferences to %s' % prefs_global_file_path)
            
        return True

    def unpack_bundle(self, bundle_path):
        start = time.time()
        script_file_name, ext = os.path.splitext(os.path.abspath(__file__))
        script_file_name += '.py'
        self.log('script file: %s' % script_file_name)
        script = None

        try:
            with open(script_file_name, 'r') as scriptfile:
                script = scriptfile.read()
                scriptfile.close()
        except Exception as e:
            self.show_exception(e)
            return False
        
        if not script:
            return False
        
        logfile = None
        logfile_path = '/var/tmp/flameTimewarpML_install.log'
        try:
            open(logfile_path, "w").close()
            logfile = open(logfile_path, 'w+')
        except:
            pass
        
        if sys.platform == 'darwin':
            import subprocess
            log_cmd = """tell application "Terminal" to activate do script "tail -f """ + os.path.abspath(logfile_path) + '; exit"'
            subprocess.Popen(['osascript', '-e', log_cmd])
        else:
            log_cmd = """konsole --caption flameTimewarpML -e /bin/bash -c 'trap exit SIGINT SIGTERM; tail -f """ + os.path.abspath(logfile_path) +"; sleep 2'"
            os.system(log_cmd)
            
        self.log('bundle_id: %s size %sMb' % (self.bundle_id, len(script)//(1024 ** 2)), logfile)
        
        if os.path.isdir(bundle_path):
            bundle_backup_folder = os.path.abspath(bundle_path + '.previous')
            try:
                cmd = 'mv ' + os.path.abspath(bundle_path) + ' ' + bundle_backup_folder
                self.log('backing up existing bundle folder', logfile)
                self.log('Executing command: %s' % cmd, logfile)
                os.system(cmd)
            except Exception as e:
                self.show_exception(e)
                return False

        try:
            self.log('creating new bundle folder: %s' % bundle_path, logfile)
            os.makedirs(bundle_path)
        except Exception as e:
            self.show_exception(e)
            return False

        start_position = script.rfind('# bundle payload starts here') + 33
        payload = script[start_position:-4]
        payload_dest = os.path.join(self.bundle_location, 'bundle.tar')
        
        try:
            import base64
            self.log('unpacking payload: %s' % payload_dest, logfile)
            with open(payload_dest, 'wb') as payload_file:
                payload_file.write(base64.b64decode(payload))
                payload_file.close()
            cmd = 'tar xf ' + payload_dest + ' -C ' + self.bundle_location + '/'
            self.log('Executing command: %s' % cmd, logfile)
            status = os.system(cmd)
            self.log('exit status %s' % os.WEXITSTATUS(status), logfile)
            self.log('cleaning up %s' % payload_dest, logfile)
            os.remove(payload_dest)
        except Exception as e:
            self.show_exception(e)
            return False

        delta = time.time() - start
        self.log('bundle extracted to %s' % bundle_path, logfile)
        self.log('extracting bundle took %s sec' % str(delta), logfile)

        del payload
        del script

        env_folder = os.path.join(self.bundle_location, 'miniconda3') 
        self.install_env(env_folder, logfile)
        self.install_env_packages(env_folder, logfile)

        cmd = 'rm -rf ' + os.path.join(self.bundle_location, 'bundle', 'miniconda.package')
        self.log('Executing command: %s' % cmd, logfile)
        os.system(cmd)

        try:
            with open(os.path.join(bundle_path, 'bundle_id'), 'w+') as bundle_id_file:
                bundle_id_file.write(self.bundle_id)
        except Exception as e:
            self.show_exception(e)
            return False
        

        self.log('flameTimewarpML has finished installing its bundle and required packages', logfile)

        try:
            logfile.close()
            os.system('killall tail')
        except:
            pass

        self.show_complete_message(env_folder)

        return True
                    
    def install_env(self, env_folder, logfile):
        env_backup_folder = os.path.abspath(env_folder + '.previous')
        if os.path.isdir(env_backup_folder):
            try:
                cmd = 'rm -rf ' + env_backup_folder
                self.log('Executing command: %s' % cmd, logfile)
                os.system(cmd)
            except Exception as e:
                self.show_exception(e)
                return False
            
        if os.path.isdir(env_folder):
            try:
                cmd = 'mv ' + env_folder + ' ' + env_backup_folder
                self.log('Executing command: %s' % cmd, logfile)
                os.system(cmd)
            except Exception as e:
                self.show_exception(e)
                return False

        start = time.time()
        self.log('installing Miniconda3...', logfile)
        self.log('installing into %s' % env_folder, logfile)
        
        if sys.platform == 'darwin':
            installer_file = os.path.join(self.bundle_location, 'bundle', 'miniconda.package', 'Miniconda3-latest-MacOSX-x86_64.sh')
        else:
            installer_file = os.path.join(self.bundle_location, 'bundle', 'miniconda.package', 'Miniconda3-latest-Linux-x86_64.sh')

        cmd = '/bin/sh ' + installer_file + ' -b -p ' + env_folder
        cmd += ' 2>&1 | tee > ' + os.path.join(self.bundle_location, 'miniconda_install.log')
        self.log('Executing command: %s' % cmd, logfile)
        status = os.system(cmd)
        self.log('exit status %s' % os.WEXITSTATUS(status), logfile)
        delta = time.time() - start
        self.log('installing Miniconda took %s sec' % str(delta), logfile)

    def install_env_packages(self, env_folder, logfile):
        start = time.time()
        self.log('installing Miniconda packages...', logfile)
        cmd = """/bin/bash -c 'eval "$(""" + os.path.join(env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
        cmd += 'pip3 install -r ' + os.path.join(self.bundle_location, 'bundle', 'requirements.txt') + ' --no-index --find-links '
        cmd += os.path.join(self.bundle_location, 'bundle', 'miniconda.package', 'packages')
        cmd += ' 2>&1 | tee > '
        cmd += os.path.join(self.bundle_location, 'miniconda_packages_install.log')
        cmd += "'"

        self.log('Executing command: %s' % cmd, logfile)        
        status = os.system(cmd)
        self.log('exit status %s' % os.WEXITSTATUS(status), logfile)
        delta = time.time() - start
        self.log('installing Miniconda packages took %s sec' % str(delta), logfile)

    def show_exception(self, e):
        from PySide2 import QtWidgets
        import traceback

        msg = 'flameTimewrarpML: %s' % e
        dmsg = pformat(traceback.format_exc())

        try:
            import flame
        except:
            print (msg)
            print (dmsg)
            return False
        
        def show_error_mbox():
            mbox = QtWidgets.QMessageBox()
            mbox.setWindowTitle('flameTimewrarpML')
            mbox.setText(msg)
            mbox.setDetailedText(dmsg)
            mbox.setStyleSheet('QLabel{min-width: 800px;}')
            mbox.exec_()

        flame.schedule_idle_event(show_error_mbox)
        return True

    def show_unpack_message(self, bundle_path):
        from PySide2 import QtWidgets

        msg = 'flameTimeWarpML is going to unpack its bundle\nin background and run additional package scrips.\nCheck console for details.'
        dmsg = 'flameTimeWarpML needs Python3 environment that is newer then the one provided with Flame '
        dmsg += 'as well as some additional ML and computer-vision dependancies like PyTorch and OpenCV. '
        dmsg += 'flameTimeWarpML is going to unpack its bundle into "%s" ' % self.bundle_location
        dmsg += 'and then it will create there Miniconda3 installation with additional packages needed'

        mbox = QtWidgets.QMessageBox()
        mbox.setWindowTitle('flameTimewrarpML')
        mbox.setText(msg)
        mbox.setDetailedText(dmsg)
        mbox.setStandardButtons(QtWidgets.QMessageBox.Ok|QtWidgets.QMessageBox.Cancel)
        mbox.setStyleSheet('QLabel{min-width: 400px;}')
        btn_Continue = mbox.button(QtWidgets.QMessageBox.Ok)
        btn_Continue.setText('Continue')
        mbox.exec_()
        if mbox.clickedButton() == mbox.button(QtWidgets.QMessageBox.Cancel):
            return False
        else:
            return True

    def show_complete_message(self, bundle_path):
        from PySide2 import QtWidgets

        msg = 'flameTimewarpML has finished unpacking its bundle and required packages. You can start using it now.'
        dmsg = 'Bundle location: %s\n' % self.bundle_location
        dmsg += '* Flame scipt written by Andrii Toloshnyy (c) 2021\n'
        dmsg += '* RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation:\n'
        dmsg += '  Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang, '
        dmsg += 'arXiv preprint arXiv:2011.06294, 2020\n'
        dmsg += '* Miniconda3: (c) 2017 Continuum Analytics, Inc. (dba Anaconda, Inc.). https://www.anaconda.com. All Rights Reserved\n'
        dmsg += '* For info on additional packages see miniconda_packages_install.log'

        try:
            import flame
        except:
            print (msg)
            print (dmsg)
            return False
        
        def show_error_mbox():
            mbox = QtWidgets.QMessageBox()
            mbox.setWindowTitle('flameTimewrarpML')
            mbox.setText(msg)
            mbox.setDetailedText(dmsg)
            mbox.setStyleSheet('QLabel{min-width: 800px;}')
            mbox.exec_()

        flame.schedule_idle_event(show_error_mbox)
        return True


class flameMenuApp(object):
    def __init__(self, framework):
        self.name = self.__class__.__name__
        self.framework = framework
        self.menu_group_name = menu_group_name
        self.debug = DEBUG
        self.dynamic_menu_data = {}

        # flame module is only avaliable when a 
        # flame project is loaded and initialized
        self.flame = None
        try:
            import flame
            self.flame = flame
        except:
            self.flame = None
        
        self.prefs = self.framework.prefs_dict(self.framework.prefs, self.name)
        self.prefs_user = self.framework.prefs_dict(self.framework.prefs_user, self.name)
        self.prefs_global = self.framework.prefs_dict(self.framework.prefs_global, self.name)

        from PySide2 import QtWidgets
        self.mbox = QtWidgets.QMessageBox()

    @property
    def flame_extension_map(self):
        return {
            'Alias': 'als',
            'Cineon': 'cin',
            'Dpx': 'dpx',
            'Jpeg': 'jpg',
            'Maya': 'iff',
            'OpenEXR': 'exr',
            'Pict': 'pict',
            'Pixar': 'picio',
            'Sgi': 'sgi',
            'SoftImage': 'pic',
            'Targa': 'tga',
            'Tiff': 'tif',
            'Wavefront': 'rla',
            'QuickTime': 'mov',
            'MXF': 'mxf',
            'SonyMXF': 'mxf'
        }
        
    def __getattr__(self, name):
        def method(*args, **kwargs):
            print ('calling %s' % name)
        return method

    def log(self, message):
        self.framework.log('[' + self.name + '] ' + message)

    def rescan(self, *args, **kwargs):
        if not self.flame:
            try:
                import flame
                self.flame = flame
            except:
                self.flame = None

        if self.flame:
            self.flame.execute_shortcut('Rescan Python Hooks')
            self.log('Rescan Python Hooks')

    def get_export_preset_fields(self, preset):
        
        self.log('Flame export preset parser')

        # parses Flame Export preset and returns a dict of a parsed values
        # of False on error.
        # Example:
        # {'type': 'image',
        #  'fileType': 'OpenEXR',
        #  'fileExt': 'exr',
        #  'framePadding': 8
        #  'startFrame': 1001
        #  'useTimecode': 0
        # }
        
        from xml.dom import minidom

        preset_fields = {}

        # Flame type to file extension map

        flame_extension_map = {
            'Alias': 'als',
            'Cineon': 'cin',
            'Dpx': 'dpx',
            'Jpeg': 'jpg',
            'Maya': 'iff',
            'OpenEXR': 'exr',
            'Pict': 'pict',
            'Pixar': 'picio',
            'Sgi': 'sgi',
            'SoftImage': 'pic',
            'Targa': 'tga',
            'Tiff': 'tif',
            'Wavefront': 'rla',
            'QuickTime': 'mov',
            'MXF': 'mxf',
            'SonyMXF': 'mxf'
        }

        preset_path = ''

        if os.path.isfile(preset.get('PresetFile', '')):
            preset_path = preset.get('PresetFile')
        else:
            path_prefix = self.flame.PyExporter.get_presets_dir(
                self.flame.PyExporter.PresetVisibility.values.get(preset.get('PresetVisibility', 2)),
                self.flame.PyExporter.PresetType.values.get(preset.get('PresetType', 0))
            )
            preset_file = preset.get('PresetFile')
            if preset_file.startswith(os.path.sep):
                preset_file = preset_file[1:]
            preset_path = os.path.join(path_prefix, preset_file)

        self.log('parsing Flame export preset: %s' % preset_path)
        
        preset_xml_doc = None
        try:
            preset_xml_doc = minidom.parse(preset_path)
        except Exception as e:
            message = 'flameMenuSG: Unable parse xml export preset file:\n%s' % e
            self.mbox.setText(message)
            self.mbox.exec_()
            return False

        preset_fields['path'] = preset_path

        preset_type = preset_xml_doc.getElementsByTagName('type')
        if len(preset_type) > 0:
            preset_fields['type'] = preset_type[0].firstChild.data

        video = preset_xml_doc.getElementsByTagName('video')
        if len(video) < 1:
            message = 'flameMenuSG: XML parser error:\nUnable to find xml video tag in:\n%s' % preset_path
            self.mbox.setText(message)
            self.mbox.exec_()
            return False
        
        filetype = video[0].getElementsByTagName('fileType')
        if len(filetype) < 1:
            message = 'flameMenuSG: XML parser error:\nUnable to find video::fileType tag in:\n%s' % preset_path
            self.mbox.setText(message)
            self.mbox.exec_()
            return False

        preset_fields['fileType'] = filetype[0].firstChild.data
        if preset_fields.get('fileType', '') not in flame_extension_map:
            message = 'flameMenuSG:\nUnable to find extension corresponding to fileType:\n%s' % preset_fields.get('fileType', '')
            self.mbox.setText(message)
            self.mbox.exec_()
            return False
        
        preset_fields['fileExt'] = flame_extension_map.get(preset_fields.get('fileType'))

        name = preset_xml_doc.getElementsByTagName('name')
        if len(name) > 0:
            framePadding = name[0].getElementsByTagName('framePadding')
            startFrame = name[0].getElementsByTagName('startFrame')
            useTimecode = name[0].getElementsByTagName('useTimecode')
            if len(framePadding) > 0:
                preset_fields['framePadding'] = int(framePadding[0].firstChild.data)
            if len(startFrame) > 0:
                preset_fields['startFrame'] = int(startFrame[0].firstChild.data)
            if len(useTimecode) > 0:
                preset_fields['useTimecode'] = useTimecode[0].firstChild.data

        return preset_fields

    def sanitized(self, text):
        import re

        if text is None:
            return None
        
        text = text.strip()
        exp = re.compile(u'[^\w\.-]', re.UNICODE)

        if isinstance(text, unicode):
            result = exp.sub('_', value)
        else:
            decoded = text.decode('utf-8')
            result = exp.sub('_', decoded).encode('utf-8')

        return re.sub('_\_+', '_', result)

    def create_timestamp_uid(self):
        # generates UUID for the batch setup
        import uuid
        from datetime import datetime
        
        uid = ((str(uuid.uuid1()).replace('-', '')).upper())
        timestamp = (datetime.now()).strftime('%Y%b%d_%H%M').upper()
        return timestamp + '_' + uid[:3]


class flameTimewrapML(flameMenuApp):
    def __init__(self, framework):
        flameMenuApp.__init__(self, framework)
        self.env_folder = os.path.join(self.framework.bundle_location, 'miniconda3')
        
        self.loops = []
        self.threads = True

        if not self.prefs.master.get(self.name):
            self.prefs['working_folder'] = '/var/tmp'
        self.working_folder = self.prefs['working_folder']
        if not os.path.isdir(self.working_folder):
            self.working_folder = '/var/tmp'

        self.new_speed = 1
        self.cpu = False

    def build_menu(self):
        def scope_clip(selection):
            import flame
            for item in selection:
                if isinstance(item, (flame.PyClip)):
                    return True
            return False

        if not self.flame:
            return []
        
        if not os.path.isfile(os.path.join(self.framework.bundle_location, 'bundle', 'bundle_id')):
            return []
        
        menu = {'actions': []}
        menu['name'] = self.menu_group_name

        menu_item = {}
        menu_item['name'] = 'Slow Down clip(s) with ML'
        menu_item['execute'] = self.slowmo
        menu_item['isEnabled'] = scope_clip
        menu_item['waitCursor'] = False
        menu['actions'].append(menu_item)

        menu_item = {}
        menu_item['name'] = 'Version: ' + __version__
        menu_item['execute'] = self.slowmo
        menu_item['isEnabled'] = False
        menu['actions'].append(menu_item)

        return menu

    def slowmo(self, selection):
        result = self.slowmo_dialog()
        if not result:
            return False

        working_folder = str(result.get('working_folder', '/var/tmp'))
        speed = result.get('speed', 1)
        cmd_strings = []
        number_of_clips = 0

        import flame
        for item in selection:
            if isinstance(item, (flame.PyClip)):
                number_of_clips += 1

                clip = item
                clip_name = clip.name.get_value()
                
                output_folder = os.path.abspath(
                    os.path.join(
                        working_folder, 
                        self.sanitized(clip_name) + '_TWML' + str(2 ** speed) + '_' + self.create_timestamp_uid()
                        )
                    )

                if os.path.isdir(output_folder):
                    from PySide2 import QtWidgets

                    msg = 'Folder %s exists' % output_folder
                    mbox = QtWidgets.QMessageBox()
                    mbox.setWindowTitle('flameTimewrarpML')
                    mbox.setText(msg)
                    mbox.setStandardButtons(QtWidgets.QMessageBox.Ok|QtWidgets.QMessageBox.Cancel)
                    mbox.setStyleSheet('QLabel{min-width: 400px;}')
                    btn_Continue = mbox.button(QtWidgets.QMessageBox.Ok)
                    btn_Continue.setText('Owerwrite')
                    mbox.exec_()
                    if mbox.clickedButton() == mbox.button(QtWidgets.QMessageBox.Cancel):
                        return False
                    cmd = 'rm -f ' + output_folder + '/*'
                    self.log('Executing command: %s' % cmd)
                    os.system(cmd)

                self.export_clip(item, output_folder)

                cmd = 'python3 '
                if self.cpu:
                    cmd = 'export OMP_NUM_THREADS=1; python3 '
                cmd += os.path.join(self.framework.bundle_location, 'bundle', 'inference_sequence.py')
                cmd += ' --input ' + os.path.join(output_folder, 'source') + ' --output ' + output_folder
                cmd += ' --exp=' + str(speed)
                if self.cpu:
                    cmd += ' --cpu'
                cmd += "; "
                cmd_strings.append(cmd)
                
                new_clip_name = clip_name + '_TWML' + str(2 ** speed)
                watcher = threading.Thread(target=self.import_watcher, args=(output_folder, clip, new_clip_name))
                watcher.daemon = True
                watcher.start()
                self.loops.append(watcher)
        
        if sys.platform == 'darwin':
            cmd_prefix = """tell application "Terminal" to activate do script "clear; """
            # cmd_prefix += """ echo " & quote & "Received """
            # cmd_prefix += str(number_of_clips)
            #cmd_prefix += ' clip ' if number_of_clips < 2 else ' clips '
            # cmd_prefix += 'to process, press Ctrl+C to cancel" & quote &; '
            cmd_prefix += """/bin/bash -c 'eval " & quote & "$("""
            cmd_prefix += os.path.join(self.env_folder, 'bin', 'conda')
            cmd_prefix += """ shell.bash hook)" & quote & "; conda activate; """
            cmd_prefix += 'cd ' + os.path.join(self.framework.bundle_location, 'bundle') + '; '
            
            ml_cmd = cmd_prefix
           
            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            ml_cmd += """'; exit" """

            import subprocess
            subprocess.Popen(['osascript', '-e', ml_cmd])
        
        else:
            cmd_prefix = """konsole -e /bin/bash -c 'eval "$(""" + os.path.join(self.env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
            cmd_prefix += 'cd ' + os.path.join(self.framework.bundle_location, 'bundle') + '; '

            ml_cmd = cmd_prefix
            ml_cmd += 'echo "Received ' + str(number_of_clips)
            ml_cmd += ' clip ' if number_of_clips < 2 else ' clips '
            ml_cmd += 'to process, press Ctrl+C to cancel"; '
            ml_cmd += 'trap exit SIGINT SIGTERM; '

            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            ml_cmd +="'"
            self.log('Executing command: %s' % ml_cmd)
            os.system(ml_cmd)

        flame.execute_shortcut('Refresh Thumbnails')

    def slowmo_dialog(self, *args, **kwargs):
        from PySide2 import QtWidgets, QtCore

        self.new_speed_list = {
            1: '1/2',
            2: '1/4',
            3: '1/8',
            4: '1/16' 
        }
        
        # flameMenuNewBatch_prefs = self.framework.prefs.get('flameMenuNewBatch', {})
        # self.asset_task_template =  flameMenuNewBatch_prefs.get('asset_task_template', {})

        window = QtWidgets.QDialog()
        window.setMinimumSize(280, 180)
        window.setWindowTitle('Slow down clip(s) with ML')
        window.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        window.setStyleSheet('background-color: #313131')

        screen_res = QtWidgets.QDesktopWidget().screenGeometry()
        window.move((screen_res.width()/2)-150, (screen_res.height() / 2)-180)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignTop)

        # New Speed hbox
        new_speed_hbox = QtWidgets.QHBoxLayout()
        new_speed_hbox.setAlignment(QtCore.Qt.AlignLeft)

        # New Speed label

        lbl_NewSpeed = QtWidgets.QLabel('New Clip(s) Speed ', window)
        lbl_NewSpeed.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_NewSpeed.setMinimumHeight(28)
        lbl_NewSpeed.setAlignment(QtCore.Qt.AlignCenter)
        new_speed_hbox.addWidget(lbl_NewSpeed)

        # Spacer
        lbl_NewSpeedSpacer = QtWidgets.QLabel('', window)
        lbl_NewSpeedSpacer.setAlignment(QtCore.Qt.AlignCenter)
        lbl_NewSpeedSpacer.setMinimumSize(8, 28)
        new_speed_hbox.addWidget(lbl_NewSpeedSpacer)

        # New Speed Selector
        btn_NewSpeedSelector = QtWidgets.QPushButton(window)
        btn_NewSpeedSelector.setText(self.new_speed_list.get(self.new_speed))
        def selectNewSpeed(new_speed_id):
            self.new_speed = new_speed_id
            btn_NewSpeedSelector.setText(self.new_speed_list.get(self.new_speed))
        btn_NewSpeedSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_NewSpeedSelector.setMinimumSize(80, 28)
        # btn_NewSpeedSelector.move(40, 102)
        btn_NewSpeedSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')
        btn_NewSpeedSelector_menu = QtWidgets.QMenu()

        for new_speed_id in sorted(self.new_speed_list.keys()):
            code = self.new_speed_list.get(new_speed_id, '1/2')
            action = btn_NewSpeedSelector_menu.addAction(code)
            action.triggered[()].connect(lambda new_speed_id=new_speed_id: selectNewSpeed(new_speed_id))
        btn_NewSpeedSelector.setMenu(btn_NewSpeedSelector_menu)
        new_speed_hbox.addWidget(btn_NewSpeedSelector)

        # Spacer
        lbl_NewSpeedSpacer = QtWidgets.QLabel('', window)
        lbl_NewSpeedSpacer.setAlignment(QtCore.Qt.AlignCenter)
        lbl_NewSpeedSpacer.setMinimumSize(48, 28)
        new_speed_hbox.addWidget(lbl_NewSpeedSpacer)

        if not sys.platform == 'darwin':
            # Cpu Proc button
            
            def enableCpuProc():
                if self.cpu:
                    btn_CpuProc.setStyleSheet('QPushButton {color: #989898; background-color: #373737; border-top: 1px inset #555555; border-bottom: 1px inset black}')
                    self.cpu = False
                else:
                    btn_CpuProc.setStyleSheet('QPushButton {font:italic; background-color: #4f4f4f; color: #d9d9d9; border-top: 1px inset black; border-bottom: 1px inset #555555}')
                    self.cpu = True

            btn_CpuProc = QtWidgets.QPushButton('CPU Proc', window)
            btn_CpuProc.setFocusPolicy(QtCore.Qt.NoFocus)
            btn_CpuProc.setMinimumSize(88, 28)
            # btn_CpuProc.move(0, 34)
            if self.cpu:
                btn_CpuProc.setStyleSheet('QPushButton {font:italic; background-color: #4f4f4f; color: #d9d9d9; border-top: 1px inset black; border-bottom: 1px inset #555555}')
            else:
                btn_CpuProc.setStyleSheet('QPushButton {color: #989898; background-color: #373737; border-top: 1px inset #555555; border-bottom: 1px inset black}')
            btn_CpuProc.pressed.connect(enableCpuProc)

            new_speed_hbox.addWidget(btn_CpuProc, alignment = QtCore.Qt.AlignRight)

        vbox.addLayout(new_speed_hbox)

        # Work Folder Label

        lbl_WorkFolder = QtWidgets.QLabel('Export folder', window)
        lbl_WorkFolder.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_WorkFolder.setMinimumHeight(28)
        lbl_WorkFolder.setMaximumHeight(28)
        lbl_WorkFolder.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(lbl_WorkFolder)

        # Work Folder Text Field

        hbox_workfolder = QtWidgets.QHBoxLayout()
        hbox_workfolder.setAlignment(QtCore.Qt.AlignLeft)


        def chooseFolder():
            result_folder = str(QtWidgets.QFileDialog.getExistingDirectory(window, "Open Directory", self.working_folder, QtWidgets.QFileDialog.ShowDirsOnly))
            if result_folder =='':
                return
            self.working_folder = result_folder
            txt_WorkFolder.setText(self.working_folder)
            self.prefs['working_folder'] = self.working_folder

            #dialog = QtWidgets.QFileDialog()
            #dialog.setWindowTitle('Select export folder')
            #dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
            #dialog.setDirectory(self.working_folder)
            #dialog.setFileMode(QtWidgets.QFileDialog.Directory)
            #path = QtWidgets.QFileDialog.getExistingDirectory()
            # dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
            #
            # if dialog.exec_() == QtWidgets.QDialog.Accepted:
            #    file_full_path = str(dialog.selectedFiles()[0])

        def txt_WorkFolder_textChanged():
            self.working_folder = txt_WorkFolder.text()
        txt_WorkFolder = QtWidgets.QLineEdit('', window)
        txt_WorkFolder.setFocusPolicy(QtCore.Qt.ClickFocus)
        txt_WorkFolder.setMinimumSize(280, 28)
        txt_WorkFolder.setStyleSheet('QLineEdit {color: #9a9a9a; background-color: #373e47; border-top: 1px inset #black; border-bottom: 1px inset #545454}')
        txt_WorkFolder.setText(self.working_folder)
        txt_WorkFolder.textChanged.connect(txt_WorkFolder_textChanged)
        hbox_workfolder.addWidget(txt_WorkFolder)

        btn_changePreset = QtWidgets.QPushButton('Choose', window)
        btn_changePreset.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_changePreset.setMinimumSize(88, 28)
        btn_changePreset.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #424142; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                   'QPushButton:pressed {font:italic; color: #d9d9d9}')
        btn_changePreset.clicked.connect(chooseFolder)
        hbox_workfolder.addWidget(btn_changePreset, alignment = QtCore.Qt.AlignLeft)

        vbox.addLayout(hbox_workfolder)


        # Spacer Label

        lbl_Spacer = QtWidgets.QLabel('', window)
        lbl_Spacer.setStyleSheet('QFrame {color: #989898; background-color: #313131}')
        lbl_Spacer.setMinimumHeight(4)
        lbl_Spacer.setMaximumHeight(4)
        lbl_Spacer.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(lbl_Spacer)

        # Create and Cancel Buttons
        hbox_Create = QtWidgets.QHBoxLayout()

        select_btn = QtWidgets.QPushButton('Create', window)
        select_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        select_btn.setMinimumSize(128, 28)
        select_btn.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #424142; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                'QPushButton:pressed {font:italic; color: #d9d9d9}')
        select_btn.clicked.connect(window.accept)
        select_btn.setAutoDefault(True)
        select_btn.setDefault(True)

        cancel_btn = QtWidgets.QPushButton('Cancel', window)
        cancel_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        cancel_btn.setMinimumSize(128, 28)
        cancel_btn.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #424142; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                'QPushButton:pressed {font:italic; color: #d9d9d9}')
        cancel_btn.clicked.connect(window.reject)

        hbox_Create.addWidget(cancel_btn)
        hbox_Create.addWidget(select_btn)

        vbox.addLayout(hbox_Create)

        window.setLayout(vbox)
        if window.exec_():
            self.framework.save_prefs()
            return {
                'speed': self.new_speed,
                'working_folder': self.working_folder
            }
        else:
            return {}

    def export_clip(self, clip, output_folder):
        import flame
        import traceback

        export_dir = os.path.join(output_folder, 'source')
        if not os.path.isdir(export_dir):
            self.log('creating folders: %s' % export_dir)
            try:
                os.makedirs(export_dir)
            except Exception as e:
                from PySide2 import QtWidgets, QtCore
                msg = 'flameTimewrarpML: %s' % e
                dmsg = pformat(traceback.format_exc())
                
                def show_error_mbox():
                    mbox = QtWidgets.QMessageBox()
                    mbox.setWindowTitle('flameTimewrarpML')
                    mbox.setText(msg)
                    mbox.setDetailedText(dmsg)
                    mbox.setStyleSheet('QLabel{min-width: 800px;}')
                    mbox.exec_()
            
                flame.schedule_idle_event(show_error_mbox)
                return False

        class ExportHooks(object):
            def preExport(self, info, userData, *args, **kwargs):
                pass
            def postExport(self, info, userData, *args, **kwargs):
                pass
            def preExportSequence(self, info, userData, *args, **kwargs):
                pass
            def postExportSequence(self, info, userData, *args, **kwargs):
                pass
            def preExportAsset(self, info, userData, *args, **kwargs):
                pass
            def postExportAsset(self, info, userData, *args, **kwargs):
                del args, kwargs
                pass
            def exportOverwriteFile(self, path, *args, **kwargs):
                del path, args, kwargs
                return "overwrite"

        exporter = self.flame.PyExporter()
        exporter.foreground = True

        export_preset_folder = self.flame.PyExporter.get_presets_dir(self.flame.PyExporter.PresetVisibility.values.get(2),
                        self.flame.PyExporter.PresetType.values.get(0))
        export_preset = os.path.join(export_preset_folder, 'OpenEXR', 'OpenEXR (16-bit fp PIZ).xml')
        exporter.export(clip, export_preset, export_dir, hooks=ExportHooks())

    def import_watcher(self, path, clip, new_clip_name):
        import hashlib
        lockfile_name = hashlib.sha1(path.encode()).hexdigest().upper() + '.lock'
        lockfile = os.path.join(self.framework.bundle_location, 'bundle', 'locks', lockfile_name)
        cmd = 'echo "' + path + '">' + lockfile
        self.log('Executing command: %s' % cmd)
        os.system(cmd)

        flame_path = None
        
        def import_flame_clip():
            import flame
            new_clips = flame.import_clips(flame_path, clip.parent)
            
            if len(new_clips) > 0:
                new_clip = new_clips[0]
                if new_clip:
                    new_clip.name.set_value(new_clip_name)
            
            flame.execute_shortcut('Refresh Thumbnails')

            # Colour Mgmt logic for future settin
            '''
            for version in new_clip.versions:
                for track in version.tracks:
                    for segment in track.segments:
                        segment.create_effect('Source Colour Mgmt')
            '''
            # End of Colour Mgmt logic for future settin


            # Hard Commit Logic for future setting
            '''
            for version in new_clip.versions:
                for track in version.tracks:
                    for segment in track.segments:
                        segment.create_effect('Source Image')
            
            new_clip.open_as_sequence()
            flame.execute_shortcut('Hard Commit Selection in Timeline')
            flame.execute_shortcut('Refresh Thumbnails')
            '''
            # End of Hard Commit Logic for future setting


        while self.threads:
            if not os.path.isfile(lockfile):
                cmd = 'rm -f "' + os.path.join(path, 'source') + '/"*'
                self.log('Executing command: %s' % cmd)
                os.system(cmd)
                try:
                    os.rmdir(os.path.join(path, 'source'))
                except Exception as e:
                    self.log('Error removing %s: %s' % (os.path.join(path, 'source'), e))

                file_names = os.listdir(path)

                if file_names:

                    file_names.sort()
                    first_frame, ext = os.path.splitext(file_names[0])
                    last_frame, ext = os.path.splitext(file_names[-1])
                    flame_path = os.path.join(path, '[' + first_frame + '-' + last_frame + ']' + '.exr')
                
                    import flame
                    flame.schedule_idle_event(import_flame_clip)

                break
            time.sleep(0.1)

    def terminate_loops(self):
        self.threads = False
        
        for loop in self.loops:
            loop.join()


# --- FLAME STARTUP SEQUENCE ---
# Flame startup sequence is a bit complicated
# If the app installed in /opt/Autodesk/<user>/python
# project hooks are not called at startup. 
# One of the ways to work around it is to check 
# if we are able to import flame module straght away. 
# If it is the case - flame project is already loaded 
# and we can start our constructor. Otherwise we need 
# to wait for app_initialized hook to be called - that would 
# mean the project is finally loaded. 
# project_changed_dict hook seem to be a good place to wrap things up

# main objects:
# app_framework takes care of preferences and general stuff
# apps is a list of apps to load inside the main program

app_framework = None
apps = []

# Exception handler
def exeption_handler(exctype, value, tb):
    from PySide2 import QtWidgets
    import traceback
    msg = 'flameTimewrarpML: Python exception %s in %s' % (value, exctype)
    mbox = QtWidgets.QMessageBox()
    mbox.setWindowTitle('flameTimewrarpML')
    mbox.setText(msg)
    mbox.setDetailedText(pformat(traceback.format_exception(exctype, value, tb)))
    mbox.setStyleSheet('QLabel{min-width: 800px;}')
    mbox.exec_()
    sys.__excepthook__(exctype, value, tb)
sys.excepthook = exeption_handler

# register clean up logic to be called at Flame exit
def cleanup(local_apps, Local_app_framework):
    global app_framework
    global apps
    
    if apps:
        if DEBUG:
            print ('[DEBUG %s] unloading apps:\n%s' % ('flameMenuSG', pformat(apps)))
        while len(apps):
            app = apps.pop()
            if DEBUG:
                print ('[DEBUG %s] unloading: %s' % ('flameMenuSG', app.name))
            app.terminate_loops()
            del app        
        apps = []

    if app_framework:
        print ('PYTHON\t: %s cleaning up' % app_framework.bundle_name)
        app_framework.save_prefs()
        app_framework = None

atexit.register(cleanup, apps, app_framework)

def load_apps(apps, app_framework):
    apps.append(flameTimewrapML(app_framework))
    app_framework.apps = apps
    if DEBUG:
        print ('[DEBUG %s] loaded:\n%s' % (app_framework.bundle_name, pformat(apps)))

def project_changed_dict(info):
    global app_framework
    global apps
    cleanup(apps, app_framework)

def app_initialized(project_name):
    global app_framework
    global apps
    if not app_framework:
        app_framework = flameAppFramework()
        print ('PYTHON\t: %s initializing' % app_framework.bundle_name)
    if not apps:
        load_apps(apps, app_framework)

try:
    import flame
    app_initialized(flame.project.current_project.name)
except:
    pass

def get_media_panel_custom_ui_actions():

    menu = []
    selection = []

    try:
        import flame
        selection = flame.media_panel.selected_entries
    except:
        pass

    for app in apps:
        if app.__class__.__name__ == 'flameTimewrapML':
            app_menu = []
            app_menu = app.build_menu()
            if app_menu:
                menu.append(app_menu)
    return menu

# bundle payload starts here
'''
BUNDLE_PAYLOAD
'''