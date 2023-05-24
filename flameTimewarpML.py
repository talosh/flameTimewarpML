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
import hashlib
import pickle

from PySide2 import QtWidgets, QtCore, QtGui


from pprint import pprint
from pprint import pformat



'''
from adsk.libwiretapPythonClientAPI import WireTapClient
from adsk.libwiretapPythonClientAPI import WireTapServerHandle
from adsk.libwiretapPythonClientAPI import WireTapNodeHandle
from adsk.libwiretapPythonClientAPI import WireTapClipFormat
from adsk.libwiretapPythonClientAPI import WireTapInt
'''

# Configurable settings
menu_group_name = 'Timewarp ML'
DEBUG = False
app_name = 'flameTimewarpML'
__version__ = 'v0.5.0.dev.002'

class flameAppFramework(object):
    # flameAppFramework class takes care of preferences

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
            return list(self.master[self.name].keys())

        @classmethod
        def fromkeys(cls, keys, v=None):
            return self.master[self.name].fromkeys(keys, v)
        
        def __repr__(self):
            return '{0}({1})'.format(type(self).__name__, self.master[self.name].__repr__())

        def master_keys(self):
            return list(self.master.keys())

    def __init__(self, *args, **kwargs):
        self.name = self.__class__.__name__
        self.app_name = kwargs.get('app_name', 'flameApp')
        self.bundle_name = self.sanitize_name(self.app_name)
        self.version = __version__
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
            self.flame_project_name = 'UnknownFlameProject'
            self.flame_user_name = 'UnknownFlameUser'
        
        import socket
        self.hostname = socket.gethostname()
        
        if sys.platform == 'darwin':
            self.prefs_folder = os.path.join(
                os.path.expanduser('~'),
                 'Library',
                 'Preferences',
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

        # menu auto-refresh defaults

        if not self.prefs_global.get('menu_auto_refresh'):
            self.prefs_global['menu_auto_refresh'] = {
                'media_panel': True,
                'batch': True,
                'main_menu': True
            }

        self.apps = []

        self.bundle_location = '/var/tmp'
        self.bundle_path = os.path.join(
            self.bundle_location,
            self.bundle_name
        )

        if (sys.platform == 'darwin'):
            if os.getenv('FLAMETWML_BUNDLE_MAC'):
                self.bundle_location = os.path.dirname(os.getenv('FLAMETWML_BUNDLE_MAC'))
                self.bundle_path = os.getenv('FLAMETWML_BUNDLE_MAC')
        elif sys.platform.startswith('linux') and FLAMETWML_BUNDLE_LINUX:
            if os.getenv('FLAMETWML_BUNDLE_LINUX'):
                self.bundle_location = os.path.dirname(os.getenv('FLAMETWML_BUNDLE_LINUX'))
                self.bundle_path = os.getenv('FLAMETWML_BUNDLE_LINUX')

        # site-packages check and payload unpack if nessesary
        self.site_packages_folder = os.path.join(
            self.bundle_path,
            'site-packages'
        )

        if not self.check_bundle_id():
            threading.Thread(
                target=self.unpack_bundle,
                args=(os.path.dirname(self.site_packages_folder), )
            ).start()

    def log(self, message):
        try:
            print ('[%s] %s' % (self.bundle_name, str(message)))
        except:
            pass

    def log_debug(self, message):
        if self.debug:
            try:
                print ('[DEBUG %s] %s' % (self.bundle_name, str(message)))
            except:
                pass

    def load_prefs(self):
        import json
        
        prefix = self.prefs_folder + os.path.sep + self.bundle_name
        prefs_file_path = prefix + '.' + self.flame_user_name + '.' + self.flame_project_name + '.prefs.json'
        prefs_user_file_path = prefix + '.' + self.flame_user_name  + '.prefs.json'
        prefs_global_file_path = prefix + '.prefs.json'

        try:
            with open(prefs_file_path, 'r') as prefs_file:
                self.prefs = json.load(prefs_file)
            self.log_debug('preferences loaded from %s' % prefs_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs, indent=4))
        except Exception as e:
            self.log('unable to load preferences from %s' % prefs_file_path)
            self.log_debug(e)

        try:
            with open(prefs_user_file_path, 'r') as prefs_file:
                self.prefs_user = json.load(prefs_file)
            self.log_debug('preferences loaded from %s' % prefs_user_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_user, indent=4))
        except Exception as e:
            self.log('unable to load preferences from %s' % prefs_user_file_path)
            self.log_debug(e)

        try:
            with open(prefs_global_file_path, 'r') as prefs_file:
                self.prefs_global = json.load(prefs_file)
            self.log_debug('preferences loaded from %s' % prefs_global_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_global, indent=4))
        except Exception as e:
            self.log('unable to load preferences from %s' % prefs_global_file_path)
            self.log_debug(e)

        return True

    def save_prefs(self):
        import json

        if not os.path.isdir(self.prefs_folder):
            try:
                os.makedirs(self.prefs_folder)
            except:
                self.log('unable to create folder %s' % self.prefs_folder)
                return False

        prefix = self.prefs_folder + os.path.sep + self.bundle_name
        prefs_file_path = prefix + '.' + self.flame_user_name + '.' + self.flame_project_name + '.prefs.json'
        prefs_user_file_path = prefix + '.' + self.flame_user_name  + '.prefs.json'
        prefs_global_file_path = prefix + '.prefs.json'

        try:
            with open(prefs_file_path, 'w') as prefs_file:
                json.dump(self.prefs, prefs_file, indent=4)
            self.log_debug('preferences saved to %s' % prefs_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs, indent=4))
        except Exception as e:
            self.log('unable to save preferences to %s' % prefs_file_path)
            self.log_debug(e)

        try:
            with open(prefs_user_file_path, 'w') as prefs_file:
                json.dump(self.prefs_user, prefs_file, indent=4)
            self.log_debug('preferences saved to %s' % prefs_user_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_user, indent=4))
        except Exception as e:
            self.log('unable to save preferences to %s' % prefs_user_file_path)
            self.log_debug(e)

        try:
            with open(prefs_global_file_path, 'w') as prefs_file:
                json.dump(self.prefs_global, prefs_file, indent=4)
            self.log_debug('preferences saved to %s' % prefs_global_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_global, indent=4))
        except Exception as e:
            self.log('unable to save preferences to %s' % prefs_global_file_path)
            self.log_debug(e)
            
        return True

    def check_bundle_id(self):
        bundle_id_file_path = os.path.join(
            self.bundle_path,
            'bundle_id'
            )
        bundle_id = self.version

        if (os.path.isdir(self.bundle_path) and os.path.isfile(bundle_id_file_path)):
            self.log('checking existing bundle id %s' % bundle_id_file_path)
            try:
                with open(bundle_id_file_path, 'r') as bundle_id_file:
                    if bundle_id_file.read() == bundle_id:
                        self.log('bundle folder exists with id matching current version')
                        bundle_id_file.close()
                        return True
                    else:
                        self.log('existing env bundle id does not match current one')
                        return False
            except Exception as e:
                self.log(pformat(e))
                return False
        elif not os.path.isdir(self.bundle_path):
            self.log('bundle folder does not exist: %s' % self.bundle_path)
            return False
        elif not os.path.isfile(bundle_id_file_path):
            self.log('bundle id file does not exist: %s' % bundle_id_file_path)
            return False

    def unpack_bundle(self, bundle_path):
        start = time.time()
        script_file_name, ext = os.path.splitext(os.path.abspath(__file__))
        script_file_name += '.py'
        self.log('script file: %s' % script_file_name)
        script = None
        payload = None

        try:
            with open(script_file_name, 'r+') as scriptfile:
                script = scriptfile.read()
                start_position = script.rfind('# bundle payload starts here')
                
                if script[start_position -1: start_position] != '\n':
                    scriptfile.close()
                    return False

                start_position += 33
                payload = script[start_position:-4]
                # scriptfile.truncate(start_position - 34)
                scriptfile.close()
        except Exception as e:
            self.log_exception(e)
            return False
        
        del script
        if not payload:
            return False

        bundle_backup_folder = ''
        if os.path.isdir(bundle_path):
            bundle_backup_folder = os.path.abspath(bundle_path + '.previous')
            if os.path.isdir(bundle_backup_folder):
                try:
                    cmd = 'rm -rf "' + os.path.abspath(bundle_backup_folder) + '"'
                    self.log('removing previous backup folder')
                    self.log('Executing command: %s' % cmd)
                    os.system(cmd)
                except Exception as e:
                    self.log_exception(e)
                    return False
            try:
                cmd = 'mv "' + os.path.abspath(bundle_path) + '" "' + bundle_backup_folder + '"'
                self.log('backing up existing bundle folder')
                self.log('Executing command: %s' % cmd)
                os.system(cmd)
            except Exception as e:
                self.log_exception(e)
                return False

        try:
            self.log('creating new bundle folder: %s' % bundle_path)
            os.makedirs(bundle_path)
        except Exception as e:
            self.log_exception(e)
            return False

        payload_dest = os.path.join(
            bundle_path, 
            self.sanitize_name(self.bundle_name + '.' + __version__ + '.bundle.tar.gz')
            )
        
        try:
            import base64
            self.log('unpacking payload: %s' % payload_dest)
            with open(payload_dest, 'wb') as payload_file:
                payload_file.write(base64.b64decode(payload))
                payload_file.close()
            cmd = 'tar xf "' + payload_dest + '" -C "' + bundle_path + '/"'
            self.log('Executing command: %s' % cmd)
            status = os.system(cmd)
            self.log('exit status %s' % os.WEXITSTATUS(status))

            # self.log('cleaning up %s' % payload_dest, logfile)
            # os.remove(payload_dest)
        
        except Exception as e:
            self.log_exception(e)
            return False

        delta = time.time() - start
        self.log('bundle extracted to %s' % bundle_path)
        self.log('extracting bundle took %s sec' % '{:.1f}'.format(delta))

        del payload
        try:
            os.remove(payload_dest)
        except Exception as e:
            self.log_exception(e)

        try:
            with open(os.path.join(bundle_path, 'bundle_id'), 'w') as bundle_id_file:
                bundle_id_file.write(self.version)
        except Exception as e:
            self.log_exception(e)
            return False
        
        return True

    def log_exception(self, e):
        self.log(pformat(e))
        self.log_debug(pformat(traceback.format_exc()))

    def sanitize_name(self, name_to_sanitize):
        import re
        if name_to_sanitize is None:
            return None
        
        stripped_name = name_to_sanitize.strip()
        exp = re.compile(u'[^\w\.-]', re.UNICODE)

        result = exp.sub('_', stripped_name)
        return re.sub('_\_+', '_', result)


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
        self.framework.log(message)

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

        if sys.version_info < (3,):
            if isinstance(text, unicode):
                result = exp.sub('_', text)
            else:
                decoded = text.decode('utf-8')
                result = exp.sub('_', decoded).encode('utf-8')
        else:
            if isinstance(text, str):
                result = exp.sub('_', text)
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


class flameTimewarpML(flameMenuApp):

    class Progress(QtWidgets.QWidget):

        class Ui_Progress(object):
            def setupUi(self, Progress):
                Progress.setObjectName("Progress")
                Progress.setStyleSheet("#Progress {background-color: #242424;} #frame {border: 1px solid #474747; border-radius: 5px;}\n")

                self.verticalLayout = QtWidgets.QVBoxLayout(Progress)  # Change from horizontal layout to vertical layout
                self.verticalLayout.setSpacing(0)
                self.verticalLayout.setContentsMargins(0, 0, 0, 0)
                self.verticalLayout.setObjectName("verticalLayout")

                # Create a new widget for the stripe at the top
                self.stripe_widget = QtWidgets.QWidget(Progress)
                self.stripe_widget.setStyleSheet("background-color: #474747;")
                self.stripe_widget.setFixedHeight(24)  # Adjust this value to change the height of the stripe

                # Create a label inside the stripe widget
                self.stripe_label = QtWidgets.QLabel("Your text here")  # Replace this with the text you want on the stripe
                self.stripe_label.setStyleSheet("color: #cbcbcb;")  # Change this to set the text color

                # Create a layout for the stripe widget and add the label to it
                stripe_layout = QtWidgets.QHBoxLayout()
                stripe_layout.addWidget(self.stripe_label)
                stripe_layout.addStretch(1)
                stripe_layout.setContentsMargins(18, 0, 0, 0)  # This will ensure the label fills the stripe widget

                # Set the layout to stripe_widget
                self.stripe_widget.setLayout(stripe_layout)

                # Add the stripe widget to the top of the main window's layout
                self.verticalLayout.addWidget(self.stripe_widget)
                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

                self.src_horisontal_layout = QtWidgets.QHBoxLayout(Progress)  # Change from horizontal layout to vertical layout
                self.src_horisontal_layout.setSpacing(0)
                self.src_horisontal_layout.setContentsMargins(0, 0, 0, 0)
                self.src_horisontal_layout.setObjectName("srcHorisontalLayout")

                self.src_frame_one = QtWidgets.QFrame(Progress)
                self.src_frame_one.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.src_frame_one.setFrameShadow(QtWidgets.QFrame.Raised)
                self.src_frame_one.setObjectName("frame")

                self.src_frame_two = QtWidgets.QFrame(Progress)
                self.src_frame_two.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.src_frame_two.setFrameShadow(QtWidgets.QFrame.Raised)
                self.src_frame_two.setObjectName("frame")

                self.src_horisontal_layout.addWidget(self.src_frame_one)
                self.src_horisontal_layout.addWidget(self.src_frame_two)

                self.verticalLayout.addLayout(self.src_horisontal_layout)
                self.verticalLayout.setStretchFactor(self.src_horisontal_layout, 4)

                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

                self.int_horisontal_layout = QtWidgets.QHBoxLayout(Progress)  # Change from horizontal layout to vertical layout
                self.int_horisontal_layout.setSpacing(0)
                self.int_horisontal_layout.setContentsMargins(0, 0, 0, 0)
                self.int_horisontal_layout.setObjectName("intHorisontalLayout")

                self.int_frame_flow = QtWidgets.QFrame(Progress)
                self.int_frame_flow.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_flow.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_flow.setObjectName("frame")

                self.int_frame_wrp1 = QtWidgets.QFrame(Progress)
                self.int_frame_wrp1.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_wrp1.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_wrp1.setObjectName("frame")

                self.int_frame_wrp2 = QtWidgets.QFrame(Progress)
                self.int_frame_wrp2.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_wrp2.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_wrp2.setObjectName("frame")

                self.int_frame_mix = QtWidgets.QFrame(Progress)
                self.int_frame_mix.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_mix.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_mix.setObjectName("frame")

                self.int_horisontal_layout.addWidget(self.int_frame_flow)
                self.int_horisontal_layout.addWidget(self.int_frame_wrp1)
                self.int_horisontal_layout.addWidget(self.int_frame_wrp2)
                self.int_horisontal_layout.addWidget(self.int_frame_mix)

                self.verticalLayout.addLayout(self.int_horisontal_layout)
                self.verticalLayout.setStretchFactor(self.int_horisontal_layout, 2)

                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

                self.res_frame = QtWidgets.QFrame(Progress)
                self.res_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.res_frame.setFrameShadow(QtWidgets.QFrame.Raised)
                self.res_frame.setObjectName("frame")
                self.verticalLayout.addWidget(self.res_frame)
                self.verticalLayout.setStretchFactor(self.res_frame, 8)

                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space


                # Create a new horizontal layout for the bottom of the window
                bottom_layout = QtWidgets.QHBoxLayout()

                # Add a close button to the bottom layout
                self.close_button = QtWidgets.QPushButton("Close")
                self.close_button.clicked.connect(Progress.close)
                self.close_button.setContentsMargins(10, 4, 4, 4)
                self.set_button_style(self.close_button)
                bottom_layout.addWidget(self.close_button)

                # Add some stretch to the bottom layout to push the close button to the left
                bottom_layout.addStretch(1)

                # Add the bottom layout to the main layout
                self.verticalLayout.addLayout(bottom_layout)

                self.retranslateUi(Progress)
                QtCore.QMetaObject.connectSlotsByName(Progress)

            def retranslateUi(self, Progress):
                Progress.setWindowTitle("Form")
                # self.progress_header.setText("Timewarp ML")
                # self.progress_message.setText("Reading images....")

            def set_button_style(self, button):
                button.setMinimumSize(QtCore.QSize(150, 28))
                button.setMaximumSize(QtCore.QSize(150, 28))
                button.setFocusPolicy(QtCore.Qt.NoFocus)
                button.setStyleSheet('QPushButton {color: rgb(154, 154, 154); background-color: rgb(58, 58, 58); border: none; font: 14px}'
                'QPushButton:hover {border: 1px solid rgb(90, 90, 90)}'
                'QPushButton:pressed {color: rgb(159, 159, 159); background-color: rgb(66, 66, 66); border: 1px solid rgb(90, 90, 90)}'
                'QPushButton:disabled {color: rgb(116, 116, 116); background-color: rgb(58, 58, 58); border: none}'
                'QPushButton::menu-indicator {subcontrol-origin: padding; subcontrol-position: center right}'
                'QToolTip {color: rgb(170, 170, 170); background-color: rgb(71, 71, 71); border: 10px solid rgb(71, 71, 71)}')

        def __init__(self, selection, **kwargs):
            self.mode = kwargs.get('mode', 'Timewarp')

            try:
                H = selection[0].height
                W = selection[0].width
            except:
                W = 1280
                H = 720

            super().__init__()

            # now load in the UI that was created in the UI designer
            self.ui = self.Ui_Progress()
            self.ui.setupUi(self)

            self.ui.stripe_label.setText(self.mode)

            # Record the mouse position on a press event.
            self.mousePressPos = None

            # make it frameless and have it stay on top
            self.setWindowFlags(
                QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
            )


            desktop = QtWidgets.QApplication.desktop()
            screen_geometry = desktop.screenGeometry(desktop.primaryScreen())

            max_width = screen_geometry.width() * 0.8
            max_height = screen_geometry.height() * 0.8

            desired_width = W  # or whatever the aspect ratio calculation yields
            desired_height = 1.89 * H  # or whatever the aspect ratio calculation yields

            scale_factor = min(max_width / desired_width, max_height / desired_height)
            scaled_width = desired_width * scale_factor
            scaled_height = desired_height * scale_factor

            # Set the window's dimensions
            self.setGeometry(0, 0, scaled_width, scaled_height)
            # Move the window to the center of the screen
            screen_center = screen_geometry.center()
            self.move(screen_center.x() - scaled_width // 2, screen_center.y() - scaled_height // 2 - 100)
            self.show()

        def set_progress(self, pixmap):
            self.ui.label.setPixmap(pixmap)
            QtWidgets.QApplication.processEvents()

        def mousePressEvent(self, event):
            # Record the position at which the mouse was pressed.
            self.mousePressPos = event.globalPos()
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self.mousePressPos is not None:
                # Calculate the new position of the window.
                newPos = self.pos() + (event.globalPos() - self.mousePressPos)
                # Move the window to the new position.
                self.move(newPos)
                # Update the position at which the mouse was pressed.
                self.mousePressPos = event.globalPos()
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            self.mousePressPos = None
            super().mouseReleaseEvent(event)

        def closeEvent(self, event):
            self.deleteLater()
            super().closeEvent(event)

    def __init__(self, framework):
        flameMenuApp.__init__(self, framework)
        self.threads = True

        if not self.prefs.master.get(self.name):
            # set general defaults
            self.prefs['working_folder'] = '/var/tmp'
            self.prefs['slowmo_flow_scale'] = 1.0
            self.prefs['dedup_flow_scale'] = 1.0
            self.prefs['fluidmorph_flow_scale'] = 1.0
            self.prefs['fltw_flow_scale'] = 1.0

        if (self.prefs.get('version') != __version__) or not os.path.isdir(str(self.prefs.get('trained_models_folder', ''))):
            # set version-specific defaults
            self.prefs['trained_models_folder'] = os.path.join(
                self.framework.bundle_path,
                'trained_models', 'default', 'v2.3.model'
                )

        self.prefs['version'] = __version__
        self.framework.save_prefs()

        # Module defaults
        self.new_speed = 1
        self.dedup_mode = 0
        self.cpu = False
        self.flow_scale = 1.0

        self.flow_scale_list = {
            2.0:  'Analyze 2x Resolution',
            1.0:  'Analyze Full Resolution',
            0.5:  'Analyze 1/2 Resolution',
            0.25: 'Analyze 1/4 Resolution',
        }

        self.trained_models_path = os.path.join(
            self.framework.bundle_path,
            'trained_models', 
            'default',
        )

        # self.scan_trained_models_folder()

    def __getattr__(self, name):
        def method(*args, **kwargs):
            if name == 'Timewarp':
                return self.Progress(args[0], parent = self, mode = 'Timewarp')
        return method

    def build_menu(self):
        def scope_clip(selection):
            import flame
            for item in selection:
                if isinstance(item, (flame.PyClip)):
                    return True
            return False

        if not self.flame:
            return []
        
        if self.check_bundle_id:
            if not os.path.isfile(
                os.path.join(
                    self.framework.bundle_path,
                    'bundle_id')):
                return []
        
        menu = {'actions': []}
        menu['name'] = self.menu_group_name

        '''
        menu_item = {}
        menu_item['name'] = 'Slow Down clip(s) with ML'
        menu_item['execute'] = self.slowmo
        menu_item['isVisible'] = scope_clip
        menu_item['waitCursor'] = False
        menu['actions'].append(menu_item)

        menu_item = {}
        menu_item['name'] = 'Fill / Remove Duplicate Frames'
        menu_item['execute'] = self.dedup
        menu_item['isVisible'] = scope_clip
        menu_item['waitCursor'] = False
        menu['actions'].append(menu_item)

        menu_item = {}
        menu_item['name'] = 'Create Fluidmorph Transition'
        menu_item['execute'] = self.fluidmorph
        menu_item['isVisible'] = scope_clip
        menu_item['waitCursor'] = False
        menu['actions'].append(menu_item)
        '''
        
        menu_item = {}
        menu_item['name'] = "Timewarp"
        menu_item['execute'] = self.Timewarp
        menu_item['isVisible'] = scope_clip
        menu_item['waitCursor'] = False
        menu['actions'].append(menu_item)

        menu_item = {}
        menu_item['name'] = 'Version: ' + __version__
        menu_item['execute'] = self.slowmo
        menu_item['isEnabled'] = False
        menu_item['isVisible'] = scope_clip

        menu['actions'].append(menu_item)

        return menu

    def slowmo(self, selection):
        result = self.slowmo_dialog()
        if not result:
            return False

        working_folder = str(result.get('working_folder', '/var/tmp'))
        speed = result.get('speed', 1)
        flow_scale = result.get('flow_scale', 1.0)
        hold_konsole = result.get('hold_konsole', False)

        cmd_strings = []
        number_of_clips = 0

        import flame
        for item in selection:
            if isinstance(item, (flame.PyClip)):
                number_of_clips += 1

                clip = item
                clip_name = clip.name.get_value()

                result_folder = os.path.abspath(
                    os.path.join(
                        working_folder,
                        self.sanitized(clip_name) + '_TWML' + str(2 ** speed) + '_' + self.create_timestamp_uid()
                        )
                    )

                if os.path.isdir(result_folder):
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
                    cmd = 'rm -f ' + result_folder + '/*'
                    self.log('Executing command: %s' % cmd)
                    os.system(cmd)

                source_clip_folder = os.path.join(result_folder, 'source')
                if clip.bit_depth == 32:
                    export_preset = os.path.join(self.framework.bundle_path, 'openexr32bit.xml')
                    self.export_clip(clip, source_clip_folder, export_preset)
                else:
                    self.export_clip(clip, source_clip_folder)

                cmd_package = {}
                cmd_package['cmd_name'] = os.path.join(self.framework.bundle_path, 'inference_sequence.py')
                cmd_package['cpu'] = self.cpu
                
                cmd_quoted_args = {}
                cmd_quoted_args['input'] = source_clip_folder
                cmd_quoted_args['output'] = result_folder
                cmd_quoted_args['model'] = self.prefs.get('trained_models_folder')

                cmd_args = {}
                cmd_args['exp'] = str(speed)
                cmd_args['flow_scale'] = flow_scale
                cmd_args['bit_depth'] = clip.bit_depth

                cmd_package['quoted_args'] = cmd_quoted_args
                cmd_package['args'] = cmd_args

                lockfile_name = hashlib.sha1(result_folder.encode()).hexdigest().upper() + '.lock'
                lockfile_path = os.path.join(self.framework.bundle_path, 'locks', lockfile_name)

                try:
                    lockfile = open(lockfile_path, 'wb')
                    pickle.dump(cmd_package, lockfile)
                    lockfile.close()
                    if self.debug:
                        self.log('lockfile saved to %s' % lockfile_path)
                        self.log('lockfile contents:\n' + pformat(cmd_package))
                except Exception as e:
                    self.log('unable to save lockfile to %s' % lockfile_path)
                    self.log(e)

                cmd = 'python3 '
                cmd += os.path.join(self.framework.bundle_path, 'command_wrapper.py') + ' '
                cmd += lockfile_path
                cmd += "; "
                cmd_strings.append(cmd)
                
                new_clip_name = clip_name + '_TWML' + str(2 ** speed)
                watcher = threading.Thread(
                    target=self.import_watcher, 
                    args=(
                        result_folder, 
                        new_clip_name, 
                        clip.parent, 
                        [source_clip_folder],
                        lockfile_path
                        )
                    )
                watcher.daemon = True
                watcher.start()
                self.loops.append(watcher)

        self.refresh_x11_windows_list()
        
        if sys.platform == 'darwin':
            cmd_prefix = """tell application "Terminal" to activate do script "clear; """
            cmd_prefix += """/bin/bash -c 'eval " & quote & "$("""
            cmd_prefix += os.path.join(self.env_folder, 'bin', 'conda')
            cmd_prefix += """ shell.bash hook)" & quote & "; conda activate; """
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '
            
            ml_cmd = cmd_prefix
           
            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            ml_cmd += """'; exit" """

            import subprocess
            subprocess.Popen(['osascript', '-e', ml_cmd])
        
        elif self.gnome_terminal:
            cmd_prefix = 'gnome-terminal '
            cmd_prefix += """-- /bin/bash -c 'eval "$(""" + os.path.join(self.env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '

            ml_cmd = cmd_prefix
            ml_cmd += 'echo "Received ' + str(number_of_clips)
            ml_cmd += ' clip ' if number_of_clips < 2 else ' clips '
            ml_cmd += 'to process, press Ctrl+C to cancel"; '
            ml_cmd += 'trap exit SIGINT SIGTERM; '

            for cmd_string in cmd_strings:
                ml_cmd += cmd_string
            if hold_konsole:
                ml_cmd += 'echo "Commands finished. You can close this window"; sleep infinity'
            ml_cmd +="'"
            self.log('Executing command: %s' % ml_cmd)
            os.system(ml_cmd)

        else:
            cmd_prefix = 'konsole '
            if hold_konsole:
                cmd_prefix += '--hold '
            cmd_prefix += """-e /bin/bash -c 'eval "$(""" + os.path.join(self.env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '
            
            ml_cmd = cmd_prefix
            ml_cmd += 'echo "Received ' + str(number_of_clips)
            ml_cmd += ' clip ' if number_of_clips < 2 else ' clips '
            ml_cmd += 'to process, press Ctrl+C to cancel"; '
            ml_cmd += 'trap exit SIGINT SIGTERM; '

            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            if hold_konsole:
                ml_cmd += 'echo "Commands finished. You can close this window"'
            ml_cmd +="'"
            self.log('Executing command: %s' % ml_cmd)
            os.system(ml_cmd)

        flame.execute_shortcut('Refresh Thumbnails')
        self.raise_window_thread = threading.Thread(target=self.raise_last_window, args=())
        self.raise_window_thread.daemon = True
        self.raise_window_thread.start()

    def slowmo_dialog(self, *args, **kwargs):
        from PySide2 import QtWidgets, QtCore

        if not self.scan_trained_models_folder():
            return {}

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

        # Spacer
        lbl_Spacer = QtWidgets.QLabel('', window)
        lbl_Spacer.setStyleSheet('QFrame {color: #989898; background-color: #313131}')
        lbl_Spacer.setMinimumHeight(4)
        lbl_Spacer.setMaximumHeight(4)
        lbl_Spacer.setAlignment(QtCore.Qt.AlignCenter)


        vbox = QtWidgets.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignTop)

        # New Speed hbox
        new_speed_hbox = QtWidgets.QHBoxLayout()
        # new_speed_hbox.setAlignment(QtCore.Qt.AlignCenter)

        # New Speed label

        lbl_NewSpeed = QtWidgets.QLabel('New Speed ', window)
        lbl_NewSpeed.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_NewSpeed.setMinimumHeight(28)
        lbl_NewSpeed.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        new_speed_hbox.addWidget(lbl_NewSpeed)

        # New Speed Selector
        btn_NewSpeedSelector = QtWidgets.QPushButton(window)
        btn_NewSpeedSelector.setText(self.new_speed_list.get(self.new_speed))
        
        def selectNewSpeed(new_speed_id):
            self.new_speed = new_speed_id
            btn_NewSpeedSelector.setText(self.new_speed_list.get(self.new_speed))

        btn_NewSpeedSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_NewSpeedSelector.setMinimumSize(80, 28)
        btn_NewSpeedSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')
        btn_NewSpeedSelector_menu = QtWidgets.QMenu()

        for new_speed_id in sorted(self.new_speed_list.keys()):
            code = self.new_speed_list.get(new_speed_id, '1/2')
            action = btn_NewSpeedSelector_menu.addAction(code)            
            x = lambda chk=False, new_speed_id=new_speed_id: selectNewSpeed(new_speed_id)
            action.triggered[()].connect(x)

        btn_NewSpeedSelector.setMenu(btn_NewSpeedSelector_menu)
        new_speed_hbox.addWidget(btn_NewSpeedSelector)

        # Flow Scale Selector

        btn_FlowScaleSelector = QtWidgets.QPushButton(window)
        self.current_flow_scale = self.prefs.get('slowmo_flow_scale', 1.0)
        btn_FlowScaleSelector.setText(self.flow_scale_list.get(self.current_flow_scale))

        def selectFlowScale(flow_scale):
            self.current_flow_scale = flow_scale
            self.prefs['slowmo_flow_scale'] = flow_scale
            btn_FlowScaleSelector.setText(self.flow_scale_list.get(self.current_flow_scale))

        btn_FlowScaleSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_FlowScaleSelector.setMinimumSize(180, 28)
        btn_FlowScaleSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')
        btn_FlowScaleSelector_menu = QtWidgets.QMenu()
        for flow_scale in sorted(self.flow_scale_list.keys(), reverse=True):
            code = self.flow_scale_list.get(flow_scale, 1.0)
            action = btn_FlowScaleSelector_menu.addAction(code)            
            x = lambda chk=False, flow_scale=flow_scale: selectFlowScale(flow_scale)
            action.triggered[()].connect(x)

        btn_FlowScaleSelector.setMenu(btn_FlowScaleSelector_menu)
        new_speed_hbox.addWidget(btn_FlowScaleSelector)

        # Cpu Proc button

        if not sys.platform == 'darwin':            
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
            if self.cpu:
                btn_CpuProc.setStyleSheet('QPushButton {font:italic; background-color: #4f4f4f; color: #d9d9d9; border-top: 1px inset black; border-bottom: 1px inset #555555}')
            else:
                btn_CpuProc.setStyleSheet('QPushButton {color: #989898; background-color: #373737; border-top: 1px inset #555555; border-bottom: 1px inset black}')
            btn_CpuProc.pressed.connect(enableCpuProc)
            new_speed_hbox.addWidget(btn_CpuProc)

        ### Model Selector START

        current_model_name = self.model_map.get(self.prefs.get('trained_models_folder'))
        
        # Model Selector Button
        btn_ModelSelector = QtWidgets.QPushButton(window)
        btn_ModelSelector.setText(current_model_name)
        
        def selectModel(trained_models_folder):
            self.prefs['trained_models_folder'] = trained_models_folder
            btn_ModelSelector.setText(self.model_map.get(trained_models_folder))

        btn_ModelSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_ModelSelector.setMinimumSize(140, 28)
        btn_ModelSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')

        btn_ModelSelector_menu = QtWidgets.QMenu()
        for trained_models_folder in sorted(self.model_map.keys()):
            
            code = self.model_map.get(trained_models_folder)
            action = btn_ModelSelector_menu.addAction(code)
            x = lambda chk=False, trained_models_folder=trained_models_folder: selectModel(trained_models_folder)
            action.triggered[()].connect(x)
    
        btn_ModelSelector.setMenu(btn_ModelSelector_menu)
        new_speed_hbox.addWidget(btn_ModelSelector)

        ### Model Selector END

        vbox.addLayout(new_speed_hbox)
        vbox.addWidget(lbl_Spacer)

        # Work Folder

        def chooseFolder():
            result_folder = str(QtWidgets.QFileDialog.getExistingDirectory(window, "Open Directory", self.working_folder, QtWidgets.QFileDialog.ShowDirsOnly))
            if result_folder =='':
                return
            self.working_folder = result_folder
            txt_WorkFolder.setText(self.working_folder)
            self.prefs['working_folder'] = self.working_folder

        def txt_WorkFolder_textChanged():
            self.working_folder = txt_WorkFolder.text()

        # Work Folder Label

        lbl_WorkFolder = QtWidgets.QLabel('Export folder', window)
        lbl_WorkFolder.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_WorkFolder.setMinimumHeight(28)
        lbl_WorkFolder.setMaximumHeight(28)
        lbl_WorkFolder.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(lbl_WorkFolder)

        # Work Folder ENV Variable selector

        if os.getenv('FLAMETWML_WORK_FOLDER'):
            lbl_WorkFolderPath = QtWidgets.QLabel(self.working_folder, window)
            lbl_WorkFolderPath.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
            lbl_WorkFolderPath.setMinimumHeight(28)
            lbl_WorkFolderPath.setMaximumHeight(28)
            lbl_WorkFolderPath.setAlignment(QtCore.Qt.AlignCenter)
            vbox.addWidget(lbl_WorkFolderPath)

        else:
            # Work Folder Text Field
            hbox_workfolder = QtWidgets.QHBoxLayout()
            hbox_workfolder.setAlignment(QtCore.Qt.AlignLeft)

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
            if os.getenv('FLAMETWML_WORK_FOLDER'):
                self.working_folder = os.getenv('FLAMETWML_WORK_FOLDER')
                self.prefs['working_folder'] = os.getenv('FLAMETWML_WORK_FOLDER')

            modifiers = QtWidgets.QApplication.keyboardModifiers()
            self.framework.save_prefs()
            return {
                'speed': self.new_speed,
                'flow_scale': self.current_flow_scale,
                'working_folder': self.working_folder,
                'hold_konsole': True if modifiers == QtCore.Qt.ControlModifier else False
            }
        else:
            return {}

    def dedup(self, selection):
        result = self.dedup_dialog()
        if not result:
            return False

        working_folder = str(result.get('working_folder', '/var/tmp'))
        mode = result.get('mode', 0)
        flow_scale = result.get('flow_scale', 1.0)
        hold_konsole = result.get('hold_konsole', False)

        cmd_strings = []
        number_of_clips = 0

        import flame
        for item in selection:
            if isinstance(item, (flame.PyClip)):
                number_of_clips += 1

                clip = item
                clip_name = clip.name.get_value()
                
                result_folder = os.path.abspath(
                    os.path.join(
                        working_folder, 
                        self.sanitized(clip_name) + '_DUPFR' + '_' + self.create_timestamp_uid()
                        )
                    )

                if os.path.isdir(result_folder):
                    from PySide2 import QtWidgets
                    msg = 'Folder %s exists' % result_folder
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
                    cmd = 'rm -f ' + result_folder + '/*'
                    self.log('Executing command: %s' % cmd)
                    os.system(cmd)

                source_clip_folder = os.path.join(result_folder, 'source')
                if clip.bit_depth == 32:
                    export_preset = os.path.join(self.framework.bundle_path, 'openexr32bit.xml')
                    self.export_clip(clip, source_clip_folder, export_preset)
                else:
                    self.export_clip(clip, source_clip_folder)

                cmd_package = {}
                cmd_package['cmd_name'] = os.path.join(self.framework.bundle_path, 'inference_dpframes.py')
                cmd_package['cpu'] = self.cpu
                
                cmd_quoted_args = {}
                cmd_quoted_args['input'] = source_clip_folder
                cmd_quoted_args['output'] = result_folder
                cmd_quoted_args['model'] = self.prefs.get('trained_models_folder')

                cmd_args = {}
                cmd_args['flow_scale'] = flow_scale
                cmd_args['bit_depth'] = clip.bit_depth
                if mode:
                    cmd_args['remove'] = ''

                cmd_package['quoted_args'] = cmd_quoted_args
                cmd_package['args'] = cmd_args

                lockfile_name = hashlib.sha1(result_folder.encode()).hexdigest().upper() + '.lock'
                lockfile_path = os.path.join(self.framework.bundle_path, 'locks', lockfile_name)

                try:
                    lockfile = open(lockfile_path, 'wb')
                    pickle.dump(cmd_package, lockfile)
                    lockfile.close()
                    if self.debug:
                        self.log('lockfile saved to %s' % lockfile_path)
                        self.log('lockfile contents:\n' + pformat(cmd_package))
                except Exception as e:
                    self.log('unable to save lockfile to %s' % lockfile_path)
                    self.log(e)

                cmd = 'python3 '
                cmd += os.path.join(self.framework.bundle_path, 'command_wrapper.py') + ' '
                cmd += lockfile_path
                cmd += "; "
                cmd_strings.append(cmd)
                
                new_clip_name = clip_name + '_DUPFR'
                watcher = threading.Thread(
                    target=self.import_watcher, 
                    args=(
                        result_folder, 
                        new_clip_name, 
                        clip.parent, 
                        [source_clip_folder],
                        lockfile_path
                        )
                    )
                watcher.daemon = True
                watcher.start()
                self.loops.append(watcher)
        
        self.refresh_x11_windows_list()

        if sys.platform == 'darwin':
            cmd_prefix = """tell application "Terminal" to activate do script "clear; """
            # cmd_prefix += """ echo " & quote & "Received """
            # cmd_prefix += str(number_of_clips)
            #cmd_prefix += ' clip ' if number_of_clips < 2 else ' clips '
            # cmd_prefix += 'to process, press Ctrl+C to cancel" & quote &; '
            cmd_prefix += """/bin/bash -c 'eval " & quote & "$("""
            cmd_prefix += os.path.join(self.env_folder, 'bin', 'conda')
            cmd_prefix += """ shell.bash hook)" & quote & "; conda activate; """
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '
            
            ml_cmd = cmd_prefix
           
            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            ml_cmd += """'; exit" """

            import subprocess
            subprocess.Popen(['osascript', '-e', ml_cmd])

        elif self.gnome_terminal:
            cmd_prefix = 'gnome-terminal '
            cmd_prefix += """-- /bin/bash -c 'eval "$(""" + os.path.join(self.env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '

            ml_cmd = cmd_prefix
            ml_cmd += 'echo "Received ' + str(number_of_clips)
            ml_cmd += ' clip ' if number_of_clips < 2 else ' clips '
            ml_cmd += 'to process, press Ctrl+C to cancel"; '
            ml_cmd += 'trap exit SIGINT SIGTERM; '

            for cmd_string in cmd_strings:
                ml_cmd += cmd_string
            if hold_konsole:
                ml_cmd += 'echo "Commands finished. You can close this window"; sleep infinity'
            ml_cmd +="'"
            self.log('Executing command: %s' % ml_cmd)
            os.system(ml_cmd)

        else:
            cmd_prefix = 'konsole '
            if hold_konsole:
                cmd_prefix += '--hold '
            cmd_prefix += """-e /bin/bash -c 'eval "$(""" + os.path.join(self.env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '

            ml_cmd = cmd_prefix
            ml_cmd += 'echo "Received ' + str(number_of_clips)
            ml_cmd += ' clip ' if number_of_clips < 2 else ' clips '
            ml_cmd += 'to process, press Ctrl+C to cancel"; '
            ml_cmd += 'trap exit SIGINT SIGTERM; '

            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            ml_cmd += 'echo "Commands finished. You can close this window"'
            ml_cmd +="'"
            self.log('Executing command: %s' % ml_cmd)
            os.system(ml_cmd)

        flame.execute_shortcut('Refresh Thumbnails')
        self.raise_window_thread = threading.Thread(target=self.raise_last_window, args=())
        self.raise_window_thread.daemon = True
        self.raise_window_thread.start()

    def dedup_dialog(self, *args, **kwargs):
        from PySide2 import QtWidgets, QtCore

        self.scan_trained_models_folder()
        self.modes_list = {
            0: 'Interpolate',
            1: 'Remove', 
        }
        self.dedup_mode = self.prefs.get('dedup_mode', 0)
        
        window = QtWidgets.QDialog()
        window.setMinimumSize(280, 180)
        window.setWindowTitle('Remove duplicate frames')
        window.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        window.setStyleSheet('background-color: #313131')

        screen_res = QtWidgets.QDesktopWidget().screenGeometry()
        window.move((screen_res.width()/2)-150, (screen_res.height() / 2)-180)

        # Spacer
        lbl_Spacer = QtWidgets.QLabel('', window)
        lbl_Spacer.setStyleSheet('QFrame {color: #989898; background-color: #313131}')
        lbl_Spacer.setMinimumHeight(4)
        lbl_Spacer.setMaximumHeight(4)
        lbl_Spacer.setAlignment(QtCore.Qt.AlignCenter)


        vbox = QtWidgets.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignTop)
        
        # Duplicate frames action hbox
        dframes_hbox = QtWidgets.QHBoxLayout()
        # dframes_hbox.setAlignment(QtCore.Qt.AlignLeft)

        # Processing Mode Label

        lbl_Dfames = QtWidgets.QLabel('Duplicate frames: ', window)
        lbl_Dfames.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_Dfames.setMinimumHeight(28)
        lbl_Dfames.setAlignment(QtCore.Qt.AlignCenter)
        dframes_hbox.addWidget(lbl_Dfames)

        # Processing Mode Selector

        btn_DfamesSelector = QtWidgets.QPushButton(window)
        btn_DfamesSelector.setText(self.modes_list.get(self.dedup_mode))
        def selectNewMode(new_mode_id):
            self.dedup_mode = new_mode_id
            self.prefs['dedup_mode'] = new_mode_id
            self.framework.save_prefs()
            btn_DfamesSelector.setText(self.modes_list.get(self.dedup_mode))
        btn_DfamesSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_DfamesSelector.setMinimumSize(120, 28)
        btn_DfamesSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')
        btn_DfamesSelector_menu = QtWidgets.QMenu()

        for new_mode_id in sorted(self.modes_list.keys()):
            code = self.modes_list.get(new_mode_id, 'Interpolate')
            action = btn_DfamesSelector_menu.addAction(code)
            x = lambda chk=False, new_mode_id=new_mode_id: selectNewMode(new_mode_id)
            action.triggered[()].connect(x)

        btn_DfamesSelector.setMenu(btn_DfamesSelector_menu)
        dframes_hbox.addWidget(btn_DfamesSelector)

        # Flow Scale Selector

        btn_FlowScaleSelector = QtWidgets.QPushButton(window)
        self.current_flow_scale = self.prefs.get('dedup_flow_scale', 1.0)
        btn_FlowScaleSelector.setText(self.flow_scale_list.get(self.current_flow_scale))

        def selectFlowScale(flow_scale):
            self.current_flow_scale = flow_scale
            self.prefs['dedup_flow_scale'] = flow_scale
            btn_FlowScaleSelector.setText(self.flow_scale_list.get(self.current_flow_scale))

        btn_FlowScaleSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_FlowScaleSelector.setMinimumSize(180, 28)
        btn_FlowScaleSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')
        btn_FlowScaleSelector_menu = QtWidgets.QMenu()
        for flow_scale in sorted(self.flow_scale_list.keys(), reverse=True):
            code = self.flow_scale_list.get(flow_scale, 1.0)
            action = btn_FlowScaleSelector_menu.addAction(code)            
            x = lambda chk=False, flow_scale=flow_scale: selectFlowScale(flow_scale)
            action.triggered[()].connect(x)

        btn_FlowScaleSelector.setMenu(btn_FlowScaleSelector_menu)
        dframes_hbox.addWidget(btn_FlowScaleSelector)

        # Cpu Proc button

        if not sys.platform == 'darwin':            
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

            dframes_hbox.addWidget(btn_CpuProc)

        ### Model Selector START

        current_model_name = self.model_map.get(self.prefs.get('trained_models_folder'), 'Unknown')
        
        # Model Selector Button
        btn_ModelSelector = QtWidgets.QPushButton(window)
        btn_ModelSelector.setText(current_model_name)
        
        def selectModel(trained_models_folder):
            self.prefs['trained_models_folder'] = trained_models_folder
            btn_ModelSelector.setText(self.model_map.get(trained_models_folder))

        btn_ModelSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_ModelSelector.setMinimumSize(140, 28)
        btn_ModelSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')

        btn_ModelSelector_menu = QtWidgets.QMenu()
        for trained_models_folder in sorted(self.model_map.keys()):
            
            code = self.model_map.get(trained_models_folder)
            action = btn_ModelSelector_menu.addAction(code)
            x = lambda chk=False, trained_models_folder=trained_models_folder: selectModel(trained_models_folder)
            action.triggered[()].connect(x)
    
        btn_ModelSelector.setMenu(btn_ModelSelector_menu)
        dframes_hbox.addWidget(btn_ModelSelector)

        ### Model Selector END

        vbox.addLayout(dframes_hbox)
        vbox.addWidget(lbl_Spacer)

        # Work Folder

        def chooseFolder():
            result_folder = str(QtWidgets.QFileDialog.getExistingDirectory(window, "Open Directory", self.working_folder, QtWidgets.QFileDialog.ShowDirsOnly))
            if result_folder =='':
                return
            self.working_folder = result_folder
            txt_WorkFolder.setText(self.working_folder)
            self.prefs['working_folder'] = self.working_folder

        def txt_WorkFolder_textChanged():
            self.working_folder = txt_WorkFolder.text()

        # Work Folder Label

        lbl_WorkFolder = QtWidgets.QLabel('Export folder', window)
        lbl_WorkFolder.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_WorkFolder.setMinimumHeight(28)
        lbl_WorkFolder.setMaximumHeight(28)
        lbl_WorkFolder.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(lbl_WorkFolder)

        # Work Folder ENV Variable selector

        if os.getenv('FLAMETWML_WORK_FOLDER'):
            lbl_WorkFolderPath = QtWidgets.QLabel(self.working_folder, window)
            lbl_WorkFolderPath.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
            lbl_WorkFolderPath.setMinimumHeight(28)
            lbl_WorkFolderPath.setMaximumHeight(28)
            lbl_WorkFolderPath.setAlignment(QtCore.Qt.AlignCenter)
            vbox.addWidget(lbl_WorkFolderPath)

        else:
            # Work Folder Text Field
            hbox_workfolder = QtWidgets.QHBoxLayout()
            hbox_workfolder.setAlignment(QtCore.Qt.AlignLeft)

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
            if os.getenv('FLAMETWML_WORK_FOLDER'):
                self.working_folder = os.getenv('FLAMETWML_WORK_FOLDER')
                self.prefs['working_folder'] = os.getenv('FLAMETWML_WORK_FOLDER')

            modifiers = QtWidgets.QApplication.keyboardModifiers()
            self.framework.save_prefs()
            return {
                'mode': self.dedup_mode,
                'working_folder': self.working_folder,
                'flow_scale': self.current_flow_scale,
                'hold_konsole': True if modifiers == QtCore.Qt.ControlModifier else False
            }
        else:
            return {}

    def fluidmorph(self, selection):
        def usage_message():
            from PySide2 import QtWidgets, QtCore
            msg = 'Please select two clips of the same dimentions and length'
            mbox = QtWidgets.QMessageBox()
            mbox.setWindowTitle('flameTimewrarpML')
            mbox.setText(msg)
            mbox.exec_()

        import flame
        clips = []
        for item in selection:
            if isinstance(item, (flame.PyClip)):
                clips.append(item)

        if len(clips) != 2:
            usage_message()
            return
                
        result = self.fluidmorph_dialog(clips = clips)
        if not result:
            return False

        working_folder = str(result.get('working_folder', '/var/tmp'))
        incoming_clip = clips[result.get('incoming')]
        outgoing_clip = clips[result.get('outgoing')]
        flow_scale = result.get('flow_scale', 1.0)
        hold_konsole = result.get('hold_konsole', False)
        cmd_strings = []

        incoming_clip_name = incoming_clip.name.get_value()
        outgoing_clip_name = outgoing_clip.name.get_value()
        result_folder = os.path.abspath(
            os.path.join(
                working_folder, 
                self.sanitized(incoming_clip_name) + '_FLUID' + '_' + self.create_timestamp_uid()
                )
            )

        if os.path.isdir(result_folder):
            from PySide2 import QtWidgets
            msg = 'Folder %s exists' % result_folder
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
            cmd = 'rm -f ' + result_folder + '/*'
            self.log('Executing command: %s' % cmd)
            os.system(cmd)

        incoming_folder = os.path.join(result_folder, 'incoming')
        outgoing_folder = os.path.join(result_folder, 'outgoing')
        if incoming_clip.bit_depth == 32:
            export_preset = os.path.join(self.framework.bundle_path, 'openexr32bit.xml')
            self.export_clip(incoming_clip, incoming_folder, export_preset)
            self.export_clip(outgoing_clip, outgoing_folder, export_preset)
        else:
            self.export_clip(incoming_clip, incoming_folder)
            self.export_clip(outgoing_clip, outgoing_folder)

        cmd_package = {}
        cmd_package['cmd_name'] = os.path.join(self.framework.bundle_path, 'inference_fluidmorph.py')
        cmd_package['cpu'] = self.cpu
        
        cmd_quoted_args = {}
        cmd_quoted_args['incoming'] = incoming_folder
        cmd_quoted_args['outgoing'] = outgoing_folder
        cmd_quoted_args['output'] = result_folder
        cmd_quoted_args['model'] = self.prefs.get('trained_models_folder')

        cmd_args = {}
        cmd_args['flow_scale'] = flow_scale
        cmd_args['bit_depth'] = incoming_clip.bit_depth

        cmd_package['quoted_args'] = cmd_quoted_args
        cmd_package['args'] = cmd_args

        lockfile_name = hashlib.sha1(result_folder.encode()).hexdigest().upper() + '.lock'
        lockfile_path = os.path.join(self.framework.bundle_path, 'locks', lockfile_name)

        try:
            lockfile = open(lockfile_path, 'wb')
            pickle.dump(cmd_package, lockfile)
            lockfile.close()
            if self.debug:
                self.log('lockfile saved to %s' % lockfile_path)
                self.log('lockfile contents:\n' + pformat(cmd_package))
        except Exception as e:
            self.log('unable to save lockfile to %s' % lockfile_path)
            self.log(e)

        cmd = 'python3 '
        cmd += os.path.join(self.framework.bundle_path, 'command_wrapper.py') + ' '
        cmd += lockfile_path
        cmd += "; "
        cmd_strings.append(cmd)
        
        new_clip_name = incoming_clip_name + '_FLUID'
        watcher = threading.Thread(
            target=self.import_watcher, 
            args=(
                result_folder, 
                new_clip_name, 
                incoming_clip.parent, 
                [incoming_folder, outgoing_folder],
                lockfile_path
                )
            )
        watcher.daemon = True
        watcher.start()
        self.loops.append(watcher)

        self.refresh_x11_windows_list()

        if sys.platform == 'darwin':
            cmd_prefix = """tell application "Terminal" to activate do script "clear; """
            # cmd_prefix += """ echo " & quote & "Received """
            # cmd_prefix += str(number_of_clips)
            #cmd_prefix += ' clip ' if number_of_clips < 2 else ' clips '
            # cmd_prefix += 'to process, press Ctrl+C to cancel" & quote &; '
            cmd_prefix += """/bin/bash -c 'eval " & quote & "$("""
            cmd_prefix += os.path.join(self.env_folder, 'bin', 'conda')
            cmd_prefix += """ shell.bash hook)" & quote & "; conda activate; """
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '
            
            ml_cmd = cmd_prefix
           
            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            ml_cmd += """'; exit" """

            import subprocess
            subprocess.Popen(['osascript', '-e', ml_cmd])

        elif self.gnome_terminal:
            cmd_prefix = 'gnome-terminal '
            cmd_prefix += """-- /bin/bash -c 'eval "$(""" + os.path.join(self.env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '
            ml_cmd = cmd_prefix
            # ml_cmd += 'echo "Received ' + str(number_of_clips)
            # ml_cmd += ' clip ' if number_of_clips < 2 else ' clips '
            # ml_cmd += 'to process, press Ctrl+C to cancel"; '
            ml_cmd += 'trap exit SIGINT SIGTERM; '

            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            if hold_konsole:
                ml_cmd += 'echo "Commands finished. You can close this window"; sleep infinity'
            ml_cmd +="'"
            self.log('Executing command: %s' % ml_cmd)
            os.system(ml_cmd)

        else:
            cmd_prefix = 'konsole '
            if hold_konsole:
                cmd_prefix += '--hold '
            cmd_prefix += """-e /bin/bash -c 'eval "$(""" + os.path.join(self.env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '

            ml_cmd = cmd_prefix
            # ml_cmd += 'echo "Received ' + str(number_of_clips)
            # ml_cmd += ' clip ' if number_of_clips < 2 else ' clips '
            # ml_cmd += 'to process, press Ctrl+C to cancel"; '
            ml_cmd += 'trap exit SIGINT SIGTERM; '

            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            if hold_konsole:
                ml_cmd += 'echo "Commands finished. You can close this window"'
            ml_cmd +="'"
            self.log('Executing command: %s' % ml_cmd)
            os.system(ml_cmd)

        flame.execute_shortcut('Refresh Thumbnails')
        self.raise_window_thread = threading.Thread(target=self.raise_last_window, args=())
        self.raise_window_thread.daemon = True
        self.raise_window_thread.start()

    def fluidmorph_dialog(self, *args, **kwargs):
        from PySide2 import QtWidgets, QtCore
        
        self.scan_trained_models_folder()

        clips = kwargs.get('clips')
        self.incoming_clip_id = 0
        self.outgoing_clip_id = 1
        
        self.clip_names_list = {
            0: clips[0].name.get_value(),
            1: clips[1].name.get_value(), 
        }

        pprint (self.clip_names_list)

        window = QtWidgets.QDialog()
        window.setMinimumSize(280, 180)
        window.setWindowTitle('Create Fluidmorph Transition')
        window.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        window.setStyleSheet('background-color: #313131')

        screen_res = QtWidgets.QDesktopWidget().screenGeometry()
        window.move((screen_res.width()/2)-150, (screen_res.height() / 2)-180)

        # Spacer
        lbl_Spacer = QtWidgets.QLabel('', window)
        lbl_Spacer.setStyleSheet('QFrame {color: #989898; background-color: #313131}')
        lbl_Spacer.setMinimumHeight(4)
        lbl_Spacer.setMaximumHeight(4)
        lbl_Spacer.setAlignment(QtCore.Qt.AlignCenter)


        vbox = QtWidgets.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignTop)
        
        '''
        # CLIP order indicator label
        lbl_text = 'Transition: '
        lbl_text += self.clip_names_list.get(self.incoming_clip_id) + ' -> ' + self.clip_names_list.get(self.outgoing_clip_id)
        lbl_ClipOrder = QtWidgets.QLabel(lbl_text, window)
        lbl_ClipOrder.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_ClipOrder.setMinimumHeight(28)
        lbl_ClipOrder.setMaximumHeight(28)
        lbl_ClipOrder.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(lbl_ClipOrder)
        '''

        # Duplicate frames action hbox
        dframes_hbox = QtWidgets.QHBoxLayout()
        # dframes_hbox.setAlignment(QtCore.Qt.AlignLeft)

        # Processing Mode Label

        lbl_Dfames = QtWidgets.QLabel('Start from: ', window)
        lbl_Dfames.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_Dfames.setMinimumHeight(28)
        lbl_Dfames.setAlignment(QtCore.Qt.AlignCenter)
        dframes_hbox.addWidget(lbl_Dfames)

        # Processing Mode Selector

        btn_DfamesSelector = QtWidgets.QPushButton(window)
        btn_DfamesSelector.setText(self.clip_names_list.get(self.incoming_clip_id))
        def selectNewMode(new_incoming_id):
            self.outgoing_clip_id = self.incoming_clip_id
            self.incoming_clip_id = new_incoming_id
            btn_DfamesSelector.setText(self.clip_names_list.get(new_incoming_id))
            lbl_text = self.clip_names_list.get(self.incoming_clip_id) + ' -> ' + self.clip_names_list.get(self.outgoing_clip_id)
            # lbl_ClipOrder.setText(lbl_text)
        btn_DfamesSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_DfamesSelector.setMinimumSize(120, 28)
        btn_DfamesSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')
        btn_DfamesSelector_menu = QtWidgets.QMenu()

        for new_incoming_id in sorted(self.clip_names_list.keys()):
            name = self.clip_names_list.get(new_incoming_id)
            action = btn_DfamesSelector_menu.addAction(name)
            x = lambda chk=False, new_incoming_id=new_incoming_id: selectNewMode(new_incoming_id)
            action.triggered[()].connect(x)
        btn_DfamesSelector.setMenu(btn_DfamesSelector_menu)
        dframes_hbox.addWidget(btn_DfamesSelector)

        # Flow Scale Selector

        btn_FlowScaleSelector = QtWidgets.QPushButton(window)
        self.current_flow_scale = self.prefs.get('fluidmorph_flow_scale', 1.0)
        btn_FlowScaleSelector.setText(self.flow_scale_list.get(self.current_flow_scale))

        def selectFlowScale(flow_scale):
            self.current_flow_scale = flow_scale
            self.prefs['fluidmorph_flow_scale'] = flow_scale
            btn_FlowScaleSelector.setText(self.flow_scale_list.get(self.current_flow_scale))

        btn_FlowScaleSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_FlowScaleSelector.setMinimumSize(180, 28)
        btn_FlowScaleSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')
        btn_FlowScaleSelector_menu = QtWidgets.QMenu()
        for flow_scale in sorted(self.flow_scale_list.keys(), reverse=True):
            code = self.flow_scale_list.get(flow_scale, 1.0)
            action = btn_FlowScaleSelector_menu.addAction(code)            
            x = lambda chk=False, flow_scale=flow_scale: selectFlowScale(flow_scale)
            action.triggered[()].connect(x)

        btn_FlowScaleSelector.setMenu(btn_FlowScaleSelector_menu)
        dframes_hbox.addWidget(btn_FlowScaleSelector)

        # Cpu Proc button

        if not sys.platform == 'darwin':            
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

            dframes_hbox.addWidget(btn_CpuProc)

        ### Model Selector START

        current_model_name = self.model_map.get(self.prefs.get('trained_models_folder'), 'Unknown')
        
        # Model Selector Button
        btn_ModelSelector = QtWidgets.QPushButton(window)
        btn_ModelSelector.setText(current_model_name)
        
        def selectModel(trained_models_folder):
            self.prefs['trained_models_folder'] = trained_models_folder
            btn_ModelSelector.setText(self.model_map.get(trained_models_folder))

        btn_ModelSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_ModelSelector.setMinimumSize(140, 28)
        btn_ModelSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')

        btn_ModelSelector_menu = QtWidgets.QMenu()
        for trained_models_folder in sorted(self.model_map.keys()):
            
            code = self.model_map.get(trained_models_folder)
            action = btn_ModelSelector_menu.addAction(code)
            x = lambda chk=False, trained_models_folder=trained_models_folder: selectModel(trained_models_folder)
            action.triggered[()].connect(x)
    
        btn_ModelSelector.setMenu(btn_ModelSelector_menu)
        dframes_hbox.addWidget(btn_ModelSelector)

        ### Model Selector END

        vbox.addLayout(dframes_hbox)
        vbox.addWidget(lbl_Spacer)

        # Work Folder

        def chooseFolder():
            result_folder = str(QtWidgets.QFileDialog.getExistingDirectory(window, "Open Directory", self.working_folder, QtWidgets.QFileDialog.ShowDirsOnly))
            if result_folder =='':
                return
            self.working_folder = result_folder
            txt_WorkFolder.setText(self.working_folder)
            self.prefs['working_folder'] = self.working_folder

        def txt_WorkFolder_textChanged():
            self.working_folder = txt_WorkFolder.text()

        # Work Folder Label

        lbl_WorkFolder = QtWidgets.QLabel('Export folder', window)
        lbl_WorkFolder.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_WorkFolder.setMinimumHeight(28)
        lbl_WorkFolder.setMaximumHeight(28)
        lbl_WorkFolder.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(lbl_WorkFolder)

        # Work Folder ENV Variable selector

        if os.getenv('FLAMETWML_WORK_FOLDER'):
            lbl_WorkFolderPath = QtWidgets.QLabel(self.working_folder, window)
            lbl_WorkFolderPath.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
            lbl_WorkFolderPath.setMinimumHeight(28)
            lbl_WorkFolderPath.setMaximumHeight(28)
            lbl_WorkFolderPath.setAlignment(QtCore.Qt.AlignCenter)
            vbox.addWidget(lbl_WorkFolderPath)

        else:
            # Work Folder Text Field
            hbox_workfolder = QtWidgets.QHBoxLayout()
            hbox_workfolder.setAlignment(QtCore.Qt.AlignLeft)

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
            if os.getenv('FLAMETWML_WORK_FOLDER'):
                self.working_folder = os.getenv('FLAMETWML_WORK_FOLDER')
                self.prefs['working_folder'] = os.getenv('FLAMETWML_WORK_FOLDER')

            modifiers = QtWidgets.QApplication.keyboardModifiers()
            self.framework.save_prefs()
            return {
                'incoming': self.incoming_clip_id,
                'outgoing': self.outgoing_clip_id,
                'working_folder': self.working_folder,
                'flow_scale': self.current_flow_scale,
                'hold_konsole': True if modifiers == QtCore.Qt.ControlModifier else False
            }
        else:
            return {}

    def fltw(self, selection):
        import flame

        sys.path.insert(0, self.framework.site_packages_folder)
        import numpy as np
        del sys.path[0]
        from PySide2 import QtWidgets, QtCore, QtGui
        class WireTapException(Exception):
            def __init__(self, msg):
                flame.messages.show_in_dialog(
                    title = 'flameTimewrarpML',
                    message = msg,
                    type = 'error',
                    buttons = ['Ok']
                )

        def sequence_message():
            from PySide2 import QtWidgets, QtCore
            msg = 'Please select single-track clips with no versions or edits'
            mbox = QtWidgets.QMessageBox()
            mbox.setWindowTitle('flameTimewrarpML')
            mbox.setText(msg)
            mbox.exec_()
        
        def effect_message():
            flame.messages.show_in_dialog(
                title = 'flameTimewrarpML',
                message = 'Please select clips with Timewarp Timeline FX',
                type = 'error',
                buttons = ['Ok']
            )

        def parse_message(e):
            from PySide2 import QtWidgets, QtCore
            import traceback
            msg = 'Error parsing TW setup file: ' + pformat(e)
            mbox = QtWidgets.QMessageBox()
            mbox.setWindowTitle('flameTimewrarpML')
            mbox.setText(msg)
            mbox.setDetailedText(pformat(traceback.format_exc()))
            mbox.setStyleSheet('QLabel{min-width: 800px;}')
            mbox.exec_()

        def dictify(r, root=True):
            from copy import copy

            if root:
                return {r.tag : dictify(r, False)}

            d = copy(r.attrib)
            if r.text:
                d["_text"]=r.text
            for x in r.findall("./*"):
                if x.tag not in d:
                    d[x.tag]=[]
                d[x.tag].append(dictify(x,False))
            return d
        
        verified_clips = []
        temp_setup_path = '/var/tmp/temporary_tw_setup.timewarp_node'

        progress = self.publish_progress_dialog()
        progress.show()

        if not selection:
            sequence_message()
            return
        clip = selection[0]
        if not isinstance(clip, (flame.PyClip)):
            sequence_message()
            return
        if len(clip.versions) != 1:
            sequence_message()
            return
        if len (clip.versions[0].tracks) != 1:
            sequence_message()
            return
        if len (clip.versions[0].tracks[0].segments) != 1:
            sequence_message()

        clip_matched = (clip.versions[0].tracks[0].segments[0].match(clip.parent, include_timeline_fx = False))
        clip_matched.commit()

        # Initialize the Wiretap Client API.
        #
        wiretap_client = WireTapClient()
        if not wiretap_client.init():
            raise WireTapException("Unable to initialize Wiretap client API.")

        server_handle = WireTapServerHandle('localhost')
        clip_node_id = flame.PyClip.get_wiretap_node_id(clip_matched)
        clip_node_handle = WireTapNodeHandle(server_handle, clip_node_id)
        num_frames = WireTapInt()

        if not clip_node_handle.getNumFrames(num_frames):
            raise WireTapException(
                "Unable to obtain number of frames: %s." % clip_node_handle.lastError()
            )
        
        fmt = WireTapClipFormat()
        if not clip_node_handle.getClipFormat(fmt):
            raise WireTapException("Unable to obtain clip format: %s." % clip.lastError())
        
        pprint (fmt.formatTag())
        pprint (fmt.bitsPerPixel())
        print  (fmt.numChannels())
        
        time.sleep(2)
        return

        buff = "0" * fmt.frameBufferSize()

        for frame_number in range(0, num_frames):
            print("Reading frame %i." % frame_number)
            if not clip_node_handle.readFrame(frame_number, buff, fmt.frameBufferSize()):
                raise WireTapException(
                    "Unable to obtain read frame %i: %s." % (frame_number, clip.lastError())
                )
            print("Successfully read frame %i." % frame_number)

            buff_tail = frameBufferSize() - fmt.height() * fmt.width() * (fmt.bitsPerPixel()/8)
            arr = np.frombuffer(buff.encode(), dtype=np.float16)[:-8]
            arr = arr.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
            frame = ((arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255).astype(np.uint8)
            resized_img = resize_nearest(frame, (800, 600))
            resized_img = np.flip(resized_img, axis=0)
            # resized_img = np.flip(resized_img, axis=1)
            resized_img = np.flip(resized_img, axis=2)
            resized_img = np.ascontiguousarray(resized_img)
            height, width, channels = resized_img.shape
            bytesPerLine = channels * width
            img = QtGui.QImage(resized_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap = QtGui.QPixmap.fromImage(img)
            progress.set_progress(pixmap)
            # lbl.setPixmap(pixmap)

            if not new_clip_node_handle.writeFrame(
                frame_number, buff, new_fmt.frameBufferSize()
            ):
                raise WireTapException(
                    "Unable to obtain write frame %i: %s."
                    % (frame_number, clip_node_handle.lastError())
                )
            print("Successfully wrote frame %i." % frame_number)

        library.acquire_exclusive_access()
        library.open()
        if library.clips:
            flame.media_panel.move(source_entries = library.clips[0], destination = clip.parent, duplicate_action = 'add')
        flame.delete(library)
        flame.delete(clip_matched)

        progress.hide()

        effects = clip.versions[0].tracks[0].segments[0].effects
        if not effects:
            # effect_message()
            return
        


        return

        verified_clips.append((clip, tw_setup_string))
        
        os.remove(temp_setup_path)

        result = self.fltw_dialog()
        if not result:
            return False

        working_folder = str(result.get('working_folder', '/var/tmp'))
        # speed = result.get('speed', 1)
        flow_scale = result.get('flow_scale', 1.0)
        hold_konsole = result.get('hold_konsole', False)

        cmd_strings = []
        number_of_clips = 0

        for clip, tw_setup_string in verified_clips:
            number_of_clips += 1
            clip_name = clip.name.get_value()

            result_folder = os.path.abspath(
                os.path.join(
                    working_folder, 
                    self.sanitized(clip_name) + '_TWML' + '_' + self.create_timestamp_uid()
                    )
                )

            if os.path.isdir(result_folder):
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
                cmd = 'rm -f ' + result_folder + '/*'
                self.log('Executing command: %s' % cmd)
                os.system(cmd)

            clip.render()
            
            source_clip_folder = os.path.join(result_folder, 'source')
            
            if clip.bit_depth == 32:
                export_preset = os.path.join(self.framework.bundle_path, 'source_export32.xml')
            else:
                export_preset = os.path.join(self.framework.bundle_path, 'source_export.xml')

            tw_setup_path = os.path.join(source_clip_folder, 'tw_setup.timewarp_node')
            self.export_clip(clip, source_clip_folder, export_preset)
            with open(tw_setup_path, 'a') as tw_setup_file:
                tw_setup_file.write(tw_setup_string)
                tw_setup_file.close()

            '''
            seg_data = {}
            seg_data['record_duration'] = clip.versions[0].tracks[0].segments[0].record_duration.relative_frame
            seg_data['record_in'] = clip.versions[0].tracks[0].segments[0].record_in.relative_frame
            seg_data['record_out'] = clip.versions[0].tracks[0].segments[0].record_out.relative_frame
            seg_data['source_duration'] = clip.versions[0].tracks[0].segments[0].source_duration.relative_frame
            seg_data['source_in'] = clip.versions[0].tracks[0].segments[0].source_in.relative_frame
            seg_data['source_out'] = clip.versions[0].tracks[0].segments[0].source_out.relative_frame
            pprint (seg_data)
            '''

            record_in = clip.versions[0].tracks[0].segments[0].record_in.relative_frame
            record_out = clip.versions[0].tracks[0].segments[0].record_out.relative_frame

            cmd_package = {}
            cmd_package['cmd_name'] = os.path.join(self.framework.bundle_path, 'inference_flame_tw.py')
            cmd_package['cpu'] = self.cpu
            
            cmd_quoted_args = {}
            cmd_quoted_args['input'] = source_clip_folder
            cmd_quoted_args['output'] = result_folder
            cmd_quoted_args['model'] = self.prefs.get('trained_models_folder')
            cmd_quoted_args['setup'] = tw_setup_path

            cmd_args = {}
            cmd_args['record_in'] = record_in
            cmd_args['record_out'] = record_out
            cmd_args['flow_scale'] = flow_scale
            cmd_args['bit_depth'] = clip.bit_depth

            cmd_package['quoted_args'] = cmd_quoted_args
            cmd_package['args'] = cmd_args

            lockfile_name = hashlib.sha1(result_folder.encode()).hexdigest().upper() + '.lock'
            lockfile_path = os.path.join(self.framework.bundle_path, 'locks', lockfile_name)

            try:
                lockfile = open(lockfile_path, 'wb')
                pickle.dump(cmd_package, lockfile)
                lockfile.close()
                if self.debug:
                    self.log('lockfile saved to %s' % lockfile_path)
                    self.log('lockfile contents:\n' + pformat(cmd_package))
            except Exception as e:
                self.log('unable to save lockfile to %s' % lockfile_path)
                self.log(e)

            cmd = 'python3 '
            cmd += os.path.join(self.framework.bundle_path, 'command_wrapper.py') + ' '
            cmd += lockfile_path
            cmd += "; "
            cmd_strings.append(cmd)
            
            new_clip_name = clip_name + '_TWML'
            watcher = threading.Thread(
                target=self.import_watcher, 
                args=(
                    result_folder, 
                    new_clip_name, 
                    clip.parent, 
                    [source_clip_folder],
                    lockfile_path
                    )
                )
            watcher.daemon = True
            watcher.start()
            self.loops.append(watcher)

        self.refresh_x11_windows_list()

        if sys.platform == 'darwin':
            cmd_prefix = """tell application "Terminal" to activate do script "clear; """
            # cmd_prefix += """ echo " & quote & "Received """
            # cmd_prefix += str(number_of_clips)
            #cmd_prefix += ' clip ' if number_of_clips < 2 else ' clips '
            # cmd_prefix += 'to process, press Ctrl+C to cancel" & quote &; '
            cmd_prefix += """/bin/bash -c 'eval " & quote & "$("""
            cmd_prefix += os.path.join(self.env_folder, 'bin', 'conda')
            cmd_prefix += """ shell.bash hook)" & quote & "; conda activate; """
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '
            
            ml_cmd = cmd_prefix
           
            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            ml_cmd += """'; exit" """

            import subprocess
            subprocess.Popen(['osascript', '-e', ml_cmd])

        elif self.gnome_terminal:
            cmd_prefix = 'gnome-terminal '
            cmd_prefix += """-- /bin/bash -c 'eval "$(""" + os.path.join(self.env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '
            ml_cmd = cmd_prefix
            ml_cmd += 'echo "Received ' + str(number_of_clips)
            ml_cmd += ' clip ' if number_of_clips < 2 else ' clips '
            ml_cmd += 'to process, press Ctrl+C to cancel"; '
            ml_cmd += 'trap exit SIGINT SIGTERM; '

            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            if hold_konsole:
                ml_cmd += 'echo "Commands finished. You can close this window"; sleep infinity'
            ml_cmd +="'"
            self.log('Executing command: %s' % ml_cmd)
            os.system(ml_cmd)

        else:
            cmd_prefix = 'konsole '
            if hold_konsole:
                cmd_prefix += '--hold '
            cmd_prefix += """-e /bin/bash -c 'eval "$(""" + os.path.join(self.env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
            cmd_prefix += 'cd ' + self.framework.bundle_path + '; '

            ml_cmd = cmd_prefix
            ml_cmd += 'echo "Received ' + str(number_of_clips)
            ml_cmd += ' clip ' if number_of_clips < 2 else ' clips '
            ml_cmd += 'to process, press Ctrl+C to cancel"; '
            ml_cmd += 'trap exit SIGINT SIGTERM; '

            for cmd_string in cmd_strings:
                ml_cmd += cmd_string

            if hold_konsole:
                ml_cmd += 'echo "Commands finished. You can close this window"'
            ml_cmd +="'"
            self.log('Executing command: %s' % ml_cmd)
            os.system(ml_cmd)

        flame.execute_shortcut('Refresh Thumbnails')
        self.raise_window_thread = threading.Thread(target=self.raise_last_window, args=())
        self.raise_window_thread.daemon = True
        self.raise_window_thread.start()

    def fltw_dialog(self, *args, **kwargs):
        from PySide2 import QtWidgets, QtCore
        
        self.scan_trained_models_folder()

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

        # Spacer
        lbl_Spacer = QtWidgets.QLabel('', window)
        lbl_Spacer.setStyleSheet('QFrame {color: #989898; background-color: #313131}')
        lbl_Spacer.setMinimumHeight(2)
        lbl_Spacer.setMaximumHeight(2)
        lbl_Spacer.setAlignment(QtCore.Qt.AlignCenter)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignTop)

        # New Speed hbox
        new_speed_hbox = QtWidgets.QHBoxLayout()
        # new_speed_hbox.setAlignment(QtCore.Qt.AlignCenter)

        # New Speed label

        lbl_NewSpeed = QtWidgets.QLabel('Processing Options: ', window)
        lbl_NewSpeed.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_NewSpeed.setMinimumHeight(28)
        lbl_NewSpeed.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        new_speed_hbox.addWidget(lbl_NewSpeed)

        # Flow Scale Selector

        btn_FlowScaleSelector = QtWidgets.QPushButton(window)
        self.current_flow_scale = self.prefs.get('fltw_flow_scale', 1.0)
        btn_FlowScaleSelector.setText(self.flow_scale_list.get(self.current_flow_scale))

        def selectFlowScale(flow_scale):
            self.current_flow_scale = flow_scale
            self.prefs['fltw_flow_scale'] = flow_scale
            btn_FlowScaleSelector.setText(self.flow_scale_list.get(self.current_flow_scale))

        btn_FlowScaleSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_FlowScaleSelector.setMinimumSize(180, 28)
        btn_FlowScaleSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')
        btn_FlowScaleSelector_menu = QtWidgets.QMenu()
        for flow_scale in sorted(self.flow_scale_list.keys(), reverse=True):
            code = self.flow_scale_list.get(flow_scale, 1.0)
            action = btn_FlowScaleSelector_menu.addAction(code)            
            x = lambda chk=False, flow_scale=flow_scale: selectFlowScale(flow_scale)
            action.triggered[()].connect(x)

        btn_FlowScaleSelector.setMenu(btn_FlowScaleSelector_menu)
        new_speed_hbox.addWidget(btn_FlowScaleSelector)

        # Cpu Proc button

        if not sys.platform == 'darwin':            
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
            if self.cpu:
                btn_CpuProc.setStyleSheet('QPushButton {font:italic; background-color: #4f4f4f; color: #d9d9d9; border-top: 1px inset black; border-bottom: 1px inset #555555}')
            else:
                btn_CpuProc.setStyleSheet('QPushButton {color: #989898; background-color: #373737; border-top: 1px inset #555555; border-bottom: 1px inset black}')
            btn_CpuProc.pressed.connect(enableCpuProc)
            new_speed_hbox.addWidget(btn_CpuProc)

        ### Model Selector START

        current_model_name = self.model_map.get(self.prefs.get('trained_models_folder'), 'Unknown')
        
        # Model Selector Button
        btn_ModelSelector = QtWidgets.QPushButton(window)
        btn_ModelSelector.setText(current_model_name)
        
        def selectModel(trained_models_folder):
            self.prefs['trained_models_folder'] = trained_models_folder
            btn_ModelSelector.setText(self.model_map.get(trained_models_folder))

        btn_ModelSelector.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_ModelSelector.setMinimumSize(140, 28)
        btn_ModelSelector.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #29323d; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}'
                                    'QPushButton::menu-indicator {image: none;}')

        btn_ModelSelector_menu = QtWidgets.QMenu()
        for trained_models_folder in sorted(self.model_map.keys()):
            
            code = self.model_map.get(trained_models_folder)
            action = btn_ModelSelector_menu.addAction(code)
            x = lambda chk=False, trained_models_folder=trained_models_folder: selectModel(trained_models_folder)
            action.triggered[()].connect(x)
    
        btn_ModelSelector.setMenu(btn_ModelSelector_menu)
        new_speed_hbox.addWidget(btn_ModelSelector)

        ### Model Selector END
        
        vbox.addLayout(new_speed_hbox)
        vbox.addWidget(lbl_Spacer)

        # Work Folder

        def chooseFolder():
            result_folder = str(QtWidgets.QFileDialog.getExistingDirectory(window, "Open Directory", self.working_folder, QtWidgets.QFileDialog.ShowDirsOnly))
            if result_folder =='':
                return
            self.working_folder = result_folder
            txt_WorkFolder.setText(self.working_folder)
            self.prefs['working_folder'] = self.working_folder

        def txt_WorkFolder_textChanged():
            self.working_folder = txt_WorkFolder.text()

        # Work Folder Label

        lbl_WorkFolder = QtWidgets.QLabel('Export folder', window)
        lbl_WorkFolder.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_WorkFolder.setMinimumHeight(28)
        lbl_WorkFolder.setMaximumHeight(28)
        lbl_WorkFolder.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(lbl_WorkFolder)

        # Work Folder ENV Variable selector

        if os.getenv('FLAMETWML_WORK_FOLDER'):
            lbl_WorkFolderPath = QtWidgets.QLabel(self.working_folder, window)
            lbl_WorkFolderPath.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
            lbl_WorkFolderPath.setMinimumHeight(28)
            lbl_WorkFolderPath.setMaximumHeight(28)
            lbl_WorkFolderPath.setAlignment(QtCore.Qt.AlignCenter)
            vbox.addWidget(lbl_WorkFolderPath)

        else:
            # Work Folder Text Field
            hbox_workfolder = QtWidgets.QHBoxLayout()
            hbox_workfolder.setAlignment(QtCore.Qt.AlignLeft)

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
            if os.getenv('FLAMETWML_WORK_FOLDER'):
                self.working_folder = os.getenv('FLAMETWML_WORK_FOLDER')
                self.prefs['working_folder'] = os.getenv('FLAMETWML_WORK_FOLDER')

            modifiers = QtWidgets.QApplication.keyboardModifiers()
            self.framework.save_prefs()
            return {
                'working_folder': self.working_folder,
                'flow_scale': self.current_flow_scale,
                'hold_konsole': True if modifiers == QtCore.Qt.ControlModifier else False
            }
        else:
            return {}

    def dialog_model_path(self,  window, vbox):
        from PySide2 import QtWidgets, QtCore

        # Trained Model Path label

        lbl_WorkFolder = QtWidgets.QLabel('Trained Model', window)
        lbl_WorkFolder.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_WorkFolder.setMinimumHeight(28)
        lbl_WorkFolder.setMaximumHeight(28)
        lbl_WorkFolder.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(lbl_WorkFolder)

        # Trained Model Path Text Field

        hbox_trainedmodelfolder = QtWidgets.QHBoxLayout()
        hbox_trainedmodelfolder.setAlignment(QtCore.Qt.AlignLeft)

        def show_missing_model_files(model_files):
            msg = 'One of the modules files not found. Make sure %s are in folder' % pformat(model_files)
            mbox = QtWidgets.QMessageBox()
            mbox.setWindowTitle('flameTimewrarpML')
            mbox.setText(msg)
            mbox.exec_()

        def chooseFolder():
            dialog = QtWidgets.QFileDialog(window)
            dialog.setWindowTitle('Select any of Trained Model pkl files')
            dialog.setNameFilter('PKL files (*.pkl)')
            dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
            dialog.setDirectory(self.prefs.get('trained_models_folder'))
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                file_names = dialog.selectedFiles()
                if file_names:
                    result_folder = os.path.dirname(file_names[0])
                else:
                    return
            
            model_files = [
                'contextnet.pkl',
                'flownet.pkl',
                'unet.pkl'
            ]
            
            for model_file in model_files:
                model_file_path = os.path.join(result_folder, model_file)
                if not os.path.isfile(model_file_path):
                    show_missing_model_files(model_files)
                    return

            txt_TrainedModelFolder.setText(result_folder)
            self.prefs['trained_models_folder'] = result_folder
    
        def txt_TrainedModelFolder_textChanged():
            self.prefs['trained_models_folder'] = txt_TrainedModelFolder.text()
    
        txt_TrainedModelFolder = QtWidgets.QLineEdit('', window)
        txt_TrainedModelFolder.setFocusPolicy(QtCore.Qt.ClickFocus)
        txt_TrainedModelFolder.setMinimumSize(280, 28)
        txt_TrainedModelFolder.setStyleSheet('QLineEdit {color: #9a9a9a; background-color: #373e47; border-top: 1px inset #black; border-bottom: 1px inset #545454}')
        txt_TrainedModelFolder.setText(self.prefs.get('trained_models_folder'))
        txt_TrainedModelFolder.textChanged.connect(txt_TrainedModelFolder_textChanged)
        hbox_trainedmodelfolder.addWidget(txt_TrainedModelFolder)

        btn_changePreset = QtWidgets.QPushButton('Choose', window)
        btn_changePreset.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_changePreset.setMinimumSize(88, 28)
        btn_changePreset.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #424142; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                   'QPushButton:pressed {font:italic; color: #d9d9d9}')
        btn_changePreset.clicked.connect(chooseFolder)
        hbox_trainedmodelfolder.addWidget(btn_changePreset, alignment = QtCore.Qt.AlignLeft)

        vbox.addLayout(hbox_trainedmodelfolder)

    def export_clip(self, clip, export_dir, export_preset = None):
        import flame
        import traceback

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

        if not export_preset:
            export_preset_folder = self.flame.PyExporter.get_presets_dir(self.flame.PyExporter.PresetVisibility.values.get(2),
                            self.flame.PyExporter.PresetType.values.get(0))
            export_preset = os.path.join(export_preset_folder, 'OpenEXR', 'OpenEXR (16-bit fp PIZ).xml')

        exporter.export(clip, export_preset, export_dir, hooks=ExportHooks())

    def import_watcher(self, import_path, new_clip_name, destination, folders_to_cleanup, lockfile):
        flame_friendly_path = None
        def import_flame_clip():
            import flame
            new_clips = flame.import_clips(flame_friendly_path, destination)
            
            if len(new_clips) > 0:
                new_clip = new_clips[0]
                if new_clip:
                    new_clip.name.set_value(new_clip_name)
            try:
                flame.execute_shortcut('Refresh Thumbnails')
            except:
                pass

            # Colour Mgmt logic for future setting
            '''
            for version in new_clip.versions:
                for track in version.tracks:
                    for segment in track.segments:
                        segment.create_effect('Source Colour Mgmt')
            '''
            # End of Colour Mgmt logic for future settin

            if os.getenv('FLAMETWML_HARDCOMMIT') == 'True':
                # Hard Commit Logic for future setting
                for version in new_clip.versions:
                    for track in version.tracks:
                        for segment in track.segments:
                            segment.create_effect('Source Image')
                
                new_clip.open_as_sequence()
                new_clip.render()
                
                try:
                    flame.execute_shortcut('Hard Commit Selection in Timeline')
                    flame.execute_shortcut('Close Current Sequence')
                    flame.execute_shortcut('Refresh Thumbnails')
                except:
                    pass
                # End of Hard Commit Logic for future setting

        while self.threads:
            if not os.path.isfile(lockfile):
                self.log('Importing result from: %s' % import_path)
                file_names = [f for f in os.listdir(import_path) if f.endswith('.exr')]
                if file_names:
                    file_names.sort()
                    first_frame, ext = os.path.splitext(file_names[0])
                    last_frame, ext = os.path.splitext(file_names[-1])
                    flame_friendly_path = os.path.join(import_path, '[' + first_frame + '-' + last_frame + ']' + '.exr')

                    import flame
                    flame.schedule_idle_event(import_flame_clip)

                # clean-up source files used
                self.log('Cleaning up temporary files used: %s' % pformat(folders_to_cleanup))
                for folder in folders_to_cleanup:
                    cmd = 'rm -f "' + os.path.abspath(folder) + '/"*'
                    self.log('Executing command: %s' % cmd)
                    os.system(cmd)
                    try:
                        os.rmdir(folder)
                    except Exception as e:
                        self.log('Error removing %s: %s' % (folder, e))

                if os.getenv('FLAMETWML_HARDCOMMIT') == 'True':
                    time.sleep(1)
                    cmd = 'rm -f "' + os.path.abspath(import_path) + '/"*'
                    self.log('Executing command: %s' % cmd)
                    os.system(cmd)
                    try:
                        os.rmdir(import_path)
                    except Exception as e:
                        self.log('Error removing %s: %s' % (import_path, e))
                break
            time.sleep(0.1)

    def terminate_loops(self):
        self.threads = False
        
        for loop in self.loops:
            loop.join()

    def scan_trained_models_folder(self):
        ''''
        self.model_map = {(str) path: (str) name, ...}
        '''

        self.model_map = {}

        if not os.path.isdir(self.trained_models_path):
            msg = 'No trained models folder found: ' + self.trained_models_path + ' does not exist. Can not continue.'
            self.framework.show_error_msg(msg)
            return self.model_map

        folder_items = os.listdir(self.trained_models_path)
        if not folder_items:
            msg = 'No trained models found in: ' + self.trained_models_path + '. Can not continue.'
            self.framework.show_error_msg(msg)
            return self.model_map

        trained_models = []
        for folder_item in sorted(folder_items):
            folder_item = os.path.join(self.trained_models_path, folder_item)
            if os.path.isdir(folder_item):
                if folder_item.endswith('.model'):
                    trained_models.append(folder_item)

        for trained_models_folder in sorted(trained_models):
            self.model_map[trained_models_folder] = ' Model ' + os.path.basename(trained_models_folder).rstrip('.model') + ' '

        current_model_name = self.model_map.get(self.prefs.get('trained_models_folder'))
        if not current_model_name:
            trained_model_folders = sorted(self.model_map.keys())
            if not trained_model_folders:
                msg = 'No trained models found in: ' + self.trained_models_path + '. Can not continue.'
                self.framework.show_error_msg(msg)
                return self.model_map
            else:
                self.prefs['trained_models_folder'] = trained_model_folders[-1]

        return self.model_map

    def refresh_x11_windows_list(self):
        import flame
        if sys.platform == 'darwin' or flame.get_version_major() != '2022':
            self.x11_windows_list = []
            return 

        from subprocess import check_output
        wmctrl_path = '/usr/bin/wmctrl'
        if not os.path.isfile(wmctrl_path):
            wmctrl_path = os.path.join(self.framework.bundle_path, 'bin', 'wmctrl')
        out = ''
        try:
            out = check_output([wmctrl_path, '-lp'])
        except:
            pass
        if out:
            self.x11_windows_list = out.decode().splitlines()
        else:
            self.x11_windows_list = []

    def raise_last_window(self):
        import flame
        if sys.platform == 'darwin' or flame.get_version_major() != '2022':
            self.x11_windows_list = []
            return 

        from subprocess import check_output
        wmctrl_path = '/usr/bin/wmctrl'
        if not os.path.isfile(wmctrl_path):
            wmctrl_path = os.path.join(self.framework.bundle_path, 'bin', 'wmctrl')
        
        if not os.path.isfile(wmctrl_path):
            return
        
        out = ''
        try:
            out = check_output([wmctrl_path, '-lp'])
        except:
            return
        if out:
            pprint (out.decode)
            lines = out.decode().splitlines()
            for line in lines:
                if line not in self.x11_windows_list:
                    if 'Flame' not in line:
                        cmd = wmctrl_path + ' -ia ' + line.split()[0]
                        time.sleep(2)
                        for i in range(5):
                            os.system(cmd)

    def publish_progress_dialog(self):
        
        class Ui_Progress(object):
            def setupUi(self, Progress):
                Progress.setObjectName("Progress")
                Progress.setStyleSheet("#Progress {background-color: #242424;} #frame {border: 1px solid #474747; border-radius: 5px;}\n")

                self.verticalLayout = QtWidgets.QVBoxLayout(Progress)  # Change from horizontal layout to vertical layout
                self.verticalLayout.setSpacing(0)
                self.verticalLayout.setContentsMargins(0, 0, 0, 0)
                self.verticalLayout.setObjectName("verticalLayout")

                # Create a new widget for the stripe at the top
                self.stripe_widget = QtWidgets.QWidget(Progress)
                self.stripe_widget.setStyleSheet("background-color: #474747;")
                self.stripe_widget.setFixedHeight(24)  # Adjust this value to change the height of the stripe

                # Create a label inside the stripe widget
                self.stripe_label = QtWidgets.QLabel("Your text here")  # Replace this with the text you want on the stripe
                self.stripe_label.setStyleSheet("color: #cbcbcb;")  # Change this to set the text color

                # Create a layout for the stripe widget and add the label to it
                stripe_layout = QtWidgets.QHBoxLayout()
                stripe_layout.addWidget(self.stripe_label)
                stripe_layout.addStretch(1)
                stripe_layout.setContentsMargins(18, 0, 0, 0)  # This will ensure the label fills the stripe widget

                # Set the layout to stripe_widget
                self.stripe_widget.setLayout(stripe_layout)

                # Add the stripe widget to the top of the main window's layout
                self.verticalLayout.addWidget(self.stripe_widget)
                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

                self.src_horisontal_layout = QtWidgets.QHBoxLayout(Progress)  # Change from horizontal layout to vertical layout
                self.src_horisontal_layout.setSpacing(0)
                self.src_horisontal_layout.setContentsMargins(0, 0, 0, 0)
                self.src_horisontal_layout.setObjectName("srcHorisontalLayout")

                self.src_frame_one = QtWidgets.QFrame(Progress)
                self.src_frame_one.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.src_frame_one.setFrameShadow(QtWidgets.QFrame.Raised)
                self.src_frame_one.setObjectName("frame")

                self.src_frame_two = QtWidgets.QFrame(Progress)
                self.src_frame_two.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.src_frame_two.setFrameShadow(QtWidgets.QFrame.Raised)
                self.src_frame_two.setObjectName("frame")

                self.src_horisontal_layout.addWidget(self.src_frame_one)
                self.src_horisontal_layout.addWidget(self.src_frame_two)

                self.verticalLayout.addLayout(self.src_horisontal_layout)
                self.verticalLayout.setStretchFactor(self.src_horisontal_layout, 4)

                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

                self.int_horisontal_layout = QtWidgets.QHBoxLayout(Progress)  # Change from horizontal layout to vertical layout
                self.int_horisontal_layout.setSpacing(0)
                self.int_horisontal_layout.setContentsMargins(0, 0, 0, 0)
                self.int_horisontal_layout.setObjectName("intHorisontalLayout")

                self.int_frame_flow = QtWidgets.QFrame(Progress)
                self.int_frame_flow.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_flow.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_flow.setObjectName("frame")

                self.int_frame_wrp1 = QtWidgets.QFrame(Progress)
                self.int_frame_wrp1.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_wrp1.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_wrp1.setObjectName("frame")

                self.int_frame_wrp2 = QtWidgets.QFrame(Progress)
                self.int_frame_wrp2.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_wrp2.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_wrp2.setObjectName("frame")

                self.int_frame_mix = QtWidgets.QFrame(Progress)
                self.int_frame_mix.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_mix.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_mix.setObjectName("frame")

                self.int_horisontal_layout.addWidget(self.int_frame_flow)
                self.int_horisontal_layout.addWidget(self.int_frame_wrp1)
                self.int_horisontal_layout.addWidget(self.int_frame_wrp2)
                self.int_horisontal_layout.addWidget(self.int_frame_mix)

                self.verticalLayout.addLayout(self.int_horisontal_layout)
                self.verticalLayout.setStretchFactor(self.int_horisontal_layout, 2)

                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

                self.res_frame = QtWidgets.QFrame(Progress)
                self.res_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.res_frame.setFrameShadow(QtWidgets.QFrame.Raised)
                self.res_frame.setObjectName("frame")
                self.verticalLayout.addWidget(self.res_frame)
                self.verticalLayout.setStretchFactor(self.res_frame, 8)

                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space


                # Create a new horizontal layout for the bottom of the window
                bottom_layout = QtWidgets.QHBoxLayout()

                # Add a close button to the bottom layout
                self.close_button = QtWidgets.QPushButton("Close")
                self.close_button.clicked.connect(Progress.close)
                self.close_button.setContentsMargins(10, 4, 4, 4)
                self.set_button_style(self.close_button)
                bottom_layout.addWidget(self.close_button)

                # Add some stretch to the bottom layout to push the close button to the left
                bottom_layout.addStretch(1)

                # Add the bottom layout to the main layout
                self.verticalLayout.addLayout(bottom_layout)

                self.retranslateUi(Progress)
                QtCore.QMetaObject.connectSlotsByName(Progress)

            def retranslateUi(self, Progress):
                Progress.setWindowTitle("Form")
                # self.progress_header.setText("Timewarp ML")
                # self.progress_message.setText("Reading images....")

            def set_button_style(self, button):
                button.setMinimumSize(QtCore.QSize(150, 28))
                button.setMaximumSize(QtCore.QSize(150, 28))
                button.setFocusPolicy(QtCore.Qt.NoFocus)
                button.setStyleSheet('QPushButton {color: rgb(154, 154, 154); background-color: rgb(58, 58, 58); border: none; font: 14px}'
                'QPushButton:hover {border: 1px solid rgb(90, 90, 90)}'
                'QPushButton:pressed {color: rgb(159, 159, 159); background-color: rgb(66, 66, 66); border: 1px solid rgb(90, 90, 90)}'
                'QPushButton:disabled {color: rgb(116, 116, 116); background-color: rgb(58, 58, 58); border: none}'
                'QPushButton::menu-indicator {subcontrol-origin: padding; subcontrol-position: center right}'
                'QToolTip {color: rgb(170, 170, 170); background-color: rgb(71, 71, 71); border: 10px solid rgb(71, 71, 71)}')


        class Progress(QtWidgets.QWidget):
            """
            Overlay widget that reports toolkit bootstrap progress to the user.
            """

            PROGRESS_HEIGHT = 640
            PROGRESS_WIDTH = 960
            PROGRESS_PADDING = 48

            def __init__(self):
                W = 1280
                H = 720

                super().__init__()

                # now load in the UI that was created in the UI designer
                self.ui = Ui_Progress()
                self.ui.setupUi(self)

                # Record the mouse position on a press event.
                self.mousePressPos = None

                # make it frameless and have it stay on top
                self.setWindowFlags(
                    QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
                )


                desktop = QtWidgets.QApplication.desktop()
                screen_geometry = desktop.screenGeometry(desktop.primaryScreen())

                max_width = screen_geometry.width() * 0.8
                max_height = screen_geometry.height() * 0.8

                desired_width = W  # or whatever the aspect ratio calculation yields
                desired_height = 1.89 * H  # or whatever the aspect ratio calculation yields

                scale_factor = min(max_width / desired_width, max_height / desired_height)
                scaled_width = desired_width * scale_factor
                scaled_height = desired_height * scale_factor

                # Set the window's dimensions
                self.setGeometry(0, 0, scaled_width, scaled_height)
                # Move the window to the center of the screen
                screen_center = screen_geometry.center()
                self.move(screen_center.x() - scaled_width // 2, screen_center.y() - scaled_height // 2)

            def set_progress(self, pixmap):
                self.ui.label.setPixmap(pixmap)
                QtWidgets.QApplication.processEvents()

            def mousePressEvent(self, event):
                # Record the position at which the mouse was pressed.
                self.mousePressPos = event.globalPos()
                super().mousePressEvent(event)

            def mouseMoveEvent(self, event):
                if self.mousePressPos is not None:
                    # Calculate the new position of the window.
                    newPos = self.pos() + (event.globalPos() - self.mousePressPos)
                    # Move the window to the new position.
                    self.move(newPos)
                    # Update the position at which the mouse was pressed.
                    self.mousePressPos = event.globalPos()
                super().mouseMoveEvent(event)

            def mouseReleaseEvent(self, event):
                self.mousePressPos = None
                super().mouseReleaseEvent(event)

            def closeEvent(self, event):
                self.deleteLater()
                super().closeEvent(event)

        return Progress()


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
    
    exception_text = traceback.format_exception(exctype, value, tb)
    if 'flameTimewrarpML.py' in pformat(exception_text):
        msg = 'flameTimewrarpML: Python exception %s in %s' % (value, exctype)
        mbox = QtWidgets.QMessageBox()
        mbox.setWindowTitle('flameTimewrarpML')
        mbox.setText(msg)
        mbox.setDetailedText(pformat(exception_text))
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
    apps.append(flameTimewarpML(app_framework))
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
    
    app_initialized.__dict__["waitCursor"] = False

    if not app_framework:
        app_framework = flameAppFramework(app_name = app_name)
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
        if app.__class__.__name__ == 'flameTimewarpML':
            app_menu = []
            app_menu = app.build_menu()
            if app_menu:
                menu.append(app_menu)
    return menu


# bundle payload starts here
'''
BUNDLE_PAYLOAD
'''
