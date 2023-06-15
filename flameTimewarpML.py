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
import queue
import threading
import atexit
import hashlib
import pickle

from PySide2 import QtWidgets, QtCore, QtGui

from adsk.libwiretapPythonClientAPI import (
    WireTapClient,
    WireTapClientUninit,
    WireTapServerId,
    WireTapServerHandle,
    WireTapNodeHandle,
    WireTapClipFormat,
    WireTapInt,
    WireTapStr,
)

from pprint import pprint
from pprint import pformat

# Configurable settings
menu_group_name = 'Timewarp ML'
DEBUG = False
app_name = 'flameTimewarpML'
__version__ = 'v0.5.0.dev.003'

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
        self.app_name = self.framework.app_name
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

    def message(self, message, type = 'Error'):
        try:
            import flame
            flame.messages.show_in_dialog(
                title = self.app_name,
                message = message,
                type = 'error',
                buttons = ['Ok']
            )
        except Exception as e:
            self.log('unable to use flame message')
            self.log(pformat(e))

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
                Progress.setStyleSheet("#Progress {background-color: #242424;} #frame {border: 1px solid #474747; border-radius: 4px;}\n")
                                
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
                self.image_one_label = QtWidgets.QLabel(self.src_frame_one)
                self.image_one_label.setAlignment(QtCore.Qt.AlignCenter)
                frame_one_layout = QtWidgets.QVBoxLayout()
                frame_one_layout.setSpacing(0)
                frame_one_layout.setContentsMargins(0, 0, 0, 0)
                frame_one_layout.addWidget(self.image_one_label)
                self.src_frame_one.setLayout(frame_one_layout)

                self.src_frame_two = QtWidgets.QFrame(Progress)
                self.src_frame_two.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.src_frame_two.setFrameShadow(QtWidgets.QFrame.Raised)
                self.src_frame_two.setObjectName("frame")
                self.image_two_label = QtWidgets.QLabel(self.src_frame_two)
                self.image_two_label.setAlignment(QtCore.Qt.AlignCenter)
                frame_two_layout = QtWidgets.QVBoxLayout()
                frame_two_layout.setSpacing(0)
                frame_two_layout.setContentsMargins(0, 0, 0, 0)
                frame_two_layout.addWidget(self.image_two_label)
                self.src_frame_two.setLayout(frame_two_layout)

                self.src_horisontal_layout.addWidget(self.src_frame_one)
                self.src_horisontal_layout.addWidget(self.src_frame_two)

                self.verticalLayout.addLayout(self.src_horisontal_layout)
                self.verticalLayout.setStretchFactor(self.src_horisontal_layout, 4)

                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

                self.int_horisontal_layout = QtWidgets.QHBoxLayout(Progress)
                self.int_horisontal_layout.setSpacing(0)
                self.int_horisontal_layout.setContentsMargins(0, 0, 0, 0)
                self.int_horisontal_layout.setObjectName("intHorisontalLayout")

                self.int_frame_flow1 = QtWidgets.QFrame(Progress)
                self.int_frame_flow1.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_flow1.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_flow1.setObjectName("frame")
                self.flow1_label = QtWidgets.QLabel(self.int_frame_flow1)
                self.flow1_label.setAlignment(QtCore.Qt.AlignCenter)
                flow1_layout = QtWidgets.QVBoxLayout()
                flow1_layout.setSpacing(0)
                flow1_layout.setContentsMargins(0, 0, 0, 0)
                flow1_layout.addWidget(self.flow1_label)
                self.int_frame_flow1.setLayout(flow1_layout)

                self.int_frame_flow2 = QtWidgets.QFrame(Progress)
                self.int_frame_flow2.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_flow2.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_flow2.setObjectName("frame")
                self.flow2_label = QtWidgets.QLabel(self.int_frame_flow2)
                self.flow2_label.setAlignment(QtCore.Qt.AlignCenter)
                flow2_layout = QtWidgets.QVBoxLayout()
                flow2_layout.setSpacing(0)
                flow2_layout.setContentsMargins(0, 0, 0, 0)
                flow2_layout.addWidget(self.flow2_label)
                self.int_frame_flow2.setLayout(flow2_layout)

                self.int_frame_flow3 = QtWidgets.QFrame(Progress)
                self.int_frame_flow3.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_flow3.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_flow3.setObjectName("frame")
                self.flow3_label = QtWidgets.QLabel(self.int_frame_flow3)
                self.flow3_label.setAlignment(QtCore.Qt.AlignCenter)
                flow3_layout = QtWidgets.QVBoxLayout()
                flow3_layout.setSpacing(0)
                flow3_layout.setContentsMargins(0, 0, 0, 0)
                flow3_layout.addWidget(self.flow3_label)
                self.int_frame_flow3.setLayout(flow3_layout)

                self.int_frame_flow4 = QtWidgets.QFrame(Progress)
                self.int_frame_flow4.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.int_frame_flow4.setFrameShadow(QtWidgets.QFrame.Raised)
                self.int_frame_flow4.setObjectName("frame")
                self.flow4_label = QtWidgets.QLabel(self.int_frame_flow4)
                self.flow4_label.setAlignment(QtCore.Qt.AlignCenter)
                flow4_layout = QtWidgets.QVBoxLayout()
                flow4_layout.setSpacing(0)
                flow4_layout.setContentsMargins(0, 0, 0, 0)
                flow4_layout.addWidget(self.flow4_label)
                self.int_frame_flow4.setLayout(flow4_layout)

                self.int_horisontal_layout.addWidget(self.int_frame_flow1)
                self.int_horisontal_layout.addWidget(self.int_frame_flow2)
                self.int_horisontal_layout.addWidget(self.int_frame_flow3)
                self.int_horisontal_layout.addWidget(self.int_frame_flow4)

                self.verticalLayout.addLayout(self.int_horisontal_layout)
                self.verticalLayout.setStretchFactor(self.int_horisontal_layout, 2)

                self.verticalLayout.addSpacing(4)  # Add a 4-pixel space

                self.res_frame = QtWidgets.QFrame(Progress)
                self.res_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
                self.res_frame.setFrameShadow(QtWidgets.QFrame.Raised)
                self.res_frame.setObjectName("frame")
                self.image_res_label = QtWidgets.QLabel(self.res_frame)
                self.image_res_label.setAlignment(QtCore.Qt.AlignCenter)
                frame_res_layout = QtWidgets.QVBoxLayout()
                frame_res_layout.setSpacing(0)
                frame_res_layout.setContentsMargins(8, 8, 8, 8)
                frame_res_layout.addWidget(self.image_res_label)
                self.res_frame.setLayout(frame_res_layout)

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
                bottom_layout.addWidget(self.close_button, alignment=QtCore.Qt.AlignLeft)

                self.info_label = QtWidgets.QLabel('Frame:', Progress)
                self.info_label.setContentsMargins(10, 4, 10, 4)
                self.info_label.setStyleSheet("color: #cbcbcb;")
                bottom_layout.addWidget(self.info_label)
                bottom_layout.setStretchFactor(self.info_label, 1)

                # mode selector button
                self.mode_selector = QtWidgets.QPushButton('Normal')
                self.mode_selector.setContentsMargins(10, 4, 10, 4)
                self.set_selector_button_style(self.mode_selector)
                self.mode_selector.setMinimumSize(QtCore.QSize(80, 28))
                self.mode_selector.setMaximumSize(QtCore.QSize(80, 28))
                bottom_layout.addWidget(self.mode_selector, alignment=QtCore.Qt.AlignRight)
                bottom_layout.addSpacing(4)

                # flow res selector button
                self.flow_res_selector = QtWidgets.QPushButton('Use Full Resolution')
                self.flow_res_selector.setContentsMargins(10, 4, 10, 4)
                self.set_selector_button_style(self.flow_res_selector)
                bottom_layout.addWidget(self.flow_res_selector, alignment=QtCore.Qt.AlignRight)
                bottom_layout.addSpacing(4)

                # Create a new QPushButton
                self.render_button = QtWidgets.QPushButton("Render")
                self.render_button.clicked.connect(Progress.render)
                self.render_button.setContentsMargins(4, 4, 10, 4)
                self.set_button_style(self.render_button)
                bottom_layout.addWidget(self.render_button, alignment=QtCore.Qt.AlignRight)

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

            def set_selector_button_style(self, button):
                button.setMinimumSize(QtCore.QSize(150, 28))
                button.setMaximumSize(QtCore.QSize(150, 28))
                button.setFocusPolicy(QtCore.Qt.NoFocus)
                button.setStyleSheet('QPushButton {color: rgb(154, 154, 154); background-color: rgb(44, 54, 68); border: none; font: 14px}'
                'QPushButton:hover {border: 1px solid rgb(90, 90, 90)}'
                'QPushButton:pressed {color: rgb(159, 159, 159); background-color: rgb(44, 54, 68); border: 1px solid rgb(90, 90, 90)}'
                'QPushButton:disabled {color: rgb(116, 116, 116); background-color: rgb(58, 58, 58); border: none}'
                'QPushButton::menu-indicator {image: none;}')

        def __init__(self, selection, **kwargs):
            super().__init__()
            self.mode = kwargs.get('mode', 'Timewarp')
            self.twml = kwargs.get('parent')
            self.app_name = self.twml.app_name
            
            # clean temp folder
            self.temp_folder = os.path.join(
                self.twml.framework.bundle_path,
                'temp'
            )
            if os.path.isdir(self.temp_folder):
                cmd = 'rm -rf ' + self.temp_folder + os.path.sep + '*'
                try:
                    os.system(cmd)
                except:
                    pass

            self.twml.progress = self
            self.current_frame = 1
            self.threads = True

            if selection:
                self.clip_parent = selection[0].parent

            if not self.twml.import_numpy():
                return
            
            self.frames_map = self.twml.compose_frames_map(selection, self.mode)
            if not self.frames_map:
                return
            
            self.min_frame = min(self.frames_map.keys())
            self.max_frame = max(self.frames_map.keys())
            
            self.destination_node_id = self.twml.create_destination_node(
                selection, 
                len(self.frames_map.keys())
                )
                        
            self.current_frame = min(self.frames_map.keys())
            self.rendering = False

            try:
                H = selection[0].height
                W = selection[0].width
            except:
                W = 1280
                H = 720
            
            # now load in the UI that was created in the UI designer
            self.ui = self.Ui_Progress()
            self.ui.setupUi(self)

            # set up mode menu
            mode_menu = QtWidgets.QMenu(self)
            for mode_number in sorted(self.twml.modes.keys(), reverse=True):
                code = self.twml.modes.get(mode_number, 1.0)
                action = mode_menu.addAction(code)
                x = lambda chk=False, mode_number=mode_number: self.twml.select_mode(mode_number)
                action.triggered[()].connect(x)
            self.ui.mode_selector.setMenu(mode_menu)

            # set up flow res menu
            flow_res_menu = QtWidgets.QMenu(self)
            for flow_res in sorted(self.twml.flow_res.keys(), reverse=True):
                code = self.twml.flow_res.get(flow_res, 1.0)
                action = flow_res_menu.addAction(code)
                x = lambda chk=False, flow_res=flow_res: self.twml.select_flow_res(flow_res)
                action.triggered[()].connect(x)
            self.ui.flow_res_selector.setMenu(flow_res_menu)

            self.ui.stripe_label.setText(self.mode)
            self.ui.info_label.setText('initializing PyTorch environment...')

            # Record the mouse position on a press event.
            self.mousePressPos = None

            # make it frameless and have it stay on top
            # self.setWindowFlags(
            #    QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
            # )


            self.setWindowFlags(
                QtCore.Qt.Window | QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint
            )

            desktop = QtWidgets.QApplication.desktop()
            screen_geometry = desktop.screenGeometry(desktop.primaryScreen())

            max_width = screen_geometry.width() * 0.8
            max_height = screen_geometry.height() * 0.8

            desired_width = W  # or whatever the aspect ratio calculation yields
            desired_height = 1.88 * H  # Coeeficient to accomodate additional rows

            scale_factor = min(max_width / desired_width, max_height / desired_height)
            scaled_width = desired_width * scale_factor
            scaled_height = desired_height * scale_factor

            # Check that scaled_width is not less than the minimum
            if scaled_width < 1024:
                scaled_width = 1024

            # Set the window's dimensions
            self.setGeometry(0, 0, scaled_width, scaled_height)
            # Move the window to the center of the screen
            screen_center = screen_geometry.center()
            self.move(screen_center.x() - scaled_width // 2, screen_center.y() - scaled_height // 2 - 100)
            self.show()

            # QtCore.QTimer.singleShot(0, self.init_torch)
            QtCore.QTimer.singleShot(0, self.after_show)

            self.message_queue = queue.Queue()
            self.message_thread = threading.Thread(target=self.process_messages)
            self.message_thread.daemon = True
            self.message_thread.start()

        def showEvent(self, event):
            super().showEvent(event)
            self.raise_()
            self.activateWindow()
            max_label_width = self.ui.info_label.width()
            self.ui.info_label.setMaximumWidth(max_label_width)

        def init_torch(self):
            if not self.twml.torch:
                if not self.twml.import_torch():
                    self.close()
                    return

        def after_show(self):
            self.setFixedSize(self.size())
            self.init_torch()
            self.process_current_frame()

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key_Left:
                self.left_arrow_pressed()
            elif event.key() == QtCore.Qt.Key_Right:
                self.right_arrow_pressed()
            else:
                super().keyPressEvent(event)  # Pass the event to the parent's handler

        def left_arrow_pressed(self):
            self.current_frame = self.current_frame - 1 if self.current_frame > self.min_frame else self.min_frame
            self.info('Frame ' + str(self.current_frame))
            self.process_current_frame()

        def right_arrow_pressed(self):
            self.current_frame = self.current_frame + 1 if self.current_frame < self.max_frame else self.max_frame
            self.info('Frame ' + str(self.current_frame))
            self.process_current_frame()

        def render(self):
            self.rendering = not self.rendering
            button_text = 'Stop' if self.rendering else 'Render'
            self.ui.render_button.setText(button_text)
            QtWidgets.QApplication.instance().processEvents()
            time.sleep(0.001)
            self.ui.render_button.setText(button_text)
            QtWidgets.QApplication.instance().processEvents()
            if self.rendering:
                self.render_loop()

        def render_loop(self):
            render_loop_thread = threading.Thread(target=self._render_loop)
            render_loop_thread.daemon = True
            render_loop_thread.start()

        def _render_loop(self):
            for frame in self.frames_map.keys():
                if not self.threads:
                    break
                if not self.rendering:
                    break
                if self.frames_map[frame].get('saved'):
                    self.info('Frame ' + str(self.current_frame) + ': Already saved')
                    continue
                self.current_frame = frame
                print ('current frame: %s' % self.current_frame)
                print ('calling process_current_frame()')
                self.process_current_frame()
            return

        def process_current_frame(self):
            frame_thread = threading.Thread(target=self._process_current_frame)
            frame_thread.start()
            frame_thread.join()

        def _process_current_frame(self):
            self.current_frame_data = self.frames_map.get(self.current_frame)

            self.destination = self.current_frame_data['outgoing']['clip'].parent
            self.info('Frame ' + str(self.current_frame) + ': reading incoming source image data...')

            inc_frame_number = self.current_frame_data['incoming']['frame_number'] - 1
            
            incoming_image_data = self.read_image_data(
                self.current_frame_data['incoming']['clip'], 
                inc_frame_number
                )
            
            self.update_interface_image(
                incoming_image_data[::2, ::2, :], 
                self.ui.image_one_label,
                text = 'src frame: ' + str(inc_frame_number + 1)
                )
            
            if not self.threads:
                return
            
            self.info('Frame ' + str(self.current_frame) + ': reading outgoing source image data...')

            outg_frame_number = self.current_frame_data['outgoing']['frame_number'] - 1
            outgoing_image_data = self.read_image_data(
                self.current_frame_data['outgoing']['clip'], 
                outg_frame_number
                )
            
            self.update_interface_image(
                outgoing_image_data[::2, ::2, :], 
                self.ui.image_two_label,
                text = 'src frame: ' + str(outg_frame_number + 1)
                )
            
            if not self.threads:
                return
            
            ratio = self.current_frame_data['ratio']
            if ratio == 0.0:
                result_image_data = incoming_image_data
                self.update_interface_image(
                    incoming_image_data[::4, ::4, :],
                    self.ui.flow1_label,
                    text = 'copy of frame: ' + str(inc_frame_number + 1)
                    )
                self.update_interface_image(
                    incoming_image_data[::4, ::4, :], 
                    self.ui.flow2_label,
                    text = 'copy of frame: ' + str(inc_frame_number + 1)
                    )
                self.update_interface_image(
                    incoming_image_data[::4, ::4, :], 
                    self.ui.flow3_label,
                    text = 'copy of frame: ' + str(inc_frame_number + 1)
                    )
                self.update_interface_image(
                    incoming_image_data[::4, ::4, :], 
                    self.ui.flow4_label,
                    text = 'copy of frame: ' + str(inc_frame_number + 1)
                    )

            elif ratio == 1.0:
                result_image_data = outgoing_image_data
                self.update_interface_image(
                    incoming_image_data[::4, ::4, :],
                    self.ui.flow1_label,
                    text = 'copy of frame: ' + str(outg_frame_number + 1)
                    )
                self.update_interface_image(
                    incoming_image_data[::4, ::4, :], 
                    self.ui.flow2_label,
                    text = 'copy of frame: ' + str(outg_frame_number + 1)
                    )
                self.update_interface_image(
                    incoming_image_data[::4, ::4, :], 
                    self.ui.flow3_label,
                    text = 'copy of frame: ' + str(outg_frame_number + 1)
                    )
                self.update_interface_image(
                    incoming_image_data[::4, ::4, :], 
                    self.ui.flow4_label,
                    text = 'copy of frame: ' + str(outg_frame_number + 1)
                    )

            else:

                self.info('Frame ' + str(self.current_frame) + ': Processing...')

                flow = self.twml.flownet(incoming_image_data, outgoing_image_data, ratio, self.twml.flownet_model_path)
                if not self.threads:
                    return
                result_image_data = flow

            if not self.threads:
                return

            self.update_interface_image(
                result_image_data, 
                self.ui.image_res_label,
                text = 'frame: ' + str(self.current_frame)
                )
            
            if self.current_frame_data.get('saved'):
                self.info('Frame ' + str(self.current_frame))
                return

            self.info('Frame ' + str(self.current_frame) + ': Saving...')
            self.save_result_frame(
                result_image_data,
                self.current_frame - 1
            )
            self.current_frame_data['saved'] = True
            self.info('Frame ' + str(self.current_frame))

        def read_image_data(self, clip, frame_number):
            import flame
            np = self.twml.np

            try:
                server_handle = WireTapServerHandle('localhost')
                clip_node_id = clip.get_wiretap_node_id()
                clip_node_handle = WireTapNodeHandle(server_handle, clip_node_id)
                fmt = WireTapClipFormat()
                if not clip_node_handle.getClipFormat(fmt):
                    raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())
                num_frames = WireTapInt()
                if not clip_node_handle.getNumFrames(num_frames):
                    raise Exception(
                        "Unable to obtain number of frames: %s." % clip_node_handle.lastError()
                    )

                buff = "0" * fmt.frameBufferSize()

                if not clip_node_handle.readFrame(int(frame_number), buff, fmt.frameBufferSize()):
                    raise Exception(
                        'Unable to obtain read frame %i: %s.' % (frame_number, clip_node_handle.lastError())
                    )
                
                frame_buffer_size = fmt.frameBufferSize()
                
                bits_per_channel = fmt.bitsPerPixel() // fmt.numChannels()

                if bits_per_channel == 8:
                    dt = np.uint8
                elif bits_per_channel == 10:
                    dt = np.uint16
                    # for value in bytes(buff, 'latin-1'):
                    #     print (bin(value))
                    byte_array = np.frombuffer(bytes(buff, 'latin-1'), dtype='>u4')
                    # byte_array = np.frombuffer(bytes(buff, 'latin-1'), dtype='<u4')
                    values_10bit = np.empty((len(byte_array) * 3,), dtype=np.uint16)
                    # Extract the three 10-bit values from each 4-byte sequence
                    for i, value in enumerate(byte_array):
                        values_10bit[i*3] = (value >> 22) & 0x3FF  # first 10 bits
                        values_10bit[i*3 + 1] = (value >> 12) & 0x3FF  # next 10 bits
                        values_10bit[i*3 + 2] = (value >> 2) & 0x3FF  # last 10 bits

                    values_16bit = (values_10bit // 1023) * 65535                    
                    buff = values_16bit.astype('<u2').tobytes().decode('latin-1')
                    frame_buffer_size = len(buff)

                    del byte_array
                    del values_10bit
                    del values_16bit

                elif bits_per_channel == 16 and not('float' in fmt.formatTag()):
                    dt = np.uint16
                elif (bits_per_channel == 16) and ('float' in fmt.formatTag()):
                    dt = np.float16
                elif bits_per_channel == 32:
                    dt = np.float32
                else:
                    raise Exception('Unknown image format')
                
                buff_tail = (frame_buffer_size // np.dtype(dt).itemsize) - (fmt.height() * fmt.width() * fmt.numChannels())
                image_array = np.frombuffer(bytes(buff, 'latin-1'), dtype=dt)[:-1 * buff_tail]
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = np.flip(image_array, axis=0)

                del buff

                if image_array.dtype == np.uint8:
                    return image_array.astype(np.float32) / 255
                elif image_array.dtype == np.uint16:
                    return image_array.astype(np.float32) / 65535
                else:
                    return image_array.astype(np.float32)

            except Exception as e:
                self.message('Error: %s' % e)

            finally:
                server_handle = None
                clip_node_handle = None

        def process_messages(self):
            while self.threads:
                try:
                    item = self.message_queue.get_nowait()
                except queue.Empty:
                    if not self.threads:
                        break
                    time.sleep(0.05)
                    continue
                if item is None:
                    time.sleep(0.05)
                    continue
                if not isinstance(item, dict):
                    self.message_queue.task_done()
                    time.sleep(0.05)
                    continue
                item_type = item.get('type')
                if not item_type:
                    self.message_queue.task_done()
                    time.sleep(0.05)
                    continue
                elif item_type == 'info':
                    message = item.get('message')
                    self._info(f'{message}')
                elif item_type == 'message':
                    message = item.get('message')
                    self._message(f'{message}')
                elif item_type == 'image':
                    self._update_interface_image(
                        item.get('image'),
                        item.get('image_label'),
                        item.get('text')
                    )
                else:
                    self.message_queue.task_done()
                    time.sleep(0.05)
                    continue
                
                self.message_queue.task_done()
                time.sleep(0.05)
            return

        def update_interface_image(self, array, image_label, text = None):
            if self.message_queue.qsize() > 8:
                if image_label != self.ui.image_res_label:
                    return
            
            item = {
                'type': 'image',
                'image': array,
                'image_label': image_label,
                'text': text
            }
            self.message_queue.put(item)

        def _update_interface_image(self, array, image_label, text = None):
            np = self.twml.np

            if array is None:
                return

            # colourmanagement should go here
            if (array.dtype == np.float16) or (array.dtype == np.float32):
                img = np.clip(array, 0, 1) * 255
                img = img.astype(np.uint8)
            else:
                img = array.astype(np.uint8)

            # Convert the numpy array to a QImage
            height, width, _ = img.shape
            bytes_per_line = 3 * width
            qt_image = QtGui.QImage(img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            qt_pixmap = QtGui.QPixmap.fromImage(qt_image)
            parent_frame = image_label.parent()
            scaled_pixmap = qt_pixmap.scaled(
                parent_frame.size() * 0.9, 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation)

            if text:
                margin = 4
                origin_x = 2
                origin_y = 2

                painter = QtGui.QPainter(scaled_pixmap)
                font = QtGui.QFont("Discreet", 12)
                painter.setFont(font)
                metrics = QtGui.QFontMetrics(font)
                text_width = metrics.horizontalAdvance(text)
                text_height = metrics.height()

                '''
                rect_x = origin_x
                rect_y = scaled_pixmap.height() - text_height - margin * 2 - origin_y
                rect_width = text_width + margin * 2 + 2
                rect_height = text_height + margin * 2
                color = QtGui.QColor(0, 0, 0)
                radius = 2
                painter.setBrush(color)
                painter.setOpacity(0.2)
                painter.drawRoundedRect(rect_x, rect_y, rect_width, rect_height, radius, radius)
                '''

                painter.setOpacity(1.0)
                painter.setPen(QtGui.QColor(255, 255, 255))
                text_x = margin + origin_x
                text_y = scaled_pixmap.height() - margin -origin_y
                painter.drawText(text_x, text_y, text)

                painter.end()

            image_label.setPixmap(scaled_pixmap)
            QtWidgets.QApplication.instance().processEvents()
            time.sleep(0.001)
            image_label.setPixmap(scaled_pixmap)
            QtWidgets.QApplication.instance().processEvents()

        def info(self, message):
            item = {
                'type': 'info',
                'message': message
            }
            self.message_queue.put(item)

        def _info(self, message):
            self.ui.info_label.setText(str(message))
            QtWidgets.QApplication.instance().processEvents()
            time.sleep(0.001)
            self.ui.info_label.setText(str(message))
            QtWidgets.QApplication.instance().processEvents()

        def message(self, message):
            item = {
                'type': 'message',
                'message': message
            }
            self.message_queue.put(item)

        def _message(self, message):
            import flame
            self.info(message)
            flame.messages.show_in_console(self.app_name + ': ' + str(message), 'info', 8)

        def save_result_frame(self, image_data, frame_number):
            import flame
            import numpy as np

            ext = '.exr' if 'float' in self.twml.fmt.formatTag() else '.dpx'
                
            file_path = os.path.join(
                self.temp_folder,
                str(frame_number) + ext
            )

            # def wiretap_test():
            try:
                if not os.path.isdir(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path))


                height, width, depth = image_data.shape
                red = image_data[:, :, 0]
                green = image_data[:, :, 1]
                blue = image_data[:, :, 2]
                if depth > 3:
                    alpha = image_data[:, :, 3]
                else:
                    alpha = np.array([])

                if file_path.endswith('exr'):
                    if self.twml.bits_per_channel == 32:
                        self.twml.write_exr(
                            file_path,
                            width,
                            height,
                            red,
                            green,
                            blue,
                            alpha = alpha,
                            half_float = False
                        )
                    else:
                        self.twml.write_exr(
                            file_path,
                            width,
                            height,
                            red.astype(np.float16),
                            green.astype(np.float16),
                            blue.astype(np.float16),
                            alpha = alpha.astype(np.float16)
                        )

                else:
                    self.twml.write_dpx(
                        file_path,
                        width,
                        height,
                        red,
                        green,
                        blue,
                        alpha = alpha,
                        bit_depth = self.twml.bits_per_channel
                    )

                gateway_server_id = WireTapServerId('Gateway', 'localhost')
                gateway_server_handle = WireTapServerHandle(gateway_server_id)
                clip_node_handle = WireTapNodeHandle(gateway_server_handle, file_path + '@CLIP')
                fmt = WireTapClipFormat()
                if not clip_node_handle.getClipFormat(fmt):
                    raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())
                
                buff = "0" * fmt.frameBufferSize()

                if not clip_node_handle.readFrame(0, buff, fmt.frameBufferSize()):
                    raise Exception(
                        'Unable to obtain read frame %i: %s.' % (frame_number, clip_node_handle.lastError())
                    )
                
                server_handle = WireTapServerHandle('localhost')
                destination_node_handle = WireTapNodeHandle(server_handle, self.destination_node_id)
                dest_fmt = WireTapClipFormat()
                if not destination_node_handle.getClipFormat(dest_fmt):
                    raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())

                if not destination_node_handle.writeFrame(
                    frame_number, buff, dest_fmt.frameBufferSize()
                ):
                    raise Exception(
                        "Unable to obtain write frame %i: %s."
                        % (frame_number, destination_node_handle.lastError())
                    )

                os.remove(file_path)

            except Exception as e:
                self.message('Error: %s' % e)
            finally:
                gateway_server_handle = None
                clip_node_handle = None
                server_handle = None
                destination_node_handle = None

            # flame.schedule_idle_event(wiretap_test)

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
            import flame

            self.threads = False
            self.twml.threads = False
            self.message_queue.join()
            self.message_thread.join()
            
            result_clip = None
            self.twml.temp_library.acquire_exclusive_access()

            flame.execute_shortcut('Save Project')
            flame.execute_shortcut('Refresh Thumbnails')
            self.twml.temp_library.commit()
            result_clip = flame.find_by_wiretap_node_id(self.destination_node_id)

            if not result_clip:
                # try harder
                flame.execute_shortcut('Save Project')
                flame.execute_shortcut('Refresh Thumbnails')
                self.twml.temp_library.commit()
                ch = self.twml.temp_library.children
                for c in ch:
                    if c.name.get_value() == self.twml.destination_node_name:
                        result_clip = c
            
            if not result_clip:
                flame.execute_shortcut('Save Project')
                flame.execute_shortcut('Refresh Thumbnails')
                self.twml.temp_library.commit()
                result_clip = flame.find_by_wiretap_node_id(self.destination_node_id)

            if not result_clip:
                # try harder
                flame.execute_shortcut('Save Project')
                flame.execute_shortcut('Refresh Thumbnails')
                self.twml.temp_library.commit()
                ch = self.twml.temp_library.children
                for c in ch:
                    if c.name.get_value() == self.twml.destination_node_name:
                        result_clip = c
            
            if result_clip:
                try:
                    copied_clip = flame.media_panel.copy(
                        source_entries = result_clip, destination = self.clip_parent
                        )
                    self.twml.temp_library.acquire_exclusive_access()
                    flame.delete(self.twml.temp_library)
                    copied_clip.render()
                    copied_clip.commit()
                except:
                    pass


            self.twml.progress = None
            self.twml.torch = None
            self.deleteLater()
            super().closeEvent(event)

            def rescan_hooks():
                flame.execute_shortcut('Rescan Python Hooks')

            flame.schedule_idle_event(rescan_hooks)

    def __init__(self, framework):
        flameMenuApp.__init__(self, framework)

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

        self.flow_res = {
            2.0: 'Use 2x Resolution',
            1.0: 'Use Full Resolution',
            0.5: 'Use 1/2 Resolution',
            0.25: 'Use 1/4 Resolution',
            0.125: 'Use 1/8 Resolution',
        }

        self.modes = {
            1: 'Fast',
            2: 'Normal'
        }

        self.trained_models_path = os.path.join(
            self.framework.bundle_path,
            'trained_models', 
            'default',
        )

        self.model_path = os.path.join(
            self.trained_models_path,
            'v4.6.model'
            )

        self.flownet_model_path = os.path.join(
            self.trained_models_path,
            'v2.4.model',
            'flownet.pkl'
        )

        self.progress = None
        self.torch = None
        self.threads = True

        # this enables fallback to CPU on Macs
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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

    def import_torch(self):
        import flame
        flame.messages.show_in_console('TimewarpML: Initializing PyTorch backend', 'info', 8)
        self.torch = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.rand(10, device = 'cuda')
            elif torch.backends.mps.is_available():
                from torch import mps
                torch.rand(10, device = 'mps')
                mps.empty_cache()
            else:
                torch.rand(10)
            self.torch = torch
        except:
            try:
                if not self.framework.site_packages_folder in sys.path:
                    sys.path.append(self.framework.site_packages_folder)
                import torch
                if torch.cuda.is_available():
                    torch.rand(10, device = 'cuda')
                elif torch.backends.mps.is_available():
                    from torch import mps
                    torch.rand(10, device = 'mps')
                    mps.empty_cache()
                else:
                    torch.rand(10)
                self.torch = torch
            except Exception as e:
                msg_str = 'Unable to import PyTorch module.\n'
                msg_str += 'Please make sure PyTorch is installed and working '
                msg_str += "with installed graphics card and Flame's python version "
                msg_str += '.'.join(str(num) for num in sys.version_info[:3])
                self.message(msg_str)
                self.log(msg)
                self.log(pformat(e))

        flame.messages.clear_console()

        print ('import')
        from torch import mps
        print (mps.driver_allocated_memory())
        return self.torch
        
    def import_numpy(self):
        import flame
        flame.messages.show_in_console('TimewarpML: Initializing Numpy module', 'info', 1)
        self.np = None
        try:
            import numpy
            self.np = numpy
        except:
            try:
                if not self.framework.site_packages_folder in sys.path:
                    sys.path.append(self.framework.site_packages_folder)
                import numpy
                self.np = numpy
            except Exception as e:
                msg_str = 'Unable to import Numpy module.\n'
                msg_str += 'Please make sure Numpy is installed and working '
                msg_str += "with installed graphics card and Flame's python version "
                msg_str += '.'.join(str(num) for num in sys.version_info[:3])
                self.message(msg_str)
                self.log(msg)
                self.log(pformat(e))

        flame.messages.clear_console()
        return self.np

    def compose_frames_map(self, selection, mode):

        '''
        {
            frame_number: {
                ratio: float,
                incoming: {clip: path, data: np.array([]), metadata: {}}
                outgoing: {clip: path, data: np.array([]), metadata: {}}
                temp_library:
                destination:
                cleanup: [{type, path}]
            }
        }
        '''
        
        if mode == 'Timewarp':
            return self.compose_frames_map_fltw(selection)
        else:
            return {}

    def compose_frames_map_fltw(self, selection):
        import flame
        def sequence_message():
            flame.messages.show_in_dialog(
                        title = self.app_name,
                        message = 'Please select single-track clips with no versions or edits',
                        type = 'error',
                        buttons = ['Ok']
            )
        
        def effect_message():
            flame.messages.show_in_dialog(
                title = self.app_name,
                message = 'Please select clips with Timewarp Timeline FX',
                type = 'error',
                buttons = ['Ok']
            )

        # sanity checks

        if len(selection) < 1:
            sequence_message()
            return {}
        
        clip = selection[0]
        self.clip = clip
        self.clip_parent = clip.parent

        effects = clip.versions[0].tracks[0].segments[0].effects

        if not isinstance(clip, (flame.PyClip)):
            sequence_message()
            return {}
        elif len(clip.versions) != 1:
            sequence_message()
            return {}
        elif len (clip.versions[0].tracks) != 1:
            sequence_message()
            return {}
        elif len (clip.versions[0].tracks[0].segments) != 1:
            sequence_message()
            return {}
        
        timewarp_effect = None
        for effect in effects:
            if effect.type == 'Timewarp':
                timewarp_effect = effect
                break
        
        if not timewarp_effect:
            effect_message()
            return {}
        
        temp_setup_path = '/var/tmp/temporary_tw_setup.timewarp_node'
        try:
            timewarp_effect.save_setup(temp_setup_path)
            with open(temp_setup_path, 'r') as tw_setup_file:
                tw_setup_string = tw_setup_file.read()
                tw_setup_file.close()
        except Exception as e:
            self.message(pformat(e))
            return {}

        # match with 'include_timeline_fx = False' does not always work
        # therefore we need to delete original timewarp 
        # before matching and add it again from setup

        flame.delete(timewarp_effect)
        clip.commit()

        frame_value_map = self.bake_flame_tw_setup(tw_setup_string)
        if not frame_value_map:
            self.message('Unable to parse Timewarp effect setup')
            return {}
        
        try:
            clip_matched = clip.versions[0].tracks[0].segments[0].match(clip.parent, include_timeline_fx = True)
            effects = clip_matched.versions[0].tracks[0].segments[0].effects
            timewarp_effect = None
            for effect in effects:
                if effect.type == 'Timewarp':
                    timewarp_effect = effect
                    break
            flame.delete(timewarp_effect)

            head = clip_matched.versions[0].tracks[0].segments[0].head
            head = 0 if head == 'infinite' else head
            tail = clip_matched.versions[0].tracks[0].segments[0].tail
            tail = 0 if tail == 'infinite' else tail
            if head:
                clip_matched.versions[0].tracks[0].segments[0].trim_head(-1 * head)
            if tail:
                clip_matched.versions[0].tracks[0].segments[0].trim_tail(-1 * tail)
            clip_matched.render()
            clip_matched.commit()
        except Exception as e:
            self.message(pformat(e))
            return {}
        
        new_timewarp = clip.versions[0].tracks[0].segments[0].create_effect('Timewarp')
        new_timewarp.load_setup(temp_setup_path)
        os.remove(temp_setup_path)

        flame.execute_shortcut('Save Project')
        flame.execute_shortcut('Refresh Thumbnails')
                
        clip_matched.name.set_value(self.sanitized(clip.name.get_value()) + '_twml_src')
        temp_library_name = self.app_name + '_' + self.sanitized(clip.name.get_value()) + '_' + self.create_timestamp_uid()
        self.temp_library_name = temp_library_name
        self.temp_library = flame.projects.current_project.create_shared_library(temp_library_name)
        self.temp_library.acquire_exclusive_access()
        self.temp_library.open()
        flame.projects.current_project.refresh_shared_libraries()
        clip_matched = flame.media_panel.move(source_entries = clip_matched, destination = self.temp_library, duplicate_action = 'replace')[0]
        clip_matched.commit()
        flame.projects.current_project.refresh_shared_libraries()

        frames_map = {}
        for frame in frame_value_map.keys():
            frames_map[frame] = {
                'ratio': frame_value_map[frame] - int(frame_value_map[frame]),
                'incoming': {
                    'clip': clip_matched,
                    'wiretap_node_id': flame.PyClip.get_wiretap_node_id(clip_matched),
                    'frame_number': int(frame_value_map[frame])
                    },
                'outgoing': {
                    'clip': clip_matched,
                    'wiretap_node_id': flame.PyClip.get_wiretap_node_id(clip_matched),
                    'frame_number': int(frame_value_map[frame]) + 1
                    },
                'temp_library': self.temp_library,
                'destination': clip.parent
            }

        return frames_map

    def create_destination_node(self, selection, num_frames):
        import flame
        np = self.np

        clip = selection[0]
        self.destination_node_name = clip.name.get_value() + '_TWML'
        destination_node_id = ''
        # def CreateDestNode():
        try:
            server_handle = WireTapServerHandle('localhost')
            clip_node_id = clip.get_wiretap_node_id()
            clip_node_handle = WireTapNodeHandle(server_handle, clip_node_id)
            fmt = WireTapClipFormat()
            if not clip_node_handle.getClipFormat(fmt):
                raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())
            
            bits_per_channel = fmt.bitsPerPixel() // fmt.numChannels()
            self.bits_per_channel = bits_per_channel
            self.format_tag = fmt.formatTag()
            self.fmt = fmt

            self.temp_library.release_exclusive_access()
            node_id = self.temp_library.get_wiretap_node_id()
            parent_node_handle = WireTapNodeHandle(server_handle, node_id)
            destination_node_handle = WireTapNodeHandle()

            if not parent_node_handle.createClipNode(
                self.destination_node_name,  # display name
                fmt,  # clip format
                "CLIP",  # extended (server-specific) type
                destination_node_handle,  # created node returned here
            ):
                raise Exception(
                    "Unable to create clip node: %s." % parent_node_handle.lastError()
                )

            if not destination_node_handle.setNumFrames(int(num_frames)):
                raise Exception(
                    "Unable to set the number of frames: %s." % clip_node_handle.lastError()
                )
            
            dest_fmt = WireTapClipFormat()
            if not destination_node_handle.getClipFormat(dest_fmt):
                raise Exception(
                    "Unable to obtain clip format: %s." % clip_node_handle.lastError()
                )
            
            destination_node_id = destination_node_handle.getNodeId().id()

        except Exception as e:
            self.message('Error creating destination wiretap node: %s' % e)
        finally:
            server_handle = None
            clip_node_handle = None
            parent_node_handle = None
            destination_node_handle = None

        return destination_node_id

        # flame.schedule_idle_event(CreateDestNode)

    def bake_flame_tw_setup(self, tw_setup_string):
        np = self.np
        import xml.etree.ElementTree as ET

        # parses tw setup from flame and returns dictionary
        # with baked frame - value pairs
        
        def dictify(r, root=True):
            def string_to_value(s):
                if (s.find('-') <= 0) and s.replace('-', '', 1).isdigit():
                    return int(s)
                elif (s.find('-') <= 0) and (s.count('.') < 2) and \
                        (s.replace('-', '', 1).replace('.', '', 1).isdigit()):
                    return float(s)
                elif s == 'True':
                    return True
                elif s == 'False':
                    return False
                else:
                    return s

            from copy import copy

            if root:
                return {r.tag: dictify(r, False)}

            d = copy(r.attrib)
            if r.text:
                # d["_text"] = r.text
                d = r.text
            for x in r.findall('./*'):
                if x.tag not in d:
                    v = dictify(x, False)
                    if not isinstance (d, dict):
                        d = {}
                    if isinstance (v, str):
                        d[x.tag] = string_to_value(v)
                    else:
                        d[x.tag] = []
                if isinstance(d[x.tag], list):
                    d[x.tag].append(dictify(x, False))
            return d

        class FlameChannellInterpolator:
            # An attempt of a python rewrite of Julit Tarkhanov's original
            # Flame Channel Parsr written in Ruby.

            class ConstantSegment:
                def __init__(self, from_frame, to_frame, value):
                    self._mode = 'constant'
                    self.start_frame = from_frame
                    self.end_frame = to_frame
                    self.v1 = value

                def mode(self):
                    return self._mode

                def defines(self, frame):
                    return (frame < self.end_frame) and (frame >= self.start_frame)

                def value_at(self, frame):
                    return self.v1

            class LinearSegment(ConstantSegment):
                def __init__(self, from_frame, to_frame, value1, value2):
                    self.vint = (value2 - value1)
                    super().__init__(from_frame, to_frame, value1)
                    self._mode = 'linear'

                def value_at(self, frame):
                    on_t_interval = (frame - self.start_frame) / (self.end_frame - self.start_frame)
                    return self.v1 + (on_t_interval * self.vint)
                
                '''
                self.HERMATRIX = np.array([
                    [2,  -2,  1,  1],
                    [-3, 3,   -2, -1],
                    [0,   0,  1,  0],
                    [1,   0,  0,  0]
                ])
                '''

            class HermiteSegment(LinearSegment):
                def __init__(self, from_frame, to_frame, value1, value2, tangent1, tangent2):
                    self.start_frame, self.end_frame = from_frame, to_frame
                    frame_interval = (self.end_frame - self.start_frame)
                    self._mode = 'hermite'

                    self.HERMATRIX = np.array([
                        [0,  0,  0,  1],
                        [1,  1,  1,  1],
                        [0,  0,  1,  0],
                        [3,  2,  1,  0]
                    ])
                    self.HERMATRIX = np.linalg.inv(self.HERMATRIX)

                    # Default tangents in flame are 0, so when we do None.to_f this is what we will get
                    # CC = {P1, P2, T1, T2}
                    p1, p2, t1, t2 = value1, value2, tangent1 * frame_interval, tangent2 * frame_interval
                    self.hermite = np.array([p1, p2, t1, t2])
                    self.basis = np.dot(self.HERMATRIX, self.hermite)

                def value_at(self, frame):
                    if frame == self.start_frame:
                        return self.hermite[0]

                    # Get the 0 < T < 1 interval we will interpolate on
                    # Q[frame_] = P[ ( frame - 149 ) / (time_to - time_from)]
                    t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

                    # S[s_] = {s^3, s^2, s^1, s^0}
                    multipliers_vec = np.array([t ** 3, t ** 2, t ** 1, t ** 0])

                    # P[s_] = S[s].h.CC
                    interpolated_scalar = np.dot(self.basis, multipliers_vec)
                    return interpolated_scalar

            class HermiteSegmentQuartic(LinearSegment):
                '''
                Make sure not to confuse them with quartic (= degree 4) Hermite splines, 
                which are defined by 5 values per segment: function value and first derivative at both ends, 
                and one of the second derivatives.
                '''
                '''
                P(x0) = y0    =>    a0 + a1*x0 + a2*x0^2 + a3*x0^3 + a4*x0^4 = y0   --(1)
                P(x1) = y1    =>    a0 + a1*x1 + a2*x1^2 + a3*x1^3 + a4*x1^4 = y1   --(2)
                
                P'(x0) = tx0    =>    a1 + 2*a2*x0 + 3*a3*x0^2 + 4*a4*x0^3 = tx0   --(3)
                P'(x1) = tx1    =>    a1 + 2*a2*x1 + 3*a3*x1^2 + 4*a4*x1^3 = tx1   --(4)

                We have four equations (equations (1)-(4)) and five unknowns (a0, a1, a2, a3, a4). 
                To solve this system of equations, we can rewrite it in matrix form:

                A * X = B

                where A is the coefficient matrix, 
                X is the column vector of unknowns, 
                and B is the column vector of constants.

                A = | 1   x0   x0^2   x0^3   x0^4 |
                    | 1   x1   x1^2   x1^3   x1^4 |
                    | 0   1    2*x0   3*x0^2 4*x0^3 |
                    | 0   1    2*x1   3*x1^2 4*x1^3 |

                X = | a0 |
                    | a1 |
                    | a2 |
                    | a3 |
                    | a4 |

                B = | y0 |
                    | y1 |
                    | tx0 |
                    | tx1 |

                To solve for X, we can compute X = inv(A) * B, where inv(A) is the inverse of matrix A.
                Once we have the values of a0, a1, a2, a3, and a4, we can substitute them back 
                into the quartic polynomial P(x) to obtain the interpolated values for any desired x.

                P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
                
                '''
                def __init__(self, from_frame, to_frame, value1, value2, tangent1, tangent2):
                    self.start_frame, self.end_frame = from_frame, to_frame
                    frame_interval = (self.end_frame - self.start_frame)
                    self._mode = 'hermite'
                    self.a = 0
                    self.b = 0
                    self.value1 = value1
                    self.value2 = value2
                    self.tangent1 = tangent1
                    self.tangent2 = tangent2
                    self.frame_interval = frame_interval

                    self.HERMATRIX = np.array([
                                [2, -2, 1, 1],
                                [-3, 3, -2, -1],
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [2, 0, 0, 0]
                            ])
                    
                    # self.HERMATRIX = np.linalg.inv(self.HERMATRIX)
                    # pprint (self.HERMATRIX)

                    # Default tangents in flame are 0, so when we do None.to_f this is what we will get
                    # CC = {P1, P2, T1, T2}
                    p1, p2, t1, t2 = value1, value2, tangent1 * frame_interval, tangent2 * frame_interval
                    self.hermite = np.array([p1, p2, t1, t2])
                    pprint (self.hermite)
                    self.basis = np.dot(self.HERMATRIX, self.hermite)
                    pprint (self.basis)

                def value_at(self, frame):
                    if frame == self.start_frame:
                        return self.hermite[0]

                    # Get the 0 < T < 1 interval we will interpolate on
                    t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

                    # S[s_] = {s^4, s^3, s^2, s^1, s^0}
                    multipliers_vec = np.array([t ** 4, t ** 3, t ** 2, t ** 1, t ** 0])

                    # cubic functions
                    a0 = (1 - 3 * (t ** 2) + 2 * (t ** 3))
                    a1 = (3 * (t ** 2) - 2 * (t ** 3))
                    b0 = (t - 2* (t ** 2) + t**3)
                    b1 = -1 * (t ** 2) + (t ** 3)

                    # quatric functions
                    alpha = self.a
                    beta = self.b

                    aa0 = 1 + (alpha - 3)*(t ** 2) + 2 * (1 - alpha) * (t ** 3) + alpha * (t ** 4)
                    aa1 = (3 - alpha) * (t **2) + 2 * (alpha - 1) * (t ** 3) - alpha * (t ** 4)
                    bb0 = t + (beta - 2) * (t ** 2) + (1 - 2 * beta) * (t ** 3) + beta * (t ** 4)
                    bb1 = -1 * (beta + 1) * (t ** 2) + (2 * beta + 1) * (t ** 3) - beta * (t ** 4)

                    # P[s_] = S[s].h.CC
                    # interpolated_scalar = np.dot(self.basis, multipliers_vec)
                    p1, p2, t1, t2 = self.value1, self.value2, self.tangent1 * self.frame_interval, self.tangent2 * self.frame_interval
                    # interpolated_scalar = a0*p1 + a1*p2 + b0*t1 + b1*t2
                    interpolated_scalar = aa0*p1 + aa1*p2 + bb0*t1 + bb1*t2
                    return interpolated_scalar

            class HermiteSegmentQuartic5x5(LinearSegment):
                '''
                Make sure not to confuse them with quartic (= degree 4) Hermite splines, 
                which are defined by 5 values per segment: function value and first derivative at both ends, 
                and one of the second derivatives.
                '''
                '''
                P(x0) = y0    =>    a0 + a1*x0 + a2*x0^2 + a3*x0^3 + a4*x0^4 = y0   --(1)
                P(x1) = y1    =>    a0 + a1*x1 + a2*x1^2 + a3*x1^3 + a4*x1^4 = y1   --(2)
                
                P'(x0) = tx0    =>    a1 + 2*a2*x0 + 3*a3*x0^2 + 4*a4*x0^3 = tx0   --(3)
                P'(x1) = tx1    =>    a1 + 2*a2*x1 + 3*a3*x1^2 + 4*a4*x1^3 = tx1   --(4)

                We have four equations (equations (1)-(4)) and five unknowns (a0, a1, a2, a3, a4). 
                To solve this system of equations, we can rewrite it in matrix form:

                A * X = B

                where A is the coefficient matrix, 
                X is the column vector of unknowns, 
                and B is the column vector of constants.

                A = | 1   x0   x0^2   x0^3   x0^4 |
                    | 1   x1   x1^2   x1^3   x1^4 |
                    | 0   1    2*x0   3*x0^2 4*x0^3 |
                    | 0   1    2*x1   3*x1^2 4*x1^3 |

                X = | a0 |
                    | a1 |
                    | a2 |
                    | a3 |
                    | a4 |

                B = | y0 |
                    | y1 |
                    | tx0 |
                    | tx1 |

                To solve for X, we can compute X = inv(A) * B, where inv(A) is the inverse of matrix A.
                Once we have the values of a0, a1, a2, a3, and a4, we can substitute them back 
                into the quartic polynomial P(x) to obtain the interpolated values for any desired x.

                P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
                
                '''
                def __init__(self, from_frame, to_frame, value1, value2, tangent1, tangent2):
                    self.start_frame, self.end_frame = from_frame, to_frame
                    frame_interval = (self.end_frame - self.start_frame)
                    self._mode = 'hermite'

                    self.HERMATRIX = np.array([
                                [0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1],
                                [0, 0, 0, 1, 0],
                                [4, 3, 2, 1, 0],
                                [0, 0, 2, 0, 0]
                            ])
                    
                    self.HERMATRIX = np.linalg.inv(self.HERMATRIX)
                    pprint (self.HERMATRIX)

                    # Default tangents in flame are 0, so when we do None.to_f this is what we will get
                    # CC = {P1, P2, T1, T2}
                    p1, p2, t1, t2 = value1, value2, tangent1 * frame_interval, tangent2 * frame_interval
                    self.hermite = np.array([p1, p2, t1, t2, 0])
                    pprint (self.hermite)
                    self.basis = np.dot(self.HERMATRIX, self.hermite)
                    pprint (self.basis)

                def value_at(self, frame):
                    if frame == self.start_frame:
                        return self.hermite[0]

                    # Get the 0 < T < 1 interval we will interpolate on
                    t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

                    # S[s_] = {s^4, s^3, s^2, s^1, s^0}
                    multipliers_vec = np.array([t ** 4, t ** 3, t ** 2, t ** 1, t ** 0])

                    # P[s_] = S[s].h.CC
                    interpolated_scalar = np.dot(self.basis, multipliers_vec)
                    return interpolated_scalar

            class BezierSegment(LinearSegment):
                class Pt:
                    def __init__(self, x, y, tanx, tany):
                        self.x = x
                        self.y = y
                        self.tanx = tanx
                        self.tany = tany
                
                def __init__(self, x1, x2, y1, y2, t1x, t1y, t2x, t2y):
                    super().__init__(x1, x2, y1, y2)
                    self.a = self.Pt(x1, y1, t1x, t1y)
                    self.b = self.Pt(x2, y2, t2x, t2y)
                    self._mode = 'bezier'

                def value_at(self, frame):
                    if frame == self.start_frame:
                        return self.a.y
                    
                    t = self.approximate_t(frame, self.a.x, self.a.tanx, self.b.tanx, self.b.x)
                    vy = self.bezier(t, self.a.y, self.a.tany, self.b.tany, self.b.y)
                    return vy
                
                def bezier(self, t, a, b, c, d):
                    return a + (a*(-3) + b*3)*(t) + (a*3 - b*6 + c*3)*(t**2) + (-a + b*3 - c*3 + d)*(t**3)
                
                def clamp(self, value):
                    if value < 0:
                        return 0.0
                    elif value > 1:
                        return 1.0
                    else:
                        return value
                
                APPROXIMATION_EPSILON = 1.0e-09
                VERYSMALL = 1.0e-20
                MAXIMUM_ITERATIONS = 100
                
                def approximate_t(self, atX, p0x, c0x, c1x, p1x):
                    if atX - p0x < self.VERYSMALL:
                        return 0.0
                    elif p1x - atX < self.VERYSMALL:
                        return 1.0

                    u, v = 0.0, 1.0
                    
                    for i in range(self.MAXIMUM_ITERATIONS):
                        a = (p0x + c0x) / 2.0
                        b = (c0x + c1x) / 2.0
                        c = (c1x + p1x) / 2.0
                        d = (a + b) / 2.0
                        e = (b + c) / 2.0
                        f = (d + e) / 2.0
                        
                        if abs(f - atX) < self.APPROXIMATION_EPSILON:
                            return self.clamp((u + v) * 0.5)
                        
                        if f < atX:
                            p0x = f
                            c0x = e
                            c1x = c
                            u = (u + v) / 2.0
                        else:
                            c0x = a
                            c1x = d
                            p1x = f
                            v = (u + v) / 2.0
                    
                    return self.clamp((u + v) / 2.0)

            class ConstantPrepolate(ConstantSegment):
                def __init__(self, to_frame, base_value):
                    super().__init__(float('-inf'), to_frame, base_value)
                    self._mode = 'ConstantPrepolate'

                def value_at(self, frame):
                    return self.v1

            class ConstantExtrapolate(ConstantSegment):
                def __init__(self, from_frame, base_value):
                    super().__init__(from_frame, float('inf'), base_value)
                    self._mode = 'ConstantExtrapolate'

                def value_at(self, frame):
                    return self.v1
                
            class LinearPrepolate(ConstantPrepolate):
                def __init__(self, to_frame, base_value, tangent):
                    self.tangent = float(tangent)
                    super().__init__(to_frame, base_value)
                    self._mode = 'LinearPrepolate'

                def value_at(self, frame):
                    frame_diff = (self.end_frame - frame)
                    return self.v1 + (self.tangent * frame_diff)
                
            class LinearExtrapolate(ConstantExtrapolate):
                def __init__(self, from_frame, base_value, tangent):
                    self.tangent = float(tangent)
                    super().__init__(from_frame, base_value)
                    self._mode = 'LinearExtrapolate'

                def value_at(self, frame):
                    frame_diff = (frame - self.start_frame)
                    return self.v1 + (self.tangent * frame_diff)

            class ConstantFunction(ConstantSegment):
                def __init__(self, value):
                    super().__init__(float('-inf'), float('inf'), value)
                    self._mode = 'ConstantFunction'

                def defines(self, frame):
                    return True

                def value_at(self, frame):
                    return self.v1

            def __init__(self, channel):
                self.segments = []
                self.extrap = channel.get('Extrap', 'constant')

                if channel.get('Size', 0) == 0:
                    self.segments = [FlameChannellInterpolator.ConstantFunction(channel.get('Value', 0))]
                elif channel.get('Size') == 1 and self.extrap == 'constant':
                    self.segments = [FlameChannellInterpolator.ConstantFunction(channel.get('Value', 0))]
                elif channel.get('Size') == 1 and self.extrap == 'linear':
                    kframes = channel.get('KFrames')
                    frame = list(kframes.keys())[0]
                    base_value = kframes[frame].get('Value')
                    left_tangent = kframes[frame].get('LHandle_dY') / kframes[frame].get('LHandle_dX') * -1
                    right_tangent = kframes[frame].get('RHandle_dY') / kframes[frame].get('RHandle_dX')
                    self.segments = [
                        FlameChannellInterpolator.LinearPrepolate(frame, base_value, left_tangent),
                        FlameChannellInterpolator.LinearExtrapolate(frame, base_value, right_tangent)
                    ]
                else:
                    self.segments = self.create_segments_from_channel(channel)

            def sample_at(self, frame):
                if self.extrap == 'cycle':
                    return self.sample_from_segments(self.frame_number_in_cycle(frame))
                elif self.extrap == 'revcycle':
                    return self.sample_from_segments(self.frame_number_in_revcycle(frame))
                else:
                    return self.sample_from_segments(frame)

            def first_defined_frame(self):
                first_f = self.segments[0].end_frame
                if first_f == float('-inf'):
                    return 1
                return first_f

            def last_defined_frame(self):
                last_f = self.segments[-1].start_frame
                if last_f == float('inf'):
                    return 100
                return last_f

            def frame_number_in_revcycle(self, frame):
                animated_across = self.last_defined_frame() - self.first_defined_frame()
                offset = abs(frame - self.first_defined_frame())
                absolute_unit = offset % animated_across
                cycles = offset // animated_across
                if cycles % 2 == 0:
                    return self.first_defined_frame() + absolute_unit
                else:
                    return self.last_defined_frame() - absolute_unit

            def frame_number_in_cycle(self, frame):
                animated_across = self.last_defined_frame() - self.first_defined_frame()
                offset = frame - self.first_defined_frame()
                modulo = offset % animated_across
                return self.first_defined_frame() + modulo

            def create_segments_from_channel(self, channel):
                kframes = channel.get('KFrames')
                index_frames = list(kframes.keys())
                # First the prepolating segment
                segments = [self.pick_prepolation(channel.get('Extrap', 'constant'), kframes[index_frames[0]], kframes[index_frames[1]])]

                # Then all the intermediate segments, one segment between each pair of keys
                for index, key in enumerate(index_frames[:-1]):
                    segments.append(self.key_pair_to_segment(kframes[key], kframes[index_frames[index + 1]]))

                # and the extrapolator
                segments.append(self.pick_extrapolation(channel.get('Extrap', 'constant'), kframes[index_frames[-2]], kframes[index_frames[-1]]))
                return segments

            def sample_from_segments(self, at_frame):
                for segment in self.segments:
                    if segment.defines(at_frame):
                        return segment.value_at(at_frame)
                raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')
            
            def segment_mode(self, at_frame):
                for segment in self.segments:
                    if segment.defines(at_frame):
                        return segment.mode()
                raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')
            
            def get_segment(self, at_frame):
                for segment in self.segments:
                    if segment.defines(at_frame):
                        return segment
                raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')

            def pick_prepolation(self, extrap_symbol, first_key, second_key):
                if extrap_symbol == 'linear' and second_key:
                    if first_key.get('CurveMode') != 'linear':
                        first_key_left_slope = first_key.get('LHandle_dY') / first_key.get('LHandle_dX') * -1
                        return FlameChannellInterpolator.LinearPrepolate(
                            first_key.get('Frame'), 
                            first_key.get('Value'), 
                            first_key_left_slope)
                    else:
                        # For linear keys the tangent actually does not do anything, so we need to look a frame
                        # ahead and compute the increment
                        increment = (second_key.get('Value') - first_key.get('Value')) / (second_key.get('Frame') - first_key.get('Frame'))
                        return FlameChannellInterpolator.LinearPrepolate(first_key.get('Frame'), first_key.get('Value'), increment)
                else:
                    return FlameChannellInterpolator.ConstantPrepolate(first_key.get('Frame'), first_key.get('Value'))
            
            def pick_extrapolation(self, extrap_symbol, previous_key, last_key):
                if extrap_symbol != 'constant':
                    if previous_key and (last_key.get('CurveMode')  == 'linear' or last_key.get('CurveOrder')  == 'linear'):
                        # For linear keys the tangent actually does not do anything, so we need to look a frame
                        # ahead and compute the increment
                        increment = (last_key.get('Value') - previous_key.get('Value')) / (last_key.get('Frame') - previous_key.get('Frame'))
                        return FlameChannellInterpolator.LinearExtrapolate(last_key.get('Frame'), last_key.get('Value'), increment)
                    else:
                        last_key_right_slope = last_key.get('LHandle_dY') / last_key.get('LHandle_dX')
                        return FlameChannellInterpolator.LinearExtrapolate(last_key.get('Frame'), last_key.get('Value'), last_key_right_slope)
                else:
                    return FlameChannellInterpolator.ConstantExtrapolate(last_key.get('Frame'), last_key.get('Value'))

            def key_pair_to_segment(self, key, next_key):
                key_left_tangent = key.get('LHandle_dY') / key.get('LHandle_dX') * -1
                key_right_tangent = key.get('RHandle_dY') / key.get('RHandle_dX')
                next_key_left_tangent = next_key.get('LHandle_dY') / next_key.get('LHandle_dX') * -1
                next_key_right_tangent = next_key.get('RHandle_dY') / next_key.get('RHandle_dX')

                if key.get('CurveMode') == 'bezier':
                    return FlameChannellInterpolator.BezierSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'),
                        key.get('Value'), 
                        next_key.get('Value'),
                        float(key.get('Frame')) + float(key.get('RHandle_dX')), 
                        float(key.get('Value')) + float(key.get('RHandle_dY')),
                        float(next_key.get('Frame')) + float(next_key.get('LHandle_dX')),
                        float(next_key.get('Value')) + float(next_key.get('LHandle_dY'))
                        )
                
                elif (key.get('CurveMode') in ['natural', 'hermite']) and (key.get('CurveOrder') == 'cubic'):
                    return FlameChannellInterpolator.HermiteSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'), 
                        key.get('Value'), 
                        next_key.get('Value'),
                        key_right_tangent, 
                        next_key_left_tangent
                        )
                elif (key.get('CurveMode') in ['natural', 'hermite']) and (key.get('CurveOrder') == 'quartic'):
                    return FlameChannellInterpolator.HermiteSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'), 
                        key.get('Value'), 
                        next_key.get('Value'),
                        key_right_tangent, 
                        next_key_left_tangent
                        )
                elif key.get('CurveMode') == 'constant':
                    return FlameChannellInterpolator.ConstantSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'), 
                        key.get('Value')
                        )
                else:  # Linear and safe
                    return FlameChannellInterpolator.LinearSegment(
                        key.get('Frame'), 
                        next_key.get('Frame'), 
                        key.get('Value'), 
                        next_key.get('Value')
                        )

        tw_setup_xml = ET.fromstring(tw_setup_string)
        tw_setup = dictify(tw_setup_xml)

        start_frame = int(tw_setup['Setup']['Base'][0]['Range'][0]['Start'])
        end_frame = int(tw_setup['Setup']['Base'][0]['Range'][0]['End'])
        # TW_Timing_size = int(tw_setup['Setup']['State'][0]['TW_Timing'][0]['Channel'][0]['Size'][0]['_text'])

        TW_SpeedTiming_size = tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size']
        TW_RetimerMode = tw_setup['Setup']['State'][0]['TW_RetimerMode']

        frame_value_map = {}

        if TW_RetimerMode == 1:
            # 'Timing' channel is enough
            tw_channel = 'TW_Timing'
            channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
            if 'KFrames' in channel.keys():
                channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Value'])}
            interpolator = FlameChannellInterpolator(channel)
            for frame_number in range (start_frame, end_frame+1):
                frame_value_map[frame_number] = interpolator.sample_at(frame_number)
            return frame_value_map

        else:
            # speed - based timewarp seem to
            # work in a different way
            # depending on a segment mode

            tw_channel = 'TW_Speed'
            channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
            if 'KFrames' in channel.keys():
                channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Value'])}
            speed_channel = dict(channel)
            tw_channel = 'TW_SpeedTiming'
            channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
            if 'KFrames' in channel.keys():
                channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Value'])}
            speed_timing_channel = dict(channel)

            speed_interpolator = FlameChannellInterpolator(speed_channel)
            timing_interpolator = FlameChannellInterpolator(speed_timing_channel)

            for frame_number in range (start_frame, end_frame+1):
                frame_value_map[frame_number] = timing_interpolator.sample_at(frame_number)
                    
        return frame_value_map

    def write_exr(self, filename, width, height, red, green, blue, alpha, half_float = True, pixelAspectRatio = 1.0):
        np = self.np
        import struct

        MAGIC = 20000630
        VERSION = 2
        UINT = 0
        HALF = 1
        FLOAT = 2

        channels_list = ['B', 'G', 'R'] if not alpha.size else ['A', 'B', 'G', 'R']

        def write_attr(f, name, type, value):
            f.write(name.encode('utf-8') + b'\x00')
            f.write(type.encode('utf-8') + b'\x00')
            f.write(struct.pack('<I', len(value)))
            f.write(value)

        def get_channels_attr(channels_list):
            channel_list = b''
            for channel_name in channels_list:
                name_padded = channel_name[:254] + '\x00'
                bit_depth = 1 if half_float else 2
                pLinear = 0
                reserved = (0, 0, 0)  # replace with your values if needed
                xSampling = 1  # replace with your value
                ySampling = 1  # replace with your value
                channel_list += struct.pack(
                    f"<{len(name_padded)}s i B 3B 2i",
                    name_padded.encode(), 
                    bit_depth, 
                    pLinear, 
                    *reserved, 
                    xSampling, 
                    ySampling
                    )
            channel_list += struct.pack('c', b'\x00')

                # channel_list += (f'{i}\x00').encode('utf-8')
                # channel_list += struct.pack("<i4B", HALF, 1, 1, 0, 0)
            return channel_list
        
        def get_box2i_attr(x_min, y_min, x_max, y_max):
            return struct.pack('<iiii', x_min, y_min, x_max, y_max)

        with open(filename, 'wb') as f:
            # Magic number and version field
            f.write(struct.pack('I', 20000630))  # Magic number
            f.write(struct.pack('H', 2))  # Version field
            f.write(struct.pack('H', 0))  # Version field
            write_attr(f, 'channels', 'chlist', get_channels_attr(channels_list))
            write_attr(f, 'compression', 'compression', b'\x00')  # no compression
            write_attr(f, 'dataWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
            write_attr(f, 'displayWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
            write_attr(f, 'lineOrder', 'lineOrder', b'\x00')  # increasing Y
            write_attr(f, 'pixelAspectRatio', 'float', struct.pack('<f', pixelAspectRatio))
            write_attr(f, 'screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
            write_attr(f, 'screenWindowWidth', 'float', struct.pack('<f', 1.0))
            f.write(b'\x00')  # end of header

            # Scan line offset table size and position
            line_offset_pos = f.tell()
            pixel_data_start = line_offset_pos + 8 * height
            bytes_per_channel = 2 if half_float else 4
            # each scan line starts with 4 bytes for y coord and 4 bytes for pixel data size
            bytes_per_scan_line = width * len(channels_list) * bytes_per_channel + 8 

            for y in range(height):
                f.write(struct.pack('<Q', pixel_data_start + y * bytes_per_scan_line))

            channel_data = {'R': red, 'G': green, 'B': blue, 'A': alpha}

            # Pixel data
            for y in range(height):
                f.write(struct.pack('I', y))  # Line number
                f.write(struct.pack('I', bytes_per_channel * len(channels_list) * width))  # Pixel data size
                for channel in sorted(channels_list):
                    f.write(channel_data[channel][y].tobytes())
            f.close

    def write_dpx(self, filename, width, height, red, green, blue, alpha, bit_depth):
        import struct
        np = self.np

        depth = 3 if not alpha.size else 4
        if bit_depth == 8:
            dt = np.uint8
            red = (red * 255).astype(dt)
            green = (green * 255).astype(dt)
            blue = (blue * 255).astype(dt)
            alpha = (alpha * 255).astype(dt)
        elif bit_depth == 16:
            dt = np.uint16
            red = (red * 65535).astype(dt)
            green = (green * 65535).astype(dt)
            blue = (blue * 65535).astype(dt)
            alpha = (alpha * 65535).astype(dt)
        else:
            dt = np.float32

        arr = np.ones((height, width, depth), dtype=dt)

        arr[:,:,0] = red
        arr[:,:,1] = green
        arr[:,:,2] = blue
        if alpha.size:
            arr[:,:,3] = alpha
            
        file_size = 8192 + arr.size * bit_depth // 8

        new_meta = {}
        new_meta['colorimetry'] = 4
        new_meta['copyright'] = '\x00' * 200
        new_meta['creator'] = '\x00' * 100
        new_meta['data_sign'] = 0
        new_meta['depth'] = bit_depth
        new_meta['descriptor'] = 50 if not alpha.size else 51
        new_meta['ditto'] = 1
        new_meta['dpx_version'] = 'V1.0\x00\x00\x00\x00'
        new_meta['encoding'] = 0
        new_meta['encryption_key'] = 4294967295
        new_meta['endianness'] = 'be'
        new_meta['file_size'] = file_size
        new_meta['filename'] = os.path.basename(filename) + '\x00' * (100 - len(filename))
        new_meta['height'] = height
        new_meta['image_element_count'] = 1
        new_meta['image_element_description'] = 'IMAGE DESCRIPTION DATA        \x00P'
        new_meta['image_padding'] = 0
        new_meta['input_device_name'] = '\x00' * 32
        new_meta['input_device_sn'] = '\x00' * 32
        new_meta['line_padding'] = 0
        new_meta['magic'] = 'SDPX'
        new_meta['offset'] = 8192
        new_meta['orientation'] = 0
        new_meta['packing'] = 0 if bit_depth != 10 else 1
        new_meta['project_name'] = '\x00' * 200
        new_meta['timestamp'] = '\x00' * 24
        new_meta['transfer_characteristic'] = 4
        new_meta['width'] = width

        propertymap = [
            #(field name, offset, length, type)

            ('magic', 0, 4, 'magic'),
            ('offset', 4, 4, 'I'),
            ('dpx_version', 8, 8, 'utf8'),
            ('file_size', 16, 4, 'I'),
            ('ditto', 20, 4, 'I'),
            ('filename', 36, 100, 'utf8'),
            ('timestamp', 136, 24, 'utf8'),
            ('creator', 160, 100, 'utf8'),
            ('project_name', 260, 200, 'utf8'),
            ('copyright', 460, 200, 'utf8'),
            ('encryption_key', 660, 4, 'I'),

            ('orientation', 768, 2, 'H'),
            ('image_element_count', 770, 2, 'H'),
            ('width', 772, 4, 'I'),
            ('height', 776, 4, 'I'),

            ('data_sign', 780, 4, 'I'),
            ('descriptor', 800, 1, 'B'),
            ('transfer_characteristic', 801, 1, 'B'),
            ('colorimetry', 802, 1, 'B'),
            ('depth', 803, 1, 'B'),
            ('packing', 804, 2, 'H'),
            ('encoding', 806, 2, 'H'),
            ('line_padding', 812, 4, 'I'),
            ('image_padding', 816, 4, 'I'),
            ('image_element_description', 820, 32, 'utf8'),

            ('input_device_name', 1556, 32, 'utf8'),
            ('input_device_sn', 1588, 32, 'utf8')
        ]

        def writeDPX(f, image, meta):
            endianness = ">" if meta['endianness'] == 'be' else "<"
            for p in propertymap:
                if p[0] in meta:
                    f.seek(p[1])
                    if p[3] == 'magic':
                        bytes = ('SDPX' if meta['endianness'] == 'be' else 'XPDS').encode(encoding='UTF-8')
                    elif p[3] == 'utf8':
                        bytes = meta[p[0]].encode(encoding='UTF-8')
                    else:
                        bytes = struct.pack(endianness + p[3], meta[p[0]])
                    f.write(bytes)
            if meta['depth'] == 10:
                raw = ((((image[:,:,0] * 0x000003FF).astype(np.dtype(np.int32)) & 0x000003FF) << 22) 
                        | (((image[:,:,1] * 0x000003FF).astype(np.dtype(np.int32)) & 0x000003FF) << 12)
                        | (((image[:,:,2] * 0x000003FF).astype(np.dtype(np.int32)) & 0x000003FF) << 2)
                    )
            else:
                raw = image.flatten()

            if meta['endianness'] == 'be':
                raw = raw.byteswap()

            f.seek(meta['offset'])
            raw.tofile(f, sep="")

        with open(filename, 'wb') as f:
            writeDPX(f, arr, new_meta)
            f.close()

    def select_flow_res(self, flow_res):
        print ('select flow res')
        print (flow_res)

    def select_mode(self, mode_number):
        print ('select mode')
        print (mode_number)

    def flownet(self, img0, img1, ratio, model_path):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint

        import numpy as np
        
        if sys.platform == 'darwin':
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.set_printoptions(profile="full")
        torch.set_grad_enabled(False)

        print ('start')
        from torch import mps
        print (mps.driver_allocated_memory())

        # flip to BGR
        img0 = np.flip(img0, axis=2).transpose(2, 0, 1).copy()
        img1 = np.flip(img1, axis=2).transpose(2, 0, 1).copy()
        img0 = (torch.tensor(img0).to(device)).unsqueeze(0)
        img1 = (torch.tensor(img1).to(device)).unsqueeze(0)

        n, c, h, w = img0.shape
        
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        print ('padding')
        from torch import mps
        print (mps.driver_allocated_memory())

        # print (img0)
        # print (img1)

        print ('processing ratio %s' % ratio)

        def warp(tenInput, tenFlow):
            backwarp_tenGrid = {}
            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
                    1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
                    1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat(
                    [tenHorizontal, tenVertical], 1).to(device)

            tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                                tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            # return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
            cpu_g = g.to('cpu')
            cpu_tenInput = g.to('cpu')
            cpu_result = torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bicubic', padding_mode='border', align_corners=True)
            return cpu_result.to(device)

        def conv_leaky(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),        
                nn.LeakyReLU(0.2, True)
            )
        
        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
                nn.PReLU(out_planes)
            )

        def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                        kernel_size=4, stride=2, padding=1, bias=True),
                nn.PReLU(out_planes)
            )

        class Conv2(nn.Module):
            def __init__(self, in_planes, out_planes, stride=2):
                super(Conv2, self).__init__()
                self.conv1 = checkpoint(conv, in_planes, out_planes, 3, stride, 1)
                self.conv2 = checkpoint(conv, out_planes, out_planes, 3, 1, 1)

            def forward(self, x):
                x = checkpoint(self.conv1, x)
                x = checkpoint(self.conv2, x)
                return x

        class ContextNet(nn.Module):
            def __init__(self):
                print ('contextnet init')
                c = 32
                super(ContextNet, self).__init__()
                self.conv0 = Conv2(3, c)
                self.conv1 = Conv2(c, c)
                self.conv2 = Conv2(c, 2*c)
                self.conv3 = Conv2(2*c, 4*c)
                self.conv4 = Conv2(4*c, 8*c)

            def forward(self, x, flow):
                print ('contextnet forward')

                x = checkpoint(self.conv0, x)
                x = checkpoint(self.conv1, x)
                flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", 
                                    align_corners=False) * 0.5
                f1 = warp(x, flow)
                x = checkpoint(self.conv2, x)
                flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                                    align_corners=False) * 0.5
                f2 = warp(x, flow)
                x = checkpoint(self.conv3, x)
                flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                                    align_corners=False) * 0.5
                f3 = warp(x, flow)
                x = checkpoint(self.conv4, x)
                flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                                    align_corners=False) * 0.5
                f4 = warp(x, flow)
                return [f1, f2, f3, f4]

        class FusionNet(nn.Module):
            def __init__(self):
                super(FusionNet, self).__init__()
                c = 32
                self.conv0 = Conv2(10, c)
                self.down0 = Conv2(c, 2*c)
                self.down1 = Conv2(4*c, 4*c)
                self.down2 = Conv2(8*c, 8*c)
                self.down3 = Conv2(16*c, 16*c)
                self.up0 = deconv(32*c, 8*c)
                self.up1 = deconv(16*c, 4*c)
                self.up2 = deconv(8*c, 2*c)
                self.up3 = deconv(4*c, c)
                self.conv = nn.ConvTranspose2d(c, 4, 4, 2, 1)

            def forward(self, img0, img1, flow, c0, c1, context_scale):
                print ('fusionnet forward')
                warped_img0 = warp(img0, flow[:, :2])
                print ('warped_img0')
                warped_img1 = warp(img1, flow[:, 2:4])
                print ('warped_img1')

                x = checkpoint(self.conv0, torch.cat((warped_img0, warped_img1, flow), 1))
                print ('x = checkpoint(self.conv0, torch.cat((warped_img0, warped_img1, flow), 1))')

                s0 = checkpoint(self.down0, x)
                print ('s0 = checkpoint(self.down0, x)')
                s0 = F.interpolate(s0, scale_factor=context_scale, mode="bilinear", align_corners=False)
                print (s0.shape)

                s1 = checkpoint(self.down1, torch.cat((s0, c0[0], c1[0]), 1))
                print ('s1 = checkpoint(self.down1, torch.cat((s0, c0[0], c1[0]), 1))')

                s2 = checkpoint(self.down2, torch.cat((s1, c0[1], c1[1]), 1))
                print ('s2 = checkpoint(self.down2, torch.cat((s1, c0[1], c1[1]), 1))')
    
                s3 = checkpoint(self.down3, torch.cat((s2, c0[2], c1[2]), 1))
                print ('s3 = checkpoint(self.down3, torch.cat((s2, c0[2], c1[2]), 1))')

                x = checkpoint(self.up0, torch.cat((s3, c0[3], c1[3]), 1))
                
                del(s3)
                mps.empty_cache()
                
                x = checkpoint(self.up1, torch.cat((x, s2), 1))
                
                del(s2)
                mps.empty_cache()

                x = checkpoint(self.up2, torch.cat((x, s1), 1))

                del(s1)
                mps.empty_cache()

                x = checkpoint(self.up3, torch.cat((x, s0), 1))

                del(s0)
                mps.empty_cache()

                x = checkpoint(self.conv, x)
                return x, warped_img0, warped_img1

        class ResConv(nn.Module):
            def __init__(self, c, dilation=1):
                super(ResConv, self).__init__()
                self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
                self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = nn.LeakyReLU(0.2, True)

            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class IFBlock(nn.Module):
            def __init__(self, in_planes, c=64):
                super(IFBlock, self).__init__()
                self.conv0 = nn.Sequential(
                    conv_leaky(in_planes, c//2, 3, 2, 1),
                    conv_leaky(c//2, c, 3, 2, 1),
                    )
                self.convblock = nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.lastconv = nn.Sequential(
                    nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
                    nn.PixelShuffle(2)
                )

            def forward(self, x, flow, scale):
                x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                if flow is not None:
                    flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, flow), 1)
                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                return flow, mask

        class IFNet(nn.Module):
            def __init__(self, progress):
                super(IFNet, self).__init__()
                self.block0 = IFBlock(7, c=192)
                self.block1 = IFBlock(8+4, c=128)
                self.block2 = IFBlock(8+4, c=96)
                self.block3 = IFBlock(8+4, c=64)
                self.progress = progress
                # self.contextnet = ContextNet()
                # self.unet = Unet()

            def forward(self, img0, img1, timestep=0.5, scale_list=[8, 4, 2, 1], fastmode=True, ensemble=False):
                from torch import mps
                print ('timestep: %s' % timestep)
                # pprint (dir(mps))
                # print (mps.current_allocated_memory())
                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                # print (mps.current_allocated_memory())

                flow_list = []
                merged = []
                mask_list = []
                warped_img0 = img0
                warped_img1 = img1
                flow = None
                mask = None
                block = [self.block0, self.block1, self.block2, self.block3]
                # print (mps.current_allocated_memory())

                for i in range(len(scale_list)):
                    self.progress.info('Frame ' + str(self.progress.current_frame) + ': Processing: Optical flow pass {} of 4'.format(str(i+1)))
                    print ('block %s' % i)
                    if flow is None:
                        flow, mask = checkpoint(block[i], torch.cat((img0[:, :3], img1[:, :3], timestep), 1), None, scale_list[i])
                        if ensemble:
                            f1, m1 = checkpoint(block[i], torch.cat((img1[:, :3], img0[:, :3], 1-timestep), 1), None, scale_list[i])
                            flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                            mask = (mask + (-m1)) / 2
                    else:
                        f0, m0 = checkpoint(block[i], torch.cat((warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1), flow, scale_list[i])
                        if ensemble:
                            f1, m1 = checkpoint(block[i], torch.cat((warped_img1[:, :3], warped_img0[:, :3], 1-timestep, -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale_list[i])
                            f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                            m0 = (m0 + (-m1)) / 2
                        flow = flow + f0
                        mask = mask + m0

                    display_flow1 = flow[:, :2].cpu().detach().numpy()
                    display_flow1 = np.pad(display_flow1, ((0, 0), (0, 1), (0, 0), (0, 0)))
                    display_flow1 = display_flow1.transpose((0, 2, 3, 1)).squeeze(axis=0)
                    display_flow1 = (display_flow1 + 1) / 2
                    display_flow1 = np.flip(display_flow1, axis=2)
                    self.progress.update_interface_image(
                        display_flow1.copy(), 
                        self.progress.ui.flow1_label,
                        text = 'Flow pass {} of 4'.format(str(i+1))
                        )
                    display_flow2 = flow[:, 2:4].cpu().detach().numpy()
                    display_flow2 = np.pad(display_flow2, ((0, 0), (0, 1), (0, 0), (0, 0)))
                    display_flow2 = display_flow2.transpose((0, 2, 3, 1)).squeeze(axis=0)
                    display_flow2 = (display_flow2 + 1) / 2
                    display_flow2 = np.flip(display_flow2, axis=2)
                    self.progress.update_interface_image(
                        display_flow2.copy(), 
                        self.progress.ui.flow2_label,
                        text = 'Flow pass {} of 4'.format(str(i+1))
                        )

                    mask_list.append(mask)
                    flow_list.append(flow)
                    warped_img0 = warp(img0, flow[:, :2])
                    display_warped_0 = warped_img0.cpu().detach().numpy().transpose((0, 2, 3, 1)).squeeze(axis=0)
                    display_warped_0 = np.flip(display_warped_0, axis=2)
                    self.progress.update_interface_image(
                        display_warped_0.copy(), 
                        self.progress.ui.flow3_label,
                        text = 'Warp incoming pass {} of 4'.format(str(i+1))
                        )

                    warped_img1 = warp(img1, flow[:, 2:4])
                    display_warped_1 = warped_img0.cpu().detach().numpy().transpose((0, 2, 3, 1)).squeeze(axis=0)
                    display_warped_1 = np.flip(display_warped_1, axis=2)
                    self.progress.update_interface_image(
                        display_warped_1.copy(), 
                        self.progress.ui.flow4_label,
                        text = 'Warp outgoing pass {} of 4'.format(str(i+1))
                        )
        
                    merged.append((warped_img0, warped_img1))
                print ('sigmoid')
                mask_list[3] = torch.sigmoid(mask_list[3])
                merged[3] = merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])
                if not fastmode:
                    pass
                    '''
                    c0 = self.contextnet(img0, flow[:, :2])
                    c1 = self.contextnet(img1, flow[:, 2:4])
                    tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
                    res = tmp[:, :3] * 2 - 1
                    merged[3] = torch.clamp(merged[3] + res, 0, 1)
                    '''
                return flow_list, mask_list[3], merged

        class FlownetModel:
            def __init__(self, progress):
                self.progress = progress
                self.flownet = IFNet(progress)
                # self.contextnet = ContextNet()
                self.flownet.eval()
                self.flownet.to(device)

            def load_model(self, path, rank=0):
                self.progress.info('Frame ' + str(self.progress.current_frame) + ': Processing: Loading optical flow...')
                
                def convert(param):
                    return {
                        k.replace("module.", ""): v
                        for k, v in param.items()
                        if "module." in k
                    }
                
                # print ('{}/flownet.pkl'.format(path))
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location=device)), False)
                # self.contextnet.load_state_dict(convert(torch.load('/var/tmp/flameTimewarpML/trained_models/default/v2.4.model/contextnet.pkl', map_location=device)), False)
                # self.fusionnet.load_state_dict(convert(torch.load('/var/tmp/flameTimewarpML/trained_models/default/v2.4.model/unet.pkl', map_location=device)), False)

            def inference(self, img0, img1, timestep=0.5, scale=1.0):
                # imgs = torch.cat((img0, img1), 1)
                scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
                flow_list, mask, merged = self.flownet(img0, img1, timestep, scale_list)
                flow = flow_list[3]

                '''
                img0 = img0.cpu().detach()
                img1 = img1.cpu().detach()

                print ('flow')
                print (flow.dtype)
                print (flow.device)

                print ('img0')
                print (img0.type)
                print (img0.device)

                c0 = self.contextnet(img0, flow[:, :2])
                c1 = self.contextnet(img1, flow[:, 2:4])
                '''

                del self.flownet
                # del self.contextnet
                from torch import mps
                mps.empty_cache()

                return merged[3], flow_list

        class ContextNetModel:
            def __init__(self, progress):
                self.progress = progress
                self.contextnet = ContextNet()
                self.contextnet.eval()
                self.contextnet.to(device)

            def load_model(self, path, rank=0):
                self.progress.info('Frame ' + str(self.progress.current_frame) + ': Processing: Loading context net...')
                
                def convert(param):
                    return {
                        k.replace("module.", ""): v
                        for k, v in param.items()
                        if "module." in k
                    }
                self.contextnet.load_state_dict(convert(torch.load('/var/tmp/flameTimewarpML/trained_models/default/v2.4.model/contextnet.pkl', map_location=device)), False)

        class FusionNetModel:
            def __init__(self, progress):
                self.progress = progress
                self.fusionnet = FusionNet()
                self.fusionnet.eval()
                self.fusionnet.to(device)

            def load_model(self, path, rank=0):
                self.progress.info('Frame ' + str(self.progress.current_frame) + ': Processing: Loading context net...')
                
                def convert(param):
                    return {
                        k.replace("module.", ""): v
                        for k, v in param.items()
                        if "module." in k
                    }
                self.fusionnet.load_state_dict(convert(torch.load('/var/tmp/flameTimewarpML/trained_models/default/v2.4.model/unet.pkl', map_location=device)), False)

        class IFBlock2(nn.Module):
            def __init__(self, in_planes, scale=1, c=64):
                super(IFBlock2, self).__init__()
                self.scale = scale
                self.conv0 = nn.Sequential(
                    conv(in_planes, c, 3, 2, 1),
                    conv(c, 2*c, 3, 2, 1),
                    )
                self.convblock = nn.Sequential(
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                )        
                self.conv1 = nn.ConvTranspose2d(2*c, 4, 4, 2, 1)
                    
            def forward(self, x):
                if self.scale != 1:
                    x = F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear",
                                    align_corners=False)
                x = self.conv0(x)
                x = self.convblock(x)
                x = self.conv1(x)
                flow = x
                if self.scale != 1:
                    flow = F.interpolate(flow, scale_factor=self.scale, mode="bilinear",
                                        align_corners=False)
                return flow

        class IFNet2(nn.Module):
            def __init__(self):
                super(IFNet2, self).__init__()
                self.block0 = IFBlock2(6, scale=8, c=192)
                self.block1 = IFBlock2(10, scale=4, c=128)
                self.block2 = IFBlock2(10, scale=2, c=96)
                self.block3 = IFBlock2(10, scale=1, c=48)

            def forward(self, x, UHD=False):
                if UHD:
                    x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
                flow0 = self.block0(x)
                F1 = flow0
                F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 2.0
                warped_img0 = warp(x[:, :3], F1_large[:, :2])
                warped_img1 = warp(x[:, 3:], F1_large[:, 2:4])
                flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1_large), 1))
                F2 = (flow0 + flow1)
                F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 2.0
                warped_img0 = warp(x[:, :3], F2_large[:, :2])
                warped_img1 = warp(x[:, 3:], F2_large[:, 2:4])
                flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2_large), 1))
                F3 = (flow0 + flow1 + flow2)
                F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 2.0
                warped_img0 = warp(x[:, :3], F3_large[:, :2])
                warped_img1 = warp(x[:, 3:], F3_large[:, 2:4])
                flow3 = self.block3(torch.cat((warped_img0, warped_img1, F3_large), 1))
                F4 = (flow0 + flow1 + flow2 + flow3)
                return F4, [F1, F2, F3, F4]

        class FlownetV2:
            def __init__(self, progress):
                self.progress = progress
                self.flownet = IFNet2()
                self.flownet.eval()
                self.flownet.to(device)

            def load_model(self, path, rank=0):
                self.progress.info('Frame ' + str(self.progress.current_frame) + ': Processing: Loading context net...')
                
                def convert(param):
                    return {
                        k.replace("module.", ""): v
                        for k, v in param.items()
                        if "module." in k
                    }
                self.flownet.load_state_dict(convert(torch.load('/var/tmp/flameTimewarpML/trained_models/default/v2.4.model/flownet.pkl', map_location=device)), False)

        context_scale = 2
        print ('context scale %s' % context_scale)

        flownet_model = FlownetModel(self.progress)
        print ('Flownet init')
        from torch import mps
        print (mps.driver_allocated_memory())
        flownet_model.load_model(self.model_path)
        print ('Flownet after model load')
        from torch import mps
        print (mps.driver_allocated_memory())
        pred, flow_list = flownet_model.inference(img0, img1, timestep=ratio, scale=context_scale)
        for flow in flow_list:
            print ('flow shape')
            print (flow.shape)
        flow = flow_list[3]

        '''
        flownet_model = FlownetV2(self.progress)
        flownet_model.load_model(self.model_path)
        imgs = torch.cat((img0, img1), 1)
        flow2, _ = flownet_model.flownet(imgs, False)
        print ('flow2 shape')
        print (flow2.shape)

        # flow = flow2
        '''

        print ('Flownet after inference')
        from torch import mps
        print (mps.driver_allocated_memory())

        del(flownet_model)
        mps.empty_cache()

        print ('after del flownet model')
        from torch import mps
        print (mps.driver_allocated_memory())

        context_net_model = ContextNetModel(self.progress)
        context_net_model.load_model(self.model_path)

        print ('after loading contextnet')
        from torch import mps
        print (mps.driver_allocated_memory())

        flow_c = F.interpolate(flow, scale_factor=context_scale/2, mode="bilinear", align_corners=False) * 0.5
        c0 = context_net_model.contextnet(img0, flow_c[:, :2])
        print ('c0')
        for a in c0:
            print(a.shape)

        print ('after first contextnet')
        from torch import mps
        print (mps.driver_allocated_memory())

        c1 = context_net_model.contextnet(img1, flow_c[:, 2:4])

        print ('after second contextnet')
        from torch import mps
        print (mps.driver_allocated_memory())

        del (flow_c)
        del context_net_model
        mps.empty_cache()

        fusion_net_model = FusionNetModel(self.progress)
        
        print ('after fusion_net_model')
        from torch import mps
        print (mps.driver_allocated_memory())

        fusion_net_model.load_model(self.model_path)

        print ('fusion_net_model.load_model')
        from torch import mps
        print (mps.driver_allocated_memory())
        
        # flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        refine_output, warped_img0, warped_img1 = fusion_net_model.fusionnet(
            img0, img1, flow, c0, c1, context_scale)
        
        del (c0)
        del (c1)
        del (flow)
        del (flow_list)
        del (fusion_net_model)
        mps.empty_cache()

        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        res = F.interpolate(res, scale_factor=1/context_scale, mode="bilinear", align_corners=False)
        print ('res')
        print (res.shape)
        mask = torch.sigmoid(refine_output[:, 3:4])
        mask = F.interpolate(mask, scale_factor=1/context_scale, mode="bilinear", align_corners=False)
        print ('mask')
        print (mask.shape)

        print ('warped_img0')
        print (warped_img0.shape)
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        
        print ('after del contextnet and empty cache')
        from torch import mps
        print (mps.driver_allocated_memory())

        res_img = pred[0].cpu().detach().numpy().transpose(1, 2, 0)[:h, :w]
        res_img = np.flip(res_img, axis=2).copy()

        del (res)
        del (mask)
        del (merged_img)
        del (pred)

        mps.empty_cache()
        print ('after empty_cache')
        from torch import mps
        print (mps.driver_allocated_memory())


        return res_img

        
        # print (ratio)
        # print (img0)




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
            raise WireTapException("Unable to obtain clip format: %s." % clip_node_handle.lastError())
        
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
            # app.terminate_loops()
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
