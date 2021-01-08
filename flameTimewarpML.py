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
DEBUG = True

__version__ = 'v0.0.4.002'

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
            from PySide2 import QtWidgets
            msg = 'Sorry folks, this stuff is linux-only at the moment'
            mbox = QtWidgets.QMessageBox()
            mbox.setWindowTitle('flameTimewrarpML')
            mbox.setText(msg)
            mbox.setStyleSheet('QLabel{min-width: 800px;}')
            mbox.exec_()
            sys.exit()

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

        if not self.prefs_global.get('bundle_location'):
            self.bundle_location = os.path.join(
                os.path.expanduser('~'),
                self.bundle_name)
        else:
            self.bundle_location = self.prefs_global.get('bundle_location')

        #    self.prefs_global['menu_auto_refresh'] = {
        #        'media_panel': True,
        #        'batch': True,
        #        'main_menu': True
        #    }
        
        self.apps = []

        self.bundle_id = str(abs(hash(__version__)))
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

    def log(self, message):
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
            self.log('preferences contents:\n' + pformat(self.prefs))
        except:
            self.log('unable to load preferences from %s' % prefs_file_path)

        try:
            prefs_file = open(prefs_user_file_path, 'r')
            self.prefs_user = pickle.load(prefs_file)
            prefs_file.close()
            self.log('preferences loaded from %s' % prefs_user_file_path)
            self.log('preferences contents:\n' + pformat(self.prefs_user))
        except:
            self.log('unable to load preferences from %s' % prefs_user_file_path)

        try:
            prefs_file = open(prefs_global_file_path, 'r')
            self.prefs_global = pickle.load(prefs_file)
            prefs_file.close()
            self.log('preferences loaded from %s' % prefs_global_file_path)
            self.log('preferences contents:\n' + pformat(self.prefs_global))

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
            
        self.log('bundle_id: %s lenght %s' % (self.bundle_id, len(script)))
        
        if os.path.isdir(bundle_path):
            try:
                cmd = 'rm -rf ' + os.path.abspath(bundle_path)
                self.log('removing existing bundle folder')
                self.log('executing: %s' % cmd)
                os.system(cmd)
            except Exception as e:
                self.show_exception(e)
                return False

        try:
            self.log('creating new bundle folder: %s' % bundle_path)
            os.makedirs(bundle_path)
        except Exception as e:
            self.show_exception(e)
            return False

        start_position = script.rfind('# bundle payload starts here') + 33
        payload = script[start_position:-4]
        payload_dest = os.path.join(self.bundle_location, 'bundle.tar')
        
        try:
            import base64
            self.log('unpacking payload: %s' % payload_dest)
            with open(payload_dest, 'wb') as payload_file:
                payload_file.write(base64.b64decode(payload))
                payload_file.close()
            cmd = 'tar xf ' + payload_dest + ' -C ' + self.bundle_location + '/'
            self.log('executing: %s' % cmd)
            os.system(cmd)
            self.log('cleaning up %s' % payload_dest)
            os.remove(payload_dest)
        except Exception as e:
            self.show_exception(e)
            return False

        delta = time.time() - start
        self.log('bundle extracted to %s' % bundle_path)
        self.log('extracting bundle took %s sec' % str(delta))

        del payload
        del script

        env_folder = os.path.join(self.bundle_location, 'miniconda3') 
        self.install_env(env_folder)
        self.install_env_packages(env_folder)

        try:
            with open(os.path.join(bundle_path, 'bundle_id'), 'w+') as bundle_id_file:
                bundle_id_file.write(self.bundle_id)
        except Exception as e:
            self.show_exception(e)
            return False
        
        self.log('flameTimewarpML has finished installing its packages')

        return True
                    
    def install_env(self, env_folder):
        env_backup_folder = env_folder + '_previous'
        if os.path.isdir(env_folder):
            try:
                cmd = 'mv ' + env_folder + ' ' + env_backup_folder
                self.log('executing %s' % cmd)
                os.system(cmd)
            except Exception as e:
                self.show_exception(e)
                return False

        start = time.time()
        self.log('installing Miniconda into %s' % env_folder)
        installer_file = os.path.join(self.bundle_location, 'bundle', 'miniconda.package', 'Miniconda3-Linux-x86_64.sh')
        cmd = '/bin/sh ' + installer_file + ' -b -p ' + env_folder
        cmd += ' 2>&1 | tee ' + os.path.join(self.bundle_location, 'miniconda_install.log')
        self.log('executing: %s' % cmd)
        os.system(cmd)
        delta = time.time() - start
        self.log('installing Miniconda took %s sec' % str(delta))

    def install_env_packages(self, env_folder):
        start = time.time()
        self.log('installing Miniconda packages')
        cmd = """/usr/bin/bash -c 'eval "$(""" + os.path.join(env_folder, 'bin', 'conda') + ' shell.bash hook)"; conda activate; '
        cmd += 'pip3 install -r ' + os.path.join(self.bundle_location, 'bundle', 'requirements.txt') + ' --no-index --find-links '
        cmd += os.path.join(self.bundle_location, 'bundle', 'miniconda.package', 'packages')
        cmd += ' 2>&1 | tee '
        cmd += os.path.join(self.bundle_location, 'miniconda_packages_install.log')
        cmd += "'"

        self.log('Executing command: %s' % cmd)
        os.system(cmd)
        delta = time.time() - start
        self.log('installing Miniconda packages took %s sec' % str(delta))

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

        msg = 'flameTimeWarpML is going to unpack its bundle\nin background and run additional package scrips.\nThis may take a little while, check Flame console'
        dmsg = 'To be able to run flameTimeWarpML needs python environment that is newer then the one provided with Flame '
        dmsg += 'as well as some additional ML and computer-vision dependancies like PyTorch and OpenCV should be avaliable. '
        dmsg += 'flameTimeWarpML is going to unpack its bundle into "%s" ' % bundle_path
        dmsg += 'and then from there create a separate Miniconda3 installation and within this separate '
        dmsg += 'virtual python environment it is going to install dependencies needed'

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


class flameTimewrapML(flameMenuApp):
    def __init__(self, framework):
        flameMenuApp.__init__(self, framework)

    def build_menu(self):
        def scope_clip(selection):
            import flame
            for item in selection:
                if isinstance(item, (flame.PyClip)):
                    return True
            return False

        if not self.flame:
            return []
        
        menu = {'actions': []}
        menu['name'] = self.menu_group_name

        menu_item = {}
        menu_item['name'] = 'Create Slowmotion with ML'
        menu_item['execute'] = self.slowmo
        menu_item['isEnabled'] = scope_clip
        menu['actions'].append(menu_item)

        menu_item = {}
        menu_item['name'] = 'Version: ' + __version__
        menu_item['execute'] = self.slowmo
        menu_item['isEnabled'] = False
        menu['actions'].append(menu_item)

        return menu

    def slowmo(self, selection):
        result = self.slowmo_dialog()
        if result:
            import flame
            for item in selection:
                if isinstance(item, (flame.PyClip)):
                    self.export_clip(item, result.get('folder'))
                    cmd = """konsole -e bash -c 'eval "$(/opt/Autodesk/flame_2020.2/miniconda3/bin/conda shell.bash hook)"; conda activate; sleep 3'"""
                    os.system(cmd)

    def slowmo_dialog(self, *args, **kwargs):
        from PySide2 import QtWidgets, QtCore

        self.new_speed = 1
        self.new_speed_list = {
            1: '1/2',
            2: '1/4',
            3: '1/8',
            4: '1/16' 
        }

        self.working_folder = '/var/tmp'
        # flameMenuNewBatch_prefs = self.framework.prefs.get('flameMenuNewBatch', {})
        # self.asset_task_template =  flameMenuNewBatch_prefs.get('asset_task_template', {})

        window = QtWidgets.QDialog()
        window.setMinimumSize(280, 180)
        window.setWindowTitle('Create Slowmotion with ML')
        window.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        window.setStyleSheet('background-color: #313131')

        screen_res = QtWidgets.QDesktopWidget().screenGeometry()
        window.move((screen_res.width()/2)-150, (screen_res.height() / 2)-180)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignTop)

        # New Speed hbox
        new_speed_hbox = QtWidgets.QHBoxLayout()
        new_speed_hbox.setAlignment(QtCore.Qt.AlignCenter)

        # New Speed label

        lbl_NewSpeed = QtWidgets.QLabel('Slow Motion Speed', window)
        lbl_NewSpeed.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_NewSpeed.setMinimumHeight(28)
        lbl_NewSpeed.setMaximumHeight(28)
        lbl_NewSpeed.setAlignment(QtCore.Qt.AlignCenter)
        new_speed_hbox.addWidget(lbl_NewSpeed)

        lbl_NewSpeedSpacer = QtWidgets.QLabel('', window)
        lbl_NewSpeedSpacer.setAlignment(QtCore.Qt.AlignCenter)
        lbl_NewSpeedSpacer.setMinimumSize(28, 28)
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

        vbox.addLayout(new_speed_hbox)

        # Work Folder Label

        lbl_WorkFolder = QtWidgets.QLabel('Working folder', window)
        lbl_WorkFolder.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_WorkFolder.setMinimumHeight(28)
        lbl_WorkFolder.setMaximumHeight(28)
        lbl_WorkFolder.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(lbl_WorkFolder)

        # Work Folder Text Field
        def txt_WorkFolder_textChanged():
            self.working_folder = txt_WorkFolder.text()
        txt_WorkFolder = QtWidgets.QLineEdit('', window)
        txt_WorkFolder.setFocusPolicy(QtCore.Qt.ClickFocus)
        txt_WorkFolder.setMinimumSize(280, 28)
        txt_WorkFolder.setStyleSheet('QLineEdit {color: #9a9a9a; background-color: #373e47; border-top: 1px inset #black; border-bottom: 1px inset #545454}')
        txt_WorkFolder.setText(self.working_folder)
        txt_WorkFolder.textChanged.connect(txt_WorkFolder_textChanged)
        vbox.addWidget(txt_WorkFolder)

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
            return {
                'speed': self.new_speed,
                'folder': self.working_folder
            }
        else:
            return {}

    def export_clip(self, clip, folder):
        import flame
        import traceback

        clip_name = clip.name.get_value()
        export_dir = os.path.join(folder, clip_name)
        if not os.path.isdir(export_dir):
            self.log('creating folders: %s' % export_dir)
            try:
                os.makedirs(export_dir)
            except Exception as e:
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
def cleanup(apps, app_framework):    
    if apps:
        if DEBUG:
            print ('[DEBUG %s] unloading apps:\n%s' % ('flameMenuSG', pformat(apps)))
        while len(apps):
            app = apps.pop()
            if DEBUG:
                print ('[DEBUG %s] unloading: %s' % ('flameMenuSG', app.name))
            del app        
        del apps

    if app_framework:
        print ('PYTHON\t: %s cleaning up' % app_framework.bundle_name)
        app_framework.save_prefs()
        del app_framework

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