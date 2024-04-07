# type: ignore
'''
flameTimewarpML
Flame 2023.2 and higher
written by Andrii Toloshnyy
andriy.toloshnyy@gmail.com
'''

import os
import sys
import importlib
from pprint import pprint

try:
    from PySide6 import QtWidgets, QtCore, QtGui
    using_pyside6 = True
except ImportError:
    from PySide2 import QtWidgets, QtCore, QtGui
    using_pyside6 = False

import flameTimewarpML_framework
importlib.reload(flameTimewarpML_framework)
from flameTimewarpML_framework import flameAppFramework

from pyflame_lib_flameTimewarpML import * # Import pyflame library for UI elements

settings = {
    'menu_group_name': 'Timewarp ML',
    'debug': False,
    'app_name': 'flameTimewarpML',
    'prefs_folder': os.getenv('FLAMETWML_PREFS'),
    'bundle_folder': os.getenv('FLAMETWML_BUNDLE'),
    'packages_folder': os.getenv('FLAMETWML_PACKAGES'),
    'temp_folder': os.getenv('FLAMETWML_TEMP'),
    'version': 'v0.4.5 dev 001',
}

class DatasetDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        try:
            self.fw = flameAppFramework()
        except:
            self.fw = None

        self.dataset_folder = self.fw.prefs.get('dataset_folder', os.path.expanduser('~'))
        if os.getenv('FLAMESMML_DATASET_FOLDER'):
            self.dataset_folder = os.getenv('FLAMESMML_DATASET_FOLDER')
        self.fw.prefs['dataset_folder'] = self.dataset_folder
        self.fw.save_prefs()

        self.setMinimumSize(480, 80)
        self.setWindowTitle('flameSimpleML: Choose folder to create dataset in:')

        # Setting up layouts
        self.layout = QtWidgets.QVBoxLayout(self)
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.pathLayout = QtWidgets.QHBoxLayout()

        # Spacer
        lbl_Spacer = QtWidgets.QLabel('', self)
        lbl_Spacer.setStyleSheet('QFrame {color: #989898; background-color: #313131}')
        lbl_Spacer.setMinimumHeight(4)
        lbl_Spacer.setMaximumHeight(4)
        lbl_Spacer.setAlignment(QtCore.Qt.AlignCenter)

        lbl_WorkFolder = QtWidgets.QLabel('Export folder', self)
        lbl_WorkFolder.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
        lbl_WorkFolder.setMinimumHeight(28)
        lbl_WorkFolder.setMaximumHeight(28)
        lbl_WorkFolder.setAlignment(QtCore.Qt.AlignCenter)
        self.pathLayout.addWidget(lbl_WorkFolder)

        def txt_WorkFolder_textChanged():
            self.dataset_folder = txt_WorkFolder.text()

        if os.getenv('FLAMESMML_DATASET_FOLDER'):
            lbl_WorkFolderPath = QtWidgets.QLabel(self.dataset_folder, self)
            lbl_WorkFolderPath.setStyleSheet('QFrame {color: #989898; background-color: #373737}')
            lbl_WorkFolderPath.setMinimumHeight(28)
            lbl_WorkFolderPath.setMaximumHeight(28)
            lbl_WorkFolderPath.setAlignment(QtCore.Qt.AlignCenter)
            self.pathLayout.addWidget(lbl_WorkFolderPath)

        else:
            # Work Folder Text Field
            hbox_workfolder = QtWidgets.QHBoxLayout()
            hbox_workfolder.setAlignment(QtCore.Qt.AlignLeft)

            txt_WorkFolder = QtWidgets.QLineEdit('', self)
            txt_WorkFolder.setFocusPolicy(QtCore.Qt.ClickFocus)
            txt_WorkFolder.setMinimumSize(280, 28)
            txt_WorkFolder.setStyleSheet('QLineEdit {color: #9a9a9a; background-color: #373e47; border-top: 1px inset #black; border-bottom: 1px inset #545454}')
            txt_WorkFolder.setText(self.dataset_folder)
            txt_WorkFolder.textChanged.connect(txt_WorkFolder_textChanged)
            hbox_workfolder.addWidget(txt_WorkFolder)

            btn_changePreset = QtWidgets.QPushButton('Choose', self)
            btn_changePreset.setFocusPolicy(QtCore.Qt.NoFocus)
            btn_changePreset.setMinimumSize(88, 28)
            btn_changePreset.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #424142; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                    'QPushButton:pressed {font:italic; color: #d9d9d9}')
            btn_changePreset.clicked.connect(self.chooseFolder)
            hbox_workfolder.addWidget(btn_changePreset, alignment = QtCore.Qt.AlignLeft)

            self.pathLineEdit = txt_WorkFolder
            self.pathLayout.addLayout(hbox_workfolder)

        select_btn = QtWidgets.QPushButton('Export', self)
        select_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        select_btn.setMinimumSize(128, 28)
        select_btn.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #424142; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                'QPushButton:pressed {font:italic; color: #d9d9d9}')
        select_btn.clicked.connect(self.accept)
        select_btn.setAutoDefault(True)
        select_btn.setDefault(True)

        cancel_btn = QtWidgets.QPushButton('Cancel', self)
        cancel_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        cancel_btn.setMinimumSize(128, 28)
        cancel_btn.setStyleSheet('QPushButton {color: #9a9a9a; background-color: #424142; border-top: 1px inset #555555; border-bottom: 1px inset black}'
                                'QPushButton:pressed {font:italic; color: #d9d9d9}')
        cancel_btn.clicked.connect(self.reject)

        # Adding widgets to the button layout
        self.buttonLayout.addWidget(cancel_btn)
        self.buttonLayout.addWidget(select_btn)

        # Adding layouts to the main layout
        self.layout.addLayout(self.pathLayout)
        self.layout.addWidget(lbl_Spacer)
        self.layout.addLayout(self.buttonLayout)

        self.setStyleSheet('background-color: #313131')
        
        # self.setWindowFlags(QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def chooseFolder(self):
        result_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', self.dataset_folder, QtWidgets.QFileDialog.ShowDirsOnly))
        if result_folder =='':
            return
        self.dataset_folder = result_folder
        self.fw.prefs['dataset_folder'] = self.dataset_folder
        self.fw.save_prefs()
        self.pathLineEdit.setText(self.dataset_folder)

    def fltw_dialog(self, *args, **kwargs):        
        # self.scan_trained_models_folder()
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


def get_media_panel_custom_ui_actions():
    def scope_clip(selection):
        try:
            import flame
            for item in selection:
                if isinstance(item, (flame.PyClip)):
                    return True
        except Exception as e:
            print (f'[{settings["app_name"]}]: Exception: {e}')
        return False
    
    def sanitized(text):
        import re

        if text is None:
            return None
        
        text = text.strip()
        exp = re.compile(u'[^\w\.-]', re.UNICODE)

        if isinstance(text, str):
            result = exp.sub('_', text)
        else:
            decoded = text.decode('utf-8')
            result = exp.sub('_', decoded).encode('utf-8')

        return re.sub('_\_+', '_', result)

    def create_timestamp_uid():
        import random
        import uuid
        from datetime import datetime

        def number_to_letter(number):
            # Map each digit to a letter
            mapping = {
                '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E',
                '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'
            }
            return ''.join(mapping.get(char, char) for char in number)

        uid = ((str(uuid.uuid4()).replace('-', '')).upper())
        uid = ''.join(random.sample(number_to_letter(uid), 4))
        timestamp = (datetime.now()).strftime('%Y%b%d_%H%M').upper()
        return f'{timestamp}_{uid}'
    
    def export_clip(clip, export_dir, export_preset = None):
        import flame

        if not os.path.isdir(export_dir):
            try:
                os.makedirs(export_dir)
            except Exception as e:
                issue = 'Unable to create folder to export'
                dialog = flame.messages.show_in_dialog(
                    title =issue,
                    message = f'{issue}:\n{export_dir}\n\nError:\n{e}',
                    type = 'error',
                    buttons = ['Ok'])
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

        exporter = flame.PyExporter()
        exporter.foreground = True

        if not export_preset:
            for visibility in range(4):
                export_preset_folder = flame.PyExporter.get_presets_dir(flame.PyExporter.PresetVisibility.values.get(visibility),
                                flame.PyExporter.PresetType.values.get(0))
                export_preset = os.path.join(export_preset_folder, 'OpenEXR', 'OpenEXR (16-bit fp Uncompressed).xml')
                if os.path.isfile(export_preset):
                    break

        exporter.export(clip, export_preset, export_dir, hooks=ExportHooks())

    def apply_model(selection):
        import flame

        apply_dialog = ApplyModelDialog()

        if apply_dialog.exec():
            first_clip_name = selection[0].name.get_value()
            result_folder = os.path.abspath(
                os.path.join(
                    apply_dialog.working_folder,
                    f'{sanitized(first_clip_name)}_ML_{create_timestamp_uid()}'
                    )
                )
            source_folder = os.path.abspath(
                os.path.join(
                    result_folder,
                    'src'
                    )
                )
            clip_number = 1
            for item in selection:
                if isinstance(item, (flame.PyClip)):
                    clip = item
                    source_clip_folder = os.path.join(source_folder, f'{clip_number:02}')
                    export_clip(clip, source_clip_folder)
                    clip_number += 1

            first_clip_parent = selection[0].parent

            flameSimpleMLInference(
                source_folder=source_folder,
                result_folder=result_folder,
                first_clip_parent = first_clip_parent,
                settings=settings
                )

    def train_model(selection):
        import flame

        if len (selection) < 2:
            dialog = flame.messages.show_in_dialog(
                title ='Dataset creaton error',
                message = 'Please select at least two clips. The channels of first selected clip, or several clips channels combined will act as input channels, and the last selected clip will be the target',
                type = 'error',
                buttons = ['Ok'])
            return

        dataset_dialog = DatasetDialog()

        if dataset_dialog.exec():
            first_clip_name = selection[0].name.get_value()
            dataset_folder = os.path.abspath(
                os.path.join(
                    dataset_dialog.dataset_folder,
                    f'{sanitized(first_clip_name)}_dataset_{create_timestamp_uid()}'
                    )
                )

            source_folder = os.path.abspath(
                os.path.join(
                    dataset_folder,
                    'source'
                    )
                )

            target_folder = os.path.abspath(
                os.path.join(
                    dataset_folder,
                    'target'
                    )
                )
            
            selected_clips = list(selection)
            target_clip = selected_clips.pop()
            export_clip(target_clip, target_folder)

            clip_number = 1
            for source_clip in selected_clips:
                source_clip_folder = os.path.join(source_folder, f'{clip_number:02}')
                export_clip(source_clip, source_clip_folder)
                clip_number += 1

            flame_version = flame.get_version()
            python_executable_path = f'/opt/Autodesk/python/{flame_version}/bin/python'
            script_folder = os.path.abspath(os.path.dirname(__file__))
            app_name = settings.get('app_name')
            version = settings.get('version')
            command = f'{python_executable_path} {script_folder}/train.py {dataset_folder}'
            msg = f'GUI for model training is not yet implemented in {app_name} {version}\n'
            msg += f'Training is currently possible with a command-line script. Please run'
            msg += f'\n\n"{command}"\n\n'
            msg += 'use --help flag for more options'
            dialog = flame.messages.show_in_dialog(
                title ='Train Model GUI is not yet implemented',
                message = msg,
                type = 'info',
                buttons = ['Copy', 'Ok'])
            if dialog == 'Copy':
                try:
                    from PySide6.QtWidgets import QApplication
                except ImportError:
                    from PySide2.QtWidgets import QApplication

                app = QApplication.instance()
                if not app:
                    app = QApplication(sys.argv)
                clipboard = app.clipboard()
                clipboard.setText(command)

    def flame_timewarp(selection):
        import flame
        import xml.etree.ElementTree as ET

        if not selection:
            return

        def sequence_message():
            dialog = flame.messages.show_in_dialog(
                title = f'{settings["app_name"]}',
                message = 'Please select single-track clips with no versions or edits',
                type = 'error',
                buttons = ['Ok'])

        def effect_message():
            dialog = flame.messages.show_in_dialog(
                title = f'{settings["app_name"]}',
                message = 'Please select clips with Timewarp Timeline FX',
                type = 'error',
                buttons = ['Ok'])
            
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

        for clip in selection:
            if isinstance(clip, (flame.PyClip)):
                if len(clip.versions) != 1:
                    sequence_message()
                    return
                if len (clip.versions[0].tracks) != 1:
                    sequence_message()
                    return
                if len (clip.versions[0].tracks[0].segments) != 1:
                    sequence_message()
                
                effects = clip.versions[0].tracks[0].segments[0].effects
                if not effects:
                    effect_message()
                    return

                verified = False
                for effect in effects:
                    if effect.type == 'Timewarp':
                        effect.save_setup(temp_setup_path)
                        with open(temp_setup_path, 'r') as tw_setup_file:
                            tw_setup_string = tw_setup_file.read()
                            tw_setup_file.close()
                            
                        tw_setup_xml = ET.fromstring(tw_setup_string)
                        tw_setup = dictify(tw_setup_xml)
                        try:
                            start = int(tw_setup['Setup']['Base'][0]['Range'][0]['Start'])
                            end = int(tw_setup['Setup']['Base'][0]['Range'][0]['End'])
                            TW_Timing_size = int(tw_setup['Setup']['State'][0]['TW_Timing'][0]['Channel'][0]['Size'][0]['_text'])
                            TW_SpeedTiming_size = int(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size'][0]['_text'])
                            TW_RetimerMode = int(tw_setup['Setup']['State'][0]['TW_RetimerMode'][0]['_text'])
                        except Exception as e:
                            parse_message(e)
                            return

                        # pprint (tw_setup)
                                
                        verified = True
                
                if not verified:
                    effect_message()
                    return

                verified_clips.append((clip, tw_setup_string))
        
        os.remove(temp_setup_path)
        dialog = DatasetDialog()
        dialog.fltw_dialog()

    def about_dialog():
        pass

    def dedup():
        pass

    def fluidmorph():
        pass

    menu = [
        {
            'name': settings['menu_group_name'],
            'actions': [
                {
                    'name': 'Fill / Remove Duplicate Frames',
                    'execute': apply_model,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': 'Create Fluidmorph Transition',
                    'execute': train_model,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': "Timewarp from Flame's TW effect",
                    'execute': flame_timewarp,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': f'Version: {settings["version"]}',
                    'execute': about_dialog,
                    'isVisible': scope_clip,
                    'isEnabled': False,
                },
            ],
        }
    ]

    return menu