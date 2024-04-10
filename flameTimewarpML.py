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

class ApplyModelDialog():

    def __init__(self, selection, mode):
        import flame
        import xml.etree.ElementTree as ET

        self.selection = selection
        self.mode = mode
        self.settings = settings

        try:
            self.fw = flameAppFramework(settings = settings)
        except:
            self.fw = None

        self.working_folder = self.fw.prefs.get('working_folder', os.path.expanduser('~'))
        if os.getenv('FLAMETWML_WORK_FOLDER'):
            self.working_folder = os.getenv('FLAMETWML_WORK_FOLDER')
        self.fw.prefs['working_folder'] = self.working_folder
        self.fw.save_prefs()

        self.model_path = self.fw.prefs.get(
            'model_path',
            os.path.join(os.path.dirname(__file__), 'models', 'flameTWML_v2.4.pth')
        )
        self.fw.prefs['model_path'] = self.model_path
        self.fw.save_prefs()

        self.verified_clips = self.verify_selection(selection, mode)
        if not self.verified_clips:
            return

        self.main_window()

    def verify_selection(self, selection, mode):
        import flame
        import xml.etree.ElementTree as ET

        if not selection:
            return False

        if mode == 'timewarp':
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
            
            def parse_message(e):
                dialog = flame.messages.show_in_dialog(
                    title = f'{settings["app_name"]}',
                    message = f'Error parsing TW setup file: {e}',
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
                        return False
                    if len (clip.versions[0].tracks) != 1:
                        sequence_message()
                        return False
                    if len (clip.versions[0].tracks[0].segments) != 1:
                        sequence_message()
                        return False
                    
                    effects = clip.versions[0].tracks[0].segments[0].effects
                    if not effects:
                        effect_message()
                        return False

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
                                return False
                            # pprint (tw_setup)
                            verified = True
                    
                    if not verified:
                        effect_message()
                        return

                    verified_clips.append((clip, tw_setup_string))
            
            os.remove(temp_setup_path)
            return verified_clips

        return []

    def main_window(self):

        def open_browser():
            """
            Open Flame file browser to choose export path.
            """

            path = pyflame.file_browser(
                path=self.export_path_entry.text(),
                title='Choose export path',
                select_directory=True,
                window_to_hide=[self.window],
            )

            if path:
                self.export_path_entry.setText(path)
                self.working_folder = path
                self.fw.prefs['working_folder'] = self.working_folder
                self.fw.save_prefs()

        def open_model_browser():
            self.window.hide()
            import flame
            flame.browser.show(
                title = 'Select flameTimewarpML Model:',
                extension = 'pth',
                default_path = os.path.dirname(self.model_path),
                multi_selection = False)
            if len(flame.browser.selection) > 0:
                self.model_path = flame.browser.selection[0]
                self.fw.prefs['model_path'] = self.model_path
                self.fw.save_prefs()
            self.window.show()

        # Create export and apply window
        self.window = PyFlameQDialog(
            width=800,
            height=200,
            title=f'{settings["app_name"]} <small>{settings["version"]}',
        )

        # Labels
        self.export_path_label = PyFlameLabel(
            text='Export Path',
        )

        self.model_path_label = PyFlameLabel(
            text='Model Path',
        )

        # Entries
        self.export_path_entry = PyFlameLineEdit(
            text=self.working_folder,
            max_width=1000,
        )

        self.model_path_entry = PyFlameLineEdit(
            text=self.model_path,
            max_width=1000,
        )

        # Buttons
        self.path_browse_button = PyFlameButton(
            text='Browse',
            connect=open_browser,
        )

        self.model_browse_button = PyFlameButton(
            text='Browse',
            connect=open_model_browser,
        )

        self.export_and_apply_button = PyFlameButton(
            text='Export and Apply',
            connect=self.apply,
            color=Color.BLUE,
        )

        self.cancel_button = PyFlameButton(
            text='Cancel',
            connect=self.window.close,
        )

        # Window layout
        grid_layout = QtWidgets.QGridLayout()

        grid_layout.setRowMinimumHeight(1, 30)
        grid_layout.setColumnMinimumWidth(2, 150)
        grid_layout.setColumnMinimumWidth(3, 150)

        grid_layout.addWidget(self.export_path_label, 0, 0)
        grid_layout.addWidget(self.export_path_entry, 0, 1, 1, 4)
        grid_layout.addWidget(self.path_browse_button, 0, 5)
 
        grid_layout.addWidget(self.model_path_label, 1, 0)
        grid_layout.addWidget(self.model_path_entry, 1, 1, 1, 4)
        grid_layout.addWidget(self.model_browse_button, 1, 5)

        grid_layout.addWidget(self.cancel_button, 2, 4)
        grid_layout.addWidget(self.export_and_apply_button, 2, 5,)

        # Add layout to window
        self.window.add_layout(grid_layout)

        self.window.show()   

    def apply(self):
        self.window.hide()

        if self.mode == 'timewarp':
            self.apply_timewarp()
            
        # Close export and apply window
        self.window.close()

    def apply_timewarp(self):
        hold_konsole = self.settings.get('debug')
        cmd_strings = []
        number_of_clips = 0

        for clip, tw_setup_string in self.verified_clips:
            number_of_clips += 1
            clip_name = clip.name.get_value()
            tw_clip_name = self.fw.sanitized(clip_name) + '_TWML' + '_' + self.fw.create_timestamp_uid()

            result_folder = os.path.abspath(
                os.path.join(
                    self.working_folder, 
                    tw_clip_name
                    )
                )
            
            if os.path.isdir(result_folder):
                self.window.hide()
                dialog = flame.messages.show_in_dialog(
                    title = f'{settings["app_name"]}',
                    message = 'Please select single-track clips with no versions or edits',
                    type = 'question',
                    buttons = ['Owerwrite'],
                    cancel_button = 'Cancel')
                
                if dialog == 'Cancel':
                    return False
                
                self.window.show()

            clip.render()
            source_clip_folder = os.path.join(result_folder, 'source')

            if clip.bit_depth == 32:
                export_preset = os.path.join(os.path.dirname(__file__), 'presets', 'source_export32.xml')
            else:
                export_preset = os.path.join(os.path.dirname(__file__), 'presets', 'source_export.xml')
            
            self.export_clip(clip, source_clip_folder, export_preset=export_preset)

            record_in = clip.versions[0].tracks[0].segments[0].record_in.relative_frame
            record_out = clip.versions[0].tracks[0].segments[0].record_out.relative_frame

            json_info = {}
            json_info['mode'] = 'timewarp'
            json_info['input'] = source_clip_folder
            json_info['output'] = result_folder
            json_info['clip_name'] = tw_clip_name
            json_info['model'] = self.fw.prefs.get('trained_models_folder')
            json_info['setup'] = tw_setup_string
            json_info['record_in'] = record_in
            json_info['record_out'] = record_out

            lockfile_path = os.path.join(
                result_folder,
                f'{tw_clip_name}.json'
            )

            import json
            try:
                with open(lockfile_path, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
            except Exception as e:
                dialog = flame.messages.show_in_dialog(
                    title = f'{settings["app_name"]}',
                    message = f'Unable to save {lockfile_path}: {e}',
                    type = 'error',
                    buttons = ['Ok'])
                return False

            self.run_inference()

    def run_inference(self):
        import platform

        conda_python_path = os.path.join(
            os.path.dirname(__file__),
            'packages',
            '.miniconda',
            'twml',
            'bin',
            'python'
            )
        inference_script_path = os.path.join(
            os.path.dirname(__file__),
            'pytorch',
            'inference.py'
        )

        import subprocess
        subprocess.Popen([conda_python_path, inference_script_path])

        '''
        if platform.system() == 'Darwin':

            cmd_prefix = """tell application "Terminal" to activate do script "clear; """
            cmd_prefix += f'source {conda_env_path}/bin/activate;'
            cmd_prefix += f'{conda_env_path}/bin/python"'
            ml_cmd = cmd_prefix

            import subprocess
            subprocess.Popen(['osascript', '-e', ml_cmd])
        else:
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
        '''

    def export_clip(self, clip, export_dir, export_preset = None):
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
    
    def about_dialog():
        pass

    def timewarp(selection):
        ApplyModelDialog(selection, mode='timewarp')

    def fluidmorph(selection):
        ApplyModelDialog(selection, mode='fluidmorph')

    def deduplicate(selection):
        ApplyModelDialog(selection, mode='deduplicate')

    menu = [
        {
            'name': settings['menu_group_name'],
            'actions': [
                {
                    'name': 'Fill / Remove Duplicate Frames',
                    'execute': deduplicate,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': 'Create Fluidmorph Transition',
                    'execute': fluidmorph,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': "Timewarp from Flame's TW effect",
                    'execute': timewarp,
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