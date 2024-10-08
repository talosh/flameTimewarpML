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
import threading
import time

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
    'version': 'v0.4.5 dev 004',
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
        except Exception as e:
            dialog = flame.messages.show_in_dialog(
                title = f'{settings["app_name"]}',
                message = f'Unable to init TimewarpmML framework: {e}',
                type = 'error',
                buttons = ['Ok'])
            return False

        self.working_folder = self.fw.prefs.get('working_folder', os.path.expanduser('~'))
        if os.getenv('FLAMETWML_WORK_FOLDER'):
            self.working_folder = os.getenv('FLAMETWML_WORK_FOLDER')
        self.fw.prefs['working_folder'] = self.working_folder
        self.fw.save_prefs()

        self.model_path = self.fw.prefs.get(
            'model_path',
            os.path.join(os.path.dirname(__file__), 'models', 'flownet4.pth')
        )
        self.fw.prefs['model_path'] = self.model_path
        self.fw.save_prefs()

        self.verified_clips = self.verify_selection(selection, mode)
        if not self.verified_clips:
            return

        self.loops = []
        self.threads = True
        
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

        elif mode == 'fluidmorph':
            verified_clips = []
            for item in selection:
                if isinstance(item, (flame.PyClip)):
                    verified_clips.append(item)

            if len(verified_clips) != 2:
                dialog = flame.messages.show_in_dialog(
                title = f'{settings["app_name"]}',
                message = 'Please select two clips of the same dimentions and length',
                type = 'error',
                buttons = ['Ok'])
                return
            return verified_clips
        
        elif mode == 'finetune':
            verified_clips = []
            for item in selection:
                if isinstance(item, (flame.PyClip)):
                    verified_clips.append(item)
            if not verified_clips:
                dialog = flame.messages.show_in_dialog(
                title = f'{settings["app_name"]}',
                message = 'Please select at least one clip to fine-tune',
                type = 'error',
                buttons = ['Ok'])
                return
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

            if not os.path.isfile(self.model_path):
                self.model_path = os.path.join(
                    os.path.dirname(__file__), 
                    'models',
                    'flownet4.pth'
                    )

            flame.browser.show(
                title = 'Select flameTimewarpML Model:',
                extension = 'pth',
                default_path = os.path.dirname(self.model_path),
                multi_selection = False)
            if len(flame.browser.selection) > 0:
                self.model_path = flame.browser.selection[0]
                self.model_path_entry.setText(self.model_path)
                self.fw.prefs['model_path'] = self.model_path
                self.fw.save_prefs()
            self.window.show()

        def iterative():
            self.fw.prefs['iterative'] = self.iterative_button.isChecked()
            self.fw.save_prefs()
        
        def half():
            self.fw.prefs['half'] = self.half_button.isChecked()
            self.fw.save_prefs()

        # Create export and apply window
        window_title = f'{settings["app_name"]} <small>{settings["version"]}'
        if self.mode == 'timewarp':
            window_title += ' [Timewarp from Flame]'
        elif self.mode == 'fluidmorph':
            window_title += ' [Fluidmorph]'
        elif self.mode == 'finetune':
            return self.main_window_finetune()

        self.window = PyFlameDialogWindow(
            width=800,
            height=256,
            title=window_title
        )

        # Labels
        self.options_label = PyFlameLabel(
            text='Options',
        )

        self.export_path_label = PyFlameLabel(
            text='Export Path',
        )

        self.model_path_label = PyFlameLabel(
            text='Model Path',
        )

        # Entries
        self.export_path_entry = PyFlameLineEdit(
            text=self.working_folder,
            max_width=True
        )

        self.model_path_entry = PyFlameLineEdit(
            text=self.model_path,
            max_width=True,
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

        self.iterative_button = PyFlamePushButton(
            text='Iterative',
            button_checked = self.fw.prefs.get('iterative', False),
            tooltip = 'Iteratively updates ratio - might help with smoother motion (Slower) ',
            connect=iterative
        )

        self.half_button = PyFlamePushButton(
            text='Use less VRAM',
            button_checked = self.fw.prefs.get('half', False),
            tooltip = 'Use less GPU memory to help with "CUDA: Out of memory" errors (Slower) ',
            connect=half
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
        grid_layout = PyFlameGridLayout()
        grid_layout.setRowMinimumHeight(0, 30)
        grid_layout.setRowMinimumHeight(1, 30)
        grid_layout.setRowMinimumHeight(2, 30)
        grid_layout.setRowMinimumHeight(3, 30)
        
        grid_layout.setColumnMinimumWidth(0, 120)
        grid_layout.setColumnMinimumWidth(1, 120)
        grid_layout.setColumnMinimumWidth(2, 120)
        grid_layout.setColumnMinimumWidth(3, 120)

        grid_layout.addWidget(self.options_label, 0, 0)
        grid_layout.addWidget(self.half_button, 0, 4)
        grid_layout.addWidget(self.iterative_button, 0, 5)

        grid_layout.addWidget(self.export_path_label, 1, 0)
        grid_layout.addWidget(self.export_path_entry, 1, 1, 1, 4)
        grid_layout.addWidget(self.path_browse_button, 1, 5)
 
        grid_layout.addWidget(self.model_path_label, 2, 0)
        grid_layout.addWidget(self.model_path_entry, 2, 1, 1, 4)
        grid_layout.addWidget(self.model_browse_button, 2, 5)

        grid_layout.addWidget(self.cancel_button, 3, 4)
        grid_layout.addWidget(self.export_and_apply_button, 3, 5)

        # Add layout to window
        self.window.add_layout(grid_layout)

        self.window.show()   

    def main_window_finetune(self):
        if not self.verified_clips:
            return

        self.src_model_path = self.model_path
        self.res_model_path = os.path.join(
            os.path.dirname(self.model_path),
            f'{self.verified_clips[0].name.get_value()}.pth'
        )
        self.finetune_export_path = os.path.join(
            self.working_folder,
            f'{self.verified_clips[0].name.get_value()}_TW_FINETUNE'
        )

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

        def open_dest_browser():
            """
            Open Flame file browser to choose export path.
            """

            path = pyflame.file_browser(
                path=self.res_model_path_entry.text(),
                title='Choose export path',
                select_directory=True,
                window_to_hide=[self.window],
            )

            if path:
                self.res_model_path_entry.setText(path)
                self.res_model_path = os.path.join(
                    os.path.dirname(self.model_path),
                    f'{self.verified_clips[0].name.get_value()}.pth'
                )

        def open_src_model_browser():
            if self.scratch_button.isChecked():
                return

            self.window.hide()
            import flame

            if not os.path.isfile(self.model_path):
                self.model_path = os.path.join(
                    os.path.dirname(__file__), 
                    'models',
                    'flownet4.pth'
                    )

            flame.browser.show(
                title = 'Select flameTimewarpML Model:',
                extension = 'pth',
                default_path = os.path.dirname(self.model_path),
                multi_selection = False)
            if len(flame.browser.selection) > 0:
                self.src_model_path = flame.browser.selection[0]
                self.src_model_path_entry.setText(self.src_model_path)
                self.fw.prefs['finetune_src_model_path'] = self.src_model_path
                self.fw.save_prefs()
            self.window.show()

        def open_dest_model_browser():
            self.window.hide()

            if not os.path.isfile(self.res_model_path):
                import shutil
                shutil.copy(self.src_model_path, self.res_model_path)

            if not os.path.isfile(self.res_model_path):
                open_dest_browser()
            else:
                flame.browser.show(
                    title = 'Select flameTimewarpML Model:',
                    extension = 'pth',
                    default_path = os.path.dirname(self.res_model_path),
                    multi_selection = False)
                if len(flame.browser.selection) > 0:
                    self.res_model_path = flame.browser.selection[0]
                    self.res_model_path_entry.setText(self.res_model_path)
                    self.fw.prefs['finetune_res_model_path'] = self.res_model_path
                    self.fw.save_prefs()
            self.window.show()

        def scratch():
            self.fw.prefs['finetune_scratch'] = self.scratch_button.isChecked()
            self.src_model_path_entry.read_only = self.scratch_button.isChecked()
            self.fw.save_prefs()

        def generalize():
            self.fw.prefs['finetune_generalize'] = self.gen_button.isChecked()
            self.fw.save_prefs()

        def large_patch():
            self.fw.prefs['finetune_1k_patch'] = self.large_patch_button.isChecked()
            self.fw.save_prefs()

        # Create export and apply window
        window_title = f'{settings["app_name"]} <small>{settings["version"]}'
        window_title += ' [Finetune]'

        self.window = PyFlameDialogWindow(
            width=940,
            height=256,
            title=window_title
        )

        # Labels
        self.options_label = PyFlameLabel(
            text='Options',
        )

        self.export_path_label = PyFlameLabel(
            text='Shot(s) Path',
        )

        self.src_model_path_label = PyFlameLabel(
            text='Initial Model Weights',
        )

        self.res_model_path_label = PyFlameLabel(
            text='Result Model Weights',
        )

        # Entries
        self.export_path_entry = PyFlameLineEdit(
            text=self.finetune_export_path,
            max_width=True,
        )

        self.src_model_path_entry = PyFlameLineEdit(
            text=self.src_model_path,
            max_width=True,
        )

        self.res_model_path_entry = PyFlameLineEdit(
            text=self.res_model_path,
            max_width=True,
        )

        # Buttons
        self.path_browse_button = PyFlameButton(
            text='Browse',
            connect=open_browser,
        )

        self.src_model_browse_button = PyFlameButton(
            text='Browse',
            connect=open_src_model_browser,
        )

        self.res_model_browse_button = PyFlameButton(
            text='Browse',
            connect=open_dest_model_browser,
        )

        self.scratch_button = PyFlamePushButton(
            text='From scratch',
            button_checked = self.fw.prefs.get('finetune_scratch', False),
            connect=scratch
        )

        self.gen_button = PyFlamePushButton(
            text='Generalize',
            button_checked = self.fw.prefs.get('finetune_generalize', False),
            connect=generalize
        )

        self.large_patch_button = PyFlamePushButton(
            text='1K Patch',
            button_checked = self.fw.prefs.get('finetune_1k_patch', False),
            connect=large_patch
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
        grid_layout.setRowMinimumHeight(0, 30)
        grid_layout.setRowMinimumHeight(1, 30)
        grid_layout.setRowMinimumHeight(2, 30)
        grid_layout.setRowMinimumHeight(3, 30)
        grid_layout.setRowMinimumHeight(4, 30)
        grid_layout.setRowMinimumHeight(5, 30)
        
        grid_layout.setColumnMinimumWidth(0, 150)
        grid_layout.setColumnMinimumWidth(1, 150)
        grid_layout.setColumnMinimumWidth(2, 150)
        grid_layout.setColumnMinimumWidth(3, 150)
        grid_layout.setColumnMinimumWidth(4, 150)

        grid_layout.addWidget(self.options_label, 0, 0)

        grid_layout.addWidget(self.scratch_button, 0, 3)
        grid_layout.addWidget(self.large_patch_button, 0, 4)
        grid_layout.addWidget(self.gen_button, 0, 5)

        grid_layout.addWidget(self.export_path_label, 1, 0)
        grid_layout.addWidget(self.export_path_entry, 1, 1, 1, 4)
        grid_layout.addWidget(self.path_browse_button, 1, 5)
 
        grid_layout.addWidget(self.src_model_path_label, 2, 0)
        grid_layout.addWidget(self.src_model_path_entry, 2, 1, 1, 4)
        grid_layout.addWidget(self.src_model_browse_button, 2, 5)

        grid_layout.addWidget(self.res_model_path_label, 3, 0)
        grid_layout.addWidget(self.res_model_path_entry, 3, 1, 1, 4)
        grid_layout.addWidget(self.res_model_browse_button, 3, 5)

        grid_layout.addWidget(self.cancel_button, 4, 4)
        grid_layout.addWidget(self.export_and_apply_button, 4, 5)

        # Add layout to window
        self.window.add_layout(grid_layout)

        self.window.show()   

    def apply(self):
        self.window.hide()

        if self.mode == 'finetune':
            self.apply_finetune()
            self.window.close()
            return

        working_folder = self.fw.prefs.get('working_folder')
        if not os.path.isdir(working_folder):
            import flame
            dialog = flame.messages.show_in_dialog(
                title = f'{self.settings["app_name"]}',
                message = f'Unable to find working folder: {working_folder}',
                type = 'error',
                buttons = ['Ok']
            )
            self.window.show()
            return

        model_file = self.fw.prefs.get('model_path')
        if not os.path.isfile(model_file):
            import flame
            dialog = flame.messages.show_in_dialog(
                title = f'{self.settings["app_name"]}',
                message = f'Unable to find model state file: {model_file}',
                type = 'error',
                buttons = ['Ok']
            )
            self.window.show()
            return

        if self.mode == 'timewarp':
            self.apply_timewarp()
        elif self.mode == 'fluidmorph':
            self.apply_fluidmorph()

        # Close export and apply window
        self.window.close()

    def apply_timewarp(self):
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
                export_preset = self.create_export_preset(
                        os.path.join(
                            os.path.dirname(__file__), 
                            'presets', 
                            'source_export32.xml'
                        )
                    )
            else:
                export_preset = self.create_export_preset(
                        os.path.join(
                            os.path.dirname(__file__), 
                            'presets', 
                            'source_export.xml'
                        )
                    )
            
            self.export_clip(clip, source_clip_folder, export_preset=export_preset)

            record_in = clip.versions[0].tracks[0].segments[0].record_in.relative_frame
            record_out = clip.versions[0].tracks[0].segments[0].record_out.relative_frame

            json_info = {}
            json_info['mode'] = 'timewarp'
            json_info['input'] = source_clip_folder
            json_info['output'] = result_folder
            json_info['clip_name'] = tw_clip_name
            json_info['model_path'] = self.fw.prefs.get('model_path')
            json_info['setup'] = tw_setup_string
            json_info['record_in'] = record_in
            json_info['record_out'] = record_out
            json_info['settings'] = self.settings
            json_info['cpu'] = self.fw.prefs.get('cpu')
            json_info['half'] = self.fw.prefs.get('half')

            lockfile_path = os.path.join(
                result_folder,
                f'{tw_clip_name}.json'
            )

            try:
                import json
                with open(lockfile_path, 'w') as json_file:
                    json.dump(json_info, json_file, indent=4)
            except Exception as e:
                dialog = flame.messages.show_in_dialog(
                    title = f'{settings["app_name"]}',
                    message = f'Unable to save {lockfile_path}: {e}',
                    type = 'error',
                    buttons = ['Ok'])
                return False

            # '''
            new_clip_name = clip_name + '_TWML'
            watcher = threading.Thread(
                target=self.import_watcher, 
                args=(
                    result_folder, 
                    new_clip_name, 
                    clip, 
                    [source_clip_folder],
                    lockfile_path
                    )
                )
            watcher.daemon = True
            watcher.start()
            self.loops.append(watcher)
            # '''

            self.run_inference(lockfile_path)

    def apply_fluidmorph(self):
            incoming_clip = self.verified_clips[0]
            outgoing_clip = self.verified_clips[1]

            clip_name = incoming_clip.name.get_value()
            tw_clip_name = self.fw.sanitized(clip_name) + '_TWML' + '_' + self.fw.create_timestamp_uid()        
    
            result_folder = os.path.abspath(
            os.path.join(
                self.working_folder, 
                tw_clip_name
                )
            )

            incoming_folder = os.path.join(result_folder, 'incoming')
            outgoing_folder = os.path.join(result_folder, 'outgoing')

            if incoming_clip.bit_depth == 32:
                export_preset = self.create_export_preset(
                        os.path.join(
                            os.path.dirname(__file__), 
                            'presets', 
                            'openexr32bit.xml'
                        )
                    )
                self.export_clip(incoming_clip, incoming_folder, export_preset=export_preset)
                self.export_clip(outgoing_clip, outgoing_folder, export_preset=export_preset)
            else:
                export_preset = self.create_export_preset(
                        os.path.join(
                            os.path.dirname(__file__), 
                            'presets', 
                            'openexr16bit.xml'
                        )
                    )
                self.export_clip(incoming_clip, incoming_folder, export_preset=export_preset)
                self.export_clip(outgoing_clip, outgoing_folder, export_preset=export_preset)

            json_info = {}
            json_info['mode'] = 'fluidmorph'
            json_info['incoming'] = incoming_folder
            json_info['outgoing'] = outgoing_folder
            json_info['output'] = result_folder
            json_info['clip_name'] = tw_clip_name
            json_info['model_path'] = self.fw.prefs.get('model_path')
            json_info['settings'] = self.settings
            json_info['cpu'] = self.fw.prefs.get('cpu')
            json_info['half'] = self.fw.prefs.get('half')

            lockfile_path = os.path.join(
                result_folder,
                f'{tw_clip_name}.json'
            )

            try:
                import json
                with open(lockfile_path, 'w') as json_file:
                    json.dump(json_info, json_file, indent=4)
            except Exception as e:
                dialog = flame.messages.show_in_dialog(
                    title = f'{settings["app_name"]}',
                    message = f'Unable to save {lockfile_path}: {e}',
                    type = 'error',
                    buttons = ['Ok'])
                return False

            # '''
            new_clip_name = clip_name + '_TWML'
            watcher = threading.Thread(
                target=self.import_watcher, 
                args=(
                    result_folder, 
                    new_clip_name, 
                    incoming_clip, 
                    [incoming_folder, outgoing_folder],
                    lockfile_path
                    )
                )
            watcher.daemon = True
            watcher.start()
            self.loops.append(watcher)
            # '''

            self.run_inference(lockfile_path)

    def apply_finetune(self):
        self.src_model_path = self.src_model_path_entry.text()

        if not os.path.isfile(self.src_model_path):
            import flame
            dialog = flame.messages.show_in_dialog(
                title = f'{self.settings["app_name"]}',
                message = f'Unable to find initial model state file: {self.src_model_path}',
                type = 'error',
                buttons = ['Ok']
            )
            self.window.show()
            return

        self.res_model_path = self.res_model_path_entry.text()
        
        if not os.path.isfile(self.res_model_path):
            import shutil
            shutil.copy(self.src_model_path, self.res_model_path)

        dataset_root_path = self.export_path_entry.text()
        export_root_path = dataset_root_path

        if self.fw.prefs.get('finetune_fast'):
            export_root_path = os.path.join(
                export_root_path,
                'fast'
            )
        else:
            export_root_path = os.path.join(
                export_root_path,
                'normal'
            )

        clip_names = set()

        for clip_number, clip in enumerate(self.verified_clips):
            if clip.name.get_value() in clip_names:
                clip_name = f'{clip.name.get_value()}_{clip_number}'
            else:
                clip_name = f'{clip.name.get_value()}'
            
            clip_names.add(clip_name)

            if clip.bit_depth == 32:
                export_preset = self.create_export_preset(
                        os.path.join(
                            os.path.dirname(__file__), 
                            'presets', 
                            'source_export32bit.xml'
                        )
                    )
            else:
                export_preset = self.create_export_preset(
                        os.path.join(
                            os.path.dirname(__file__), 
                            'presets', 
                            'source_export.xml'
                        )
                    )
                
            print (f'exporting {clip} as {clip_name}')
            
            self.export_clip(
                clip, 
                os.path.join(
                    export_root_path,
                    clip_name
                ), 
                export_preset=export_preset)

        json_info = {}
        json_info['mode'] = 'finetune'

        json_info['dataset_path'] = dataset_root_path # 'Path to the dataset'
        json_info['lr'] = 1e-6 if self.fw.prefs.get('finetune_generalize') else 4e-6 # 'Learning rate (default: 1e-6)'
        json_info['pulse'] = 9999 # 'Period in steps to pulse learning rate (float) (default: 10K)'
        json_info['pulse_amplitude'] = 25 # 'Learning rate pulse amplitude (percentage) (default: 25)'
        json_info['onecycle'] = -1 # 'Train one cycle for N epochs (default: None)'
        json_info['state_file'] = self.res_model_path # 'Path to the pre-trained model state dict file (optional)'
        json_info['model'] = None # 'Model name (optional)'
        json_info['legacy_model'] = None # 'Model name (optional)'
        json_info['device'] = 0 # 'Graphics card index (default: 0)'
        json_info['batch_size'] = 4 if self.fw.prefs.get('finetune_generalize') else 2 # 'Batch size (int) (default: 2)'
        json_info['first_epoch'] = -1 # 'Epoch (int) (default: Saved)'
        json_info['epochs'] = -1 # 'Number of epoch to run (int) (default: Unlimited)'
        json_info['reset_stats'] = True # 'Reset saved step, epoch and loss stats'
        json_info['eval'] = 9999 if self.fw.prefs.get('finetune_eval') else -1 # 'Evaluate after N steps'
        json_info['eval_first'] = True
        json_info['eval_samples'] = -1 # 'Evaluate N random training samples'
        json_info['eval_seed'] = -1 # 'Random seed to select samples if --eval_samples set'
        json_info['eval_buffer'] = 8 # 'Write buffer size for evaluated images'
        json_info['eval_keep_all'] = False # 'Keep eval results for each eval step'
        json_info['frame_size'] = 1024 if self.fw.prefs.get('finetune_1k_patch') else 448 # 'Frame size in pixels (default: 448)'
        json_info['all_gpus'] = False
        json_info['freeze'] = False
        json_info['acescc'] = 0 # 'Percentage of ACEScc encoded frames (default: 40))'
        json_info['generalize'] = 85 if self.fw.prefs.get('finetune_generalize') else 1 # 'Generalization level (0 - 100) (default: 85)'
        json_info['weight_decay'] = -1 # 'AdamW weight decay (default: calculated from --generalize value)'
        json_info['preview'] = 100 # 'Save preview each N steps (default: 1000)'
        json_info['save'] = 1000 # 'Save model state dict each N steps (default: 1000)'
        json_info['repeat'] = 1 # 'Repeat each triade N times with augmentation (default: 1)'
        json_info['iterations'] = 1 # 'Run each flow refinement N times (default: 1)'
        json_info['compile'] = False # 'Compile with torch.compile'

        json_file_path = os.path.join(
            export_root_path,
            f'{self.verified_clips[0].name.get_value()}.json'
        )

        try:
            import json
            with open(json_file_path, 'w') as json_file:
                json.dump(json_info, json_file, indent=4)
        except Exception as e:
            dialog = flame.messages.show_in_dialog(
                title = f'{settings["app_name"]}',
                message = f'Unable to save {json_file_path}: {e}',
                type = 'error',
                buttons = ['Ok'])
            return False
        
        self.run_finetune(json_file_path)

    def run_finetune(self, json_file_path):
        import platform

        conda_env_path = os.path.join(
            os.path.dirname(__file__),
            'packages',
            '.miniconda',
            'appenv'
        )

        conda_python_path = os.path.join(
            conda_env_path,
            'bin',
            'python'
            )

        inference_script_path = os.path.join(
            os.path.dirname(__file__),
            'pytorch',
            'flameTimewarpML_finetune.py'
        )

        env = os.environ.copy()
        env['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
            conda_env_path,
            'plugins',
            'platforms'
        )

        # print (f'command: {conda_python_path} {inference_script_path} {lockfile_path}')

        import subprocess
        subprocess.Popen([conda_python_path, inference_script_path, json_file_path], env=env)

    def run_inference(self, lockfile_path):
        import platform

        conda_env_path = os.path.join(
            os.path.dirname(__file__),
            'packages',
            '.miniconda',
            'appenv'
        )

        conda_python_path = os.path.join(
            conda_env_path,
            'bin',
            'python'
            )

        inference_script_path = os.path.join(
            os.path.dirname(__file__),
            'pytorch',
            'flameTimewarpML_inference.py'
        )

        env = os.environ.copy()
        env['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
            conda_env_path,
            'plugins',
            'platforms'
        )

        # print (f'command: {conda_python_path} {inference_script_path} {lockfile_path}')

        import subprocess
        subprocess.Popen([conda_python_path, inference_script_path, lockfile_path], env=env)

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

    def import_watcher(self, import_path, new_clip_name, clip, folders_to_cleanup, lockfile):
        flame_friendly_path = None
        destination = clip.parent
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

        while self.threads:
            if not os.path.isfile(lockfile):

                # clean-up source files used
                print(f'Cleaning up temporary files used: {folders_to_cleanup}')
                for folder in folders_to_cleanup:
                    cmd = 'rm -f "' + os.path.abspath(folder) + '/"*'
                    print('Executing command: %s' % cmd)
                    os.system(cmd)
                    try:
                        os.rmdir(folder)
                    except Exception as e:
                        print('Error removing %s: %s' % (folder, e))
            
                print('Importing result from: %s' % import_path)
                flame_friendly_path = import_path
                flame.schedule_idle_event(import_flame_clip)
                
                '''
                file_names = [f for f in os.listdir(import_path) if f.endswith('.exr')]
                print (file_names)

                if file_names:
                    file_names.sort()
                    first_name, ext = os.path.splitext(file_names[0])
                    last_name, ext = os.path.splitext(file_names[-1])
                    first_frame = first_name.split('.')[-1]
                    last_frame = last_name.split('.')[-1]
                    flame_friendly_path = os.path.join(import_path, '[' + first_frame + '-' + last_frame + ']' + '.exr')

                    print (flame_friendly_path)

                    import flame
                    flame.schedule_idle_event(import_flame_clip)
                '''

                '''
                if os.getenv('FLAMETWML_HARDCOMMIT') == 'True':
                    time.sleep(1)
                    cmd = 'rm -f "' + os.path.abspath(import_path) + '/"*'
                    print('Executing command: %s' % cmd)
                    os.system(cmd)
                    try:
                        os.rmdir(import_path)
                    except Exception as e:
                        print('Error removing %s: %s' % (import_path, e))
                '''
                break
            time.sleep(0.1)

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

    def create_export_preset(self, export_preset_path):
        import flame

        def find_files_with_all_path_patterns(directory, patterns):
            import os
            import fnmatch

            matches = []
            for root, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    if all(fnmatch.fnmatch(full_path, pattern) for pattern in patterns):
                        matches.append(full_path)
            return matches

        def find_version_in_file(file_path):
            import os
            import re
            version_pattern = re.compile(r'<preset version="(\d+)">')
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        match = version_pattern.search(line)
                        if match:
                            return match.group(1)
                return None
            except (IOError, OSError) as e:
                print(f"Error reading file {file_path}: {e}")
                return None

        def update_version_in_file(src_path, dest_path, new_version):
            import os
            import xml.etree.ElementTree as ET

            try:
                # Parse the source XML file
                tree = ET.parse(src_path)
                root = tree.getroot()

                if 'version' in root.attrib:
                    root.set('version', str(new_version))

                # Write the updated XML to the destination file
                tree.write(dest_path, encoding='utf-8', xml_declaration=True)

                # print(f"Updated version in file saved to {dest_path}")
                return dest_path
            except ET.ParseError as e:
                print(f"Error parsing XML file {src_path}: {e}")
                return None
            except (IOError, OSError) as e:
                print(f"Error processing file: {e}")
                return None

        try:
            flame_presets_location = flame.PyExporter.get_presets_base_dir(
                        flame.PyExporter.PresetVisibility.Autodesk
                    )
            
            matching_files = find_files_with_all_path_patterns(flame_presets_location, ['*OpenEXR*.xml', '*file*', '*sequence*'])
            new_version = find_version_in_file(matching_files[0])

            # print (f'new version: {new_version}')

            dest_preset_path = os.path.join(
                    '/var/tmp',
                    os.path.basename(export_preset_path)
                )
            
            if os.path.isfile(dest_preset_path):
                # print (f'removing {dest_preset_path}')
                os.remove(dest_preset_path)

            preset_path = update_version_in_file(
                export_preset_path,
                dest_preset_path,
                new_version
            )

            if preset_path:
                return preset_path
            else:
                return export_preset_path

        except:
            return export_preset_path

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

    def finetune(selection):
        ApplyModelDialog(selection, mode='finetune')

    def deduplicate(selection):
        dialog = flame.messages.show_in_dialog(
            title = f'{settings["app_name"]}',
            message = f'Not yet implemented in {settings["version"]}',
            type = 'error',
            buttons = ['Ok'])

    menu = [
        {
            'name': settings['menu_group_name'],
            'actions': [
                {
                    'name': 'Fill / Remove Duplicate Frames',
                    'order': 1,
                    'execute': deduplicate,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': 'Create Fluidmorph Transition',
                    'order': 2,
                    'execute': fluidmorph,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': "Timewarp from Flame's TW effect",
                    'order': 3,
                    'execute': timewarp,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': "Fine-tune model on selected clips",
                    'order': 4,
                    'execute': finetune,
                    'isVisible': scope_clip,
                    'waitCursor': False,
                },
                {
                    'name': f'Version: {settings["version"]}',
                    'order': 5,
                    'execute': about_dialog,
                    'isVisible': scope_clip,
                    'isEnabled': False,
                },
            ],
        }
    ]

    return menu