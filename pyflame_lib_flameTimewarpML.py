"""
Script Name: PyFlame Custom Library
Version: 2.1.6
Written by: Michael Vaglienty
With contributions from: Kieran Hanrahan
Creation Date: 10.31.20
Update Date: 01.31.24

This file contains a library various custom UI widgets that can be used to build QT windows similar to the look of Flame along with some other useful functions.

This file should be placed in same folder as main script.

To avoid conflicts with having multiple copies within /opt/Autodesk/shared/python, file should be renamed to: pyflame_lib_<script name>.py

Usage:

    To import and use the file, add the following line to the top of your script:

        from pyflame_lib_<script name> import *

    To use PyFlameFunctions use pyflame.FUCNTION_NAME:
        pyflame.message_print
        pyflame.get_flame_version
        ...

    To use widgets use widget_name:
        PyFlamePushButtonMenu
        PyFlamePushButton
        ...

Updates:

    v2.1.6 01.31.24

        Fixed PySide6 errors/font in slider calculator.

        Added new class: PyFlameLineEditFileBrowser - Line Edit widget that opens a flame file browser when clicked.

        PyFlameLineEdit: Added read_only argument. This will make the line edit read only and unselectable with dark background. Default is False.

        PyFlameSlider: Added rate argument. This controls the sensitivity of the slider. The value should be between 1 and 10.
        1 is the most sensitive and 10 is the least sensitive. Default is 10.

    v2.1.5 01.25.24

        Updated PyFlameTokenPushButton:
            Added default argument values:
                text='Add Token'
                token_dict={}
                token_dest=None
            Added new method:
                add_menu_options(new_options): Add new menu options to the existing token menu and clear old options.

    v2.1.4 01.21.24

        Updated PySide.
        Improved UI scaling for different screen resolutions.
        Fixed issue with PyFlameConfig not properly returning boolean values.

    v2.1.3 11.21.23

        Updated pyflame.get_export_preset_version() to check default jpeg export preset for preset version.
        This function no longer needs to be updated manually.

    v2.1.2 11.17.23

        Updated Token Push Button Menu widget. Added ability to clean destination(line edit widget) before adding the token.

    v2.1.1 11.06.23

        Added pyflame functions for dealing with export preset versions:
            pyflame.get_export_preset_version()
            pyflame.update_export_preset()

    v2.1.0 08.14.23

        All widgets have been updated to be more consistent.

        All widgets have been changed from Flame to PyFlame.
        For example: FlamePushButtonMenu -> PyFlamePushButtonMenu
        Widgets should be left in this file and not moved individually
        to other files. This will cause problems since some widgets rely
        on other widgets/functions in this file.

        Widget documentation has been improved.
"""

import ast
import datetime
import os
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from enum import Enum
from functools import partial
from subprocess import PIPE, Popen
from typing import Callable, Dict, List, Optional, Tuple, Union

import flame

# Try to import PySide6, otherwise import PySide2
# ------------------------------------------------------ #

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    from PySide2 import QtCore, QtGui, QtWidgets

if QtCore.__version_info__[0] < 6:
    main_window_res = QtWidgets.QDesktopWidget()
else:
    main_window_res = QtGui.QGuiApplication.primaryScreen()

# ------------------------------------------------------ #

PYFLAME_FONT = 'Discreet' # Font used in all PyFlame UI elements
PYFLAME_FONT_SIZE = 13 # Default font size used in all PyFlame UI elements

# -------------------------------- PyFlame Enums -------------------------------- #
# These are used by different PyFlame widgets and functions
# Not intended to be used outside of this file

class Color(Enum):
    """
    Enum for button color options.

    Parameters:
        GRAY (int): For gray button.
        BLUE (int): For blue button.
        RED (int): For red button.
    """

    GRAY = 1
    BLUE = 2
    RED = 3

class Style(Enum):
    """
    Enum for PyFlameLabel style options.

    Parameters:
        NORMAL (int): Standard label without any additional styling. Text is left aligned by default.
        UNDERLINE (int): Text is underlined. Text is centered by default.
        BACKGROUND (int): Adds a darker background to the label. Text is left aligned by default.
        BORDER (int): Adds a white border around the label with a dark background. Text is centered by default.

    Default text alignment can be overridden by setting the PyFlameLabel align argument.
    """

    NORMAL = 1
    UNDERLINE = 2
    BACKGROUND = 3
    BORDER = 4

class Align(Enum):
    """
    Enum for PyFlameLabel and PyFlamePushButtonMenu text alignment.

    Parameters:
        LEFT (int): Align text to the left.
        RIGHT (int): Align text to the right.
        CENTER (int): Align text to the center.
    """

    LEFT = 1
    RIGHT = 2
    CENTER = 3

class MessageType(Enum):
    """
    Enum for PyFlameMessageWindow and pyflame.message_print message types.

    Parameters:
        INFO (int): Information message type.
        OPERATION_COMPLETE (int): Operation complete message type.
        CONFIRM (int): Confirmation message type.
        ERROR (int): Error message type.
        WARNING (int): Warning message type.
    """
    INFO = 1
    OPERATION_COMPLETE = 2
    CONFIRM = 3
    ERROR = 4
    WARNING = 5

class LineColor(Enum):
    """
    Enum for PyFlameWindow side bar color options.

    Parameters:
        GRAY (int): For gray line.
        BLUE (int): For blue line.
        RED (int): For red line.
        GREEN (int): For green line.
        YELLOW (int): For yellow line.
        TEAL (int): For teal line.
    """

    GRAY = 1
    BLUE = 2
    RED = 3
    GREEN = 4
    YELLOW = 5
    TEAL = 6

class BrowserType(Enum):
    """
    Enum for PyFlameLineEditFileBrowser browser type options.

    Parameters:
        FILE (int): File browser.
        DIRECTORY (int): Directory browser.
    """

    FILE = 1
    DIRECTORY = 2

# -------------------------------- PyFlame Functions-------------------------------- #
# Various useful functions
# Can be accessed by pyflame.method_name
# Example:
#     pyflame.message_print(
#         message='Config not found.',
#         script_name=SCRIPT_NAME,
#         type=MessageType.ERROR
#         )

class PyFlameFunctions():
    """
    PyFlame class containing various useful functions.
    """

    @staticmethod
    def message_print(message: str, script_name: str='PYTHON HOOKS', type=MessageType.INFO, time: int=3) -> None:
        """
        Prints messages to terminal and Flame message area(2023.1+).

        Parameters:
            message (str): Message to print.
            script_name (str): Name of script. This is displayed in the Flame message area.
                Default: PYTHON HOOKS
            type (MessageType): Type of message. (message, error, warning).
                -MessageType.INFO: Standard message.
                -MessageType.ERROR: Error message. Text in terminal will be yellow.
                -MessageType.WARNING: Warning message. Text in terminal will be red.
                Default: MessageType.INFO
            time (int): Amount of time to display message for in seconds. Default is 3 seconds.

        Returns:
            None: The method does not return a value.

        Example:
            pyflame.message_print(message='Config not found.', script_name=SCRIPT_NAME, type=MessageType.ERROR)
        """

        # Validate arguments
        if not isinstance(message, str):
            raise TypeError('Pyflame Message Print: message must be a string.')
        elif not isinstance(script_name, str):
            raise TypeError('Pyflame Message Print: script_name must be a string.')
        valid_message_types = {MessageType.INFO, MessageType.ERROR, MessageType.WARNING}
        if type not in valid_message_types:
            raise ValueError(f'PyFlame Message Print: Invalid message type: {type}. Accepted types: INFO, ERROR, WARNING.')
        elif not isinstance(time, int):
            raise TypeError('Pyflame Message Print: time must be an integer.')

        # Print to terminal/shell
        if type == MessageType.INFO:
            print(f'--> {message}\n') # Print message text normally
        elif type == MessageType.ERROR:
            print(f'\033[93m--> {message}\033[0m\n') # Print message text in yellow
        elif type == MessageType.WARNING:
            print(f'\033[91m--> {message}\033[0m\n') # Print message text in red

        script_name = script_name.upper()

        # Print to Flame Message Window - Flame 2023.1 and later
        # Warning and error intentionally swapped to match color of message window
        try:
            if type == MessageType.INFO:
                flame.messages.show_in_console(f'{script_name}: {message}', 'info', time)
            elif type == MessageType.ERROR:
                flame.messages.show_in_console(f'{script_name}: {message}', 'warning', time)
            elif type == MessageType.WARNING:
                flame.messages.show_in_console(f'{script_name}: {message}', 'error', time)
        except:
            pass

    @staticmethod
    def get_flame_version() -> float:
            """
            Gets version of flame and returns float value.

            Example:
                flame_version = pyflame.get_flame_version()

            Returns:
                flame_version (float): 2022.0
                    2022 -> 2022.0
                    2022.1.1 -> 2022.1
                    2022.1.pr145 -> 2022.1
            """

            flame_version = flame.get_version()

            if 'pr' in flame_version:
                flame_version = flame_version.rsplit('.pr', 1)[0]
            if len(flame_version) > 6:
                flame_version = flame_version[:6]
            flame_version = float(flame_version)

            print('Flame Version:', flame_version, '\n')

            return flame_version

    @staticmethod
    def get_flame_python_packages_path() -> str:
        """
        Get path to Flame's python packages folder.

        Returns:
            python_packages_path (str): Path to Flame's python packages folder.

        Raises:
            FileNotFoundError: If no python3.* folder is found in the python lib path.

        Example:
            python_packages_path = pyflame.pyflame_get_flame_python_packages_path()
        """

        flame_version = flame.get_version()

        python_lib_path = f'/opt/Autodesk/python/{flame_version}/lib'

        # Find the folder in the python lib path that starts with 'python3.'
        for folder in os.listdir(python_lib_path):
            if folder.startswith('python3.'):
                python_package_folder = os.path.join(python_lib_path, folder, 'site-packages')
                print('Flame Python Packages Folder:', python_package_folder, '\n')
                return python_package_folder

        raise FileNotFoundError('No python3.* folder found in the python lib path.')

    @staticmethod
    def file_browser(path: str='/opt/Autodesk', title: Optional[str]=None, extension: Optional[List[str]]=None, select_directory: bool=False,
                     multi_selection: bool=False, include_resolution: bool=False, use_flame_browser: bool=True,
                     window_to_hide=None) -> Optional[Union[str, list]]:
        """
        Opens QT file browser window(Flame 2022 - Flame 2023). Flame's file browser is used 2023.1 and later.

        Parameters:
            title (str): File browser window title.
            extension (list) File extension filter. None to list directories.
                Default: None
            path (str): Open file browser to this path.
                Default: /opt/Autodesk
            select_directory (bool): Ability to select directories.
                Default: False
            multi_selection (bool): Ability to select multiple files/folders.
                Default: False
            include_resolution (bool): Enable resolution controls in flame browser.
                Default: False
            use_flame_browser (bool): Use Flame's file browser if using Flame 2023.1 or later.
                Default: True
            window_to_hide (list[QWidget.QWindow]): Hide Qt window while file browser window is open. window is restored when browser is closed.
                Default: None

        Example:
            path = pyflame_file_browser('Load Undistort ST Map(EXR)', ['exr'], self.undistort_map_path)

        Returns:
            path (str, list): Path to selected file or directory.
                When Multi Selection is enabled, the file browser will return a list.
                Otherwise it will return a string.
        """

        # Check argument values
        if not isinstance(path, str):
            raise TypeError('Pyflame File Browser: path must be a string.')
        elif title is not None and not isinstance(title, str):
            raise TypeError('Pyflame File Browser: title must be None or a string.')
        elif extension is not None and not isinstance(extension, list):
            raise TypeError('Pyflame File Browser: extension must None or a list.')
        elif not isinstance(select_directory, bool):
            raise TypeError('Pyflame File Browser: select_directory must be a boolean.')
        elif not isinstance(multi_selection, bool):
            raise TypeError('Pyflame File Browser: multi_selection must be a boolean.')
        elif not isinstance(include_resolution, bool):
            raise TypeError('Pyflame File Browser: include_resolution must be a boolean.')
        if not isinstance(use_flame_browser, bool):
            raise TypeError('Pyflame File Browser: use_flame_browser must be a boolean.')
        if window_to_hide is not None and not isinstance(window_to_hide, list):
            raise TypeError('Pyflame File Browser: window_to_hide must be None or a list.')

        if not title and not extension:
            title = 'Select Directory'
        elif not title and extension:
            title = 'Select File'

        # Clean up path
        while os.path.isdir(path) is not True:
            path = path.rsplit('/', 1)[0]
            if '/' not in path and not os.path.isdir(path):
                path = '/opt/Autodesk'
            print('Browser path:', path, '\n')

        # Open file browser
        if pyflame.get_flame_version() >= 2023.1 and use_flame_browser:

            # Hide Qt window while browser is open
            if window_to_hide:
                for window in window_to_hide:
                    window.hide()

            # Open Flame file browser
            flame.browser.show(
                title=title,
                extension=extension,
                default_path=path,
                select_directory=select_directory,
                multi_selection=multi_selection,
                include_resolution=include_resolution
                )

            # Restore Qt windows
            if window_to_hide:
                for window in window_to_hide:
                    window.show()

            # Return file path(s) from Flame file browser
            if flame.browser.selection:
                if multi_selection:
                    return flame.browser.selection
                return flame.browser.selection[0]
        else:
            browser = QtWidgets.QFileDialog()
            browser.setDirectory(path)

            if select_directory or not extension:
                browser.setFileMode(QtWidgets.QFileDialog.Directory)
            else:
                browser.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                filter_str = ';;'.join(f'*.{ext}' for ext in extension)
                browser.setNameFilter(filter_str)

                browser.selectNameFilter(filter_str)

            if browser.exec_():
                return str(browser.selectedFiles()[0])

            print('\n--> Import cancelled \n')
            return

    @staticmethod
    def open_in_finder(path: str) -> None:
        """
        Open path in System Finder

        Parameters:
            path (str): Path to open in Finder
        """

        # Validate argument types
        if not isinstance(path, str):
            raise TypeError('Pyflame Open In Finder: Path must be a string')

        if not os.path.exists(path):
            pyflame.message_print(message=f'Path does not exist: {path}', type=MessageType.ERROR)
            return

        # Open path in Finder
        if platform.system() == 'Darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])

    @staticmethod
    def refresh_hooks(script_name: str='PYTHON HOOKS') -> None:
        """
        Refresh python hooks and print message to terminal and Flame message window

        Parameters:
            script_name: (str) Name of script.
                This is displayed in the Flame message area.
                Default: PYTHON HOOKS

        Returns:
            None: The method does not return a value.

        Example:
            import pyflame
            pyflame.refresh_hooks(script_name='SCRIPT NAME')
        """

        if not isinstance(script_name, str):
            raise TypeError('Pyflame Refresh Hooks: script_name must be a string')

        # Refresh python hooks
        flame.execute_shortcut('Rescan Python Hooks')

        # Print message to terminal and Flame message area
        pyflame.message_print(
            message='Python hooks refreshed.',
            script_name=script_name,
            )

    @staticmethod
    def resolve_path_tokens(tokenized_path: str, flame_pyobject=None, date=None) -> str:
        """
        Resolves paths with tokens.

        Parameters:
            tokenized_path (str): Path with tokens to be translated.
            flame_pyobject (flame.PyClip, optional) Flame PyClip/PySegment/PyBatch Object.
            date: (datetime, optional) Date/time to use for token translation. Default is None. If None is passed datetime value will be gotten each time function is run.

        Supported tokens are:
            <ProjectName>, <ProjectNickName>, <UserName>, <UserNickName>, <YYYY>, <YY>, <MM>, <DD>, <Hour>, <Minute>, <AMPM>, <ampm>

        Tokens aviailble for Flame PyObjects:
            PyClip and PySegment:
                <ShotName>, <SeqName>, <SEQNAME>, <ClipName>, <Resolution>, <ClipHeight>, <ClipWidth>, <TapeName>
            PyBatch:
                <ShotName>, <SeqName>, <SEQNAME>

        Returns:
            resolved_path (str): Resolved path.

        Example:
            export_path = pyflame.translate_path_tokens(
                tokenized_path=custom_export_path,
                flame_pyobject=clip,
                date=date
                )
        """

        if not isinstance(tokenized_path, str):
            raise TypeError('Pyflame Translate Path Tokens: tokenized_path must be a string')

        def get_seq_name(name):

            # Get sequence name abreviation from shot name

            seq_name = re.split('[^a-zA-Z]', name)[0]

            return seq_name

        print('Tokenized path to resolve:', tokenized_path)

        # Get time values for token conversion

        if not date:
            date = datetime.datetime.now()

        yyyy = date.strftime('%Y')
        yy = date.strftime('%y')
        mm = date.strftime('%m')
        dd = date.strftime('%d')
        hour = date.strftime('%I')
        if hour.startswith('0'):
            hour = hour[1:]
        minute = date.strftime('%M')
        ampm_caps = date.strftime('%p')
        ampm = str(date.strftime('%p')).lower()

        # Replace tokens in path

        resolved_path = re.sub('<ProjectName>', flame.project.current_project.name, tokenized_path)
        resolved_path = re.sub('<ProjectNickName>', flame.project.current_project.nickname, resolved_path)
        resolved_path = re.sub('<UserName>', flame.users.current_user.name, resolved_path)
        resolved_path = re.sub('<UserNickName>', flame.users.current_user.nickname, resolved_path)
        resolved_path = re.sub('<YYYY>', yyyy, resolved_path)
        resolved_path = re.sub('<YY>', yy, resolved_path)
        resolved_path = re.sub('<MM>', mm, resolved_path)
        resolved_path = re.sub('<DD>', dd, resolved_path)
        resolved_path = re.sub('<Hour>', hour, resolved_path)
        resolved_path = re.sub('<Minute>', minute, resolved_path)
        resolved_path = re.sub('<AMPM>', ampm_caps, resolved_path)
        resolved_path = re.sub('<ampm>', ampm, resolved_path)

        if flame_pyobject:

            if isinstance(flame_pyobject, flame.PyClip):

                clip = flame_pyobject

                clip_name = str(clip.name)[1:-1]

                # Get shot name from clip

                try:
                    if clip.versions[0].tracks[0].segments[0].shot_name != '':
                        shot_name = str(clip.versions[0].tracks[0].segments[0].shot_name)[1:-1]
                    else:
                        shot_name = pyflame.resolve_shot_name(clip_name)
                except:
                    shot_name = ''

                # Get tape name from clip

                try:
                    tape_name = str(clip.versions[0].tracks[0].segments[0].tape_name)
                except:
                    tape_name = ''

                # Get Seq Name from shot name

                seq_name = get_seq_name(shot_name)

                # Replace clip tokens in path

                resolved_path = re.sub('<ShotName>', shot_name, resolved_path)
                resolved_path = re.sub('<SeqName>', seq_name, resolved_path)
                resolved_path = re.sub('<SEQNAME>', seq_name.upper(), resolved_path)
                resolved_path = re.sub('<ClipName>', str(clip.name)[1:-1], resolved_path)
                resolved_path = re.sub('<Resolution>', str(clip.width) + 'x' + str(clip.height), resolved_path)
                resolved_path = re.sub('<ClipHeight>', str(clip.height), resolved_path)
                resolved_path = re.sub('<ClipWidth>', str(clip.width), resolved_path)
                resolved_path = re.sub('<TapeName>', tape_name, resolved_path)

            elif isinstance(flame_pyobject, flame.PySegment):

                segment = flame_pyobject

                segment_name = str(segment.name)[1:-1]

                # Get shot name from clip

                try:
                    if segment.shot_name != '':
                        shot_name = str(segment.shot_name)[1:-1]
                    else:
                        shot_name = pyflame.resolve_shot_name(segment_name)
                except:
                    shot_name = ''

                # Get tape name from segment

                try:
                    tape_name = str(segment.tape_name)
                except:
                    tape_name = ''

                # Get Seq Name from shot name

                seq_name = get_seq_name(shot_name)

                # Replace segment tokens in path

                resolved_path = re.sub('<ShotName>', shot_name, resolved_path)
                resolved_path = re.sub('<SeqName>', seq_name, resolved_path)
                resolved_path = re.sub('<SEQNAME>', seq_name.upper(), resolved_path)
                resolved_path = re.sub('<ClipName>', segment_name, resolved_path)
                resolved_path = re.sub('<Resolution>', 'Unable to Resolve', resolved_path)
                resolved_path = re.sub('<ClipHeight>', 'Unable to Resolve', resolved_path)
                resolved_path = re.sub('<ClipWidth>', 'Unable to Resolve', resolved_path)
                resolved_path = re.sub('<TapeName>', tape_name, resolved_path)

            elif isinstance(flame_pyobject, flame.PyBatch):

                batch = flame_pyobject

                shot_name = ''

                for node in batch.nodes:
                    if node.type in ('Render', 'Write File'):
                        if node.shot_name:
                            shot_name = str(node.shot_name)[1:-1]
                            break

                if not shot_name:
                    shot_name = pyflame.resolve_shot_name(str(batch.name)[1:-1])

                # Get Seq Name from shot name

                seq_name = get_seq_name(shot_name)

                resolved_path = re.sub('<ShotName>', shot_name, resolved_path)
                resolved_path = re.sub('<SeqName>', seq_name, resolved_path)
                resolved_path = re.sub('<SEQNAME>', seq_name.upper(), resolved_path)

        print('Resolved path:', resolved_path, '\n')

        return resolved_path

    @staticmethod
    def resolve_shot_name(name: str) -> str:
        """
        Resolves a shot name from a provided string. This function is intended to handle
        two formats: a camera source name like 'A010C0012' or a standard name where the
        shot name precedes other identifiers (e.g. 'pyt_0010_comp').

        Parameters:
            name (str): The name to be resolved into a shot name.

        Returns:
            str: The resolved shot name.

        Raises:
            TypeError: If the provided 'name' is not a string.

        Examples:
            Using a camera source name:
                shot_name = pyflame.resolve_shot_name('A010C0012')
                print(shot_name)  # Outputs: A010C001

            Using a standard name:
                shot_name = pyflame.resolve_shot_name('pyt_0010_comp')
                print(shot_name)  # Outputs: pyt_0010
        """


        # Validate argument types
        if not isinstance(name, str):
            raise TypeError('Pyflame Resolve Shot Name: name must be a string.')

        # Check if the name follows the format of a camera source (e.g. A010C0012).
        # If so, take the first 8 characters as the shot name.
        # The regex ^A\d{3}C\d{3} matches strings that start with 'A', followed by
        # three digits, followed by 'C', followed by three more digits.
        if re.match(r'^A\d{3}C\d{3}', name):
            shot_name = name[:8]
        else:
            # If the name is not a camera source, we assume it's in a different format
            # that requires splitting to find the shot name.
            # We split the name using digit sequences as delimiters.
            shot_name_split = re.split(r'(\d+)', name)

            # After splitting, we need to reassemble the shot name.
            # If there is at least one split, we check if the second element in the
            # split is alphanumeric. If it is, we concatenate the first two elements.
            # If it's not alphanumeric, we concatenate the first three elements.
            if len(shot_name_split) > 1:
                if shot_name_split[1].isalnum():
                    shot_name = shot_name_split[0] + shot_name_split[1]
                else:
                    shot_name = shot_name_split[0] + shot_name_split[1] + shot_name_split[2]
            else:
                # If the name wasn't split (no digits found), we keep the original name.
                shot_name = name

        return shot_name

    @staticmethod
    def untar(tar_file_path: str, untar_path: str, sudo_password: Optional[str]=None) -> bool:
        """
        Untar a tar file.

        Parameters:
            tar_file_path (str): Path to tar file to untar including filename.tgz/tar.
            untar_path (str): Untar destination path.
            sudo_password (bool, optional): Password for sudo.

        Returns:
            bool: True if untar successful, False if not.

        Example:
            pyflame.unzip('/home/user/file.tar', '/home/user/untarred')
        """

        # Validate arguments
        if not isinstance(untar_path, str):
            raise TypeError('pyflame.untar: untar_path must be a string')
        if sudo_password is not None and not isinstance(sudo_password, str):
            raise TypeError('pyflame.untar: sudo must be a string or None')

        # Untar
        untar_command = f'tar -xvf {tar_file_path} -C {untar_path}'
        untar_command = untar_command.split()

        if sudo_password:
            process = Popen(['sudo', '-S'] + untar_command, stdin=PIPE, stderr=PIPE, universal_newlines=True)
            stdout, stderr = process.communicate(sudo_password + '\n')
            if stderr:
                print(stderr)

        else:
            process = Popen(untar_command, stdin=PIPE, stderr=PIPE, universal_newlines=True)

        # Check if files exist in untar_path
        files_exist = False
        if os.path.isdir(untar_path):
            files = os.listdir(untar_path)
            if files:
                files_exist = True

        if files_exist:
            print('--> Untar successful.\n')
            return True
        else:
            print('--> Untar failed.\n')
            return False

    @staticmethod
    def gui_resize(value: int) -> int:
        """
        Provides scaling for Qt UI elements based on the current screen's height
        relative to a standard height of 3190 pixels(HighDPI(Retina) resolution of
        Mac Studio Display).

        Parameters:
            value (int): Value to be scaled.

        Returns:
            int: The value scaled for the current screen resolution.

        Example:
            self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        """

        # Validate argument type
        if not isinstance(value, int):
            raise TypeError('gui_resize: Value must be an integer.')

        # Baseline resolution from mac studio display
        base_screen_height = 3190

        # Get current screen resolution
        screen_resolution = main_window_res.screenGeometry()

        # Check if high DPI scaling is enabled. If so, double the screen height.
        if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
            screen_height = screen_resolution.height() * 2
        else:
            screen_height = screen_resolution.height()

        # Calculate screen ratio
        screen_ratio = round(screen_height / base_screen_height, 1)

        if screen_ratio >= 1.0:
            screen_ratio = screen_ratio * .9

        # Scale value based on screen ratio
        scaled_value = int(float(value) * screen_ratio * 1.1)

        return scaled_value

    @staticmethod
    def font_resize(value: int) -> int:
        """
        Provides scaling for fonts to be used in Qt UI elements.
        Fonts are first scaled with the gui_resize method. Then if the
        current display is a High DPI display(Retina Displays) the
        result is returned. If the current display is not a High DPI
        display the the value is scaled further by 0.8 so fonts don't
        appear to big.

        Parameters:
            value (int): Value to be scaled.

        Returns:
            int: The font size value scaled for the current screen resolution.

        Example:
            font.setPointSize(pyflame.font_resize(13)
        """

        # Validate argument types
        if not isinstance(value, int):
            raise ValueError('PyFlameButton: Value must be an integer.')

        # Scale font size through gui_resize method
        scaled_size = pyflame.gui_resize(value)

        # If screen is High DPI return scaled value, if not return scaled value * .8 to scale smaller.
        if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling')  and platform.system() == 'Darwin':
            return scaled_size
        else:
            return int(scaled_size * 0.8)

    @staticmethod
    def get_export_preset_version(preset_path: str) -> (str, str):
        """
        Get current export preset version and current Flame preset export version.
        This should be updated with each new version of Flame.

        Scripts that use this:
            Create Export Menus
            SynthEyes Export
            Create Shot

        Parameters:
            preset_path (str): Path of preset to check/update.

        Returns:
            current_export_version (str): Version of current preset export.
            export_version (str): Export preset version for currernt version of Flame.
        """

        print('Checking export preset version...')

        def get_current_export_version() -> str:
            """
            Get export version for current export preset XML.
            """

            # Open export preset XML
            export_preset_xml_tree = ET.parse(preset_path)
            root = export_preset_xml_tree.getroot()

            # Get version export preset is currently set to
            for setting in root.iter('preset'):
                current_export_version = setting.get('version')
                print(f'Current export preset version: {current_export_version}')

            return current_export_version

        def get_export_version() -> str:
            """
            Get export version for current version of Flame from the default
            Jpeg export preset XML.
            """

            # Open Flame default Jpeg export preset XML
            preset_dir = flame.PyExporter.get_presets_dir(
                flame.PyExporter.PresetVisibility.Autodesk, flame.PyExporter.PresetType.Image_Sequence
            )
            preset_path = os.path.join(
                preset_dir, "Jpeg", "Jpeg (8-bit).xml"
            )

            preset_xml_tree = ET.parse(preset_path)
            root = preset_xml_tree.getroot()

            # Get version default export preset is currently set to
            for setting in root.iter('preset'):
                default_export_version = setting.get('version')
                print(f'Flame default export preset version: {default_export_version}')

            return default_export_version

        current_export_version = get_current_export_version()
        export_version = get_export_version()

        return current_export_version, export_version

    @staticmethod
    def update_export_preset(preset_path: str) -> None:
        """
        Update export preset file version to match current version of flame being used.

        Parameters:
            preset_path (str): Path of preset to check/update.
        """

        current_export_version, export_version = pyflame.get_export_preset_version(
            preset_path=preset_path,
            )

        # If preset version if different than current export version then update preset xml
        if current_export_version != export_version:

            # Open preset XML file
            export_preset_xml_tree = ET.parse(preset_path)
            root = export_preset_xml_tree.getroot()

            # Update preset version in preset XML
            for element in root.iter('preset'):
                element.set('version', export_version)
            # Write out updated preset XML file
            export_preset_xml_tree.write(preset_path)

            print(f'Export preset updated to: {export_version}\n')
        print('Export preset current, nothing to update.\n')

pyflame = PyFlameFunctions()

# -------------------------------- PyFlame Config -------------------------------- #

class PyFlameConfig():
    """
    This class is used to create, load, and save XML configuration files for scripts.

    When initialized, an instance of PyFlameConfig is returned with attributes based on provided config_values dict.
    If a config file doesn't exist, attributes will be set from the provided config_values dict.
    If a config file does exist, attributes will be set from the config file and config_values dict will be ignored.

    Parameters:
        script_name (str): The name of the script for which the configuration is being set. This name is used in
            creating or updating the XML config file.
        script_path (str): The absolute path to the script. This is used to locate the XML config file.
        config_values (dict): A dictionary containing default configuration values. Keys and values must be strings.
            These will be used to create attributes for the class instance.
        config_xml_path (str, optional): The absolute path of the configuration XML file if it is different from the
            default (script_path + 'config/config.xml').
            In most cases, this should not be passed.

    Returns:
        instance: Returns an instance of the PyFlameConfig class with attributes set as per the configuration values.

    Example usage:
        To Load/Create a config file:

            self.settings = PyFlameConfig(
                script_name=SCRIPT_NAME,
                script_path=SCRIPT_PATH,
                    config_values={
                    'camera_path': '/opt/Autodesk',
                    'scene_scale': '100',
                    'import_type': 'Action Objects',
                    'st_map_setup': 'False',
                    }
                )

        To save settings:
            self.settings.save_config(
                script_name=SCRIPT_NAME,
                script_path=SCRIPT_PATH,
                config_values={
                    'camera_path': self.path_entry.text(),
                    'scene_scale': self.scene_scale_entry.text(),
                    'import_type': self.import_type.text(),
                    'st_map_setup': self.st_map_setup.isChecked(),
                    }
                )

        To get setting values:
            print(self.settings.camera_path)
            >'/opt/Autodesk'
    """

    def __init__(self: 'PyFlameConfig',
                 script_name: str,
                 script_path: str,
                 config_values: Dict[str, str],
                 config_xml_path: Optional[str]=None
        ):

        script_name = script_name.replace(' ', '_')

        self.load_config(script_name, script_path, config_values, config_xml_path)

        # Validate argument types

        if not isinstance(script_name, str):
            raise TypeError('script_name: script_name must be a string.')
        if not isinstance(script_path, str):
            raise TypeError('script_path: script_path must be a string.')
        elif not isinstance(config_values, dict):
            raise TypeError('config_values: config_values must be a dict.')
        for key, value in config_values.items():
            if not isinstance(key, str):
                raise TypeError('config_keys: config_values keys must be strings.')
            if value is not None and not isinstance(value, str):
                raise TypeError('config_values: config_values values must be strings.')
        if config_xml_path and not isinstance(config_xml_path, str):
            raise TypeError('config_xml_path: config_xml_path must be a string.')

    def _convert_value_type(self, value: str) -> Union[str, bool, int, float, List, Dict]:
        """
        Convert string to bool, list, dict, int, or float if needed

        Parameters:
            value (str): String representation of the value to be converted.

        Returns:
            various types: The converted value. Could be one of bool, list, dict, int, or float.
        """

        # Check if the value is already a boolean
        if isinstance(value, bool):
            return value

        # If the value is a string, perform conversions

        if value in {'True', 'False'}:
            return ast.literal_eval(value)  # convert string to bool
        elif value is None:
            return ''
        else:
            try:  # check if it can be converted to int or float
                float_value = float(value)
                if float_value.is_integer():
                    return int(float_value)  # if it's a whole number, return as int
                else:
                    return float_value  # return as float
            except ValueError:
                pass

            # check if it's a list or dict
            if (value.startswith('[') and value.endswith(']')) or (value.startswith('{') and value.endswith('}')):
                try:
                    return ast.literal_eval(value)  # convert string to list or dict
                except ValueError:
                    pass

            return value  # if all else fails, return the string as is

    def load_config(self, script_name: str, script_path: str, config_values: Dict[str, str], config_xml_path: Optional[str]=None):
        """
        Creates attributes based on config_values dict. If config file doesn't exist, attributes will be set from the provided config_values dict.
        If config file exists, attributes will be set from the config file.
        New config values can be added to the config_values dict that may not exist in the config file. These will be added to the config file when it is saved using save_config.

        Parameters:
            script_name (str): Name of script.
            script_path (str): Path to script.
            config_values (dict): Default config values. Keys and values must be strings.
            config_xml_path (str, optional): Path of config XML file if other than default.

        Returns:
            None: The method does not return a value. It sets the config as attributes of the instance.
        """

        # If config_xml_path is not specified, set it to default

        if not config_xml_path:
            config_xml_path = os.path.join(script_path, 'config/config.xml')
        print(f'    Config xml path: {config_xml_path}\n')

        def load_xml(config_values):
            """
            Load config file and return dict of values.
            """

            # Load XML config file
            xml_tree = ET.parse(config_xml_path)
            root = xml_tree.getroot()

            for child in root:
                    xml_value = child.text
                    for key, value in config_values.items():
                        if child.tag == key:
                            config_values[key] = xml_value
            return config_values

        # Load XML config file if one exists
        if os.path.isfile(config_xml_path):
            config_values = load_xml(config_values)

        # Convert config to attributes
        print('    Config values:\n')

        for key, value in config_values.items():
            value = self._convert_value_type(value) # Convert value to correct type
            setattr(self, key, value)
        print('\n')

        pyflame.message_print(message='Config loaded.', script_name=script_name)

    def save_config(self, script_name: str, script_path: str, config_values: Dict[str, str], config_xml_path: Optional[str]=None):
        """
        Save config file. If config_xml_path is not specified, it will be set to the default.
        Only values in the config_values dict will be saved/updated to the config file.

        Parameters:
            script_name (str): Name of script.
            script_path (str): Path to script.
            config_values (dict): Config values. Keys and values must be strings.
            config_xml_path (str, optional): Path of config XML file if other than default.

        Returns:
            None: The method does not return a value. It saves the config as attributes of the instance.
        """

        def save_xml(config_values, config_xml_path):

            def indent_new_element(element, level=1):
                """
                Add new element indentation to XML file.
                """

                i = '\n'
                if len(element):
                    if not element.text or not element.text.strip():
                        element.text = i + "    "
                    if not element.tail or not element.tail.strip():
                        element.tail = i
                    for elem in element:
                        indent_new_element(elem, level+1)
                    if not element.tail or not element.tail.strip():
                        element.tail = i
                else:
                    if level and (not element.tail or not element.tail.strip()):
                        element.tail = i

            def fix_indentation(file_path, spaces='    '):
                """
                Fixes indentation of XML file. All lines except first and last line will be indented by 4 spaces.
                """

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                for i in range(1, len(lines)-1):
                    if not lines[i].startswith(spaces):
                        lines[i] = spaces + lines[i]
                with open(file_path, 'w') as f:
                    f.writelines(lines)

            # Create config file if it doesn't exist
            if not os.path.exists(config_xml_path):
                print('        Creating new config file...\n')
                os.makedirs(config_path, exist_ok=True)
                with open(config_xml_path, 'w') as f:
                    f.write(f'<{script_name}_settings>\n')
                    f.write(f'</{script_name}_settings>')
                print(f'            {config_xml_path} created.\n')

            print('    Config Values:\n')

            # Save settings to config file
            xml_tree = ET.parse(config_xml_path)
            root = xml_tree.getroot()

            # loop through config_values dict and update XML file
            for key, value in config_values.items():
                xml_value = root.find(f'.//{key}')
                value = str(value)

                # Remove quotes from string values
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # If setting doesn't exist in config file, add it, if it does, update it
                if xml_value is None:
                    xml_value = ET.Element(key)
                    indent_new_element(xml_value)
                    root.append(xml_value)
                    xml_value.text = value
                    print(f'        added: {key}: {value}')
                else:
                    xml_value.text = value
                    print(f'        updated: {key}: {value}')

            print('\n')

            xml_tree.write(config_xml_path)

            # Fix xml indentation
            fix_indentation(config_xml_path)

        print(f'Saving config...\n')

        script_name = script_name.replace(' ', '_')

        # Set config paths
        if not config_xml_path:
            config_path = os.path.join(script_path, 'config')
            config_xml_path = os.path.join(config_path, 'config.xml')
        else:
            config_path = os.path.dirname(config_xml_path)

        print(f'    Config path: {config_path}')
        print(f'    Config xml path: {config_xml_path}\n')

        # Save config
        save_xml(config_values, config_xml_path)

        # Set config values as attributes
        for key, value in config_values.items():
            value = self._convert_value_type(value) # Convert value to correct type
            setattr(self, key, value)

        pyflame.message_print(message='Config saved.', script_name=script_name)

# -------------------------------- PyFlame Widget Classes -------------------------------- #

class PyFlameButton(QtWidgets.QPushButton):
    """
    Custom Qt Flame Button Widget.

    Parameters:
        text (str): Text shown on button.
        connect (callable): Function to execute when clicked.
        width (int): Button width.
            Default: 150
        height (int): Button height.
            Default: 28
        color (Color): Button color.
            -Color.GRAY: Standard gray button.
            -Color.BLUE: Blue button.
            -Color.RED: Red button.
            Default: Color.GRAY
        font (str): Button font.
            Default: PYFLAME_FONT
        font_size (int): Button font size.
            Default: PYFLAME_FONT_SIZE
        tooltip (str, optional): Button tooltip text.
            Default: None

    Methods:
        set_button_color(color): Set the color of the button after its creation.
            -Color.GRAY: Standard gray button.
            -Color.BLUE: Blue button.
            -Color.RED: Red button.

    Example:
        button = PyFlameButton(
            text='Button Name',
            connect=some_function,
            color=Color.BLUE
            )

    Usage:
        To change color of button after it's created:
            button.set_button_color(Color.RED)

        To enable/disable button:
            button.setEnabled(True)
            button.setEnabled(False)
    """

    def __init__(self: 'PyFlameButton',
                 text: str,
                 connect: Callable[..., None],
                 width: int=150,
                 height: int=28,
                 color: Color=Color.GRAY,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 tooltip: Optional[str]=None
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(text, str):
            raise TypeError('PyFlameButton: text must be a string.')
        elif not callable(connect):
            raise TypeError('PyFlameButton: connect must be a callable function or method.')
        elif not isinstance(width, int) or width <= 0:
            raise ValueError('PyFlameButton: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise ValueError('PyFlameButton: height must be a positive integer.')
        elif not isinstance(color, Color):
            raise ValueError('PyFlameButton: color must be an instance of Color Enum. '
                            'Options are: Color.GRAY, Color.BLUE, or Color.RED.')
        elif not isinstance(font, str):
            raise TypeError('PyFlameButton: font must be a string')
        elif not (isinstance(font_size, int) and font_size > 0):
            raise ValueError('PyFlameButton: font_size must be a positive integer.')
        elif tooltip is not None and not isinstance(tooltip, str):
            raise TypeError('PyFlameButton: tooltip must be a string, or None.')

        # Set button font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Build button
        self.setText(text)
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.clicked.connect(connect)
        if tooltip is not None:
            self.setToolTip(tooltip)

        self._set_stylesheet(color)

    def set_button_color(self, color: Color) -> None:
        """
        Set the color of the button after its creation.
        """

        self._set_stylesheet(color)
        self.update()  # Refresh the button's appearance

    def _set_stylesheet(self, color: Color) -> None:

        if color == Color.GRAY:
            self.setStyleSheet("""
                QPushButton {
                    color: rgb(190, 190, 190);
                    background-color: rgb(58, 58, 58);
                    border: none;
                    }
                QPushButton:hover{
                    border: 1px solid rgb(90, 90, 90);
                    }
                QPushButton:pressed{
                    color: rgb(159, 159, 159);
                    background-color: rgb(66, 66, 66);
                    border: 1px solid rgb(90, 90, 90);
                    }
                QPushButton:disabled{
                    color: rgb(116, 116, 116);
                    background-color: rgb(58, 58, 58);
                    border: none;
                    }
                QPushButton::menu-indicator{
                    subcontrol-origin: padding;
                    subcontrol-position: center right;
                    }
                QToolTip{
                    color: rgb(170, 170, 170);
                    background-color: rgb(71, 71, 71);
                    border: 10px solid rgb(71, 71, 71);
                    }
                """)

        elif color == Color.BLUE:
            self.setStyleSheet("""
                QPushButton{
                    color: rgb(190, 190, 190);
                    background-color: rgb(0, 110, 175);
                    border: none;
                    }
                QPushButton:hover{
                    border: 1px solid rgb(90, 90, 90);
                    }
                QPushButton:pressed{
                    color: rgb(159, 159, 159);
                    border: 1px solid rgb(90, 90, 90);
                    }
                QPushButton:disabled{
                    color: rgb(116, 116, 116);
                    background-color: rgb(58, 58, 58);
                    border: none;
                    }
                QToolTip{
                    color: rgb(170, 170, 170);
                    background-color: rgb(71, 71, 71);
                    border: 10px solid rgb(71, 71, 71)
                    }
                """)

        elif color == Color.RED:
            self.setStyleSheet("""
                QPushButton{
                    color: rgb(190, 190, 190);
                    background-color: rgb(200, 29, 29);
                    border: none;
                    }
                QPushButton:hover{
                    border: 1px solid rgb(90, 90, 90);
                    }
                QPushButton:pressed{
                    color: rgb(159, 159, 159);
                    border: 1px solid rgb(90, 90, 90);
                    }
                QPushButton:disabled{
                    color: rgb(116, 116, 116);
                    background-color: rgb(58, 58, 58);
                    border: none;
                    }
                QToolTip{
                    color: rgb(170, 170, 170);
                    background-color: rgb(71, 71, 71);
                    border: 10px solid rgb(71, 71, 71);
                    }
                """)

class PyFlameLabel(QtWidgets.QLabel):
    """
    Custom Qt Flame Label Widget

    Parameters:
        text (str): Label text.
        style (Style): Select from different styles.
            -Style.NORMAL: Standard label without any additional styling. Text is left aligned.
            -Style.UNDERLINE: Underlines label text. Text is centered.
            -Style.BACKGROUND: Adds a darker background to the label. Text is left aligned.
            -Style.BORDER: Adds a white border around the label with a dark background. Text is centered.
            Default: Style.NORMAL
        align (Align, optional): Align text to left, right, or center. Overrides LabelStyle alignment.
            -Align.LEFT: Aligns text to the left side of the label.
            -Align.RIGHT: Aligns text to the right side of the label.
            -Align.CENTER: Centers text within the label.
            Default: None
        height (int): Label height.
            Default: 25
        width (int): Label width.
            Default: 150
        max_width (int, optional): Maximum label width. Use if width is being set by layout.
            Default: 0
        max_height (int, optional): Maximum label height. Use if height is being set by layout.
            Default: 0
        underline_color (tuple): Color of underline when using Style.UNDERLINE. Tuple must contain 4 values.
            The fourth value is the alpha value which is a float between 0 and 1.
            Default: (40, 40, 40, 1)
        font (str): Label font.
            Default: PYFLAME_FONT
        font_size (int): Label font size.
            Default: PYFLAME_FONT_SIZE

    Example:
        label = PyFlameLabel(
            text='Label Name',
            style=Style.UNDERLINE,
            align=Align.LEFT,
            width=300
            )

    Usage:
        To set label text:
            label.setText('New Text')

        To enable/disable label:
            label.setEnabled(True)
            label.setEnabled(False)
    """

    def __init__(self: 'PyFlameLabel',
                 text: str,
                 style: Style=Style.NORMAL,
                 align: Optional[Align]=None,
                 width: int=150,
                 height: int=28,
                 max_width: Optional[int]=0,
                 max_height: Optional[int]=0,
                 underline_color: tuple=(40, 40, 40, 1),
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(text, str):
            raise TypeError('PyFlameLabel: text must be a string')
        elif not isinstance(style, Style):
            raise TypeError('PyFlameLabel: style must be an instance of Style Enum. '
                            'Style.NORMAL, Style.UNDERLINE, Style.BACKGROUND, or Style.BORDER.')
        elif align is not None and not isinstance(align, Align):
            raise TypeError('PyFlameLabel: align must be an instance of Align Enum, or None. '
                            'Options are: Align.LEFT, Align.RIGHT, Align.CENTER, or None.')
        elif not isinstance(width, int) or width <= 0:
            raise ValueError('PyFlameLabel: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise ValueError('PyFlameLabel: height must be a positive integer.')
        elif not isinstance(max_width, int) or max_width < 0:
            raise ValueError('PyFlameLabel: max_width must be a positive integer.')
        elif not isinstance(max_height, int) or max_height < 0:
            raise ValueError('PyFlameLabel: max_height must be a positive integer.')
        elif not isinstance(underline_color, tuple) or len(underline_color) != 4:
            raise TypeError('PyFlameLabel: underline_color must be a rgba value tuple. The tuple must contain 4 values.'
                            'Such as: (40, 40, 40, 0.5).')
        elif not isinstance(font, str):
            raise TypeError('PyFlameLabel: font must be a string.')
        elif not (isinstance(font_size, int) and font_size > 0):
            raise ValueError('PyFlameLabel: font_size must be a positive integer.')

        self.underline_color = underline_color

        # Set label font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Build label
        self.setText(text)
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        if max_width:
            self.setMaximumWidth(pyflame.gui_resize(max_width))
        if max_height:
            self.setMaximumHeight(pyflame.gui_resize(max_height))
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        # Set label stylesheet based on style
        if align == Align.LEFT:
            self.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        elif align == Align.RIGHT:
            self.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        elif align == Align.CENTER:
            self.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignCenter)
        else:  # None
            if style == Style.NORMAL or style == Style.BACKGROUND:
                self.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
            else:  # UNDERLINE or BORDER
                self.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignCenter)

        self._set_stylesheet(style)

    def _set_stylesheet(self, style: Style) -> None:

        if style == Style.NORMAL:
            self.setStyleSheet("""
                QLabel{
                    color: rgb(190, 190, 190);
                    }
                QLabel:disabled{
                    color: rgb(106, 106, 106);
                    }
                """)
        elif style == Style.UNDERLINE:
            self.setStyleSheet(f"""
                QLabel{{
                    color: rgb(190, 190, 190);
                    border-bottom: 1px inset rgba{self.underline_color};
                    }}
                QLabel:disabled{{
                    color: rgb(106, 106, 106);
                    }}
                """)
        elif style == Style.BACKGROUND:
            self.setStyleSheet("""
                QLabel{
                    color: rgb(190, 190, 190);
                    background-color: rgb(30, 30, 30);
                    padding-left: 5px;
                    }
                QLabel:disabled{
                    color: rgb(106, 106, 106);
                    }
                """)
        elif style == Style.BORDER:
            self.setStyleSheet("""
                QLabel{
                    color: rgb(190, 190, 190);
                    border: 1px solid rgb(64, 64, 64);
                    }
                QLabel:disabled{
                    color: rgb(106, 106, 106);
                    }
                """)

class PyFlameLineEdit(QtWidgets.QLineEdit):
    """
    Custom Qt Flame Line Edit Widget

    Parameters:
        text (str): Line edit text.
        width (int): Width of lineedit widget.
            Default: 150
        height (int): Height of lineedit widget.
            Default: 28
        max_width (int, optional): Maximum width of widget. Use if width is being set by layout.
            Default: 0
        text_changed (callable, optional): Function to call when text is changed.
            Default: None
        placeholder_text (str, optional): Temporary text to display when line edit is empty.
            Default: None
        read_only (bool): Set line edit to read only. Gives line edit a darker background and gray text. Text is not selectable.
            Default: False
        font (str): Line edit font.
            Default: PYFLAME_FONT
        font_size (int): Line edit font size.
            Default: PYFLAME_FONT_SIZE

    Example:
        entry = PyFlameLineEdit(
            text='Some text here'
            )

    Usage:
        To get text from line edit:
            entry.text()

        To set text in line edit:
            entry.setText('Some text here')

        To enable/disable line edit:
            entry.setEnabled(True)
            entry.setEnabled(False)

        To set line edit as focus (cursor will be in line edit):
            entry.setFocus()
    """

    def __init__(self: 'PyFlameLineEdit',
                 text: str,
                 width: int=150,
                 height: int=28,
                 max_width: Optional[int]=0,
                 text_changed: Optional[Callable]=None,
                 placeholder_text: Optional[str]=None,
                 read_only: bool=False,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(text, str) and not isinstance(text, int):
            raise TypeError('PyFlameLineEdit: text must be string or int.')
        elif not isinstance(width, int) or width <= 0:
            raise ValueError('PyFlameLineEdit: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise ValueError('PyFlameLineEdit: height must be a positive integer.')
        elif not isinstance(max_width, int) or max_width < 0:
            raise TypeError('PyFlameLineEdit: max_width must be a positive integer.')
        elif text_changed is not None and not callable(text_changed):
            raise TypeError('PyFlameLineEdit: text_changed must be a callable function or method, or None.')
        elif placeholder_text is not None and not isinstance(placeholder_text, str):
            raise TypeError('PyFlameLineEdit: placeholder_text must be a string, or None.')
        if not isinstance(read_only, bool):
            raise TypeError('PyFlameLineEdit: read_only must be a boolean (True or False).')
        elif not isinstance(font, str):
            raise TypeError('PyFlameLineEdit: font must be a string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlameLineEdit: font_size must be a positive integer.')

        self.read_only = read_only

        # Set font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Build line edit
        self.setText(str(text))
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        if max_width:
            self.setMaximumWidth(pyflame.gui_resize(max_width))
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        if text_changed is not None:
            self.textChanged.connect(text_changed)
        if placeholder_text is not None:
            self.setPlaceholderText(placeholder_text)

        if read_only:
            self.setReadOnly(True)

        self._set_stylesheet()

    def _set_stylesheet(self) -> None:
        """
        If the line edit is set to read only, use a darker background color and change the text color to gray.
        Otherwise, use the default Flame line edit stylesheet.
        """

        if self.read_only:
            self.setStyleSheet("""
                QLineEdit{
                    color: rgb(190, 190, 190);
                    background-color: rgb(30, 30, 30);
                    padding-left: 5px;
                    }
                QLineEdit:hover{
                    border: 1px solid rgb(90, 90, 90);
                    padding-left: 6px;
                    }
                QLineEdit:disabled{
                    color: rgb(106, 106, 106);
                    background-color: rgb(55, 55, 55);
                    border: 1px solid rgb(55, 55, 55);
                    padding-left: 5px;
                    }
                QToolTip{
                    color: rgb(170, 170, 170);
                    background-color: rgb(71, 71, 71);
                    border: none;
                    }
                """)
        else:
            self.setStyleSheet("""
                QLineEdit{
                    color: rgb(154, 154, 154);
                    background-color: rgb(55, 65, 75);
                    selection-color: rgb(38, 38, 38);
                    selection-background-color: rgb(184, 177, 167);
                    border: 1px solid rgb(55, 65, 75);
                    padding-left: 5px;
                    }
                QLineEdit:focus{
                    background-color: rgb(73, 86, 99);
                    padding-left: 5px;
                    }
                QLineEdit:hover{
                    border: 1px solid rgb(90, 90, 90);
                    padding-left: 5px;
                    }
                QLineEdit:disabled{
                    color: rgb(106, 106, 106);
                    background-color: rgb(55, 55, 55);
                    border: 1px solid rgb(55, 55, 55);
                    padding-left: 5px;
                    }
                QToolTip{
                    color: rgb(170, 170, 170);
                    background-color: rgb(71, 71, 71);
                    border: none;
                    }
                """)

    def mousePressEvent(self, event):
        if self.read_only:
            self.clearFocus()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.read_only:
            event.ignore()
        else:
            super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        if not self.read_only:  # This allows typing when not read-only.
            super().keyPressEvent(event)
        else:
            pass

    def mouseDoubleClickEvent(self, event):
        if self.read_only:
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

class PyFlameLineEditFileBrowser(QtWidgets.QLineEdit):
    """
    Custom Qt Flame Line Edit File Browser Widget

    Line Edit widget that opens a Flame file browser when clicked on.

    Parameters:
        text (str): Line edit text.
        width: (int) Width of widget.
            Default: 150
        height: (int) Height of widget.
            Default: 28
        max_width (int, optional): Maximum width of widget. Use if width is being set by layout.
            Default: 0
        placeholder_text (str, optional): Temporary text to display when line edit is empty.
            Default: None
        browser_type: BrowserType=BrowserType.FILE,
        browser_type: (BrowserType) Type of browser to open. Select from BrowserType.FILE or BrowserType.DIRECTORY.
            Default: BrowserType.FILE
        browser_ext: (list) List of file extensions to filter by when browser_type is BrowserType.FILE. Ignore if browser_type is BrowserType.DIRECTORY.
            Default: []
        browser_title: (str) Title of browser window.
            Default: 'Select File'
        browser_window_to_hide: (QtWidgets.QWidget) Window to hide when browser is open.
            Default: None
        connect: (callable) Function to call when text is changed.
            Default: None
        font (str): Line edit font.
            Default: PYFLAME_FONT
        font_size (int): Line edit font size.
            Default: PYFLAME_FONT_SIZE

    Attributes:
        path (str): The selected file or directory path.

    Example:
        path_entry = PyFlameLineEditFileBrowser(
            text=some_path,
            width=600,
            browser_type=BrowserType.FILE,
            browser_ext=[
                'exr',
                ],
            browser_title='Select Image',
            browser_window_to_hide=[self.window],
            )

    Usage:
        To get path:
            path_entry.path

        To get text from line edit:
            path_entry.text()

        To set text in line edit:
            path_entry.setText('Some text here')

        To enable/disable line edit:
            path_entry.setEnabled(True)
            path_entry.setEnabled(False)

        To set line edit as focus (cursor will be in line edit):
            path_entry.setFocus()
    """

    clicked = QtCore.Signal()

    def __init__(self,
                 text: str,
                 width: int=150,
                 height: int=28,
                 max_width: Optional[int]=0,
                 placeholder_text: Optional[str]=None,
                 browser_type: BrowserType=BrowserType.FILE,
                 browser_ext: List[str]=[],
                 browser_title: str='Select File',
                 browser_window_to_hide: Optional[QtWidgets.QWidget]=None,
                 connect: Optional[Callable[..., None]] = None,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ):
        super().__init__()

        self.path = self.text()

        # Set font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Build line edit
        self.setText(text)
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        if max_width:
            self.setMaximumWidth(pyflame.gui_resize(max_width))
        self.setReadOnly(True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        # Set placeholder text if specified
        if placeholder_text is not None:
            self.setPlaceholderText(placeholder_text)

        # If browser title is not specified, set it based on browser type
        if browser_title == '':
            if browser_type == BrowserType.FILE:
                browser_title = 'Select File'
            elif browser_type == BrowserType.DIRECTORY:
                browser_title = 'Select Directory'

        # Set browser select directory based on browser type
        if browser_type == BrowserType.FILE:
            browser_select_directory = False
        elif browser_type == BrowserType.DIRECTORY:
            browser_select_directory = True

        def open_file_browser():
            """
            Open flame file browser to select file or directory
            """

            new_path = pyflame.file_browser(
                path=self.text(),
                title=browser_title,
                extension=browser_ext,
                select_directory=browser_select_directory,
                window_to_hide=browser_window_to_hide,
            )

            if new_path:
                self.setText(new_path)
                self.path = new_path

        self.clicked.connect(open_file_browser)

        # Connect to function if one is specified
        if connect:
            self.clicked.connect(connect)

        self._set_stylesheet()

    def _set_stylesheet(self) -> None:
        self.setStyleSheet("""
            QLineEdit{
                color: rgb(154, 154, 154);
                background-color: rgb(55, 65, 75);
                selection-color: rgb(38, 38, 38);
                selection-background-color: rgb(184, 177, 167);
                border: 1px solid rgb(55, 65, 75);
                padding-left: 5px;
            }
            QLineEdit:focus{
                background-color: rgb(73, 86, 99);
            }
            QLineEdit:hover{
                border: 1px solid rgb(90, 90, 90);
            }
            QLineEdit:disabled{
                color: rgb(106, 106, 106);
                background-color: rgb(55, 55, 55);
                border: 1px solid rgb(55, 55, 55);
            }
            QToolTip{
                color: rgb(170, 170, 170);
                background-color: rgb(71, 71, 71);
                border: none;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
            return
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pass

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            super().keyPressEvent(event)
        else:
            pass

class PyFlameListWidget(QtWidgets.QListWidget):
    """
    Custom Qt Flame List Widget

    Parameters:
        width (int): Widget width.
            Default: 200
        height (int): Widget height.
            Default: 250
        max_width (int, optional): Widget maximum width. Use if width is being set by layout.
            Default: 0
        max_height (int, optional): Widget maximum height. Use if height is being set by layout.
            Default: 0
        font (str): List widget font.
            Default: PYFLAME_FONT
        font_size (int): List widget font size.
            Default: PYFLAME_FONT_SIZE

    Methods:
        add_items(items: List[str]): Add a list of strings to the list widget.

    Example:
        list_widget = PyFlameListWidget(
            width=300,
            height=300
            )

    Usage:
        To add items to list widget:
            list_widget.add_items([item1, item2, item3])

        To enable/disable list widget:
            list_widget.setEnabled(True)
            list_widget.setEnabled(False)
    """

    def __init__(self: 'PyFlameListWidget',
                 width: int=200,
                 height: int=250,
                 max_width: Optional[int]=0,
                 max_height: Optional[int]=0,
                 tooltip: Optional[str]=None,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(width, int) or width <= 0:
            raise TypeError('PyFlameListWidget: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise TypeError('PyFlameListWidget: height must be a positive integer.')
        elif not isinstance(max_width, int) or max_width < 0:
            raise TypeError('PyFlameListWidget: max_width must be a positive integer.')
        elif not isinstance(max_height, int) or max_height < 0:
            raise TypeError('PyFlameListWidget: max_height must be a positive integer.')
        elif tooltip is not None and not isinstance(tooltip, str):
            raise TypeError('PyFlameListWidget: tooltip must be a string, or None.')
        elif not isinstance(font, str):
            raise TypeError('PyFlameListWidget: font must be a string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlameListWidget: font_size must be a positive integer.')

        # Set label font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Build list widget
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        if max_width:
            self.setMaximumWidth(pyflame.gui_resize(max_width))
        if max_height:
            self.setMaximumHeight(pyflame.gui_resize(max_height))
        self.spacing()
        self.setUniformItemSizes(True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)

        # Set tooltip if specified
        if tooltip is not None:
            self.setToolTip(tooltip)

        # Set stylesheet
        self._set_stylesheet()

    def add_items(self, items: List[str]) -> None:
            """
            Add a list of strings to the list widget.

            Parameters:
                items (List[str]): The list of strings to be added.
            """
            if not isinstance(items, list):
                raise TypeError('PyFlameListWidget: items must be a list of strings.')
            if not all(isinstance(item, str) for item in items):
                raise TypeError('PyFlameListWidget: All items must be strings.')

            self.addItems(items)

    def _set_stylesheet(self) -> None:

        self.setStyleSheet(f"""
            QListWidget{{
                color: rgb(154, 154, 154);
                background-color: rgb(30, 30, 30);
                alternate-background-color: rgb(36, 36, 36);
                outline: 3px rgb(0, 0, 0);
                }}
            QListWidget::item{{
                padding-top: {pyflame.gui_resize(5)}px;  /* Increase top padding */
                padding-bottom: {pyflame.gui_resize(5)}px;  /* Increase bottom padding */
                }}
            QListWidget::item:selected{{
                color: rgb(217, 217, 217);
                background-color: rgb(71, 71, 71);
                border: 1px solid rgb(71, 71, 71);
                }}
            QScrollBar::handle{{
                background: rgb(49, 49, 49);
                }}
            QScrollBar:vertical{{
                width: {pyflame.gui_resize(20)}px;  /* Adjust the width of the vertical scrollbar */
                }}
            QScrollBar:horizontal{{
                height: {pyflame.gui_resize(20)}px;  /* Adjust the height of the horizontal scrollbar */
                }}
            QToolTip{{
                color: rgb(170, 170, 170);
                background-color: rgb(71, 71, 71);
                border: 10px solid rgb(71, 71, 71);
                }}
            """)

class PyFlamePushButton(QtWidgets.QPushButton):
    """
    Custom Qt Flame Push Button Widget

    Parameters:
        text (str): Text displayed on button.
        button_checked (bool): True or False.
        connect (callable, optional): Function to be called when button is pressed.
            Default: None
        height (int): Button height.
            Default: 28
        width (int): Button width.
            Default: 150
        max_width (int, optional): Maximum button width. Use if width is being set by layout.
            Default: 0
        tooltip (str, optional): Button tooltip text.
            Default: None
        font (str): Button font.
            Default: PYFLAME_FONT
        font_size (int): Button font size.
            Default: PYFLAME_FONT_SIZE

    Example:
        pushbutton = PyFlamePushButton(
            text='Button Name',
            button_checked=False,
            )
    Usage:
        To get button state:
            pushbutton.isChecked()

        To set button state:
            pushbutton.setChecked(True)

        To enable/disable button:
            pushbutton.setEnabled(True)
            pushbutton.setEnabled(False)
    """

    def __init__(self: 'PyFlamePushButton',
                 text: str,
                 button_checked: bool,
                 connect: Optional[Callable[..., None]]=None,
                 width: int=150,
                 height: int=28,
                 max_width: Optional[int]=0,
                 tooltip: Optional[str]=None,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types

        if not isinstance(text, str):
            raise TypeError('PyFlamePushButton: text must be string.')
        elif not isinstance(button_checked, bool):
            raise TypeError('PyFlamePushButton: button_checked must be bool.')
        elif connect is not None and not callable(connect):
            raise TypeError('PyFlamePushButton: connect must be a callable function or method, or None.')
        elif not isinstance(width, int) or width <= 0:
            raise TypeError('PyFlamePushButton: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise TypeError('PyFlamePushButton: height must be a positive integer.')
        elif not isinstance(max_width, int) or max_width < 0:
            raise TypeError('PyFlamePushButton: max_width must be a positive integer.')
        elif tooltip is not None and not isinstance(tooltip, str):
            raise TypeError('PyFlamePushButton: tooltip must be a string, or None.')
        elif not isinstance(font, str):
            raise TypeError('PyFlamePushButton: font must be string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlamePushButton: font_size must be a positive integer.')

        # Set button font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Build push button
        self.setText(text)
        self.setCheckable(True)
        self.setChecked(button_checked)
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        if max_width:
            self.setMaximumWidth(pyflame.gui_resize(max_width))
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.clicked.connect(connect)
        if tooltip is not None:
            self.setToolTip(tooltip)

        self._set_stylesheet()

    def _set_stylesheet(self) -> None:

        # Push button stylesheet
        self.setStyleSheet("""
            QPushButton{
                color: rgb(190, 190, 190);
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: .93 rgb(58, 58, 58), stop: .94 rgb(44, 54, 68));
                text-align: left;
                border-top: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: .93 rgb(58, 58, 58), stop: .94 rgb(44, 54, 68));
                border-bottom: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: .93 rgb(58, 58, 58), stop: .94 rgb(44, 54, 68));
                border-left: 1px solid rgb(58, 58, 58);
                border-right: 1px solid rgb(44, 54, 68);
                padding-left: 5px;
                }
            QPushButton:checked{
                color: rgb(217, 217, 217);
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: .93 rgb(71, 71, 71), stop: .94 rgb(50, 101, 173));
                text-align: left;
                border-top: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: .93 rgb(71, 71, 71), stop: .94 rgb(50, 101, 173));
                border-bottom: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: .93 rgb(71, 71, 71), stop: .94 rgb(50, 101, 173));
                border-left: 1px solid rgb(71, 71, 71);
                border-right: 1px solid rgb(50, 101, 173);
                padding-left: 5px;
                font: italic;
                }
            QPushButton:hover{
                border: 1px solid rgb(90, 90, 90);
                }
            QPushButton:disabled{
                color: rgb(106, 106, 106);
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: .93 rgb(58, 58, 58), stop: .94 rgb(50, 50, 50));
                font: light;
                border: none;
                }
            QToolTip{
                color: rgb(170, 170, 170);
                background-color: rgb(71, 71, 71);
                border: 10px solid rgb(71, 71, 71);
                }
            """)

class PyFlamePushButtonMenu(QtWidgets.QPushButton):
    """
    Custom Qt Flame Menu Push Button Widget

    Parameters:
        text (str): Text displayed on button.
        menu_options (list): Options shown in menu when button is pressed.
        width (int): Button width.
            Default: 150
        height (int): Button height.
            Default: 28
        max_width (int, optional): Button maximum width. Use if width is being set by layout.
            Default: 0
        connect (callable, optional): Function to be called when button is changed.
            Default: None
        align (Align): Align text to left, center or right.
            -Align.LEFT: Aligns the text to the left side of the label.
            -Align.RIGHT: Aligns the text to the right side of the label.
            -Align.CENTER: Centers the text within the label.
            Default: Align.CENTER
        font (str): Button font.
            Default: PYFLAME_FONT
        font_size (int): Button font size.
            Default: PYFLAME_FONT_SIZE

    Methods:
        update_menu(text, menu_options, connect): Use to update an existing button menu.

    Example:
        item_options = [
            'Item 1',
            'Item 2',
            'Item 3',
            'Item 4'
            ]
        menu_push_button = PyFlamePushButtonMenu(
            text='push_button_name',
            menu_options=item_options,
            align=Align.LEFT
            )

    Usage:
        To get current menu text:
            menu_push_button.text()

        To update an existing button menu:
            PyFlamePushButtonMenu.update_menu(
                text='Current Menu Selection',
                menu_options=item_options
                )

        To enable/disable list widget:
            list_widget.setEnabled(True)
            list_widget.setEnabled(False)
    """

    def __init__(self: 'PyFlamePushButtonMenu',
                 text: str,
                 menu_options: List[str],
                 width: int=150,
                 height: int=28,
                 max_width: Optional[int]=0,
                 connect: Optional[Callable[..., None]]=None,
                 align: Align=Align.CENTER,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(text, str):
            raise TypeError(f'PyFlamePushButtonMenu: text must be string.')
        elif not isinstance(menu_options, list):
            raise TypeError('PyFlamePushButtonMenu: menu_options must be list.')
        elif not isinstance(width, int) or width <= 0:
            raise ValueError('PyFlamePushButtonMenu: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise ValueError('PyFlamePushButtonMenu: height must be a positive integer.')
        elif not isinstance(max_width, int) or max_width < 0:
            raise TypeError('PyFlamePushButtonMenu: max_width must be a positive integer.')
        elif connect is not None and not callable(connect):
            raise TypeError('PyFlamePushButtonMenu: connect must be a callable function or method, or None.')
        elif not isinstance(align, Align):
            raise TypeError('PyFlamePushButtonMenu: align must be an instance of Align Enum. '
                            'Options are: Align.LEFT, Align.RIGHT, Align.CENTER.')
        elif not isinstance(font, str):
            raise TypeError('PyFlamePushButtonMenu: font must be string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlamePushButtonMenu: font_size must be a positive integer.')

        # Set button font
        self.font_size = pyflame.font_resize(font_size)
        font = QtGui.QFont(font)
        font.setPointSize(self.font_size)
        self.setFont(font)
        self.font = font

        # Build push button menu
        self.setText(text)
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        if max_width:
            self.setMaximumWidth(pyflame.gui_resize(max_width))
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        # Create menus
        self.pushbutton_menu = QtWidgets.QMenu(self)
        self.pushbutton_menu.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushbutton_menu.aboutToShow.connect(self._match_push_button_width) # Match menu width to button width
        self.pushbutton_menu.setMinimumWidth(width)

        # Menu stylesheet
        self._set_menu_stylesheet()

        # Add menu options
        for menu in menu_options:
            self.pushbutton_menu.addAction(menu, partial(self._create_menu, menu, connect))

        self.setMenu(self.pushbutton_menu)

        self._set_button_stylesheet(align)

    def _set_button_stylesheet(self, align: Align) -> None:

        self.setStyleSheet(f"""
            QPushButton{{
                color: rgb(190, 190, 190);
                background-color: rgb(45, 55, 68);
                border: none;
                text-align:{align.value};
                padding-left: 10px;
                }}
            QPushButton:disabled{{
                color: rgb(116, 116, 116);
                background-color: rgb(45, 55, 68);
                border: none;
                }}
            QPushButton:hover{{
                border: 1px solid rgb(90, 90, 90);
                }}
            QPushButton::menu-indicator{{
                image: none;
                }}
            QToolTip{{
                color: rgb(170, 170, 170);
                background-color: rgb(71, 71, 71);
                border: 10px solid rgb(71, 71, 71);
                }}
            """)

    def _set_menu_stylesheet(self) -> None:

        self.pushbutton_menu.setStyleSheet(f"""
            QMenu{{
                color: rgb(190, 190, 190);
                background-color: rgb(45, 55, 68);
                border: none;
                font: {self.font_size}px "Discreet";
                }}
            QMenu::item:selected{{
                color: rgb(217, 217, 217);
                background-color: rgb(58, 69, 81);
                }}
            """)

    def _match_push_button_width(self):

        self.pushbutton_menu.setMinimumWidth(self.size().width())

    def _create_menu(self, menu, connect):

        self.setText(menu)

        if connect:
            connect()

    def update_menu(self, text: str, menu_options: List[str], connect=None):
        """
        Use to update an existing button menu.
        """

        self.setText(text)

        self.pushbutton_menu.clear()

        for menu in menu_options:
            self.pushbutton_menu.addAction(menu, partial(self._create_menu, menu, connect))

class PyFlameSlider(QtWidgets.QLineEdit):
    """
    Custom Qt Flame Slider Widget

    Parameters:
        start_value (int or float): Initial value of the slider.
        min_value (int or float): Minimum value of the slider.
        max_value (int or float): Maximum value of the slider.
        value_is_float (bool): If True, the slider value is a float.
            Default: False
        rate (int or float): Sensitivity of the slider. The value should be between 1 and 10. Lower values are more sensitive.
            Default: 10
        width (int): Width of the slider widget.
            Default: 110
        height (int): Height of the slider widget.
            Default: 28
        connect (callable, optional): Function to call when the slider value is changed.
            Default: None
        font (str): Slider font.
            Default: PYFLAME_FONT
        font_size (int): Slider font size.
            Default: PYFLAME_FONT_SIZE
        tooltip (str, optional): Slider tooltip text.
            Default: None

    Example:
        slider = PyFlameSlider(
            start_value=0,
            min_value=-20,
            max_value=20,
            )

    Usage:
        To return the slider value as an integer or float, use:
            slider.get_value()

        To return the slider value as a string, use:
            slider.text()

        To enable/disable slider:
            slider.setEnabled(True)
            slider.setEnabled(False)
    """

    def __init__(self: 'PyFlameSlider',
                 start_value: int,
                 min_value: int,
                 max_value: int,
                 value_is_float: bool=False,
                 rate: Union[int, float]=10,
                 width: int=150,
                 height: int=28,
                 connect=None,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 tooltip: Optional[str]=None,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(start_value, (int, float)):
            raise TypeError('PyFlameSlider: start_value must be integer or float.')
        elif not isinstance(min_value, (int, float)):
            raise TypeError('PyFlameSlider: min_value must be integer or float.')
        elif not isinstance(max_value, (int, float)):
            raise TypeError('PyFlameSlider: max_value must be integer or float.')
        elif not isinstance(value_is_float, bool):
            raise TypeError('PyFlameSlider: value_is_float must be a boolean.')
        elif not isinstance(rate, (int, float)) or rate < 1 or rate > 10:
            raise TypeError('PyFlameSlider: rate must be an integer or float between 1 and 10.')
        elif not isinstance(width, int) or width <= 0:
            raise TypeError('PyFlameSlider: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise TypeError('PyFlameSlider: height must be a positive integer.')
        elif connect is not None and not callable(connect):
            raise TypeError('PyFlameSlider: connect must be a callable function or method, or None.')
        elif not isinstance(font, str):
            raise TypeError('PyFlameSlider: font must be a string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlameSlider: font_size must be a positive integer.')
        elif tooltip is not None and not isinstance(tooltip, str):
            raise TypeError('PyFlameSlider: tooltip must be a string, or None.')

        # Set slider font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size + 1))
        self.font = font
        self.setFont(self.font)

        # Scale button size for screen resolution
        self.width = pyflame.gui_resize(width)
        self.height = pyflame.gui_resize(height)

        # Build slider
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumHeight(self.height)
        self.setMinimumWidth(self.width)
        self.setMaximumWidth(self.width)

        if tooltip is not None:
            self.setToolTip(tooltip)

        if value_is_float:
            self.spinbox_type = 'Float'
        else:
            self.spinbox_type = 'Integer'

        self.rate = rate *.1
        self.min = min_value
        self.max = max_value
        self.steps = 1
        self.value_at_press = None
        self.pos_at_press = None
        self.setValue(start_value)
        self.setReadOnly(True)
        self.textChanged.connect(self.value_changed)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self._set_stylesheet()

        self.clearFocus()

        class Slider(QtWidgets.QSlider):

            def __init__(self, start_value, min_value, max_value, width):
                super(Slider, self).__init__()

                self.setMaximumHeight(pyflame.gui_resize(4))
                self.setMinimumWidth(width)
                self.setMaximumWidth(width)
                self.setMinimum(min_value)
                self.setMaximum(max_value)
                self.setValue(start_value)
                self.setOrientation(QtCore.Qt.Horizontal)

                # Slider stylesheet

                self.setStyleSheet(f"""
                    QSlider{{
                        color: rgb(55, 65, 75);
                        background-color: rgb(39, 45, 53);
                        }}
                    QSlider::groove{{
                        color: rgb(39, 45, 53);
                        background-color: rgb(39, 45, 53);
                        }}
                    QSlider::handle:horizontal{{
                        background-color: rgb(102, 102, 102);
                        width: {pyflame.gui_resize(3)}px;
                        }}
                    QSlider::disabled{{
                        color: rgb(106, 106, 106);
                        background-color: rgb(55, 65, 75);
                        }}
                    """)

                self.setDisabled(True)
                self.raise_()

        def set_slider():
            slider666.setValue(float(self.text()))
            slider666.setFont(self.font)

        slider666 = Slider(start_value, min_value, max_value, pyflame.gui_resize(width))
        self.textChanged.connect(set_slider)

        self.textChanged.connect(connect) # Connect to function when slider value is changed

        self.vbox = QtWidgets.QVBoxLayout(self)
        self.vbox.addWidget(slider666)
        self.vbox.setContentsMargins(0, pyflame.gui_resize(24), 0, 0)

    def _set_stylesheet(self) -> None:

        # Slider stylesheet
        self.setStyleSheet("""
            QLineEdit {
                color: rgb(154, 154, 154);
                background-color: rgb(55, 65, 75);
                selection-color: rgb(38, 38, 38);
                selection-background-color: rgb(184, 177, 167);
                border: none;
                padding-left: 5px;
                }
            QLineEdit:hover{
                border: 1px solid rgb(90, 90, 90);
                }
            QLineEdit:disabled{
                color: rgb(106, 106, 106);
                background-color: rgb(55, 65, 75);
                }
            QToolTip{
                color: rgb(170, 170, 170);
                background-color: rgb(71, 71, 71);
                border: 10px solid rgb(71, 71, 71);
                }
            """)

    def get_value(self):
        """
        Returns the slider's current value as an int or float.

        If the slider's value is of type 'Integer', this method will return the value as an int.
        Otherwise, the value will be returned as a float.

        Returns:
            Union[int, float]: The current slider value.
        """
        if self.spinbox_type == 'Integer':
            return int(self.text())
        else:
            return float(self.text())

    def _calculator(self):

        def clear():
            calc_lineedit.setText('')

        def button_press(key):

            if self.clean_line == True:
                calc_lineedit.setText('')

            calc_lineedit.insert(key)

            self.clean_line = False

        def plus_minus():

            if calc_lineedit.text():
                calc_lineedit.setText(str(float(calc_lineedit.text()) * -1))

        def add_sub(key):

            if calc_lineedit.text() == '':
                calc_lineedit.setText('0')

            if '**' not in calc_lineedit.text():
                try:
                    calc_num = eval(calc_lineedit.text().lstrip('0'))

                    calc_lineedit.setText(str(calc_num))

                    calc_num = float(calc_lineedit.text())

                    if calc_num == 0:
                        calc_num = 1
                    if key == 'add':
                        self.setValue(float(self.text()) + float(calc_num))
                    else:
                        self.setValue(float(self.text()) - float(calc_num))

                    self.clean_line = True
                except:
                    pass

        def enter():

            if self.clean_line == True:
                return calc_window.close()

            if calc_lineedit.text():
                try:

                    # If only single number set slider value to that number

                    self.setValue(float(calc_lineedit.text()))
                except:

                    # Do math

                    new_value = calculate_entry()
                    self.setValue(float(new_value))

            close_calc()

        def equals():

            if calc_lineedit.text() == '':
                calc_lineedit.setText('0')

            if calc_lineedit.text() != '0':

                calc_line = calc_lineedit.text().lstrip('0')
            else:
                calc_line = calc_lineedit.text()

            if '**' not in calc_lineedit.text():
                try:
                    calc = eval(calc_line)
                except:
                    calc = 0

                calc_lineedit.setText(str(calc))
            else:
                calc_lineedit.setText('1')

        def calculate_entry():

            calc_line = calc_lineedit.text().lstrip('0')

            if '**' not in calc_lineedit.text():
                try:
                    if calc_line.startswith('+'):
                        calc = float(self.text()) + eval(calc_line[-1:])
                    elif calc_line.startswith('-'):
                        calc = float(self.text()) - eval(calc_line[-1:])
                    elif calc_line.startswith('*'):
                        calc = float(self.text()) * eval(calc_line[-1:])
                    elif calc_line.startswith('/'):
                        calc = float(self.text()) / eval(calc_line[-1:])
                    else:
                        calc = eval(calc_line)
                except:
                    calc = 0
            else:
                calc = 1

            calc_lineedit.setText(str(float(calc)))

            return calc

        def close_calc():

            calc_window.close()

            self.setStyleSheet("""
                QLineEdit{
                    color: rgb(154, 154, 154);
                    background-color: rgb(55, 65, 75);
                    selection-color: rgb(154, 154, 154);
                    selection-background-color: rgb(55, 65, 75);
                    border: none;
                    padding-left: 5px;
                    }
                QLineEdit:hover{
                    border: 1px solid rgb(90, 90, 90);
                    }
                """)

        def revert_color():

            self.setStyleSheet("""
                QLineEdit{
                    color: rgb(154, 154, 154);
                    background-color: rgb(55, 65, 75);
                    selection-color: rgb(154, 154, 154);
                    selection-background-color: rgb(55, 65, 75);
                    border: none;
                    padding-left: 5px;
                    }
                QLineEdit:hover{
                    border: 1px solid rgb(90, 90, 90);
                    }
                """)

        self.clean_line = False

        calc_window = QtWidgets.QWidget()
        calc_window.setMinimumSize(QtCore.QSize(210, 280))
        calc_window.setMaximumSize(QtCore.QSize(210, 280))
        calc_window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Popup)
        calc_window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        calc_window.destroyed.connect(revert_color)
        calc_window.move(QtGui.QCursor.pos().x() - 110, QtGui.QCursor.pos().y() - 290)
        calc_window.setStyleSheet("""
            background-color: rgb(36, 36, 36)
        """)

        # Label
        calc_label = QtWidgets.QLabel('Calculator', calc_window)
        calc_label.setAlignment(QtCore.Qt.AlignCenter)
        calc_label.setMinimumHeight(28)
        font = QtGui.QFont(PYFLAME_FONT)
        font.setPointSize(pyflame.font_resize(PYFLAME_FONT_SIZE))
        calc_label.setFont(font)
        calc_label.setStyleSheet("""
            color: rgb(154, 154, 154);
            background-color: rgb(57, 57, 57);
        """)

        #  LineEdit
        calc_lineedit = QtWidgets.QLineEdit('', calc_window)
        calc_lineedit.setMinimumHeight(28)
        calc_lineedit.setFocus()
        calc_lineedit.returnPressed.connect(enter)
        calc_lineedit.setFont(font)
        calc_lineedit.setStyleSheet("""
            QLineEdit{
                color: rgb(154, 154, 154);
                background-color: rgb(55, 65, 75);
                selection-color: rgb(38, 38, 38);
                selection-background-color: rgb(184, 177, 167);
                border: none;
                padding-left: 5px;
                }
            """)

        # Limit input to numbers and math symbols
        try:
            from PySide6.QtCore import QRegularExpression
            from PySide6.QtGui import QRegularExpressionValidator as QValidator
        except ImportError:
            from PySide2.QtCore import QRegExp as QRegularExpression
            from PySide2.QtGui import QRegExpValidator as QValidator

        regex = QRegularExpression('[0-9_,=,/,*,+,\-,.]+')
        validator = QValidator(regex)
        calc_lineedit.setValidator(validator)

        # Buttons
        def calc_null():
            # For blank button - this does nothing
            pass

        class FlameButton(QtWidgets.QPushButton):

            def __init__(self, text, size_x, size_y, connect, parent, *args, **kwargs):
                super(FlameButton, self).__init__(*args, **kwargs)

                self.setText(text)
                self.setParent(parent)
                self.setMinimumSize(size_x, size_y)
                self.setMaximumSize(size_x, size_y)
                self.setFocusPolicy(QtCore.Qt.NoFocus)
                self.clicked.connect(connect)

                # Set button font
                font = QtGui.QFont(PYFLAME_FONT)
                font.setPointSize(pyflame.font_resize(PYFLAME_FONT_SIZE))
                self.setFont(font)

                self.setStyleSheet("""
                    QPushButton{
                        color: rgb(154, 154, 154);
                        background-color: rgb(58, 58, 58);
                        border: none;
                        }
                    QPushButton:hover{
                        border: 1px solid rgb(90, 90, 90);
                        }
                    QPushButton:pressed{
                        color: rgb(159, 159, 159);
                        background-color: rgb(66, 66, 66);
                        border: none;
                        }
                    QPushButton:disabled{
                        color: rgb(116, 116, 116);
                        background-color: rgb(58, 58, 58);
                        border: none;
                        }
                    """)

        blank_button = FlameButton('', 40, 28, calc_null, calc_window)
        blank_button.setDisabled(True)
        plus_minus_button = FlameButton('+/-', 40, 28, plus_minus, calc_window)
        plus_minus_button.setStyleSheet("""
            color: rgb(154, 154, 154);
            background-color: rgb(45, 55, 68);
        """)

        add_button = FlameButton('Add', 40, 28, (partial(add_sub, 'add')), calc_window)
        sub_button = FlameButton('Sub', 40, 28, (partial(add_sub, 'sub')), calc_window)

        #  --------------------------------------- #

        clear_button = FlameButton('C', 40, 28, clear, calc_window)
        equal_button = FlameButton('=', 40, 28, equals, calc_window)
        div_button = FlameButton('/', 40, 28, (partial(button_press, '/')), calc_window)
        mult_button = FlameButton('/', 40, 28, (partial(button_press, '*')), calc_window)

        #  --------------------------------------- #

        _7_button = FlameButton('7', 40, 28, (partial(button_press, '7')), calc_window)
        _8_button = FlameButton('8', 40, 28, (partial(button_press, '8')), calc_window)
        _9_button = FlameButton('9', 40, 28, (partial(button_press, '9')), calc_window)
        minus_button = FlameButton('-', 40, 28, (partial(button_press, '-')), calc_window)

        #  --------------------------------------- #

        _4_button = FlameButton('4', 40, 28, (partial(button_press, '4')), calc_window)
        _5_button = FlameButton('5', 40, 28, (partial(button_press, '5')), calc_window)
        _6_button = FlameButton('6', 40, 28, (partial(button_press, '6')), calc_window)
        plus_button = FlameButton('+', 40, 28, (partial(button_press, '+')), calc_window)

        #  --------------------------------------- #

        _1_button = FlameButton('1', 40, 28, (partial(button_press, '1')), calc_window)
        _2_button = FlameButton('2', 40, 28, (partial(button_press, '2')), calc_window)
        _3_button = FlameButton('3', 40, 28, (partial(button_press, '3')), calc_window)
        enter_button = FlameButton('Enter', 40, 61, enter, calc_window)

        #  --------------------------------------- #

        _0_button = FlameButton('0', 89, 28, (partial(button_press, '0')), calc_window)
        point_button = FlameButton('.', 40, 28, (partial(button_press, '.')), calc_window)

        gridbox = QtWidgets.QGridLayout()
        gridbox.setVerticalSpacing(5)
        gridbox.setHorizontalSpacing(5)

        gridbox.addWidget(calc_label, 0, 0, 1, 4)

        gridbox.addWidget(calc_lineedit, 1, 0, 1, 4)

        gridbox.addWidget(blank_button, 2, 0)
        gridbox.addWidget(plus_minus_button, 2, 1)
        gridbox.addWidget(add_button, 2, 2)
        gridbox.addWidget(sub_button, 2, 3)

        gridbox.addWidget(clear_button, 3, 0)
        gridbox.addWidget(equal_button, 3, 1)
        gridbox.addWidget(div_button, 3, 2)
        gridbox.addWidget(mult_button, 3, 3)

        gridbox.addWidget(_7_button, 4, 0)
        gridbox.addWidget(_8_button, 4, 1)
        gridbox.addWidget(_9_button, 4, 2)
        gridbox.addWidget(minus_button, 4, 3)

        gridbox.addWidget(_4_button, 5, 0)
        gridbox.addWidget(_5_button, 5, 1)
        gridbox.addWidget(_6_button, 5, 2)
        gridbox.addWidget(plus_button, 5, 3)

        gridbox.addWidget(_1_button, 6, 0)
        gridbox.addWidget(_2_button, 6, 1)
        gridbox.addWidget(_3_button, 6, 2)
        gridbox.addWidget(enter_button, 6, 3, 2, 1)

        gridbox.addWidget(_0_button, 7, 0, 1, 2)
        gridbox.addWidget(point_button, 7, 2)

        calc_window.setLayout(gridbox)

        calc_window.show()

    def value_changed(self):

        # If value is greater or less than min/max values set values to min/max

        if int(self.value()) < self.min:
            self.setText(str(self.min))
        if int(self.value()) > self.max:
            self.setText(str(self.max))

    def mousePressEvent(self, event):

        if event.buttons() == QtCore.Qt.LeftButton:
            self.value_at_press = self.value()
            self.pos_at_press = event.pos()
            self.setCursor(QtGui.QCursor(QtCore.Qt.SizeHorCursor))
            self.setFont(self.font)
            self.setStyleSheet("""
                QLineEdit{
                    color: rgb(217, 217, 217);
                    background-color: rgb(73, 86, 99);
                    selection-color: rgb(154, 154, 154);
                    selection-background-color: rgb(73, 86, 99);
                    border: none; padding-left: 5px;
                    }
                QLineEdit:hover{
                    border: 1px solid rgb(90, 90, 90);
                    }
                """)

    def mouseReleaseEvent(self, event):

        if event.button() == QtCore.Qt.LeftButton:

            # Open calculator if button is released within 10 pixels of button click

            if event.pos().x() in range((self.pos_at_press.x() - 10), (self.pos_at_press.x() + 10)) and event.pos().y() in range((self.pos_at_press.y() - 10), (self.pos_at_press.y() + 10)):
                self._calculator()
            else:
                self.setStyleSheet("""
                    QLineEdit{
                        color: rgb(154, 154, 154);
                        background-color: rgb(55, 65, 75);
                        selection-color: rgb(154, 154, 154);
                        selection-background-color: rgb(55, 65, 75);
                        border: none;
                        padding-left: 5px;
                        }
                    QLineEdit:hover{
                        border: 1px solid rgb(90, 90, 90);
                        }
                    """)

            self.value_at_press = None
            self.pos_at_press = None
            self.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
            return

        super(PyFlameSlider, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):

        if event.buttons() != QtCore.Qt.LeftButton:
            return

        if self.pos_at_press is None:
            return

        steps_mult = self.getStepsMultiplier(event)
        delta = event.pos().x() - self.pos_at_press.x()

        if self.spinbox_type == 'Integer':
            delta /= 10 * self.rate # Make movement less sensiteve.
        else:
            delta /= 100 * self.rate
        delta *= self.steps * steps_mult

        value = self.value_at_press + delta
        self.setValue(value)

        super(PyFlameSlider, self).mouseMoveEvent(event)

    def getStepsMultiplier(self, event):

        steps_mult = 1

        if event.modifiers() == QtCore.Qt.CTRL:
            steps_mult = 10
        elif event.modifiers() == QtCore.Qt.SHIFT:
            steps_mult = 0.10

        return steps_mult

    def setMinimum(self, value):

        self.min = value

    def setMaximum(self, value):

        self.max = value

    def setSteps(self, steps):

        if self.spinbox_type == 'Integer':
            self.steps = max(steps, 1)
        else:
            self.steps = steps

    def value(self):

        if self.spinbox_type == 'Integer':
            return int(self.text())
        else:
            return float(self.text())

    def setValue(self, value):

        if self.min is not None:
            value = max(value, self.min)

        if self.max is not None:
            value = min(value, self.max)

        if self.spinbox_type == 'Integer':
            self.setText(str(int(value)))
        else:
            # Keep float values to two decimal places

            self.setText('%.2f' % float(value))

class PyFlameTextEdit(QtWidgets.QTextEdit):
    """
    Custom Qt Flame Text Edit Widget

    Parameters:
        text (str): Text to be displayed.
        width (int): Width of text edit.
            Default: 300
        height (int): Height of text edit.
            Default: 100
        max_width (int, optional): Maximum width of text edit. Use if width is being set by layout.
            Default: 0
        max_height (int, optional): Maximum height of text edit. Use if height is being set by layout.
            Default: 0
        read_only (bool): Make text in window read only.
            Default: False
        font (str): Text font.
            Default: PYFLAME_FONT
        font_size (int): Text font size.
            Default: PYFLAME_FONT_SIZE

    Methods:
        text(): Returns the text in the text edit.
        setText(text): Sets the text in the text edit.

    Example:
        text_edit = PyFlameTextEdit(
            text='Some text here',
            width=300,
            height=150,
            read_only=True,
            font_size=30
            )

    Usage:
        To get text from text edit:
            text_edit.text()

        To set text in text edit:
            text_edit.setText('Some text here')

        To enable/disable text edit:
            text_edit.setEnabled(True)
            text_edit.setEnabled(False)
    """

    def __init__(self: 'PyFlameTextEdit',
                 text: str,
                 width: int=300,
                 height: int=100,
                 max_width: Optional[int]=0,
                 max_height: Optional[int]=0,
                 read_only: bool=False,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(text, str):
            raise TypeError('PyFlameTextEdit: text must be a string.')
        elif not isinstance(width, int) or width <= 0:
            raise ValueError('PyFlameTextEdit: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise ValueError('PyFlameTextEdit: height must be a positive integer.')
        elif not isinstance(max_width, int) or max_width < 0:
            raise TypeError('PyFlameTextEdit: max_width must be a positive integer.')
        elif not isinstance(max_height, int) or max_height < 0:
            raise TypeError('PyFlameTextEdit: max_height must be a positive integer.')
        elif not isinstance(read_only, bool):
            raise TypeError('PyFlameTextEdit: read_only must be a boolean.')
        elif not isinstance(font, str):
            raise TypeError('PyFlameTextEdit: font must be a string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlameTextEdit: font_size must be a positive integer.')

        # Set font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Build text edit
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        if max_width:
            self.setMaximumWidth(pyflame.gui_resize(max_width))
        if max_height:
            self.setMaximumHeight(pyflame.gui_resize(max_height))
        self.setText(text)
        self.setReadOnly(read_only)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self._set_text_edit_style(read_only)

    def _set_text_edit_style(self, read_only: bool):

        if read_only:
            self.setStyleSheet(f"""
                QTextEdit{{
                    color: rgb(154, 154, 154);
                    background-color: rgb(55, 65, 75);
                    selection-color: rgb(38, 38, 38);
                    selection-background-color: rgb(184, 177, 167);
                    border: none;
                    padding-left: 5px;
                    }}
                QScrollBar::handle{{
                    background: rgb(49, 49, 49);
                    }}
                QScrollBar:vertical{{
                    width: {pyflame.gui_resize(20)}px;  /* Adjust the width of the vertical scrollbar */
                    }}
                QScrollBar:horizontal{{
                    height: {pyflame.gui_resize(20)}px;  /* Adjust the height of the horizontal scrollbar */
                    }}
                """)
        else:
            self.setStyleSheet(f"""
                QTextEdit{{
                    color: rgb(154, 154, 154);
                    background-color: rgb(55, 65, 75);
                    selection-color: rgb(38, 38, 38);
                    selection-background-color: rgb(184, 177, 167);
                    border: none;
                    padding-left: 5px;
                    }}
                QTextEdit:focus{{
                    background-color: rgb(73, 86, 99);
                    }}
                QScrollBar::handle{{
                    background: rgb(49, 49, 49);
                    }}
                QScrollBar:vertical{{
                    width: {pyflame.gui_resize(20)}px;  /* Adjust the width of the vertical scrollbar */
                    }}
                QScrollBar:horizontal{{
                    height: {pyflame.gui_resize(20)}px;  /* Adjust the height of the horizontal scrollbar */
                    }}
                """)

    def text(self) -> str:
        """
        Returns the text in the text edit.
        """

        return self.toPlainText()

    def setText(self, text: str) -> None:
        """
        Sets the text in the text edit.

        Parameters:
            text (str): Text to be added to TextEdit.
        """

        self.setPlainText(text)

class PyFlameTokenPushButton(QtWidgets.QPushButton):
    """
    Custom Qt Flame Token Push Button Widget

    Parameters:
        text (str): Text displayed on button.
            Default: 'Add Token'
        token_dict (dict): Dictionary defining tokens. {'Token Name': '<Token>'}.
            Default: {}
        token_dest (QtWidgets.QLineEdit): Where token value will be applied to.
            Default: None
        clear_dest (bool): Clear destination QLineEdit before inserting token.
            Default: False
        width (int): Button width.
            Default: 150
        height (int): Button height.
            Default: 28
        max_width (int, optional): Button max width. Use if width is being set by layout.
            Default: 0
        font (str): Button font.
            Default: PYFLAME_FONT
        font_size (int): Button font size.
            Default: PYFLAME_FONT_SIZE

    Methods:
        add_menu_options(new_options): Add new menu options to the existing token menu and clear old options.

    When a token is chosen from the menu, it is inserted into the QLineEdit widget specified by token_dest.

    Example:
        token_push_button = PyFlameTokenPushButton(
            token_dict={
                'Token 1': '<Token1>',
                'Token2': '<Token2>',
                },
            token_dest=some_qlineedit_entry_field,
            clear_dest=True,
            )

    Usage:
        To enable/disable token push button:
            token_push_button.setEnabled(True)
            token_push_button.setEnabled(False)

        To add new menu options to the existing token menu:
            token_push_button.add_menu_options(new_options={'New Token Name': '<New Token>'})
    """

    def __init__(self: 'PyFlameTokenPushButton',
                 text: str='Add Token',
                 token_dict: Dict[str, str]={},
                 token_dest: QtWidgets.QLineEdit=None,
                 clear_dest: bool=False,
                 width: int=150,
                 max_width: Optional[int]=0,
                 height: int=28,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(text, str):
            raise TypeError('PyFlameTokenPushButton: text must be a string.')
        elif not isinstance(token_dict, dict):
            raise TypeError('PyFlameTokenPushButton: token_dict must be a dictionary.')
        elif not isinstance(token_dest, QtWidgets.QLineEdit):
            raise TypeError('PyFlameTokenPushButton: token_dest must be a QtWidgets.QLineEdit.')
        elif not isinstance(width, int) or width <= 0:
            raise ValueError('PyFlameTokenPushButton: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise ValueError('PyFlameTokenPushButton: height must be a positive integer.')
        elif not isinstance(max_width, int) or max_width < 0:
            raise TypeError('PyFlameTokenPushButton: max_width must be a positive integer.')
        elif not isinstance(font, str):
            raise TypeError('PyFlameTokenPushButton: font must be a string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlameTokenPushButton: font_size must be a positive integer.')

        # Set button font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Build token push button
        self.setText(text)
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        if max_width:
            self.setMaximumWidth(pyflame.gui_resize(max_width))
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        # Create the token menu
        self.token_menu = QtWidgets.QMenu(self)
        self.token_menu.setFocusPolicy(QtCore.Qt.NoFocus)

        def token_action_menu():

            def insert_token(token):
                if clear_dest:
                    token_dest.setText('')
                for key, value in token_dict.items():
                    if key == token:
                        token_name = value
                        token_dest.insert(token_name)

            for key, value in token_dict.items():
                self.token_menu.addAction(key, partial(insert_token, key))

        token_action_menu()
        self.setMenu(self.token_menu)

        self._set_stylesheet()
        self._set_menu_style_sheet()

        self.token_dict = token_dict
        self.token_dest = token_dest
        self.clear_dest = clear_dest

    def _set_stylesheet(self):

        self.setStyleSheet("""
            QPushButton{
                color: rgb(154, 154, 154);
                background-color: rgb(45, 55, 68);
                border: none;
                }
            QPushButton:hover{
                border: 1px solid rgb(90, 90, 90);
                }
            QPushButton:disabled{
                color: rgb(106, 106, 106);
                background-color: rgb(45, 55, 68);
                border: none;
                }
            QPushButton::menu-indicator{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                }
            QToolTip{
                color: rgb(170, 170, 170);
                background-color: rgb(71, 71, 71);
                border: 10px solid rgb(71, 71, 71);
                }
            """)

    def _set_menu_style_sheet(self):

        self.menu().setStyleSheet("""
            QMenu{
                color: rgb(154, 154, 154);
                background-color: rgb(45, 55, 68);
                border: none;
                }
            QMenu::item:selected{
                color: rgb(217, 217, 217);
                background-color: rgb(58, 69, 81);
                }
            """)

    def add_menu_options(self, new_options: Dict[str, str]):
            """
            Add new menu options to the existing token menu and clear old options.

            Parameters:
                new_options (dict): Dictionary of new token options to add. {'New Token Name': '<New Token>'}.
            """
            if not isinstance(new_options, dict):
                raise TypeError('add_menu_options: new_options must be a dictionary.')

            def insert_new_token(token):
                if self.clear_dest:
                    self.token_dest.setText('')
                for key, value in self.token_dict.items():
                    if key == token:
                        token_name = value
                        self.token_dest.insert(token_name)

            # Clear existing token menu and dictionary
            self.token_menu.clear()
            self.token_dict.clear()

            # Add new menu options
            for key, value in new_options.items():
                self.token_dict[key] = value
                self.token_menu.addAction(key, partial(insert_new_token, key))

class PyFlameTreeWidget(QtWidgets.QTreeWidget):
    """
    Custom Qt Flame Tree Widget

    Parameters:
        column_names (list): List of names to be used for column names in tree.
        connect (callable, optional): Function to call when item in tree is clicked on.
            Default: None
        width (int): Width of tree widget.
            Default: 100
        height (int): Height of tree widget.
            Default: 100
        max_width (int, optional): Maximum width of tree widget. Use if width is being set by layout.
            Default: 0
        max_height (int, optional): Maximum height of tree widget. Use if height is being set by layout.
            Default: 0
        font (str): Tree widget font.
            Default: PYFLAME_FONT
        font_size (int):
            Default: PYFLAME_FONT_SIZE

    Example:
        columns = [
            'Column 1',
            'Column 2',
            'Column 3',
            'Column4'
            ]
        tree = PyFlameTreeWidget(
            column_names=columns,
            width=300,
            height=200
            )

    Usage:
        To enable/disable tree widget:
            tree.setEnabled(True)
            tree.setEnabled(False)
    """

    def __init__(self: 'PyFlameTreeWidget',
                 column_names: List[str],
                 connect: Optional[Callable[..., None]]=None,
                 width: int=100,
                 height: int=100,
                 max_width: Optional[int]=0,
                 max_height: Optional[int]=0,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(column_names, list):
            raise TypeError('PyFlameTreeWidget: column_names must be a list.')
        elif connect is not None and not callable(connect):
            raise TypeError('PyFlameLineEdit: text_changed must be a callable function or method, or None.')
        elif not isinstance(width, int) or width <= 0:
            raise ValueError('PyFlameTreeWidget: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise ValueError('PyFlameTreeWidget: height must be a positive integer.')
        elif not isinstance(max_width, int) or max_width < 0:
            raise TypeError('PyFlameTreeWidget: max_width must be a positive integer.')
        elif not isinstance(max_height, int) or max_height < 0:
            raise TypeError('PyFlameTreeWidget: max_height must be a positive integer.')
        elif not isinstance(font, str):
            raise TypeError('PyFlameTreeWidget: font must be a string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlameTreeWidget: font_size must be a positive integer.')

        self.header_font = font
        self.header_font_size = pyflame.font_resize(font_size)

        # Set tree font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.gui_resize(font_size))
        self.setFont(font)

        # Build tree widget
        self.setFixedSize(pyflame.gui_resize(width), pyflame.gui_resize(height))
        if max_width:
            self.setMaximumWidth(pyflame.gui_resize(max_width))
        if max_height:
            self.setMaximumHeight(pyflame.gui_resize(max_height))
        self.setSortingEnabled(True)
        self.sortByColumn(0, QtCore.Qt.AscendingOrder)
        self.setAlternatingRowColors(True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.clicked.connect(connect)
        self.setHeaderLabels(column_names)
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.itemCollapsed.connect(self._on_item_collapsed) # Prevent top-level item from collapsing

        self._set_stylesheet()

    def _on_item_collapsed(self, item):
        """
        Prevent the top-level item from collapsing.
        """

        # Check if the item is a top-level item
        if self.indexOfTopLevelItem(item) != -1:
            # Re-expand the top-level item
            self.expandItem(item)

    def _set_stylesheet(self):

        self.setStyleSheet(f"""
            QTreeWidget{{
                color: rgb(154, 154, 154);
                background-color: rgb(30, 30, 30);
                alternate-background-color: rgb(36, 36, 36);
                border: none;
                }}
            QTreeWidget::item{{
                padding-top: {pyflame.gui_resize(5)}px;  /* Increase top padding */
                padding-bottom: {pyflame.gui_resize(5)}px;  /* Increase bottom padding */
                font-size: {self.header_font_size}px;
            }}
            QHeaderView::section{{
                color: rgb(154, 154, 154);
                background-color: rgb(57, 57, 57);
                border: none;
                padding-left: {pyflame.gui_resize(10)}px;
                font-family: {self.header_font};
                font-size: {self.header_font_size}px;
                height: {pyflame.gui_resize(28)}px;
                }}
            QTreeWidget:item:selected{{
                color: rgb(217, 217, 217);
                background-color: rgb(71, 71, 71);
                selection-background-color: rgb(153, 153, 153);
                }}
            QTreeWidget:item:selected:active{{
                color: rgb(153, 153, 153);
                border: none;
                }}
            QTreeWidget:disabled{{
                color: rgb(101, 101, 101);
                background-color: rgb(34, 34, 34);
                }}
            QMenu{{
                color: rgb(154, 154, 154);
                background-color: rgb(36, 48, 61);
                }}
            QMenu::item:selected{{
                color: rgb(217, 217, 217);
                background-color: rgb(58, 69, 81);
                }}
            QScrollBar::handle{{
                background: rgb(49, 49, 49);
                }}
            QScrollBar:vertical{{
                width: {pyflame.gui_resize(20)}px;  /* Adjust the width of the vertical scrollbar */
                }}
            QScrollBar:horizontal{{
                height: {pyflame.gui_resize(20)}px;  /* Adjust the height of the horizontal scrollbar */
                }}
            QTreeWidget::branch:has-siblings:!adjoins-item {{
                border-image: none;
                background: transparent;
            }}
            QTreeWidget::branch:has-siblings:adjoins-item {{
                border-image: none;
                background: transparent;
            }}
            QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {{
                border-image: none;
                background: transparent;
            }}
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {{
                border-image: none;
                background: transparent;
            }}
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings  {{
                border-image: none;
                background: transparent;
            }}
            QTreeWidget:item:selected {{
                color: rgb(217, 217, 217);
                background-color: rgb(71, 71, 71); // This is the color of the highlighted item
            }}
            """)

# -------------------------------- PyFlame Window Classes -------------------------------- #

class PyFlameMessageWindow(QtWidgets.QDialog):
    """
    Custom Qt Flame Message Window

    Parameters:
        message (str): Text displayed in body of window.
        script_name (str, optional): Name of script. Used to set default window title.
            If set to None defaults for message type are used.
            Defaults by message type:
                -MessageType.INFO: Python Hook
                -MessageType.OPERATION_COMPLETE: Python Hook: Operation Complete
                -MessageType.CONFIRM: Python Hook: Confirm Operation
                -MessageType.ERROR: Python Hook: Error
                -MessageType.WARNING: Python Hook: Warning
        type (MessageType): Type of message window to be shown.
            -MessageType.INFO
                Title: SCRIPT_NAME
                Title with no script_name: Python Hook
                Window lines: Blue
                Buttons: Ok
            -MessageType.OPERATION_COMPLETE
                Title: SCRIPT_NAME: Operation Complete
                Title with no script_name: Python Hook: Operation Complete
                Window lines: Blue
                Buttons: Ok
            -MessageType.CONFIRM
                Title: SCRIPT_NAME: Confirm Operation
                Title with no script_name: Python Hook: Confirm Operation
                Window lines: Grey
                Buttons: Confirm, Cancel
                Returns bool value.
            -MessageType.ERROR
                Title: SCRIPT_NAME: Error
                Title with no script_name: Python Hook: Error
                Window lines: Yellow
                Buttons: Ok
            -MessageType.WARNING
                Title: SCRIPT_NAME: Warning
                Title with no script_name: Python Hook: Warning
                Window lines: Red
                Buttons: Confirm, Cancel
                Returns bool value.
            Default: MessageType.INFO
        title (str, optional): Use to override default title for message type.
            Default: None
        time: (int) Time in seconds to display message in flame message area.
            Default: 3
        font (str): Message font.
            Default: PYFLAME_FONT
        font_size (int) Message font size.
            Default: PYFLAME_FONT_SIZE
        parent: (QtWidget, optional) Parent window.
            Default: None

    Examples:
        PyFlameMessageWindow(
            message='Unable to create folder.<br>Check folder permissions',
            type=MessageType.ERROR
            )

        To get a bool value:

            proceed = PyFlameMessageWindow(
                message='Do you want to do this?',
                type=MessageType.CONFIRM,
                ):
                more code...
    """

    def __init__(self: 'PyFlameMessageWindow',
                 message: str,
                 script_name: str=None,
                 type: MessageType=MessageType.INFO,
                 title: Optional[str]=None,
                 time: int=3,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 parent=None
                 ):
        super().__init__()

        # Validate argument types
        if not isinstance(message, str):
            raise TypeError('PyFlameMessageWindow: message must be a string.')
        elif script_name is not None and not isinstance(script_name, str):
            raise TypeError('PyFlameMessageWindow: script_name must be a string or None.')
        elif not isinstance(type, MessageType):
            raise ValueError('PyFlameMessageWindow: type must be an instance of Type Enum. '
                            'MessageType.INFO, MessageType.OPERATION_COMPLETE, MessageType.CONFIRM, MessageType.ERROR, '
                            'MessageType.WARNING.')
        elif title is not None and not isinstance(title, str):
            raise TypeError('PyFlameMessageWindow: title must be a string or None.')
        elif not isinstance(time, int):
            raise TypeError('PyFlameMessageWindow: time must be an integer.')
        elif not isinstance(font, str):
            raise TypeError('PyFlameMessageWindow: font must be a string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlameMessageWindow: font_size must be a positive integer.')

        self.type = type
        self.confirmed = False

        # Set common button
        self.button = PyFlameButton(
            text='Ok',
            connect=self.confirm,
            width=110,
            color=Color.BLUE,
        )

        self.confirm_button = PyFlameButton(
            text='Confirm',
            connect=self.confirm,
            width=110,
            color=Color.BLUE,
        )

        # Set message window type options
        if type == MessageType.INFO:
            if not title:
                if script_name:
                    title = script_name
                else:
                    title = 'Python Hook'

        elif type == MessageType.OPERATION_COMPLETE:
            if not title:
                if script_name:
                    title = f'{script_name}: Operation Complete'
                else:
                    title = 'Python Hook: Operation Complete'

        elif type == MessageType.ERROR:
            if not title:
                if script_name:
                    title = f'{script_name}: Error'
                else:
                    title = 'Python Hook: Error'

        elif type == MessageType.CONFIRM:
            if not title:
                if script_name:
                    title = f'{script_name}: Confirm Operation'
                else:
                    title = 'Python Hook: Confirm Operation'
            self.button.setText = 'Confirm'

        elif type == MessageType.WARNING:
            if not title:
                if script_name:
                    title = f'{script_name}: Warning'
                else:
                    title = 'Python Hook: Warning'
            self.confirm_button.set_button_color(Color.RED)

        message_font = font

        # Set font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Set Window size for screen
        width = 500
        height = 330
        self.width = pyflame.gui_resize(width)
        self.height = pyflame.gui_resize(height)

        # Create message window
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setFixedSize(QtCore.QSize(self.width, self.height))
        self.setStyleSheet("""
            background-color: rgb(36, 36, 36);
        """)

        resolution = main_window_res.screenGeometry()
        self.move((resolution.width() / 2) - (self.frameSize().width() / 2),
                  (resolution.height() / 2) - (self.frameSize().height() / 2))

        self.setParent(parent)

        self.grid = QtWidgets.QGridLayout()

        self.main_label = PyFlameLabel(
            text=title,
            width=width,
            font_size=24,
            )

        self.message_text_edit = QtWidgets.QTextEdit(message)
        self.message_text_edit.setDisabled(True)
        self.message_text_edit.setFont(font)
        self.message_text_edit.setStyleSheet(f"""
            QTextEdit{{
                color: rgb(190, 190, 190);
                background-color: rgb(36, 36, 36);
                selection-color: rgb(190, 190, 190);
                selection-background-color: rgb(36, 36, 36);
                border: none;
                padding-left: 20px;
                padding-right: 20px;
                }}
            """)

        # Set layout for message window
        row_height = pyflame.gui_resize(pyflame.gui_resize(30))

        self.grid.setContentsMargins(pyflame.gui_resize(10),
                                     pyflame.gui_resize(12),
                                     pyflame.gui_resize(10),
                                     pyflame.gui_resize(10),
                                     )

        if type == MessageType.CONFIRM or type == MessageType.WARNING:
            self.cancel_button = PyFlameButton(text='Cancel', connect=self.cancel, width=110)
            self.grid.addWidget(self.main_label, 0, 0)
            self.grid.setRowMinimumHeight(1, row_height)
            self.grid.addWidget(self.message_text_edit, 2, 0, 4, 8)
            self.grid.setRowMinimumHeight(9, row_height)
            self.grid.addWidget(self.cancel_button, 10, 5)
            self.grid.addWidget(self.confirm_button, 10, 6)
            self.grid.setRowMinimumHeight(11, 30)
        else:
            self.grid.addWidget(self.main_label, 0, 0)
            self.grid.setRowMinimumHeight(1, row_height)
            self.grid.addWidget(self.message_text_edit, 2, 0, 4, 8)
            self.grid.setRowMinimumHeight(9, row_height)
            self.grid.addWidget(self.button, 10, 6)
            self.grid.setRowMinimumHeight(11, row_height)
        self.setLayout(self.grid)

        self._print(message, title, time) # Print message to terminal and Flame's console area

        self.exec()

    def __bool__(self):
        return self.confirmed

    def _print(self, message: str, title: str, time: int):
        """
        Print to the terminal and Flame's console area.
        """

        message = message.replace('<br>', ' ')
        message = message.replace('<center>', '')
        message = message.replace('</center>', '')
        message = message.replace('<dd>', '')
        message = message.replace('<b>', '')
        message = message.replace('</b>', '')

        # Print to terminal/shell
        if self.type == MessageType.INFO or self.type == MessageType.OPERATION_COMPLETE or self.type == MessageType.CONFIRM:
            # Print message normally
            print(f'\n--> {title}: {message}\n')
        elif self.type == MessageType.WARNING:
            # Print message text in red
            print(f'\033[91m\n--> {title}: {message}\033[0m\n')
        elif self.type == MessageType.ERROR:
            # Print message text in yellow
            print(f'\033[93m\n--> {title}: {message}\033[0m\n')

        # Print message to the Flame message area - only works in Flame 2023.1 and later
        # Warning and error intentionally swapped to match color of message window
        title = title.upper()

        try:
            if self.type == MessageType.INFO or self.type == MessageType.OPERATION_COMPLETE or self.type == MessageType.CONFIRM:
                flame.messages.show_in_console(f'{title}: {message}', 'info', time)
            elif self.type == MessageType.ERROR:
                flame.messages.show_in_console(f'{title}: {message}', 'warning', time)
            elif self.type == MessageType.WARNING:
                flame.messages.show_in_console(f'{title}: {message}', 'error', time)
        except:
            pass

    def cancel(self):
        self.close()
        self.confirmed = False
        print('--> Cancelled\n')

    def confirm(self):
        self.close()
        self.confirmed = True
        if self.type == MessageType.CONFIRM:
            print('--> Confirmed\n')

    def paintEvent(self, event):
        """
        Draw vertical line on left side of window and a horizontal line across
        the top of the window under the title text.
        """

        # Initialize painter
        painter = QtGui.QPainter(self)

        if self.type == MessageType.CONFIRM:
            line_color = QtGui.QColor(71, 71, 71)  # Grey (fully opaque)
            horizontal_line_color = QtGui.QColor(71, 71, 71, 64)  # Grey with 25% transparency (alpha = 64)
        elif self.type == MessageType.INFO or self.type == MessageType.OPERATION_COMPLETE:
            line_color = QtGui.QColor(0, 110, 176)  # Blue (fully opaque)
            horizontal_line_color = QtGui.QColor(0, 110, 176, 64)  # Blue with 25% transparency (alpha = 64)
        elif self.type == MessageType.ERROR:
            line_color = QtGui.QColor(251, 181, 73)  # Yellow (fully opaque)
            horizontal_line_color = QtGui.QColor(251, 181, 73, 64)  # Yellow with 25% transparency (alpha = 64)
        elif self.type == MessageType.WARNING:
            line_color = QtGui.QColor(200, 29, 29)  # Red (fully opaque)
            horizontal_line_color = QtGui.QColor(200, 29, 29, 64)  # Red with 25% transparency (alpha = 64)

        # Draw 50% transparent horizontal line
        scaled_vertical_pos = pyflame.gui_resize(50)
        painter.setPen(QtGui.QPen(horizontal_line_color, .5, QtCore.Qt.SolidLine))
        painter.drawLine(0, scaled_vertical_pos, self.width, scaled_vertical_pos)

        # Draw fully opaque vertical line on left side
        scaled_bar_width = pyflame.gui_resize(4)
        painter.setPen(QtGui.QPen(line_color, scaled_bar_width, QtCore.Qt.SolidLine))
        painter.drawLine(0, 0, 0, self.height)

    def mousePressEvent(self, event):
        self.oldPosition = event.globalPos()

    def mouseMoveEvent(self, event):
        try:
            delta = QtCore.QPoint(event.globalPos() - self.oldPosition)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPosition = event.globalPos()
        except:
            pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.confirm()

class PyFlamePasswordWindow(QtWidgets.QDialog):
    """
    Custom Qt Flame Password Window

    Parameters:
        message (str): Window message.
        title (str, optional): Text shown in top left of window. If set to None, either Enter Password or Enter Username and Password will be used.
            Default: None
        user_name_prompt (bool): Set to True for username and password prompt.
            Default: False
        font (str): Window font.
            Default: PYFLAME_FONT
        font_size (int): Window font size.
            Default: PYFLAME_FONT_SIZE
        parent (QtWidget, optional): Parent window.
            Default: None

    Returns:
        password(): Returns password as a string.
        username_password(): Returns a tuple of username and password, both as strings.

    Examples:
        for password prompt:
            password_window = PyFlamePasswordWindow(message=f'System password needed to install {SCRIPT_NAME}.')
            password = password_window.password

        for username and password prompt:
            password_window = PyFlamePasswordWindow(message='Enter username and password.', user_name_prompt=True)
            username, password = password_window.username_password()
    """

    def __init__(self: 'PyFlamePasswordWindow',
                 message: str,
                 title: Optional[str]=None,
                 user_name_prompt: bool=False,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 parent: Optional[QtWidgets.QWidget]=None
                 ):
        super().__init__()

        # Validate argument types
        if not isinstance(message, str):
            raise TypeError('PyFlamePasswordWindow: message must be a string')
        elif title is not None and not isinstance(title, str):
            raise TypeError('PyFlamePasswordWindow: title must be a string or None')
        elif not isinstance(user_name_prompt, bool):
            raise TypeError('PyFlamePasswordWindow: user_name_prompt must be a boolean')
        elif not isinstance(font, str):
            raise TypeError('PyFlamePasswordWindow: font must be a string')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise TypeError('PyFlamePasswordWindow: font_size must be a positive integer')

        # Set window title if set to None
        if not title and user_name_prompt:
            title = 'Enter Username and Password'
        elif not title and not user_name_prompt:
            title = 'Enter Password'
        else:
            title = title

        self.user_name_prompt = user_name_prompt
        self.username_value = ''
        self.password_value = ''

        message_font = font

        # Set font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Set Window size for screen
        width = 500
        height = 350
        self.width = pyflame.gui_resize(width)
        self.height = pyflame.gui_resize(height)
        height_alt = pyflame.gui_resize(300)

        # Build password window

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        if user_name_prompt:
            self.setMinimumSize(QtCore.QSize(self.width, self.height))
            self.setMaximumSize(QtCore.QSize(self.width, self.height))
        else:
            self.setMinimumSize(QtCore.QSize(self.width, height_alt))
            self.setMaximumSize(QtCore.QSize(self.width, height_alt))
        self.setStyleSheet("""
            background-color: rgb(36, 36, 36);
        """)

        resolution = main_window_res.screenGeometry()
        self.move((resolution.width() / 2) - (self.frameSize().width() / 2),
                  (resolution.height() / 2) - (self.frameSize().height() / 2))

        self.setParent(parent)

        self.main_label = PyFlameLabel(
            text=title,
            width=width,
            font_size=24,
            )

        self.message_label = QtWidgets.QTextEdit(message)
        self.message_label.setMinimumWidth(self.width)
        self.message_label.setDisabled(True)
        self.message_label.setFont(font)
        self.message_label.setStyleSheet(f"""
            QTextEdit{{
                color: rgb(190, 190, 190);
                background-color: rgb(36, 36, 36);
                selection-color: rgb(190, 190, 190);
                selection-background-color: rgb(36, 36, 36);
                border: none;
                padding-left: 20px;
                padding-right: 20px;
                }}
            """)

        self.password_label = PyFlameLabel(text='      Password', width=80)

        if self.user_name_prompt:
            self.password_entry = PyFlameLineEdit(text='', width=350, max_width=350)
        else:
            self.password_entry = PyFlameLineEdit(text='', width=350, max_width=350)
        self.password_entry.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_entry.returnPressed.connect(self._set_password)

        if user_name_prompt:
            self.username_label = PyFlameLabel(text='      Username', width=80)
            self.username_entry = PyFlameLineEdit(text='', width=350)
            self.username_entry.returnPressed.connect(self._set_username_password)
            self.confirm_button = PyFlameButton(text='Confirm', connect=self._set_username_password, width=110, color=Color.BLUE)
        else:
            self.confirm_button = PyFlameButton(text='Confirm', connect=self._set_password, width=110, color=Color.BLUE)

        self.cancel_button = PyFlameButton(text='Cancel', connect=self._cancel, width=110)

        # UI Widget Layout
        self.grid = QtWidgets.QGridLayout()

        self.grid.setContentsMargins(pyflame.gui_resize(10),
                                     pyflame.gui_resize(12),
                                     pyflame.gui_resize(10),
                                     pyflame.gui_resize(10),
                                     )

        self.grid.addWidget(self.main_label, 0, 0)
        self.grid.setRowMinimumHeight(1, 30)
        self.grid.addWidget(self.message_label, 2, 0, 1, 6)
        self.grid.setRowMinimumHeight(3, 30)

        if user_name_prompt:
            self.grid.addWidget(self.username_label, 4, 0)
            self.grid.addWidget(self.username_entry, 4, 1)

        self.grid.addWidget(self.password_label, 5, 0)
        self.grid.addWidget(self.password_entry, 5, 1)
        self.grid.setRowMinimumHeight(9, 60)
        self.grid.addWidget(self.cancel_button, 10, 5)
        self.grid.addWidget(self.confirm_button, 10, 6)
        self.grid.setRowMinimumHeight(11, 30)

        message = message.replace('<br>', '')
        message = message.replace('<center>', '')
        message = message.replace('<dd>', '')

        self.password_value = ''
        self.username_value = ''

        print(f'\n--> {title}: {message}\n')

        self.setLayout(self.grid)
        self.show()

        # Set entry focus
        if user_name_prompt:
            self.username_entry.setFocus()
        else:
            self.password_entry.setFocus()

        self.exec_()

    def _cancel(self):
        self.close()
        print('--> Cancelled.\n')
        return False

    def _set_username_password(self):

        if self.password_entry.text() and self.username_entry.text():
            self.close()

            self.username_value = self.username_entry.text()
            self.password_value = self.password_entry.text()
            return

    def _set_password(self):
        import subprocess

        password = self.password_entry.text()

        if password:
            command = ['sudo', '-S', 'echo', 'Testing sudo password']
            try:
                # Run the command with sudo and pass the password through stdin
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                output, error = process.communicate(input=password+'\n')

                if process.returncode == 0:
                    print('Sudo password is correct.\n')
                    self.password_value = self.password_entry.text()
                    self.close()
                    return
                else:
                    print('Sudo password is incorrect.')
                    self.message_label.setText('Password incorrect, try again.')
            except Exception as e:
                print('Error occurred while testing sudo password:', str(e))

    def password(self) -> Optional[Union[str, bool]]:
        """
        Returns password as string or None if no password was entered.
        """

        if self.password_value:
            return self.password_value
        return None

    def username_password(self):
        """
        Returns username and password as tuple or None if no username or password was entered.
        """

        if self.username_value and self.password_value:
            return self.username_value, self.password_value
        return None, None

    def paintEvent(self, event):
        """
        Draw vertical red line on left side of window and a horizontal line across
        the top of the window under the title text.
        """

        # Initialize painter
        painter = QtGui.QPainter(self)

        # Draw 50% transparent horizontal line
        scaled_vertical_pos = pyflame.gui_resize(50)
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 29, 29, 64), .5, QtCore.Qt.SolidLine))
        painter.drawLine(0, scaled_vertical_pos, self.width, scaled_vertical_pos)

        # Draw fully opaque vertical line on left side
        scaled_line_width = pyflame.gui_resize(4)
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 29, 29), scaled_line_width, QtCore.Qt.SolidLine))
        painter.drawLine(0, 0, 0, self.height)

    def mousePressEvent(self, event):
        self.oldPosition = event.globalPos()

    def mouseMoveEvent(self, event):

        try:
            delta = QtCore.QPoint(event.globalPos() - self.oldPosition)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPosition = event.globalPos()
        except:
            pass

class PyFlameProgressWindow(QtWidgets.QDialog):
    """
    Custom Qt Flame Progress Window

    Parameters:
        num_to_do (int): total number of operations to do.
        title (str, optional): text shown in top left of window ie. Rendering...
            Default: None
        text (str, optional): message to show in window.
            Default: None
        line_color (WindowBarColor): Color of bar on left side of window.
            -LineColor.GRAY: For gray line.
            -LineColor.BLUE: For blue line.
            -LineColor.RED: For red line.
            -LineColor.GREEN: For green line.
            -LineColor.YELLOW: For yellow line.
            -LineColor.TEAL: For teal line.
            Default: LineColor.BLUE
        enable_done_button (bool): enable done button,
            Default: False
        font (str): Font to be used in window.
            Default: PYFLAME_FONT
        font_size (int): Size of font.
            Default: PYFLAME_FONT_SIZE

    Public Methods:
        set_progress_value(int): Set progress bar value.
        enable_done_button(bool): Enable or disable done button.
        set_text(str): Set text in window.

    Examples:
        To create progress bar window:
            self.progress_window = PyFlameProgressWindow(
                num_to_do=10,
                title='Rendering...',
                text='Rendering: Batch 1 of 5',
                enable_done_button=True,
                )

        To update progress bar progress value:
            self.progress_window.set_progress_value(5)

        To update text in window:
            self.progress_window.set_text('Rendering: Batch 2 of 5')

        To enable or disable done button - True or False:
            self.progress_window.enable_done_button(True)
    """

    def __init__(self: 'PyFlameProgressWindow',
                 num_to_do: int,
                 title: Optional[str]=None,
                 text: Optional[str]=None,
                 line_color: LineColor=LineColor.BLUE,
                 enable_done_button: bool=False,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 parent=None
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(num_to_do, int):
            raise TypeError('PyFlameProgressWindow: num_to_do must be an integer')
        if title is not None and not isinstance(title, str):
            raise TypeError('PyFlameProgressWindow: title must be a string or None')
        elif text is not None and not isinstance(text, str):
            raise TypeError('PyFlameProgressWindow: text must be a string or None')
        elif not isinstance(line_color, LineColor):
            raise ValueError('PyFlameProgressWindow: color must be an instance of LineColor Enum. '
                            'Options are: LineColor.GRAY, LineColor.BLUE, LineColor.RED, '
                            'LineColor.GREEN, LineColor.YELLOW, LineColor.TEAL.')
        elif not isinstance(enable_done_button, bool):
            raise TypeError('PyFlameProgressWindow: enable_done_button must be True or False')
        elif not isinstance(font, str):
            raise TypeError('PyFlameProgressWindow: font must be a string')
        elif not isinstance(font_size, int):
            raise TypeError('PyFlameProgressWindow: font_size must be an integer')

        self.line_color = line_color
        self.num_to_do = num_to_do

        if not title:
            title = 'Task Progress'

        # Set font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Set Window size for screen
        width = 500
        height = 330
        self.window_width = pyflame.gui_resize(width)
        self.window_height = pyflame.gui_resize(height)

        # Build window
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setMinimumSize(QtCore.QSize(self.window_width, self.window_height))
        self.setMaximumSize(QtCore.QSize(self.window_width, self.window_height))
        self.setStyleSheet("""
            background-color: rgb(36, 36, 36)
            """)

        resolution = main_window_res.screenGeometry()
        self.move((resolution.width() / 2) - (self.frameSize().width() / 2),
                  (resolution.height() / 2) - (self.frameSize().height() / 2))

        self.setParent(parent)

        self.grid = QtWidgets.QGridLayout()

        self.main_label = PyFlameLabel(
            text=title,
            width=width,
            font_size=24,
            )

        self.message_text_edit = QtWidgets.QTextEdit('')
        self.message_text_edit.setDisabled(True)
        self.message_text_edit.setFont(font)
        self.message_text_edit.setStyleSheet(f"""
            QTextEdit{{
                color: rgb(154, 154, 154);
                background-color: rgb(36, 36, 36);
                selection-color: rgb(190, 190, 190);
                selection-background-color: rgb(36, 36, 36);
                border: none;
                padding-left: 20px;
                padding-right: 20px;
                }}
            """)
        self.message_text_edit.setText(text)

        # Progress bar
        bar_max_height = pyflame.gui_resize(5)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximum(num_to_do)
        self.progress_bar.setMaximumHeight(bar_max_height)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar{
                color: rgb(154, 154, 154);
                background-color: rgb(45, 45, 45);
                border: none;
                }
            QProgressBar:chunk{
                background-color: rgb(0, 110, 176)
                }
            """)

        self.done_button = PyFlameButton(
            text='Done',
            connect=self.close,
            width=110,
            color=Color.BLUE,
            )
        self.done_button.setEnabled(enable_done_button)

        # Layout
        row_height = pyflame.gui_resize(30)

        self.grid.setContentsMargins(pyflame.gui_resize(10),
                                     pyflame.gui_resize(12),
                                     pyflame.gui_resize(10),
                                     pyflame.gui_resize(10),
                                     )

        self.grid.addWidget(self.main_label, 0, 0)
        self.grid.setRowMinimumHeight(1, row_height)
        self.grid.addWidget(self.message_text_edit, 2, 0, 1, 4)
        self.grid.addWidget(self.progress_bar, 8, 0, 1, 7)
        self.grid.setRowMinimumHeight(9, row_height)
        self.grid.addWidget(self.done_button, 10, 6)
        self.grid.setRowMinimumHeight(11, row_height)

        print(f'\n--> {title}\n')

        self.setLayout(self.grid)
        self.show()

    def set_text(self, text):
        """
        Use to set the text of the message text edit widget to the specified text.

        Parameters:
            text (str): The text to set in the message text edit widget.
        """

        self.message_text_edit.setText(text)

    def set_progress_value(self, value):
        """
        Use to set the value of the progress bar.

        Parameters:
            value (int): The value to set the progress bar to.
        """

        self.progress_bar.setValue(value)
        QtWidgets.QApplication.processEvents()

    def enable_done_button(self, value):
        """
        Use to enable or disable the done button.

        Parameters:
            value (bool): True to enable done button, False to disable done button.
        """

        if value:
            self.done_button.setEnabled(True)
        else:
            self.done_button.setEnabled(False)

    def showEvent(self, event):
        """
        If the window has a parent, center the window on the screen.
        """

        parent = self.parent()
        if parent:
            # Center the window on the screen
            screen_geometry =  QtWidgets.QDesktopWidget().screenGeometry()
            x = (screen_geometry.width() - self.width()) / 2
            y = (screen_geometry.height() - self.height()) / 2
            x = x/2
            y = y/2

            self.move(x, y)

            super().showEvent(event)

    def paintEvent(self, event):

        # Set line color
        if self.line_color == LineColor.GRAY:
            line_color = QtGui.QColor(71, 71, 71)  # Grey (fully opaque)
            horizontal_line_color = QtGui.QColor(71, 71, 71, 64)  # Grey with 25% transparency (alpha = 64)
        elif self.line_color == LineColor.BLUE:
            line_color = QtGui.QColor(0, 110, 176)  # Blue (fully opaque)
            horizontal_line_color = QtGui.QColor(0, 110, 176, 64)  # Blue with 25% transparency (alpha = 64)
        elif self.line_color == LineColor.RED:
            line_color = QtGui.QColor(200, 29, 29)  # Red (fully opaque)
            horizontal_line_color = QtGui.QColor(200, 29, 29, 64)  # Red with 25% transparency (alpha = 64)
        elif self.line_color == LineColor.GREEN:
            line_color = QtGui.QColor(0, 180, 13)
            horizontal_line_color = QtGui.QColor(0, 180, 13, 64) # Green with 25% transparency (alpha = 64)
        elif self.line_color == LineColor.YELLOW:
            line_color = QtGui.QColor(251, 181, 73)  # Yellow (fully opaque)
            horizontal_line_color = QtGui.QColor(251, 181, 73, 64)  # Yellow with 25% transparency (alpha = 64)
        elif self.line_color == LineColor.TEAL:
            line_color = QtGui.QColor(14, 110, 106)
            horizontal_line_color = QtGui.QColor(14, 110, 106, 64) # Teal with 25% transparency (alpha = 64)

        # Initialize painter
        painter = QtGui.QPainter(self)

        # Draw 50% transparent horizontal line below text
        scaled_line_height = pyflame.gui_resize(50)
        painter.setPen(QtGui.QPen(horizontal_line_color, .5, QtCore.Qt.SolidLine))
        painter.drawLine(0, scaled_line_height, self.window_width, scaled_line_height)

        # Draw fully opaque vertical line on left side
        scaled_line_width = pyflame.gui_resize(4)
        painter.setPen(QtGui.QPen(line_color, scaled_line_width, QtCore.Qt.SolidLine))
        painter.drawLine(0, 0, 0, self.window_height)

    def mousePressEvent(self, event):

        self.oldPosition = event.globalPos()

    def mouseMoveEvent(self, event):

        try:
            delta = QtCore.QPoint(event.globalPos() - self.oldPosition)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPosition = event.globalPos()
        except:
            pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.close()

class PyFlameQDialog(QtWidgets.QDialog):
    """
    Custom Qt Flame QDialog Widget

    Parameters:
        width (int): Width of window.
        height (int): Height of window.
        title (str): Text displayed in top left corner of window.
            Default: Python Script
        line_color (LineColor): Color of bar on left side of window.
            -LineColor.GRAY: For gray line.
            -LineColor.BLUE: For blue line.
            -LineColor.RED: For red line.
            -LineColor.GREEN: For green line.
            -LineColor.YELLOW: For yellow line.
            -LineColor.TEAL: For teal line.
            Default: LineColor.BLUE
        return_pressed (callable, optional): Function to be called when return key is pressed.
            Default: None
        font (str): Font to be used in window.
            Default: PYFLAME_FONT
        font_size (int): Size of font.
            Default: PYFLAME_FONT_SIZE

    Public Methods:
        add_layout(layout): Add layout to window.

    Examples:
        To create a window:
            window = PyFlameQDialog(
                width=400,
                height=200,
                title=f'{SCRIPT_NAME} <small>{SCRIPT_VERSION},
                return_pressed=confirm,
                )
        To add a qlayout to the window:
            window.add_layout(layout)
    """

    def __init__(self: 'PyFlameQDialog',
                 width: int,
                 height: int,
                 title: str='Python Script',
                 line_color: LineColor=LineColor.BLUE,
                 return_pressed: Optional[Callable]=None,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(width, int) or width <= 0:
            raise ValueError('PyFlameWindow: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise ValueError('PyFlameWindow: height must be a positive integer.')
        elif not isinstance(title, str):
            raise TypeError('PyFlameWindow: title must be a string.')
        elif not isinstance(line_color, LineColor):
            raise ValueError('PyFlameWindow: color must be an instance of LineColor Enum. '
                             'Options are: LineColor.GRAY, LineColor.BLUE, LineColor.RED, '
                             'LineColor.GREEN, LineColor.YELLOW, LineColor.TEAL.')
        elif return_pressed is not None and not callable(return_pressed):
            raise TypeError('PyFlameWindow: return_pressed must be a callable function or None.')
        elif not isinstance(font, str):
            raise TypeError('PyFlameWindow: font must be a string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise ValueError('PyFlameWindow: font_size must be a positive integer.')

        self.line_color = line_color

        self.font_size = pyflame.font_resize(font_size)

        # Scale window size for screen resolution
        self.width = pyflame.gui_resize(width)
        self.height = pyflame.gui_resize(height)
        self.line = pyflame.gui_resize(45)

        # Build window
        self.setFixedSize(self.width, self.height)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Set return pressed function
        self.return_pressed = return_pressed

        # Center window in linux
        resolution = main_window_res.screenGeometry()
        self.move((resolution.width() / 2) - (self.frameSize().width() / 2),
                  (resolution.height() / 2) - (self.frameSize().height() / 2))

        # Set font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Window title label
        title_label = PyFlameLabel(
            text='<span style="white-space: pre;">  ' + title, # Add space to title using CSS code. This pushes the title to the right one space.
            style=Style.UNDERLINE,
            align=Align.LEFT,
            width=width,
            underline_color=(0, 43, 66, 0.5),
            height=40,
            font_size=24,
            )

        # Window layout
        # -------------

        # Main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(title_label, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        main_layout.addSpacing(pyflame.gui_resize(7))

        # Center layout - script layout added to this layout
        self.center_layout = QtWidgets.QGridLayout()

        # Create widget to hold the center layout
        center_widget = QtWidgets.QWidget()
        center_widget.setLayout(self.center_layout)

        # Add the center layout to the main layout
        main_layout.addWidget(center_widget, alignment=QtCore.Qt.AlignCenter)

        # Add stretchable space below the center layout
        main_layout.addStretch()

        # Set main window layout
        self.setLayout(main_layout)

        # After setting the main window layout:
        self.setLayout(main_layout)





        self._set_stylesheet(font)

    def add_layout(self, layout):
        """
        Add layout from script to the main window.
        """

        self.center_layout.addLayout(layout, 1, 0)

    def _set_stylesheet(self, font):

        # Window stylesheet - Double brackets required below due to f-string
        self.setStyleSheet(f"""
            QWidget{{
                background-color: rgb(36, 36, 36);
                }}
            QTabWidget{{
                background-color: rgb(36, 36, 36);
                border: none;
                }}
            QTabWidget::tab-bar{{
                alignment: center;
                }}
            QTabBar::tab{{
                color: rgb(190, 190, 190);
                background-color: rgb(36, 36, 36);
                min-width: 20ex;
                padding: 5px;
                }}
            QTabBar::tab:selected{{
                color: rgb(186, 186, 186);
                background-color: rgb(31, 31, 31);
                border: 1px solid rgb(31, 31, 31);
                border-bottom: 1px solid rgb(0, 110, 176);
                }}
            QTabBar::tab:!selected{{
                color: rgb(186, 186, 186);
                background-color: rgb(36, 36, 36);
                border: none;
                }}
            QTabWidget::pane{{
                border-top: 1px solid rgb(49, 49, 49);
                }}
            """)

    def paintEvent(self, event):

        painter = QtGui.QPainter(self)

        if self.line_color == LineColor.GRAY:
            color = QtGui.QColor(71, 71, 71)
        elif self.line_color == LineColor.BLUE:
            color = QtGui.QColor(0, 110, 176)
        elif self.line_color == LineColor.RED:
            color = QtGui.QColor(200, 29, 29)
        elif self.line_color == LineColor.GREEN:
            color = QtGui.QColor(0, 180, 13)
        elif self.line_color == LineColor.YELLOW:
            color = QtGui.QColor(251, 181, 73)
        elif self.line_color == LineColor.TEAL:
            color = QtGui.QColor(14, 110, 106)

        # Draw bar on left side of window
        painter.setPen(QtGui.QPen(color, pyflame.gui_resize(4), QtCore.Qt.SolidLine))
        painter.drawLine(0, 0, 0, self.height)

    def mousePressEvent(self, event):

        self.oldPosition = event.globalPos()

    def mouseMoveEvent(self, event):

        try:
            delta = QtCore.QPoint(event.globalPos() - self.oldPosition)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPosition = event.globalPos()
        except:
            pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return and self.return_pressed is not None:
            self.return_pressed()

    def resizeOverlay(self, event): #added for overlay
        self.overlay.resize(event.size())

class PyFlameWindow(QtWidgets.QWidget):
    """
    Custom Qt Flame Window Widget

    Parameters:
        width (int): Width of window.
        height (int): Height of window.
        title (str): Text displayed in top left corner of window.
            Default: Python Script
        line_color (LineColor): Color of bar on left side of window.
            -LineColor.GRAY: For gray line.
            -LineColor.BLUE: For blue line.
            -LineColor.RED: For red line.
            -LineColor.GREEN: For green line.
            -LineColor.YELLOW: For yellow line.
            -LineColor.TEAL: For teal line.
            Default: LineColor.BLUE
        return_pressed (callable, optional): Function to be called when return key is pressed.
            Default: None
        font (str): Font to be used in window.
            Default: PYFLAME_FONT
        font_size (int): Size of font.
            Default: PYFLAME_FONT_SIZE

    Public Methods:
        add_layout(layout): Add layout to window.

    Example:
        To create a window:
            window = PyFlameWindow(
                width=400,
                height=200,
                title=f'{SCRIPT_NAME} <small>{SCRIPT_VERSION},
                return_pressed=confirm,
                )
        To add a qlayout to the window:
            window.add_layout(layout)
    """

    def __init__(self: 'PyFlameWindow',
                 width: int,
                 height: int,
                 title: str='Python Script',
                 line_color: LineColor=LineColor.BLUE,
                 return_pressed: Optional[Callable]=None,
                 font: str=PYFLAME_FONT,
                 font_size: int=PYFLAME_FONT_SIZE,
                 ) -> None:
        super().__init__()

        # Validate argument types
        if not isinstance(width, int) or width <= 0:
            raise ValueError('PyFlameWindow: width must be a positive integer.')
        elif not isinstance(height, int) or height <= 0:
            raise ValueError('PyFlameWindow: height must be a positive integer.')
        elif not isinstance(title, str):
            raise TypeError('PyFlameWindow: title must be a string.')
        elif not isinstance(line_color, LineColor):
            raise ValueError('PyFlameWindow: color must be an instance of LineColor Enum. '
                             'Options are: LineColor.GRAY, LineColor.BLUE, LineColor.RED, '
                             'LineColor.GREEN, LineColor.YELLOW, LineColor.TEAL.')
        elif return_pressed is not None and not callable(return_pressed):
            raise TypeError('PyFlameWindow: return_pressed must be a callable function or None.')
        elif not isinstance(font, str):
            raise TypeError('PyFlameWindow: font must be a string.')
        elif not isinstance(font_size, int) or font_size <= 0:
            raise ValueError('PyFlameWindow: font_size must be a positive integer.')

        self.line_color = line_color

        self.font_size = pyflame.font_resize(font_size)

        # Scale window size for screen resolution
        self.width = pyflame.gui_resize(width)
        self.height = pyflame.gui_resize(height)

        # Build window
        self.setFixedSize(self.width, self.height)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Set return pressed function
        self.return_pressed = return_pressed

        # Center window in linux
        resolution = main_window_res.screenGeometry()
        self.move((resolution.width() / 2) - (self.frameSize().width() / 2),
                  (resolution.height() / 2) - (self.frameSize().height() / 2))

        # Set font
        font = QtGui.QFont(font)
        font.setPointSize(pyflame.font_resize(font_size))
        self.setFont(font)

        # Window title label
        title_label = PyFlameLabel(
            text='<span style="white-space: pre;">  ' + title, # Add space to title using CSS code. This pushes the title to the right one space.
            style=Style.UNDERLINE,
            align=Align.LEFT,
            width=width,
            underline_color=(0, 43, 66, 0.5),
            height=40,
            font_size=24,
            )

        # Window layout
        # -------------

        # Title line HBox
        title_text_hbox = QtWidgets.QHBoxLayout()
        title_text_hbox.addSpacing(pyflame.gui_resize(3))
        title_text_hbox.addWidget(title_label)

        # Main layout
        main_vbox = QtWidgets.QVBoxLayout()
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.addLayout(title_text_hbox)
        main_vbox.addSpacing(pyflame.gui_resize(20))
        # Center layout - where main UI is added
        self.center_layout = QtWidgets.QGridLayout()

        # Create widget to hold the center layout
        center_widget = QtWidgets.QWidget()
        center_widget.setLayout(self.center_layout)

        # Add the center layout to the main layout
        main_vbox.addWidget(center_widget, alignment=QtCore.Qt.AlignCenter)
        main_vbox.addStretch()
        main_vbox.addSpacing(pyflame.gui_resize(20))

        self.setLayout(main_vbox)

        self._set_stylesheet(font)

    def add_layout(self, layout):
        """
        Add layout from script to the main window.
        """

        self.center_layout.addLayout(layout, 0, 0)

    def _set_stylesheet(self, font):

        # Window stylesheet
        self.setStyleSheet(f"""
            QWidget{{
                background-color: rgb(36, 36, 36);
                }}
            QTabWidget{{
                background-color: rgb(36, 36, 36);
                border: none;
                }}
            QTabWidget::tab-bar{{
                alignment: center;
                }}
            QTabBar::tab{{
                color: rgb(190, 190, 190);
                background-color: rgb(36, 36, 36);
                min-width: 20ex;
                height: {pyflame.gui_resize(20)}px;
                padding: 5px;
                }}
            QTabBar::tab:selected{{
                color: rgb(186, 186, 186);
                background-color: rgb(31, 31, 31);
                border: 1px solid rgb(31, 31, 31);
                border-bottom: 1px solid rgb(0, 110, 176);
                }}
            QTabBar::tab:!selected{{
                color: rgb(186, 186, 186);
                background-color: rgb(36, 36, 36);
                border: none;
                }}
            QTabWidget::pane{{
                border-top: 1px solid rgb(49, 49, 49);
                }}
            """)

    def paintEvent(self, event):

        painter = QtGui.QPainter(self)

        if self.line_color == LineColor.GRAY:
            color = QtGui.QColor(71, 71, 71)
        elif self.line_color == LineColor.BLUE:
            color = QtGui.QColor(0, 110, 176)
        elif self.line_color == LineColor.RED:
            color = QtGui.QColor(200, 29, 29)
        elif self.line_color == LineColor.GREEN:
            color = QtGui.QColor(0, 180, 13)
        elif self.line_color == LineColor.YELLOW:
            color = QtGui.QColor(251, 181, 73)
        elif self.line_color == LineColor.TEAL:
            color = QtGui.QColor(14, 110, 106)

        # Draw bar on left side of window
        painter.setPen(QtGui.QPen(color, pyflame.gui_resize(4), QtCore.Qt.SolidLine))
        painter.drawLine(0, 0, 0, 5000)

    def mousePressEvent(self, event):

        self.oldPosition = event.globalPos()

    def mouseMoveEvent(self, event):

        try:
            delta = QtCore.QPoint(event.globalPos() - self.oldPosition)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPosition = event.globalPos()
        except:
            pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return and self.return_pressed is not None:
            self.return_pressed()
