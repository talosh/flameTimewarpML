from __future__ import print_function

import json

# Disable this flag to allow for backward compatibility with older scripts
# pylint: disable=redefined-builtin

# Python 2/3 support
def itervalues(d):
    try:  # https://www.python.org/dev/peps/pep-0469/
        # Python 2
        return d.itervalues()  # pylint: disable=dict-iter-method
    except AttributeError:
        # Python 3
        return iter(d.values())


#############################################
# Utility functions to configure the pybox
#############################################

## Create a new floating point numeric UI element.
#  A floating point numeric is displayed as a field in the Pybox UI.
#  Once created, it must be added to either `global_elements` or
#  `render_elements`.
#   @sa BaseClass::add_global_elements
#   @sa BaseClass::add_render_elements
#
#  @param name          Name displayed in the UI.
#  @param value         Current value of the field.
#  @param default       Default value set on load.
#  @param min           Minimum value allowed.
#  @param max           Maximum value allowed.
#  @param inc           The stepping used when dragging in a float vector
#                       field.
#  @param row           Y position within a `page`. Range [0,4]
#  @param col           X position within a `page`. Range [0,3]
#  @param page          The page (or tab) where to insert the element. Range [0,5]
#  @param channel_name  Name of the Animation Channel for this UI element.
#                       Sanitized to remove any white space.
#                       If none is specified, `channel_name = name + "_chn"`
#  @param tooltip       Tooltip displayed for this element.
#  @return              A float numeric UI element as a dictionary.
def create_float_numeric(
    name,
    value=0.0,
    default=0.0,
    min=0.0,
    max=100.0,
    inc=1.0,
    row=0,
    col=0,
    page=0,
    channel_name=None,
    tooltip="",
):
    """Create a new floating point numeric"""

    if channel_name is None:
        channel_name = name + "_chn"

    # Flame remove spaces for channel names
    channel_name = channel_name.replace(" ", "_")

    return {
        "row": int(row),
        "col": int(col),
        "page": int(page),
        "min": float(min),
        "max": float(max),
        "inc": float(inc),
        "tooltip": str(tooltip),
        "name": str(name),
        "channel_name": str(channel_name),
        "default": float(default),
        "type": "Float",
        "value": float(value),
    }


def create_numeric_info(
    name, default=0.0, min=0.0, max=100.0, inc=1.0, display_name=""
):
    return {
        "min": float(min),
        "max": float(max),
        "default": float(default),
        "inc": float(inc),
        "channel_name": str(name),
        "display_name": str(display_name),
    }


## Create a new vector numeric UI element.
#  A 2D vector numeric displays 2 fields, a 3D vector numeric displays 3 fields.
#  `size` defines the dimension of the vector.
#  Once created, it must be added to either `global_elements` or
#  `render_elements`.
#   @sa BaseClass::add_global_elements
#   @sa BaseClass::add_render_elements
#
#  @param name          Name displayed in the UI.
#  @param size          Length of the vector (2 or 3)
#  @param values        Array of values of lenght = size.
#  @param numeric_info  `None` or array of {min, max, default, inc,
#                       channel_name, display_name}. Dimmension of the array is
#                       defined by `size` parameter. If an array is used, there
#                       is no need to define the parameters listed
#                       below: min, max, inc, default, channel_name. If `None`,
#                       these parameters apply to all fields.
#  @param default       Default value set on load.
#  @param min           Minimum value allowed.
#  @param max           Maximum value allowed.
#  @param inc           The stepping used when dragging in a numeric vector
#                       field.
#  @param row           Y position within a `page`. Range [0,4]
#  @param col           X position within a `page`. Range [0,3]
#  @param page          The page (or tab) where to insert the element. Range [0,5]
#  @param channel_name  Name of the Animation Channel for this UI element.
#                       Sanitized to remove any white space.
#                       If none is specified, `channel_name = name + "_chn"`
#  @param tooltip       Tooltip displayed for this element.
#  @return              A vector numeric UI element as a dictionary.
def create_vector_numeric(
    name,
    size=3,
    values=None,
    numeric_info=None,
    default=0.0,
    min=0.0,
    max=100.0,
    inc=1.0,
    row=0,
    col=0,
    page=0,
    channel_name=None,
    tooltip="",
):
    """Create a new vector numeric"""
    if size > 3:
        raise Exception("Vector size not supported")

    if channel_name is None:
        channel_name = name + "_chn"

    if values is None:
        values = [float(default)] * size

    if numeric_info is None:
        numeric_info = []
        chn_names = {
            0: "x",
            1: "y",
            2: "z",
        }
        for i in range(0, size):
            numeric_info.append(
                create_numeric_info(
                    name + "_" + chn_names[i],
                    min=min,
                    max=max,
                    default=default,
                    inc=inc,
                )
            )

    return {
        "row": int(row),
        "col": int(col),
        "page": int(page),
        "name": str(name),
        "type": "FloatVector",
        "tooltip": str(tooltip),
        "channel_name": str(channel_name),
        "values": values,
        "info": numeric_info,
    }


## Create a new pop-up UI element.
#  Once created, it must be added to either `global_elements` or
#  `render_elements`.
#   @sa BaseClass::add_global_elements
#   @sa BaseClass::add_render_elements
#
#  @param name      Name displayed in the UI.
#  @param items     Array of options for the pop-up (as string).
#  @param value     Index of the value.
#  @param default   Option displayed in the popup menu at initialization.
#                   Member of `values`.
#  @param row       Y position within a `page`. Range [0,4]
#  @param col       X position within a `page`. Range [0,3]
#  @param page      The page (or tab) where to insert the element. Range [0,5]
#  @param tooltip   Tooltip displayed for this element.
#  @return          A pop-up UI element as a dictionary.
def create_popup(name, items, value=0, default=0, row=0, col=0, page=0, tooltip=""):
    """Create a new popup parameter"""

    if not isinstance(items, list) or len(items) < 1:
        raise Exception("Items should be an array and contain at least 1 element")

    for item in items:
        item = str(item)

    if default >= len(items):
        raise Exception("Default should be a valid item")

    if value >= len(items):
        raise Exception("Value should be a valid item")

    return {
        "row": row,
        "col": col,
        "page": page,
        "name": name,
        "type": "Pup",
        "tooltip": tooltip,
        "items": items,
        "default": default,
        "value": value,
    }


## Create a new color pot UI element.
#  Once created, it must be added to either `global_elements` or
#  `render_elements`.
#   @sa BaseClass::add_global_elements
#   @sa BaseClass::add_render_elements
#
#  @param name          Name displayed in the UI.
#  @param default       The default color in the color pot. An array
#                       of 3 float RGB values, in range [0,1].
#  @param values        An array of 3 float RGB values, in range [0,1].
#  @param row           Y position within a `page`. Range [0,4]
#  @param col           X position within a `page`. Range [0,3]
#  @param page          The page (or tab) where to insert the element. Range [0,5]
#  @param channel_name  Name of the Animation Channel for this UI element.
#                       Sanitized to remove any white space.
#                       If none is specified, `channel_name = name + "_chn"`
#  @param tooltip       Tooltip displayed for this element.
#  @return              A color pot UI element as a dictionary.
def create_color(
    name, default=None, values=None, row=0, col=0, page=0, channel_name=0, tooltip=""
):
    """Create a new color pot"""

    if not channel_name:
        channel_name = name + "_chn"

    if default is None:
        default = [0.0, 0.0, 0.0]

    if values is None:
        values = default

    return {
        "row": row,
        "col": col,
        "page": page,
        "name": name,
        "type": "Color",
        "tooltip": tooltip,
        "default": default,
        "channel_name": channel_name,
        "values": values,
    }


## Create a new toggle button element.
#  Once created, it must be added to either `global_elements` or
#  `render_elements`.
#   @sa BaseClass::add_global_elements
#   @sa BaseClass::add_render_elements
#
#
#  @param name          Name displayed in the UI.
#  @param value         True (enabled) or False (disabled)
#  @param default       State of the toggle at initialization. True for activated.
#                       False for deactivated.
#  @param row           Y position within a `page`. Range [0,4]
#  @param col           X position within a `page`. Range [0,3]
#  @param page          The page (or tab) where to insert the element. Range [0,5]
#  @param tooltip       Tooltip displayed for this element.
#  @return              A toggle button UI element as dictionary.
def create_toggle_button(name, value, default=False, row=0, col=0, page=0, tooltip=""):

    return {
        "row": row,
        "col": col,
        "page": page,
        "tooltip": tooltip,
        "name": name,
        "default": default,
        "type": "Toggle",
        "value": value,
    }


## Create a new UI element used to display a Flame file browser.
#  Once created, it must be added to either `global_elements` or
#  `render_elements`.
#   @sa BaseClass::add_global_elements
#   @sa BaseClass::add_render_elements
#
#
#  @param name          Name displayed in the UI.
#  @param value         Path displayed in the browser.
#  @param extension     Filter for files with this extension. Three-letter
#                       extension, without period, such as: `py, nk, jpg`
#  @param home          Defines the location opened when the HOME button is
#                       clicked in the file browser UI.
#  @param row           Y position within a `page`. Range [0,4]
#  @param col           X position within a `page`. Range [0,3]
#  @param page          The page (or tab) where to insert the element. Range [0,5]
#  @param tooltip       Tooltip displayed for this element.
#  @return              A browser UI element as a dictionary.
def create_file_browser(
    name, value, extension, home, row=0, col=0, page=0, tooltip="", isFileSelector=True
):

    return {
        "row": row,
        "col": col,
        "page": page,
        "tooltip": tooltip,
        "name": name,
        "value": value,
        "extension": extension,
        "home": home,
        "type": "Browser",
        "isFileSelector": isFileSelector,
    }


## Create a new UI text field element
#  Once created, it must be added to either `global_elements` or
#  `render_elements`.
#   @sa BaseClass::add_global_elements
#   @sa BaseClass::add_render_elements
#
#
#  @param name          Name displayed in the UI.
#  @param value         A string, the current value of the textfield
#  @param row           Y position within a `page`. Range [0,4]
#  @param col           X position within a `page`. Range [0,3]
#  @param page          The page (or tab) where to insert the element. Range [0,5]
#  @param tooltip       Tooltip displayed for this element.
#  @return              A string field UI element as dictionary.
def create_text_field(name, value="", row=0, col=0, page=0, tooltip="", isField=True):
    return {
        "row": row,
        "col": col,
        "page": page,
        "tooltip": tooltip,
        "name": name,
        "type": "TextField",
        "value": value,
        "isField": isField,
    }


## Create a new page object. Can be provided 0 or more column names.
#  Requires BaseClass::set_ui_pages() to load in the page in the UI.
#
#  @param name Name of the page created. Displayed as the tab's name in the node.
#  @param cols The name of each column in the `page`, as a collection.
#  @return A page as a JSON object.
def create_page(name, *cols):
    """Create a new page provided a name and 0 or more columns name"""
    return {"name": name, "col": [str(i) for i in cols]}


## The class to instance to interact with the Pybox node.
#  Sample implementations are available in /opt/Autodesk/presets/2018.2.pr80/pybox/presets
#  As for basic the structure of a Pybox file, here is an example:
#   @code
#   import os
#   import sys
#   import tempfile
#   import pybox_v1 as pybox
#
#   class sample_class(pybox.BaseClass):
#
#       def initialize(self):
#           # This is where the node is initialized.
#           self.set_img_format("exr")
#
#           self.set_in_socket(0, "Front", "/tmp/in_front")
#           self.set_in_socket(2, "Matte", "/tmp/in_matte")
#
#           self.set_out_socket(0, "Result", "/tmp/out_front")
#           self.set_out_socket(1, "OutMatte", "/tmp/out_front")
#
#           self.set_state_id("setup_ui")
#           self.setup_ui()
#
#       def setup_ui(self):
#           # This is where the UI is created.
#           numeric = pybox.create_float_numeric("Value", 0, 0, 100, col=2)
#           self.add_global_elements(numeric)
#
#           page = pybox.create_page("Main Page")
#           self.set_ui_pages(page)
#
#           self.set_state_id("execute")
#           self.execute()
#
#       def execute(self):
#           pass
#
#       def teardown(self):
#           pass
#
#   def _main(argv):
#
#       # Load the json file, make sure you have read access to it
#       p = sample_class(argv[0])
#
#       # Call the appropriate function
#       p.dispatch()
#
#       # Save file
#       p.write_to_disk(argv[0])
#
#   if __name__ == '__main__':
#       _main(sys.argv[1:])
#   @endcode
#


class BaseClass(object):

    ## Function to override.
    #  First method to call, initialize the Pybox node on load. One is expected
    #  to define input and output sockets used by the Pybox node in this function.
    #  Should concludes with the following to move onto the UI definition phase:
    #   @code
    #   set_state_id("setup_ui")
    #   self.setup_ui()
    #   @endcode
    #
    #  @param self The object pointer
    def initialize(self):
        raise NotImplementedError("initialize function not implemented")

    ## Function to override.
    #  Second method to call, defines the Pybox UI. One is expected
    #  to define the UI controls the end user can interact with. Every UI element
    #  must be assigned to #add_render_elements or #add_global_elements.
    #  Should concludes with the following to move onto the execution phase:
    #   @code
    #   set_state_id("execute")
    #   self.execute()
    #   @endcode
    #
    #  @param self The object pointer
    def setup_ui(self):
        raise NotImplementedError("setup_ui function not implemented")

    ## Function to override.
    #  Third method to call, execution phase. This is where the actual processing
    #  through a third party app happens.
    #  Should concludes with the following to move to the teardown phase:
    #   @code
    #   set_state_id("teardown")
    #   self.teardown()
    #   @endcode
    #
    #  @param self The object pointer
    def execute(self):
        raise NotImplementedError("execute function not implemented")

    ## Function to override.
    #  The teardown function is called when the Pybox is destroyed. It is
    #  recommended to cleanup owned resources here.
    #
    #  @param self The object pointer
    def teardown(self):
        """The teardown function is called when the pybox is destroyed.
        It is recommanded to cleanup owned resources here."""
        raise NotImplementedError("teardown function not implemented")

    #############################################
    # Default pybox methods
    #############################################

    ## The constructor
    #  @param path  File path to the JSON file. Always the the first argument passed
    #  to the python script (always argv[0]).
    def __init__(self, path):
        with open(path) as data_file:
            self.__dict__ = json.load(data_file)

            if not self.state_id:
                raise RuntimeError("state id should not be empty")

            # Get the previous state (and remove it from cur state)
            prev_data = self.__dict__.pop("previous_data", None)

            # Diff the dynamicUI to find changed values
            if prev_data:

                def diff_dict(prev, new):
                    elements = set(prev.keys()).intersection(set(new.keys()))
                    diff = []
                    for el in elements:
                        if prev[el] != new[el]:
                            diff.append(prev[el])
                    return diff

                prev_data = json.loads(prev_data)
                self.ui_changes = diff_dict(
                    prev_data["dynamic_ui"][self.__GLOBAL_ELEMENTS],
                    self.dynamic_ui[self.__GLOBAL_ELEMENTS],
                )
                self.ui_changes.extend(
                    diff_dict(
                        prev_data["dynamic_ui"][self.__RENDER_ELEMENTS],
                        self.dynamic_ui[self.__RENDER_ELEMENTS],
                    )
                )

    ## Execute the methods specified by the state ID.
    #  Automatically calls the function associated with the state of the Pybox.
    # @sa //initialize and al.
    #
    #  @param self The object pointer
    def dispatch(self):
        """Executes the methods specified by the state id"""
        getattr(self, self.state_id)()

    ## Set the state of the Pybox in the JSON payload when the Pybox is ready to
    #  move to that next state. When a Pybox is loaded, the state is implicitly
    #  set to `initialize`.
    #
    #  @param self The object pointer
    #  @param state_id One of the following:
    #                             - `initialize`
    #                             - `setup_ui`
    #                             - `execute`
    #                             - `teardown`
    def set_state_id(self, state_id):
        if not state_id in self.__VALID_STATES:
            raise ValueError("Invalid state id : " + str(state_id))
        self.state_id = state_id

    ## Return a list of the UI that has been modified in Flame by the user.
    #
    #  @param self The object pointer
    #  @return A dictionary of changed UI elements. Key: `name`
    def get_ui_changes(self):
        """Returns a list of the UI that has been modified
        since the last iteration"""
        if not hasattr(self, "ui_changes"):
            return []
        return self.ui_changes

    ## Return the next position in the UI that can accomodate the passed UI
    #  element. This method is essential when laying out the UI programmatically,
    #  since some UI elements take more room than others.
    #
    #  @param self The object pointer
    #  @param  ui_type One of `Browser`, `FloatVector`, `Float`, `Pup`, `Color`, `Toggle`
    #  @return  A dictionary {row, col, page} as the next free location for `ui_type`.
    def get_next_available_position(self, ui_type):
        # loop over all ui and retrieve max col and row
        max_row = -1
        max_col = 0
        max_page = 0

        types = [self.__GLOBAL_ELEMENTS, self.__RENDER_ELEMENTS]
        for t in types:
            for elem in itervalues(self.dynamic_ui[t]):

                elem_type = elem["type"]
                row = elem["row"]
                col = elem["col"]
                page = elem["page"]

                # some dyn ui take up more space
                if (
                    elem_type == "Browser"
                    or elem_type == "FloatVector"
                    or elem_type == "IntVector"
                ):
                    row += 1

                if page > max_page:
                    max_page = page
                    max_col = col
                    max_row = row

                if col > max_col and page == max_page:
                    max_col = col
                    max_row = row

                if row > max_row and col == max_col and page == max_page:
                    max_row = row

        max_row += 1  # next empty position

        if max_row > 4:
            max_row = 0
            max_col += 1

        if max_col > 5:
            max_col = 0
            max_page += 1

        # check if our ui_type fits
        if ui_type == "Browser" or ui_type == "FloatVector" or ui_type == "IntVector":
            if max_row + 1 > 4:
                max_row = 0
                max_col += 1

            if max_col > 3:
                max_col = 0
                max_page += 1

        return {"row": max_row, "col": max_col, "page": max_page}

    #############################################
    # Error messages
    #############################################

    ## Returns a message logged in the aplication log.
    #
    def get_error_msg(self):
        return self.message[self.__ERROR]

    ## Set the error message in the JSON payload.
    #  Allows Flame to write the message to its log file. According to error
    #  logging level.
    #  Other method: print in shell.
    #
    def set_error_msg(self, error):
        self.message[self.__ERROR] = error

    def get_warning_msg(self):
        return self.message[self.__WARNING]

    def set_warning_msg(self, warning):
        self.message[self.__WARNING] = warning

    def get_notice_msg(self):
        return self.message[self.__NOTICE]

    def set_notice_msg(self, notice):
        self.message[self.__NOTICE] = notice

    def get_debug_msg(self):
        return self.message[self.__DEBUG]

    def set_debug_msg(self, debug):
        self.message[self.__DEBUG] = debug

    def get_dialog_msg(self):
        return self.message[self.__DIALOG]

    def set_dialog_msg(self, dialog):
        self.message[self.__DIALOG] = dialog

    #############################################
    # Metadata methods
    #############################################

    ## Return the bit depth of the current batch default.
    #
    #  @param self The object pointer
    #  @return A string
    def get_bit_depth(self):
        return self.adsk_metadata[self.__RESOLUTION][self.__BIT_DEPTH]

    ## Return the colour space of the project.
    #
    #  @param self The object pointer
    #  @return A string
    def get_colour_space(self):
        return self.adsk_metadata[self.__COLOUR_SPACE]

    ## Return the today's date.
    #
    #  @param self The object pointer
    #  @return A JSON object for `year`, `day`, `month`
    def get_date(self):
        return self.adsk_metadata[self.__DATE]

    ## Return today's day.
    #
    #  @param self The object pointer
    #  @return An integer for the day of the month.
    def get_date_day(self):
        return self.adsk_metadata[self.__DATE][self.__DATE_DAY]

    ## Return today's month
    #
    #  @param self The object pointer
    #  @return An integer for the month.
    def get_date_month(self):
        return self.adsk_metadata[self.__DATE][self.__DATE_MONTH]

    ## Return today's year.
    #
    #  @param self The object pointer
    #  @return An integer for the year.
    def get_date_year(self):
        return self.adsk_metadata[self.__DATE][self.__DATE_YEAR]

    ## Return the current Batch frame.
    #
    #  @param self The object pointer
    #  @return An integer
    def get_frame(self):
        return self.adsk_metadata[self.__FRAME]

    ## Return the frame ratio of the current batch default
    #
    #  @param self The object pointer
    #  @return A float
    def get_frame_ratio(self):
        return self.adsk_metadata[self.__RESOLUTION][self.__FRAME_RATIO]

    ## Return the frame rate.
    #
    #  @param self The object pointer
    #  @return A string
    def get_framerate(self):
        return self.adsk_metadata[self.__FRAMERATE]

    ## Return the height of the current batch default.
    #
    #  @param self The object pointer
    def get_height(self):
        return self.adsk_metadata[self.__RESOLUTION][self.__HEIGHT]

    ## Return a string as the image format as set by the #set_img_format setter.
    #
    #  @param self The object pointer
    #  @return A string
    def get_img_format(self):
        return self.adsk_metadata[self.__IMG_FORMAT]

    ## Set the format of the image exported by Batch and to be used by the
    #  external processor.
    #
    #  @param self The object pointer
    #  @param img_format One of the supported image formats:
    #         - `jpeg`
    #         - `exr`
    #         - `sgi`
    #         - `tga`
    #         - `tiff`
    def set_img_format(self, img_format):
        if not img_format in self.__VALID_IMG_FORMATS:
            raise ValueError("Invalid img format : " + str(img_format))
        self.adsk_metadata[self.__IMG_FORMAT] = img_format

    ## Return the nickname of the Flame user.
    #
    #  @param self The object pointer
    #  @return A string
    def get_nickname(self):
        return self.adsk_metadata[self.__NICKNAME]

    ## Return the name of the node in Batch, defined in the Node Name field in
    #  the Batch UI.
    #
    #  @param self The object pointer
    #  @return A string
    def get_node_name(self):
        return self.adsk_metadata[self.__NODE_NAME]

    ## Return the name of the Flame Project.
    #
    #  @param self The object pointer
    #  @return A string
    def get_project(self):
        return self.adsk_metadata[self.__PROJECT]

    ## Return the nickname of the Flame Project.
    #
    #  @param self The object pointer
    #  @return A string
    def get_project_nickname(self):
        return self.adsk_metadata[self.__PROJECT_NICKNAME]

    ## Return the record timecode of the current frame.
    #
    #  @param self The object pointer
    #  @return A string
    def get_record_time_code(self):
        return self.adsk_metadata[self.__RECORD_TIME_CODE]

    ## Return the `width`, `bit_depth`, `frame_ratio`, and `height` of the current batch default
    #
    #  @param self The object pointer
    #  @return A JSON object
    def get_resolution(self):
        return self.adsk_metadata[self.__RESOLUTION]

    ## Return the source timecode of the current frame.
    #
    #  @param self The object pointer
    #  @return A string
    def get_source_time_code(self):
        return self.adsk_metadata[self.__SOURCE_TIME_CODE]

    ## Return the name of the current shot.
    #
    #  @param self The object pointer
    #  @return A string
    def get_shot_name(self):
        return self.adsk_metadata[self.__SHOT_NAME]

    ## Return the name of the current tape.
    #
    #  @param self The object pointer
    #  @return A string
    def get_tape_name(self):
        return self.adsk_metadata[self.__TAPE_NAME]

    ## Return the name of the current Flame user.
    #
    #  @param self The object pointer
    #  @return A string
    def get_user(self):
        return self.adsk_metadata[self.__USER]

    ## Return the width of the current batch default
    #
    #  @param self The object pointer
    def get_width(self):
        return self.adsk_metadata[self.__RESOLUTION][self.__WIDTH]

    ## Return the name of the workstation where Flame currently runs.
    #
    #  @param self The object pointer
    #  @return A string
    def get_workstation(self):
        return self.adsk_metadata[self.__WORKSTATION]

    ####################
    # Sockets
    ####################

    ## Set the path for the outgoing file for the socket identified by `idx`.
    #
    #  Pad the node with as many undefined sockets as required to reach the
    #  requested `idx` value. If only 4 input sockets exist, and `idx` if 9
    #  is passed, 5 `undefined` input sockets are created. //review
    #
    #  Use #add_in_socket to add an input socket to the next available `idx`.
    #  When a node is loaded in Batch, it is always created with 3 input nodes:
    #  Front/Undefined/Matte. Consider using #remove_in_sockets before adding
    #  back input sockets.
    #
    #  @param self The object pointer
    #  @param idx  The index of the socket. 0 based.
    #  @param socket_type The type of the input socket. From the following list:
    #                     - `Front`
    #                     - `Back`
    #                     - `Matte`
    #                     - `3DMotion`
    #                     - `Background`
    #                     - `MotionVector`
    #                     - `Normal`
    #                     - `Position`
    #                     - `Uv`
    #                     - `ZDepth`
    #                     - `undefined`
    #  @param path  Where the outgoing image file is written. This is the where
    #               the external application can expect the file to be written.
    def set_in_socket(self, idx, socket_type, path):
        if not socket_type in self.__VALID_IN_SOCKET_TYPE:
            raise ValueError("Invalid in socket type : " + str(socket_type))

        # Pad with "invalid sockets" if needed
        while len(self.sockets[self.__IN]) <= idx:
            self.sockets[self.__IN].append(
                {self.__PATH: "", self.__DESCRIPTION: "undefined"}
            )

        self.sockets[self.__IN][idx] = {
            self.__PATH: path,
            self.__DESCRIPTION: socket_type,
        }

    ## Add an input socket at the next available `idx`.
    #  Use #set_in_socket to add an input socket at an arbitrary `idx`.
    #  When a node is loaded in Batch, it is always created with 3 input nodes:
    #  Front/Undefined/Matte. Consider using #remove_in_sockets before adding
    #  back input sockets.
    #
    #  @param self The object pointer
    #  @param socket_type The type of the input socket. From the following list:
    #                     - `Front`
    #                     - `Back`
    #                     - `Matte`
    #                     - `3DMotion`
    #                     - `Background`
    #                     - `MotionVector`
    #                     - `Normal`
    #                     - `Position`
    #                     - `Uv`
    #                     - `ZDepth`
    #                     - `undefined`
    #  @param path  Where the outgoing image file is written. This is the where
    #               the external application can expect the file to be written.
    def add_in_socket(self, socket_type, path):
        if not socket_type in self.__VALID_IN_SOCKET_TYPE:
            raise ValueError("Invalid in socket type : " + str(socket_type))

        # Pad with "invalid sockets" if needed
        self.sockets[self.__IN].append(
            {self.__PATH: path, self.__DESCRIPTION: socket_type}
        )

    ## Return the number of input sockets on the Pybox node.
    #
    #  @param self The object pointer
    def get_num_in_sockets(self):
        return len(self.sockets[self.__IN])

    ## Return a JSON object with path and description of the select input socket.
    #
    #  @param self The object pointer
    #  @param idx  The index of the input socket to query. 0 based.
    def get_in_socket(self, idx):
        return self.sockets[self.__IN][idx]

    ## Return a JSON object as a collection with path and description for each
    #  input socket. Ordered by idx value.
    #
    #  @param self The object pointer
    def get_in_sockets(self):
        return self.sockets[self.__IN]

    ## Return the path of the input socket at index `idx`
    #
    #  @param self The object pointer
    #  @param[in] idx The index of the input socket to query.
    def get_in_socket_path(self, idx):
        return self.sockets[self.__IN][idx][self.__PATH]

    ## Remove from the node the input socket at index `idx`.
    #
    #  @param self The object pointer
    #  @param[in] idx The index of the input socket to remove. 0 based.
    def remove_in_socket(self, idx):
        self.sockets[self.__IN].pop(idx)

    ## Remove from the node every input socket.
    #
    #  @param self The object pointer
    def remove_in_sockets(self):
        self.sockets[self.__IN] = []

    ## Set the path for the incoming file and the type for the socket identified
    #  by `idx`. Pad the node with undefined sockets to reach the requested `idx`
    #  value: If only 4 output sockets exist, and and `idx` if 9 is passed, 5
    #  `undefined` output sockets are created.
    #  Use #add_out_socket to add an output socket to the next available `idx`.
    #  When a node is loaded in Batch, it is always created with 2 output nodes:
    #  Front/Matte. Consider using #remove_out_sockets before adding back output
    #  sockets.
    #
    #  @param self The object pointer
    #  @param idx  The index of the socket. 0 based.
    #  @param socket_type The type of the output socket. From the following list:
    #                     - `Result`
    #                     - `OutMatte`
    #                     - `Out3DMotion`
    #                     - `OutBackground`
    #                     - `OutMotionVector`
    #                     - `OutNormal`
    #                     - `OutPosition`
    #                     - `OutUv`
    #                     - `OutZDepth`
    #                     - `undefined`
    #  @param path  Where the incoming image file is written. This is the where
    #               the external application is expected to write back its output.
    def set_out_socket(self, idx, socket_type, path):
        if not socket_type in self.__VALID_OUT_SOCKET_TYPE:
            raise ValueError("Invalid out socket type")

        # Pad with "invalid sockets" if needed
        while len(self.sockets[self.__OUT]) <= idx:
            self.sockets[self.__OUT].append(
                {self.__PATH: "", self.__DESCRIPTION: "undefined"}
            )

        self.sockets[self.__OUT][idx] = {
            self.__PATH: path,
            self.__DESCRIPTION: socket_type,
        }

    ## Add an output socket at the next available `idx`.
    #  Use #set_out_socket to add an output socket at an arbitrary `idx`.
    #  When a node is loaded in Batch, it is always created with 2 output nodes:
    #  Front/Undefined/Matte. Consider using #remove_out_sockets before adding
    #  back output sockets.
    #
    #  @param self The object pointer
    #  @param socket_type The type of the output socket. From the following list:
    #                     - `Result`
    #                     - `OutMatte`
    #                     - `Out3DMotion`
    #                     - `OutBackground`
    #                     - `OutMotionVector`
    #                     - `OutNormal`
    #                     - `OutPosition`
    #                     - `OutUv`
    #                     - `OutZDepth`
    #                     - `undefined`
    #  @param path  Where the incoming image file is written. This is the where
    #               the external application is expected to write back its output.
    def add_out_socket(self, socket_type, path):
        if not socket_type in self.__VALID_OUT_SOCKET_TYPE:
            raise ValueError("Invalid out socket type : " + str(socket_type))

        # Pad with "invalid sockets" if needed
        self.sockets[self.__OUT].append(
            {self.__PATH: path, self.__DESCRIPTION: socket_type}
        )

    ## Return the number of output sockets on the Pybox node.
    #
    #  @param self The object pointer
    def get_num_out_sockets(self):
        return len(self.sockets[self.__OUT])

    ## Return a JSON object with path and description of the select output socket.
    #
    #  @param self The object pointer
    #  @param[in] idx The index of the output socket to query. 0 based.
    def get_out_socket(self, idx):
        return self.sockets[self.__OUT][idx]

    ## Return a JSON object as a collection of with path and description for
    #  each output socket. Ordered by idx value.
    #
    #  @param self The object pointer
    def get_out_sockets(self):
        return self.sockets[self.__OUT]

    ## Return the path of the select output socket.
    #
    #  @param self The object pointer
    #  @param[in] idx   The index of the input socket to query.
    def get_out_socket_path(self, idx):
        return self.sockets[self.__OUT][idx][self.__PATH]

    ## Remove from the node the output socket defined by `idx`.
    #
    #  @param self The object pointer
    #  @param[in] idx The index of the input socket to remove. 0 based.
    def remove_out_socket(self, idx):
        self.sockets[self.__OUT].pop(idx)

    ## Remove from the node every input socket.
    #
    #  @param self The object pointer
    def remove_out_sockets(self):
        self.sockets[self.__OUT] = []

    ## Set 1 or more pages to use in the UI. Create a page with #create_page.
    #
    #  @param self The object pointer
    #  @param pages JSON object of a page.
    def set_ui_pages(self, *pages):
        """Set 1 or more pages to use in the UI.
        A page can be created with the create_page function."""
        self.dynamic_ui[self.__PAGES] = list(pages)

    ## Get the pages set in the Pybox node. Pages are set in the UI using
    #  #set_ui_pages.
    #
    #  @param self The object pointer.
    #  @return A JSON object, with `name` of a page and all its associated `col`.
    def set_ui_pages_array(self, pages):
        """Set 1 or more pages to use in the UI.
        A page can be created with the create_page function."""
        if isinstance(pages, list):
            self.dynamic_ui[self.__PAGES] = pages

    ## Get the pages set in the Pybox node. Pages are set in the UI using
    #  #set_ui_pages.
    #
    #  @param self The object pointer.
    #  @return A JSON object, with `name` of a page and all its associated `col`.
    def get_ui_pages(self):
        return self.dynamic_ui[self.__PAGES]

    ####################
    # Render elements
    ####################

    ## Render elements (#add_render_elements) do not trigger Python calls
    #  until the Pybox result is displayed in its Result View (F4) or pulled
    #  from a node downstream from the Pybox. As opposed to Global elements,which
    #  are UI elements that always triggger a Python call.
    #   @sa get_render_element
    #   @sa get_render_elements
    #   @sa get_render_element_value
    #   @sa set_render_element_value
    #   @sa remove_render_element
    #
    #   @param self The object pointer.
    #   @param elements Any number of UI elements `name` to add to the Render
    #                   elements array.
    def add_render_elements(self, *elements):
        for element in elements:
            if not "name" in element:
                raise ValueError("Element has no name")
            self.dynamic_ui[self.__RENDER_ELEMENTS][element["name"]] = element

    ## Get the list of UI elements that were set as render elements.
    #  @sa add_render_elements
    #
    #   @param self The object pointer.
    #   @return An array of UI `name`.
    def get_render_elements(self):
        return self.dynamic_ui[self.__RENDER_ELEMENTS]

    ## Get the properties of one UI elements that was set as a render element.
    #  @sa add_render_elements
    #
    #   @param self The object pointer.
    #   @param name The `name` of the UI element to query.
    #   @return A dictionary of the properties of a UI element.
    def get_render_element(self, name):
        if name in self.dynamic_ui[self.__RENDER_ELEMENTS]:
            return self.dynamic_ui[self.__RENDER_ELEMENTS][name]

        return {}

    ## Return the `value` property of one UI elements that was set as a
    #  render element.
    #  @sa add_render_elements
    #
    #   @param self The object pointer.
    #   @param name `name` of a UI element.
    #   @return     Actual value depends on type of the UI element queried. An
    #               array of float for vector numeric (2D or 3D) and color pot
    #               (3D) UI elements. A single value for all others.
    def get_render_element_value(self, name):
        elem = self.dynamic_ui[self.__RENDER_ELEMENTS][name]
        if elem["type"] == "FloatVector" or elem["type"] == "Color":
            return self.dynamic_ui[self.__RENDER_ELEMENTS][name][self.__VALUES]

        else:
            return self.dynamic_ui[self.__RENDER_ELEMENTS][name][self.__VALUE]

    ## Set the `value` property of one UI elements that was set as a
    #  render element.
    #  @sa add_render_elements
    #
    #   @param self The object pointer.
    #   @param name     `name` of a UI element.
    #   @param value    The value to set. Be careful: vector numeric and color pot
    #                   are arrays of float. Others are single values.
    def set_render_element_value(self, name, value):
        elem = self.dynamic_ui[self.__RENDER_ELEMENTS][name]

        if elem["type"] == "FloatVector":
            self.dynamic_ui[self.__RENDER_ELEMENTS][name][self.__VALUES] = value

        else:
            self.dynamic_ui[self.__RENDER_ELEMENTS][name][self.__VALUE] = value

    ## Remove a UI element from the Render element set.
    #  Important: a UI element must be part of a set (Render or Global) to be
    #  displayed in the Pybox node UI.
    #  @sa add_render_elements
    #
    #   @param self The object pointer.
    #   @param name     `name` of a UI element to remove.
    def remove_render_element(self, name):
        self.dynamic_ui[self.__RENDER_ELEMENTS].pop(name)

    ####################
    # Global elements
    ####################

    ## Global elements are UI elements that always triggger a Python call.
    #  As opposed to Render elements (#add_render_elements)that do not trigger
    #  Python calls until the Pybox result is displayed in its Result View (F4)
    #  or pulled from a node downstream from the Pybox.
    #   @sa get_global_element
    #   @sa get_global_elements
    #   @sa get_global_element_value
    #   @sa set_global_element_value
    #   @sa remove_global_element
    #
    #   @param self The object pointer.
    #   @param elements Any number of UI elements `name` to add to the Global
    #                   elements array.
    def add_global_elements(self, *elements):
        for element in elements:
            if not "name" in element:
                raise ValueError("Element has no name")
            self.dynamic_ui[self.__GLOBAL_ELEMENTS][element["name"]] = element

    ## Get the list of UI elements that were set as global elements.
    #  @sa add_global_elements
    #
    #   @param self The object pointer.
    #   @return An array of UI `name`.
    def get_global_elements(self):
        return self.dynamic_ui[self.__GLOBAL_ELEMENTS]

    ## Get the properties of one UI elements that was set as a global element.
    #  @sa add_global_elements
    #
    #   @param self The object pointer.
    #   @param name The `name` of the UI element to query.
    #   @return A dictionary of the properties of a UI element.
    def get_global_element(self, name):
        if name in self.dynamic_ui[self.__GLOBAL_ELEMENTS]:
            return self.dynamic_ui[self.__GLOBAL_ELEMENTS][name]

        return {}

    ## Return the `value` property of one UI elements that was set as a
    #  global element.
    #  @sa add_global_elements
    #
    #   @param self The object pointer.
    #   @param name `name` of a UI element.
    #   @return     Actual value depends on type of the UI element queried. An
    #               array of float for vector numeric (2D or 3D) and color pot
    #               (3D) UI elements. A single value for all others.
    def get_global_element_value(self, name):
        elem = self.dynamic_ui[self.__GLOBAL_ELEMENTS][name]
        if elem["type"] == "FloatVector" or elem["type"] == "Color":
            return self.dynamic_ui[self.__GLOBAL_ELEMENTS][name][self.__VALUES]

        else:
            return self.dynamic_ui[self.__GLOBAL_ELEMENTS][name][self.__VALUE]

    ## Set the `value` property of one UI elements that was set as a
    #  global element.
    #  @sa add_global_elements
    #
    #   @param self The object pointer.
    #   @param name     `name` of a UI element.
    #   @param value    The value to set. Be careful: vector numeric and color pot
    #                   are arrays of float. Others are single values.
    def set_global_element_value(self, name, value):

        elem = self.dynamic_ui[self.__GLOBAL_ELEMENTS][name]

        if elem["type"] == "FloatVector":
            self.dynamic_ui[self.__GLOBAL_ELEMENTS][name][self.__VALUES] = value

        else:
            self.dynamic_ui[self.__GLOBAL_ELEMENTS][name][self.__VALUE] = value

    ## Remove a UI element from the Global element set.
    #  Important: a UI element must be part of a set (Render or Global) to be
    #  displayed in the Pybox node UI.
    #  @sa add_global_elements
    #
    #   @param self The object pointer.
    #   @param name     `name` of a UI element to remove.
    def remove_global_element(self, name):
        self.dynamic_ui[self.__GLOBAL_ELEMENTS].pop(name)

    #############################
    # Process informations (RO)
    #############################

    ## Indicates if processing data are available.
    #  Processing happens when a node downstream the Pybox node requests a frame
    #  or the Pybox itself displays its result view (F4).
    #  There is no processing available when a Global UI element has been
    #  triggered without an active result viewport.
    #  An input socket is inactive when there is no associated data.
    #  An output socket is active when it is the currently processed ouput
    #  socket, and indicated by #get_process_output_socket.
    #
    #  The following functions all fail if `is_processing` = False.
    #  - #get_process_output_socket
    #  - #get_process_frame
    #  - #get_process_quality
    #  - #get_process_resolution_factor_x
    #  - #get_process_resolution_factor_y
    #  - #get_process_in_socket
    #  - #get_process_out_socket
    #  - #get_socket_id
    #  - #is_socket_active
    #  - #get_socket_width
    #  - #get_socket_height
    #  - #get_socket_bit_depth
    #  - #get_socket_frame_ratio
    #  - #get_socket_colour_space
    #
    #  @param self The object pointer.
    #  @return Boolean
    def is_processing(self):
        return hasattr(self, self.__PROCESS_INFORMATIONS)

    ## Get the currently processed output socket `idx`
    #  An output socket is active when it is the currently processed ouput socket.
    #
    #  @return
    def get_process_output_socket(self):
        if not self.is_processing():
            raise ValueError("No processing information")
        return self.process[self.__PROCESS_PARAMS][self.__PROCESS_OUTPUT_SOCKET]

    ## Get the frame number of the frame being processed.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #  @return A float
    def get_process_frame(self):
        if not self.is_processing():
            raise ValueError("No processing information")
        return float(self.process[self.__PROCESS_PARAMS][self.__FRAME])

    ## Indicates the quality expected by Batch, in terms to adaptive degradation.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #
    #  @return An integer where
    #   - -1: unkonwn
    #   - 0: Full resolution
    #   - 1: Degrade
    def get_process_quality(self):
        if not self.is_processing():
            raise ValueError("No processing information")
        return int(self.process[self.__PROCESS_PARAMS][self.__PROCESS_QUALITY])

    ## Indicate a scaling applied to the current socket width, with respect to the
    #  full resolution.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #
    #  @return A float, 1 for full, 0.5 for proxy.
    def get_process_resolution_factor_x(self):
        if not self.is_processing():
            raise ValueError("No processing information")
        return float(self.process[self.__PROCESS_PARAMS][self.__PROCESS_X_FACTOR])

    ## Indicate a scaling applied to the current socket height, with respect to the
    #  full resolution.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #
    #  @return A float, 1 for full, 0.5 for proxy.
    def get_process_resolution_factor_y(self):
        if not self.is_processing():
            raise ValueError("No processing information")
        return float(self.process[self.__PROCESS_PARAMS][self.__PROCESS_Y_FACTOR])

    ## Get the processing data for the given input socket.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #
    #  @param idx Index of the socket queried.
    #  @return An array of data to be passed to the following functions as `socket`.
    #   - #get_socket_id
    #   - #is_socket_active
    #   - #get_socket_width
    #   - #get_socket_height
    #   - #get_socket_bit_depth
    #   - #get_socket_frame_ratio
    #   - #get_socket_colour_space
    def get_process_in_socket(self, idx):
        if not self.is_processing():
            raise ValueError("No processing information")
        return self.process[self.__IN][idx]

    ## Get the processing data for the given output socket.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #
    #  @param idx Index of the socket queried.
    #  @return An array of data to be passed to the following functions as `socket`.
    #   - #get_socket_id
    #   - #is_socket_active
    #   - #get_socket_width
    #   - #get_socket_height
    #   - #get_socket_bit_depth
    #   - #get_socket_frame_ratio
    #   - #get_socket_colour_space
    def get_process_out_socket(self, idx):
        if not self.is_processing():
            raise ValueError("No processing information")
        return self.process[self.__OUT][idx]

    ## Get a unique name for the socket.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #  @sa BaseClass::get_process_in_socket
    #  @sa BaseClass::get_process_out_socket
    #
    #  @param socket A get_process_in_socket or get_process_out_socket object.
    #  @return A string
    def get_socket_id(self, socket):
        return socket[self.__SOCKET_ID]

    ## Get whether or not the socket is active.
    #  Actual interpretation of the return depends on the socket being queried.
    #  - If the socket is an INPUT. True: There is media associated with the
    #   socket for the current processing. False: No media associated for this
    #   socket.
    #  - If the socket is an OUTPUT. True: Media is request downstream (or in F4
    #   result view) from this socket. False: No one cares for this socket.
    #
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #  @sa BaseClass::get_process_in_socket
    #  @sa BaseClass::get_process_out_socket
    #
    #  @param socket A get_process_in_socket or get_process_out_socket object.
    #  @return Boolean.
    def is_socket_active(self, socket):
        return socket[self.__SOCKET_ACTIVE]

    ## If #is_socket_active = TRUE, get the width of the socket. In pixels.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #  @sa BaseClass::get_process_in_socket
    #  @sa BaseClass::get_process_out_socket
    #
    #  @param socket A get_process_in_socket or get_process_out_socket object.
    #  @return An integer
    def get_socket_width(self, socket):
        return int(socket[self.__RESOLUTION][self.__WIDTH])

    ## If #is_socket_active = TRUE, get the height of the socket. In pixels.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #  @sa BaseClass::get_process_in_socket
    #  @sa BaseClass::get_process_out_socket
    #
    #  @param socket A get_process_in_socket or get_process_out_socket object.
    #  @return Integer
    def get_socket_height(self, socket):
        return int(socket[self.__RESOLUTION][self.__HEIGHT])

    ## If #is_socket_active = TRUE, get the bit depth of the socket.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #  @sa BaseClass::get_process_in_socket
    #  @sa BaseClass::get_process_out_socket
    #
    #  @param socket A get_process_in_socket or get_process_out_socket object.
    #  @return An Integer as the bit depth.
    def get_socket_bit_depth(self, socket):
        return int(socket[self.__RESOLUTION][self.__BIT_DEPTH])

    ## If #is_socket_active = TRUE, get the frame ratio of the socket.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #  @sa BaseClass::get_process_in_socket
    #  @sa BaseClass::get_process_out_socket
    #
    #  @param socket A get_process_in_socket or get_process_out_socket object.
    #  @return A float
    def get_socket_frame_ratio(self, socket):
        return float(socket[self.__RESOLUTION][self.__FRAME_RATIO])

    ## If #is_socket_active = TRUE, get the resolution of the socket.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #  @sa BaseClass::get_process_in_socket
    #  @sa BaseClass::get_process_out_socket
    #
    #  @param socket A get_process_in_socket or get_process_out_socket object.
    #  @return Resolution object
    def get_socket_resolution(self, socket):
        return socket[self.__RESOLUTION]

    ## If #is_socket_active = TRUE, get the colour space of the socket.
    #  Fails if is_processing = False.
    #  @sa BaseClass::is_processing
    #  @sa BaseClass::get_process_in_socket
    #  @sa BaseClass::get_process_out_socket
    #
    #  @param socket A get_process_in_socket or get_process_out_socket object.
    #  @return       A string
    def get_socket_colour_space(self, socket):
        return socket[self.__COLOUR_SPACE]

    ####################
    # IO methods
    ####################

    ## Write the JSON payload file to disk so that it can be read by Flame.
    #  To be called just befre the script exits.
    #
    #  @param self The object pointer.
    #  @param path The path to the JSON file. Must be identical to argv[v0].
    def write_to_disk(self, path):
        """Function to be called at the end of the processing to
        write the file to disk so it can be read by Flame."""
        self.previous_data = json.dumps(self.__dict__, indent=4)
        try:
            with open(path, "w+") as output_file:
                output_file.write(json.dumps(self.__dict__, indent=4))
        except EnvironmentError:
            print("ERROR: pybox json file could not be opened")

    ####################
    # Private constants
    ####################
    __IN = "in"
    __OUT = "out"
    __PAGES = "pages"
    __RENDER_ELEMENTS = "render_elements"
    __GLOBAL_ELEMENTS = "global_elements"
    __PATH = "path"
    __DESCRIPTION = "description"
    __VALUE = "value"
    __VALUES = "values"
    __PROCESS_INFORMATIONS = "process"
    __PROCESS_PARAMS = "params"
    __PROCESS_OUTPUT_SOCKET = "active_socket"
    __PROCESS_QUALITY = "quality"
    __PROCESS_X_FACTOR = "working_res_factor_x"
    __PROCESS_Y_FACTOR = "working_res_factor_y"
    __SOCKET_ID = "description"
    __SOCKET_ACTIVE = "active"

    ####################
    # Messaging
    ####################

    __ERROR = "error"
    __WARNING = "warning"
    __NOTICE = "notice"
    __DEBUG = "debug"
    __DIALOG = "dialog"

    ####################
    # ADSK metadata
    ####################
    __BIT_DEPTH = "bit_depth"
    __COLOUR_SPACE = "colour_space"
    __DATE = "date"
    __DATE_DAY = "day"
    __DATE_MONTH = "month"
    __DATE_YEAR = "year"
    __FRAME = "frame"
    __FRAME_RATIO = "frame_ratio"
    __FRAMERATE = "framerate"
    __HEIGHT = "height"
    __IMG_FORMAT = "format"
    __NICKNAME = "nickname"
    __NODE_NAME = "node_name"
    __PROJECT_NICKNAME = "project_nickname"
    __PROJECT = "project"
    __RECORD_TIME_CODE = "record_time_code"
    __RESOLUTION = "resolution"
    __SOURCE_TIME_CODE = "source_time_code"
    __TAPE_NAME = "tape_name"
    __SHOT_NAME = "shot_name"
    __USER = "user"
    __WIDTH = "width"
    __WORKSTATION = "workstation"

    #############################################
    # Validation data
    #############################################
    __VALID_STATES = ["initialize", "setup_ui", "execute", "teardown"]
    __VALID_IMG_FORMATS = ["jpeg", "exr", "sgi", "tga", "tiff"]
    __VALID_IN_SOCKET_TYPE = [
        "Front",
        "Back",
        "Matte",
        "3DMotion",
        "Background",
        "MotionVector",
        "Normal",
        "Position",
        "Uv",
        "ZDepth",
        "undefined",
    ]

    __VALID_OUT_SOCKET_TYPE = [
        "Result",
        "OutMatte",
        "Out3DMotion",
        "OutBackground",
        "OutMotionVector",
        "OutNormal",
        "OutPosition",
        "OutUv",
        "OutZDepth",
        "undefined",
    ]
