"""
Pybox setup for Warp.

Two Front inputs, no matte inputs, two OutUv outputs. Uses a separate
model architecture from the interpolation effects, so weights live in
their own WEIGHTS_DIR subfolder rather than the shared weights/ folder.

The daemon-side processing is a stub (see effect.py's process_warp
handler) pending the model's actual forward() signature.
"""

import os
import sys
import time
import socket
import subprocess
import json
import glob
import re
import shutil
import pybox_v1 as pybox

from pathlib import Path

# Set True for verbose per-dispatch tracing (daemon calls, state, message
# block contents). Errors always print regardless of this flag.
DEBUG = False

EFFECT_NAME = 'ML Warp'
IMAGE_FORMAT = "exr"

MODEL_UI_ELEMENT     = "Model"
SCALE_UI_ELEMENT     = "Scale"
DETAIL_UI_ELEMENT    = "Detail"
SMOOTH_UI_ELEMENT    = "Smooth"

# Stops at 4: this model's halving_steps() drops the max(...,1) guard that
# the interpolation model has, so scale 2 gives [2,1,0,1] and scale 1 gives
# [1,0,0,1] — a zero stage divisor, i.e. ZeroDivisionError inside forward().
SCALE_VALUES = [64, 32, 16, 8, 4]
# No Bidirectional or Iterational toggles — those belong to the
# interpolation model's forward() signature, not this one. Detail and
# Smooth take the place the interpolation effects give to Ratio.

# Empty until this script has been told where it lives. The locate flow below
# fills it in and rewrites this line on disk.
SCRIPT_LOCATION = ''
SCRIPT_NAME     = 'fluidmorph.v001'
SOCKET_PATH     = '/dev/shm/fluidmorph_effect.sock'
STATE_PATH      = '/dev/shm/fluidmorph_effect.state.json'

# Where to look for the bundle when SCRIPT_LOCATION above is empty or wrong.
PRESET_GLOB   = '/opt/Autodesk/presets/*/pybox'
PRESET_ROOT   = '/opt/Autodesk/presets'

# The file browser needs a real, existing folder to open into — an empty
# value makes Flame's native browser fail outright ("lacks read/write
# permissions to access ."), it does not just leave the field blank. This is
# also compared against explicitly in _locate() to tell "still showing its
# untouched default" apart from "the user actually picked something", since
# both leave `value` non-empty.
LOCATION_DEFAULT_VALUE = PRESET_ROOT if os.path.isdir(PRESET_ROOT) else '/'

# This script's own filename — the locate flow searches for and rewrites only
# this file, never any sibling.
THIS_SCRIPT = 'warp.py'

# A different model architecture from the interpolation effects, so its
# checkpoints live in their own subfolder rather than the shared weights/
# folder those use.
WEIGHTS_DIR = 'weights.warp'

LOCATION_UI_ELEMENT = 'Script File'


def printd(message=None):
    """Verbose tracing — only prints when DEBUG is True."""
    if DEBUG:
        print(f'{SCRIPT_NAME} [debug]: {message}')


def printe(message=None):
    """Errors — always prints, regardless of DEBUG."""
    print(f'{SCRIPT_NAME} [error]: {message}')


def read_state():
    """Read the persisted failure state. Never raises."""
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def write_state(patch):
    """Merge `patch` into the persisted state. Never raises.

    Several independent flows share this one state file (the locate flow's
    prompted/rejected/located, and fail()'s signature/message). A plain
    overwrite here previously let a later write from one flow silently erase
    keys written by another, which produced state loss across dispatches —
    e.g. showing the "prompted" dialog again because a later write had wiped
    that key, alternating between two dialogs that should each show once.
    """
    try:
        state = read_state()
        state.update(patch)
        with open(STATE_PATH, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        printe(f'Could not write state file: {e}')


def clear_state():
    try:
        if os.path.exists(STATE_PATH):
            os.unlink(STATE_PATH)
    except Exception:
        pass

# ── Locating the effect bundle ─────────────────────────────────────────────────

def bundle_path(location, *parts):
    """Path inside the effect bundle held at `location`. None if unlocated."""
    if not location:
        return None
    return os.path.join(os.path.abspath(location), SCRIPT_NAME, *parts)


def is_valid_location(location):
    """
    True if `location` holds this effect's bundle and this script's own file.

    The layout is location/SCRIPT_NAME/<this file>, alongside WEIGHTS_DIR
    and effect/ — all three live inside the SCRIPT_NAME folder, not at
    `location` itself. WEIGHTS_DIR (not 'weights') because this effect's
    checkpoints are a different, incompatible architecture from the ones
    the other pybox scripts in this bundle use.
    """
    if not location or not os.path.isdir(location):
        return False
    return (os.path.isfile(bundle_path(location, THIS_SCRIPT))
            and os.path.isdir(bundle_path(location, WEIGHTS_DIR))
            and os.path.isfile(bundle_path(location, 'effect', 'effect.py')))


def candidate_locations():
    """Where to look, in order: the configured path, then installed presets."""
    candidates = [SCRIPT_LOCATION]
    # Newest Flame version first, so a fresh install wins over an old one.
    candidates.extend(sorted(glob.glob(PRESET_GLOB), reverse=True))
    return candidates


def resolve_location():
    """First candidate that actually holds the bundle, or None."""
    for candidate in candidate_locations():
        if is_valid_location(candidate):
            return os.path.abspath(candidate)
    return None


def write_location_to_self(new_location):
    """
    Rewrite the SCRIPT_LOCATION line in this script's own file.

    Edits the real file on disk at `new_location`, not the temp copy Flame is
    executing, which is why the node has to be rebuilt afterwards to pick the
    change up.

    Returns (ok, message).
    """
    new_location = os.path.abspath(new_location)
    target = bundle_path(new_location, THIS_SCRIPT)

    if not target or not os.path.isfile(target):
        return False, f'{THIS_SCRIPT} not found under {new_location}'

    pattern     = re.compile(r'^SCRIPT_LOCATION\s*=.*$', re.MULTILINE)
    replacement = 'SCRIPT_LOCATION = ' + repr(new_location)

    try:
        with open(target) as handle:
            source = handle.read()
    except Exception as e:
        return False, str(e)

    if not pattern.search(source):
        return False, f'no SCRIPT_LOCATION line in {THIS_SCRIPT}'

    try:
        backup = target + '.bak'
        if not os.path.exists(backup):
            shutil.copyfile(target, backup)

        tmp = target + '.tmp'
        with open(tmp, 'w') as handle:
            handle.write(pattern.sub(replacement, source, count=1))
        os.replace(tmp, target)
    except Exception as e:
        return False, str(e)

    return True, THIS_SCRIPT


RESOLVED_LOCATION = resolve_location()


def scan_weights(location):
    """Available .pth weights. Never raises — returns a placeholder instead."""
    weights_abs_path = bundle_path(location, WEIGHTS_DIR)
    if not weights_abs_path or not os.path.isdir(weights_abs_path):
        return [('None', 'None')]
    try:
        weights_files = sorted([
            os.path.join(weights_abs_path, f)
            for f in os.listdir(weights_abs_path)
            if f.endswith('.pth')
        ])
    except Exception as e:
        printe(f'Could not scan weights in {weights_abs_path}: {e}')
        return [('None', 'None')]
    if not weights_files:
        return [('None', 'None')]
    return [(os.path.splitext(os.path.basename(f))[0], f) for f in weights_files]


MODELS = scan_weights(RESOLVED_LOCATION)

effect_python = bundle_path(
    RESOLVED_LOCATION, 'packages', 'miniconda', 'appenv', 'bin', 'python'
)
effect_script = bundle_path(RESOLVED_LOCATION, 'effect', 'effect.py')


# ── Daemon management ──────────────────────────────────────────────────────────

def is_daemon_running():
    """Check if the daemon socket is up and accepting connections."""
    if not os.path.exists(SOCKET_PATH):
        return False
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(SOCKET_PATH)
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def ensure_daemon_running():
    """Start the effect daemon if it is not already running."""
    if is_daemon_running():
        return

    if not effect_python or not effect_script:
        raise RuntimeError(
            f'{SCRIPT_NAME} bundle location is unknown — cannot start the daemon'
        )

    printd(f"Starting {EFFECT_NAME} effect daemon...")

    # Remove stale socket file if present
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    subprocess.Popen(
        [effect_python, effect_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 10 seconds for the socket to appear
    for _ in range(100):
        if is_daemon_running():
            printd("Daemon is up.")
            return
        time.sleep(0.1)

    raise RuntimeError(
        f"Effect daemon failed to start — socket never appeared at {SOCKET_PATH}"
    )


def send_recv(msg_dict):
    """
    Connect, send one message, receive one response, disconnect.
    Each call is fully self-contained — no persistent connection needed.
    """

    # printd(f"send_recv {msg_dict} command...")
    if msg_dict.get('type') != 'exit':
        ensure_daemon_running()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(60.0)   # generous timeout for model load / GPU inference
    sock.connect(SOCKET_PATH)

    try:
        sock.sendall((json.dumps(msg_dict) + '\n').encode())

        response = b''
        while not response.endswith(b'\n'):
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk

        return json.loads(response.strip())

    finally:
        sock.close()


# ── Pybox class ────────────────────────────────────────────────────────────────

class Warp(pybox.BaseClass):

    def initialize(self):
        # Sockets are declared even when the bundle is missing: Flame demands
        # an output regardless, and passthrough_outputs() below keeps them fed.
        self.set_img_format(IMAGE_FORMAT)
        ext = self.get_img_format()

        tempdir = Path('/dev/shm')

        self.set_in_socket(0, "Front", str(tempdir / f"input0.{ext}"))
        self.set_in_socket(1, "Front", str(tempdir / f"input1.{ext}"))
        self.remove_in_socket(2)

        self.remove_out_sockets()
        # "OutUv" is pybox's dedicated socket type for a UV/motion-vector
        # output. Two of them, semantics to be defined once the warp
        # model's outputs are known.
        self.set_out_socket(0, "OutUv", str(tempdir / f"uv0.{ext}"))
        self.set_out_socket(1, "OutUv", str(tempdir / f"uv1.{ext}"))

        if RESOLVED_LOCATION is None:
            printe(f'Could not locate the {SCRIPT_NAME} bundle.')
            printe(f'Searched: {", ".join(candidate_locations())}')
            self.set_state_id("setup_ui")
            self.setup_ui()
            return

        printd(f'Bundle located at {RESOLVED_LOCATION}')
        ensure_daemon_running()

        # Failure state lives in /dev/shm and survives Flame restarts, so a
        # stale signature would suppress retries (and the dialog) forever.
        # A node load is an explicit fresh start.
        clear_state()

        printd("Pinging effect daemon...")
        response = send_recv({'type': 'command', 'data': 'ping'})
        printd(f"Response: {response}")

        self.set_state_id("setup_ui")
        self.setup_ui()

    # ── Locating the bundle ────────────────────────────────────────────────────

    def setup_locate_ui(self):
        """
        Minimal UI shown when the script could not locate itself: a file
        browser and nothing else. Registered as a global element so that
        picking a file triggers a Python call straight away.
        """
        browser = pybox.create_file_browser(
            LOCATION_UI_ELEMENT,
            value=LOCATION_DEFAULT_VALUE,
            extension='py',
            home=LOCATION_DEFAULT_VALUE,
            page=0, col=0, row=0,
            tooltip=(
                f"<b>Script File</b>\nSelect '{THIS_SCRIPT}' — the file this "
                f"node is running from — to record its location."
            ),
            isFileSelector=True,
        )
        self.add_global_elements(browser)

        page = pybox.create_page("Main")
        self.set_ui_pages(page)

        self.set_state_id("execute")
        self.execute()

    def passthrough_outputs(self):
        """
        Copy an input frame to every output socket with a plain file copy.

        Used while the bundle is missing: the daemon is unavailable, and this
        script runs in Flame's Python where the imaging libraries may not be,
        so no decode is possible. The input is a file Flame itself just wrote,
        so copying it is guaranteed to produce something readable — and
        without an output Flame re-requests the frame endlessly.
        """
        source = None
        for idx in range(self.get_num_in_sockets()):
            path = self.get_in_socket_path(idx)
            if path and os.path.isfile(path):
                source = path
                break
        if source is None:
            return

        for idx in range(self.get_num_out_sockets()):
            destination = self.get_out_socket_path(idx)
            if not destination:
                continue
            try:
                tmp = f'{destination}.{os.getpid()}.tmp'
                shutil.copyfile(source, tmp)
                os.replace(tmp, destination)
            except Exception as e:
                printe(f'Passthrough to {destination} failed: {e}')

    def _locate(self):
        """
        Ask the user to point at this script's own file, then record where it
        lives by rewriting that same file.

        Layout is location/SCRIPT_NAME/THIS_SCRIPT, so the selected file's
        location is two directories up, not one.

        prompted/rejected/located are tracked as plain attributes on this
        object rather than in the shared /dev/shm state file. Pybox already
        round-trips every instance attribute through this node's own JSON
        payload (write_to_disk dumps self.__dict__; __init__ restores it), so
        this is naturally scoped to one node and resets when the node is
        deleted — unlike a shared file, which would let one node's dismissed
        dialog silently suppress every future node's dialog, including a
        different script's.
        """
        self.passthrough_outputs()

        selected = self.get_global_element_value(LOCATION_UI_ELEMENT)
        selected = str(selected).strip() if selected else ''

        # The field can't start empty — an empty `value` makes Flame's native
        # browser fail to open at all. So "nothing chosen yet" means the field
        # still shows the default folder it was created with, not that it's
        # literally blank.
        if not selected or selected == LOCATION_DEFAULT_VALUE:
            if not getattr(self, '_locate_prompted', False):
                self._locate_prompted = True
                self.set_dialog_msg(
                    f'{EFFECT_NAME}\n\n'
                    f'A pybox script has no way to find out where it is '
                    f'stored on disk.\n\n'
                    f"Use the Script File field on this node's Main panel to "
                    f"select '{THIS_SCRIPT}' — the file this node is running "
                    f'from. The location will be written into it, so this is '
                    f'only needed once.'
                )
            self.set_error_msg(
                f'{EFFECT_NAME}: location unknown — set Script File'
            )
            return

        # Selecting the exact right filename first gives a crisper rejection
        # than a generic "invalid location" — the browser lists every .py
        # file in a folder, not just this one.
        if os.path.basename(selected) != THIS_SCRIPT:
            if selected != getattr(self, '_locate_rejected', None):
                self._locate_rejected = selected
                self.set_dialog_msg(
                    f'{EFFECT_NAME}\n\n'
                    f'{selected}\n\nis not \'{THIS_SCRIPT}\'. Select the file '
                    f'named exactly that.'
                )
            self.set_error_msg(
                f'{EFFECT_NAME}: wrong file selected — set Script File'
            )
            return

        # location/SCRIPT_NAME/THIS_SCRIPT — up one level from the file gives
        # the SCRIPT_NAME folder, up one more gives the location itself.
        script_dir = os.path.dirname(os.path.abspath(selected))
        location   = os.path.dirname(script_dir)

        if not is_valid_location(location):
            if selected != getattr(self, '_locate_rejected', None):
                self._locate_rejected = selected
                self.set_dialog_msg(
                    f'{EFFECT_NAME}\n\n'
                    f'{selected}\n\nis named correctly, but its folder is '
                    f'missing weights or effect files. Select the copy of '
                    f"'{THIS_SCRIPT}' that belongs to this node."
                )
            self.set_error_msg(
                f'{EFFECT_NAME}: incomplete installation — set Script File'
            )
            return

        if getattr(self, '_locate_located', None) == location:
            return

        ok, detail = write_location_to_self(location)
        self._locate_located = location
        if ok:
            printd(f'Recorded location {location}')
            self.set_dialog_msg(
                f'{EFFECT_NAME}\n\n'
                f'Location set to:\n{location}\n\n'
                f'This node must now be DELETED and re-created for the '
                f'change to take effect. Flame runs a copy of the script '
                f'taken when the node was created, so the running copy '
                f'still holds the old path.'
            )
        else:
            printe(f'Could not update script: {detail}')
            self.set_dialog_msg(
                f'{EFFECT_NAME}\n\n'
                f'Found the file at:\n{location}\n\n'
                f'but could not update it: {detail}\n\n'
                f'Set SCRIPT_LOCATION by hand in {THIS_SCRIPT}.'
            )

    # ── Normal UI ──────────────────────────────────────────────────────────────

    def setup_ui(self):
        if RESOLVED_LOCATION is None:
            self.setup_locate_ui()
            return

        model_popup = pybox.create_popup(
            MODEL_UI_ELEMENT,
            items=[m[0] for m in MODELS],
            page=0, col=0, row=0,
            tooltip="<b>Model</b>\nDisplays the model to use. Editable."
        )
        self.add_render_elements(model_popup)

        scale_popup = pybox.create_popup(
            SCALE_UI_ELEMENT,
            items=[str(s) for s in SCALE_VALUES],
            value=SCALE_VALUES.index(16),
            default=SCALE_VALUES.index(16),
            page=0, col=0, row=1,
            tooltip="<b>Scale</b>\nInitial scale factor. Editable."
        )
        self.add_render_elements(scale_popup)

        detail_value = pybox.create_float_numeric(
            DETAIL_UI_ELEMENT,
            value=0.5, default=0.5, min=0.0, max=1.0, inc=0.01,
            channel_name=DETAIL_UI_ELEMENT,
            page=0, col=0, row=2,
            tooltip="<b>Detail</b>\nLevel of detail in the estimated flow. Editable."
        )
        self.add_render_elements(detail_value)

        smooth_value = pybox.create_float_numeric(
            SMOOTH_UI_ELEMENT,
            value=0.0, default=0.0, min=0.0, max=1.0, inc=0.01,
            channel_name=SMOOTH_UI_ELEMENT,
            page=0, col=0, row=3,
            tooltip="<b>Smooth</b>\nSmooths the resulting UV field. 0 is no smoothing. Editable."
        )
        self.add_render_elements(smooth_value)

        page = pybox.create_page("Main", "Warp")
        self.set_ui_pages(page)

        self.set_state_id("execute")
        self.execute()

    # ── Failure handling ───────────────────────────────────────────────────────

    def front_colour_space(self):
        """
        Flame's colour space tag for the Front input, or None.

        Reads the live per-socket value when processing; the pybox API only
        exposes get_socket_colour_space during a process pass. Falls back to
        the project working colour space, and finally to None so the daemon
        leaves the header alone rather than writing a wrong tag.
        """
        try:
            if self.is_processing():
                sock = self.get_process_in_socket(0)
                if self.is_socket_active(sock):
                    cs = self.get_socket_colour_space(sock)
                    if cs:
                        return cs
        except Exception as e:
            printd(f'Could not read socket colour space: {e}')
        try:
            return self.get_colour_space() or None
        except Exception:
            return None

    def request_error_frame(self):
        """Ask the daemon to fill both UV outputs with a zero-displacement placeholder."""
        try:
            response = send_recv({
                'type': 'command',
                'data': 'error_frame_warp',
                'input0': self.get_in_socket_path(0),
                'input1': self.get_in_socket_path(1),
                'uv0':    self.get_out_socket_path(0),
                'uv1':    self.get_out_socket_path(1),
                'colour_space': self.front_colour_space(),
            })
            printd(f'Error frame response: {response}')
        except Exception as e:
            printe(f'Could not write error frame: {e}')

    def fail(self, signature, message):
        """
        Record the failure against the current settings, flag the frame in the
        viewport, and raise the dialog exactly once.

        The dialog is only ever set here — on a *newly seen* failure signature.
        Once the signature is persisted, subsequent dispatches take the
        already-failed branch in execute() and leave the dialog cleared, so
        closing it cannot respawn it.
        """
        printe(f'{message}')
        printd('fail() reached — setting dialog message NOW')
        write_state({'signature': signature, 'message': message})
        self.request_error_frame()
        self.set_error_msg(f'{EFFECT_NAME}: {message}')
        self.set_dialog_msg(f'{EFFECT_NAME}\n\n{message}')

    # ── Main ───────────────────────────────────────────────────────────────────

    def execute(self):
        # The message block round-trips through the JSON payload, so a dialog
        # set on a previous dispatch would be re-written by write_to_disk and
        # shown again. Clear it first — only fail() below may set it again.
        printd(f'state_id       = {getattr(self, "state_id", "<missing>")!r}')
        printd(f'has message    = {hasattr(self, "message")}')
        printd(f'message before = {getattr(self, "message", "<missing>")!r}')
        printd(f'state file     = {STATE_PATH} exists={os.path.exists(STATE_PATH)}')
        printd(f'state contents = {read_state()!r}')
        printd(f'active out sock= '
               f'{self.get_process_output_socket() if self.is_processing() else "not processing"}')

        self.set_dialog_msg("")
        self.set_error_msg("")

        try:
            if RESOLVED_LOCATION is None:
                self._locate()
            else:
                self._run()
        finally:
            printd(f'message after  = {getattr(self, "message", "<missing>")!r}')
            printd(f'dialog is set  = {bool(self.get_dialog_msg())}')

    def _run(self):
        model_index = self.get_render_element_value(MODEL_UI_ELEMENT)
        model_path  = MODELS[model_index][1]
        scale_index = self.get_render_element_value(SCALE_UI_ELEMENT)
        scale_value = SCALE_VALUES[scale_index]
        detail_value = float(self.get_render_element_value(DETAIL_UI_ELEMENT))
        smooth_value = float(self.get_render_element_value(SMOOTH_UI_ELEMENT))

        # Everything the user can change. If a run fails, we refuse to retry
        # until one of these differs from the settings that failed.
        signature = json.dumps({
            'model': model_path,
            'scale': scale_value,
            'detail': detail_value,
            'smooth': smooth_value,
        }, sort_keys=True)

        state = read_state()
        printd(f'signature now  = {signature}')
        printd(f'signature saved= {state.get("signature")!r}')
        if state.get('signature') == signature:
            printd('Settings unchanged since last failure — not retrying (no dialog).')
            self.request_error_frame()
            self.set_error_msg(
                f"{EFFECT_NAME}: {state.get('message', 'previous error')} "
                f"(change a setting to retry)"
            )
            return

        try:
            # Ask the daemon which model it currently has loaded
            status = send_recv({'type': 'command', 'data': 'status'})
            printd(f"Daemon status: {status}")

            # Membership, not equality: the daemon keeps several checkpoints
            # resident so different node types can coexist without evicting
            # each other every frame.
            if model_path not in status.get('loaded_models', []):
                printd(f"Loading model: {os.path.basename(model_path)}")
                response = send_recv({
                    'type': 'command',
                    'data': 'load_model',
                    'weight_path': model_path,
                })
                printd(f"Load response: {response}")
                if response.get('status') == 'error':
                    self.fail(signature, f"Error loading model: {response.get('message')}")
                    return
            else:
                printd(f"Model already loaded: {os.path.basename(model_path)}")

            response = send_recv({
                'type': 'command',
                'data': 'process_warp',
                'weight_path': model_path,
                'input0': self.get_in_socket_path(0),
                'input1': self.get_in_socket_path(1),
                'uv0':    self.get_out_socket_path(0),
                'uv1':    self.get_out_socket_path(1),
                'scale':     scale_value,
                'detail':    detail_value,
                'smooth':    smooth_value,
                'colour_space': self.front_colour_space(),
            })
            printd(f"Process response: {response}")

            if response.get('status') == 'error':
                self.fail(signature, f"Error processing: {response.get('message')}")
                return

        except Exception as e:
            # Daemon unreachable, socket timeout, malformed response, etc.
            self.fail(signature, f"Effect daemon error: {e}")
            return

        # Success — allow future retries again.
        clear_state()

    def teardown(self):
        try:
            send_recv({'type': 'exit'})
        except Exception:
            pass


def _main(argv):
    p = Warp(argv[0])
    p.dispatch()
    p.write_to_disk(argv[0])

if __name__ == "__main__":
    _main(sys.argv[1:])