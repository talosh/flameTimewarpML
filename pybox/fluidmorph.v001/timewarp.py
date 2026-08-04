"""
Pybox setup for Timewarp
"""

import os
import sys
import time
import math
import socket
import subprocess
import json
import pybox_v1 as pybox

from pathlib import Path

# Set True for verbose per-dispatch tracing (daemon calls, state, message
# block contents). Errors always print regardless of this flag.
DEBUG = False

EFFECT_NAME = 'ML Timewarp'
IMAGE_FORMAT = "exr"

MODEL_UI_ELEMENT = "Model"
SCALE_UI_ELEMENT   = "Scale"
RATIO_UI_ELEMENT   = "Ratio"
BIDI_UI_ELEMENT    = "Bidirectional"
ITER_UI_ELEMENT    = "Iterational"

SCALE_VALUES = [64, 32, 16, 8, 4, 2, 1]

SCRIPT_LOCATION = '/mmfs1/vault/flame/pybox'
SCRIPT_NAME     = 'fluidmorph.v001'
SOCKET_PATH     = '/dev/shm/fluidmorph_effect.sock'
STATE_PATH      = '/dev/shm/fluidmorph_effect.state.json'


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


def write_state(state):
    """Persist the failure state. Never raises."""
    try:
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

def scan_weights():
    weights_abs_path = os.path.abspath(
        os.path.join(SCRIPT_LOCATION, SCRIPT_NAME, 'weights')
    )
    weights_files = sorted([
        os.path.abspath(os.path.join(weights_abs_path, f))
        for f in os.listdir(weights_abs_path)
        if f.endswith('.pth')
    ])
    if not weights_files:
        return [('None', 'None')]
    return [(os.path.splitext(os.path.basename(f))[0], f) for f in weights_files]

MODELS = scan_weights()

effect_python = os.path.join(
    os.path.abspath(SCRIPT_LOCATION),
    SCRIPT_NAME,
    'packages', 'miniconda', 'appenv', 'bin', 'python'
)

effect_script = os.path.join(
    os.path.abspath(SCRIPT_LOCATION),
    SCRIPT_NAME,
    'effect', 'effect.py'
)


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

class Fluidmorph(pybox.BaseClass):

    def initialize(self):
        ensure_daemon_running()

        # Failure state lives in /dev/shm and survives Flame restarts, so a
        # stale signature would suppress retries (and the dialog) forever.
        # A node load is an explicit fresh start.
        clear_state()

        printd("Pinging effect daemon...")
        response = send_recv({'type': 'command', 'data': 'ping'})
        printd(f"Response: {response}")

        self.set_img_format(IMAGE_FORMAT)
        ext = self.get_img_format()

        tempdir = Path('/dev/shm')

        self.set_in_socket(0, "Front", str(tempdir / f"input0.{ext}"))
        self.set_in_socket(1, "Front", str(tempdir / f"input1.{ext}"))
        self.remove_in_socket(2)

        self.remove_out_sockets()
        self.set_out_socket(0, "Result", str(tempdir / f"result0.{ext}"))
        self.set_out_socket(1, "OutMatte", str(tempdir / f"matte0.{ext}"))

        self.set_state_id("setup_ui")
        self.setup_ui()

    def setup_ui(self):
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

        ratio_value = pybox.create_float_numeric(
            RATIO_UI_ELEMENT,
            value=0.5, default=0.5, min=-1000000.0, max=1000000.0, inc=0.01,
            channel_name=RATIO_UI_ELEMENT,
            page=0, col=0, row=2,
            tooltip="<b>Ratio</b>\nBlend ratio between the two inputs. Values outside 0-1 wrap to their fractional part, so a Timewarp Timing channel can be linked directly. Editable."
        )
        self.add_render_elements(ratio_value)

        bidirectional_toggle = pybox.create_toggle_button(
            BIDI_UI_ELEMENT,
            value=False,
            default=False,
            page=0, col=0, row=3,
            tooltip="<b>Bidirectional</b>\nAverage each stage with a reverse pass. Doubles processing time."
        )
        self.add_render_elements(bidirectional_toggle)

        iterational_toggle = pybox.create_toggle_button(
            ITER_UI_ELEMENT,
            value=False,
            default=False,
            page=0, col=0, row=4,
            tooltip="<b>Iterational</b>\nEnable iterational refinement."
        )
        self.add_render_elements(iterational_toggle)

        page = pybox.create_page("Main", "Fluidmorph")
        self.set_ui_pages(page)

        self.set_state_id("execute")
        self.execute()

    # ── Failure handling ───────────────────────────────────────────────────────

    def request_error_frame(self):
        """Ask the daemon to write input0 + saltire to the output socket."""
        try:
            response = send_recv({
                'type': 'command',
                'data': 'error_frame',
                'input0':  self.get_in_socket_path(0),
                'input1':  self.get_in_socket_path(1),
                'result0': self.get_out_socket_path(0),
                'matte0':  self.get_out_socket_path(1),
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
            self._run()
        finally:
            printd(f'message after  = {getattr(self, "message", "<missing>")!r}')
            printd(f'dialog is set  = {bool(self.get_dialog_msg())}')

    def _run(self):
        scale_index = self.get_render_element_value(SCALE_UI_ELEMENT)
        scale_value = SCALE_VALUES[scale_index]

        # The field accepts any value so a Timewarp Timing channel can be
        # linked straight in. Only the fractional part is meaningful here, so
        # wrap it and write the wrapped value back so the UI shows what is
        # actually being used.
        raw_ratio   = float(self.get_render_element_value(RATIO_UI_ELEMENT))
        ratio_value = raw_ratio - math.floor(raw_ratio)
        if ratio_value != raw_ratio:
            printd(f'Ratio {raw_ratio} wrapped to {ratio_value}')
            self.set_render_element_value(RATIO_UI_ELEMENT, ratio_value)

        bidirectional_value = bool(self.get_render_element_value(BIDI_UI_ELEMENT))
        iterational_value = bool(self.get_render_element_value(ITER_UI_ELEMENT))
        model_index = self.get_render_element_value(MODEL_UI_ELEMENT)
        model_path  = MODELS[model_index][1]

        # Everything the user can change. If a run fails, we refuse to retry
        # until one of these differs from the settings that failed.
        signature = json.dumps({
            'model': model_path,
            'scale': scale_value,
            'ratio': ratio_value,
            'bidirectional': bidirectional_value,
            'iterational': iterational_value,
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

            if status.get('loaded_model') != model_path:
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
                'data': 'process',
                'input0':  self.get_in_socket_path(0),
                'input1':  self.get_in_socket_path(1),
                'result0': self.get_out_socket_path(0),
                'matte0':  self.get_out_socket_path(1),
                'scale':   scale_value,
                'ratio':   ratio_value,
                'bidirectional': bidirectional_value,
                'iterational':   iterational_value,
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
    p = Fluidmorph(argv[0])
    p.dispatch()
    p.write_to_disk(argv[0])

if __name__ == "__main__":
    _main(sys.argv[1:])