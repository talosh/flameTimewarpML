"""
Pybox setup for Warp / Unwarp
"""

import os
import sys
import time
import socket
import subprocess
import json
import pybox_v1 as pybox

from pathlib import Path

EFFECT_NAME = 'Warp / Unwarp'
IMAGE_FORMAT = "exr"

MODEL_UI_ELEMENT = "Model"
SCALE_UI_ELEMENT   = "Scale"

SCRIPT_LOCATION = '/pybox_effect'
SCRIPT_NAME     = 'warp.v001'
SOCKET_PATH     = '/dev/shm/warp_effect.sock'

def printc(message=None):
    print(f'{SCRIPT_NAME}: {message}')

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

    printc(f"Starting {EFFECT_NAME} effect daemon...")

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
            printc("Daemon is up.")
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

    # printc(f"send_recv {msg_dict} command...")
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
        ensure_daemon_running()

        printc("Pinging effect daemon...")
        response = send_recv({'type': 'command', 'data': 'ping'})
        printc(f"Response: {response}")

        self.set_img_format(IMAGE_FORMAT)
        ext = self.get_img_format()

        tempdir = Path('/dev/shm')

        self.set_in_socket(0, "Front", str(tempdir / f"input0.{ext}"))
        self.set_in_socket(1, "Front", str(tempdir / f"input1.{ext}"))
        self.remove_in_socket(2)

        self.set_out_socket(0, "Result", str(tempdir / f"result0.{ext}"))
        self.set_out_socket(1, "Result", str(tempdir / f"result1.{ext}"))

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

        scale_value = pybox.create_float_numeric(
            SCALE_UI_ELEMENT,
            value=32, min=1, max=64, inc=0.1,
            channel_name=SCALE_UI_ELEMENT,
            page=0, col=0, row=1,
            tooltip="<b>Mix Field</b>\nPower of the effect. Editable."
        )
        self.add_render_elements(scale_value)

        page = pybox.create_page("Main", "Super Res")
        self.set_ui_pages(page)

        self.set_state_id("execute")
        self.execute()

    def execute(self):
        scale_value = self.get_render_element_value(SCALE_UI_ELEMENT)
        model_index = self.get_render_element_value(MODEL_UI_ELEMENT)
        model_path  = MODELS[model_index][1]

        # Ask the daemon which model it currently has loaded
        status = send_recv({'type': 'command', 'data': 'status'})
        printc(f"Daemon status: {status}")

        if status.get('loaded_model') != model_path:
            printc(f"Loading model: {os.path.basename(model_path)}")
            response = send_recv({
                'type': 'command',
                'data': 'load_model',
                'weight_path': model_path,
            })
            printc(f"Load response: {response}")
            if response.get('status') == 'error':
                self.set_dialog_msg(f"Error loading model: {response.get('message')}")
                return
        else:
            printc(f"Model already loaded: {os.path.basename(model_path)}")

        response = send_recv({
            'type': 'command',
            'data': 'process',
            'input0':  self.get_in_socket_path(0),
            'input1':  self.get_in_socket_path(1),
            'result0': self.get_out_socket_path(0),
            'result1': self.get_out_socket_path(1),
            'scale':   scale_value,
        })
        printc(f"Process response: {response}")

        if response.get('status') == 'error':
            self.set_dialog_msg(f"Error processing: {response.get('message')}")

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