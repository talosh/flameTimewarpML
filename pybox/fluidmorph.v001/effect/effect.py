import os
import sys
import json
import socket
import numpy as np
import torch
import torch.nn.functional as F
import OpenImageIO as oiio

from pprint import pformat

SOCKET_PATH = '/dev/shm/fluidmorph_effect.sock'

# ── Global state ───────────────────────────────────────────────────────────────
model              = None
device             = None
current_weight_path = None
# ──────────────────────────────────────────────────────────────────────────────


def create_torch_device():
    if torch.cuda.is_available():
        device_name = 'cuda'
    elif torch.backends.mps.is_available():
        device_name = 'mps'
    else:
        device_name = 'cpu'
    return torch.device(device_name)


def find_and_import_model(models_dir='models', base_name=None, model_name=None, model_file=None):
    import re
    import importlib

    models_abs_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            models_dir
        )
    )

    if model_file:
        if not model_file.endswith('.py'):
            raise ValueError(f"model_file must end with .py: {model_file}")

        full_path = os.path.join(models_abs_path, model_file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")

        module_name = model_file[:-3]
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from: {full_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'Model'):
            raise AttributeError(f"Module {full_path} has no 'Model' class")

        return getattr(module, 'Model')

    if not model_name and not base_name:
        raise ValueError("Either base_name, model_name, or model_file must be provided")

    try:
        files = os.listdir(models_abs_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Models directory not found: {models_abs_path}") from e

    py_files = [f for f in files if f.endswith('.py')]
    if not py_files:
        raise FileNotFoundError(f"No Python files found in: {models_abs_path}")

    selected_file = None

    if model_name:
        target_file = f"{model_name}.py"
        if target_file in py_files:
            selected_file = target_file
        else:
            raise FileNotFoundError(
                f"Model '{model_name}.py' not found in {models_abs_path}. "
                f"Available: {', '.join(py_files)}"
            )

    elif base_name:
        regex_pattern = rf"{re.escape(base_name)}_v(\d+)\.py"
        versions = []
        for f in py_files:
            match = re.match(regex_pattern, f)
            if match:
                versions.append((f, int(match.group(1))))

        if not versions:
            raise FileNotFoundError(
                f"No models matching '{base_name}_v*.py' in {models_abs_path}. "
                f"Available: {', '.join(py_files)}"
            )

        selected_file = sorted(versions, key=lambda x: x[1], reverse=True)[0][0]

    if selected_file:
        module_name = selected_file[:-3]
        module_path = f"{models_dir}.{module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import {module_path}: {e}") from e

        if not hasattr(module, 'Model'):
            raise AttributeError(
                f"Module {module_path} has no 'Model' class. "
                f"Available: {', '.join(dir(module))}"
            )

        return getattr(module, 'Model')

    raise FileNotFoundError(f"Failed to select a model file from {models_abs_path}")


def read_image_file(file_path, header_only=False):
    result = {'spec': None, 'image_data': None}
    inp = oiio.ImageInput.open(file_path)
    if inp:
        spec = inp.spec()
        result['spec'] = spec
        if not header_only:
            result['image_data'] = inp.read_image(0, 0, 0, spec.nchannels)
        inp.close()
    return result


def write_image_file(file_path, image_data, image_spec):
    out = oiio.ImageOutput.create(file_path)
    if out:
        out.open(file_path, image_spec)
        out.write_image(image_data)
        out.close()


# ── Error frame ────────────────────────────────────────────────────────────────

# Approximate linear-light equivalents of the Scottish flag colours.
SALTIRE_BLUE  = (0.0, 0.105, 0.34)
SALTIRE_WHITE = (1.0, 1.0, 1.0)

OVERLAY_ALPHA = 0.5     # how strongly the flag is mixed over the input
SALTIRE_WIDTH = 0.12    # thickness of the cross arms, normalised


def saltire_overlay(height, width):
    """Build an (h, w, 3) float32 image of a Scottish saltire."""
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    xs = np.linspace(0.0, 1.0, width,  dtype=np.float32).reshape(1, width)

    # Distance to each diagonal, in normalised units.
    d1 = np.abs(ys - xs) / np.sqrt(2.0)
    d2 = np.abs(ys + xs - 1.0) / np.sqrt(2.0)
    d  = np.minimum(d1, d2)

    cross = (d < (SALTIRE_WIDTH / 2.0)).astype(np.float32)[..., None]

    blue  = np.array(SALTIRE_BLUE,  dtype=np.float32).reshape(1, 1, 3)
    white = np.array(SALTIRE_WHITE, dtype=np.float32).reshape(1, 1, 3)

    return blue * (1.0 - cross) + white * cross


def write_error_frame(input_file, result_file):
    """
    Copy the input frame to the output with a semi-transparent saltire
    superimposed. Used instead of a modal dialog so a failure is visible
    in the viewport without blocking the user.

    Returns True if a frame was written.
    """
    if not input_file or not result_file:
        return False
    if not os.path.isfile(input_file):
        return False

    img = read_image_file(input_file)
    if img['image_data'] is None:
        return False

    data = np.array(img['image_data'], dtype=np.float32, copy=True)
    if data.ndim != 3 or data.shape[2] < 3:
        return False

    height, width = data.shape[0], data.shape[1]
    overlay = saltire_overlay(height, width)

    # Only touch RGB — leave alpha and any extra channels untouched.
    data[..., :3] = data[..., :3] * (1.0 - OVERLAY_ALPHA) + overlay * OVERLAY_ALPHA

    write_image_file(result_file, np.ascontiguousarray(data), img['spec'])
    return True


class EMA:
    def __init__(self, model, decay=0.995):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


# ── Message handlers ───────────────────────────────────────────────────────────

def handle_message(msg):
    global model, device, current_weight_path

    if msg.get('type') == 'command':

        if msg.get('data') == 'ping':
            return {'status': 'ok', 'message': 'pong'}

        if msg.get('data') == 'status':
            return {
                'status': 'ok',
                'loaded_model': current_weight_path,
                'device': str(device) if device else None,
            }

        if msg.get('data') == 'load_model':
            try:
                device = create_torch_device()
                weight_path = msg.get('weight_path')
                checkpoint  = torch.load(weight_path, map_location=device)
                model_file  = checkpoint['model_info']['file']
                Net         = find_and_import_model(model_file=model_file)
                model_info  = Net.get_info()
                model       = Net().get_model()().to(device).eval()
                model.load_state_dict(checkpoint['flownet_state_dict'], strict=False)
                current_weight_path = weight_path
                return {'status': 'ok', 'model_info': f'{model_info}'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}

        if msg.get('data') == 'error_frame':
            try:
                written = write_error_frame(msg.get('input0'), msg.get('result0'))
                if written:
                    return {'status': 'ok', 'message': 'error frame written'}
                return {'status': 'ok', 'message': 'no usable input for error frame'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}

        if msg.get('data') == 'process':
            if model is None:
                try:
                    write_error_frame(msg.get('input0'), msg.get('result0'))
                except Exception:
                    pass
                return {'status': 'error', 'message': 'Model not loaded. Send load_model first.'}

            try:
                input_file0 = msg.get('input0')
                input_file1 = msg.get('input1')
                result_file0 = msg.get('result0')

                if not os.path.isfile(input_file0):
                    return {'status': 'ok', 'message': 'no input connected (input0)'}
                if not os.path.isfile(input_file1):
                    return {'status': 'ok', 'message': 'no input connected (input1)'}

                with torch.no_grad():
                    img      = read_image_file(input_file0)
                    img0     = torch.from_numpy(img['image_data'])
                    img0     = img0.to(device, dtype=torch.float32, non_blocking=True).permute(2, 0, 1).unsqueeze(0)

                    img      = read_image_file(input_file1)
                    img1     = torch.from_numpy(img['image_data'])
                    img1     = img1.to(device, dtype=torch.float32, non_blocking=True).permute(2, 0, 1).unsqueeze(0)

                    torch.cuda.synchronize(device=device)
                    torch.cuda.empty_cache()
                    model.eval()

                    res_fwd, res_inv = model(
                        img1,
                        img0,
                        scale=(msg.get('scale', 8.), 1)
                    )

                    res_fwd = res_fwd.squeeze(0).permute(1, 2, 0).cpu()

                    write_image_file(result_file0, np.ascontiguousarray(res_fwd.numpy(force=True)), img['spec'])

                return {'status': 'ok', 'message': f'{res_fwd.shape}'}

            except Exception as e:
                try:
                    write_error_frame(msg.get('input0'), msg.get('result0'))
                except Exception:
                    pass
                return {'status': 'error', 'message': str(e)}

    return {'status': 'error', 'message': f'Unknown message: {msg}'}


# ── Socket server ──────────────────────────────────────────────────────────────

INACTIVITY_TIMEOUT = 600

def run_server():
    # Remove stale socket if it exists
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    server.settimeout(INACTIVITY_TIMEOUT)

    print(f'[effect] Listening on {SOCKET_PATH}', flush=True)

    try:
        while True:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                print(f'[effect] No activity for {INACTIVITY_TIMEOUT}s, exiting', flush=True)
                break
            
            try:
                buf = b''
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        msg = json.loads(line)

                        if msg.get('type') == 'exit':
                            print('[effect] Exit requested, shutting down.', flush=True)
                            conn.close()
                            server.close()
                            os.unlink(SOCKET_PATH)
                            return

                        response = handle_message(msg)
                        conn.sendall((json.dumps(response) + '\n').encode())
            except Exception as e:
                try:
                    conn.sendall((json.dumps({'status': 'error', 'message': str(e)}) + '\n').encode())
                except Exception:
                    pass
            finally:
                conn.close()
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)


if __name__ == '__main__':
    run_server()