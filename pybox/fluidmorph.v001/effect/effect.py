import os
import sys
import json
import socket
import shutil
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


def matching_spec(image_data, template=None):
    """
    Build a spec that always agrees with the array being written.

    Reusing an input spec is what produces truncated EXRs: if the header
    declares more scanlines or channels than the array actually holds,
    Flame reports "Scan line 0 is missing" and re-requests the frame forever.
    """
    height, width, nchannels = image_data.shape

    if (template is not None
            and template.width == width
            and template.height == height
            and template.nchannels == nchannels):
        return template

    fmt = template.format if template is not None else oiio.TypeDesc(oiio.FLOAT)
    return oiio.ImageSpec(width, height, nchannels, fmt)


def _tmp_path_for(dst_path):
    """
    Staging path alongside the destination. Keeps the real extension —
    OpenImageIO selects its writer from the extension, so a name like
    "result0.exr.tmp" has no format writer and fails outright.
    """
    directory, filename = os.path.split(dst_path)
    stem, ext = os.path.splitext(filename)
    return os.path.join(directory, f'.{stem}.{os.getpid()}.tmp{ext}')



def write_image_file(file_path, image_data, image_spec=None):
    """
    Write atomically via a temp file + rename, so Flame can never observe a
    partially written frame. Raises on failure instead of leaving a stub.
    """
    image_data = np.ascontiguousarray(image_data)
    spec = matching_spec(image_data, image_spec)

    tmp_path = _tmp_path_for(file_path)

    try:
        out = oiio.ImageOutput.create(tmp_path)
        if out is None:
            raise RuntimeError(
                f'Could not create output for {tmp_path}: {oiio.geterror()}'
            )

        if not out.open(tmp_path, spec):
            raise RuntimeError(f'Could not open {tmp_path}: {out.geterror()}')

        try:
            if not out.write_image(image_data):
                raise RuntimeError(f'Could not write {tmp_path}: {out.geterror()}')
        finally:
            out.close()

        # Atomic on the same filesystem — Flame sees either the old file or
        # the complete new one, never a half-written one.
        os.replace(tmp_path, file_path)

    except Exception:
        # Never leave a stray temp file behind for Flame to trip over.
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        raise


# ── Matte output ───────────────────────────────────────────────────────────────

# None = match the result's channel count. This matters for more than the
# channel count itself: when the matte array matches the result's spec exactly,
# the template spec is reused verbatim, so the matte inherits the result's
# display window, data window and pixel aspect ratio. A freshly built spec
# loses all of that, and Flame then reads the matte with different geometry
# than the result — which looks like a magnified, offset matte.
# Set to 1 for a genuinely single-channel matte, or 3 to force grayscale RGB.
MATTE_CHANNELS      = None
MATTE_CHANNEL_NAMES = ('A',)     # only used when writing a 1-channel matte


def write_matte_file(file_path, matte, template_spec=None):
    """
    Write a matte as its own image.

    Only the first channel of `matte` is used; it is replicated out to
    whatever channel count is being written.
    """
    matte = np.asarray(matte, dtype=np.float32)
    if matte.ndim == 2:
        matte = matte[..., None]
    matte = matte[..., :1]

    if MATTE_CHANNELS is None:
        nchannels = template_spec.nchannels if template_spec is not None else 1
    else:
        nchannels = MATTE_CHANNELS
    nchannels = max(int(nchannels), 1)

    if nchannels > 1:
        matte = np.repeat(matte, nchannels, axis=2)

    matte = np.ascontiguousarray(matte)
    height, width = matte.shape[0], matte.shape[1]

    if (template_spec is not None
            and template_spec.width == width
            and template_spec.height == height
            and template_spec.nchannels == nchannels):
        # Identical to the result's spec — reuse it so every window and aspect
        # attribute matches the result exactly.
        spec = template_spec
    else:
        fmt = template_spec.format if template_spec is not None else oiio.TypeDesc(oiio.FLOAT)
        spec = oiio.ImageSpec(width, height, nchannels, fmt)
        if nchannels == len(MATTE_CHANNEL_NAMES):
            spec.channelnames = list(MATTE_CHANNEL_NAMES)

    write_image_file(file_path, matte, spec)


def pad_to_multiple(tensor, mult):
    """
    Pad an NCHW tensor so H and W divide evenly by `mult`.

    The network downsamples by `scale` then by 4 more in conv0, and rebuilds
    with fixed x4 / xscale upsamples. Unless H and W divide by scale*4 the
    round trip loses rows, and the result no longer matches the input
    resolution. Returns the padded tensor and the original (h, w).
    """
    height, width = tensor.shape[2], tensor.shape[3]
    pad_h = (mult - height % mult) % mult
    pad_w = (mult - width % mult) % mult

    if pad_h == 0 and pad_w == 0:
        return tensor, (height, width)

    try:
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')
    except Exception:
        # replicate/reflect refuse pads larger than the dimension itself
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

    return padded, (height, width)


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


def write_error_frame(result_file, matte_file, *input_files):
    """
    Copy the first readable input to the output with a semi-transparent
    saltire superimposed. Used instead of a modal dialog so a failure is
    visible in the viewport without blocking the user.

    Flame re-requests the frame indefinitely if an output socket has no valid
    file, so this must succeed whenever any input is readable — and it must
    fill the matte socket too, not just the result.

    Returns True if a frame was written.
    """
    if not result_file:
        return False

    for input_file in input_files:
        if not input_file or not os.path.isfile(input_file):
            continue

        try:
            img = read_image_file(input_file)

            if img['image_data'] is None:
                raise ValueError('no pixel data')

            data = np.array(img['image_data'], dtype=np.float32, copy=True)
            if data.ndim != 3 or data.shape[2] < 3:
                raise ValueError(f'unexpected shape {data.shape}')

            height, width = data.shape[0], data.shape[1]
            overlay = saltire_overlay(height, width)

            # Only touch RGB — leave alpha and any extra channels untouched.
            data[..., :3] = data[..., :3] * (1.0 - OVERLAY_ALPHA) + overlay * OVERLAY_ALPHA

            write_image_file(result_file, data, img['spec'])
            write_error_matte(matte_file, height, width, img['spec'])
            return True

        except Exception:
            # Last resort: copy the input verbatim. It is a file Flame itself
            # just wrote, so it is guaranteed readable — an unflagged frame
            # beats no output at all, which makes Flame re-request forever.
            try:
                shutil.copyfile(input_file, result_file)
                try:
                    header = read_image_file(input_file, header_only=True)
                    spec = header['spec']
                    write_error_matte(matte_file, spec.height, spec.width, spec)
                except Exception:
                    pass
                return True
            except Exception:
                continue

    return False


def write_error_matte(matte_file, height, width, template_spec=None):
    """Solid opaque matte to accompany an error frame. Never raises."""
    if not matte_file:
        return
    try:
        write_matte_file(
            matte_file,
            np.ones((height, width, 1), dtype=np.float32),
            template_spec,
        )
    except Exception as e:
        print(f'[effect] Could not write error matte: {e}', flush=True)


# ── Iterational ratio solving ──────────────────────────────────────────────────

# The model is only trained to be accurate near these ratios; asking it for a
# small fractional step directly gives imprecise results. Any target ratio is
# reached instead by repeated bracketing that only ever requests these.
BASE_RATIOS     = (0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75)
RATIO_TOLERANCE = 0.001
MAX_RATIO_STEPS = 32


def solve_ratio_iteratively(net, img0, img1, target, scale_arg, bidirectional,
                            tolerance=RATIO_TOLERANCE, max_steps=MAX_RATIO_STEPS):
    """
    Reach an arbitrary target ratio using only well-trained base ratios.

    A bracket [lo, hi] is kept around the target, with `a` and `b` holding the
    frames at those two times. Each inference lands somewhere inside the
    bracket; whichever side the target falls on becomes the new bracket, and
    the frame just synthesised replaces the endpoint it superseded. So each
    pass hands the model a pair of frames that are progressively closer
    together, always asking for a ratio it handles well.

    Converges within 7 inferences for any target at 0.001 tolerance.

    Returns (frame, conf, steps_taken).
    """
    if target <= tolerance:
        return img0, torch.ones_like(img0[:, :1]), 0
    if target >= 1.0 - tolerance:
        return img1, torch.ones_like(img1[:, :1]), 0

    lo, hi = 0.0, 1.0
    a, b = img0, img1
    result, conf = None, None
    steps = 0

    while steps < max_steps:
        width = hi - lo
        ratio = min(BASE_RATIOS, key=lambda r: abs(r - (target - lo) / width))

        result, conf = net(a, b, timestep=ratio,
                           scale=(scale_arg, 1), bidirectional=bidirectional)
        steps += 1

        landed = lo + ratio * width

        # conf belongs to the final inference only — intermediates describe
        # brackets that were discarded.
        if abs(target - landed) <= tolerance:
            return result, conf, steps

        if target < landed:
            b, hi = result, landed
        else:
            a, lo = result, landed

    return result, conf, steps


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
                written = write_error_frame(
                    msg.get('result0'), msg.get('matte0'),
                    msg.get('input0'), msg.get('input1')
                )
                if written:
                    return {'status': 'ok', 'message': 'error frame written'}
                return {'status': 'ok', 'message': 'no usable input for error frame'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}

        if msg.get('data') == 'process':
            input_file0  = msg.get('input0')
            input_file1  = msg.get('input1')
            result_file0 = msg.get('result0')
            matte_file0  = msg.get('matte0')
            ratio        = float(msg.get('ratio', 0.5))

            if model is None:
                try:
                    write_error_frame(result_file0, matte_file0, input_file0, input_file1)
                except Exception:
                    pass
                return {'status': 'error', 'message': 'Model not loaded. Send load_model first.'}

            try:
                if not os.path.isfile(input_file0):
                    return {'status': 'ok', 'message': 'no input connected (input0)'}
                if not os.path.isfile(input_file1):
                    return {'status': 'ok', 'message': 'no input connected (input1)'}

                scale_arg = int(msg.get('scale', 8))

                with torch.no_grad():
                    img      = read_image_file(input_file0)
                    out_spec = img['spec']
                    img0     = torch.from_numpy(img['image_data'])
                    img0     = img0.to(device, dtype=torch.float32, non_blocking=True).permute(2, 0, 1).unsqueeze(0)

                    img      = read_image_file(input_file1)
                    img1     = torch.from_numpy(img['image_data'])
                    img1     = img1.to(device, dtype=torch.float32, non_blocking=True).permute(2, 0, 1).unsqueeze(0)

                    if img0.shape != img1.shape:
                        raise ValueError(
                            f'Input resolutions differ: {tuple(img0.shape[2:])} vs {tuple(img1.shape[2:])}'
                        )

                    # The net downsamples by `scale` then by 4 again in conv0.
                    # Without this the output comes back smaller than the input
                    # and the written EXR is truncated.
                    mult = max(scale_arg * 4, 4)
                    img0, (orig_h, orig_w) = pad_to_multiple(img0, mult)
                    img1, _                = pad_to_multiple(img1, mult)

                    if device.type == 'cuda':
                        torch.cuda.synchronize(device=device)
                        torch.cuda.empty_cache()
                    model.eval()

                    bidirectional = bool(msg.get('bidirectional', False))
                    iterational   = bool(msg.get('iterational', False))

                    if iterational:
                        # Padding happened once, above — every intermediate
                        # frame stays in padded space and only the final result
                        # is cropped, so nothing is re-padded per step.
                        res_fwd, res_conf, steps = solve_ratio_iteratively(
                            model, img0, img1, ratio, scale_arg, bidirectional
                        )
                    else:
                        steps = 1
                        res_fwd, res_conf = model(
                            img0,
                            img1,
                            timestep = ratio,
                            scale=(scale_arg, 1),
                            bidirectional=bidirectional,
                        )

                    # Crop the padding back off.
                    res_fwd  = res_fwd[:, :, :orig_h, :orig_w]
                    res_conf = res_conf[:, :, :orig_h, :orig_w]

                    # Last line of defence: the written array must match the
                    # source resolution exactly or Flame gets a broken frame.
                    if res_fwd.shape[2] != orig_h or res_fwd.shape[3] != orig_w:
                        res_fwd = F.interpolate(
                            res_fwd, size=(orig_h, orig_w),
                            mode='bilinear', align_corners=False
                        )
                    if res_conf.shape[2] != orig_h or res_conf.shape[3] != orig_w:
                        res_conf = F.interpolate(
                            res_conf, size=(orig_h, orig_w),
                            mode='bilinear', align_corners=False
                        )

                    res_fwd  = res_fwd.squeeze(0).permute(1, 2, 0).cpu()
                    res_conf = res_conf.squeeze(0).permute(1, 2, 0).cpu()

                    write_image_file(result_file0, res_fwd.numpy(force=True), out_spec)

                    if matte_file0:
                        write_matte_file(
                            matte_file0, res_conf.numpy(force=True), out_spec
                        )

                return {
                    'status': 'ok',
                    'message': (
                        f'{tuple(res_fwd.shape)} in {steps} inference(s); '
                        f'result spec {out_spec.width}x{out_spec.height}x{out_spec.nchannels}, '
                        f'matte {tuple(res_conf.shape)}'
                    ),
                }

            except Exception as e:
                try:
                    write_error_frame(msg.get('result0'), msg.get('matte0'),
                                      msg.get('input0'), msg.get('input1'))
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