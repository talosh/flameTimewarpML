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
# Models are cached by weight path, not held in a single slot: several pybox
# nodes share this one daemon and may each want a different checkpoint (warp
# uses its own weights folder entirely). With a single slot, two node types in
# one Batch would evict each other's model on every frame — a full torch.load
# per dispatch, each way.
loaded_models      = {}    # weight_path -> model
model_lru          = []    # weight_paths, least-recently-used first
device             = None

# Cap on simultaneously resident models. Each one costs GPU memory, so this
# trades memory for avoiding reloads.
MAX_LOADED_MODELS  = 4
# ──────────────────────────────────────────────────────────────────────────────


def model_not_loaded_message(weight_path):
    """
    Why the model lookup failed, distinguishing the two very different causes.

    A message with no weight_path at all means the pybox script predates the
    per-path model cache and is talking to a newer daemon — the model may well
    be loaded, just not addressable. Saying "send load_model first" there is
    actively misleading, because the script already did.
    """
    if not weight_path:
        return ('No weight_path in the request — this pybox script is older '
                'than the daemon. Redeploy the pybox scripts, or restart the '
                'daemon to match. '
                f'Currently loaded: {[os.path.basename(p) for p in model_lru]}')
    return (f'Model not loaded: {os.path.basename(weight_path)}. '
            f'Send load_model first. '
            f'Currently loaded: {[os.path.basename(p) for p in model_lru]}')


def touch_model(weight_path):
    """Mark `weight_path` as most recently used."""
    if weight_path in model_lru:
        model_lru.remove(weight_path)
    model_lru.append(weight_path)


def get_loaded_model(weight_path):
    """The cached model for `weight_path`, or None. Marks it most recent."""
    if not weight_path:
        return None
    net = loaded_models.get(weight_path)
    if net is not None:
        touch_model(weight_path)
    return net


def evict_models_if_needed():
    """Drop least-recently-used models beyond MAX_LOADED_MODELS."""
    while len(model_lru) > MAX_LOADED_MODELS:
        oldest = model_lru.pop(0)
        loaded_models.pop(oldest, None)
        print(f'[effect] Evicted model {os.path.basename(oldest)}', flush=True)
        if device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()


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


# ── Single-channel auxiliary outputs (matte, conf) ─────────────────────────────

# None = match the template spec's channel count. This matters for more than
# the channel count itself: when the array matches the template exactly, the
# template spec's geometry is reused, so the file inherits the result's display
# window, data window and pixel aspect ratio. A freshly built spec loses all
# of that, and Flame then reads the file with different geometry than the
# result — which looks like a magnified, offset image.
# Set to 1 to force genuinely single-channel, or 3 to force grayscale RGB.
MATTE_CHANNELS      = None
MATTE_CHANNEL_NAMES = ('A',)     # only used when writing a 1-channel file

# Matte and conf are data, not colour. Reusing the result's spec verbatim also
# copies its colour metadata — the project's working colourspace tag and R,G,B
# channel names — so Flame colour-manages them, applying a transform to values
# that carry no colour at all. 'Raw' is OpenImageIO's convention for "leave
# these values alone".
DATA_COLORSPACE = 'Raw'

# Flame writes and reads its own colour space tag as this EXR header
# attribute, and honours it on read when Colour Management is set to
# "From File or Rules". OIIO's own oiio:ColorSpace is not what Flame looks at,
# which is why tagging that alone changed nothing.
FLAME_COLOURSPACE_ATTR = 'autodeskColorSpace'

# Flame's own tag for non-colour data. It retags the Result of any node that
# declares a matte output as "Matte" automatically, which is the behaviour
# being overridden here by writing the Front input's real colour space onto
# the Result.
FLAME_DATA_COLOURSPACE = 'Matte'


def apply_flame_colourspace(spec, colour_space):
    """
    Stamp Flame's colour space tag onto a spec. No-op if `colour_space` is
    falsy, so a caller that could not determine one leaves the header alone
    rather than writing something wrong.
    """
    if not colour_space:
        return spec
    try:
        spec.attribute(FLAME_COLOURSPACE_ATTR, str(colour_space))
    except Exception as e:
        print(f'[effect] Could not set {FLAME_COLOURSPACE_ATTR}: {e}', flush=True)
    return spec


def data_spec_from(template_spec, width, height, nchannels, channel_names=None):
    """
    A spec for a data output (matte, conf, UV): the template's geometry, but
    none of its colour metadata.

    The template is copied whenever its resolution matches, then the channel
    count is overridden — a differing channel count must not cost us the
    display window, data window and pixel aspect ratio, or Flame reads the
    file with different geometry than the result and it appears magnified and
    offset. That bites UV especially, which is 2-channel against a 3-channel
    result.

    The template must be COPIED, not returned: the caller shares that exact
    object with the result write, so mutating it in place would retag the
    result image too.
    """
    if (template_spec is not None
            and template_spec.width == width
            and template_spec.height == height):
        spec = oiio.ImageSpec(template_spec)
        spec.nchannels = nchannels
    else:
        fmt = template_spec.format if template_spec is not None else oiio.TypeDesc(oiio.FLOAT)
        spec = oiio.ImageSpec(width, height, nchannels, fmt)

    # Setting oiio:ColorSpace is not enough on its own: a spec read back from
    # an EXR also carries colorInteropID (and possibly chromaticities), which
    # re-declare the colourspace and win over it on the next write. They have
    # to be erased first or the data tag silently does not survive the round
    # trip.
    for attr in ('colorInteropID', 'chromaticities', 'oiio:ColorSpace',
                 FLAME_COLOURSPACE_ATTR):
        try:
            spec.erase_attribute(attr)
        except Exception:
            pass
    spec.attribute('oiio:ColorSpace', DATA_COLORSPACE)
    apply_flame_colourspace(spec, FLAME_DATA_COLOURSPACE)

    # Some readers key off the channel names rather than the colourspace tag,
    # so R,G,B would still invite a colour interpretation. UV overrides this
    # (see write_uv_file) — Flame will not read a UV socket otherwise.
    if channel_names and len(channel_names) == nchannels:
        spec.channelnames = list(channel_names)
    elif nchannels == 1:
        spec.channelnames = list(MATTE_CHANNEL_NAMES)
    else:
        spec.channelnames = [f'Y{i}' if i else 'Y' for i in range(nchannels)]

    return spec


def write_matte_file(file_path, matte, template_spec=None):
    """
    Write a single-channel-style image — used for both the Matte and the
    Conf outputs, whichever `matte` and `file_path` are passed for.

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

    spec = data_spec_from(template_spec, width, height, nchannels)
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


def write_error_frame(result_file, matte_file, conf_file, *input_files,
                      colour_space=None):
    """
    Copy the first readable input to the output with a semi-transparent
    saltire superimposed. Used instead of a modal dialog so a failure is
    visible in the viewport without blocking the user.

    Flame re-requests the frame indefinitely if an output socket has no valid
    file, so this must succeed whenever any input is readable — and it must
    fill every declared output socket, not just the result.

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

            apply_flame_colourspace(img['spec'], colour_space)
            write_image_file(result_file, data, img['spec'])
            write_error_matte(matte_file, height, width, img['spec'])
            write_error_conf(conf_file, height, width, img['spec'])
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
                    write_error_conf(conf_file, spec.height, spec.width, spec)
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


def write_error_conf(conf_file, height, width, template_spec=None):
    """
    Zero confidence to accompany an error frame. Never raises.

    Zero rather than the matte's opaque-one: the matte error convention says
    "treat this frame as fully covered", which is the safe default for
    downstream compositing; confidence has no equivalent safe-high default,
    so zero — "we have no confidence in this frame" — is the honest value.
    """
    if not conf_file:
        return
    try:
        write_matte_file(
            conf_file,
            np.zeros((height, width, 1), dtype=np.float32),
            template_spec,
        )
    except Exception as e:
        print(f'[effect] Could not write error conf: {e}', flush=True)


# ── Warp (UV) outputs ────────────────────────────────────────────────────────

def probe_input_geometry(*input_files):
    """
    (height, width, spec) of the first readable input file, or
    (None, None, None) if none are readable. Never raises.
    """
    for path in input_files:
        if not path or not os.path.isfile(path):
            continue
        try:
            header = read_image_file(path, header_only=True)
            spec = header['spec']
            return spec.height, spec.width, spec
        except Exception:
            continue
    return None, None, None


# Warp model constraints, both found by testing rather than stated anywhere:
#
#  - Its halving_steps() drops the max(...,1) guard the interpolation model
#    has, so a scale below 4 produces a zero stage divisor and a
#    ZeroDivisionError inside forward(). warp.py's dropdown stops at 4, and
#    this is the backstop.
#  - Each stage interpolates to round(dim/scale), pads to a multiple of 4,
#    then conv0 halves twice. ResConv's reflect padding needs at least 2
#    pixels, so round(dim/scale) must be >= 5. At scale 64 that means roughly
#    320 lines — a proxy render can easily fall under it.
WARP_MIN_SCALE      = 4
WARP_MIN_STAGE_SIZE = 5


def safe_warp_scale(scale, height, width):
    """
    Largest usable scale <= `scale` for this resolution.

    Halves until every stage stays above the model's minimum, so a proxy or
    degraded render quietly loses some scale rather than erroring out and
    flagging the frame.
    """
    scale = max(int(scale), WARP_MIN_SCALE)
    smallest = min(height, width)
    while scale > WARP_MIN_SCALE and round(smallest / scale) < WARP_MIN_STAGE_SIZE:
        scale //= 2
    return scale


# Flame rejects a bare 2-channel EXR on a UV socket ("Unsupported image
# channel format"), and will not accept data-style channel names there either.
# UV therefore goes out as 3 channels with a zero third and ordinary R,G,B
# names — the model's own commented-out result_fwd/result_rev lines pad the
# same way, so this matches what the architecture expects downstream.
UV_CHANNELS      = 3
UV_CHANNEL_NAMES = ('R', 'G', 'B')


def write_uv_file(file_path, uv, template_spec=None):
    """
    Write a UV image, padded out to UV_CHANNELS. Never raises.

    Goes through write_image_file directly rather than write_matte_file:
    UV carries two distinct values, not one replicated across channels.
    """
    if not file_path:
        return
    try:
        uv = np.asarray(uv, dtype=np.float32)
        if uv.ndim == 2:
            uv = uv[..., None]

        height, width = uv.shape[0], uv.shape[1]

        if uv.shape[2] < UV_CHANNELS:
            pad = np.zeros(
                (height, width, UV_CHANNELS - uv.shape[2]), dtype=np.float32
            )
            uv = np.concatenate([uv, pad], axis=2)
        else:
            uv = uv[..., :UV_CHANNELS]

        uv = np.ascontiguousarray(uv)
        write_image_file(
            file_path,
            uv,
            data_spec_from(
                template_spec, width, height, UV_CHANNELS,
                channel_names=UV_CHANNEL_NAMES,
            ),
        )
    except Exception as e:
        print(f'[effect] Could not write UV {file_path}: {e}', flush=True)


def write_zero_uv(file_path, height, width, template_spec=None):
    """
    All-zero 2-channel UV placeholder. Never raises.

    Written on any warp failure so Flame always has a readable frame on both
    UV sockets — without one it re-requests forever.
    """
    write_uv_file(
        file_path, np.zeros((height, width, 2), dtype=np.float32), template_spec
    )


# ── Iterational ratio solving ──────────────────────────────────────────────────

# The model is only trained to be accurate near these ratios; asking it for a
# small fractional step directly gives imprecise results. Any target ratio is
# reached instead by repeated bracketing that only ever requests these.
BASE_RATIOS     = (0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75)
RATIO_TOLERANCE = 0.001
MAX_RATIO_STEPS = 32


def _black_matte_like(reference, provided, other):
    """
    `provided` if both `provided` and `other` are present, else a black
    tensor matching `reference`'s batch/H/W — mirrors the model's own
    endpoint fallback, needed here too since the endpoint shortcuts below
    return before ever calling the model.
    """
    if provided is not None and other is not None:
        return provided
    channels = 1
    if provided is not None:
        channels = provided.shape[1]
    elif other is not None:
        channels = other.shape[1]
    return torch.zeros(
        (reference.shape[0], channels, reference.shape[2], reference.shape[3]),
        dtype=reference.dtype, device=reference.device,
    )


def solve_ratio_iteratively(net, img0, img1, target, scale_arg, bidirectional,
                            matte0=None, matte1=None,
                            tolerance=RATIO_TOLERANCE, max_steps=MAX_RATIO_STEPS):
    """
    Reach an arbitrary target ratio using only well-trained base ratios.

    A bracket [lo, hi] is kept around the target, with `a` and `b` holding the
    frames at those two times. Each inference lands somewhere inside the
    bracket; whichever side the target falls on becomes the new bracket, and
    the frame just synthesised replaces the endpoint it superseded. So each
    pass hands the model a pair of frames that are progressively closer
    together, always asking for a ratio it handles well.

    Matte is carried forward the same way: each step's synthesised matte
    (the matte belonging to the synthesised intermediate frame) replaces the
    matte on whichever side the frame it came from replaced, keeping the two
    aligned to the same synthesised timestep throughout the search.

    Converges within 7 inferences for any target at 0.001 tolerance.

    Returns (frame, conf, matte, steps_taken).
    """
    if target <= tolerance:
        return img0, torch.ones_like(img0[:, :1]), _black_matte_like(img0, matte0, matte1), 0
    if target >= 1.0 - tolerance:
        return img1, torch.ones_like(img1[:, :1]), _black_matte_like(img1, matte0, matte1), 0

    lo, hi = 0.0, 1.0
    a, b = img0, img1
    a_matte, b_matte = matte0, matte1
    result, conf, matte_out = None, None, None
    steps = 0

    while steps < max_steps:
        width = hi - lo
        ratio = min(BASE_RATIOS, key=lambda r: abs(r - (target - lo) / width))

        result, conf, matte_out = net(a, b, timestep=ratio,
                                      scale=(scale_arg, 1), bidirectional=bidirectional,
                                      matte0=a_matte, matte1=b_matte)
        steps += 1

        landed = lo + ratio * width

        # conf belongs to the final inference only — intermediates describe
        # brackets that were discarded. Matte is carried forward (see
        # docstring), so it stays correct at every step, not just the last.
        if abs(target - landed) <= tolerance:
            return result, conf, matte_out, steps

        if target < landed:
            b, b_matte, hi = result, matte_out, landed
        else:
            a, a_matte, lo = result, matte_out, landed

    return result, conf, matte_out, steps


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
    global device

    if msg.get('type') == 'command':

        if msg.get('data') == 'ping':
            return {'status': 'ok', 'message': 'pong'}

        if msg.get('data') == 'status':
            return {
                'status': 'ok',
                'loaded_models': list(model_lru),
                'device': str(device) if device else None,
            }

        if msg.get('data') == 'load_model':
            try:
                weight_path = msg.get('weight_path')
                if not weight_path:
                    return {'status': 'error', 'message': 'no weight_path given'}

                # Already resident — the whole point of the cache. Cheap
                # enough that the pybox side may call this unconditionally.
                if weight_path in loaded_models:
                    touch_model(weight_path)
                    return {'status': 'ok', 'message': 'already loaded'}

                device      = create_torch_device()
                checkpoint  = torch.load(weight_path, map_location=device)
                model_file  = checkpoint['model_info']['file']
                Net         = find_and_import_model(model_file=model_file)
                model_info  = Net.get_info()
                net         = Net().get_model()().to(device).eval()
                net.load_state_dict(checkpoint['flownet_state_dict'], strict=False)

                loaded_models[weight_path] = net
                touch_model(weight_path)
                evict_models_if_needed()
                return {'status': 'ok', 'model_info': f'{model_info}'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}

        # ── Warp (UV) — stub pending the model's actual architecture ────────────
        # load_model above is already generic (it imports whatever model file
        # the checkpoint's own model_info.file names), so it needs no changes
        # to load a warp checkpoint once one exists. Only real inference below
        # is missing.

        if msg.get('data') == 'error_frame_warp':
            try:
                height, width, spec = probe_input_geometry(
                    msg.get('input0'), msg.get('input1')
                )
                if height is None:
                    return {'status': 'ok', 'message': 'no usable input for warp error frame'}
                write_zero_uv(msg.get('uv0'), height, width, spec)
                write_zero_uv(msg.get('uv1'), height, width, spec)
                return {'status': 'ok', 'message': 'zero UV error frame written'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}

        if msg.get('data') == 'process_warp':
            uv0         = msg.get('uv0')
            uv1         = msg.get('uv1')
            input_file0 = msg.get('input0')
            input_file1 = msg.get('input1')

            def warp_fallback():
                """Zero UV on both sockets so Flame never loops on a missing frame."""
                height, width, spec = probe_input_geometry(input_file0, input_file1)
                if height is not None:
                    write_zero_uv(uv0, height, width, spec)
                    write_zero_uv(uv1, height, width, spec)

            net = get_loaded_model(msg.get('weight_path'))
            if net is None:
                warp_fallback()
                return {'status': 'error',
                        'message': model_not_loaded_message(msg.get('weight_path'))}

            try:
                if not input_file0 or not os.path.isfile(input_file0):
                    return {'status': 'ok', 'message': 'no input connected (input0)'}
                if not input_file1 or not os.path.isfile(input_file1):
                    return {'status': 'ok', 'message': 'no input connected (input1)'}

                scale_arg = int(msg.get('scale', 16))
                # 'detail' drives the model's FiLM conditioning (its forward
                # calls the argument `sharpness`). 'smooth' is the UV
                # smoothing amount, applied after the model returns.
                detail = float(msg.get('detail', 0.5))
                smooth = float(msg.get('smooth', 0.0))

                with torch.no_grad():
                    img      = read_image_file(input_file0)
                    out_spec = img['spec']
                    apply_flame_colourspace(out_spec, msg.get('colour_space'))
                    img0 = torch.from_numpy(img['image_data'])
                    img0 = img0.to(device, dtype=torch.float32, non_blocking=True).permute(2, 0, 1).unsqueeze(0)

                    img  = read_image_file(input_file1)
                    img1 = torch.from_numpy(img['image_data'])
                    img1 = img1.to(device, dtype=torch.float32, non_blocking=True).permute(2, 0, 1).unsqueeze(0)

                    if img0.shape != img1.shape:
                        raise ValueError(
                            f'Input resolutions differ: {tuple(img0.shape[2:])} vs {tuple(img1.shape[2:])}'
                        )

                    # Head's first conv takes exactly 3 channels, so drop any
                    # alpha or extra channels the incoming EXR carries.
                    img0 = img0[:, :3]
                    img1 = img1[:, :3]

                    orig_h, orig_w = img0.shape[2], img0.shape[3]

                    # Proxy/degraded renders can be too small for the
                    # requested scale; step down rather than fail the frame.
                    used_scale = safe_warp_scale(scale_arg, orig_h, orig_w)
                    if used_scale != scale_arg:
                        print(
                            f'[effect] scale {scale_arg} too high for '
                            f'{orig_w}x{orig_h}, using {used_scale}',
                            flush=True,
                        )

                    if device.type == 'cuda':
                        torch.cuda.synchronize(device=device)
                        torch.cuda.empty_cache()
                    net.eval()

                    # No pad_to_multiple here, unlike `process`. This model
                    # pads to its own maxdepth internally, crops back, then
                    # explicitly interpolates to (h, w) — so it already
                    # returns the input resolution. timestep is hardcoded to
                    # 0.5 inside its forward, which is why there is no Ratio
                    # control on this effect.
                    #
                    # Both UI controls go straight into forward(): Detail as
                    # its `sharpness` FiLM conditioning, Smooth as `smooth`.
                    # Smoothing happens inside the model, on the flow before
                    # flow_to_uv — nothing is applied to the UV out here.
                    uv_fwd, uv_rev = net(
                        img0, img1,
                        scale=(used_scale, 1),
                        sharpness=detail,
                        smooth=smooth,
                    )

                    # Belt and braces: a written array that disagrees with the
                    # source resolution is what produces truncated EXRs.
                    def fit(t):
                        if t.shape[2] != orig_h or t.shape[3] != orig_w:
                            return F.interpolate(
                                t, size=(orig_h, orig_w),
                                mode='bilinear', align_corners=False
                            )
                        return t

                    uv_fwd = fit(uv_fwd).squeeze(0).permute(1, 2, 0).cpu()
                    uv_rev = fit(uv_rev).squeeze(0).permute(1, 2, 0).cpu()

                    write_uv_file(uv0, uv_fwd.numpy(force=True), out_spec)
                    write_uv_file(uv1, uv_rev.numpy(force=True), out_spec)

                return {
                    'status': 'ok',
                    'message': (
                        f'uv {tuple(uv_fwd.shape)} scale={used_scale}'
                        + (f' (requested {scale_arg})' if used_scale != scale_arg else '')
                        + f' detail={detail} smooth={smooth}'
                    ),
                }

            except Exception as e:
                warp_fallback()
                return {'status': 'error', 'message': str(e)}

        if msg.get('data') == 'error_frame':
            try:
                written = write_error_frame(
                    msg.get('result0'), msg.get('matte0'), msg.get('conf0'),
                    msg.get('input0'), msg.get('input1'),
                    colour_space=msg.get('colour_space'),
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
            conf_file0   = msg.get('conf0')
            matte_in0    = msg.get('matte_in0')
            matte_in1    = msg.get('matte_in1')
            ratio        = float(msg.get('ratio', 0.5))

            # Which checkpoint this node wants — several may be resident at
            # once, so the message has to say, not the daemon assume.
            net = get_loaded_model(msg.get('weight_path'))
            if net is None:
                try:
                    write_error_frame(result_file0, matte_file0, conf_file0,
                                      input_file0, input_file1,
                                      colour_space=msg.get('colour_space'))
                except Exception:
                    pass
                return {'status': 'error',
                        'message': model_not_loaded_message(msg.get('weight_path'))}

            try:
                if not os.path.isfile(input_file0):
                    return {'status': 'ok', 'message': 'no input connected (input0)'}
                if not os.path.isfile(input_file1):
                    return {'status': 'ok', 'message': 'no input connected (input1)'}

                scale_arg = int(msg.get('scale', 8))

                with torch.no_grad():
                    img      = read_image_file(input_file0)
                    out_spec = img['spec']

                    # Flame retags the Result of any node that declares a
                    # matte output as "Matte". Writing the Front input's real
                    # colour space here overrides that on read (when Colour
                    # Management is "From File or Rules"). No-op when the
                    # pybox side could not determine one.
                    apply_flame_colourspace(out_spec, msg.get('colour_space'))

                    img0     = torch.from_numpy(img['image_data'])
                    img0     = img0.to(device, dtype=torch.float32, non_blocking=True).permute(2, 0, 1).unsqueeze(0)

                    img      = read_image_file(input_file1)
                    img1     = torch.from_numpy(img['image_data'])
                    img1     = img1.to(device, dtype=torch.float32, non_blocking=True).permute(2, 0, 1).unsqueeze(0)

                    if img0.shape != img1.shape:
                        raise ValueError(
                            f'Input resolutions differ: {tuple(img0.shape[2:])} vs {tuple(img1.shape[2:])}'
                        )

                    # Input mattes are optional — absent, unreadable, or a
                    # bad shape all just mean "no matte", not an error. The
                    # model treats either side missing as "output black".
                    def load_optional_matte(path, ref_shape):
                        if not path or not os.path.isfile(path):
                            return None
                        try:
                            m = read_image_file(path)
                            if m['image_data'] is None:
                                return None
                            t = torch.from_numpy(m['image_data'])
                            t = t.to(device, dtype=torch.float32, non_blocking=True).permute(2, 0, 1).unsqueeze(0)
                        except Exception as e:
                            print(f'[effect] Could not read matte {path}: {e}', flush=True)
                            return None
                        if t.shape[2:] != ref_shape[2:]:
                            print(
                                f'[effect] Matte {path} resolution '
                                f'{tuple(t.shape[2:])} does not match input '
                                f'{tuple(ref_shape[2:])} — ignoring.',
                                flush=True,
                            )
                            return None
                        return t

                    matte0 = load_optional_matte(matte_in0, img0.shape)
                    matte1 = load_optional_matte(matte_in1, img1.shape)

                    # The net downsamples by `scale` then by 4 again in conv0.
                    # Without this the output comes back smaller than the input
                    # and the written EXR is truncated.
                    mult = max(scale_arg * 4, 4)
                    img0, (orig_h, orig_w) = pad_to_multiple(img0, mult)
                    img1, _                = pad_to_multiple(img1, mult)
                    if matte0 is not None:
                        matte0, _ = pad_to_multiple(matte0, mult)
                    if matte1 is not None:
                        matte1, _ = pad_to_multiple(matte1, mult)

                    if device.type == 'cuda':
                        torch.cuda.synchronize(device=device)
                        torch.cuda.empty_cache()
                    net.eval()

                    bidirectional = bool(msg.get('bidirectional', False))
                    iterational   = bool(msg.get('iterational', False))

                    if iterational:
                        # Padding happened once, above — every intermediate
                        # frame stays in padded space and only the final result
                        # is cropped, so nothing is re-padded per step.
                        res_fwd, res_conf, res_matte, steps = solve_ratio_iteratively(
                            net, img0, img1, ratio, scale_arg, bidirectional,
                            matte0=matte0, matte1=matte1,
                        )
                    else:
                        steps = 1
                        res_fwd, res_conf, res_matte = net(
                            img0,
                            img1,
                            timestep = ratio,
                            scale=(scale_arg, 1),
                            bidirectional=bidirectional,
                            matte0=matte0,
                            matte1=matte1,
                        )

                    # Crop the padding back off.
                    res_fwd   = res_fwd[:, :, :orig_h, :orig_w]
                    res_conf  = res_conf[:, :, :orig_h, :orig_w]
                    res_matte = res_matte[:, :, :orig_h, :orig_w]

                    # Last line of defence: the written array must match the
                    # source resolution exactly or Flame gets a broken frame.
                    def fit(t):
                        if t.shape[2] != orig_h or t.shape[3] != orig_w:
                            return F.interpolate(
                                t, size=(orig_h, orig_w),
                                mode='bilinear', align_corners=False
                            )
                        return t

                    res_fwd   = fit(res_fwd)
                    res_conf  = fit(res_conf)
                    res_matte = fit(res_matte)

                    res_fwd   = res_fwd.squeeze(0).permute(1, 2, 0).cpu()
                    res_conf  = res_conf.squeeze(0).permute(1, 2, 0).cpu()
                    res_matte = res_matte.squeeze(0).permute(1, 2, 0).cpu()

                    write_image_file(result_file0, res_fwd.numpy(force=True), out_spec)

                    if matte_file0:
                        write_matte_file(
                            matte_file0, res_matte.numpy(force=True), out_spec
                        )
                    if conf_file0:
                        write_matte_file(
                            conf_file0, res_conf.numpy(force=True), out_spec
                        )

                return {
                    'status': 'ok',
                    'message': (
                        f'{tuple(res_fwd.shape)} in {steps} inference(s); '
                        f'result spec {out_spec.width}x{out_spec.height}x{out_spec.nchannels}, '
                        f'matte {"warped" if (matte0 is not None and matte1 is not None) else "black"}, '
                        f'conf {tuple(res_conf.shape)}'
                    ),
                }

            except Exception as e:
                try:
                    write_error_frame(msg.get('result0'), msg.get('matte0'), msg.get('conf0'),
                                      msg.get('input0'), msg.get('input1'),
                                      colour_space=msg.get('colour_space'))
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