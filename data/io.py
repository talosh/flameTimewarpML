"""
io.py
=====

Decode one EXR frame to the canonical CPU tensor the cache stores: read ->
take RGB -> arcsinh tonemap -> to (C, H, W) float32 -> resize to canonical size.

Order matters: tonemap *then* resize, so resampling happens in the compressed
arcsinh space (more stable than resampling raw HDR magnitudes) — same order the
original reader used. arcsinh(x*2)/2 and the clamp to >=0 match the original.

OpenImageIO and torch are imported lazily. `default_reader` is the production
path; tests inject a `reader(path, out_h, out_w, channels)` returning the same
tensor from synthetic data, so the pixel pipeline is exercisable without EXRs.
"""

from __future__ import annotations

from typing import Optional


def read_exr_hwc(path: str, channels: int = 3):
    """Read an EXR as an (H, W, C) float32 numpy array (first `channels` only)."""
    import numpy as np
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise IOError(f"cannot open EXR: {path}")
    try:
        spec = inp.spec()
        nch = min(channels, spec.nchannels)
        data = inp.read_image(0, 0, 0, nch)
    finally:
        inp.close()
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:                       # single channel
        arr = arr[:, :, None]
    return arr[:, :, :channels]


def default_reader(path: str, out_h: int, out_w: int,
                   channels: int = 3, tonemap: bool = True):
    """Full load: decode -> tonemap -> CHW float32 tensor -> resize to (out_h, out_w)."""
    import numpy as np
    import torch
    from .resize import resize_chw

    hwc = read_exr_hwc(path, channels)                  # (H, W, C) float32
    if tonemap:
        hwc = np.arcsinh(hwc * 2.0) / 2.0
    np.clip(hwc, 0.0, None, out=hwc)                    # HDR can carry negatives
    chw = torch.from_numpy(np.ascontiguousarray(hwc.transpose(2, 0, 1)))
    return resize_chw(chw, out_h, out_w)
