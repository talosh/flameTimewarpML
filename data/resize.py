"""
resize.py
=========

Pixel resize only. The *shape* math (what target size a frame gets) lives in
sampler.py so the planner and dataset agree; this module just moves pixels to a
target the caller already computed.

We dropped LANCZOS for speed. `area` is the downscale path — it's the common
case (EXR sources are large), it's antialiased, and, crucially for HDR data, it
stays non-negative for non-negative input (it's a local average), unlike bicubic
which overshoots into negatives. `bilinear` handles the rare upscale.

torch is imported lazily so `import data` works without it.
"""

from __future__ import annotations


def resize_chw(tensor, out_h: int, out_w: int):
    """Resize a (C, H, W) float tensor to (C, out_h, out_w).

    Downscale (target area <= source) uses `area`; upscale uses `bilinear`.
    A no-op when already at the target size.
    """
    import torch.nn.functional as F  # noqa: F401  (lazy)

    c, h, w = tensor.shape
    if h == out_h and w == out_w:
        return tensor
    x = tensor.unsqueeze(0)
    if out_h * out_w <= h * w:
        x = F.interpolate(x, size=(out_h, out_w), mode="area")
    else:
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x.squeeze(0)
