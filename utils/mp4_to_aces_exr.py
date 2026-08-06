#!/usr/bin/env python3
"""
mp4_to_aces_exr.py

Batch-converts MP4 files (recursively) into linear ACES (AP0 or AP1) 16-bit
half-float EXR image sequences (PIZ compressed). Single-threaded, sequential.

Pipeline per file (kept deliberately simple and phase-separated):
    1.  Walk input dir, find *.mp4 (any case), recursively.
    2.  Sanitize every path component to plain ASCII.
    3.  Recreate the folder structure under the output dir; each source file
        becomes its own folder containing the EXR sequence.
    4.  ffprobe the file for color_transfer / color_primaries / pix_fmt.
    5.  Run ffmpeg to write an uncompressed 16-bit TIFF sequence into a
        _tiff/ subfolder of the output folder, then WAIT for ffmpeg to finish
        and exit. By default only a window of --window frames sampled from the
        MIDDLE of the clip is decoded/written (clips are assumed continuous);
        pass --window 0 to convert the whole clip. At this point ffmpeg (and
        its decoder memory) is gone and the TIFFs are sitting on disk. TIFF
        values are still gamma/PQ/HLG-encoded; ffmpeg only does YUV->RGB
        matrixing here.
    6.  Convert the TIFFs one at a time, single-threaded:
          --format exr (default): read TIFF -> bilinear resize (torch CPU) so
            the short side == --short-side -> remove transfer curve -> convert
            gamut to AP0/AP1 -> write linear half-float EXR (PIZ).
          --format png: read TIFF -> bilinear resize only -> write PNG. NO
            color conversion (no transfer removal, no gamut change) -- pixels
            stay in the source's gamma-encoded RGB, quantized to --png-bits
            (8 or 16) with zlib level --png-compression (0-9).
        Only one frame is ever resident in memory.
    7.  Once all outputs are written, delete the _tiff/ subfolder.
    8.  Move on to the next file.

Resume (--resume): re-running with --resume skips work already on disk --
fully-converted clips are skipped without even spawning their child process,
and a clip that died partway is finished by reconverting only its missing
frames. Existence is the only check, so it assumes the existing frames were
made with the same settings (short-side, format, window); run without --resume
to force a clean reconvert.

Why this shape: ffmpeg finishing and exiting before conversion means the two
memory-hungry stages never overlap, and reading TIFFs back one at a time keeps
peak memory to a single frame. The only requirement is that the OUTPUT disk has
room for one file's worth of uncompressed TIFFs transiently (they're deleted
per file) -- and that the output dir is NOT on a tmpfs/RAM-backed mount.

Memory / process isolation:
    Some native libraries (torch's math backend, image I/O) accumulate memory
    internally in ways Python's GC cannot return to the OS, so a single
    long-lived process slowly grows its RSS across many clips until it gets
    OOM-killed. To make the run robust regardless of the exact cause, the
    default mode processes files in a FRESH child process of this same script
    (one child per --batch-size files) and waits for each child to finish and
    exit before starting the next. When a child exits, the OS reclaims all of
    its memory. This keeps the run sequential and single-threaded while
    bounding memory to at most --batch-size files' worth. If a child is killed
    (e.g. OOM on one pathological file), the parent logs it and moves on
    instead of the whole run dying. Use --in-process to disable this and run
    everything in one process (for debugging).

Dependencies:
    pip install numpy torch
    OpenImageIO python bindings (oiio) available on PYTHONPATH -- used to read
    the intermediate TIFFs and write the final EXRs.
    ffmpeg + ffprobe binaries available on PATH (or pass --ffmpeg/--ffprobe)

Notes / simplifications (read before treating this as color-critical):
    - HDR handling is a "best effort" default, not a full display-referred
      tone-mapping pipeline:
        * PQ (ST.2084): full EOTF is applied, then normalized so that
          10000 nits -> 100.0 linear and 100 nits (diffuse white) -> 1.0.
          No tone mapping / gamut clipping is done -- highlights above
          reference white simply come out as scene-linear values > 1.0,
          which is what you want feeding an EXR intermediate.
        * HLG (ARIB STD-B67): only the inverse OETF is applied. The HLG
          spec additionally defines a display-referred OOTF (dependent on
          nominal peak luminance) which is intentionally NOT applied here,
          since this is meant to stay scene-referred. Flag this if you need
          "true" HLG-to-linear per a specific target peak.
    - SDR transfer removal uses the standard BT.709 piecewise curve (not
      pure gamma 2.4/2.2). Swap `rec709_to_linear` if you want something else.
    - Color primaries are read from ffprobe; unknown/missing tags fall back
      to bt709. Check the per-file log output for footage where that
      assumption might be wrong.
    - ffmpeg's default YUV->RGB conversion already expands limited range
      (16-235) to full range when producing rgb48le, so no extra range
      handling is done here. Double check with unusual sources.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("Missing dependency: pip install torch")

try:
    import OpenImageIO as oiio
except ImportError:
    sys.exit("Missing dependency: OpenImageIO python bindings (oiio) not found on PYTHONPATH")

torch.set_num_threads(os.cpu_count() or 1)
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------
# Color matrices (industry-standard ACES values, D60 white for AP0/AP1)
# --------------------------------------------------------------------------

# Rec.709 primaries -> CIE XYZ (D65)
M_REC709_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# Rec.2020 primaries -> CIE XYZ (D65)
M_REC2020_TO_XYZ = np.array([
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
], dtype=np.float64)

# Bradford chromatic adaptation D65 -> D60 (as used throughout ACES)
M_D65_TO_D60 = np.array([
    [1.01303, 0.00610531, -0.014971],
    [0.00769823, 0.998165, -0.00503203],
    [-0.00284131, 0.00468516, 0.924507],
], dtype=np.float64)

# CIE XYZ (D60) -> ACES AP0
M_XYZ_TO_AP0 = np.array([
    [1.0498110175, 0.0000000000, -0.0000974845],
    [-0.4959030231, 1.3733130458, 0.0982400361],
    [0.0000000000, 0.0000000000, 0.9912520182],
], dtype=np.float64)

# CIE XYZ (D60) -> ACES AP1
M_XYZ_TO_AP1 = np.array([
    [1.6410233797, -0.3248032942, -0.2364246952],
    [-0.6636628587, 1.6153315917, 0.0167563477],
    [0.0117218943, -0.0082844420, 0.9883948585],
], dtype=np.float64)


def build_gamut_matrix(source_primaries: str, target_gamut: str) -> np.ndarray:
    """Compose source-RGB -> AP0/AP1 3x3 matrix."""
    if source_primaries == "bt2020":
        src_to_xyz = M_REC2020_TO_XYZ
    else:
        src_to_xyz = M_REC709_TO_XYZ

    xyz_to_target = M_XYZ_TO_AP0 if target_gamut == "ap0" else M_XYZ_TO_AP1

    # RGB(src) -> XYZ(D65) -> XYZ(D60) -> AP0/AP1
    return xyz_to_target @ M_D65_TO_D60 @ src_to_xyz


# --------------------------------------------------------------------------
# Transfer functions (EOTFs) -> scene-linear
# --------------------------------------------------------------------------

def rec709_to_linear(v: np.ndarray) -> np.ndarray:
    """Standard BT.709 transfer curve, inverse (code value -> linear)."""
    v = np.clip(v, 0.0, None)
    return np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1.0 / 0.45))


def pq_to_linear(v: np.ndarray, ref_white_nits: float = 100.0) -> np.ndarray:
    """SMPTE ST.2084 (PQ) EOTF, code value [0,1] -> linear, normalized so
    ref_white_nits maps to 1.0 scene-linear."""
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0

    v = np.clip(v, 0.0, 1.0)
    vp = np.power(v, 1.0 / m2)
    num = np.clip(vp - c1, 0.0, None)
    den = c2 - c3 * vp
    den = np.where(den <= 0, 1e-8, den)
    lin = np.power(num / den, 1.0 / m1)  # 0..1 relative to 10000 nits
    nits = lin * 10000.0
    return nits / ref_white_nits


def hlg_to_linear(v: np.ndarray) -> np.ndarray:
    """ARIB STD-B67 (HLG) inverse OETF only (no display OOTF applied --
    result stays scene-referred, see module docstring)."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    v = np.clip(v, 0.0, 1.0)
    return np.where(
        v <= 0.5,
        (v * v) / 3.0,
        (np.exp((v - c) / a) + b) / 12.0,
    )


def remove_transfer(v: np.ndarray, transfer: str) -> np.ndarray:
    if transfer == "smpte2084":
        return pq_to_linear(v)
    if transfer == "arib-std-b67":
        return hlg_to_linear(v)
    # default / bt709 / unknown -> treat as rec709
    return rec709_to_linear(v)


# --------------------------------------------------------------------------
# Filesystem / naming helpers
# --------------------------------------------------------------------------

def sanitize_ascii(name: str) -> str:
    """Normalize unicode to closest ASCII, drop anything left over, replace
    unsafe filesystem characters with underscore."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_bytes = normalized.encode("ascii", "ignore")
    ascii_str = ascii_bytes.decode("ascii")
    safe = []
    for ch in ascii_str:
        if ch.isalnum() or ch in "-_. ":
            safe.append(ch)
        else:
            safe.append("_")
    result = "".join(safe).strip().strip(".")
    return result or "unnamed"


def find_mp4_files(input_dir: Path):
    for root, _dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                yield Path(root) / f


def sanitized_relative_output_dir(input_dir: Path, mp4_path: Path, output_dir: Path) -> Path:
    rel = mp4_path.relative_to(input_dir)
    rel_parts = [sanitize_ascii(p) for p in rel.parts[:-1]]
    stem = sanitize_ascii(rel.stem)
    return output_dir.joinpath(*rel_parts, stem)


def count_existing_frames(out_dir: Path, ext: str) -> int:
    """Count already-written output frames (top-level {stem}.NNNNNN.ext files)
    in out_dir. The _tiff/ subfolder holds .tiff intermediates and is not
    matched by this non-recursive glob."""
    if not out_dir.exists():
        return 0
    return sum(1 for _ in out_dir.glob(f"*.{ext}"))


# --------------------------------------------------------------------------
# ffprobe / ffmpeg
# --------------------------------------------------------------------------

def probe_color_info(ffprobe_bin: str, path: Path) -> dict:
    cmd = [
        ffprobe_bin, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=color_transfer,color_primaries,color_space,pix_fmt,width,height,"
        "nb_frames,avg_frame_rate,duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    s = streams[0]

    transfer = s.get("color_transfer", "unknown") or "unknown"
    primaries = s.get("color_primaries", "unknown") or "unknown"

    if transfer == "unknown":
        transfer = "bt709"
    if primaries == "unknown":
        primaries = "bt709"

    return {
        "transfer": transfer,
        "primaries": primaries,
        "pix_fmt": s.get("pix_fmt", ""),
        "width": s.get("width"),
        "height": s.get("height"),
        "nb_frames": s.get("nb_frames"),
        "avg_frame_rate": s.get("avg_frame_rate"),
        "duration": s.get("duration"),
    }


def _parse_rate(rate: str) -> float:
    """Parse an ffprobe rate like '30000/1001' or '25/1' into a float."""
    try:
        if rate and "/" in rate:
            num, den = rate.split("/")
            den = float(den)
            return float(num) / den if den else 0.0
        return float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def get_frame_count(ffprobe_bin: str, path: Path, info: dict) -> int:
    """Best-effort total frame count. Try the header's nb_frames, then
    fps*duration, then finally a full (slower) -count_frames decode. Returns
    None only if all of those fail."""
    try:
        n = int(info.get("nb_frames"))
        if n > 0:
            return n
    except (TypeError, ValueError):
        pass

    fps = _parse_rate(info.get("avg_frame_rate"))
    try:
        dur = float(info.get("duration"))
        if fps > 0 and dur > 0:
            return max(1, round(fps * dur))
    except (TypeError, ValueError):
        pass

    # Last resort: decode and count exactly.
    cmd = [
        ffprobe_bin, "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nw=1:nk=1", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        n = int(out.stdout.strip())
        if n > 0:
            return n
    except (subprocess.CalledProcessError, ValueError):
        pass
    return None


def write_tiff_sequence(ffmpeg_bin: str, src: Path, tiff_dir: Path,
                        start: int = 0, count: int = None) -> list:
    """Phase 1: run ffmpeg to dump the sequence as uncompressed 16-bit TIFFs
    into tiff_dir, then wait for it to finish and exit. Values are only
    YUV->RGB matrixed; the transfer curve is left intact for Python to remove.

    If count is given, only source frames [start, start+count) are decoded and
    written (used for the middle-window mode) -- ffmpeg does this itself so we
    never write the whole sequence just to keep the middle. Returns the sorted
    list of written TIFF paths (renumbered from 1)."""
    # Start from a clean dir so leftover TIFFs from a crashed/earlier run
    # (possibly a different window) can't get mixed into this extraction.
    if tiff_dir.exists():
        shutil.rmtree(tiff_dir, ignore_errors=True)
    tiff_dir.mkdir(parents=True, exist_ok=True)
    pattern = tiff_dir / "frame_%06d.tiff"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-i", str(src),
        "-vsync", "0",
    ]
    if count is not None:
        # trim uses source frame indices; end_frame is exclusive. Colon-
        # separated options avoid the comma-escaping the select filter needs.
        cmd += ["-vf", f"trim=start_frame={start}:end_frame={start + count}"]
        cmd += ["-frames:v", str(count)]  # hard cap, belt-and-suspenders
    cmd += ["-pix_fmt", "rgb48le", str(pattern)]

    # subprocess.run blocks until ffmpeg exits -> process is fully gone before
    # we return and start the conversion phase.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {proc.stderr.strip()}")

    frames = sorted(tiff_dir.glob("frame_*.tiff"))
    if not frames:
        raise RuntimeError("ffmpeg produced no TIFF frames")
    return frames


# --------------------------------------------------------------------------
# Image processing (Phase 2: TIFF -> EXR, one frame at a time)
# --------------------------------------------------------------------------

def read_tiff_uint16(path: Path) -> np.ndarray:
    """Read an intermediate TIFF via OIIO, forced to uint16, HWC, RGB."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"OIIO failed to open {path}: {oiio.geterror()}")
    spec = inp.spec()
    pixels = inp.read_image(oiio.UINT16)
    inp.close()
    if pixels is None:
        raise RuntimeError(f"OIIO failed to read pixels from {path}")

    arr = np.asarray(pixels, dtype=np.uint16).reshape(spec.height, spec.width, spec.nchannels)
    if arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        arr = np.repeat(arr[:, :, :1], 3, axis=2)
    return arr


def resize_bilinear_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    """Bilinear resize (torch, CPU) so the short side == short_side, aspect
    preserved. img is HWC float32."""
    h, w = img.shape[:2]
    if h <= w:
        new_h = short_side
        new_w = max(1, round(w * (short_side / h)))
    else:
        new_w = short_side
        new_h = max(1, round(h * (short_side / w)))

    with torch.inference_mode():
        t = torch.from_numpy(img).to(DEVICE).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return t.squeeze(0).permute(1, 2, 0).contiguous().numpy().copy()


def write_exr_half_piz(path: Path, rgb: np.ndarray):
    h, w, c = rgb.shape
    half = rgb.astype(np.float16)

    spec = oiio.ImageSpec(w, h, c, oiio.HALF)
    spec.attribute("compression", "piz")
    spec.channelnames = ("R", "G", "B")[:c]

    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"OIIO could not create writer for {path}: {oiio.geterror()}")
    if not out.open(str(path), spec):
        raise RuntimeError(f"OIIO failed to open {path} for writing: {out.geterror()}")
    if not out.write_image(half):
        raise RuntimeError(f"OIIO failed to write {path}: {out.geterror()}")
    out.close()


def convert_tiff_to_exr(tiff_path: Path, exr_path: Path, transfer: str, gamut_matrix: np.ndarray, short_side: int):
    raw = read_tiff_uint16(tiff_path)  # uint16, HWC, RGB

    code_values = raw.astype(np.float32)
    code_values *= (1.0 / 65535.0)
    del raw

    small = resize_bilinear_short_side(code_values, short_side)
    del code_values

    linear = remove_transfer(small, transfer)
    del small

    h, w, _ = linear.shape
    converted = (linear.reshape(-1, 3) @ gamut_matrix.T).reshape(h, w, 3).astype(np.float32)
    del linear

    write_exr_half_piz(exr_path, converted)
    del converted


def write_png(path: Path, img01: np.ndarray, bits: int, compression: int):
    """Write HWC float image (values in [0,1], still gamma-encoded) as PNG.
    bits: 8 or 16. compression: zlib level 0-9 (0 = none/fastest/largest)."""
    img01 = np.clip(img01, 0.0, 1.0)
    if bits == 16:
        data = (img01 * 65535.0 + 0.5).astype(np.uint16)
        pixtype = oiio.UINT16
    else:
        data = (img01 * 255.0 + 0.5).astype(np.uint8)
        pixtype = oiio.UINT8

    h, w, c = data.shape
    spec = oiio.ImageSpec(w, h, c, pixtype)
    spec.attribute("png:compressionLevel", int(compression))

    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"OIIO could not create writer for {path}: {oiio.geterror()}")
    if not out.open(str(path), spec):
        raise RuntimeError(f"OIIO failed to open {path} for writing: {out.geterror()}")
    if not out.write_image(data):
        raise RuntimeError(f"OIIO failed to write {path}: {out.geterror()}")
    out.close()


def convert_tiff_to_png(tiff_path: Path, png_path: Path, short_side: int, bits: int, compression: int):
    """PNG path: resize only. No transfer removal, no gamut conversion --
    pixels stay in the source's gamma-encoded RGB, just quantized to 8/16-bit."""
    raw = read_tiff_uint16(tiff_path)

    code_values = raw.astype(np.float32)
    code_values *= (1.0 / 65535.0)
    del raw

    small = resize_bilinear_short_side(code_values, short_side)
    del code_values

    write_png(png_path, small, bits, compression)
    del small


# --------------------------------------------------------------------------
# Per-file orchestration (sequential, single-threaded)
# --------------------------------------------------------------------------

def process_one_file(mp4_path: Path, input_dir: Path, output_dir: Path, args):
    out_dir = sanitized_relative_output_dir(input_dir, mp4_path, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiff_dir = out_dir / "_tiff"

    print(f"[{mp4_path.name}] probing color info...")
    info = probe_color_info(args.ffprobe, mp4_path)
    print(f"[{mp4_path.name}] transfer={info['transfer']} primaries={info['primaries']} pix_fmt={info['pix_fmt']}")

    is_png = args.format == "png"
    ext = "png" if is_png else "exr"
    gamut_matrix = None if is_png else build_gamut_matrix(info["primaries"], args.gamut)

    window = args.window

    # Total frame count is needed for the middle window and/or resume check;
    # probe it at most once.
    total = None
    if window > 0 or args.resume:
        total = get_frame_count(args.ffprobe, mp4_path, info)

    # ---- Resume: skip this clip entirely if its output is already complete ----
    if args.resume:
        if window > 0:
            expected = window if total is None else min(window, total)
        else:
            expected = total  # None if unknown
        existing = count_existing_frames(out_dir, ext)
        if expected is not None and existing >= expected:
            print(f"[{mp4_path.name}] resume: {existing}/{expected} frame(s) already present, skipping.")
            return

    # ---- Work out the frame window (middle N frames) ----
    start, count = 0, None  # defaults: whole clip
    if window > 0:
        if total is None:
            print(f"[{mp4_path.name}] WARN: couldn't determine frame count; "
                  f"taking first {window} frames instead of the middle.")
            start, count = 0, window
        elif total <= window:
            print(f"[{mp4_path.name}] only {total} frame(s) <= window {window}; converting all.")
            start, count = 0, None
        else:
            start = (total - window) // 2
            count = window
            print(f"[{mp4_path.name}] middle window: frames {start}..{start + count - 1} of {total}")

    try:
        # ---- Phase 1: ffmpeg writes the TIFF sequence (window only), then exits ----
        print(f"[{mp4_path.name}] writing TIFF sequence with ffmpeg...")
        tiff_frames = write_tiff_sequence(args.ffmpeg, mp4_path, tiff_dir, start=start, count=count)
        print(f"[{mp4_path.name}] ffmpeg done, {len(tiff_frames)} TIFFs on disk. Converting to {ext.upper()}...")

        # ---- Phase 2: convert TIFFs -> output one at a time ----
        skipped = 0
        for i, tiff_path in enumerate(tiff_frames, start=1):
            out_path = out_dir / f"{out_dir.name}.{i:06d}.{ext}"
            if args.resume and out_path.exists():
                skipped += 1
                continue
            if is_png:
                convert_tiff_to_png(tiff_path, out_path, args.short_side, args.png_bits, args.png_compression)
            else:
                convert_tiff_to_exr(tiff_path, out_path, info["transfer"], gamut_matrix, args.short_side)
            if i % 100 == 0:
                print(f"[{mp4_path.name}] {i}/{len(tiff_frames)} {ext.upper()}s written...")

        tail = f" ({skipped} already present)" if skipped else ""
        print(f"[{mp4_path.name}] done ({len(tiff_frames)} frames){tail} -> {out_dir}")
    finally:
        # ---- Phase 3: always remove the intermediate TIFFs ----
        if tiff_dir.exists():
            shutil.rmtree(tiff_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Worker: process a given list of files in THIS process, sequentially.
# --------------------------------------------------------------------------

def run_worker(mp4_files, input_dir: Path, output_dir: Path, args) -> int:
    failures = 0
    for mp4_path in mp4_files:
        try:
            process_one_file(mp4_path, input_dir, output_dir, args)
        except Exception as e:
            print(f"[ERROR] {mp4_path}: {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
    return failures


# --------------------------------------------------------------------------
# Parent: spawn a fresh child of THIS script per batch, so each child's
# memory is fully reclaimed by the OS when it exits.
# --------------------------------------------------------------------------

def _shared_arg_list(args) -> list:
    """Rebuild the passthrough CLI flags for a child invocation."""
    flags = [
        "--short-side", str(args.short_side),
        "--window", str(args.window),
        "--format", args.format,
        "--gamut", args.gamut,
        "--png-bits", str(args.png_bits),
        "--png-compression", str(args.png_compression),
        "--ffmpeg", args.ffmpeg,
        "--ffprobe", args.ffprobe,
    ]
    if args.resume:
        flags.append("--resume")
    return flags


def run_parent(mp4_files, input_dir: Path, output_dir: Path, args):
    # Resume pre-filter: skip files whose output is clearly already complete,
    # so we don't pay a child-process spawn (torch import) just to skip them.
    # Certain case without probing: for a fixed window, having >= window frames
    # on disk means done. Uncertain cases (window 0, or fewer-than-window that
    # might be a genuinely short clip) are left to the child's authoritative
    # check after it probes.
    if args.resume and args.window > 0:
        ext = "png" if args.format == "png" else "exr"
        kept, skipped = [], 0
        for p in mp4_files:
            out_dir = sanitized_relative_output_dir(input_dir, p, output_dir)
            if count_existing_frames(out_dir, ext) >= args.window:
                skipped += 1
            else:
                kept.append(p)
        if skipped:
            print(f"[resume] {skipped} file(s) already complete, skipping; {len(kept)} to process.")
        mp4_files = kept
        if not mp4_files:
            print("Nothing left to do.")
            return

    batch_size = max(1, args.batch_size)
    batches = [mp4_files[i:i + batch_size] for i in range(0, len(mp4_files), batch_size)]
    total = len(mp4_files)
    done = 0

    for bidx, batch in enumerate(batches, start=1):
        cmd = [
            args.python, os.path.abspath(__file__),
            str(input_dir), str(output_dir),
            *_shared_arg_list(args),
            "--_worker_files", *[str(p) for p in batch],
        ]
        print(f"=== batch {bidx}/{len(batches)} ({len(batch)} file(s), {done}/{total} done) -> fresh process ===")
        proc = subprocess.run(cmd)
        done += len(batch)
        if proc.returncode != 0:
            # Child may have been OOM-killed (-9) or errored. Log and continue;
            # the next batch runs in a brand-new process with a clean slate.
            print(
                f"[WARN] batch {bidx} child exited with code {proc.returncode} "
                f"(files: {', '.join(p.name for p in batch)}). Continuing.",
                file=sys.stderr,
            )

    print(f"All batches attempted ({total} file(s)).")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="Folder to search for .mp4 files (recursive)")
    ap.add_argument("output_dir", type=Path, help="Folder to write EXR sequences into")
    ap.add_argument("--short-side", type=int, default=448, help="Target short-side resolution (default: 448)")
    ap.add_argument("--window", type=int, default=24,
                    help="Convert only a window of this many frames sampled from "
                         "the MIDDLE of each clip (clips are assumed continuous). "
                         "Use 0 (or negative) to convert the whole clip. (default: 24)")
    ap.add_argument("--format", choices=["exr", "png"], default="exr",
                    help="Output format. 'exr' = linear ACES half-float (resize + "
                         "transfer removal + gamut convert). 'png' = resize only, "
                         "no color conversion, stays in source gamma. (default: exr)")
    ap.add_argument("--gamut", choices=["ap0", "ap1"], default="ap1",
                    help="Target ACES gamut for EXR output (ignored for --format png). (default: ap1)")
    ap.add_argument("--png-bits", type=int, choices=[8, 16], default=8,
                    help="PNG bit depth (only for --format png). (default: 8)")
    ap.add_argument("--png-compression", type=int, default=6,
                    help="PNG zlib compression level 0-9 (only for --format png); "
                         "0 = none/fastest/largest, 9 = smallest/slowest. (default: 6)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip work already done: fully-converted clips are skipped "
                         "entirely, and individual frames that already exist are not "
                         "reconverted. Assumes existing frames were made with the same "
                         "settings (short-side, format, window); use without --resume "
                         "to force a full reconvert.")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg binary")
    ap.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe binary")
    ap.add_argument(
        "--batch-size", type=int, default=1,
        help="Files to process per child process before it exits and its "
             "memory is reclaimed (default: 1 = strictest memory bound; raise "
             "to amortize per-child startup at the cost of a higher ceiling).",
    )
    ap.add_argument("--in-process", action="store_true",
                    help="Process everything in a single process (no per-batch "
                         "child spawning). For debugging; may grow memory over a run.")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use for child processes (default: this one).")
    # Internal: when present, this invocation IS a worker and should process
    # exactly these files in-process, then exit.
    ap.add_argument("--_worker_files", nargs="*", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not (0 <= args.png_compression <= 9):
        ap.error("--png-compression must be between 0 and 9")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Worker invocation: process the explicit file list and exit.
    if args._worker_files is not None:
        files = [Path(p) for p in args._worker_files]
        failures = run_worker(files, input_dir, output_dir, args)
        sys.exit(1 if failures else 0)

    # Parent (or in-process) invocation: enumerate and dispatch.
    mp4_files = list(find_mp4_files(input_dir))
    if not mp4_files:
        print("No .mp4 files found.")
        return
    print(f"Found {len(mp4_files)} mp4 file(s).")

    if args.in_process:
        run_worker(mp4_files, input_dir, output_dir, args)
    else:
        run_parent(mp4_files, input_dir, output_dir, args)


if __name__ == "__main__":
    main()