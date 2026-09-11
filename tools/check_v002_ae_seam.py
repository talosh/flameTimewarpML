#!/usr/bin/env python
"""Colour-seam audit for warpnet4_v002's embedded AE input pipeline.

Compares, on REAL dataset EXRs, what the v002 model feeds to its frozen
ae_encoder against what the ts_test dataset fed to the encoder during AE
training. This is the distribution-shift measurement that justifies (or
flags) training film_z on the production path.

Three paths per frame (all start from the same native EXR, read once):

  A   PRODUCTION   : reader clip(0,None) -> area resize to frame_size
                     (sampler resized_hw, multiple 16) -> model ae_prepare
                     (nan_to_num rails + soft_limit + area resize to 64
                     + encode)
  A'  MODEL-COLOUR : reader clip(0,None) -> area resize DIRECTLY to 64 ->
                     model ae_prepare (no second resize — target size)
  B   TRAINING     : ts_test path verbatim (nan_to_num defaults +
                     clip(0,None) -> area resize to 64 -> encode_linear)

Decomposition:
  A  vs B  = total production-vs-training seam (what film_z will see)
  A' vs B  = colour/sanitisation seam only (soft_limit rails vs raw)
  A  vs A' = resize-staging seam only (two-step area vs one-step area)

For each encoding (ap1, ap0, rec709, acescct) we report encoded-tensor
bit-identity / max|Δ|, plus z_global |Δ| through the trained encoder on
consecutive frame pairs (what film_z actually consumes).

Usage (from the repo root):

  env LD_LIBRARY_PATH=/home/flame/miniconda3/envs/appenv/lib \
      CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
      /home/flame/miniconda3/envs/appenv/bin/python tools/check_v002_ae_seam.py \
      [--clips 4] [--frames-per-clip 3] [--frame-size 224]
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dataset-root', default='/mnt/StorageMedia/dataset_image/pexels_exr')
    ap.add_argument('--clips', type=int, default=4, help='number of clips to audit')
    ap.add_argument('--frames-per-clip', type=int, default=3)
    ap.add_argument('--frame-size', type=int, default=224,
                    help='production short side (live run uses 224)')
    ap.add_argument('--ae', default='/mnt/StorageMedia/dev/ts_test/runs/m6_full/ckpt_final.pt')
    ap.add_argument('--ts-test', default='/mnt/StorageMedia/dev/ts_test')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    from models.warpnet4_v002 import Model as M2
    from data.io import read_exr_hwc
    from data.resize import resize_chw
    from data.sampler import resized_hw, ROT_NONE

    sys.path.insert(0, args.ts_test)
    from src.data.latent_ae_dataset import (
        read_exr_hwc as ts_read, resize_chw as ts_resize,
        encode_linear as ts_encode, SHORT_SIDE, ENCODINGS)

    m = M2().get_training_model()()
    ck = torch.load(args.ae, map_location='cpu', weights_only=False)
    m.ae_encoder.load_state_dict(ck['encoder'], strict=True)
    m.eval()

    sys.path.insert(0, args.ts_test)
    from src.models.vit_latent_ae import LatentEncoder as StandaloneAE
    ref = StandaloneAE()
    ref.load_state_dict(ck['encoder'], strict=True)
    ref.eval()

    captured = []

    def _hook(mod, inp):
        captured.append(inp)

    m.ae_encoder.register_forward_pre_hook(_hook)

    # ---- deterministic clip selection: stride over the sorted clip list
    clips = sorted(os.path.join(root, d)
                   for root, dirs, _ in os.walk(args.dataset_root)
                   for d in dirs
                   if any(f.endswith('.exr') for f in os.listdir(os.path.join(root, d))))
    step = max(1, len(clips) // args.clips)
    clips = clips[::step][:args.clips]
    print(f'[seam] {len(clips)} clips: {", ".join(os.path.relpath(c, args.dataset_root) for c in clips)}')

    # throwaway warp-side input at production size — the AE branch is
    # independent of it, but the dilated conv stack needs the full pyramid
    warp_in = torch.randn(1, 3, 224, 398)

    def model_encoded(lin, enc):
        """Run the real model pipeline on one linear frame; return the
        encoded tensor the ae_encoder actually receives (hook-captured).
        `lin` is 3D CHW; the model API takes 4D [B,C,H,W]."""
        captured.clear()
        with torch.no_grad():
            m(warp_in, warp_in, sharpness=0.5, ae_in=(lin[None], lin[None]), encoding=enc)
        a_enc, b_enc, ratio = captured[0]
        return a_enc

    def crop_common(*ts):
        """Top-left crop to the common spatial size (snap-to-16 at frame_size
        slightly changes the aspect, so the two-step and one-step 64-side
        targets can differ by a couple of pixels — that is itself a seam)."""
        h = min(t.shape[-2] for t in ts)
        w = min(t.shape[-1] for t in ts)
        return tuple(t[..., :h, :w] for t in ts)

    # per-encoding accumulators
    stats = {e: dict(n_px=0, n_mis_AB=0, n_mis_AAp=0,
                     bitid_A=0, bitid_Ap=0,
                     maxd_A=0.0, maxd_A_match=0.0, maxd_A_mis=0.0,
                     maxd_Ap=0.0, maxd_stag_match=0.0, maxd_stag_mis=0.0)
             for e in ENCODINGS}
    z_diffs = {e: [] for e in ENCODINGS}
    lin_stats = []

    for clip in clips:
        frames = sorted(f for f in os.listdir(clip) if f.endswith('.exr'))
        n = len(frames)
        idxs = list(range(0, n, max(1, n // args.frames_per_clip)))[:args.frames_per_clip]
        for i in idxs:
            path = os.path.join(clip, frames[i])
            hwc = read_exr_hwc(path)
            h, w = hwc.shape[:2]
            n_nan = int(np.isnan(hwc).sum())
            n_pinf = int(np.isposinf(hwc).sum())
            n_ninf = int(np.isneginf(hwc).sum())
            finite = hwc[np.isfinite(hwc)]
            above_knee = float((finite > 1.1 - 0.002).mean()) if finite.size else 0.0
            shadow = float(((finite > 0.0) & (finite < 0.002)).mean()) if finite.size else 0.0
            lin_stats.append((os.path.basename(path), f'{h}x{w}',
                              n_nan, n_pinf + n_ninf,
                              float(finite.min()) if finite.size else 0.0,
                              float(finite.max()) if finite.size else 0.0,
                              above_knee, shadow))

            # A: production two-step (reader clip -> frame_size)
            hp = hwc.copy()
            np.clip(hp, 0.0, None, out=hp)
            chw = torch.from_numpy(np.ascontiguousarray(hp.transpose(2, 0, 1)))
            out_h, out_w = resized_hw(h, w, ROT_NONE, frame_size=args.frame_size)
            linA = resize_chw(chw, out_h, out_w)
            # A': model-colour, one step to 64
            out64_h, out64_w = (SHORT_SIDE, max(SHORT_SIDE, int(round(max(h, w) * (SHORT_SIDE / min(h, w)))))) \
                if h <= w else (max(SHORT_SIDE, int(round(max(h, w) * (SHORT_SIDE / min(h, w))))), SHORT_SIDE)
            linAp = resize_chw(chw, out64_h, out64_w)

            # B: ts_test training path verbatim
            hb = ts_read(path)
            np.nan_to_num(hb, copy=False)
            np.clip(hb, 0.0, None, out=hb)
            chwb = torch.from_numpy(np.ascontiguousarray(hb.transpose(2, 0, 1)))
            linB = ts_resize(chwb, out64_h, out64_w)

            for e in ENCODINGS:
                a_enc = model_encoded(linA, e)
                ap_enc = model_encoded(linAp, e)
                b_enc = ts_encode(linB, e).unsqueeze(0)

                s = stats[e]
                mis_AB = int(a_enc.shape != b_enc.shape)
                s['n_mis_AB'] += mis_AB
                s['n_mis_AAp'] += int(a_enc.shape != ap_enc.shape)
                ac, bc = crop_common(a_enc, b_enc)
                ac2, bc2, apc = crop_common(a_enc, b_enc, ap_enc)
                npx = ac.numel()
                s['n_px'] += npx
                s['bitid_A'] += int(torch.equal(ac, bc)) and npx
                s['bitid_Ap'] += int(torch.equal(apc, bc2)) and npx
                d = (ac - bc).abs().max().item()
                s['maxd_A'] = max(s['maxd_A'], d)
                s['maxd_A_match' if not mis_AB else 'maxd_A_mis'] = \
                    max(s['maxd_A_match' if not mis_AB else 'maxd_A_mis'], d)
                s['maxd_Ap'] = max(s['maxd_Ap'], (apc - bc2).abs().max().item())
                ds = (ac2 - apc).abs().max().item()
                key = 'maxd_stag_match' if a_enc.shape == ap_enc.shape else 'maxd_stag_mis'
                s[key] = max(s[key], ds)

                # z through the trained encoder, same (a, a) self-pair both sides
                with torch.no_grad():
                    zA = m.ae_encoder(a_enc, a_enc, torch.tensor([0.5]))[1]
                    zB = ref(b_enc, b_enc, torch.tensor([0.5]))[1]
                z_diffs[e].append((zA - zB).norm().item())

    print(f'\n[seam] native linear stats (frames, size, nan, inf, min, max, '
          f'frac>1.088, frac in (0,0.002)):')
    for name, size, nn, ni, lo, hi, ak, sh in lin_stats:
        print(f'  {name} {size}: nan={nn} inf={ni} min={lo:.4g} max={hi:.4g} '
              f'above_knee={ak:.2e} shadow_band={sh:.2e}')

    apl = "A'"
    n_frames = len(lin_stats)
    hdr = '  {:<10} {:>7} {:>7} {:>13} {:>14} {:>14} {:>14} {:>14} {:>14} {:>15} {:>14}'.format(
        'enc', 'szA!=B', 'szA!=' + apl, 'A vs B bitid',
        'A vs B max|d| match', 'A vs B max|d| mis',
        apl + ' vs B bitid', apl + ' vs B max|d|',
        'A vs ' + apl + ' max|d| match', 'A vs ' + apl + ' max|d| mis',
        'A vs ' + apl + ' max|d| (all)')
    print(f'\n[seam] encoded-tensor comparison per encoding '
          f'({n_frames} frames; sizes compared on common top-left crop; '
          f'match/mis = A-vs-B size matched or not):')
    print(hdr)
    for e in ENCODINGS:
        s = stats[e]
        print(f'  {e:<10} {s["n_mis_AB"]:>7} {s["n_mis_AAp"]:>7} '
              f'{s["bitid_A"] / s["n_px"]:>13.4f} {s["maxd_A_match"]:>14.6f} '
              f'{s["maxd_A_mis"]:>14.6f} {s["bitid_Ap"] / s["n_px"]:>14.4f} '
              f'{s["maxd_Ap"]:>15.6f} {s["maxd_stag_match"]:>14.6f} '
              f'{s["maxd_stag_mis"]:>15.6f} {s["maxd_A"]:>14.6f}')

    print(f'\n[seam] z_global |A - B| (L2, frame pairs, trained encoder):')
    for e in ENCODINGS:
        vals = z_diffs[e]
        print(f'  {e:<10} n={len(vals):<3} mean={np.mean(vals):.6f} max={np.max(vals):.6f}')
    print('\n[seam] DONE')


if __name__ == '__main__':
    main()
