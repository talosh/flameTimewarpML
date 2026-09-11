#!/usr/bin/env python
"""One-shot merge: warpnet4_v001 weights + trained ts_test AE encoder
-> warpnet4_v002 checkpoint.

Usage (from the repo root):

  PYTHONPATH=. /home/flame/miniconda3/envs/appenv/bin/python \
      tools/merge_warpnet_v002.py \
      --base weights/Warpnet4_v001_mergebase.pth \
      --ae   /mnt/StorageMedia/dev/ts_test/runs/m6_full/ckpt_final.pt \
      --out  weights/Warpnet4_v002.pth

What it does
  1. Loads the v001 state dict into a fresh v002 model (strict=False).
     The missing keys must be EXACTLY the new film_z + ae_encoder keys;
     any unexpected key fails the merge.
  2. Bakes in the trained AE encoder (strict load, latent_version "1").
  3. PASSTHROUGH gate: v001(base) vs v002(merged) on a fixed-seed input
     must give max|Δ| == 0.0 over all 4 flow levels. film_z is zero at
     this point, so the AE (whatever its weights) is multiplied by a
     zero matrix — this gate verifies the whole merge in one check.
  4. AE SEAM gate: the embedded ae_encoder vs the standalone ts_test
     LatentEncoder (same weights, same input) must give bit-identical
     z_global — catches any vendoring drift in the v002 model file.
  5. Saves the merged checkpoint in the trainer's key format and
     verifies a strict round-trip reload.

The output carries NO optimizer state: the parameter count changed, so
the v001 optimizer state cannot load. The trainer's try/except falls
back to a fresh optimizer (it prints 'unable to load optimizer state').

Exit code 0 = all gates passed.
"""
import argparse
import os
import random
import sys
import time

import torch


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', required=True,
                    help='warpnet4_v001 checkpoint (trainer key format)')
    ap.add_argument('--ae', required=True,
                    help='ts_test AE checkpoint (7-key format, ["encoder"])')
    ap.add_argument('--out', required=True, help='merged v002 checkpoint path')
    ap.add_argument('--ts-test', default='/mnt/StorageMedia/dev/ts_test',
                    help='ts_test project root (for the seam gate)')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from models.warpnet4_v001 import Model as M1
    from models.warpnet4_v002 import Model as M2

    print(f'[merge] base: {args.base}')
    print(f'[merge] ae:   {args.ae}')
    base = torch.load(args.base, map_location='cpu', weights_only=False)
    ae_ckpt = torch.load(args.ae, map_location='cpu', weights_only=False)
    lv = ae_ckpt.get('latent_version')
    assert lv == M2.info['ae_latent_version'], \
        f'AE latent_version {lv!r} != model {M2.info["ae_latent_version"]!r}'
    print(f'[merge] AE latent_version: {lv} (OK)')

    m1 = M1().get_training_model()()
    m2 = M2().get_training_model()()

    # --- Gate 1: base state dict into v002, missing == exactly the new keys
    r = m2.load_state_dict(base['flownet_state_dict'], strict=False)
    missing = sorted(r.missing_keys)
    unexpected = sorted(r.unexpected_keys)
    expected_new = {k for k in m2.state_dict()
                    if k.startswith('ae_encoder.') or '.film_z.' in k}
    assert not unexpected, f'GATE-1 FAIL: {len(unexpected)} unexpected keys, e.g. {unexpected[:8]}'
    diff = sorted(set(missing) ^ expected_new)
    assert not diff, f'GATE-1 FAIL: missing keys != expected new keys; diff {diff[:8]}'
    print(f'GATE-1 BASE-MERGE OK: {len(base["flownet_state_dict"])} keys loaded, '
          f'{len(missing)} new keys at init '
          f'({sum(1 for k in missing if ".film_z." in k)} film_z + '
          f'{sum(1 for k in missing if k.startswith("ae_encoder."))} ae_encoder)')

    # --- Gate 2: trained AE encoder, strict
    m2.ae_encoder.load_state_dict(ae_ckpt['encoder'], strict=True)
    print(f'GATE-2 AE-BAKE OK: {len(ae_ckpt["encoder"])} tensors loaded strict')

    # --- Gate 3: passthrough identity v001(base) vs v002(merged)
    m1.load_state_dict(base['flownet_state_dict'], strict=True)
    m1.eval()
    m2.eval()
    torch.manual_seed(args.seed + 1)
    img0 = torch.randn(2, 3, 224, 398) * 0.5
    img1 = torch.randn(2, 3, 224, 398) * 0.5
    sharpness = 0.37
    with torch.no_grad():
        # FlownetCas draws its random pyramid scale from the GLOBAL random
        # module (skewed_randint) — seed it identically before each call so
        # both models see the same scale sequence.
        random.seed(999)
        o1 = m1(img0, img1, sharpness=sharpness)
        random.seed(999)
        o2 = m2(img0, img1, sharpness=sharpness)
    worst = 0.0
    for i in range(4):
        d = (o1['flow_list'][i] - o2['flow_list'][i]).abs().max().item()
        worst = max(worst, d)
        assert d == 0.0, f'GATE-3 FAIL: level {i} max|Δ| = {d} (must be 0.0)'
    print(f'GATE-3 PASSTHROUGH OK: max|Δ| over 4 flow levels = {worst} (exactly 0.0)')

    # --- Gate 4: AE seam — embedded encoder vs standalone ts_test encoder
    sys.path.insert(0, args.ts_test)
    from src.models.vit_latent_ae import LatentEncoder as StandaloneAE
    ref = StandaloneAE()
    ref.load_state_dict(ae_ckpt['encoder'], strict=True)
    ref.eval()
    torch.manual_seed(args.seed + 2)
    a = torch.rand(1, 3, 64, 114).clamp(-1.0, 1.0)
    b = torch.rand(1, 3, 64, 114).clamp(-1.0, 1.0)
    ratio = torch.tensor([0.5])
    with torch.no_grad():
        z_embed = m2.ae_encoder(a, b, ratio)[1]
        z_ref = ref(a, b, ratio)[1]
    d = (z_embed - z_ref).abs().max().item()
    assert d == 0.0, f'GATE-4 FAIL: embedded vs standalone z_global max|Δ| = {d}'
    print(f'GATE-4 AE-SEAM OK: embedded vs ts_test standalone z_global max|Δ| = {d}')

    # --- Save (trainer key format; no optimizer state — see docstring)
    out = {
        'flownet_state_dict': m2.state_dict(),
        'model_info': M2.info,
        'lr': base.get('lr', 2e-5),
        'step': 0,
        'epoch': 0,
        'start_timestamp': time.time(),
        'trained_model_path': args.out,
    }
    torch.save(out, args.out)

    # --- Gate 5: strict round-trip
    m3 = M2().get_training_model()()
    ck = torch.load(args.out, map_location='cpu', weights_only=False)
    m3.load_state_dict(ck['flownet_state_dict'], strict=True)
    assert ck['model_info']['file'] == 'warpnet4_v002.py', ck['model_info']
    assert ck['model_info']['ae_latent_version'] == lv
    print('GATE-5 ROUNDTRIP OK: strict reload + model_info (file, ae_latent_version)')

    n_params = sum(p.numel() for p in m2.parameters())
    n_trainable = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f'[merge] params: {n_params/1e6:.2f}M total, {n_trainable/1e6:.2f}M trainable '
          f'(ae_encoder frozen)')
    print(f'[merge] DONE -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
