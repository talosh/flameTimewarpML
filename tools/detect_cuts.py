"""
detect_cuts.py
==============

Find sequences that contain a hidden *scene cut* — a hard content change with no
break in frame numbering, so `descriptions.py` sees one continuous run. Those are
the dangerous ones: a training window spanning the cut asks the model to
interpolate between two unrelated shots.

Method (cheap by design — it touches every frame in the dataset):
  1. decode each frame at thumbnail size (default 64px), tonemap, grayscale
  2. for each adjacent pair compute a dissimilarity = mean-abs-diff + histogram
     distance
  3. score each boundary against that sequence's own *median* dissimilarity, so
     a fast pan (high motion everywhere) isn't flagged while a cut in an
     otherwise-static shot is. An absolute floor suppresses noise-only spikes.
  4. re-read only the flagged pairs at preview size and write an HTML report

Outputs: an HTML report with before/after thumbnails (open in a browser), plus
JSON and CSV for scripting.

Run from the project root:

    python -m detect_cuts /path/to/dataset --out cuts_report
    python -m detect_cuts /path/to/dataset --threshold 6 --workers 16
    python -m detect_cuts /path/to/dataset --limit 500        # quick sample

Then open cuts_report/index.html.
"""

from __future__ import annotations

import os
import io
import csv
import sys
import json
import time
import base64
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# --------------------------------------------------------------------------
# frame loading (OIIO; imported lazily so --self_test works without it)
# --------------------------------------------------------------------------

def load_gray_thumb(path: str, size: int) -> np.ndarray:
    """Decode one EXR to a small tonemapped grayscale array in [0, 1]-ish."""
    import OpenImageIO as oiio
    from PIL import Image

    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise IOError(f"cannot open {path}")
    try:
        spec = inp.spec()
        nch = min(3, spec.nchannels)
        arr = np.asarray(inp.read_image(0, 0, 0, nch), dtype=np.float32)
    finally:
        inp.close()
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = np.arcsinh(np.clip(arr, 0, None) * 2.0) / 2.0          # HDR -> compressed
    if arr.shape[2] >= 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr[:, :, 0]
    im = Image.fromarray(gray, mode="F").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def load_rgb_preview(path: str, width: int) -> "Image.Image":
    """Decode one EXR to a viewable RGB PIL image for the report."""
    import OpenImageIO as oiio
    from PIL import Image

    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise IOError(f"cannot open {path}")
    try:
        spec = inp.spec()
        nch = min(3, spec.nchannels)
        arr = np.asarray(inp.read_image(0, 0, 0, nch), dtype=np.float32)
        h, w = spec.height, spec.width
    finally:
        inp.close()
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = np.arcsinh(np.clip(arr, 0, None) * 2.0) / 2.0
    # simple display transform: normalize to the 99.5th percentile, gamma 2.2
    hi = float(np.percentile(arr, 99.5)) or 1.0
    arr = np.clip(arr / hi, 0, 1) ** (1 / 2.2)
    im = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    if w > 0 and h > 0:
        im = im.resize((width, max(1, int(width * h / w))), Image.BILINEAR)
    return im


# --------------------------------------------------------------------------
# dissimilarity + scoring (pure numpy — unit-testable)
# --------------------------------------------------------------------------

def pair_distance(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Dissimilarity of two grayscale thumbs: pixel MAD + histogram L1.

    MAD catches content displacement; the histogram term catches tonal/scene
    changes that survive spatial shuffling (a cut to a differently-lit shot).
    """
    mad = float(np.mean(np.abs(a - b)))
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        hist_d = 0.0
    else:
        ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
        hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
        ha = ha.astype(np.float64) / max(1, ha.sum())
        hb = hb.astype(np.float64) / max(1, hb.sum())
        hist_d = float(np.abs(ha - hb).sum()) / 2.0        # 0..1
    return 0.5 * mad + 0.5 * hist_d


def score_distances(d: np.ndarray, min_abs: float, eps: float = 1e-6):
    """Score each boundary against the sequence's own median dissimilarity.

    Returns (scores, median). A boundary is interesting when it is both a large
    *relative* spike (>> the sequence's typical frame-to-frame change) and above
    an absolute floor, so static or noisy clips don't produce phantom cuts.
    """
    if d.size == 0:
        return np.zeros(0, dtype=np.float64), 0.0
    med = float(np.median(d))
    scores = d / (med + eps)
    scores = np.where(d >= min_abs, scores, 0.0)
    return scores, med


def analyse_series(thumbs, threshold: float, min_abs: float):
    """thumbs: list of (frame_index, gray array). Returns (cuts, stats)."""
    if len(thumbs) < 2:
        return [], {"median": 0.0, "n_pairs": 0, "mean": 0.0, "max": 0.0}
    idxs = [t[0] for t in thumbs]
    d = np.array([pair_distance(thumbs[i][1], thumbs[i + 1][1])
                  for i in range(len(thumbs) - 1)], dtype=np.float64)
    scores, med = score_distances(d, min_abs)
    cuts = []
    for i, sc in enumerate(scores):
        if sc >= threshold:
            cuts.append({
                "before_index": int(idxs[i]),
                "after_index": int(idxs[i + 1]),
                "distance": float(d[i]),
                "score": float(sc),
            })
    stats = {"median": med, "n_pairs": int(d.size),
             "mean": float(d.mean()), "max": float(d.max())}
    return cuts, stats


# --------------------------------------------------------------------------
# per-sequence worker
# --------------------------------------------------------------------------

_WORK = {}


def _init_worker(size, threshold, min_abs, stride, max_frames):
    _WORK.update(size=size, threshold=threshold, min_abs=min_abs,
                 stride=stride, max_frames=max_frames)


def _scan_sequence(payload):
    """payload: (seq_id, folder, basenames). Returns a result dict."""
    seq_id, folder, names = payload
    size = _WORK["size"]
    stride = _WORK["stride"]
    max_frames = _WORK["max_frames"]

    try:
        indices = list(range(0, len(names), max(1, stride)))
        if max_frames and len(indices) > max_frames:
            indices = indices[:max_frames]
        thumbs = []
        for i in indices:
            thumbs.append((i, load_gray_thumb(os.path.join(folder, names[i]), size)))
        cuts, stats = analyse_series(thumbs, _WORK["threshold"], _WORK["min_abs"])
        for c in cuts:
            c["before_name"] = names[c["before_index"]]
            c["after_name"] = names[c["after_index"]]
        return {"seq_id": seq_id, "folder": folder, "n_frames": len(names),
                "n_scanned": len(thumbs), "cuts": cuts, "stats": stats, "error": None}
    except Exception as e:
        return {"seq_id": seq_id, "folder": folder, "n_frames": len(names),
                "n_scanned": 0, "cuts": [], "stats": {},
                "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _b64_png(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_report(results, out_dir, args, elapsed):
    os.makedirs(out_dir, exist_ok=True)

    flagged = [r for r in results if r["cuts"]]
    flagged.sort(key=lambda r: max(c["score"] for c in r["cuts"]), reverse=True)
    errors = [r for r in results if r["error"]]
    total_cuts = sum(len(r["cuts"]) for r in results)

    # machine-readable
    with open(os.path.join(out_dir, "cuts.json"), "w") as f:
        json.dump({"root": args.dataset_path, "params": vars(args),
                   "n_sequences": len(results), "n_flagged": len(flagged),
                   "n_cuts": total_cuts, "results": flagged}, f, indent=2)
    with open(os.path.join(out_dir, "cuts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "folder", "n_frames", "before_name", "after_name",
                    "before_index", "after_index", "distance", "score"])
        for r in flagged:
            for c in r["cuts"]:
                w.writerow([r["seq_id"], r["folder"], r["n_frames"],
                            c["before_name"], c["after_name"], c["before_index"],
                            c["after_index"], f"{c['distance']:.5f}", f"{c['score']:.2f}"])

    # visual report — render previews for the top N cuts only
    shown = []
    budget = args.max_report
    for r in flagged:
        for c in sorted(r["cuts"], key=lambda x: -x["score"]):
            if len(shown) >= budget:
                break
            shown.append((r, c))
        if len(shown) >= budget:
            break

    cards = []
    for r, c in shown:
        try:
            before = _b64_png(load_rgb_preview(
                os.path.join(r["folder"], c["before_name"]), args.preview_width))
            after = _b64_png(load_rgb_preview(
                os.path.join(r["folder"], c["after_name"]), args.preview_width))
            imgs = (f'<img src="data:image/png;base64,{before}" alt="before">'
                    f'<span class="arrow">&#8594;</span>'
                    f'<img src="data:image/png;base64,{after}" alt="after">')
        except Exception as e:
            imgs = f'<div class="err">preview failed: {e}</div>'
        cards.append(f"""
      <div class="card">
        <div class="imgs">{imgs}</div>
        <div class="meta">
          <div class="score">score {c['score']:.1f}<span class="dist"> (d={c['distance']:.4f}, median={r['stats'].get('median', 0):.4f})</span></div>
          <div class="files"><code>{c['before_name']}</code> &#8594; <code>{c['after_name']}</code></div>
          <div class="path">{r['folder']}</div>
          <div class="path">frames {c['before_index']} &#8594; {c['after_index']} of {r['n_frames']}</div>
        </div>
      </div>""")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cut detection report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#141414; color:#e8e8e8; }}
 header {{ padding: 20px 28px; background:#1d1d1d; border-bottom:1px solid #333; position:sticky; top:0; }}
 h1 {{ margin:0 0 6px; font-size:19px; font-weight:600; }}
 .sub {{ color:#9a9a9a; font-size:13px; line-height:1.7; }}
 .sub b {{ color:#e8e8e8; }}
 .wrap {{ padding: 20px 28px 60px; }}
 .card {{ background:#1d1d1d; border:1px solid #333; border-radius:8px; padding:14px; margin-bottom:16px; }}
 .imgs {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
 .imgs img {{ border-radius:4px; background:#000; max-width:{args.preview_width}px; }}
 .arrow {{ color:#e0a030; font-size:22px; }}
 .meta {{ margin-top:10px; font-size:12px; color:#9a9a9a; line-height:1.7; }}
 .score {{ color:#e0a030; font-weight:600; font-size:14px; }}
 .dist {{ color:#777; font-weight:400; font-size:12px; }}
 .files code {{ background:#262626; padding:1px 5px; border-radius:3px; color:#cfcfcf; }}
 .path {{ color:#6f6f6f; font-family: ui-monospace, Menlo, monospace; font-size:11px; }}
 .err {{ color:#d06060; }}
 .none {{ color:#7bbf7b; padding:24px 0; }}
</style></head><body>
<header>
  <h1>Cut detection report</h1>
  <div class="sub">
    root <b>{args.dataset_path}</b><br>
    scanned <b>{len(results)}</b> sequences in {elapsed:.1f}s &middot;
    flagged <b>{len(flagged)}</b> sequences &middot; <b>{total_cuts}</b> suspected cuts
    &middot; showing top <b>{len(shown)}</b>
    {f'&middot; <span style="color:#d06060">{len(errors)} errors</span>' if errors else ''}<br>
    threshold {args.threshold}x median &middot; min_abs {args.min_abs} &middot;
    thumb {args.size}px &middot; stride {args.stride}
  </div>
</header>
<div class="wrap">
{''.join(cards) if cards else '<div class="none">No suspected cuts found with these settings. Try a lower --threshold.</div>'}
</div></body></html>"""

    index = os.path.join(out_dir, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(html)

    if errors:
        with open(os.path.join(out_dir, "errors.txt"), "w") as f:
            for r in errors:
                f.write(f"{r['folder']}\t{r['error']}\n")

    return index, len(flagged), total_cuts, len(errors)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Detect hidden scene cuts inside frame sequences")
    ap.add_argument("dataset_path", nargs="?", help="dataset root to scan")
    ap.add_argument("--out", default="cuts_report", help="output folder (default: cuts_report)")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="flag when a boundary is this many times the sequence median (default 5)")
    ap.add_argument("--min_abs", type=float, default=0.04,
                    help="absolute dissimilarity floor; suppresses noise spikes in static clips (default 0.04)")
    ap.add_argument("--size", type=int, default=64, help="analysis thumbnail size in px (default 64)")
    ap.add_argument("--stride", type=int, default=1, help="analyse every Nth frame (default 1)")
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames analysed per sequence (0 = all)")
    ap.add_argument("--workers", type=int, default=8, help="parallel processes (default 8)")
    ap.add_argument("--limit", type=int, default=0, help="only scan the first N sequences (0 = all)")
    ap.add_argument("--max_window", type=int, default=12, help="manifest max_window (default 12)")
    ap.add_argument("--max_report", type=int, default=200, help="max cuts rendered in the HTML (default 200)")
    ap.add_argument("--preview_width", type=int, default=340, help="report thumbnail width (default 340)")
    ap.add_argument("--min_length", type=int, default=3, help="ignore sequences shorter than this (default 3)")
    ap.add_argument("--self_test", action="store_true", help="run detector self-tests on synthetic data and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.dataset_path:
        ap.error("dataset_path is required (or use --self_test)")

    from data import build_manifest

    t0 = time.time()
    manifest = build_manifest(args.dataset_path, max_window=args.max_window,
                              read_headers=False, verbose=True)
    seqs = [s for s in manifest.sequences if s.count >= args.min_length]
    if args.limit:
        seqs = seqs[:args.limit]
    if not seqs:
        print("no sequences found")
        return
    n_frames = sum(s.count for s in seqs)
    print(f"[cuts] scanning {len(seqs)} sequences / {n_frames} frames "
          f"with {args.workers} workers (thumb {args.size}px, stride {args.stride})")

    payloads = [(s.seq_id, s.folder, [s.basename_at(i) for i in range(s.count)])
                for s in seqs]

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers),
                             initializer=_init_worker,
                             initargs=(args.size, args.threshold, args.min_abs,
                                       args.stride, args.max_frames)) as ex:
        futures = [ex.submit(_scan_sequence, p) for p in payloads]
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(futures):
                hit = sum(1 for r in results if r["cuts"])
                print(f"\r[cuts] {done}/{len(futures)} sequences, {hit} flagged", end="", flush=True)
    print()

    elapsed = time.time() - t0
    index, n_flagged, n_cuts, n_err = write_report(results, args.out, args, elapsed)
    print(f"[cuts] done in {elapsed:.1f}s — {n_flagged} sequences flagged, "
          f"{n_cuts} suspected cuts" + (f", {n_err} errors" if n_err else ""))
    print(f"[cuts] open {os.path.abspath(index)}")
    print(f"[cuts] also wrote cuts.json / cuts.csv in {os.path.abspath(args.out)}")


# --------------------------------------------------------------------------
# self test (synthetic — no OIIO needed)
# --------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(0)
    ok = fail = 0

    def check(cond, msg):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL: {msg}")

    def smooth_plate(h, w, seed, cells=8):
        """Low-frequency texture — real footage is spatially smooth, so a 1px pan
        changes little. (Per-pixel noise would make any pan look like a cut.)"""
        r = np.random.default_rng(seed)
        small = r.random((max(2, h // cells), max(2, w // cells))).astype(np.float32)
        yi = np.linspace(0, small.shape[0] - 1, h)
        xi = np.linspace(0, small.shape[1] - 1, w)
        y0 = np.clip(np.floor(yi).astype(int), 0, small.shape[0] - 2)
        x0 = np.clip(np.floor(xi).astype(int), 0, small.shape[1] - 2)
        fy = (yi - y0)[:, None]
        fx = (xi - x0)[None, :]
        a = small[np.ix_(y0, x0)]
        b = small[np.ix_(y0, x0 + 1)]
        c = small[np.ix_(y0 + 1, x0)]
        d = small[np.ix_(y0 + 1, x0 + 1)]
        return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy).astype(np.float32)

    def pan(n, size=64, speed=1.0, seed=0):
        """A smoothly panning shot: one smooth plate sampled at shifting offsets."""
        base = smooth_plate(size, size * 4, seed)
        out = []
        for i in range(n):
            off = int(i * speed) % (size * 3)
            out.append((i, base[:, off:off + size].copy()))
        return out

    print("1. clean pan -> no cuts")
    thumbs = pan(20, speed=1.0, seed=1)
    cuts, stats = analyse_series(thumbs, threshold=5.0, min_abs=0.04)
    check(len(cuts) == 0, f"clean pan flagged {len(cuts)} cuts")

    print("2. pan with a hard cut -> exactly one cut at the seam")
    a = pan(12, speed=1.0, seed=2)
    b = pan(12, speed=1.0, seed=99)
    thumbs = [(i, f[1]) for i, f in enumerate(a + b)]
    cuts, stats = analyse_series(thumbs, threshold=5.0, min_abs=0.04)
    check(len(cuts) == 1, f"expected 1 cut, got {len(cuts)}")
    if cuts:
        check(cuts[0]["before_index"] == 11 and cuts[0]["after_index"] == 12,
              f"cut located at {cuts[0]['before_index']}->{cuts[0]['after_index']} (want 11->12)")
        check(cuts[0]["score"] > 5.0, f"score {cuts[0]['score']:.1f} should exceed threshold")

    print("3. static shot with sensor noise -> no cuts (min_abs floor)")
    base = smooth_plate(64, 64, seed=7)
    thumbs = [(i, np.clip(base + rng.normal(0, 0.004, base.shape), 0, 1).astype(np.float32))
              for i in range(20)]
    cuts, _ = analyse_series(thumbs, threshold=5.0, min_abs=0.04)
    check(len(cuts) == 0, f"noisy static clip flagged {len(cuts)} cuts")

    print("4. fast motion (high median) -> not flagged")
    thumbs = pan(20, speed=12.0, seed=3)
    cuts, stats = analyse_series(thumbs, threshold=5.0, min_abs=0.04)
    check(len(cuts) == 0, f"fast pan flagged {len(cuts)} (median={stats['median']:.4f})")

    print("5. two cuts -> both found")
    seg = lambda s: [f[1] for f in pan(8, speed=1.0, seed=s)]
    frames = seg(10) + seg(20) + seg(30)
    thumbs = [(i, f) for i, f in enumerate(frames)]
    cuts, _ = analyse_series(thumbs, threshold=5.0, min_abs=0.04)
    check(len(cuts) == 2, f"expected 2 cuts, got {len(cuts)}")
    if len(cuts) == 2:
        check([c["after_index"] for c in cuts] == [8, 16],
              f"cut positions {[c['after_index'] for c in cuts]} (want [8, 16])")

    print("6. dissolve (gradual) -> not a hard cut")
    A = smooth_plate(64, 64, seed=41)
    B = smooth_plate(64, 64, seed=42)
    thumbs = [(i, (A * (1 - i / 19) + B * (i / 19)).astype(np.float32)) for i in range(20)]
    cuts, _ = analyse_series(thumbs, threshold=5.0, min_abs=0.04)
    check(len(cuts) == 0, f"dissolve flagged {len(cuts)} (expected 0 — gradual, not hard)")

    print("7. edge cases")
    check(analyse_series([], 5.0, 0.04)[0] == [], "empty input")
    check(analyse_series([(0, rng.random((8, 8)).astype(np.float32))], 5.0, 0.04)[0] == [],
          "single frame")
    d = np.array([0.1, 0.1, 0.1])
    sc, med = score_distances(d, min_abs=0.0)
    check(np.allclose(sc, 1.0) and abs(med - 0.1) < 1e-9, f"uniform distances -> score 1 ({sc})")
    x = rng.random((32, 32)).astype(np.float32)
    check(pair_distance(x, x) == 0.0, "identical frames -> distance 0")
    check(pair_distance(np.zeros((16, 16), np.float32), np.ones((16, 16), np.float32)) > 0.9,
          "black vs white -> near-max distance")

    print(f"\n{ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)