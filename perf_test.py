"""
perf_test.py — data-pipeline performance & memory harness for the new dataset.

Runs a short loop that builds the pipeline, pulls batches, moves them to CUDA,
and discards them — measuring throughput, CPU/GPU/pinned memory, cache hit rate,
and surfacing the failure modes that matter on slow storage (starvation, pinned
RAM blowup, worker stalls, variable-shape VRAM spikes).

Run from the project root (so `import data` resolves):

    ./packages/appenv/bin/python -m perf_test /path/to/train_data \
        --steps 300 --batch_size 16 --frame_size 448 \
        --num_workers 8 --pool_size 48 --reuse 4 --cache_items 256

Compare configs by re-running with different flags. Use --profile_stalls to see
where time goes, and --no_pool / --no_cuda to isolate layers.
"""
import os, sys, time, argparse, statistics, gc, threading

def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def get_rss():
    try:
        import resource
        # ru_maxrss is KB on Linux, bytes on macOS
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v * 1024 if sys.platform != "darwin" else v
    except Exception:
        return 0


def _devidx(dev):
    return dev.index if getattr(dev, "index", None) is not None else 0


def _cuda_try(fn, *a, **k):
    """Best-effort CUDA call. Some builds reject even a valid device arg on the
    memory/sync APIs, so instrumentation must never crash the actual run."""
    try:
        return fn(*a, **k)
    except Exception:
        return None


def cuda_mem(torch):
    # no-arg forms use the current device and are the most build-portable
    try:
        return {
            "max_alloc": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        }
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description="Data pipeline perf/memory harness")
    ap.add_argument("dataset_path")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup_steps", type=int, default=20, help="steps excluded from timing")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--frame_size", type=int, default=448)
    ap.add_argument("--max_window", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--cache_items", type=int, default=256)
    ap.add_argument("--pool_size", type=int, default=48)
    ap.add_argument("--reuse", type=int, default=4)
    ap.add_argument("--pool_order", type=str, default="random", choices=["random", "sequential"])
    ap.add_argument("--pad_tolerance", type=float, default=0.10)
    ap.add_argument("--rotation_prob", type=float, default=0.5)
    ap.add_argument("--max_long_side", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--no_pool", action="store_true", help="bypass reuse-pool, iterate DataLoader directly")
    ap.add_argument("--no_cuda", action="store_true", help="skip the H2D copy (measure CPU side only)")
    ap.add_argument("--simulate_compute_ms", type=float, default=0.0, help="sleep per step to mimic model time")
    ap.add_argument("--profile_stalls", action="store_true", help="report per-step wait vs compute split")
    args = ap.parse_args()

    import torch
    from data import (build_manifest, split_sequences, TimewarpDataset,
                      TimewarpBatchSampler, build_dataloader, BatchPool)

    dev = torch.device(f"cuda:{args.device}") if (torch.cuda.is_available() and not args.no_cuda) else torch.device("cpu")
    use_cuda = dev.type == "cuda"
    if use_cuda:
        # force a CUDA context; if the device can't actually be used (driver/build
        # mismatch) fall back to CPU timing rather than dying on a stats call
        _cuda_try(torch.cuda.set_device, _devidx(dev))
        if _cuda_try(lambda: torch.zeros(1, device=dev)) is None:
            print(" !! CUDA is_available() but this device can't be used — falling back to CPU timing.")
            print("    Likely this interpreter's torch/CUDA doesn't match the driver. Try --no_cuda,")
            print("    or run with the same python you train with.")
            dev = torch.device("cpu")
            use_cuda = False
    print(f"== perf harness ==\n device={dev}  pin_memory={args.pin_memory}  workers={args.num_workers}")

    # ---- build pipeline (time the scan/manifest separately — it's one-time) ----
    t0 = time.time()
    manifest = build_manifest(args.dataset_path, max_window=args.max_window,
                              read_headers=True, verbose=True)
    t_scan = time.time() - t0
    split = split_sequences(manifest.sequences, fractions=(1.0, 0.0, 0.0),
                            seed=args.seed, verbose=True)
    seqs = split.train
    if not seqs:
        print("!! no usable sequences found — check the path / EXR naming")
        return
    total_windows = sum(s.num_windows(True) for s in seqs if s.height is not None)
    print(f" manifest: {len(seqs)} sequences, {manifest.total_frames()} frames, "
          f"{total_windows} windows, scan {t_scan:.1f}s")

    dataset = TimewarpDataset(seqs, frame_size=args.frame_size, seed=args.seed)
    sampler = TimewarpBatchSampler(
        seqs, batch_size=args.batch_size, frame_size=args.frame_size,
        pad_tolerance=args.pad_tolerance, rotation_prob=args.rotation_prob,
        max_long_side=(args.max_long_side or None),
        steps_per_epoch=args.steps + args.warmup_steps + 5,
        seed=args.seed, verbose=True)
    loader = build_dataloader(dataset, sampler, num_workers=args.num_workers,
                              cache_items=args.cache_items, pin_memory=args.pin_memory,
                              return_mask=False)

    if args.no_pool:
        source = loader
        print(" source: DataLoader (pool bypassed)")
    else:
        source = BatchPool(loader, steps_per_epoch=args.steps + args.warmup_steps + 5,
                           size=args.pool_size, reuse=args.reuse, order=args.pool_order, seed=args.seed)
        print(f" source: BatchPool size={args.pool_size} reuse={args.reuse} order={args.pool_order}")

    if use_cuda:
        _cuda_try(torch.cuda.reset_peak_memory_stats)

    # ---- the loop ----
    step_times, wait_times, move_times = [], [], []
    shapes = {}
    peak_batch_bytes = 0
    n = 0
    baseline_rss = get_rss()
    loop_start = time.time()
    last = time.time()

    it = iter(source)
    try:
        while n < args.steps + args.warmup_steps:
            t_wait0 = time.time()
            batch = next(it)                     # <-- blocks here if starved
            t_wait = time.time() - t_wait0

            t_move0 = time.time()
            b = {}
            bb = 0
            for k in ("img0", "img1", "img2"):
                v = batch[k]
                bb += v.element_size() * v.nelement()
                b[k] = v.to(dev, non_blocking=args.pin_memory) if not args.no_cuda else v
            if use_cuda:
                try:
                    torch.cuda.synchronize()              # make the H2D cost real/measurable
                except Exception:
                    pass
            t_move = time.time() - t_move0
            peak_batch_bytes = max(peak_batch_bytes, bb)

            # record the padded shape distribution (variable-size VRAM risk)
            sh = tuple(batch["img0"].shape[1:])
            shapes[sh] = shapes.get(sh, 0) + 1

            if args.simulate_compute_ms:
                time.sleep(args.simulate_compute_ms / 1000.0)

            del b, batch                          # discard (as requested)

            now = time.time()
            if n >= args.warmup_steps:
                step_times.append(now - last)
                wait_times.append(t_wait)
                move_times.append(t_move)
            last = now
            n += 1
            if n % 50 == 0:
                print(f"  step {n}/{args.steps + args.warmup_steps} ...")
    finally:
        if hasattr(it, "close"):
            it.close()

    wall = time.time() - loop_start
    timed = len(step_times)
    if timed == 0:
        print("!! no timed steps — increase --steps")
        return

    # ---- report ----
    def stat(xs):
        xs = sorted(xs)
        p50 = xs[len(xs)//2]
        p95 = xs[min(len(xs)-1, int(len(xs)*0.95))]
        return statistics.mean(xs), p50, p95, max(xs)

    m_step, p50, p95, mx = stat(step_times)
    ips = 1.0 / m_step
    print("\n================ RESULTS ================")
    print(f" timed steps        : {timed}  (after {args.warmup_steps} warmup)")
    print(f" throughput         : {ips:.2f} batches/s  |  {ips*args.batch_size:.1f} samples/s")
    print(f" step time  mean/p50/p95/max : {m_step*1e3:.1f} / {p50*1e3:.1f} / {p95*1e3:.1f} / {mx*1e3:.1f} ms")
    if args.profile_stalls:
        mw, w50, w95, wmx = stat(wait_times)
        mv, v50, v95, vmx = stat(move_times)
        print(f" wait-for-batch     : mean {mw*1e3:.1f}ms  p95 {w95*1e3:.1f}ms  max {wmx*1e3:.1f}ms  (<- starvation shows here)")
        print(f" H2D move           : mean {mv*1e3:.1f}ms  p95 {v95*1e3:.1f}ms")
        stall_frac = sum(wait_times) / sum(step_times)
        print(f" fraction of time WAITING on data : {stall_frac*100:.1f}%")
        if stall_frac > 0.25:
            print("   >> data-bound: raise --reuse / --num_workers / --pool_size, or --cache_items")

    print(f"\n peak batch (3 imgs): {human(peak_batch_bytes)}   "
          f"est. pool RAM (size x batch): {human(peak_batch_bytes * args.pool_size)}")
    rss = get_rss()
    print(f" process RSS         : {human(rss)}  (grew {human(rss - baseline_rss)} during loop)")
    if use_cuda:
        cm = cuda_mem(torch)
        if cm:
            print(f" CUDA peak alloc     : {human(cm['max_alloc'])}   peak reserved: {human(cm['max_reserved'])}")
        else:
            print(" CUDA peak alloc     : (stats API unavailable on this build)")

    print(f"\n padded shapes seen  : {len(shapes)} distinct")
    for sh, c in sorted(shapes.items(), key=lambda x: -x[1])[:8]:
        print(f"    {sh}: {c}  ({human(3*sh[0]*sh[1]*sh[2]*args.batch_size*4)}/batch img-set)")
    if len(shapes) > 1:
        big = max(shapes, key=lambda s: s[0]*s[1])
        small = min(shapes, key=lambda s: s[0]*s[1])
        ratio = (big[0]*big[1]) / (small[0]*small[1])
        print(f"    largest/smallest area ratio: {ratio:.2f}x  "
              f"(>>2x means VRAM must fit the biggest bucket, not the average)")

    if not args.no_pool and hasattr(source, "pool"):
        pass
    # cache stats live per-worker; surface the main-process one if workers==0
    if args.num_workers == 0 and dataset.cache is not None:
        st = dataset.cache.stats()
        print(f"\n frame cache (workers=0): {st['items']} items, {human(st['bytes'])}, "
              f"hit rate {st['hit_rate']*100:.1f}%")
    else:
        print("\n note: frame-cache hit rate is per-worker; run with --num_workers 0 to see it here.")

    print("=========================================")


if __name__ == "__main__":
    main()
