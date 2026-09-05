"""Validate the 'fixed' (stab) window mode against a brute-force reference."""
from data.descriptions import num_windows, window_at

ok = fail = 0
def check(c, m):
    global ok, fail
    if c: ok += 1
    else:
        fail += 1; print(f"  FAIL: {m}")

def brute(L, W, bidir, mode):
    out = []
    Weff = min(W, L)
    sizes = [Weff] if mode == "fixed" else list(range(3, Weff + 1))
    for w in sizes:
        if w < 3: continue
        for pos in range(0, L - w + 1):
            for gt_off in range(1, w - 1):
                out.append((pos, pos + gt_off, pos + w - 1, gt_off / (w - 1)))
                if bidir:
                    out.append((pos + w - 1, pos + gt_off, pos, 1 - gt_off / (w - 1)))
    return out

class S:
    def __init__(self, L, W): self.count = L; self.max_window = W; self.seq_id = "s"

print("1. fixed mode matches brute force (count + enumeration)")
for L in [3, 4, 5, 8, 13, 24, 40]:
    for W in [3, 5, 12, 24, 60]:
        for bidir in [True, False]:
            ref = brute(L, W, bidir, "fixed")
            n = num_windows(L, W, bidir, "fixed")
            check(n == len(ref), f"count L={L} W={W} bidir={bidir}: {n} vs {len(ref)}")
            got = [window_at(S(L, W), k, bidir, "fixed") for k in range(n)]
            gt = [(g.start, g.gt, g.end, round(g.ratio, 9)) for g in got]
            rt = [(a, b, c, round(d, 9)) for a, b, c, d in ref]
            check(gt == rt, f"enumeration L={L} W={W} bidir={bidir}")

print("2. full mode unchanged (regression)")
for L in [5, 13, 20]:
    for W in [3, 12]:
        for bidir in [True, False]:
            ref = brute(L, W, bidir, "full")
            n = num_windows(L, W, bidir, "full")
            check(n == len(ref), f"full count L={L} W={W}: {n} vs {len(ref)}")
            got = [window_at(S(L, W), k, bidir, "full") for k in range(n)]
            gt = [(g.start, g.gt, g.end, round(g.ratio, 9)) for g in got]
            check(gt == [(a,b,c,round(d,9)) for a,b,c,d in ref], f"full enum L={L} W={W}")
# default arg is still 'full'
check(num_windows(20, 12) == num_windows(20, 12, True, "full"), "default mode is full")

print("3. fixed mode uses only the max-window baseline")
L, W = 30, 24
specs = [window_at(S(L, W), k, False, "fixed") for k in range(num_windows(L, W, False, "fixed"))]
spans = {abs(s.end - s.start) for s in specs}
check(spans == {W - 1}, f"all spans == max_window-1 ({W-1}); got {spans}")
check(all(min(s.start, s.end) < s.gt < max(s.start, s.end) for s in specs), "gt strictly interior")
# every interior gt position is covered for a given window position
first = [s for s in specs if s.start == 0]
check(sorted(s.gt for s in first) == list(range(1, W - 1)),
      f"all interior gts at pos0: {sorted(s.gt for s in first)[:5]}...")

print("4. fixed yields far fewer windows than full")
nf = num_windows(48, 24, True, "full")
nx = num_windows(48, 24, False, "fixed")
check(nx < nf / 10, f"fixed {nx} vs full {nf} (expect a big reduction)")
print(f"     L=48 W=24: full(bidir)={nf}  fixed(uni)={nx}")

print("5. short runs degrade gracefully")
check(num_windows(2, 24, True, "fixed") == 0, "len 2 -> 0 windows")
check(num_windows(3, 24, True, "fixed") == 2, f"len 3 -> 2 (fw+bw), got {num_windows(3,24,True,'fixed')}")
check(num_windows(3, 24, False, "fixed") == 1, "len 3 uni -> 1")
# W clamps to L
check(num_windows(10, 100, False, "fixed") == num_windows(10, 10, False, "fixed"), "W clamps to L")

print("6. ratio correctness at the extremes")
sp = [window_at(S(25, 25), k, False, "fixed") for k in range(num_windows(25, 25, False, "fixed"))]
check(abs(min(s.ratio for s in sp) - 1/24) < 1e-9, "min ratio = 1/(W-1)")
check(abs(max(s.ratio for s in sp) - 23/24) < 1e-9, "max ratio = (W-2)/(W-1)")

print(f"\n{ok} passed, {fail} failed")