"""Pixel-path tests: rotation, flips, collate padding+mask, cache integration.

Uses a synthetic reader (no EXR / OIIO needed) but does need torch. Skips
cleanly where torch is absent; run it on the training box to validate."""
try:
    import torch
except ImportError:
    print("SKIP: torch not available in this environment — run on the training box")
    raise SystemExit(0)

from data.descriptions import Sequence
from data.sampler import SampleSpec, resized_hw, family_and_long, ROT_NONE, ROT_CW, ROT_CCW
from data.dataset import TimewarpDataset, rotate_chw
from data.collate import collate_timewarp
from data.cache import FrameCache

ok = fail = 0
def check(cond, msg):
    global ok, fail
    if cond: ok += 1
    else:
        fail += 1
        print(f"  FAIL: {msg}")

FS = 448

def make_seq(seq_id, h, w, count=12):
    return Sequence(seq_id=seq_id, folder=f"/d/{seq_id}", left="a.", tail=".exr",
                    pad=4, step=1, start=1, count=count, height=h, width=w, max_window=12)

# a reader that returns a position-encoded tensor of the requested size and
# counts calls per path (so we can see cache hits)
CALLS = {}
def fake_reader(path, out_h, out_w, channels=3):
    CALLS[path] = CALLS.get(path, 0) + 1
    ys = torch.arange(out_h).float().view(1, out_h, 1).expand(1, out_h, out_w)
    xs = torch.arange(out_w).float().view(1, 1, out_w).expand(1, out_h, out_w)
    idc = torch.full((1, out_h, out_w), float(hash(path) % 997))
    return torch.cat([ys, xs, idc], dim=0)[:channels]

# ---- 1. rotation: shape matches the planner's geometry --------------------
print("1. rotation shape")
for (h, w) in [(1080, 1920), (1920, 1080), (1000, 3000), (500, 500)]:
    ch, cw = resized_hw(h, w, ROT_NONE, FS)          # canonical
    canon = torch.zeros(3, ch, cw)
    for rot in (ROT_NONE, ROT_CW, ROT_CCW):
        out = rotate_chw(canon, rot)
        exp_h, exp_w = resized_hw(h, w, rot, FS)
        check(tuple(out.shape[1:]) == (exp_h, exp_w),
              f"h={h} w={w} rot={rot}: {tuple(out.shape[1:])} vs ({exp_h},{exp_w})")

# ---- 2. rotation is a real 90 turn and invertible -------------------------
print("2. rotation invertible")
t = fake_reader("/p/x", FS, 800)
cw = rotate_chw(t, ROT_CW)
back = rotate_chw(cw, ROT_CCW)                        # CW then CCW -> identity
check(torch.equal(back, t), "CW then CCW restores original")
check(not torch.equal(cw, t) if t.shape[1] != t.shape[2] else True, "CW changes non-square tensor")
four = t
for _ in range(4):
    four = rotate_chw(four, ROT_CW)
check(torch.equal(four, t), "4x CW == identity")

# ---- 3. flips consistent across the triplet + deterministic ---------------
print("3. flips")
seq = make_seq("s0", 1080, 1920)
ds = TimewarpDataset([seq], frame_size=FS, reader=fake_reader,
                     hflip_prob=1.0, vflip_prob=0.0, cflip_prob=0.0, seed=5)
spec = SampleSpec("s0", 0, 3, 11, 0.3, ROT_NONE)
a = ds[spec]
# horizontal flip applied -> img differs from an unflipped load, flipped on dim W
ch, cw = resized_hw(1080, 1920, ROT_NONE, FS)
raw = fake_reader(seq.path_at(0), ch, cw)
check(torch.equal(a["img0"], raw.flip([2])), "hflip applied to dim W")
# same transform on all three frames: build a dataset returning identical frames
class ConstReader:
    def __call__(self, path, out_h, out_w, channels=3):
        g = torch.arange(out_h * out_w).float().view(1, out_h, out_w)
        return torch.cat([g, g + 1, g + 2], 0)[:channels]
ds_c = TimewarpDataset([seq], frame_size=FS, reader=ConstReader(),
                       hflip_prob=0.5, vflip_prob=0.5, cflip_prob=0.5, seed=9)
s = ds_c[SampleSpec("s0", 0, 1, 2, 0.5, ROT_NONE)]
# all three frames came from identical content -> after identical aug they stay equal
check(torch.equal(s["img0"], s["img1"]) and torch.equal(s["img1"], s["img2"]),
      "same flip on whole triplet")
# determinism: same spec+epoch -> identical output
check(torch.equal(ds[spec]["img0"], ds[spec]["img0"]), "getitem deterministic")
ds.set_epoch(1)
b = ds[spec]
# different epoch may change the flip draw; at least it must not error and stays valid shape
check(b["img0"].shape == a["img0"].shape, "epoch change keeps shape")

# ---- 4. collate: padding + mask ------------------------------------------
print("4. collate")
def sample(h, w, ratio=0.5, sid="s"):
    img = torch.ones(3, h, w)
    return {"img0": img, "img1": img * 2, "img2": img * 3, "ratio": ratio,
            "spec": SampleSpec(sid, 0, 1, 2, ratio, ROT_NONE)}
# same family (landscape): H=448 fixed, W varies within band
samples = [sample(448, 800), sample(448, 816), sample(448, 848)]
batch = collate_timewarp(samples, return_mask=True)
check(batch["img0"].shape == (3, 3, 448, 848), f"padded to max W: {tuple(batch['img0'].shape)}")
check(batch["ratio"].shape == (3,), "ratio batched")
check(len(batch["specs"]) == 3, "specs carried")
# mask: ones over real region, zeros in pad
m = batch["mask"]
check(m.shape == (3, 1, 448, 848), f"mask shape {tuple(m.shape)}")
check(m[0, 0, :, :800].min() == 1.0 and m[0, 0, :, 800:].max() == 0.0, "sample 0 mask boundary at 800")
check(m[2, 0].min() == 1.0, "sample 2 (widest) fully valid")
# padded pixels are zero in the image too
check(batch["img0"][0, :, :, 800:].abs().max() == 0.0, "padding is zero in image")
# real region preserved
check(batch["img1"][0, :, :, :800].min() == 2.0, "real content intact after pad")

# ---- 5. dataset end-to-end + cache hits -----------------------------------
print("5. dataset + cache")
CALLS.clear()
seqA = make_seq("A", 1080, 1920, count=12)
cache = FrameCache(max_items=64)
ds2 = TimewarpDataset([seqA], frame_size=FS, reader=fake_reader, cache=cache, seed=1)
out = ds2[SampleSpec("A", 0, 3, 11, 0.25, ROT_NONE)]
check(set(out.keys()) == {"img0", "img1", "img2", "ratio", "spec"}, f"keys {set(out.keys())}")
check(out["ratio"] == 0.25, "ratio passed through")
ch, cw = resized_hw(1080, 1920, ROT_NONE, FS)
check(tuple(out["img0"].shape) == (3, ch, cw), f"frame shape {tuple(out['img0'].shape)}")
# three distinct frames -> three reader calls
check(sum(CALLS.values()) == 3, f"3 reads for 3 frames: {sum(CALLS.values())}")
# a second sample sharing frame index 0 -> cache hit, no new read for that path
n_before = sum(CALLS.values())
_ = ds2[SampleSpec("A", 0, 5, 9, 0.4, ROT_NONE)]   # frame 0 shared with previous
shared_path = seqA.path_at(0)
check(CALLS[shared_path] == 1, f"shared frame cached (1 read): {CALLS[shared_path]}")
check(cache.hits >= 1, f"cache registered hit(s): {cache.hits}")

# ---- 6. rotated sample has planner-consistent shape -----------------------
print("6. rotated sample shape")
outr = ds2[SampleSpec("A", 0, 3, 11, 0.25, ROT_CW)]
fam, lng = family_and_long(1080, 1920, ROT_CW, FS)
eh, ew = resized_hw(1080, 1920, ROT_CW, FS)
check(tuple(outr["img0"].shape[1:]) == (eh, ew), f"rotated frame {tuple(outr['img0'].shape[1:])} vs ({eh},{ew})")
check(max(eh, ew) == lng, "long side consistent with planner")

print(f"\n{ok} passed, {fail} failed")
