"""Sampler tests — geometry, bucketing, batch validity, determinism, DDP."""
import math
from collections import Counter
from data.descriptions import Sequence
from data.sampler import (
    snap, rotated_hw, resized_hw, family_and_long, _band,
    normalize_rotation_weights, assign_rotations, build_buckets,
    shard_sequences, TimewarpBatchSampler, SampleSpec,
    LANDSCAPE, PORTRAIT, ROT_NONE, ROT_CW, ROT_CCW,
)
import random

ok = fail = 0
def check(cond, msg):
    global ok, fail
    if cond: ok += 1
    else:
        fail += 1
        print(f"  FAIL: {msg}")

def make_seq(seq_id, h, w, count=12, max_window=12):
    """Directly build a Sequence with a known native size."""
    return Sequence(seq_id=seq_id, folder=f"/d/{seq_id}", left="a.", tail=".exr",
                    pad=4, step=1, start=1, count=count, height=h, width=w,
                    max_window=max_window)

FS = 448

# ---- 1. geometry ----------------------------------------------------------
print("1. geometry")
check(snap(910, 16) == 912 and snap(7, 16) == 16 and snap(448, 16) == 448, "snap")
check(rotated_hw(1080, 1920, ROT_NONE) == (1080, 1920), "rot0 no swap")
check(rotated_hw(1080, 1920, ROT_CW) == (1920, 1080), "rot90 swaps")
check(rotated_hw(1080, 1920, ROT_CCW) == (1920, 1080), "rot-90 swaps")

H, W = resized_hw(1080, 1920, ROT_NONE, FS)
check(H == FS and W % 16 == 0 and W > H, f"landscape resize ({H},{W})")
check(W == snap(1920 * FS / 1080, 16), f"long side scaled ({H},{W})")

# rotating a landscape by 90 makes it portrait, same long magnitude
fam0, long0 = family_and_long(1080, 1920, ROT_NONE, FS)
fam90, long90 = family_and_long(1080, 1920, ROT_CW, FS)
check(fam0 == LANDSCAPE and fam90 == PORTRAIT, f"rotation flips family {fam0}/{fam90}")
check(long0 == long90, f"same long magnitude {long0}/{long90}")

# both axes always /16
for (h, w) in [(1080, 1920), (1920, 1080), (500, 500), (720, 1280), (2160, 3840), (1000, 3000)]:
    for rot in (ROT_NONE, ROT_CW, ROT_CCW):
        Hh, Ww = resized_hw(h, w, rot, FS)
        check(Hh % 16 == 0 and Ww % 16 == 0 and min(Hh, Ww) == FS,
              f"resize invariants h={h} w={w} rot={rot} -> ({Hh},{Ww})")

# square -> landscape by convention, exactly frame_size square
Hs, Ws = resized_hw(500, 500, ROT_NONE, FS)
check(Hs == FS and Ws == FS, f"square -> ({Hs},{Ws})")

# ---- 2. rotation assignment ----------------------------------------------
print("2. rotation assignment")
weights = normalize_rotation_weights(rotation_prob=0.5)
check(abs(weights[ROT_NONE] - 0.5) < 1e-9 and abs(weights[ROT_CW] - 0.25) < 1e-9, f"weights {weights}")

seqs = [make_seq(f"s{i:05d}", 1080, 1920) for i in range(20000)]
r1 = assign_rotations(seqs, random.Random(1), weights)
r2 = assign_rotations(seqs, random.Random(1), weights)
check(r1 == r2, "rotation assignment deterministic for same seed")
r3 = assign_rotations(seqs, random.Random(2), weights)
check(r1 != r3, "different seed -> different rotations")
dist = Counter(r1.values())
check(abs(dist[ROT_NONE] / len(seqs) - 0.5) < 0.02, f"~50% unrotated: {dist[ROT_NONE]/len(seqs):.3f}")
check(abs(dist[ROT_CW] / len(seqs) - 0.25) < 0.02, f"~25% CW: {dist[ROT_CW]/len(seqs):.3f}")

# custom weights, all rotation
w_all = normalize_rotation_weights(weights={ROT_CW: 1, ROT_CCW: 1})
r_all = assign_rotations(seqs[:1000], random.Random(1), w_all)
check(ROT_NONE not in set(r_all.values()), "custom weights exclude rot 0")

# ---- 3. bucketing bounds --------------------------------------------------
print("3. bucketing")
# a spread of aspect ratios
mixed = []
for i in range(2000):
    # vary long side widely: 16:9, 4:3, 21:9, square, portrait variants
    shapes = [(1080, 1920), (1080, 1440), (1080, 2520), (1000, 1000),
              (1920, 1080), (1440, 1080), (2160, 3840)]
    h, w = shapes[i % len(shapes)]
    mixed.append(make_seq(f"m{i:05d}", h, w))
rots = {s.seq_id: ROT_NONE for s in mixed}          # fix rotation for a clean check
buckets = build_buckets(mixed, rots, frame_size=FS, pad_tolerance=0.10)

for key, b in buckets.items():
    fam, band = key
    # every sequence in a bucket has the claimed family
    fams = {family_and_long(s.height, s.width, r, FS)[0] for s, r in zip(b.seqs, b.rots)}
    check(fams == {fam}, f"bucket {key} single family: {fams}")
    # long side spread within (1+tol)
    if b.longs:
        ratio = max(b.longs) / min(b.longs)
        check(ratio <= 1.10 + 1e-6, f"bucket {key} long ratio {ratio:.4f} <= 1.10")

# max_long_side drops the extreme ones
warns = []
b2 = build_buckets(mixed, rots, frame_size=FS, max_long_side=1000, warnings=warns)
kept = sum(len(b.seqs) for b in b2.values())
check(kept < len(mixed) and any("max_long_side" in w for w in warns),
      f"max_long_side filter kept {kept}/{len(mixed)}")

# ---- 4. batch validity ----------------------------------------------------
print("4. batch validity")
BS = 8
sampler = TimewarpBatchSampler(mixed, batch_size=BS, frame_size=FS,
                               steps_per_epoch=200, seed=42, rotation_prob=0.5)
by_id = {s.seq_id: s for s in mixed}

n_batches = 0
for batch in sampler:
    n_batches += 1
    check(len(batch) == BS, f"batch size {len(batch)}")
    # all same family (recompute from native + spec rotation)
    fams = set()
    longs = []
    for spec in batch:
        s = by_id[spec.seq_id]
        fam, lng = family_and_long(s.height, s.width, spec.rotation, FS)
        fams.add(fam)
        longs.append(lng)
        # valid frame indices and ratio
        check(0 <= spec.start < s.count and 0 <= spec.gt < s.count and 0 <= spec.end < s.count,
              f"indices in range: {spec}")
        check(0.0 <= spec.ratio <= 1.0, f"ratio in [0,1]: {spec.ratio}")
        check(spec.rotation in (ROT_NONE, ROT_CW, ROT_CCW), f"rotation valid {spec.rotation}")
    check(len(fams) == 1, f"batch mixes families: {fams}")
    if longs:
        check(max(longs) / min(longs) <= 1.10 + 1e-6, f"batch long spread {max(longs)/min(longs):.4f}")
check(n_batches == 200, f"yielded {n_batches} batches (want 200)")

# ---- 5. distinct sequences within a batch (large bucket) ------------------
print("5. within-batch diversity")
# many identical-shape sequences -> one big bucket -> batches should be distinct seqs
same = [make_seq(f"same{i:05d}", 1080, 1920) for i in range(500)]
s5 = TimewarpBatchSampler(same, batch_size=16, frame_size=FS, steps_per_epoch=50,
                          seed=7, rotation_prob=0.0)  # no rotation -> all one bucket
all_distinct = True
for batch in s5:
    ids = [spec.seq_id for spec in batch]
    if len(set(ids)) != len(ids):
        all_distinct = False
        break
check(all_distinct, "large bucket -> distinct sequences per batch")

# small bucket (fewer sequences than batch) must still fill
tiny = [make_seq(f"tiny{i}", 1080, 1920) for i in range(3)]
s_tiny = TimewarpBatchSampler(tiny, batch_size=8, frame_size=FS, steps_per_epoch=10,
                              seed=1, rotation_prob=0.0)
tiny_ok = all(len(b) == 8 for b in s_tiny)
check(tiny_ok, "small bucket fills batch with repeats")

# ---- 6. determinism -------------------------------------------------------
print("6. determinism")
def collect(sampler):
    return [[(s.seq_id, s.start, s.gt, s.end, round(s.ratio, 6), s.rotation) for s in b]
            for b in sampler]

a = TimewarpBatchSampler(mixed, batch_size=BS, frame_size=FS, steps_per_epoch=30, seed=99, epoch=0)
b = TimewarpBatchSampler(mixed, batch_size=BS, frame_size=FS, steps_per_epoch=30, seed=99, epoch=0)
check(collect(a) == collect(b), "same seed+epoch -> identical stream")

a.set_epoch(1)
check(collect(a) != collect(b), "different epoch -> different stream")

c = TimewarpBatchSampler(mixed, batch_size=BS, frame_size=FS, steps_per_epoch=30, seed=100, epoch=0)
check(collect(c) != collect(b), "different seed -> different stream")

# re-iterating the same sampler at the same epoch reproduces
b2 = TimewarpBatchSampler(mixed, batch_size=BS, frame_size=FS, steps_per_epoch=30, seed=99, epoch=0)
check(collect(b2) == collect(b2), "re-iteration reproduces")

# ---- 7. DDP sharding ------------------------------------------------------
print("7. DDP sharding")
pool = [make_seq(f"p{i:05d}", 1080, 1920) for i in range(1000)]
WS = 4
shards = [shard_sequences(pool, rank=r, world_size=WS) for r in range(WS)]
# disjoint and complete
ids_per = [set(s.seq_id for s in sh) for sh in shards]
union = set().union(*ids_per)
check(len(union) == len(pool), f"shards cover all: {len(union)}/{len(pool)}")
for i in range(WS):
    for j in range(i + 1, WS):
        check(ids_per[i].isdisjoint(ids_per[j]), f"shards {i},{j} disjoint")
# balanced
sizes = [len(sh) for sh in shards]
check(max(sizes) - min(sizes) <= 1, f"balanced shards: {sizes}")

# per-rank samplers run same #steps but different content
rank_samplers = [TimewarpBatchSampler(pool, batch_size=BS, frame_size=FS,
                                      steps_per_epoch=20, seed=5, rank=r, world_size=WS)
                 for r in range(WS)]
lens = [len(s) for s in rank_samplers]
check(len(set(lens)) == 1 and lens[0] == 20, f"equal step counts across ranks: {lens}")
streams = [collect(s) for s in rank_samplers]
# each rank draws from its own shard -> seq ids disjoint across rank streams
rank_ids = [set(spec[0] for batch in st for spec in batch) for st in streams]
disjoint_ranks = all(rank_ids[i].isdisjoint(rank_ids[j])
                     for i in range(WS) for j in range(i + 1, WS))
check(disjoint_ranks, "rank streams use disjoint sequences")

# ---- 8. coverage estimate -------------------------------------------------
print("8. step estimate")
s8 = TimewarpBatchSampler(mixed, batch_size=BS, frame_size=FS, seed=1)  # steps=None
total_windows = sum(s.num_windows(True) for s in mixed)
check(s8.steps_per_epoch == math.ceil(total_windows / BS),
      f"nominal steps {s8.steps_per_epoch} vs {math.ceil(total_windows/BS)}")

print(f"\n{ok} passed, {fail} failed")
