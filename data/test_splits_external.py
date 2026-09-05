"""External held-out root tests, incl. the nested-root trap and symlinks."""
import os, tempfile
from data.descriptions import detect_sequences_in_folder
from data.splits import split_sequences, find_overlap

ok = fail = 0
def check(cond, msg):
    global ok, fail
    if cond: ok += 1
    else:
        fail += 1
        print(f"  FAIL: {msg}")

def mkseqs(folder, n_frames=10, stem="a"):
    return detect_sequences_in_folder(
        folder, [f"{stem}.{n:04d}.exr" for n in range(1, n_frames + 1)])

# ---- 1. external test used verbatim, fractions renormalised --------------
print("1. external test root")
pool = []
for i in range(100):
    pool += mkseqs(f"/data/train/clip_{i:03d}")
ext_test = []
for i in range(7):
    ext_test += mkseqs(f"/data/oos_test/shot_{i:03d}")

s = split_sequences(pool, test_sequences=ext_test,
                    fractions=(0.9, 0.1, 0.1), seed=42, verbose=False)
check(len(s.test) == len(ext_test), f"test from external: {len(s.test)} vs {len(ext_test)}")
check(s.is_externally_held_out("test") and not s.is_externally_held_out("val"),
      f"sources {s.sources}")
# f_te ignored -> val = round(100 * 0.1/1.0) = 10, train = 90
check(len({x.folder for x in s.val}) == 10, f"val units {len({x.folder for x in s.val})}")
check(len({x.folder for x in s.train}) == 90, f"train units {len({x.folder for x in s.train})}")
# no external test folder appears in train/val
tv = {x.folder for x in s.train} | {x.folder for x in s.val}
check(tv.isdisjoint({x.folder for x in s.test}), "external test never in train/val")

# ---- 2. THE TRAP: test root nested inside train root ---------------------
print("2. nested test root is excluded from pool")
pool_nested = []
for i in range(50):
    pool_nested += mkseqs(f"/data/clips/clip_{i:03d}")
# scan of /data/clips also picked these up:
for i in range(5):
    pool_nested += mkseqs(f"/data/clips/test/shot_{i:03d}")
ext = []
for i in range(5):
    ext += mkseqs(f"/data/clips/test/shot_{i:03d}")

s2 = split_sequences(pool_nested, test_sequences=ext,
                     fractions=(0.9, 0.05, 0.05), seed=1, verbose=False)
tv2 = {x.folder for x in s2.train} | {x.folder for x in s2.val}
check(not any("/test/" in f for f in tv2), f"nested test leaked: {[f for f in tv2 if '/test/' in f]}")
check(len(s2.excluded) == 5, f"excluded count {len(s2.excluded)}")

# error mode
try:
    split_sequences(pool_nested, test_sequences=ext, on_overlap="error", verbose=False)
    check(False, "on_overlap='error' should raise")
except ValueError:
    check(True, "")

# ignore mode keeps them
s3 = split_sequences(pool_nested, test_sequences=ext, on_overlap="ignore", verbose=False)
tv3 = {x.folder for x in s3.train} | {x.folder for x in s3.val}
check(any("/test/" in f for f in tv3), "ignore mode should keep overlap")

# held_out_roots catches nesting even with no external sequences
s4 = split_sequences(pool_nested, held_out_roots=["/data/clips/test"], verbose=False)
tv4 = {x.folder for x in s4.train} | {x.folder for x in s4.val} | {x.folder for x in s4.test}
check(not any("/test/" in f for f in tv4), "held_out_roots alone excludes")

# ---- 3. symlink resolution (real filesystem) ----------------------------
print("3. symlinked test dir")
with tempfile.TemporaryDirectory() as tmp:
    real_test = os.path.join(tmp, "real_test")
    os.makedirs(os.path.join(real_test, "shotZ"))
    link_dir = os.path.join(tmp, "train", "linked_test")
    os.makedirs(os.path.join(tmp, "train"))
    os.symlink(real_test, link_dir)

    # pool sees it through the symlink, held-out set through the real path
    pool_sym = mkseqs(os.path.join(link_dir, "shotZ"))
    ext_sym = mkseqs(os.path.join(real_test, "shotZ"))
    overlapping, _ = find_overlap(pool_sym, ext_sym)
    check(len(overlapping) == len(pool_sym),
          f"symlink overlap detected: {len(overlapping)}/{len(pool_sym)}")

# ---- 4. content-signature advisory --------------------------------------
print("4. content signature warning")
pool_copy = mkseqs("/data/train/copied_shot", n_frames=10, stem="a")
ext_orig = mkseqs("/data/oos/original_shot", n_frames=10, stem="a")
_, warns = find_overlap(pool_copy, ext_orig)
check(len(warns) == 1 and "copy" in warns[0], f"content warning: {warns}")
# advisory only — not excluded
s5 = split_sequences(pool_copy, test_sequences=ext_orig, verbose=False)
check(len(s5.train) + len(s5.val) == len(pool_copy), "content match not auto-excluded")
check(len(s5.overlap_warnings) == 1, "warning surfaced on Split")

# ---- 5. external val + external test -------------------------------------
print("5. both external")
s6 = split_sequences(pool, val_sequences=mkseqs("/data/val/v0"),
                     test_sequences=ext_test, seed=3, verbose=False)
check(len({x.folder for x in s6.train}) == 100, f"all pool -> train: {len({x.folder for x in s6.train})}")
check(s6.is_externally_held_out("val") and s6.is_externally_held_out("test"), "both external")

# ---- 6. determinism unchanged --------------------------------------------
print("6. determinism")
a = split_sequences(pool, test_sequences=ext_test, seed=42, verbose=False)
b = split_sequences(pool, test_sequences=ext_test, seed=42, verbose=False)
ids = lambda xs: sorted(x.seq_id for x in xs)
check(ids(a.train) == ids(b.train) and ids(a.val) == ids(b.val), "deterministic")

print(f"\n{ok} passed, {fail} failed")
