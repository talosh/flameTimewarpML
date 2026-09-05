"""FrameCache tests — torch-free (uses explicit byte sizes / dummy objects)."""
from data.cache import FrameCache, default_sizeof

ok = fail = 0
def check(cond, msg):
    global ok, fail
    if cond: ok += 1
    else:
        fail += 1
        print(f"  FAIL: {msg}")

# ---- 1. hit / miss + loader-called-once ----------------------------------
print("1. hit/miss")
calls = {}
def loader_for(key):
    def _l():
        calls[key] = calls.get(key, 0) + 1
        return f"value-{key}"
    return _l

c = FrameCache(max_items=10)
v1 = c.get_or_load("a", loader_for("a"), nbytes=100)
v2 = c.get_or_load("a", loader_for("a"), nbytes=100)   # hit -> loader NOT called
check(v1 == "value-a" and v2 == "value-a", "returns value")
check(calls["a"] == 1, f"loader called once, got {calls['a']}")
check(c.hits == 1 and c.misses == 1, f"hits={c.hits} misses={c.misses}")

# ---- 2. LRU eviction by item count ---------------------------------------
print("2. LRU by count")
c = FrameCache(max_items=3)
for k in ["a", "b", "c"]:
    c.get_or_load(k, lambda k=k: k, nbytes=1)
check(len(c) == 3, f"len {len(c)}")
c.get_or_load("d", lambda: "d", nbytes=1)              # evicts "a" (oldest)
check(len(c) == 3, "still 3 after insert")
# "a" should be gone -> re-loading it is a miss (loader runs)
ran = {"a": False}
def load_a():
    ran["a"] = True
    return "a"
c.get_or_load("a", load_a, nbytes=1)
check(ran["a"], "a was evicted (re-load ran)")

# ---- 3. LRU ordering: access refreshes recency ---------------------------
print("3. recency refresh")
c = FrameCache(max_items=3)
for k in ["a", "b", "c"]:
    c.get_or_load(k, lambda k=k: k, nbytes=1)
c.get_or_load("a", lambda: "SHOULD_NOT_RUN", nbytes=1)   # touch a -> now MRU
c.get_or_load("d", lambda: "d", nbytes=1)                # evicts b (now oldest), not a
present = {}
for k in ["a", "b", "c", "d"]:
    ran = {"x": False}
    def l(k=k, ran=ran):
        ran["x"] = True
        return k
    c.get_or_load(k, l, nbytes=1)
    present[k] = not ran["x"]   # present == loader did NOT run
# after touching a then inserting d (with c present): expect a,c,d present, b evicted
check(present["a"], "a retained (was touched)")
check(not present["b"], "b evicted (became oldest)")

# ---- 4. eviction by byte budget ------------------------------------------
print("4. byte budget")
c = FrameCache(max_items=None, max_bytes=250)
c.get_or_load("a", lambda: "a", nbytes=100)
c.get_or_load("b", lambda: "b", nbytes=100)
check(c.nbytes == 200 and len(c) == 2, f"bytes={c.nbytes} len={len(c)}")
c.get_or_load("c", lambda: "c", nbytes=100)   # 300 > 250 -> evict oldest (a) -> 200
check(c.nbytes == 200 and len(c) == 2, f"after budget evict: bytes={c.nbytes} len={len(c)}")
# a single item larger than budget is stored then immediately trimmed to <= itself
c2 = FrameCache(max_items=None, max_bytes=50)
c2.get_or_load("big", lambda: "big", nbytes=100)
check(len(c2) <= 1, f"oversized item handling: len={len(c2)}")

# ---- 5. default_sizeof on numpy ------------------------------------------
print("5. sizeof")
import numpy as np
arr = np.zeros((3, 100, 100), dtype=np.float32)
check(default_sizeof(arr) == arr.nbytes == 3 * 100 * 100 * 4, f"numpy nbytes {default_sizeof(arr)}")
class Fake:  # torch-like
    def element_size(self): return 4
    def nelement(self): return 3 * 64 * 64
check(default_sizeof(Fake()) == 4 * 3 * 64 * 64, f"torch-like sizeof {default_sizeof(Fake())}")

# ---- 6. hit-rate stats ---------------------------------------------------
print("6. stats")
c = FrameCache(max_items=100)
for _ in range(3):
    c.get_or_load("k", lambda: "k", nbytes=1)   # 1 miss + 2 hits
st = c.stats()
check(abs(st["hit_rate"] - 2/3) < 1e-9, f"hit_rate {st['hit_rate']}")

print(f"\n{ok} passed, {fail} failed")
