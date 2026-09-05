"""BatchPool tests — torch-free. Fake 'batches' are just ids; a fake DataLoader
is any object whose iter() yields them (optionally slowly)."""
import time
import threading
import itertools
from collections import Counter
from data.pool import BatchPool

ok = fail = 0
def check(cond, msg):
    global ok, fail
    if cond: ok += 1
    else:
        fail += 1
        print(f"  FAIL: {msg}")


class ListLoader:
    """Yields a fixed list each epoch; fresh iterator per iter()."""
    def __init__(self, items, delay=0.0):
        self.items = items
        self.delay = delay
    def __iter__(self):
        def gen():
            for x in self.items:
                if self.delay:
                    time.sleep(self.delay)
                yield x
        return gen()
    def __len__(self):
        return len(self.items)


class CounterLoader:
    """Infinite unique ids; counts how many were actually pulled."""
    def __init__(self, delay=0.0):
        self.delay = delay
        self.n_pulled = 0
        self._lock = threading.Lock()
    def __iter__(self):
        def gen():
            while True:
                with self._lock:
                    self.n_pulled += 1
                    v = self.n_pulled
                if self.delay:
                    time.sleep(self.delay)
                yield v
        return gen()


class ErrLoader:
    def __init__(self, n_ok=2):
        self.n_ok = n_ok
    def __iter__(self):
        def gen():
            for i in range(self.n_ok):
                yield f"ok{i}"
            raise ValueError("boom in producer")
        return gen()
    def __len__(self):
        return self.n_ok


# ---- 1. passthrough: exact order, exact count, deterministic --------------
print("1. passthrough")
items = [f"b{i}" for i in range(50)]
p = BatchPool(ListLoader(items), steps_per_epoch=50, reuse=1, order="sequential")
check(p.is_passthrough(), "reuse=1+seq is passthrough")
out = list(p)
check(out == items, "passthrough yields producer order exactly")
check(len(out) == 50, f"count {len(out)}")
# deterministic
out2 = list(BatchPool(ListLoader(items), steps_per_epoch=50, reuse=1, order="sequential"))
check(out == out2, "passthrough reproducible")

# passthrough loops the producer when steps exceed its length
loop = list(BatchPool(ListLoader(["a", "b", "c"]), steps_per_epoch=7,
                      reuse=1, order="sequential"))
check(loop == ["a", "b", "c", "a", "b", "c", "a"], f"loops producer: {loop}")

# ---- 2. pooled: total count, and the hard reuse invariant -----------------
print("2. pooled counts + reuse invariant")
STEPS, REUSE, SIZE = 500, 5, 32
cl = CounterLoader()
p = BatchPool(cl, steps_per_epoch=STEPS, size=SIZE, reuse=REUSE, order="random",
              warmup=SIZE, seed=1)
served = list(p)
check(len(served) == STEPS, f"served {len(served)} == {STEPS}")
counts = Counter(served)
check(max(counts.values()) <= REUSE, f"NO id served more than reuse: max={max(counts.values())}")
# most ids hit exactly reuse; only those left in the pool at the end are short
exact = sum(1 for c in counts.values() if c == REUSE)
check(exact >= len(counts) - SIZE, f"all but <=SIZE served exactly reuse ({exact}/{len(counts)})")
# unique fresh batches ~= steps/reuse
uniq = len(counts)
lo, hi = STEPS // REUSE, STEPS // REUSE + SIZE + 2
check(lo - 2 <= uniq <= hi, f"unique fresh ~ steps/reuse: {uniq} in [{lo},{hi}]")
# producer wasn't over-pulled: pulled <= served-unique + buffered slack
check(cl.n_pulled <= uniq + SIZE + 2, f"bounded pulls: pulled={cl.n_pulled}, uniq={uniq}")
# every served id came from the producer (positive ints)
check(all(isinstance(x, int) and x > 0 for x in served), "served ids are producer values")

# ---- 3. reuse=1 random still serves each once -----------------------------
print("3. reuse=1 random")
cl3 = CounterLoader()
s3 = list(BatchPool(cl3, steps_per_epoch=100, size=16, reuse=1, order="random", seed=2))
c3 = Counter(s3)
check(len(s3) == 100 and max(c3.values()) == 1, "reuse=1: every served batch is fresh")
check(len(c3) == 100, f"100 distinct fresh batches: {len(c3)}")

# ---- 4. latency hiding: serving beats naive producer time -----------------
print("4. latency hiding")
DELAY, N = 0.01, 200
naive = DELAY * N                                   # ~2.0s if every step waited a decode
slow = CounterLoader(delay=DELAY)
t0 = time.time()
n = sum(1 for _ in BatchPool(slow, steps_per_epoch=N, size=20, reuse=20,
                             order="random", warmup=6, seed=3))
elapsed = time.time() - t0
check(n == N, f"served all {n}")
check(elapsed < 0.6 * naive, f"latency hidden: {elapsed:.3f}s < {0.6*naive:.3f}s (naive {naive:.2f}s)")
# with reuse 20, fresh pulls ~ N/20 = 10
check(slow.n_pulled <= N // 20 + 20 + 2, f"few fresh pulls under reuse: {slow.n_pulled}")

# ---- 5. producer exception propagates to the serve side -------------------
print("5. error propagation")
raised = False
try:
    for _ in BatchPool(ErrLoader(n_ok=2), steps_per_epoch=50, size=8, reuse=3,
                       order="random", warmup=2, loop_producer=False):
        pass
except ValueError as e:
    raised = "boom" in str(e)
check(raised, "producer ValueError surfaces on the serving thread")

# ---- 6. multiple epochs + set_epoch ---------------------------------------
print("6. epochs")
cl6 = CounterLoader()
p6 = BatchPool(cl6, steps_per_epoch=80, size=16, reuse=4, order="random", seed=10)
e0 = list(p6)
p6.set_epoch(1)
e1 = list(p6)
check(len(e0) == 80 and len(e1) == 80, f"both epochs full: {len(e0)},{len(e1)}")
check(max(Counter(e0).values()) <= 4 and max(Counter(e1).values()) <= 4, "reuse invariant holds each epoch")
# different epoch seed -> different serve order (ids are fresh across epochs anyway,
# but the per-epoch pool RNG differs, so behaviour isn't identical by construction)
check(e0 != e1, "epochs differ")

# ---- 7. early break cleans up the filler thread ---------------------------
print("7. early break / no thread leak")
base = threading.active_count()
p7 = BatchPool(CounterLoader(delay=0.002), steps_per_epoch=10000,
               size=32, reuse=3, order="random", seed=5)
it = iter(p7)
for _ in range(5):
    next(it)
it.close()                     # triggers generator finally -> stop + join
time.sleep(0.3)
check(threading.active_count() <= base + 1, f"filler thread stopped (active {threading.active_count()} vs base {base})")

# ---- 8. sequential+reuse serves FIFO-ish, invariant still holds ------------
print("8. sequential reuse")
cl8 = CounterLoader()
s8 = list(BatchPool(cl8, steps_per_epoch=60, size=8, reuse=3, order="sequential", seed=1))
check(len(s8) == 60 and max(Counter(s8).values()) <= 3, "sequential reuse respects invariant")

# ---- 9. steps default from len(dataloader) --------------------------------
print("9. steps default")
p9 = BatchPool(ListLoader(list(range(123))))     # no explicit steps
check(p9.steps_per_epoch == 123, f"defaults to len(loader): {p9.steps_per_epoch}")
check(len(p9) == 123, "len() matches")

print(f"\n{ok} passed, {fail} failed")