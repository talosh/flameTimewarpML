"""
cache.py
========

A per-worker LRU frame cache, keyed by path, holding decoded+tonemapped+resized
frames. This targets I/O *volume*: overlapping sliding-window triplets share two
of three frames with their neighbours, so caching decoded frames cuts redundant
EXR reads. (Latency is the reuse-pool's job, one layer up.)

One instance lives per DataLoader worker (created in worker_init_fn); processes
don't share memory cheaply, so there's no cross-worker sharing. Sizing: memory ~=
max_items x (C x H x W x 4) bytes, or cap directly with max_bytes.

torch-free and generic — it stores any object and measures it via `nbytes`
(numpy) / element_size x nelement (torch) / an explicit size, so its logic is
fully testable without torch.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Optional


def default_sizeof(value) -> int:
    # torch tensor
    es = getattr(value, "element_size", None)
    ne = getattr(value, "nelement", None)
    if callable(es) and callable(ne):
        return es() * ne()
    # numpy array
    nb = getattr(value, "nbytes", None)
    if isinstance(nb, int):
        return nb
    return 1  # opaque object: count by item instead of bytes


class FrameCache:
    def __init__(self, max_items: int = 256, max_bytes: Optional[int] = None,
                 sizeof: Callable = default_sizeof):
        if max_items is not None and max_items < 1 and max_bytes is None:
            raise ValueError("need a positive max_items or a max_bytes budget")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._sizeof = sizeof
        self._d: "OrderedDict[str, tuple]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._d)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def get_or_load(self, key: str, loader: Callable,
                    nbytes: Optional[int] = None):
        """Return cached value for `key`, else call `loader()`, store, and return.
        `loader` runs *outside* the lock (it may be slow I/O)."""
        with self._lock:
            hit = self._d.get(key)
            if hit is not None:
                self.hits += 1
                self._d.move_to_end(key)
                return hit[0]

        value = loader()                       # slow path, unlocked
        nb = nbytes if nbytes is not None else self._sizeof(value)

        with self._lock:
            existing = self._d.get(key)
            if existing is not None:           # another thread beat us (rare)
                self._d.move_to_end(key)
                return existing[0]
            self.misses += 1
            self._d[key] = (value, nb)
            self._bytes += nb
            self._evict_locked()
            return value

    def _evict_locked(self) -> None:
        while self._d:
            over_items = self.max_items is not None and len(self._d) > self.max_items
            over_bytes = self.max_bytes is not None and self._bytes > self.max_bytes
            if not (over_items or over_bytes):
                break
            _, (v, nb) = self._d.popitem(last=False)   # evict oldest
            self._bytes -= nb

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"items": len(self._d), "bytes": self._bytes,
                "hits": self.hits, "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0}
