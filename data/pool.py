"""
pool.py
=======

A bounded reuse-pool wrapping a DataLoader. It targets I/O *latency*: the frame
cache (cache.py) reduces how many EXRs are decoded, but a miss is still a
synchronous decode of a huge file on the critical path. The pool holds N
pre-composed padded batches in CPU RAM and serves a random one per step while a
background thread keeps refilling from the (slow) DataLoader — so serving never
blocks on a decode once the pool is warm.

The reuse factor is the throughput knob. Each fresh batch is served ~`reuse`
times before eviction, so the producer only has to supply ~1 fresh batch per
`reuse` serves — i.e. it tolerates storage that's `reuse`x too slow to feed one
fresh batch per step. Turning `reuse` up trades data freshness for throughput.

Per epoch the pool yields `steps_per_epoch` batches (optimizer steps, unchanged
by reuse); only ~steps_per_epoch/reuse of them are freshly decoded, the rest are
reuses. Across epochs the sampler reshuffles (new rotations/windows), so
coverage catches up. Train-only: run val/test through the DataLoader directly
(or a pool with reuse=1), or you silently reweight them.

Reproducibility: `reuse=1, order='sequential'` is a pure passthrough — no thread,
no randomness — and reproduces the DataLoader/sampler stream exactly. Any other
setting is a stochastic stream (random serve + background-fill timing), which is
fine for training and matches the original's streaming reader.

Device-agnostic: it stores whatever the DataLoader yields (CPU tensors); the
training loop moves the served batch to GPU as it already does. No torch here.
"""

from __future__ import annotations

import threading
from typing import Optional

from .sampler import _seed_for
import random


class _ProducerExhausted(Exception):
    pass


class BatchPool:
    def __init__(
        self,
        dataloader,
        *,
        steps_per_epoch: Optional[int] = None,
        size: int = 64,
        reuse: int = 4,
        order: str = "random",         # "random" | "sequential"
        warmup: Optional[int] = None,  # entries to prefill before first serve
        seed: int = 1234,
        epoch: int = 0,
        loop_producer: bool = True,    # re-open the loader if it exhausts mid-epoch
        wait_timeout: float = 0.25,
    ):
        if order not in ("random", "sequential"):
            raise ValueError("order must be 'random' or 'sequential'")
        if size < 1:
            raise ValueError("size must be >= 1")
        if reuse < 1:
            raise ValueError("reuse must be >= 1")
        self.dataloader = dataloader
        self._explicit_steps = steps_per_epoch
        self.size = size
        self.reuse = reuse
        self.order = order
        self.warmup = min(size, warmup if warmup is not None else size)
        self.seed = seed
        self.epoch = epoch
        self.loop_producer = loop_producer
        self.wait_timeout = wait_timeout

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def steps_per_epoch(self) -> int:
        if self._explicit_steps is not None:
            return self._explicit_steps
        return len(self.dataloader)     # == len(batch_sampler) for a batch_sampler loader

    def __len__(self) -> int:
        return self.steps_per_epoch

    def is_passthrough(self) -> bool:
        return self.reuse == 1 and self.order == "sequential"

    # -- one epoch --------------------------------------------------------
    def __iter__(self):
        if self.is_passthrough():
            yield from self._passthrough()
        else:
            yield from self._pooled()

    # deterministic, thread-free
    def _passthrough(self):
        steps = self.steps_per_epoch
        it = iter(self.dataloader)
        for _ in range(steps):
            try:
                yield next(it)
            except StopIteration:
                if not self.loop_producer:
                    return
                it = iter(self.dataloader)
                try:
                    yield next(it)
                except StopIteration:
                    return

    # buffered + reuse
    def _pooled(self):
        steps = self.steps_per_epoch
        rng = random.Random(_seed_for(self.seed, self.epoch, "pool"))

        pool = []                      # list of [batch, serve_count]
        lock = threading.Lock()
        not_full = threading.Condition(lock)
        not_empty = threading.Condition(lock)
        stop = threading.Event()
        state = {"exc": None, "done": False}   # producer exhausted / errored

        def filler():
            producer = iter(self.dataloader)
            try:
                while not stop.is_set():
                    with lock:
                        while len(pool) >= self.size and not stop.is_set():
                            not_full.wait(self.wait_timeout)
                        if stop.is_set():
                            return
                    try:
                        batch = next(producer)
                    except StopIteration:
                        if not self.loop_producer:
                            raise _ProducerExhausted
                        producer = iter(self.dataloader)
                        try:
                            batch = next(producer)
                        except StopIteration:
                            raise _ProducerExhausted
                    with lock:
                        pool.append([batch, 0])
                        not_empty.notify()
            except _ProducerExhausted:
                pass
            except BaseException as e:           # surface to the serving thread
                with lock:
                    state["exc"] = e
                    not_empty.notify_all()
            finally:
                with lock:
                    state["done"] = True
                    not_empty.notify_all()

        t = threading.Thread(target=filler, name="batchpool-filler", daemon=True)
        t.start()
        try:
            # warmup: wait until enough entries are buffered (or producer finished)
            with lock:
                while len(pool) < self.warmup and not state["done"] and state["exc"] is None:
                    not_empty.wait(self.wait_timeout)

            served = 0
            while served < steps:
                with lock:
                    while len(pool) == 0 and not state["done"] and state["exc"] is None:
                        not_empty.wait(self.wait_timeout)
                    if state["exc"] is not None:
                        raise state["exc"]
                    if len(pool) == 0 and state["done"]:
                        break            # producer dry and empty
                    idx = rng.randrange(len(pool)) if self.order == "random" else 0
                    entry = pool[idx]
                    entry[1] += 1
                    batch = entry[0]
                    if entry[1] >= self.reuse:
                        pool.pop(idx)
                        not_full.notify()
                served += 1
                yield batch
        finally:
            stop.set()
            with lock:
                not_full.notify_all()
                not_empty.notify_all()
            t.join(timeout=5.0)


def build_pool(dataloader, *, steps_per_epoch=None, size=64, reuse=4,
               order="random", warmup=None, seed=1234, epoch=0):
    """Convenience constructor mirroring build_dataloader's style."""
    return BatchPool(dataloader, steps_per_epoch=steps_per_epoch, size=size,
                     reuse=reuse, order=order, warmup=warmup, seed=seed, epoch=epoch)