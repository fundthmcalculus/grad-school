"""Decompose the speedup: how much is the kernel, how much is just cores?

The original bench compared this kernel on 8 threads against aeon's
DEFAULT n_jobs=1. aeon supports n_jobs, so that comparison conflates two
different things. This one varies one factor at a time.
"""

import time
import numpy as np
from aeon.distances import dtw_pairwise_distance
import dtw_simd

rng = np.random.default_rng(0)
CASES = [
    ("600x96", np.ascontiguousarray(rng.normal(size=(600, 96)))),
    ("150x1024", np.ascontiguousarray(rng.normal(size=(150, 1024)))),
]


def t(fn, *a, **k):
    fn(*a, **k)  # warm the JIT / caches
    t0 = time.perf_counter()
    fn(*a, **k)
    return time.perf_counter() - t0


print(
    f"{'case':>10} {'aeon 1thr':>10} {'aeon 8thr':>10} {'simd 1thr':>10} {'simd 8thr':>10}"
    f" | {'aeon par':>9} {'simd par':>9} {'KERNEL 1v1':>11} {'fair 8v8':>9}"
)
for name, X in CASES:
    a1 = t(dtw_pairwise_distance, X, n_jobs=1)
    a8 = t(dtw_pairwise_distance, X, n_jobs=8)
    s1 = t(dtw_simd.dtw_pairwise, X, 1)
    s8 = t(dtw_simd.dtw_pairwise, X, 8)
    # correctness still holds at every setting
    assert np.allclose(dtw_pairwise_distance(X, n_jobs=8), dtw_simd.dtw_pairwise(X, 8))
    print(
        f"{name:>10} {a1:9.2f}s {a8:9.2f}s {s1:9.2f}s {s8:9.2f}s"
        f" | {a1/a8:8.2f}x {s1/s8:8.2f}x {a1/s1:10.2f}x {a8/s8:8.2f}x"
    )
