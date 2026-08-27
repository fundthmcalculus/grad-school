# dtw-simd: OpenMP + AVX-512 all-pairs DTW

Cython kernel matching `aeon.distances.dtw_pairwise_distance` semantics
(full-window, squared local cost, no sqrt) exactly, built because aeon's numba
build was measured SINGLE-THREADED on this machine and the G2 harness's cost
is entirely the DTW matrix build.

Design: OpenMP `prange` over anchor rows; within a row, pairs are processed in
lockstep batches of 8 with the DP laid out `(L, 8)` so the innermost lane loop
is unit-stride and auto-vectorizes (8 float64 = one AVX-512 vector).

Measured, **one variable at a time** (`fair_bench.py`; 4 physical cores):

| case | aeon 1 thr | aeon 8 thr | this 1 thr | this 8 thr | kernel (1v1) | fair (8v8) |
|---|---|---|---|---|---|---|
| 600x96 | 5.09s | 1.69s | 1.07s | 0.35s | **4.8x** | **4.8x** |
| 150x1024 | 34.35s | 11.39s | 10.36s | 3.01s | **3.3x** | **3.8x** |

**The honest number is ~3.3-4.8x, at equal core budget.** An earlier version of
this file claimed "10-12x". That compared this kernel on 8 threads against
`dtw_pairwise_distance`'s DEFAULT `n_jobs=1`, conflating two factors: aeon
parallelises perfectly well when asked, and BOTH kernels gain ~3x from cores
here, so the parallel speedup is not attributable to this code. Only the
per-core column is. The superseded claim is recorded rather than quietly
deleted -- it is exactly the AGENTS.md trap "attribute changes to one variable
at a time", walked into.

In production (`REPRO_G2_DTW_IMPL=simd`), against aeon *as the harness had been
calling it* (n_jobs=1): ECG5000's full 5000x5000 build 630s -> 59s;
StarLightCurves' 9236x1024 build ~30h -> 4.6h. Against a properly parallelised
aeon those same builds would have been roughly 210s and ~10h, so this kernel's
own contribution to StarLightCurves is ~10h -> 4.6h, not ~30h -> 4.6h.

Build:
    uv run --project ../../tribble-cluster --with cython --with setuptools \
        python setup.py build_ext --inplace
Verify + benchmark:
    uv run --project ../../tribble-cluster --with aeon python bench.py

-march=native (local experiment). A distributable build must use
-march=x86-64-v3 (the -march=native SIGILL incident is tribble-fis#124).

History note: bench.py's equality check caught a real race during development
(a fixed-size C array is NOT privatized by Cython prange -- scalars and
pointers are), and the harness's runtime equality gate then caught a latent
zero-edge bug in `ivat_mf.minimax_transform_fast` (sparse MST drops
exact-zero distances, i.e. duplicate points). Verification gates earn their
keep.
