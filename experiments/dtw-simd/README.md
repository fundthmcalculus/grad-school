# dtw-simd: OpenMP + AVX-512 all-pairs DTW

Cython kernel matching `aeon.distances.dtw_pairwise_distance` semantics
(full-window, squared local cost, no sqrt) exactly, built because aeon's numba
build was measured SINGLE-THREADED on this machine and the G2 harness's cost
is entirely the DTW matrix build.

Design: OpenMP `prange` over anchor rows; within a row, pairs are processed in
lockstep batches of 8 with the DP laid out `(L, 8)` so the innermost lane loop
is unit-stride and auto-vectorizes (8 float64 = one AVX-512 vector).

Measured (8-core AVX-512 box, under one-core contention): **10-12x aeon**,
verified equal to 2e-13. In production (`REPRO_G2_DTW_IMPL=simd`): ECG5000's
full 5000x5000 build 630s -> 59s; StarLightCurves' full 9236x1024 build drops
from ~30h (infeasible) to ~2-4h.

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
