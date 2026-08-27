# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Batched, OpenMP + SIMD all-pairs DTW (squared local cost, no window).

Semantics match aeon.distances.dtw_pairwise_distance exactly (verified in
bench.py): full-window DTW over equal-length univariate float64 series with
SQUARED difference local cost and no square root at the end.

Why this is fast where numba's is not:
  * OpenMP `prange` over anchor rows (aeon's build was measured single-core).
  * Within a row, pairs are processed in lockstep batches of B=8: the DP
    recurrence is sequential in (t1, t2) but INDEPENDENT across pairs, so the
    innermost loop runs over the batch lane with unit stride and
    auto-vectorizes -- 8 float64 lanes is exactly one AVX-512 vector (one SSE2
    vector still covers 2, so the layout wins on any x86-64).
  * The batch's series are pre-packed into (L, B) layout so the cost load is
    also unit-stride.

Memory per thread: two (L+1)*B float64 DP rows -- 131 KB at L=1024, L2-resident.
"""

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free

cnp.import_array()

DEF BATCH = 8
cdef double INF = float("inf")


cdef inline double _min3(double a, double b, double c) nogil:
    cdef double m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m


cdef void _dtw_batch(
    const double* x,          # (L,) anchor series
    const double* ypack,      # (L, BATCH) packed batch, lane-contiguous
    double* prev,             # (L+1)*BATCH scratch
    double* cur,              # (L+1)*BATCH scratch
    double* out,              # (BATCH,) results
    Py_ssize_t L,
) noexcept nogil:
    cdef Py_ssize_t t1, t2, b
    cdef double xv, c, m
    # dp[0][0] = 0, dp[0][t2>0] = INF
    for b in range(BATCH):
        prev[b] = 0.0
    for t2 in range(1, L + 1):
        for b in range(BATCH):
            prev[t2 * BATCH + b] = INF
    for t1 in range(1, L + 1):
        xv = x[t1 - 1]
        # dp[t1][0] = INF
        for b in range(BATCH):
            cur[b] = INF
        for t2 in range(1, L + 1):
            for b in range(BATCH):  # unit-stride lane loop: auto-vectorizes
                c = xv - ypack[(t2 - 1) * BATCH + b]
                c = c * c
                m = _min3(
                    prev[(t2 - 1) * BATCH + b],   # diagonal
                    prev[t2 * BATCH + b],         # up
                    cur[(t2 - 1) * BATCH + b],    # left
                )
                cur[t2 * BATCH + b] = c + m
        # swap rows
        for t2 in range(0, (L + 1) * BATCH):
            prev[t2] = cur[t2]
    for b in range(BATCH):
        out[b] = prev[L * BATCH + b]


def dtw_pairwise(double[:, ::1] X, int n_threads=0):
    """All-pairs DTW matrix for (n, L) float64 series. n_threads<=0: OpenMP default."""
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t L = X.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] D = np.zeros((n, n), dtype=np.float64)
    cdef double[:, ::1] Dv = D
    cdef Py_ssize_t i, j, j0, b, t, nb
    cdef double* ypack
    cdef double* prev
    cdef double* cur
    # `out` must be heap scratch like the others: Cython privatizes scalars
    # and pointers across prange threads, NOT fixed C arrays -- a stack array
    # here is SHARED and races (caught by bench.py: 110/40k corrupted cells).
    cdef double* out
    cdef int nt = n_threads if n_threads > 0 else 0

    with nogil:
        for i in prange(n - 1, schedule="dynamic", num_threads=nt if nt > 0 else 8):
            ypack = <double*> malloc(L * BATCH * sizeof(double))
            prev = <double*> malloc((L + 1) * BATCH * sizeof(double))
            cur = <double*> malloc((L + 1) * BATCH * sizeof(double))
            out = <double*> malloc(BATCH * sizeof(double))
            j0 = i + 1
            while j0 < n:
                nb = n - j0
                if nb > BATCH:
                    nb = BATCH
                # pack nb series into (L, BATCH), padding lanes with series j0
                for t in range(L):
                    for b in range(BATCH):
                        if b < nb:
                            ypack[t * BATCH + b] = X[j0 + b, t]
                        else:
                            ypack[t * BATCH + b] = X[j0, t]
                _dtw_batch(&X[i, 0], ypack, prev, cur, out, L)
                for b in range(nb):
                    Dv[i, j0 + b] = out[b]
                    Dv[j0 + b, i] = out[b]
                j0 = j0 + BATCH
            free(ypack)
            free(prev)
            free(cur)
            free(out)
    return D
