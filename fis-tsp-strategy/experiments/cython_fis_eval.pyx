# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""A C translation of ``fis.fis_eval1``, to answer whether Cython can beat numba here.

This is a transcription, not an improvement: same table lerp, same product t-norm, same
early break at a firing strength below 1e-12, same weighted-average defuzzification. Making
it *different* would answer a different question — the point is to hold the algorithm fixed
and vary only the compiler, so that any difference is attributable to code generation.

The loop lives inside the C function for the same reason it lives inside the jitted one on the
numba side: crossing the Python boundary costs on the order of a microsecond, which is several
times the kernel itself, so a per-call comparison would measure the boundary on both sides and
report a tie regardless of what the code generators did.

Typed memoryviews with bounds and wraparound checking off are the fair setting, because numba
does not bounds-check either — FINDINGS §10.4 records what that cost once. Anything less would
be handicapping Cython; anything more would be comparing a checked implementation against an
unchecked one.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int8_t

cdef int MF_RES = 64
cdef int N_TERMS = 3


cdef inline void _memberships(
    double[::1] x, double[:, :, ::1] tab, double[:, ::1] mu, int n_in
) noexcept nogil:
    cdef int i, t, j
    cdef double xi, f, a, v0
    for i in range(n_in):
        xi = x[i]
        if xi < 0.0:
            xi = 0.0
        elif xi > 1.0:
            xi = 1.0
        f = xi * MF_RES
        j = <int>f
        if j >= MF_RES:
            j = MF_RES - 1
        a = f - j
        for t in range(N_TERMS):
            v0 = tab[i, t, j]
            mu[i, t] = v0 + a * (tab[i, t, j + 1] - v0)


cdef inline double _eval1(
    double[::1] x, double[:, ::1] mu, double[:, :, ::1] tab,
    int8_t[:, ::1] ant, double[:, ::1] cons,
) noexcept nogil:
    cdef int n_rules = ant.shape[0]
    cdef int n_in = ant.shape[1]
    cdef int r, i
    cdef int8_t a
    cdef double w, num = 0.0, den = 0.0
    _memberships(x, tab, mu, n_in)
    for r in range(n_rules):
        w = 1.0
        for i in range(n_in):
            a = ant[r, i]
            if a >= 0:
                w *= mu[i, a]
                if w < 1e-12:
                    break
        if w < 1e-12:
            continue
        den += w
        num += w * cons[r, 0]
    if den <= 1e-12:
        return 0.5
    return num / den


def loop_eval1(
    long reps,
    double[::1] x,
    double[:, ::1] mu,
    double[:, :, ::1] tab,
    int8_t[:, ::1] ant,
    double[:, ::1] cons,
):
    """``_eval1`` ``reps`` times, mirroring the jitted driver exactly.

    ``x[0]`` is perturbed and the result accumulated for the same reason as on the numba side:
    without a consumed result and a varying input, an optimiser is entitled to hoist the call
    out of the loop or delete it, and the kernel then benchmarks at zero.
    """
    cdef double acc = 0.0
    cdef long i
    with nogil:
        for i in range(reps):
            x[0] = (i & 63) / 64.0
            acc += _eval1(x, mu, tab, ant, cons)
    return acc
