# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
"""Cython hot loop shared by ode12/ode23/ode45/ode56/ode67/ode78.

Two "FPU tricks" live here, applied uniformly to every method through one
generic embedded-RK stage engine rather than six hand-duplicated ones:

1. **Fused multiply-add** (``libc.math.fma``) for every stage combination and
   for the final weighted sums. FMA computes ``a*b+c`` with a single rounding
   instead of two, which is both more accurate and -- when the extension is
   built with ``-mfma``/``-march=native`` -- lowers directly to the hardware
   FMA3 instruction instead of separate mulsd/addsd, so the compiler-level
   "trick" and the numerical-accuracy trick are the same line of code.

2. **A nogil fast path for compiled right-hand sides.** The generic engine
   always needs the GIL to call an arbitrary Python ``fun(t, y)``, so that is
   the default path. But if the caller passes a right-hand side compiled to a
   flat C function pointer (e.g. a ``numba.cfunc`` with signature
   ``void(float64, CPointer(float64), CPointer(float64), intc)``, exposed via
   its ``.address`` attribute), the *entire* per-step stage loop -- every
   evaluation, every combination -- runs without ever touching the GIL. See
   ``rhs_ptr_t`` and ``step_generic_fast`` below.

Both paths funnel their arithmetic through the same ``_combine`` /
``_weighted_sum`` nogil kernels: loop order is stage-outer /
state-component-inner (so each stage's contribution streams over a
contiguous row of ``K`` and the per-stage scale factor ``h * coeff`` is
hoisted out of the inner loop), and a zero coefficient short-circuits its
row entirely -- these tableaus are sparse (e.g. Verner's schemes have
several structural zeros per row) so skipping them is a real, not just
cosmetic, saving.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport fma

cnp.import_array()

ctypedef void (*rhs_ptr_t)(double t, double* y, double* dy, int n) noexcept nogil


cdef inline void _combine(double[::1] out, double[::1] y, double h,
                           double[:, ::1] K, double[::1] coeffs,
                           int n_terms, int n) noexcept nogil:
    """out[:] = y[:] + h * sum_{i<n_terms} coeffs[i] * K[i, :], via fma."""
    cdef int i, j
    cdef double hc
    for j in range(n):
        out[j] = y[j]
    for i in range(n_terms):
        hc = h * coeffs[i]
        if hc == 0.0:
            continue
        for j in range(n):
            out[j] = fma(hc, K[i, j], out[j])


cdef inline void _weighted_sum(double[::1] out, double h, double[:, ::1] K,
                                double[::1] coeffs, int n_terms, int n) noexcept nogil:
    """out[:] = h * sum_{i<n_terms} coeffs[i] * K[i, :], via fma."""
    cdef int i, j
    cdef double hc
    for j in range(n):
        out[j] = 0.0
    for i in range(n_terms):
        hc = h * coeffs[i]
        if hc == 0.0:
            continue
        for j in range(n):
            out[j] = fma(hc, K[i, j], out[j])


cdef inline object y_arr_view(double[::1] buf):
    """Expose a memoryview as a plain ndarray for a Python-callable RHS."""
    return np.asarray(buf)


def step_generic_py(object fun, double t, double[::1] y, double h,
                     double[:, ::1] A, double[::1] b, double[::1] e,
                     double[::1] c, int n_stages, tuple args,
                     double[::1] k1_carry):
    """One embedded-RK step attempt, calling ``fun`` as a Python callable.

    ``k1_carry``, if not ``None``, is used directly as stage 1 instead of
    calling ``fun(t, y)`` again -- valid both across FSAL step boundaries
    (stage ``n_stages`` of an FSAL tableau equals ``f(t+h, y_new)``, which is
    exactly stage 1 of the next step) and across step-size retries at a fixed
    ``(t, y)`` (stage 1 never depends on ``h``, so a rejected-and-retried
    step never needs to recompute it either).

    Returns ``(y_new, err, K)`` as freshly allocated ndarrays.
    """
    cdef int n = y.shape[0]
    cdef cnp.ndarray K_arr = np.empty((n_stages, n), dtype=np.float64)
    cdef double[:, ::1] K = K_arr
    cdef cnp.ndarray y_stage_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] y_stage = y_stage_arr
    cdef int i
    cdef double ti

    if k1_carry is not None:
        K_arr[0, :] = k1_carry
    else:
        if args:
            f0 = fun(t, y_arr_view(y), *args)
        else:
            f0 = fun(t, y_arr_view(y))
        K_arr[0, :] = np.asarray(f0, dtype=np.float64)

    for i in range(1, n_stages):
        _combine(y_stage, y, h, K, A[i, :], i, n)
        ti = t + c[i] * h
        if args:
            fi = fun(ti, y_arr_view(y_stage), *args)
        else:
            fi = fun(ti, y_arr_view(y_stage))
        K_arr[i, :] = np.asarray(fi, dtype=np.float64)

    cdef cnp.ndarray y_new_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray err_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] y_new = y_new_arr
    cdef double[::1] err = err_arr
    _combine(y_new, y, h, K, b, n_stages, n)
    _weighted_sum(err, h, K, e, n_stages, n)
    return y_new_arr, err_arr, K_arr


def step_generic_fast(size_t rhs_address, double t, double[::1] y, double h,
                       double[:, ::1] A, double[::1] b, double[::1] e,
                       double[::1] c, int n_stages, double[::1] k1_carry):
    """One embedded-RK step attempt, calling a compiled RHS through a raw
    function pointer with signature ``void(double t, double* y, double* dy,
    int n)``. The whole stage loop runs without the GIL.
    """
    cdef int n = y.shape[0]
    cdef rhs_ptr_t f = <rhs_ptr_t><void*>rhs_address
    cdef cnp.ndarray K_arr = np.empty((n_stages, n), dtype=np.float64)
    cdef double[:, ::1] K = K_arr
    cdef cnp.ndarray y_stage_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] y_stage = y_stage_arr
    cdef cnp.ndarray y_new_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray err_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] y_new = y_new_arr
    cdef double[::1] err = err_arr
    cdef bint has_carry = k1_carry is not None
    cdef int i
    cdef double ti

    if has_carry:
        K_arr[0, :] = k1_carry

    with nogil:
        if not has_carry:
            f(t, &y[0], &K[0, 0], n)
        for i in range(1, n_stages):
            _combine(y_stage, y, h, K, A[i, :], i, n)
            ti = t + c[i] * h
            f(ti, &y_stage[0], &K[i, 0], n)
        _combine(y_new, y, h, K, b, n_stages, n)
        _weighted_sum(err, h, K, e, n_stages, n)

    return y_new_arr, err_arr, K_arr


def eval_fast(size_t rhs_address, double t, double[::1] y):
    """Evaluate a compiled RHS once (used for initial-step selection, and
    for the trailing f(t+h, y_new) that non-FSAL methods need for dense
    output). Not on the hot path, so a plain (GIL-holding) call is fine."""
    cdef int n = y.shape[0]
    cdef rhs_ptr_t f = <rhs_ptr_t><void*>rhs_address
    cdef cnp.ndarray dy_arr = np.empty(n, dtype=np.float64)
    cdef double[::1] dy = dy_arr
    with nogil:
        f(t, &y[0], &dy[0], n)
    return dy_arr
