# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
"""Cython core for ``odeexp``, an exponential (Rosenbrock-Euler) integrator.

The idea behind an exponential integrator: instead of approximating
``y' = f(t, y)`` with polynomials (what every Runge-Kutta method in this
package does), linearize about the current state via the Jacobian
``J = df/dy`` and integrate the resulting ``y' = J y + (f(y) - J y)`` *exactly*
along its linear part. The nonlinear remainder is treated to first order,
giving the "exponential Rosenbrock-Euler" step

    y_new = y + h * phi1(h J) f(y),      phi1(z) = (e^z - 1) / z

which is exact for linear systems and second order for general nonlinear
ones (Hochbruck & Ostermann, "Exponential integrators", Acta Numerica 2010).
It is the right tool when ``J``'s eigenvalues are large and negative (stiff
decay that would force a classical explicit RK method down to tiny steps)
but the nonlinear part is mild.

**The trick used to compute phi1(hJ)v without ever forming phi1 itself:**
for the augmented matrix

    M = [[h*J,  h*v],
         [ 0,    0 ]]      (size (n+1) x (n+1))

it is a standard identity (Al-Mohy & Higham, "Computing the action of the
matrix exponential", 2011) that

    exp(M) = [[expm(h*J),  h*phi1(h*J)*v],
              [    0,            1      ]]

so the last column of ``exp(M)`` (top ``n`` entries) *is* the exponential
Rosenbrock-Euler increment. One dense matrix exponential replaces a whole
phi-function machinery.

``exp(M)`` itself is computed by scaling-and-squaring with a plain Taylor
series (scale ``M`` down by a power of two until its norm is comfortably
inside the Taylor series' fast-convergence radius, sum ~18 terms via
fused multiply-add, square back up) rather than a Pade approximant, trading
a few extra matrix products for not needing a linear solve -- a reasonable
trade at the small-to-moderate ``n`` this integrator targets (dense ``n x n``
Jacobians; large stiff systems would want a Krylov-subspace action method
instead, which is out of scope here).
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fma, fabs, ceil, log2

cnp.import_array()

cdef int _TAYLOR_TERMS = 18
cdef double _SCALING_THETA = 0.5


cdef inline void _matmul(double[:, ::1] out, double[:, ::1] X, double[:, ::1] Y,
                          int m) noexcept nogil:
    """out = X @ Y for square m x m matrices, via fma."""
    cdef int i, k, j
    cdef double xik
    for i in range(m):
        for j in range(m):
            out[i, j] = 0.0
    for i in range(m):
        for k in range(m):
            xik = X[i, k]
            if xik == 0.0:
                continue
            for j in range(m):
                out[i, j] = fma(xik, Y[k, j], out[i, j])


cdef inline double _inf_norm(double[:, ::1] X, int m) noexcept nogil:
    cdef int i, j
    cdef double row, best
    best = 0.0
    for i in range(m):
        row = 0.0
        for j in range(m):
            row += fabs(X[i, j])
        if row > best:
            best = row
    return best


def matrix_exponential(double[:, ::1] M):
    """Dense matrix exponential via scaling-and-squaring + Taylor series."""
    cdef int m = M.shape[0]
    cdef double nrm = _inf_norm(M, m)
    cdef int s = 0
    if nrm > _SCALING_THETA:
        s = <int> ceil(log2(nrm / _SCALING_THETA))
        if s < 0:
            s = 0
    cdef double scale = 1.0 / (2 ** s)

    cdef cnp.ndarray Ms_arr = np.multiply(np.asarray(M), scale)
    cdef double[:, ::1] Ms = Ms_arr

    cdef cnp.ndarray R_arr = np.eye(m, dtype=np.float64)
    cdef cnp.ndarray term_arr = np.eye(m, dtype=np.float64)
    cdef cnp.ndarray tmp_arr = np.empty((m, m), dtype=np.float64)
    cdef double[:, ::1] R = R_arr
    cdef double[:, ::1] term = term_arr
    cdef double[:, ::1] tmp = tmp_arr

    cdef int k, i, j
    with nogil:
        for k in range(1, _TAYLOR_TERMS + 1):
            _matmul(tmp, term, Ms, m)
            for i in range(m):
                for j in range(m):
                    tmp[i, j] /= k
            for i in range(m):
                for j in range(m):
                    term[i, j] = tmp[i, j]
                    R[i, j] += tmp[i, j]
        for _ in range(s):
            _matmul(tmp, R, R, m)
            for i in range(m):
                for j in range(m):
                    R[i, j] = tmp[i, j]

    return R_arr


def exprb2_increment(double[:, ::1] J, double[::1] fy, double h):
    """Return ``h * phi1(h J) fy`` via the augmented-matrix exponential trick.

    ``J`` is the (n, n) Jacobian, ``fy`` the (n,) right-hand-side value at the
    current state, ``h`` the step size.
    """
    cdef int n = J.shape[0]
    cdef int m = n + 1
    cdef cnp.ndarray M_arr = np.zeros((m, m), dtype=np.float64)
    cdef double[:, ::1] M = M_arr
    cdef int i, j
    for i in range(n):
        for j in range(n):
            M[i, j] = h * J[i, j]
        M[i, n] = h * fy[i]
    E = matrix_exponential(M_arr)
    return np.ascontiguousarray(E[:n, n])
