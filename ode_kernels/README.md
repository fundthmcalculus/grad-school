# ode_kernels

Cython-accelerated embedded Runge-Kutta ODE integrators -- `ode12`, `ode23`,
`ode45`, `ode56`, `ode67`, `ode78` -- plus `odeexp`, an exponential
Rosenbrock-Euler integrator, all built around one adaptive step-size
controller styled on `scipy.integrate.solve_ivp`.

```python
from ode_kernels import ode45

result = ode45(lambda t, y: [-y[0]], (0.0, 5.0), [1.0])
print(result.y[0, -1], result.nfev)
```

## The family

| method  | pair                       | order | error order | stages | source of coefficients |
|---------|----------------------------|:-----:|:-----------:|:------:|-------------------------|
| `ode12` | Heun-Euler                 | 2     | 1           | 2      | classical, hand-derivable |
| `ode23` | Bogacki-Shampine           | 3     | 2           | 4      | `scipy.integrate._ivp.rk.RK23` (bit-identical tableau) |
| `ode45` | Dormand-Prince             | 5     | 4           | 7      | `scipy.integrate._ivp.rk.RK45` (bit-identical tableau, scipy's own default method) |
| `ode56` | Verner 6(5) ("Vern6")      | 6     | 5           | 9      | `OrdinaryDiffEqVerner.jl`, `Vern6Tableau` |
| `ode67` | Verner 7(6) ("Vern7")      | 7     | 6           | 10     | `OrdinaryDiffEqVerner.jl`, `Vern7Tableau` |
| `ode78` | Fehlberg 7(8) ("RKF78")    | 8     | 7           | 13     | NASA Trick, `rkf78_butcher_tableau.cc` |

Every tableau is transcribed from one of those primary sources (not from
memory) -- see `tableaus.py` for the full citations and the exact literals.
That matters more the higher the order gets: a single wrong digit in a
13-stage tableau still runs, still looks plausible at loose tolerances, and
silently fails to achieve its advertised accuracy. Two independent checks
guard against that:

* **Structural consistency** (`tests/test_tableaus.py`): every stage row of
  `A` sums to its abscissa `c_i`, and `sum(b) == 1`. Necessary, not
  sufficient -- a transcription error can still satisfy these.
* **Empirical order of convergence** (`tests/test_convergence.py`): each
  method is run at a sequence of fixed step sizes through the raw stage
  kernel (bypassing the adaptive controller) on `y' = -y`, and the observed
  log-log slope of global error vs. `h` is checked against the claimed
  order. This is the check that actually catches a wrong coefficient.

`ode23` and `ode45` are additionally checked step-for-step against
`scipy.integrate.solve_ivp(method="RK23"/"RK45")` on a nonlinear problem
(`tests/test_vs_scipy.py`) -- since they share a tableau and step-size
controller, they should (and do) land within tight tolerance of scipy's own
answer, not just an independently-accurate one.

## Design notes / the "advanced FPU" tricks

**One generic engine, not six unrolled ones.** All six methods share a
single Cython stage-combination kernel (`_rk_kernels.pyx`), parameterized by
each tableau's `(A, b, e, c, n_stages)` at call time rather than
hand-unrolled per method. This was a deliberate simplicity-over-microopt
trade: `n_stages` is small (2-13) and known at the start of each call, so
GCC still auto-vectorizes/unrolls the inner loops under `-O3 -funroll-loops`
without needing six near-duplicate hand-written steppers, each an
independent opportunity for a copy-paste bug in the exact place correctness
matters most.

**Fused multiply-add** (`libc.math.fma`) for every stage combination and
weighted sum. `fma(a, b, c)` computes `a*b + c` with a single rounding
instead of two separate roundings for a multiply then an add -- both more
accurate, and, built with `-march=native` on an FMA3-capable target, it
lowers directly to the hardware fused-multiply-add instruction instead of
separate `mulsd`/`addsd`. The accuracy win and the instruction-level win are
the same line of code.

**Structural-zero skipping.** These tableaus are sparse -- Verner's schemes
in particular have several zero entries per row (`a92 = a93 = 0`, etc.) --
so the stage-combination loop hoists `h * coeff` out of the inner loop and
skips a row entirely when that product is zero, both saving the work and
avoiding `0 * anything` edge cases.

**A nogil fast path for compiled right-hand sides.** The default path calls
an arbitrary Python `fun(t, y)`, which needs the GIL every stage regardless
of what else is fast. But if `fun` exposes a `.address` attribute pointing
at a C function with signature `void(double t, double* y, double* dy, int
n)` -- what a `numba.cfunc`-decorated function gives you -- the *entire*
per-step stage loop runs without ever touching the GIL:

```python
from numba import cfunc, types
import numpy as np
from ode_kernels import ode45

@cfunc(types.void(types.double, types.CPointer(types.double),
                   types.CPointer(types.double), types.intc))
def rhs(t, y, dy, n):
    dy[0] = -y[0]

result = ode45(rhs, (0.0, 5.0), [1.0])   # same call, no Python per stage
```

`args` isn't supported together with the fast path (there's no way to thread
arbitrary Python objects through a raw C signature); use a closure baked
into the compiled function instead.

**Deliberately not `-ffast-math`.** The build (`setup.py`) uses
`-O3 -march=native -funroll-loops -fno-math-errno`, probing at build time
and dropping `-march=native` if the compiler rejects it. `-ffast-math` is
left out on purpose: it would relax the IEEE semantics the adaptive
controller actually depends on -- `error_norm == 0` exact-equality checks,
NaN propagation for blow-up detection -- to chase a speedup in a loop that
is bandwidth- and call-overhead-bound, not transcendental-function-bound.
Trading step-control correctness for that would be a bad trade.

**FSAL, done as a general property of the tableau, not per-method
special-casing.** `ode23`, `ode45`, and `ode56` are First-Same-As-Last: the
final stage's abscissa is 1 and its row of `A` equals `b`, so that stage
*is* `f(t+h, y_new)` already. The driver (`_driver.py`) detects this from
the tableau's `fsal` flag and carries that stage into the next step's
(and, within a rejected step, every retried attempt's) stage 1, verified in
`tests/test_tableaus.py::test_fsal_flag_matches_last_row_of_a` and covered
by the `nfev` accounting checked in `test_vs_scipy.py`.

## `odeexp`

An exponential Rosenbrock-Euler integrator: instead of a polynomial
approximation to `y' = f(t, y)`, it linearizes about the current state via
the Jacobian `J = df/dy` and integrates the linear part *exactly* via the
matrix `phi1` function, treating the nonlinear remainder to first order:

```
y_new = y + h * phi1(h J) f(y),      phi1(z) = (e^z - 1) / z
```

exact for linear systems, second order for general nonlinear ones
(Hochbruck & Ostermann, "Exponential integrators", *Acta Numerica* 2010).
Useful when `J`'s eigenvalues are large and negative (fast linear decay that
would pin an explicit RK method to tiny steps) but the nonlinear part is
mild.

Two implementation points worth knowing about:

* **`phi1(hJ)v` without ever forming `phi1`.** For the augmented matrix
  `M = [[hJ, hv], [0, 0]]`, `exp(M)`'s top-right block is exactly
  `h * phi1(hJ) * v` (Al-Mohy & Higham, 2011). One dense matrix exponential
  of a size-`(n+1)` matrix replaces a whole phi-function machinery. `exp(M)`
  itself is computed by scaling-and-squaring with a plain 18-term Taylor
  series rather than a Pade approximant -- more matrix products, no linear
  solve needed, a reasonable trade at the small dense Jacobians this
  targets.
* **Non-autonomous correction.** A right-hand side with explicit
  `t`-dependence needs `df/dt`, not just `df/dy`, or the method quietly
  degrades to first order and the step-size controller compensates by
  shrinking `h` far more than necessary -- this is exactly what surfaced
  the bug during development (see `_expdriver.py`'s `_exprb2_step`
  docstring). Fixed by autonomizing: treat time as an extra state
  (`dt/ds = 1`) and linearize the resulting `(n+1)`-dimensional augmented
  system, extending the same augmented-matrix trick by one more row/column.

Adaptivity uses classical Richardson step doubling (one step of `h` vs. two
of `h/2`) rather than an embedded pair, since exponential Rosenbrock-Euler
has no natural embedded companion -- 3x the linear algebra per attempt, an
honest cost for adaptivity on a method whose whole point is large steps
through stiff linear behavior.

Only handles dense Jacobians of modest size (it forms `J` and `exp(M)`
explicitly every step); a large stiff system would want a Krylov-subspace
action method instead, out of scope here.

## Layout

```
tableaus.py       Butcher tableau data + consistency checker (source of truth)
_common.py        OdeResult, dense-output interpolants, initial-step heuristic
_driver.py        adaptive step-size controller for the RK family
_methods.py       public ode12..ode78 entry points
_rk_kernels.pyx   Cython: generic embedded-RK stage engine (Python-callable +
                  nogil fast-path RHS)
_expint.pyx       Cython: dense matrix exponential + the augmented-matrix
                  phi1-action trick
_expdriver.py     odeexp: Jacobian (analytic or finite-difference), the
                  non-autonomous correction, step-doubling adaptivity
setup.py          build script (Cython + aggressive-but-safe compiler flags)
tests/            tableau consistency, empirical order, vs. scipy, dense
                  output, fast-RHS path, odeexp
benchmark.py      speed/accuracy vs. scipy.integrate.solve_ivp on 3 problems
```

## Build

```bash
cd ode_kernels
pip install cython numpy scipy pytest numba   # numba only needed for the
                                               # fast-RHS test/example
python setup.py build_ext --inplace
```

## Test

```bash
cd ode_kernels && python -m pytest tests/ -v
```

## Benchmark

```bash
python -m ode_kernels.benchmark
```

Compares every method against the closest scipy equivalent (`ode23`/`ode45`
against `RK23`/`RK45` directly; all six against a `DOP853`-at-1e-13
reference trajectory for accuracy) on a 20-dimensional linear decay, the
Van der Pol oscillator, and the 7-body Pleiades problem -- writes a table to
stdout and `figures/benchmark.png`.

## What's not here

* **Event detection** (`solve_ivp`'s `events=`) -- not implemented.
* **Implicit / stiff methods.** All six RK methods are explicit; `odeexp`
  handles *mild* stiffness via its linear part but still needs an explicit
  treatment of the nonlinear remainder, so it is not a substitute for a
  proper implicit stiff solver (BDF, Radau) on genuinely stiff nonlinear
  problems.
* **Vectorized RHS** (`solve_ivp`'s `vectorized=True`, batched columns of
  `y`) -- each stage evaluates `fun` on one state vector at a time.
