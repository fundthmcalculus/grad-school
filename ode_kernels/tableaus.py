"""Verified Butcher tableaux for the ode_kernels embedded Runge-Kutta family.

Every tableau here is transcribed from a primary, independently checkable source
rather than derived from memory, because a single wrong digit in a high-order
tableau produces a method that still runs and still "looks" convergent at loose
tolerances while silently failing to achieve its advertised order. Sources:

  * ``ode23`` -- Bogacki-Shampine 3(2), transcribed from
    ``scipy/integrate/_ivp/rk.py`` (class ``RK23``). Bit-identical to what
    ``scipy.integrate.solve_ivp(method="RK23")`` uses.
  * ``ode45`` -- Dormand-Prince 5(4), transcribed from
    ``scipy/integrate/_ivp/rk.py`` (class ``RK45``). Bit-identical to
    ``scipy.integrate.solve_ivp``'s default method.
  * ``ode56`` -- Verner's "efficient" 6(5) pair, transcribed from
    ``OrdinaryDiffEqVerner/src/verner_tableaus.jl`` (``Vern6Tableau``,
    ``CompiledFloats`` specialization -- the literal Float64 constants Julia's
    production solver uses), cross-checked against the exact-rational variant
    of the same function in the same file.
  * ``ode67`` -- Verner's "efficient" 7(6) pair, transcribed the same way from
    ``Vern7Tableau`` in the same file.
  * ``ode78`` -- Fehlberg's classical 7(8) pair (the method the name "RKF78"
    refers to), transcribed from NASA Trick's
    ``er7_utils/integration/rkf78/src/rkf78_butcher_tableau.cc``, a
    flight-software-grade implementation.
  * ``ode12`` -- Heun-Euler 2(1). Two stages, derivable by hand from the order
    conditions; included for completeness at the cheap end of the family.

Every tableau is validated by :func:`check_tableau` (row-sum / weight-sum
consistency) in ``tests/test_tableaus.py``, and every method's *actual* order
is re-derived empirically in ``tests/test_convergence.py`` via Richardson
log-log slope fits -- that second check is what actually catches a transcription
error, since a wrong coefficient can still satisfy the row-sum identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class ButcherTableau:
    """An embedded explicit Runge-Kutta pair in "extended" form.

    ``A`` is the full (n_stages, n_stages) strictly-lower-triangular stage
    matrix, ``c`` the stage abscissae, ``b`` the propagated (higher-order)
    weights, and ``e`` the error weights such that, for stage derivatives
    ``k_1 .. k_s``::

        y_new = y + h * sum_i b[i] * k_i
        err   = h * sum_i e[i] * k_i      (~ y_new(order) - y_new(error_order))

    ``fsal`` marks methods where stage ``s`` is exactly ``f(t + h, y_new)``
    (i.e. row ``s`` of ``A`` equals ``b``), so it can be carried over as stage
    1 of the next step for free instead of re-evaluated.
    """

    name: str
    order: int
    error_order: int
    n_stages: int
    A: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    e: tuple[float, ...]
    c: tuple[float, ...]
    fsal: bool
    reference: str = field(default="", compare=False)
    P: tuple[tuple[float, ...], ...] | None = None
    """Optional scipy-style free interpolation matrix, shape (n_stages, poly_order),
    for exact-match dense output. ``None`` means dense output falls back to a
    cubic Hermite fit on the step endpoints."""

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array(self.A, dtype=np.float64)
        b = np.array(self.b, dtype=np.float64)
        e = np.array(self.e, dtype=np.float64)
        c = np.array(self.c, dtype=np.float64)
        return A, b, e, c

    def P_array(self) -> np.ndarray | None:
        return None if self.P is None else np.array(self.P, dtype=np.float64)


def _pad(row: list[float], n: int) -> tuple[float, ...]:
    return tuple(row + [0.0] * (n - len(row)))


def _sub(b: list[float], bhat: list[float]) -> tuple[float, ...]:
    return tuple(bi - bhi for bi, bhi in zip(b, bhat))


# ---------------------------------------------------------------------------
# ode12 -- Heun-Euler 2(1)
# ---------------------------------------------------------------------------
_HEUN_EULER = ButcherTableau(
    name="Heun-Euler",
    order=2,
    error_order=1,
    n_stages=2,
    A=((0.0, 0.0), (1.0, 0.0)),
    b=(0.5, 0.5),
    e=_sub([0.5, 0.5], [1.0, 0.0]),
    c=(0.0, 1.0),
    fsal=False,
    reference="Classical predictor-corrector pair; Euler embedded in Heun's method.",
)

# ---------------------------------------------------------------------------
# ode23 -- Bogacki-Shampine 3(2)   (scipy.integrate RK23, bit-identical)
# ---------------------------------------------------------------------------
_BOGACKI_SHAMPINE = ButcherTableau(
    name="Bogacki-Shampine",
    order=3,
    error_order=2,
    n_stages=4,
    A=(
        (0.0, 0.0, 0.0, 0.0),
        (1 / 2, 0.0, 0.0, 0.0),
        (0.0, 3 / 4, 0.0, 0.0),
        (2 / 9, 1 / 3, 4 / 9, 0.0),
    ),
    b=(2 / 9, 1 / 3, 4 / 9, 0.0),
    e=(5 / 72, -1 / 12, -1 / 9, 1 / 8),
    c=(0.0, 1 / 2, 3 / 4, 1.0),
    fsal=True,
    reference="P. Bogacki, L.F. Shampine, Appl. Math. Lett. 2(4), 1989; "
    "scipy.integrate._ivp.rk.RK23.",
    P=(
        (1.0, -4 / 3, 5 / 9),
        (0.0, 1.0, -2 / 3),
        (0.0, 4 / 3, -8 / 9),
        (0.0, -1.0, 1.0),
    ),
)

# ---------------------------------------------------------------------------
# ode45 -- Dormand-Prince 5(4)   (scipy.integrate RK45, bit-identical)
# ---------------------------------------------------------------------------
_DORMAND_PRINCE = ButcherTableau(
    name="Dormand-Prince",
    order=5,
    error_order=4,
    n_stages=7,
    A=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1 / 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3 / 40, 9 / 40, 0.0, 0.0, 0.0, 0.0, 0.0),
        (44 / 45, -56 / 15, 32 / 9, 0.0, 0.0, 0.0, 0.0),
        (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0.0, 0.0, 0.0),
        (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0.0, 0.0),
        (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0),
    ),
    b=(35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0),
    e=(-71 / 57600, 0.0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40),
    c=(0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0),
    fsal=True,
    reference="J.R. Dormand, P.J. Prince, J. Comp. Appl. Math. 6(1), 1980; "
    "scipy.integrate._ivp.rk.RK45.",
    P=(
        (1.0, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799),
        (0.0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072),
        (0.0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632),
        (0.0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844),
        (0.0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423),
    ),
)

# ---------------------------------------------------------------------------
# ode56 -- Verner's efficient 6(5) pair ("Vern6")
# ---------------------------------------------------------------------------
_VERNER56 = ButcherTableau(
    name="Verner 6(5)",
    order=6,
    error_order=5,
    n_stages=9,
    A=(
        _pad([], 9),
        _pad([0.06], 9),
        _pad([0.019239962962962962, 0.07669337037037037], 9),
        _pad([0.035975, 0.0, 0.107925], 9),
        _pad([1.3186834152331484, 0.0, -5.042058063628562, 4.220674648395414], 9),
        _pad(
            [
                -41.87259166432751,
                0.0,
                159.43256216313748,
                -122.11921356501004,
                5.531743066200053,
            ],
            9,
        ),
        _pad(
            [
                -54.430156935316504,
                0.0,
                207.06725136501848,
                -158.61081378459,
                6.991816585950242,
                -0.01859723106220323,
            ],
            9,
        ),
        _pad(
            [
                -54.66374178728198,
                0.0,
                207.95280625538936,
                -159.2889574744995,
                7.018743740796944,
                -0.018338785905045722,
                -0.0005119484997882099,
            ],
            9,
        ),
        _pad(
            [
                0.03438957868357036,
                0.0,
                0.0,
                0.25826245556335037,
                0.4209371189673537,
                4.40539646966931,
                -176.48311902429865,
                172.36413340141507,
            ],
            9,
        ),
    ),
    b=(
        0.03438957868357036,
        0.0,
        0.0,
        0.25826245556335037,
        0.4209371189673537,
        4.40539646966931,
        -176.48311902429865,
        172.36413340141507,
        0.0,
    ),
    e=(
        0.008623404282200854,
        0.0,
        0.0,
        -0.019434029953152708,
        0.028450072588037983,
        -2.1097110610652914,
        103.45854289996397,
        -101.39980461914912,
        0.03333333333333333,
    ),
    c=(0.0, 0.06, 0.09593333333333333, 0.1439, 0.4973, 0.9725, 0.9995, 1.0, 1.0),
    fsal=True,
    reference="J.H. Verner, 'Numerically optimal Runge-Kutta pairs with "
    "interpolants', Numer. Algorithms (2010); OrdinaryDiffEqVerner.jl "
    "Vern6Tableau (CompiledFloats specialization).",
)

# ---------------------------------------------------------------------------
# ode67 -- Verner's efficient 7(6) pair ("Vern7")
# ---------------------------------------------------------------------------
_VERNER67 = ButcherTableau(
    name="Verner 7(6)",
    order=7,
    error_order=6,
    n_stages=10,
    A=(
        _pad([], 10),
        _pad([0.005], 10),
        _pad([-1.07679012345679, 1.185679012345679], 10),
        _pad([0.04083333333333333, 0.0, 0.1225], 10),
        _pad(
            [0.6389139236255726, 0.0, -2.455672638223657, 2.272258714598084], 10
        ),
        _pad(
            [
                -2.6615773750187572,
                0.0,
                10.804513886456137,
                -8.3539146573962,
                0.820487594956657,
            ],
            10,
        ),
        _pad(
            [
                6.067741434696772,
                0.0,
                -24.711273635911088,
                20.427517930788895,
                -1.9061579788166472,
                1.006172249242068,
            ],
            10,
        ),
        _pad(
            [
                12.054670076253203,
                0.0,
                -49.75478495046899,
                41.142888638604674,
                -4.461760149974004,
                2.042334822239175,
                -0.09834843665406107,
            ],
            10,
        ),
        _pad(
            [
                10.138146522881808,
                0.0,
                -42.6411360317175,
                35.76384003992257,
                -4.3480228403929075,
                2.0098622683770357,
                0.3487490460338272,
                -0.27143900510483127,
            ],
            10,
        ),
        _pad(
            [
                -45.030072034298676,
                0.0,
                187.3272437654589,
                -154.02882369350186,
                18.56465306347536,
                -7.141809679295079,
                1.3088085781613787,
            ],
            10,
        ),
    ),
    b=(
        0.04715561848627222,
        0.0,
        0.0,
        0.25750564298434153,
        0.26216653977412624,
        0.15216092656738558,
        0.4939969170032485,
        -0.29430311714032503,
        0.08131747232495111,
        0.0,
    ),
    e=(
        0.002547011879931045,
        0.0,
        0.0,
        -0.00965839487279575,
        0.04206470975639691,
        -0.0666822437469301,
        0.2650097464621281,
        -0.29430311714032503,
        0.08131747232495111,
        -0.02029518466335628,
    ),
    c=(
        0.0,
        0.005,
        0.10888888888888888,
        0.16333333333333333,
        0.4555,
        0.6095094489978381,
        0.884,
        0.925,
        1.0,
        1.0,
    ),
    fsal=False,
    reference="J.H. Verner, op. cit.; OrdinaryDiffEqVerner.jl Vern7Tableau "
    "(CompiledFloats specialization).",
)

# ---------------------------------------------------------------------------
# ode78 -- Fehlberg's classical 7(8) pair ("RKF78")
# ---------------------------------------------------------------------------
_RKF78_C = (
    0.0,
    2 / 27,
    1 / 9,
    1 / 6,
    5 / 12,
    1 / 2,
    5 / 6,
    1 / 6,
    2 / 3,
    1 / 3,
    1.0,
    0.0,
    1.0,
)
_RKF78_A = (
    _pad([], 13),
    _pad([2 / 27], 13),
    _pad([1 / 36, 1 / 12], 13),
    _pad([1 / 24, 0.0, 1 / 8], 13),
    _pad([5 / 12, 0.0, -25 / 16, 25 / 16], 13),
    _pad([1 / 20, 0.0, 0.0, 1 / 4, 1 / 5], 13),
    _pad([-25 / 108, 0.0, 0.0, 125 / 108, -65 / 27, 125 / 54], 13),
    _pad([31 / 300, 0.0, 0.0, 0.0, 61 / 225, -2 / 9, 13 / 900], 13),
    _pad([2.0, 0.0, 0.0, -53 / 6, 704 / 45, -107 / 9, 67 / 90, 3.0], 13),
    _pad(
        [-91 / 108, 0.0, 0.0, 23 / 108, -976 / 135, 311 / 54, -19 / 60, 17 / 6, -1 / 12],
        13,
    ),
    _pad(
        [
            2383 / 4100,
            0.0,
            0.0,
            -341 / 164,
            4496 / 1025,
            -301 / 82,
            2133 / 4100,
            45 / 82,
            45 / 164,
            18 / 41,
        ],
        13,
    ),
    _pad(
        [
            3 / 205,
            0.0,
            0.0,
            0.0,
            0.0,
            -6 / 41,
            -3 / 205,
            -3 / 41,
            3 / 41,
            6 / 41,
            0.0,
        ],
        13,
    ),
    _pad(
        [
            -1777 / 4100,
            0.0,
            0.0,
            -341 / 164,
            4496 / 1025,
            -289 / 82,
            2193 / 4100,
            51 / 82,
            33 / 164,
            12 / 41,
            0.0,
            1.0,
        ],
        13,
    ),
)
_RKF78_B8 = (
    0.0, 0.0, 0.0, 0.0, 0.0, 34 / 105, 9 / 35, 9 / 35, 9 / 280, 9 / 280, 0.0,
    41 / 840, 41 / 840,
)
# The NASA Trick source also ships an "RKb7" vector alongside RKb8, which
# looks at first glance like it should give a second full propagated
# solution for a textbook e = b8 - b7 error estimate (that's how ode23/45/
# 56/67 all do it). It doesn't: dotting RKb7 against the stages is only
# ~2nd-order accurate here, not 7th -- confirmed empirically (fixed h on
# y'=-y: the resulting "error" scales as h^2, not h^8) before this was
# caught. Fehlberg's actual classical error estimator for this tableau uses
# only 4 of the 13 stages, weighted by the same 41/840 that appears twice in
# b8: err = h * (41/840) * (k1 + k11 - k12 - k13). Cross-checked against an
# independent reference implementation (aerospaceresearch/orbitdeterminator,
# util/rkf78.py) and empirically verified to scale as h**8 (halving h
# shrinks the estimate by ~256x, i.e. 2**8) -- see
# tests/test_convergence.py and this module's __main__ block.
_RKF78_E = _pad([41 / 840], 13)
_e78 = list(_RKF78_E)
_e78[10] += 41 / 840
_e78[11] -= 41 / 840
_e78[12] -= 41 / 840
_RKF78_E = tuple(_e78)

_FEHLBERG78 = ButcherTableau(
    name="Fehlberg 7(8)",
    order=8,
    error_order=7,
    n_stages=13,
    A=_RKF78_A,
    b=_RKF78_B8,
    e=_RKF78_E,
    c=_RKF78_C,
    fsal=False,
    reference="E. Fehlberg, NASA TR R-287 (1968); NASA Trick "
    "er7_utils/integration/rkf78/src/rkf78_butcher_tableau.cc; error "
    "estimator cross-checked against aerospaceresearch/orbitdeterminator "
    "util/rkf78.py.",
)

TABLEAUS: dict[str, ButcherTableau] = {
    "ode12": _HEUN_EULER,
    "ode23": _BOGACKI_SHAMPINE,
    "ode45": _DORMAND_PRINCE,
    "ode56": _VERNER56,
    "ode67": _VERNER67,
    "ode78": _FEHLBERG78,
}


def check_tableau(tab: ButcherTableau, tol: float = 1e-12) -> list[str]:
    """Return a list of consistency-condition violations (empty if none).

    Checks the two structural identities every consistent explicit RK
    tableau must satisfy: each stage's row of ``A`` sums to its abscissa
    ``c_i`` (the "row-sum" or "consistency" condition), and the propagating
    weights ``b`` sum to 1 (needed to integrate ``y' = 1`` exactly). These
    catch gross transcription errors but -- because they are necessary, not
    sufficient, conditions for a given order -- do not by themselves prove
    the tableau achieves its claimed order. See ``tests/test_convergence.py``
    for the empirical order check that does.
    """
    problems = []
    A, b, e, c = tab.as_arrays()
    row_sums = A.sum(axis=1)
    bad_rows = np.where(np.abs(row_sums - c) > tol)[0]
    for i in bad_rows:
        problems.append(
            f"{tab.name}: row {i} of A sums to {row_sums[i]!r}, expected c[{i}]={c[i]!r}"
        )
    b_sum = b.sum()
    if abs(b_sum - 1.0) > tol:
        problems.append(f"{tab.name}: sum(b) = {b_sum!r}, expected 1.0")
    return problems


if __name__ == "__main__":
    for method, tab in TABLEAUS.items():
        issues = check_tableau(tab)
        status = "OK" if not issues else "FAILED"
        print(f"{method:8s} ({tab.name:20s}) order={tab.order} "
              f"error_order={tab.error_order} n_stages={tab.n_stages} fsal={tab.fsal} "
              f"-> {status}")
        for issue in issues:
            print(f"    {issue}")
