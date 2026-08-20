"""Convert a TRIBBLE-constructed fuzzy inference system into a ReLU network.

The construction rests on one identity (Bede, Kreinovich & Toth, NAFIPS 2023 --
see ``papers/nn-fis-equivalence/``): a triangular membership function *is* a
short sum of ReLUs of the input, exactly, with no approximation anywhere.

    T(x; a, b, c) = s_a * relu(x - a) - (s_a + s_c) * relu(x - b) + s_c * relu(x - c)
    s_a = 1 / (b - a),  s_c = 1 / (c - b)

so a fuzzy term is a hidden-layer motif and its apex/foot knots are the ReLU
bias terms. Everything in this module follows from that:

* :func:`membership_to_relu` turns any of the package's membership shapes into
  that expansion (Gaussians are first fitted to triangles by the package's own
  :mod:`tribblefis.triangle_fit`, which is the only lossy step and is reported
  as such).
* :func:`fis_to_relu_net_1d` uses it to convert a one-dimensional Ruspini-
  partitioned zeroth-order TSK system into a one-hidden-layer ReLU network
  *analytically* -- no data, no fitting, agreement at machine precision. That
  is the theorem, made executable, and ``test_fis2nn.py`` pins it.
* :func:`analytic_seed_from_fis` is the practical n-dimensional version, and is
  what the experiment actually uses. It backs the equivalence out into *every*
  weight rather than only the biases: the FIS's own one-dimensional profiles are
  sampled at its knots and decomposed by second differences, so consequents and
  gating reach the network too. In more than one dimension the FIS output is not
  piecewise linear -- the product t-norm and the firing-strength normalization
  both leave the PWL class -- so what the seed carries is the FIS's additive
  part, exactly, and the experiment measures the residual.
* :func:`warm_start_from_fis` is the weaker variant kept for comparison: FIS
  knots for the first layer, and least squares for everything else.

Written against numpy only, deliberately: the point is that the converted
network is an ordinary MLP that any framework can consume, and a 60-line Adam
loop keeps the comparison between initializations free of framework defaults
that would otherwise differ between arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:  # pandas is used only in the partial-dependence path
    import pandas as pd

from tribblefis.gauss_data import (
    GaussianMembership,
    GaussianMixtureModel,
    TrapezoidMembership,
    TriangularMembership,
)
from tribblefis.triangle_fit import (
    GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
    fit_triangle_to_gaussian,
)

# Knots closer together than this (in the scaled feature's units) are merged
# into one hidden unit. Duplicate knots are common -- two classes' Gaussians
# routinely land on the same mode -- and an exactly duplicated ReLU column makes
# the read-out's normal equations singular in a way ridge would silently paper
# over rather than a modeller noticing.
KNOT_MERGE_TOL = 1e-9


class DegenerateMembership(ValueError):
    """A membership function with no width, so no exact ReLU expansion exists.

    Separate from a plain ``ValueError`` because the two callers want opposite
    things. A caller handing in a hand-built term wants the exception: it means
    the term is malformed. A caller walking a *fitted* FIS wants to skip the
    term and carry on: a zero-width term is what a feature with (near-)zero
    variance produces, and there is nothing for an axis-aligned seed to learn
    from it either way.

    This is not hypothetical. ``fit_triangle_to_gaussian`` collapses to zero
    width when a Gaussian's sigma is 0, which is what a constant feature gives;
    N-CMAPSS DS02 has one (``Xs_T30_max``). The first version of this guard
    assumed the case unreachable and took the benchmark down on contact.
    """


@dataclass(frozen=True)
class ReLUExpansion:
    """``mu(x) = bias + sum_i coeffs[i] * relu(x - knots[i])``, exactly."""

    bias: float
    knots: np.ndarray
    coeffs: np.ndarray

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.knots.size == 0:
            return np.full_like(x, self.bias, dtype=float)
        act = np.maximum(x[:, None] - self.knots[None, :], 0.0)
        return self.bias + act @ self.coeffs


def triangle_to_relu(mf: TriangularMembership) -> ReLUExpansion:
    """Exact ReLU expansion of a triangular term, shoulders included.

    A left shoulder (``a = -inf``) is 1 up to the apex and then falls, so it
    expands to a constant minus two ReLUs; a right shoulder (``c = +inf``) rises
    and then flattens, which is two ReLUs and no constant. Both matter: a
    Ruspini partition is exactly one left shoulder, some triangles, and one
    right shoulder, so a converter that only handled interior triangles could
    not convert a partition at all.

    Raises
    ------
    DegenerateMembership
        If a side has zero width -- ``a == b`` with a finite ``a``, or
        ``b == c`` with a finite ``c``. Such a term steps from 0 to 1 at the
        apex, and a **finite sum of ReLUs is continuous**, so no expansion of
        this form can represent it.

        This used to return silently wrong weights. The rising branch below
        carried the comment "b == a (a vertical rise) contributes no ReLU: the
        term jumps to 1 at the apex, which the falling side's expansion below
        reproduces on its own." It does not. With ``T(a=1, b=1, c=2)`` the old
        code returned ``-fall * relu(x - b) + fall * relu(x - c)`` -- the
        *negation* of the correct ramp, still falling past ``c``, wrong by a
        full unit of membership. ``b == c`` mirrored it, returning a right
        shoulder pinned at 1 instead of dropping to 0.

        **Degenerate terms are reachable from fitted models**, though not the
        negating branch above. The first version of this guard reasoned that
        ``fit_triangle_to_gaussian`` yields ``a < b < c`` for any
        ``sigma > 0``, so the case was unreachable -- and then raised on a real
        N-CMAPSS DS02 fit within the hour. What that data produces is
        ``sigma == 0`` exactly, from a feature with no variance
        (``Xs_T30_max``), which fits the *fully* collapsed ``a == b == c``.

        That form was harmless in value and bad in kind: with no rise, no fall
        and a zero apex coefficient, the old code emitted no knots and an
        all-zero expansion, so a collapsed feature entered the seed as silence
        rather than as a wrong number. Fidelity on DS02 is unchanged by this
        fix. What changes is that a collapsed feature is now *reported*
        (:func:`fis_knots` warns with a count) instead of vanishing.

        The negating ``a == b < c`` branch remains a genuine wrong answer for
        any caller that constructs terms directly; it simply is not what this
        package's Gaussian fit produces.
    ValueError
        If the feet are inverted (``a > b`` or ``b > c``). The old
        ``elif b > a`` guard silently dropped the rising ReLU in that case.
    """
    a, b, c = float(mf.a), float(mf.b), float(mf.c)
    knots: list[float] = []
    coeffs: list[float] = []
    bias = 0.0

    left_shoulder = np.isneginf(a)
    right_shoulder = np.isposinf(c)

    if not left_shoulder and a == b:
        raise DegenerateMembership(
            f"zero-width rising side: a == b == {b!r}. The term steps from 0 to 1 "
            "at the apex, and a finite sum of ReLUs is continuous, so no exact "
            "expansion exists. Use a left shoulder (a = -inf) if the term really "
            "is 1 up to the apex."
        )
    if not right_shoulder and b == c:
        raise DegenerateMembership(
            f"zero-width falling side: b == c == {b!r}. The term steps from 1 to 0 "
            "at the apex, and a finite sum of ReLUs is continuous, so no exact "
            "expansion exists. Use a right shoulder (c = +inf) if the term really "
            "stays 1 past the apex."
        )
    if not left_shoulder and a > b:
        raise ValueError(f"inverted triangle: need a <= b, got a={a!r}, b={b!r}")
    if not right_shoulder and b > c:
        raise ValueError(f"inverted triangle: need b <= c, got b={b!r}, c={c!r}")

    # Rising side.
    rise = 0.0
    if left_shoulder:
        bias = 1.0  # membership is already 1 to the left of the apex
    else:
        rise = 1.0 / (b - a)
        knots.append(a)
        coeffs.append(rise)

    # Falling side.
    fall = 0.0
    if right_shoulder:
        pass  # membership stays 1 to the right of the apex
    else:
        fall = 1.0 / (c - b)
        knots.append(c)
        coeffs.append(fall)

    # The apex knot carries whatever slope change makes the two sides meet.
    apex_coeff = -(rise + fall)
    if left_shoulder:
        apex_coeff = -fall
    if right_shoulder:
        apex_coeff = -rise
    if apex_coeff != 0.0:
        knots.append(b)
        coeffs.append(apex_coeff)

    order = np.argsort(np.asarray(knots, dtype=float)) if knots else np.array([], int)
    return ReLUExpansion(
        bias=bias,
        knots=np.asarray(knots, dtype=float)[order],
        coeffs=np.asarray(coeffs, dtype=float)[order],
    )


def trapezoid_to_relu(mf: TrapezoidMembership) -> ReLUExpansion:
    """Exact ReLU expansion of a trapezoidal term (four knots instead of three).

    Raises ``ValueError`` on a vertical edge (``a == b`` or ``c == d``) for the
    same reason :func:`triangle_to_relu` does: the term steps from 0 to 1 with
    no ramp, and a finite sum of ReLUs cannot step. The previous ``if b > a``
    guard skipped the rising pair in that case and returned an expansion that
    was 0 through the plateau and negative past it.

    Unlike the triangle, **all four parameters must be finite**. Triangles get
    shoulder forms because a Ruspini partition needs them; trapezoids in this
    package never do -- ``trapz_math_fast`` builds them from finite data
    quantiles -- and ``gauss_data.TrapezoidMembership.evaluate`` cannot
    represent an infinite foot anyway: with ``a = -inf`` its rising branch
    computes ``(x - a) / (b - a)`` as ``inf / inf`` and returns NaN across the
    whole left side. Converting a shape whose own ground truth is NaN would be
    inventing semantics, so this rejects it instead.
    """
    a, b, c, d = float(mf.a), float(mf.b), float(mf.c), float(mf.d)
    knots: list[float] = []
    coeffs: list[float] = []

    if not all(np.isfinite(v) for v in (a, b, c, d)):
        raise ValueError(
            f"trapezoid feet must all be finite, got a={a!r}, b={b!r}, c={c!r}, "
            f"d={d!r}. Shoulder forms exist for triangles (a Ruspini partition "
            "needs them) but not for trapezoids, whose own `evaluate` returns "
            "NaN on an infinite foot."
        )
    if a == b:
        raise DegenerateMembership(
            f"zero-width rising edge: a == b == {b!r}. A vertical edge is "
            "discontinuous and has no exact ReLU expansion."
        )
    if c == d:
        raise DegenerateMembership(
            f"zero-width falling edge: c == d == {c!r}. A vertical edge is "
            "discontinuous and has no exact ReLU expansion."
        )
    if a > b or c > d:
        raise ValueError(
            f"inverted trapezoid: need a <= b <= c <= d, got a={a!r}, b={b!r}, "
            f"c={c!r}, d={d!r}"
        )

    rise = 1.0 / (b - a)
    knots += [a, b]
    coeffs += [rise, -rise]
    fall = 1.0 / (d - c)
    knots += [c, d]
    coeffs += [-fall, fall]

    order = np.argsort(np.asarray(knots, dtype=float)) if knots else np.array([], int)
    return ReLUExpansion(
        bias=0.0,
        knots=np.asarray(knots, dtype=float)[order],
        coeffs=np.asarray(coeffs, dtype=float)[order],
    )


def membership_to_relu(
    mf, half_width_sigma: float = GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH
) -> ReLUExpansion:
    """ReLU expansion of any membership shape the package produces.

    Triangles and trapezoids convert exactly. A Gaussian has infinite support
    and is not piecewise linear, so it is first replaced by the package's own
    MAE-optimal triangle fit (`tribblefis.triangle_fit`) -- the single lossy
    step in the whole pipeline, and the reason the experiment reports the
    triangularized FIS's accuracy separately from the Gaussian FIS's.
    """
    if isinstance(mf, TriangularMembership):
        return triangle_to_relu(mf)
    if isinstance(mf, TrapezoidMembership):
        return trapezoid_to_relu(mf)
    if isinstance(mf, GaussianMembership):
        return triangle_to_relu(fit_triangle_to_gaussian(mf, half_width_sigma))
    raise TypeError(f"no ReLU expansion for {type(mf).__name__}")


def merge_knots(values: Iterable[float], tol: float = KNOT_MERGE_TOL) -> np.ndarray:
    """Sort, drop non-finite, and merge knots within ``tol`` into their mean."""
    vals = np.asarray(sorted(float(v) for v in values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] <= tol:
            out[-1] = 0.5 * (out[-1] + v)
        else:
            out.append(v)
    return np.asarray(out, dtype=float)


def fis_knots(
    model: GaussianMixtureModel,
    features: Sequence[str],
    half_width_sigma: float = GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
    tol: float = KNOT_MERGE_TOL,
) -> dict[str, np.ndarray]:
    """Every ReLU knot the FIS implies, per feature.

    This is the object the whole "hot start" claim is about: the FIS's answer to
    *where the interesting breakpoints of each input are*, expressed in the only
    currency a ReLU network has.

    Zero-width terms are skipped, and the count is reported through
    :func:`warnings.warn` rather than swallowed. A fitted FIS really does
    contain them -- a feature with (near-)zero variance gives a Gaussian whose
    sigma underflows, and its triangle fit has no width -- and such a term
    contributes no breakpoint an axis-aligned seed could use. Skipping is the
    right answer; skipping *silently* is not, because a FIS where most terms
    are degenerate is a FIS whose conversion means nothing, and the caller
    should hear about it.
    """
    import warnings

    knots: dict[str, np.ndarray] = {}
    n_degenerate = 0
    n_total = 0
    for name in features:
        feature_model = model.feature_models.get(name)
        if feature_model is None:
            knots[name] = np.asarray([], dtype=float)
            continue
        raw: list[float] = []
        for label_model in feature_model.label_models.values():
            for mf in label_model.memberships:
                n_total += 1
                try:
                    expansion = membership_to_relu(mf, half_width_sigma)
                except DegenerateMembership:
                    n_degenerate += 1
                    continue
                raw.extend(expansion.knots.tolist())
        knots[name] = merge_knots(raw, tol)

    if n_degenerate:
        warnings.warn(
            f"fis_knots: skipped {n_degenerate} of {n_total} membership "
            f"functions with zero width (a feature with no variance fits a "
            f"Gaussian whose sigma underflows). Their features contribute no "
            f"knots to the seed.",
            RuntimeWarning,
            stacklevel=2,
        )
    return knots


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


@dataclass
class ReLUNet:
    """``y = relu(X @ W1 + b1) @ w2 + X @ v + c``.

    One hidden layer plus a linear skip. The skip is not decoration: a ReLU
    layer whose knots all sit inside the data range cannot express a nonzero
    slope to the left of the first knot, and the exact 1-D conversion needs one.
    Every arm of the experiment carries the same skip, so it cannot flatter the
    warm-started arm specifically.
    """

    W1: np.ndarray  # (n_features, n_hidden)
    b1: np.ndarray  # (n_hidden,)
    w2: np.ndarray  # (n_hidden,)
    v: np.ndarray  # (n_features,)
    c: float

    @property
    def n_hidden(self) -> int:
        return int(self.W1.shape[1])

    def hidden(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(X, dtype=float) @ self.W1 + self.b1, 0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.hidden(X) @ self.w2 + X @ self.v + self.c

    def copy(self) -> "ReLUNet":
        return ReLUNet(
            self.W1.copy(), self.b1.copy(), self.w2.copy(), self.v.copy(), float(self.c)
        )

    def n_parameters(self) -> int:
        return self.W1.size + self.b1.size + self.w2.size + self.v.size + 1


def _design(net: ReLUNet, X: np.ndarray) -> np.ndarray:
    """``[hidden | X | 1]`` -- the read-out is linear in exactly these columns."""
    X = np.asarray(X, dtype=float)
    return np.hstack([net.hidden(X), X, np.ones((X.shape[0], 1))])


def solve_readout(
    net: ReLUNet, X: np.ndarray, y: np.ndarray, l2: float = 1e-6, anchor: bool = True
) -> ReLUNet:
    """Set ``w2, v, c`` to the ridge least-squares optimum for the current layer 1.

    Closed form, no gradient steps: for fixed hidden units the output is linear
    in the read-out, which is the same argument `regression.solve_tsk_consequents`
    makes for TSK consequents at fixed firing strengths. Its cost is one linear
    solve rather than an epoch budget.

    ``anchor=True`` (the default) fits the *residual* of whatever read-out the
    net already carries and adds the correction, so the ridge penalty shrinks
    toward that read-out instead of toward zero. Applied to an analytic seed
    that matters: the plain form would solve the backed-out weights away and
    keep only the knots, which is precisely the information the seed exists to
    carry. At ``l2 -> 0`` the two forms coincide, as they should.
    """
    Phi = _design(net, X)
    y = np.asarray(y, dtype=float).ravel()
    target = y - net.predict(X) if anchor else y
    n_cols = Phi.shape[1]
    penalty = l2 * np.eye(n_cols)
    penalty[-1, -1] = 0.0  # never penalize the intercept
    beta = np.linalg.solve(Phi.T @ Phi + penalty, Phi.T @ target)
    h = net.n_hidden
    n_f = net.W1.shape[0]
    out = net.copy()
    if anchor:
        out.w2 = net.w2 + beta[:h]
        out.v = net.v + beta[h : h + n_f]
        out.c = float(net.c + beta[-1])
    else:
        out.w2 = beta[:h]
        out.v = beta[h : h + n_f]
        out.c = float(beta[-1])
    return out


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------


def _axis_aligned_net(
    n_features: int, knots_per_feature: Sequence[tuple[int, np.ndarray]]
) -> ReLUNet:
    """Build layer 1 from ``(feature_index, knots)`` pairs.

    Each hidden unit reads exactly one feature: ``relu(x_f - knot)``. That is
    what the membership expansion produces, and it makes the initial network
    additive across features. Training is free to break the axis alignment --
    and the experiment's whole second half is about watching it do so.
    """
    cols_w: list[np.ndarray] = []
    cols_b: list[float] = []
    for f_idx, knots in knots_per_feature:
        for knot in np.asarray(knots, dtype=float):
            w = np.zeros(n_features, dtype=float)
            w[f_idx] = 1.0
            cols_w.append(w)
            cols_b.append(-float(knot))
    if not cols_w:
        raise ValueError("no knots: cannot build a hidden layer")
    W1 = np.column_stack(cols_w)
    b1 = np.asarray(cols_b, dtype=float)
    h = W1.shape[1]
    return ReLUNet(
        W1=W1,
        b1=b1,
        w2=np.zeros(h),
        v=np.zeros(n_features),
        c=0.0,
    )


def warm_start_from_fis(
    model: GaussianMixtureModel,
    features: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-6,
    half_width_sigma: float = GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
) -> tuple[ReLUNet, dict[str, np.ndarray]]:
    """The conversion: FIS knots become layer 1, read-out is solved in closed form.

    ``X`` must be in the same scaled coordinates the FIS was fitted in -- the
    knots are membership-function parameters, not data-derived quantities, and
    they are only meaningful in that frame.
    """
    knots = fis_knots(model, features, half_width_sigma)
    pairs = [(i, knots[name]) for i, name in enumerate(features) if knots[name].size]
    net = _axis_aligned_net(len(features), pairs)
    return solve_readout(net, X, y, l2), knots


def quantile_start(
    X: np.ndarray, n_hidden: int, y: np.ndarray, l2: float = 1e-6, eps: float = 1e-9
) -> ReLUNet:
    """Ablation: the same architecture with knots at per-feature quantiles.

    Isolates the FIS's *placement* from the ReLU-knot parameterization itself.
    If a hot start were simply an artifact of axis-aligned knots plus a
    closed-form read-out, this arm would match the converted one.
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    per_feature = max(1, n_hidden // n_features)
    qs = (np.arange(per_feature) + 0.5) / per_feature
    pairs = []
    for f in range(n_features):
        knots = merge_knots(np.quantile(X[:, f], qs), tol=eps)
        if knots.size:
            pairs.append((f, knots))
    net = _axis_aligned_net(n_features, pairs)
    return solve_readout(net, X, y, l2)


def random_feature_start(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    l2: float = 1e-6,
) -> ReLUNet:
    """Ablation: random (He) layer 1, read-out solved in closed form.

    The classic random-features / extreme-learning-machine control. It shares
    the converted arm's closed-form read-out and differs only in where layer 1
    came from, which is the comparison that makes "the FIS knew where to put the
    knots" a falsifiable claim rather than a description.
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    W1 = rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, n_hidden))
    b1 = np.zeros(n_hidden)
    net = ReLUNet(W1=W1, b1=b1, w2=np.zeros(n_hidden), v=np.zeros(n_features), c=0.0)
    return solve_readout(net, X, y, l2)


def he_start(rng: np.random.Generator, n_features: int, n_hidden: int) -> ReLUNet:
    """The standard baseline: He-normal layer 1, small random read-out."""
    W1 = rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=n_hidden)
    v = np.zeros(n_features)
    return ReLUNet(W1=W1, b1=b1, w2=w2, v=v, c=0.0)


# ---------------------------------------------------------------------------
# Backing the equivalence out into the seed weights
# ---------------------------------------------------------------------------
#
# `warm_start_from_fis` above takes only the FIS's *knots* and then asks least
# squares for the read-out. That throws away everything the FIS knew about what
# happens between the knots -- its consequents, its rule weights, its gating.
# The functions below keep it.
#
# The route is the equivalence read at the level of the FIS's input-output
# function rather than its internal gates. Bede/Kreinovich/Toth's identity says
# a continuous piecewise-linear function of one variable *is* a one-hidden-layer
# ReLU network, with slope changes as the output weights; it does not care how
# the piecewise-linear function was produced. So instead of demanding that the
# firing strengths themselves be piecewise linear -- which is what forces the
# min/max-versus-product gating question, and what the tetrahedral construction
# of the 2025 paper exists to solve -- we take the FIS's own one-dimensional
# profiles, which are functions we can evaluate exactly, and convert *those*.
#
# The consequence worth stating plainly: **the choice of t-norm stops being
# load-bearing.** A product t-norm makes firing strengths piecewise multilinear
# and kills any exact gate-level conversion; it does not stop us evaluating the
# FIS at a knot. `analysis_gating.py` measures whether the choice still matters
# empirically, now that it no longer matters structurally.


def pwl_to_relu_weights(
    knots: np.ndarray, values: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Exact ReLU decomposition of the piecewise-linear interpolant of ``(t, v)``.

    Returns ``(base_slope, intercept, coeffs)`` such that

        g(x) = intercept + base_slope * x + sum_j coeffs[j] * relu(x - knots[j])

    reproduces every ``(knots[j], values[j])`` pair exactly and is linear
    between and beyond them. ``coeffs[j]`` is the *change* in slope at knot
    ``j`` -- the second difference of the sampled values -- which is precisely
    the output weight the equivalence assigns to that knot's hidden unit. The
    first and last knots carry no slope change (the function is extended
    linearly outside the sampled range), so their coefficients are zero.

    **Extrapolation is unbounded, and it is a real error term.** Past the last
    knot the reconstruction keeps the slope of the last *segment* forever, and
    likewise below the first. On a FIS whose Gaussian-derived knots do not span
    the data this is not a corner case: on N-CMAPSS DS02's `honest` pipeline,
    42% of test rows fall outside at least one FIS feature's knot range, and at
    the one-feature end of the fidelity sweep that extrapolation is essentially
    the whole residual (seed 0.070 relative against a best-additive 0.030).
    Callers who care should either clip inputs into the knot range before the
    hidden layer or widen the knot set; measuring the outside-range fraction
    alongside any fidelity number is the minimum, since a good fidelity score
    on a knot-spanning dataset says nothing about one where it does not span.
    """
    t = np.asarray(knots, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.ndim != 1 or t.shape != v.shape:
        raise ValueError("knots and values must be matching 1-D arrays")
    m = t.size
    coeffs = np.zeros(m, dtype=float)
    if m == 0:
        return 0.0, 0.0, coeffs
    if m == 1:
        return 0.0, float(v[0]), coeffs

    seg = np.diff(v) / np.diff(t)  # slope of each segment, length m-1
    base = float(seg[0])
    coeffs[1:-1] = np.diff(seg)  # slope change at each interior knot
    intercept = float(v[0] - base * t[0])
    return base, intercept, coeffs


def partial_dependence(
    predict_fn,
    X: "pd.DataFrame",
    feature: str,
    grid: np.ndarray,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """The FIS's average response to ``feature``, holding the joint data fixed.

    ``g_f(t) = mean_i FIS(x_i with x_i[f] := t)`` over a background sample of
    rows. This is the first-order term of the functional ANOVA decomposition of
    the FIS -- under independent inputs it is the exact projection of the FIS
    onto functions of ``feature`` alone, which is the best any additive seed can
    do, so it is the right thing to back out rather than a convenient proxy.

    It consumes ``X`` but never ``y``: this is a conversion of the FIS, not a
    refit against labels. (The module docstring's "no data" is about the exact
    1-D theorem in :func:`fis_to_relu_net_1d`; *this* path is label-free, which
    is the weaker and accurate claim.)

    **Only sound for a 0th- or 1st-order TSK.** Overwriting one column sends
    every background row to a point the joint distribution may never visit, and
    the FIS is then evaluated off its own data manifold. With affine
    consequents that extrapolates linearly and stays sane. With
    ``tsk_order="full-2nd"`` the consequent is quadratic and it does not: on
    N-CMAPSS DS02 the resulting seed sits 31x the FIS's own standard deviation
    away from it, against 1.3x for the same pipeline converted at 1st order.
    The failure is in this probe, not in the decomposition downstream -- the
    best-achievable additive fit computed the same way blows up identically.
    Restrict the grid to the feature's *conditional* support (an ALE-style
    profile), or convert a lower-order FIS, before reading anything into a
    fidelity number from a 2nd-order system.
    """
    import pandas as pd  # local: keeps the module's hard dependency numpy-only

    rows = X if background is None else X.iloc[background]
    n = len(rows)
    tiled = pd.concat([rows] * len(grid), ignore_index=True)
    tiled[feature] = np.repeat(np.asarray(grid, dtype=float), n)
    preds = np.asarray(predict_fn(tiled), dtype=float).reshape(len(grid), n)
    return preds.mean(axis=1)


def analytic_seed_from_fis(
    predict_fn,
    X: "pd.DataFrame",
    features: Sequence[str],
    knots: dict[str, np.ndarray],
    background_size: int = 256,
    seed: int = 0,
) -> ReLUNet:
    """Seed weights derived from the FIS's own response, with no label fitting.

    For each feature the FIS's partial-dependence profile is sampled at that
    feature's knots and converted by :func:`pwl_to_relu_weights`; the slope
    changes become hidden-unit output weights, the leading slopes become the
    linear skip, and the constants are summed into the bias with the additive
    decomposition's centering term.

    In one dimension there is nothing to average over, the profile *is* the FIS,
    and the seed reproduces it exactly at every knot -- the equivalence, with
    the FIS's consequents carried into the weights rather than re-estimated.
    In more dimensions it is the additive part of the FIS, exactly.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    background = (
        rng.choice(n, background_size, replace=False)
        if background_size and n > background_size
        else np.arange(n)
    )

    pairs = [(i, knots[f]) for i, f in enumerate(features) if knots[f].size]
    net = _axis_aligned_net(len(features), pairs)

    baseline = float(np.mean(np.asarray(predict_fn(X.iloc[background]), dtype=float)))

    w2 = np.zeros(net.n_hidden, dtype=float)
    v = np.zeros(len(features), dtype=float)
    c = baseline
    at = 0
    for f_idx, ks in pairs:
        profile = partial_dependence(predict_fn, X, features[f_idx], ks, background)
        base_slope, intercept, coeffs = pwl_to_relu_weights(ks, profile)
        w2[at : at + ks.size] = coeffs
        v[f_idx] = base_slope
        # Each feature's profile already contains the baseline, so every profile
        # beyond the first would re-add it; subtract it back out per feature.
        c += intercept - baseline
        at += ks.size

    net.w2 = w2
    net.v = v
    net.c = float(c)
    return net


# ---------------------------------------------------------------------------
# The exact one-dimensional conversion (the theorem, executable)
# ---------------------------------------------------------------------------


def fis_to_relu_net_1d(
    terms: Sequence[TriangularMembership], consequents: Sequence[float]
) -> ReLUNet:
    """Convert a 1-D zeroth-order TSK system on a Ruspini partition, exactly.

    Preconditions -- all three are what make the conversion an identity rather
    than an approximation:

    * one input,
    * triangular terms forming a partition of unity (so the firing-strength
      normalization is division by 1, and the only non-PWL step disappears),
    * singleton consequents (so the output is a fixed linear combination of the
      terms).

    Under them ``y(x) = sum_l m_l * T_l(x)`` is continuous piecewise linear, and
    stacking the terms' ReLU expansions gives a one-hidden-layer network equal
    to it at every point -- no data touched, no fitting. This is the executable
    form of Bede, Kreinovich & Toth's 1-D equivalence.
    """
    if len(terms) != len(consequents):
        raise ValueError("one consequent per term")
    expansions = [triangle_to_relu(t) for t in terms]
    knots = merge_knots(np.concatenate([e.knots for e in expansions] or [np.array([])]))
    index = {round(float(k), 12): i for i, k in enumerate(knots)}

    w2 = np.zeros(len(knots))
    c = 0.0
    for m, exp in zip(consequents, expansions):
        c += float(m) * exp.bias
        for knot, coeff in zip(exp.knots, exp.coeffs):
            j = index.get(round(float(knot), 12))
            if j is None:  # merged knot: fall back to nearest
                j = int(np.argmin(np.abs(knots - knot)))
            w2[j] += float(m) * float(coeff)

    return ReLUNet(
        W1=np.ones((1, len(knots))),
        b1=-knots.astype(float),
        w2=w2,
        v=np.zeros(1),
        c=float(c),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    epochs: list[int]
    train_rmse: list[float]
    test_rmse: list[float]
    val_rmse: list[float]
    seconds: list[float]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, used for the binary-classification arms."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse of :func:`sigmoid`, clipped away from the asymptotes.

    The conversion seeds a *logit*, not a probability: the network's scalar
    output is what a sigmoid is applied to, so backing the FIS out in
    probability space and then squashing it again would compose two sigmoids
    and misplace every weight. A FIS that returns a hard 0 or 1 -- which
    `TribbleClassifier` does routinely, since a normalized firing strength can
    saturate -- would otherwise map to an infinite target.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def error_rate(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Misclassification rate at a 0 logit threshold."""
    return float(
        np.mean(
            (np.asarray(logits).ravel() > 0.0) != (np.asarray(y_true).ravel() > 0.5)
        )
    )


def log_loss(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Mean binary cross-entropy, computed from logits without forming p."""
    z = np.asarray(logits, dtype=float).ravel()
    t = np.asarray(y_true, dtype=float).ravel()
    return float(np.mean(np.maximum(z, 0.0) - z * t + np.log1p(np.exp(-np.abs(z)))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = (
        np.asarray(y_true, dtype=float).ravel()
        - np.asarray(y_pred, dtype=float).ravel()
    )
    return float(np.sqrt(np.mean(d * d)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def train_adam(
    net: ReLUNet,
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 3e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    seed: int = 0,
    eval_every: int = 1,
    eval_batches: int | None = None,
    track_train: bool = True,
    y_scale: float = 1.0,
    y_center: float = 0.0,
    loss: str = "mse",
    metric_fn=None,
) -> tuple[ReLUNet, TrainHistory]:
    """Minibatch Adam on the MSE, identical for every arm.

    ``y_scale``/``y_center`` map the network's (standardized) output back to the
    target's own units for reporting, so an arm is never scored in a frame of
    its own choosing.

    ``loss="bce"`` treats the network's scalar output as a **logit** and
    optimizes binary cross-entropy instead. Only the output-layer gradient
    changes -- ``sigmoid(pred) - y`` in place of ``2 * (pred - y)`` -- because
    everything below it is the same network; keeping one training loop for both
    is what stops the regression and classification arms differing by an
    optimizer detail nobody meant to introduce. ``metric_fn(y_true, raw_pred)``
    overrides the reported curve (error rate rather than RMSE, for instance).

    ``eval_batches`` records the curve every N *minibatches* instead of every
    epoch, and the recorded "epoch" becomes fractional. At 160k rows an epoch is
    313 updates and a network can cross every quality target inside the first
    one, which makes an epoch-resolution time-to-target table read as a row of
    ties. This is the knob that makes the comparison measurable at scale rather
    than a statement about the granularity of the ruler.
    """
    import time

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    net = net.copy()

    params = ["W1", "b1", "w2", "v", "c"]
    m = {p: np.zeros_like(np.atleast_1d(getattr(net, p)), dtype=float) for p in params}
    vv = {p: np.zeros_like(np.atleast_1d(getattr(net, p)), dtype=float) for p in params}
    t = 0

    hist = TrainHistory([], [], [], [], [])

    if loss not in ("mse", "bce"):
        raise ValueError(f"loss must be 'mse' or 'bce', got {loss!r}")

    def _score(Xe, ye):
        if Xe is None or ye is None:
            return float("nan")
        if metric_fn is not None:
            return float(metric_fn(np.asarray(ye).ravel(), net.predict(Xe)))
        return rmse(
            np.asarray(ye).ravel() * y_scale + y_center,
            net.predict(Xe) * y_scale + y_center,
        )

    # Wall clock spent *measuring* rather than training. `hist.seconds` used to
    # be `perf_counter() - start`, which charged every prior evaluation pass
    # over X_test/X_val to the training time it was supposed to be reporting.
    # That cancels between arms of equal width and does not cancel otherwise --
    # and the comparison this module exists for puts a hot arm whose width is
    # fixed by the FIS's knot count (264 units on N-CMAPSS DS02) against an `he`
    # arm free to be narrow (8), a 33x difference in per-evaluation cost billed
    # to the wrong column. Subtracting it makes every recorded second a second
    # of gradient descent.
    eval_seconds = 0.0

    def record(epoch: float) -> None:
        nonlocal eval_seconds
        t_rec = time.perf_counter()
        hist.epochs.append(epoch)
        hist.seconds.append(t_rec - start - eval_seconds)
        # Scoring the training set is the most expensive part of a record, and
        # at sub-epoch cadence on 160k rows it costs more than the training it
        # is measuring. Nothing in this experiment selects on the train curve.
        hist.train_rmse.append(_score(X, y) if track_train else float("nan"))
        hist.test_rmse.append(_score(X_test, y_test))
        hist.val_rmse.append(_score(X_val, y_val))
        eval_seconds += time.perf_counter() - t_rec

    start = time.perf_counter()
    record(0.0)
    n_batches = max(1, int(np.ceil(n / batch_size)))

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        # `bi` and not `b`: `b` is the batch *size* three lines down, and
        # letting the index share the name silently disabled sub-epoch eval.
        for bi, lo in enumerate(range(0, n, batch_size)):
            idx = order[lo : lo + batch_size]
            Xb, yb = X[idx], y[idx]
            b = Xb.shape[0]

            z = Xb @ net.W1 + net.b1
            h = np.maximum(z, 0.0)
            pred = h @ net.w2 + Xb @ net.v + net.c
            if loss == "bce":
                g_out = (sigmoid(pred) - yb) / b
            else:
                g_out = (2.0 / b) * (pred - yb)

            grads = {
                "w2": h.T @ g_out,
                "v": Xb.T @ g_out,
                "c": np.atleast_1d(float(g_out.sum())),
            }
            g_h = np.outer(g_out, net.w2) * (z > 0.0)
            grads["W1"] = Xb.T @ g_h
            grads["b1"] = g_h.sum(axis=0)

            t += 1
            for p in params:
                g = grads[p]
                m[p] = beta1 * m[p] + (1 - beta1) * g
                vv[p] = beta2 * vv[p] + (1 - beta2) * (g * g)
                m_hat = m[p] / (1 - beta1**t)
                v_hat = vv[p] / (1 - beta2**t)
                step = lr * m_hat / (np.sqrt(v_hat) + eps)
                if p == "c":
                    net.c = float(net.c - step[0])
                else:
                    setattr(net, p, getattr(net, p) - step)

            if eval_batches and (bi + 1) % eval_batches == 0:
                record(epoch - 1 + (bi + 1) / n_batches)

        if eval_batches is None and (epoch % eval_every == 0 or epoch == epochs):
            record(float(epoch))
    if eval_batches:
        record(float(epochs))

    return net, hist
