"""Exact ReLU expansions of the package's membership shapes.

The construction rests on one identity (Bede, Kreinovich & Toth, NAFIPS 2023 --
see ``papers/nn-fis-equivalence/``): a triangular membership function *is* a
short sum of ReLUs of the input, exactly, with no approximation anywhere.

    T(x; a, b, c) = s_a * relu(x - a) - (s_a + s_c) * relu(x - b) + s_c * relu(x - c)
    s_a = 1 / (b - a),  s_c = 1 / (c - b)

so a fuzzy term is a hidden-layer motif and its apex/foot knots are the ReLU
bias terms. :func:`membership_to_relu` turns any of the package's membership
shapes into that expansion (Gaussians are first fitted to triangles by the
package's own :mod:`tribblefis.triangle_fit`, which is the only lossy step and
is reported as such).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

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
