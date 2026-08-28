"""Backing the FIS's own equivalence out into every network weight, not just the knots.

``fis2nn_init.warm_start_from_fis`` takes only the FIS's *knots* and then asks
least squares for the read-out. That throws away everything the FIS knew about
what happens between the knots -- its consequents, its rule weights, its
gating. The functions below keep it.

The route is the equivalence read at the level of the FIS's input-output
function rather than its internal gates. Bede/Kreinovich/Toth's identity says
a continuous piecewise-linear function of one variable *is* a one-hidden-layer
ReLU network, with slope changes as the output weights; it does not care how
the piecewise-linear function was produced. So instead of demanding that the
firing strengths themselves be piecewise linear -- which is what forces the
min/max-versus-product gating question, and what the tetrahedral construction
of the 2025 paper exists to solve -- we take the FIS's own one-dimensional
profiles, which are functions we can evaluate exactly, and convert *those*.

The consequence worth stating plainly: **the choice of t-norm stops being
load-bearing.** A product t-norm makes firing strengths piecewise multilinear
and kills any exact gate-level conversion; it does not stop us evaluating the
FIS at a knot. `analysis_gating.py` measures whether the choice still matters
empirically, now that it no longer matters structurally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pandas is used only in the partial-dependence path
    import pandas as pd

from fis2nn_init import _axis_aligned_net
from fis2nn_network import ReLUNet


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
    1-D theorem in :func:`fis2nn_convert1d.fis_to_relu_net_1d`; *this* path is
    label-free, which is the weaker and accurate claim.)

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
