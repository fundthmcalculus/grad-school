"""Overlap modeling for TRIBBLE's TSK regressor: soft output-bucket boundaries.

The idea under test
-------------------
TRIBBLE partitions the target into ``n_output_buckets`` and treats each bucket as
one rule. Every per-bucket quantity is then fit on a **hard** slice of the
training data -- ``y_bucket == r`` and nothing else:

* the antecedent membership functions, `gauss_math.fit_gaussians`, which sees
  ``X[column][y == label_value]``;
* the bucket centroid ``y_bucket_mean[r]``;
* (on the legacy path) the per-rule consequent polynomial.

A sample one cent above a bucket edge contributes to rule ``r+1`` and not at all
to rule ``r``, even though it is nearly indistinguishable from the sample one
cent below. This module makes each rule's fitting slice **overlap its
neighbours'** by a fraction of their points, so the quantities each rule is
built from vary smoothly across an edge instead of stepping. The extreme buckets
have only one neighbour, so their overlap is one-sided -- they keep a hard outer
edge because there is no data past it to blend with.

What "a percentage of overlap" means here
-----------------------------------------
Overlap is measured in **rank space, as a fraction of the neighbour's point
count** -- ``overlap=0.25`` means rule ``r`` also fits on the 25% of bucket
``r+1``'s points nearest the shared edge, and the 25% of bucket ``r-1``'s points
nearest theirs. Rank space rather than value space because that is what the
request asks for ("a certain percentage of the data points"), and because it
behaves identically under ``output_partition="uniform"`` and ``"quantile"``,
whose bucket widths differ by construction.

Two band profiles:

``shape="flat"``
    Borrowed points carry weight 1, exactly like the rule's own. The literal
    reading: the slices simply overlap.
``shape="ramp"``
    Borrowed points carry a weight falling linearly from 1 at the shared edge to
    ``1/m`` at the far end of the band. Own points stay at 1, so the resulting
    membership in rank space is a **trapezoid**: a plateau over the rule's own
    bucket and a shoulder into each neighbour. This is the smoother of the two
    and the one that has a boundary-continuous limit.

Where the overlap is applied is a separate axis from how wide it is, because the
three consumers are independent and only one of them changes what happens at
*predict* time:

* ``overlap_antecedents`` -- widens the membership functions. This is the only
  switch that reaches prediction: the firing strengths themselves get smoother,
  so the model has no hard edge at inference, not merely during fitting.
* ``overlap_means`` -- the bucket centroids become overlap-weighted means, so
  the output "ladder" the consequents correct from has its rungs pulled toward
  each other.
* ``consequent_fit="local"`` -- fits each rule's polynomial on its own
  overlap-weighted slice. This is the literal request applied to the consequent
  equations. Note that TRIBBLE's shipped default (``"global"``) already couples
  every rule to every sample through the firing-weighted stacked ridge solve, so
  overlap cannot enter it as data sharing; ``"local"`` is the arm where per-rule
  data sharing is a meaningful quantity at all, and ``local`` at ``overlap=0``
  is the necessary control that separates "local instead of global" from
  "overlapping instead of hard".

``fusion_reg`` is the same intent expressed as a penalty rather than as shared
data: it adds ``fusion_reg * sum_r ||c_{r+1} - c_r||^2`` to the global ridge, so
adjacent rules' correction coefficients are pulled toward agreement, which is
what unbounded data sharing between neighbours converges to. It stays inside the
closed-form solve (one more block of augmented rows) and, unlike ``"local"``, it
does not give up the exact firing-weighted optimum's structure.

Everything degenerates: ``overlap=0``, ``consequent_fit="global"``,
``fusion_reg=0`` reproduces `TribbleRegressor` prediction-for-prediction. That
equality is asserted in `test_overlap.py`, and it is the reason any difference
this module measures can be attributed to the overlap.
"""

from __future__ import annotations

import typing
import uuid
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from tribblefis.gauss_data import (
    FeatureModel,
    TrapezoidMembership,
    GaussianMembership,
    GaussianMixtureModel,
    LabelModel,
    DefaultNormCornorm,
)
from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    fit_gaussians,
    resolve_norm_pair,
    take_top_features,
    tsk_firing_strengths,
)
from tribblefis.regression import (
    _normalize_firing_strengths,
    build_consequent_features,
    partition_output,
    predict_tsk,
    rule_consequent_values,
    solve_tsk_consequents_from_firing,
)

VALID_SHAPES = ("flat", "ramp")
VALID_CONSEQUENT_FITS = ("global", "local", "local-residual", "shrink-local")
VALID_PREDICT_MODES = ("blend", "wta")

# Floor on a fitted sigma, mirroring the guard the library's own mixture fit
# applies: a component that collapses onto a single repeated value would make
# membership a delta function and the firing strengths all-or-nothing.
_SIGMA_FLOOR_FRAC = 1e-3


# --------------------------------------------------------------------------
# The overlap membership matrix
# --------------------------------------------------------------------------
VALID_BANDS = ("adjacent", "random")


def overlap_weights(
    y,
    labels,
    n_buckets: int,
    fraction: float,
    shape: str = "flat",
    band: str = "adjacent",
    random_state: int = 42,
) -> np.ndarray:
    """Per-(sample, bucket) fitting weights with rank-space overlap bands.

    Returns ``W`` of shape ``(n_samples, n_buckets)``. ``W[i, b] > 0`` means
    sample ``i`` takes part in fitting bucket ``b``'s rule; the value is how much
    it counts. Rows do **not** sum to 1 -- this is a fitting weight, not a
    probability, and normalizing it would undo the overlap by giving a shared
    point half a vote in each of two rules instead of a full vote in both.

    ``fraction`` is read against the *neighbour's* size: bucket ``b`` borrows
    ``round(fraction * n_{b-1})`` points from below and
    ``round(fraction * n_{b+1})`` from above. Bucket 0 borrows only from above
    and bucket ``n_buckets - 1`` only from below -- the one-sided ends.

    ``fraction=0`` returns the hard indicator matrix, so this function is the
    identity on the baseline.

    Ties in ``y`` are split by a stable sort, i.e. arbitrarily but
    reproducibly. A band boundary that lands inside a run of equal target values
    therefore takes some of them and not others; on a heavily tied target
    prefer ``shape="flat"``, whose weights do not depend on within-band rank.

    ``band="random"`` is the control, not a mode anyone would deploy. It borrows
    the same *number* of rows from each neighbour and hands them the same
    *multiset* of weights, but draws them uniformly from that neighbour instead
    of from the shared edge. Everything the adjacent band changes except the one
    thing it is supposed to change is therefore held fixed: rows per fit,
    weight distribution, the widened slice's effect on BIC component selection,
    and the number of candidates a validation sweep gets to choose from. A gain
    that the random band reproduces is not a gain from softening a boundary.
    """
    if not VALID_SHAPES.count(shape):
        raise ValueError(f"shape must be one of {VALID_SHAPES}, got {shape!r}")
    if band not in VALID_BANDS:
        raise ValueError(f"band must be one of {VALID_BANDS}, got {band!r}")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction!r}")

    y = np.asarray(y, dtype=float).ravel()
    labels = np.asarray(labels).ravel()
    W = np.zeros((len(y), n_buckets), dtype=float)

    members = [np.flatnonzero(labels == b) for b in range(n_buckets)]
    for b, idx in enumerate(members):
        W[idx, b] = 1.0

    if fraction <= 0.0:
        return W

    for b in range(n_buckets):
        for neighbour, take_high in ((b - 1, True), (b + 1, False)):
            if neighbour < 0 or neighbour >= n_buckets:
                continue                      # an end bucket: one-sided overlap
            src = members[neighbour]
            m = int(round(fraction * len(src)))
            if m <= 0:
                continue
            ordered = src[np.argsort(y[src], kind="stable")]
            # The band runs from the shared edge outward, so index 0 is the point
            # most like this bucket's own and the ramp can decay along it.
            rows = ordered[-m:][::-1] if take_high else ordered[:m]
            w = 1.0 - np.arange(m) / m if shape == "ramp" else np.ones(m)
            if band == "random":
                # Same count, same weights, boundary structure destroyed. Seeded
                # per (bucket, neighbour) so the two sides draw independently and
                # the whole matrix is reproducible.
                rng = np.random.default_rng(
                    (random_state, b, neighbour, n_buckets))
                rows = rng.choice(src, size=m, replace=False)
            W[rows, b] = np.maximum(W[rows, b], w)

    return W


def overlap_bucket_means(y, W: np.ndarray, hard_means: np.ndarray,
                         pin_extremes: bool) -> np.ndarray:
    """Overlap-weighted bucket centroids, falling back to ``hard_means`` when empty.

    ``pin_extremes`` restores the two end centroids to whatever ``hard_means``
    holds for them -- normally the observed min and max that `partition_output`
    pins there, which the weighted mean would pull inward.
    """
    y = np.asarray(y, dtype=float).ravel()
    means = np.array(hard_means, dtype=float).copy()
    totals = W.sum(axis=0)
    nonempty = totals > 0
    means[nonempty] = (W[:, nonempty] * y[:, None]).sum(axis=0) / totals[nonempty]
    if pin_extremes and len(means) >= 2:
        means[0] = hard_means[0]
        means[-1] = hard_means[-1]
    return means


# --------------------------------------------------------------------------
# Antecedents: membership functions fit on the overlapped slice
# --------------------------------------------------------------------------
def _weighted_component_moments(values: np.ndarray, weights: np.ndarray,
                                components: list) -> list:
    """Re-place fitted components' (mu, sigma) as weighted moments of their members.

    The library's `fit_gaussian_mixture_1d` has no sample-weight argument, so the
    ``ramp`` profile is applied in two steps: the component *count and partition*
    come from the unweighted fit over the band-extended slice (which is what BIC
    selection is for, and what keeps this path identical to the library's at
    ``fraction=0``), then each component's location and width are recomputed as
    weighted moments of the points nearest it. A component whose members carry no
    weight is returned untouched.
    """
    if not components:
        return components
    mus = np.array([c.mu for c in components], dtype=float)
    assign = np.argmin(np.abs(values[:, None] - mus[None, :]), axis=1)
    floor = _SIGMA_FLOOR_FRAC * max(float(np.std(values)), np.finfo(float).tiny)

    out = []
    for k, comp in enumerate(components):
        sel = assign == k
        w = weights[sel]
        total = float(w.sum())
        if total <= 0 or sel.sum() == 0:
            out.append(comp)
            continue
        v = values[sel]
        mu = float((w * v).sum() / total)
        var = float((w * (v - mu) ** 2).sum() / total)
        out.append(GaussianMembership.create(mu=mu, sigma=max(np.sqrt(var), floor)))
    return out


def build_overlap_membership_model(
    X: pd.DataFrame,
    W: np.ndarray,
    feature_names: list,
    n_gaussians: int = 0,
    max_samples: int | None = None,
    random_state: int = 42,
    shape: str = "flat",
) -> GaussianMixtureModel:
    """`create_gaussian_membership_dict`, but each label fit on its overlapped slice.

    Reuses `fit_gaussians` verbatim -- including its categorical-column branch and
    its ``max_samples`` subsampling -- by handing it a synthetic label series that
    marks exactly this bucket's overlapped support. That is deliberate: a
    re-implementation would have to be kept in step with the library's, and the
    point of this experiment is to change *which rows each rule sees*, nothing
    else about how a membership function is fit.
    """
    n_buckets = W.shape[1]
    # Only buckets that actually carry data become rules. `partition_output` can
    # leave a bucket empty (equal-width edges over a skewed target starve the
    # tails, which the library warns about), and `create_gaussian_membership_dict`
    # keys its labels off `y.unique()` -- so including an empty bucket here would
    # silently add a rule the baseline does not have and break the comparison.
    occupied = [b for b in range(n_buckets) if bool((W[:, b] > 0).any())]
    ordered_models = {}
    for name in feature_names:
        label_models = {}
        for b in occupied:
            w = W[:, b]
            support = w > 0
            # `fit_gaussians` selects with `y == label_value`; -1 is never a
            # bucket label, so this reproduces the hard mask at fraction=0.
            y_synth = pd.Series(np.where(support, b, -1), index=X.index)
            memberships = fit_gaussians(
                X, y_synth, name, b, n_gaussians,
                max_samples=max_samples, random_state=random_state, verbose=False,
            )
            if shape == "ramp" and memberships and support.any():
                col = X[name]
                numeric = not (
                    pd.api.types.is_object_dtype(col.dtype)
                    or pd.api.types.is_bool_dtype(col.dtype)
                    or pd.api.types.is_string_dtype(col.dtype)
                )
                # The categorical branch of `fit_gaussians` ignores the label
                # entirely (one near-delta per distinct value), so weighting it
                # would be meaningless -- leave those alone.
                if numeric:
                    vals = col[support].to_numpy(dtype=float)
                    memberships = _weighted_component_moments(
                        vals, w[support], memberships)
            label_models[b] = LabelModel(memberships=memberships)
        ordered_models[name] = FeatureModel(label_models=label_models)
    return GaussianMixtureModel(feature_models=ordered_models)


def _build_with_overlap_slices(build_fn, W: np.ndarray, index, feature_names):
    """Run a whole-model fitter once per bucket, each on that bucket's widened slice.

    `create_trapz_membership_dict` fits every (feature, label) pair in one call,
    so unlike `fit_gaussians` it cannot be handed a single bucket's support. This
    calls it once per bucket with a label series that marks only that bucket's
    overlapped rows -- every other row labelled -1, which is never a bucket -- and
    keeps the one label model it asked for. That is ``n_buckets`` fits of a
    ``n_buckets``-label model, so it costs more than the Gaussian path; it is the
    price of reusing the library's fitter instead of reimplementing a weighted
    trapezoid EM, and it keeps the two membership families comparable.
    """
    n_buckets = W.shape[1]
    occupied = [b for b in range(n_buckets) if bool((W[:, b] > 0).any())]
    per_feature: dict = {name: {} for name in feature_names}
    for b in occupied:
        support = W[:, b] > 0
        labels = pd.Series(np.where(support, b, -1), index=index)
        sub = build_fn(labels)
        for name in feature_names:
            fmodel = sub.feature_models.get(name)
            if fmodel is None or b not in fmodel.label_models:
                per_feature[name][b] = LabelModel(memberships=[])
            else:
                per_feature[name][b] = fmodel.label_models[b]
    return GaussianMixtureModel(feature_models={
        name: FeatureModel(label_models=per_feature[name]) for name in feature_names})


# --------------------------------------------------------------------------
# Consequents
# --------------------------------------------------------------------------
def _ridge_lstsq(A: np.ndarray, b: np.ndarray, penalty: np.ndarray,
                 l2_reg: float) -> np.ndarray:
    """Ridge solve as augmented rows on the design, matching the library's path.

    `solve_tsk_consequents_from_firing` deliberately augments rather than forming
    normal equations -- the augmented design's condition number is the square root
    of the Gram's, and small singular values get truncated instead of inverted.
    Using the same construction here keeps the local arm's numerics comparable to
    the global arm's rather than introducing a second failure mode.
    """
    if l2_reg > 0:
        root = np.sqrt(l2_reg * penalty)
        A = np.vstack([A, np.diag(root)])
        b = np.hstack([b, np.zeros_like(root)])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def solve_consequents_local(
    firing_strengths: np.ndarray,
    labels: list,
    X_train: pd.DataFrame,
    top_n_todo: list,
    y_bucket_mean,
    y_train: pd.DataFrame,
    W: np.ndarray,
    order: str = "2nd",
    l2_reg: float = 1e-6,
    basis: str = "raw",
    cross_pairs=None,
    pin_extremes: bool = False,
    residual_form: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit each rule's consequent on its own (optionally overlapping) slice.

    Rule ``r`` solves ``min sum_i W[i, b_r] * (y_i - (mean_r + phi_i . c_r))^2``
    plus the same ridge penalty the global solver uses (intercept unpenalized).
    With ``W`` the hard indicator this is the classical per-bucket local TSK fit
    -- a ridge-regularized `compute_*_order_corrections` -- and with an overlap
    band it is that fit with neighbouring data blended in.

    ``residual_form=True`` is the library's literal legacy formulation: the rule's
    constant is held at ``y_bucket_mean[bucket]`` and the corrections are fitted to
    the residual ``y - y_bucket_mean[bucket]``, exactly as
    `regression.compute_{first,second,third,full_second}_order_corrections` do
    (with a ridge added, which they lack). The default frees the intercept instead,
    which can only fit at least as well on the slice -- so running both settles
    whether the local arm's deficit is an artifact of how its constant was chosen.

    ``firing_strengths`` is consumed only for its shape and column-to-bucket map
    (``labels``); a local fit does not weight by firing, which is exactly what
    distinguishes it from `solve_tsk_consequents_from_firing`. Returns
    ``(corr_terms, y_bucket_mean_opt)`` in the same layout the shared
    `predict_tsk` path expects, so nothing downstream changes.
    """
    n_rules = firing_strengths.shape[1]
    X_rule = X_train[top_n_todo].to_numpy()
    feats = build_consequent_features(X_rule, order, basis=basis, cross_pairs=cross_pairs)
    phi = np.hstack([np.ones((X_rule.shape[0], 1)), feats])
    n_terms = feats.shape[1]

    y = np.asarray(y_train["y_value"].values, dtype=float)
    ybm = np.asarray(y_bucket_mean, dtype=float).ravel()

    corr = np.zeros((n_rules, n_terms))
    means = np.zeros(n_rules)

    penalty = np.ones(1 + n_terms)
    penalty[0] = 0.0                      # the intercept is the bucket mean

    for r in range(n_rules):
        bucket = int(labels[r])
        w = W[:, bucket]
        support = w > 0
        have_mean = ybm.size > bucket and np.isfinite(ybm[bucket])
        # `residual_form` fixes every rule's constant, not just the two extremes,
        # so it subsumes the pin. Both routes go through the same exact-constraint
        # branch below.
        pin = have_mean and (
            residual_form
            or (pin_extremes and n_rules >= 2 and r in (0, n_rules - 1)))

        if not support.any():
            # An empty rule keeps the centroid it was handed and corrects nothing.
            means[r] = ybm[bucket] if ybm.size > bucket else 0.0
            continue

        root_w = np.sqrt(w[support])
        A = phi[support] * root_w[:, None]
        b = y[support] * root_w

        if pin:
            # Same exact-constraint treatment as the global solver: move the known
            # intercept to the right-hand side and solve the rest against the
            # residual, rather than penalizing it toward the pinned value.
            value = float(ybm[bucket])
            beta = _ridge_lstsq(A[:, 1:], b - A[:, 0] * value, penalty[1:], l2_reg)
            means[r] = value
            corr[r] = beta
        else:
            beta = _ridge_lstsq(A, b, penalty, l2_reg)
            means[r] = beta[0]
            corr[r] = beta[1:]

    return corr, means


def solve_consequents_fused(
    firing_strengths: np.ndarray,
    labels: list,
    X_train: pd.DataFrame,
    top_n_todo: list,
    y_bucket_mean,
    y_train: pd.DataFrame,
    order: str = "2nd",
    l2_reg: float = 1e-6,
    basis: str = "raw",
    cross_pairs=None,
    pin_extremes: bool = False,
    fusion_reg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The global firing-weighted ridge solve, plus a neighbour-agreement penalty.

    Adds ``fusion_reg * sum_r ||c_{r+1} - c_r||^2`` over adjacent rules to the
    objective `solve_tsk_consequents_from_firing` minimizes. This is overlap
    expressed as a penalty instead of as shared rows: sharing a neighbour's data
    pulls two adjacent consequents toward a common fit, and in the limit of total
    sharing they *are* the same polynomial, which is what an infinite
    ``fusion_reg`` gives. Unlike the local arm it keeps the firing-weighted
    coupling that makes the global solve the exact optimum of the quantity the
    model is actually scored on.

    Only the **correction** coefficients are fused, never the intercepts: the
    per-rule intercepts are the output ladder's rungs and are supposed to differ.
    Fusing them would flatten the model toward a single global polynomial.

    Rule columns are in sorted bucket-label order (`FeatureModel.ordered_keys`),
    so column adjacency is bucket adjacency; ``labels`` is checked for that rather
    than assumed.

    At ``fusion_reg=0`` this reduces to the library's solve, and
    `test_overlap.py` asserts the two agree numerically -- the reason it is safe
    to read a difference at ``fusion_reg>0`` as the penalty's effect.
    """
    from tribblefis.regression import _normalize_firing_strengths

    norm_fs = _normalize_firing_strengths(firing_strengths)
    n_rules = norm_fs.shape[1]

    X_rule = X_train[top_n_todo].to_numpy()
    feats = build_consequent_features(X_rule, order, basis=basis, cross_pairs=cross_pairs)
    n_terms = feats.shape[1]
    per_rule = 1 + n_terms

    phi = np.hstack([np.ones((X_rule.shape[0], 1)), feats])
    design = (norm_fs[:, :, np.newaxis] * phi[:, np.newaxis, :]).reshape(
        X_rule.shape[0], n_rules * per_rule
    )
    y = np.asarray(y_train["y_value"].values, dtype=float)

    penalty = np.ones(n_rules * per_rule)
    penalty[::per_rule] = 0.0

    # Neighbour-difference rows: one per (adjacent rule pair, correction term).
    fuse_rows = np.zeros((0, design.shape[1]))
    if fusion_reg > 0 and n_rules >= 2 and n_terms > 0:
        order_by_label = np.argsort(np.asarray(labels))
        root = np.sqrt(fusion_reg)
        rows = []
        for a, b in zip(order_by_label[:-1], order_by_label[1:]):
            for j in range(n_terms):
                row = np.zeros(design.shape[1])
                row[int(a) * per_rule + 1 + j] = root
                row[int(b) * per_rule + 1 + j] = -root
                rows.append(row)
        fuse_rows = np.vstack(rows)

    pinned_cols: list[int] = []
    pinned_vals: list[float] = []
    if pin_extremes and n_rules >= 2 and y_bucket_mean is not None:
        ybm = np.asarray(y_bucket_mean, dtype=float).ravel()
        if ybm.size > int(np.max(labels)):
            for rule_idx in (0, n_rules - 1):
                value = float(ybm[int(labels[rule_idx])])
                if np.isfinite(value):
                    pinned_cols.append(rule_idx * per_rule)
                    pinned_vals.append(value)

    A = np.vstack([design, fuse_rows]) if len(fuse_rows) else design
    rhs = np.hstack([y, np.zeros(len(fuse_rows))]) if len(fuse_rows) else y

    if pinned_cols:
        pinned = np.asarray(pinned_cols, dtype=int)
        values = np.asarray(pinned_vals, dtype=float)
        free = np.setdiff1d(np.arange(A.shape[1]), pinned)
        beta = np.zeros(A.shape[1])
        beta[pinned] = values
        beta[free] = _ridge_lstsq(
            A[:, free], rhs - A[:, pinned] @ values, penalty[free], l2_reg)
    else:
        beta = _ridge_lstsq(A, rhs, penalty, l2_reg)

    coeffs = beta.reshape(n_rules, per_rule)
    corr = coeffs[:, 1:].copy() if n_terms > 0 else np.zeros((n_rules, 0))
    return corr, coeffs[:, 0].copy()


def _mf_bounds(mf, gaussian_k: float = 4.0) -> tuple[float, float]:
    """A plotting/scoring range for any membership shape this experiment builds.

    Deliberately not `gauss_data.mf_interval`: that raises `TypeError` on a type
    it does not know, and this has to keep working for `ClampedGaussianMembership`
    (and anything added later) without editing the library. Falls back to the
    `mu +/- gaussian_k * sigma` convention for anything carrying `mu`/`sigma`.
    """
    if hasattr(mf, "k") and hasattr(mf, "sigma"):          # clamped Gaussian
        return mf.mu - mf.k * mf.sigma, mf.mu + mf.k * mf.sigma
    if hasattr(mf, "d"):                                   # trapezoid
        return mf.a, mf.d
    if hasattr(mf, "c") and hasattr(mf, "a"):              # triangle
        return mf.a, mf.c
    return mf.mu - gaussian_k * mf.sigma, mf.mu + gaussian_k * mf.sigma


def sharpen_firing(firing_strengths: np.ndarray, gamma: float) -> np.ndarray:
    """Raise firing strengths to ``gamma`` before normalization.

    Motivated by stage 2's diagnostic rather than by theory. If per-bucket rules
    are individually good and the blend is what loses accuracy, then the blend's
    *concentration* is a knob worth having: ``gamma > 1`` sharpens toward the
    strongest rule (``gamma -> inf`` is winner-take-all), ``gamma < 1`` flattens
    toward a uniform average, and ``gamma = 1`` is TSK's own weighting. It is one
    scalar, it costs nothing, and it tests whether the local family's deficit is
    a *calibration* problem in the weights rather than a structural one.

    Must be applied identically in the solve and at predict time. That is not a
    style preference here: this codebase has been bitten by a firing-strength
    convention that differed between fit and evaluation (see
    `_normalize_firing_strengths`' docstring), so the exponent goes through one
    function called from both paths.

    Each row is divided by its own maximum before the exponent. Downstream
    normalization is scale-invariant, so this changes nothing mathematically and
    everything numerically: raising raw strengths of order 1e-2 to gamma=10
    underflows the whole row toward zero, `_normalize_firing_strengths` then sees
    a row sum under its 1e-6 floor and returns all-zeros, and the model silently
    predicts 0 for that row. Measured before this guard was added: mean maximum
    weight *fell* from 0.78 at gamma=3 to 0.02 at gamma=150, the opposite of
    sharpening. Rows that were already below the floor are left alone, so the
    all-zero convention is preserved rather than resurrected.
    """
    if gamma == 1.0:
        return firing_strengths
    out = np.clip(firing_strengths, 0.0, None)
    row_max = out.max(axis=1)
    live = out.sum(axis=1) > 1e-6
    scale = np.where(live & (row_max > 0), row_max, 1.0)
    return np.power(out / scale[:, np.newaxis], gamma) * live[:, np.newaxis]


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------
class OverlapTribbleRegressor(BaseEstimator, RegressorMixin):
    """`TribbleRegressor` with soft (overlapping) output-bucket boundaries.

    Deliberately a parallel implementation rather than a subclass:
    `TribbleRegressor.fit` is one straight-line method with no hook between
    partitioning and the membership fit, so overriding it would mean copying the
    body anyway. Copying it *here*, in the experiment, keeps the library
    untouched while the idea is being evaluated -- and makes the diff between the
    two paths readable, which is the whole point of an ablation.

    The default arguments are chosen so that
    ``OverlapTribbleRegressor()`` and ``TribbleRegressor()`` fit the same model;
    `test_overlap.py` asserts it.

    Parameters that do not exist on `TribbleRegressor`
    -------------------------------------------------
    overlap : float, default 0.0
        Band width, as a fraction of the *neighbour* bucket's point count. 0
        reproduces hard boundaries.
    overlap_shape : {"flat", "ramp"}, default "flat"
        Weight profile across the band. See `overlap_weights`.
    overlap_band : {"adjacent", "random"}, default "adjacent"
        ``"random"`` is the control arm: same row count and same weights, drawn
        anywhere in the neighbour instead of at the shared edge. See
        `overlap_weights`.
    overlap_antecedents : bool, default True
        Fit each bucket's membership functions on its overlapped slice. The only
        switch that changes predict-time behaviour (via the firing strengths).
    overlap_means : bool, default True
        Use overlap-weighted bucket centroids.
    consequent_fit : {"global", "local", "local-residual", "shrink-local"}, default "global"
        ``"global"`` is the library's exact firing-weighted stacked ridge solve.
        ``"local"`` fits each rule's polynomial on its own overlapped slice with a
        free intercept; ``"local-residual"`` does the same with the intercept held
        at the bucket centroid, which is the library's legacy
        `compute_*_order_corrections` formulation. ``"shrink-local"`` keeps the
        global solve and uses the local fit as the ridge's prior instead of zero.
    predict_mode : {"blend", "wta"}, default "blend"
        ``"blend"`` is TSK's firing-weighted average. ``"wta"`` answers each row
        with its single strongest-firing rule, which is the aggregation a set of
        good *local* approximators would want. Diagnostic: it makes the model
        piecewise-discontinuous, which is the thing overlap set out to avoid.
    blend_recalibrate : bool, default False
        After solving, fit a per-rule affine recalibration ``(a_r, b_r)`` of the
        blend against the training fold (`solve_blend_recalibration`). Cannot
        change what a rule computes, only how it is combined.
    membership : str, default "gaussian"
        Antecedent shape, one of `VALID_MEMBERSHIPS`. ``"gaussian"`` has infinite
        support, so every rule fires everywhere. ``"clamped"`` keeps the Gaussian
        fit and zeroes it past `clamp_k` sigma. ``"trapezoid"`` is the library's
        fast histogram fitter; ``"trapezoid-em"``/``"triangle-em"`` are its EM
        fitter, kept reachable but ~4000x slower. ``"ruspini"`` re-expresses each
        feature as a shared triangular partition that is both compactly supported
        *and* a partition of unity -- see `ruspinize_features`.
    trapz_ramp : float, default 0.1
        `ramp_width_ratio` for the fast trapezoid fitter -- shoulder width as a
        fraction of the bin count. Note this moves the *plateau*, not the support:
        the fitter computes ``ramp_width = bin_width * int(n_bins * ratio)``, which
        is approximately ``range * ratio`` however many bins there are. It cannot
        widen ``[a, d]``, so it cannot fix coverage.
    trapz_bins : int, default 50
        `n_bins`. This is the knob that governs support. The histogram is taken
        over each bucket's own data, so ``[a, d]`` can never exceed that bucket's
        ``[min, max]`` -- but with many bins over few samples, interior bins fall
        empty, `_find_contiguous_regions` returns several disjoint regions, and
        the gaps between them are dead zones *inside* the bucket's own range.
        Fewer bins merges those regions and closes the gaps; ``n_bins=1`` gives a
        single trapezoid spanning the whole range, which is the widest support the
        parameterization admits.
    trapz_pad : float, default 0.0
        Re-seat fitted trapezoids so their data range is the plateau and the
        support extends this fraction of the region width beyond it
        (`pad_trapezoids`). 0 keeps the library's fitted geometry, whose left edge
        sits exactly on the data minimum and therefore gives zero membership to
        every value tied with it.
    trapz_merge : float, default 0.2
        `merge_width_ratio` -- regions separated by fewer than
        ``int(n_bins * ratio)`` empty bins are merged. The other lever on support,
        and the one that closes gaps without coarsening the histogram.
    ruspini_tol : float, default 0.02
        Apex-merge tolerance for ``membership="ruspini"``, in the feature's units
        (inputs are unit-scaled, so 0.02 is 2% of the range).
    clamp_k : float, default 3.0
        Cutoff in standard deviations for ``membership="clamped"``. 3.0 is the
        convention `gauss_data.mf_interval` already uses for a Gaussian's
        effective support.
    clamp_smooth : bool, default True
        True subtracts the boundary value and rescales, so membership reaches zero
        continuously at the cutoff. False truncates, leaving a step of
        ``exp(-k^2/2)`` there.
    blend_sharpen : float, default 1.0
        Exponent applied to the firing strengths before normalization, in the
        solve and at predict time alike. >1 concentrates the blend on the
        strongest rule, <1 flattens it. 1.0 is TSK's own weighting.
        See `sharpen_firing`.
    fusion_reg : float, default 0.0
        Weight on the adjacent-rule correction-difference penalty, for
        ``consequent_fit="global"`` only. See `solve_consequents_fused`.

    Unsupported on purpose: interaction detection/selection and the RBF basis.
    They are orthogonal to overlap and each adds a branch that would have to be
    kept in step with the library's; the arms in `run_experiment.py` do not use
    them.
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=0,
        n_output_buckets=2,
        output_partition="uniform",
        tsk_order="1st",
        consequent_basis="raw",
        l2_reg=1e-6,
        pin_extremes=False,
        norm_conorm=DefaultNormCornorm,
        t_norm=None,
        t_conorm=None,
        allow_mixed_norms=False,
        random_state=42,
        max_samples=None,
        overlap=0.0,
        overlap_shape="flat",
        overlap_band="adjacent",
        overlap_antecedents=True,
        overlap_means=True,
        consequent_fit="global",
        fusion_reg=0.0,
        predict_mode="blend",
        blend_recalibrate=False,
        blend_recal_l2=0.0,
        blend_sharpen=1.0,
        membership="gaussian",
        clamp_k=3.0,
        clamp_smooth=True,
        trapz_ramp=0.1,
        trapz_bins=50,
        trapz_merge=0.2,
        trapz_pad=0.0,
        ruspini_tol=0.02,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.n_output_buckets = n_output_buckets
        self.output_partition = output_partition
        self.tsk_order = tsk_order
        self.consequent_basis = consequent_basis
        self.l2_reg = l2_reg
        self.pin_extremes = pin_extremes
        self.norm_conorm = norm_conorm
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.allow_mixed_norms = allow_mixed_norms
        self.random_state = random_state
        self.max_samples = max_samples
        self.overlap = overlap
        self.overlap_shape = overlap_shape
        self.overlap_band = overlap_band
        self.overlap_antecedents = overlap_antecedents
        self.overlap_means = overlap_means
        self.consequent_fit = consequent_fit
        self.fusion_reg = fusion_reg
        self.predict_mode = predict_mode
        self.blend_recalibrate = blend_recalibrate
        self.blend_recal_l2 = blend_recal_l2
        self.blend_sharpen = blend_sharpen
        self.membership = membership
        self.clamp_k = clamp_k
        self.clamp_smooth = clamp_smooth
        self.trapz_ramp = trapz_ramp
        self.trapz_bins = trapz_bins
        self.trapz_merge = trapz_merge
        self.trapz_pad = trapz_pad
        self.ruspini_tol = ruspini_tol

    def _norms(self):
        return resolve_norm_pair(
            self.norm_conorm, self.t_norm, self.t_conorm, self.allow_mixed_norms)

    def _build_antecedents(self, X_df, y_partitioned, soft_ante):
        """The membership model, with the overlap applied to each bucket's slice.

        Every fitter here takes the same ``(X, labels, features, ...)`` shape, so
        the overlap reaches all of them the same way: the fitter is handed a
        synthetic label series marking this bucket's *widened* support instead of
        its hard one. One implementation of "which rows does this bucket see",
        across every membership shape, rather than one per shape.

        ``"trapezoid"`` uses the histogram fitter
        (`trapz_math_fast.create_trapz_membership_dict_fast`): 0.01 s against 42 s
        for the EM fitter on concrete, which is the difference between a sweep and
        an afternoon. ``"trapezoid-em"`` and ``"triangle-em"`` keep the EM path
        reachable for a spot check; they are far too slow for the arm matrix,
        especially on the overlap path, which fits once per bucket.
        """
        if self.membership in ("trapezoid", "trapezoid-em", "triangle-em"):
            if self.membership == "trapezoid":
                from tribblefis.trapz_math_fast import (
                    create_trapz_membership_dict_fast as _fit)

                def build(labels):
                    return _fit(X_df, labels, top_n_var_names=self.top_features_,
                                n_bins=self.trapz_bins,
                                ramp_width_ratio=self.trapz_ramp,
                                merge_width_ratio=self.trapz_merge)
            else:
                from tribblefis.trapz_math import create_trapz_membership_dict
                shape = ("trapezoid" if self.membership == "trapezoid-em"
                         else "triangle")

                def build(labels):
                    return create_trapz_membership_dict(
                        X_df, labels, top_n_var_names=self.top_features_,
                        n_trapezoids=self.n_gaussians, max_samples=self.max_samples,
                        random_state=self.random_state, verbose=False, shape=shape)

            if not soft_ante:
                return build(y_partitioned["y_bucket"])
            return _build_with_overlap_slices(
                build, self.overlap_weights_, X_df.index, self.top_features_)

        if soft_ante:
            return build_overlap_membership_model(
                X_df, self.overlap_weights_, self.top_features_,
                n_gaussians=self.n_gaussians, max_samples=self.max_samples,
                random_state=self.random_state, shape=self.overlap_shape)
        return create_gaussian_membership_dict(
            X_df, y_partitioned["y_bucket"], top_n_var_names=self.top_features_,
            n_gaussians=self.n_gaussians, max_samples=self.max_samples,
            random_state=self.random_state)

    def fit(self, X, y):
        if self.consequent_fit not in VALID_CONSEQUENT_FITS:
            raise ValueError(
                f"consequent_fit must be one of {VALID_CONSEQUENT_FITS}, "
                f"got {self.consequent_fit!r}")
        if self.fusion_reg < 0:
            raise ValueError(f"fusion_reg must be >= 0, got {self.fusion_reg!r}")
        if self.fusion_reg > 0 and self.consequent_fit != "global":
            warnings.warn(
                "fusion_reg is a penalty on the global stacked solve and is "
                f"ignored for consequent_fit={self.consequent_fit!r}.",
                RuntimeWarning, stacklevel=2)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        y_array = (y.values.flatten() if isinstance(y, (pd.Series, pd.DataFrame))
                   else np.asarray(y).flatten())
        X_array, y_array = check_X_y(X, y_array, multi_output=False, y_numeric=True)
        X_df = pd.DataFrame(X_array, columns=self.feature_names_in_)
        y_series = pd.Series(y_array, name="y_value")

        y_partitioned, hard_means = partition_output(
            self.n_output_buckets, y_series, method=self.output_partition)
        hard_labels = y_partitioned["y_bucket"].to_numpy()

        self.overlap_weights_ = overlap_weights(
            y_array, hard_labels, self.n_output_buckets,
            fraction=self.overlap, shape=self.overlap_shape,
            band=self.overlap_band, random_state=self.random_state)

        self.feature_differentiators_ = calculate_gaussian_correlation(
            X_df, y_partitioned["y_bucket"], top_n=self.top_n)
        self.top_n_actual_, self.top_features_ = take_top_features(
            self.feature_differentiators_, top_p=self.top_p, top_n=self.top_n)

        # Feature ranking stays on the hard partition. The overlap is a statement
        # about which rows each *rule* is fitted from, not about which columns are
        # informative, and letting it move the feature set would confound every
        # arm with a different input space.
        if self.membership not in VALID_MEMBERSHIPS:
            raise ValueError(f"membership must be one of {VALID_MEMBERSHIPS}, "
                             f"got {self.membership!r}")

        use_overlap = self.overlap > 0.0
        soft_ante = use_overlap and self.overlap_antecedents
        self.model_ = self._build_antecedents(X_df, y_partitioned, soft_ante)
        if self.membership == "clamped":
            self.model_ = clamp_model(self.model_, k=self.clamp_k,
                                      smooth=self.clamp_smooth)
        elif self.membership == "ruspini":
            self.model_ = ruspinize_features(self.model_, merge_tol=self.ruspini_tol)
        elif self.membership == "trapezoid" and self.trapz_pad > 0:
            self.model_ = pad_trapezoids(self.model_, X_df, pad=self.trapz_pad)

        self.n_rules_ = self.model_.n_rules

        # `pin_extremes=False` here on purpose, and it is not the estimator's
        # `pin_extremes`. Re-pinning the end centroids to the hard min/max would
        # make `overlap_means` a guaranteed no-op: the global solver re-derives
        # every unpinned intercept from scratch, so the *only* centroid that
        # survives into the model is a pinned one. Softening those is therefore
        # the whole content of this switch -- at the cost of some output range,
        # which is the trade the sweep is there to price.
        means_in = (overlap_bucket_means(y_array, self.overlap_weights_, hard_means,
                                         pin_extremes=False)
                    if use_overlap and self.overlap_means else hard_means)
        self.y_bucket_mean_in_ = np.asarray(means_in, dtype=float).copy()

        firing, labels = tsk_firing_strengths(
            X_df[self.top_features_], self.model_, norms=self._norms())
        firing = sharpen_firing(firing, self.blend_sharpen)
        self.rule_labels_ = list(labels)

        local_args = (firing, labels, X_df, self.top_features_, means_in,
                      y_partitioned, self.overlap_weights_)
        local_kw = dict(order=self.tsk_order, l2_reg=self.l2_reg,
                        basis=self.consequent_basis, pin_extremes=self.pin_extremes)

        if self.consequent_fit in ("local", "local-residual"):
            self.corr_terms_, self.y_bucket_mean_ = solve_consequents_local(
                *local_args, residual_form=self.consequent_fit == "local-residual",
                **local_kw)
        elif self.consequent_fit == "shrink-local":
            # The prior is the per-bucket local fit, solved first at the same
            # overlap width and ridge strength, then used as the global solve's
            # shrinkage target instead of zero.
            prior_corr, prior_means = solve_consequents_local(*local_args, **local_kw)
            self.local_prior_ = (prior_corr, prior_means)
            self.corr_terms_, self.y_bucket_mean_ = solve_consequents_shrunk(
                firing, labels, X_df, self.top_features_, means_in, y_partitioned,
                prior_corr, prior_means, order=self.tsk_order, l2_reg=self.l2_reg,
                basis=self.consequent_basis, pin_extremes=self.pin_extremes)
        elif self.fusion_reg > 0:
            self.corr_terms_, self.y_bucket_mean_ = solve_consequents_fused(
                firing, labels, X_df, self.top_features_, means_in, y_partitioned,
                order=self.tsk_order, l2_reg=self.l2_reg, basis=self.consequent_basis,
                pin_extremes=self.pin_extremes, fusion_reg=self.fusion_reg)
        else:
            # The library's own solver, called unchanged, so the baseline arm is
            # the shipped code path and not a re-derivation of it.
            self.corr_terms_, self.y_bucket_mean_ = solve_tsk_consequents_from_firing(
                firing, labels, X_df, self.top_features_, means_in, y_partitioned,
                order=self.tsk_order, l2_reg=self.l2_reg, basis=self.consequent_basis,
                pin_extremes=self.pin_extremes, verbose=False)

        # A recalibrated blend is fitted after the consequents are frozen. It adds
        # 2*n_rules parameters, so it is scored on validation/test like everything
        # else rather than on the fold that produced it.
        self.blend_a_, self.blend_b_ = None, None
        if self.blend_recalibrate:
            rule_vals = rule_consequent_values(
                X_df, self.top_features_, labels, self.y_bucket_mean_,
                self.corr_terms_, order=self.tsk_order, basis=self.consequent_basis)
            self.blend_a_, self.blend_b_ = solve_blend_recalibration(
                rule_vals, _normalize_firing_strengths(firing),
                y_partitioned["y_value"].to_numpy(dtype=float),
                l2_reg=self.blend_recal_l2)

        # Kept for the local-approximation diagnostic: which bucket each training
        # row fell in, before any overlap widened the slices.
        self.hard_labels_ = hard_labels
        self.is_fitted_ = True
        return self

    def _rule_values_and_weights(self, X_df):
        firing, labels = tsk_firing_strengths(
            X_df[self.top_features_], self.model_, norms=self._norms())
        firing = sharpen_firing(firing, self.blend_sharpen)
        rule_vals = rule_consequent_values(
            X_df, self.top_features_, labels, self.y_bucket_mean_, self.corr_terms_,
            order=self.tsk_order, basis=self.consequent_basis)
        return _normalize_firing_strengths(firing), rule_vals, labels

    def predict(self, X):
        check_is_fitted(self)
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=self.feature_names_in_)

        if self.predict_mode not in VALID_PREDICT_MODES:
            raise ValueError(f"predict_mode must be one of {VALID_PREDICT_MODES}, "
                             f"got {self.predict_mode!r}")

        if (self.predict_mode == "blend" and not self.blend_recalibrate
                and self.blend_sharpen == 1.0):
            # The library's own path, unchanged, so the default arm stays the
            # shipped prediction rather than a re-derivation of it.
            return predict_tsk(
                X_df, self.model_, self.top_features_, self.y_bucket_mean_,
                self.corr_terms_, order=self.tsk_order, basis=self.consequent_basis,
                norms=self._norms())

        norm_fs, rule_vals, _ = self._rule_values_and_weights(X_df)
        if self.predict_mode == "blend" and not self.blend_recalibrate:
            return np.sum(norm_fs * rule_vals, axis=1)
        if self.predict_mode == "wta":
            # Winner-take-all. Rows where nothing fires stay 0 -- the same
            # convention `_normalize_firing_strengths` leaves them in, and what the
            # blended path yields for them.
            winner = np.argmax(norm_fs, axis=1)
            out = rule_vals[np.arange(len(rule_vals)), winner]
            return np.where(norm_fs.sum(axis=1) > 0, out, 0.0)

        return np.sum(norm_fs * (self.blend_a_[None, :]
                                 + self.blend_b_[None, :] * rule_vals), axis=1)

    def coverage(self, X) -> dict:
        """`coverage_report` for this model on `X`. See that function."""
        check_is_fitted(self)
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=self.feature_names_in_)
        firing, _ = tsk_firing_strengths(
            X_df[self.top_features_], self.model_, norms=self._norms())
        return coverage_report(sharpen_firing(firing, self.blend_sharpen))

    def local_approximation_r2(self, X, y, hard_labels=None) -> float:
        """R2 of each row's own-bucket rule, ignoring the blend. See `local_rule_r2`.

        ``hard_labels`` defaults to the training partition, which is only correct
        for the training fold; pass the fold's own `partition_output` labels to
        score a validation or test fold.
        """
        check_is_fitted(self)
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=self.feature_names_in_)
        _, rule_vals, labels = self._rule_values_and_weights(X_df)
        hard = self.hard_labels_ if hard_labels is None else hard_labels
        return local_rule_r2(rule_vals, y, labels, hard)

    # -- diagnostics -------------------------------------------------------
    def membership_overlap_area(self) -> float:
        """Mean pairwise overlap of adjacent buckets' membership functions.

        A scalar summary of "how soft did the antecedents actually get". For each
        feature and each adjacent bucket pair, the two `LabelModel`s' maximum-
        membership envelopes are compared on a shared grid and the overlap
        coefficient ``sum(min) / sum(max)`` is taken; the report is the mean over
        all (feature, adjacent pair). Rises with `overlap` if the mechanism is
        doing what it claims to.
        """
        check_is_fitted(self)
        scores = []
        for feature_model in self.model_.feature_models.values():
            keys = feature_model.ordered_keys
            envelopes, grids = {}, []
            for k in keys:
                for m in feature_model.label_models[k].memberships:
                    grids.extend(_mf_bounds(m))
            grids = [g for g in grids if np.isfinite(g)]
            if not grids:
                continue
            grid = np.linspace(min(grids), max(grids), 512)
            for k in keys:
                mfs = feature_model.label_models[k].memberships
                # `evaluate`, not a hardcoded Gaussian: the same diagnostic has to
                # read on trapezoid, triangle and clamped models, and a Gaussian
                # formula applied to a trapezoid would silently report the wrong
                # shape's overlap.
                envelopes[k] = (np.max([m.evaluate(grid) for m in mfs], axis=0)
                                if mfs else np.zeros_like(grid))
            for a, b in zip(keys[:-1], keys[1:]):
                lo, hi = envelopes[a], envelopes[b]
                denom = np.sum(np.maximum(lo, hi))
                if denom > 0:
                    scores.append(float(np.sum(np.minimum(lo, hi)) / denom))
        return float(np.mean(scores)) if scores else float("nan")


# --------------------------------------------------------------------------
# Follow-up: is the local family's deficit the FIT or the AGGREGATION?
#
# `solve_consequents_local` makes every rule a good approximator *of y on its own
# slice*. But prediction is a firing-weighted blend of all rules, and the firing
# strengths come from x-space membership functions fitted per y-bucket -- they are
# not indicators of "sample i belongs to bucket r". So a rule can be an excellent
# local model and still be blended in where it does not apply. Nothing in the
# first sweep separated those two failure modes, because it only ever scored the
# blended prediction.
#
# The four pieces below separate them:
#   * `local_rule_r2`      -- measures the local approximation directly.
#   * `residual_form`      -- the library's literal legacy formulation, so the
#                             local arm cannot be dismissed as a straw man.
#   * predict_mode="wta"   -- drops the blend entirely; each row is answered by
#                             its strongest rule alone.
#   * blend_recalibrate    -- keeps the local rules and re-solves only the blend.
#   * consequent_fit=
#     "shrink-local"       -- keeps the exact global objective and uses the local
#                             solution as the ridge's prior instead of zero.
# --------------------------------------------------------------------------
def local_rule_r2(rule_vals: np.ndarray, y, labels: list, hard_labels) -> float:
    """R2 of each row's *responsible* rule's own output, ignoring the blend.

    For every row, take the rule whose bucket is that row's hard `y_bucket` and
    score that rule's crisp consequent output against `y`. This is the number the
    phrase "each rule consequent function locally approximates better" refers to,
    and it is not what test R2 measures: test R2 scores
    ``sum_r w_r q_r(x)``, which can be worse than every ``q_r`` is on its own
    bucket if the weights ``w_r`` put mass on the wrong rules.

    Returns NaN when no row can be attributed (no overlap between `labels` and
    the hard bucket labels present).
    """
    y = np.asarray(y, dtype=float).ravel()
    hard = np.asarray(hard_labels).ravel()
    col_of = {int(lab): j for j, lab in enumerate(labels)}
    own = np.array([col_of.get(int(b), -1) for b in hard])
    keep = own >= 0
    if keep.sum() < 2:
        return float("nan")
    pred = rule_vals[np.flatnonzero(keep), own[keep]]
    yt = y[keep]
    denom = float(np.sum((yt - yt.mean()) ** 2))
    if denom == 0 or not np.all(np.isfinite(pred)):
        return float("nan")
    return float(1.0 - np.sum((yt - pred) ** 2) / denom)


def solve_blend_recalibration(
    rule_vals: np.ndarray,
    norm_fs: np.ndarray,
    y,
    l2_reg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-rule affine recalibration of a frozen set of rule outputs.

    Fits ``y_hat = sum_r w_r * (a_r + b_r * q_r(x))`` for fixed ``q_r`` (the local
    per-bucket consequents) and fixed ``w_r`` (the firing strengths), which is
    ``2 * n_rules`` free parameters and linear in all of them. The ridge pulls
    ``(a_r, b_r)`` toward ``(0, 1)`` -- the identity -- so ``l2_reg -> inf``
    recovers the plain blend exactly and any departure from it has to be paid for.

    This is the cheapest possible fix for an aggregation mismatch: it cannot
    change what any rule computes, only how much of it is used and where its zero
    sits. If it recovers most of the local family's deficit, the deficit was the
    blend. If it recovers little, the local fit itself was the problem.
    """
    y = np.asarray(y, dtype=float).ravel()
    n_rules = norm_fs.shape[1]
    design = np.empty((len(y), 2 * n_rules))
    design[:, 0::2] = norm_fs
    design[:, 1::2] = norm_fs * rule_vals

    prior = np.tile([0.0, 1.0], n_rules)
    if l2_reg > 0:
        root = np.sqrt(l2_reg) * np.ones(2 * n_rules)
        A = np.vstack([design, np.diag(root)])
        rhs = np.hstack([y, root * prior])
    else:
        A, rhs = design, y
    beta = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return beta[0::2].copy(), beta[1::2].copy()


def solve_consequents_shrunk(
    firing_strengths: np.ndarray,
    labels: list,
    X_train: pd.DataFrame,
    top_n_todo: list,
    y_bucket_mean,
    y_train: pd.DataFrame,
    prior_corr: np.ndarray,
    prior_means: np.ndarray,
    order: str = "2nd",
    l2_reg: float = 1e-6,
    basis: str = "raw",
    cross_pairs=None,
    pin_extremes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """The exact global firing-weighted ridge, shrunk toward the local solution.

    Identical to `solve_tsk_consequents_from_firing` except that the ridge's
    prior is the per-bucket local fit rather than zero: it minimizes
    ``||y - Phi beta||^2 + l2_reg * ||D (beta - beta_local)||^2``.

    This is the one way per-bucket local information can enter without giving up
    anything. The global solve stays the exact minimizer of the firing-weighted
    objective the model is scored on; the only thing that changes is *where the
    regularizer pulls when the data does not pin a coefficient down*. Shrinking
    toward zero says "in the absence of evidence, correct nothing"; shrinking
    toward the local fit says "in the absence of evidence, do what this rule's own
    bucket says". At ``l2_reg=0`` the two are identical, so the arm has the
    baseline as a limit rather than as a rival.

    Intercepts are unpenalized, as everywhere else, so ``prior_means`` is unused
    except for shape agreement -- kept in the signature because dropping it would
    make the asymmetry with `prior_corr` invisible at the call site.
    """
    from tribblefis.regression import _normalize_firing_strengths

    norm_fs = _normalize_firing_strengths(firing_strengths)
    n_rules = norm_fs.shape[1]
    X_rule = X_train[top_n_todo].to_numpy()
    feats = build_consequent_features(X_rule, order, basis=basis, cross_pairs=cross_pairs)
    n_terms = feats.shape[1]
    per_rule = 1 + n_terms

    phi = np.hstack([np.ones((X_rule.shape[0], 1)), feats])
    design = (norm_fs[:, :, np.newaxis] * phi[:, np.newaxis, :]).reshape(
        X_rule.shape[0], n_rules * per_rule)
    y = np.asarray(y_train["y_value"].values, dtype=float)

    penalty = np.ones(n_rules * per_rule)
    penalty[::per_rule] = 0.0
    # The prior vector, laid out to match `design`'s columns: 0 on every
    # (unpenalized) intercept column, the local fit on every correction column.
    prior = np.zeros(n_rules * per_rule)
    if n_terms > 0:
        prior.reshape(n_rules, per_rule)[:, 1:] = np.asarray(prior_corr, dtype=float)

    pinned_cols: list[int] = []
    pinned_vals: list[float] = []
    if pin_extremes and n_rules >= 2 and y_bucket_mean is not None:
        ybm = np.asarray(y_bucket_mean, dtype=float).ravel()
        if ybm.size > int(np.max(labels)):
            for rule_idx in (0, n_rules - 1):
                value = float(ybm[int(labels[rule_idx])])
                if np.isfinite(value):
                    pinned_cols.append(rule_idx * per_rule)
                    pinned_vals.append(value)

    def _solve(A, rhs, pen, pri):
        if l2_reg > 0:
            root = np.sqrt(l2_reg * pen)
            A = np.vstack([A, np.diag(root)])
            rhs = np.hstack([rhs, root * pri])
        return np.linalg.lstsq(A, rhs, rcond=None)[0]

    if pinned_cols:
        pinned = np.asarray(pinned_cols, dtype=int)
        values = np.asarray(pinned_vals, dtype=float)
        free = np.setdiff1d(np.arange(design.shape[1]), pinned)
        beta = np.zeros(design.shape[1])
        beta[pinned] = values
        beta[free] = _solve(design[:, free], y - design[:, pinned] @ values,
                            penalty[free], prior[free])
    else:
        beta = _solve(design, y, penalty, prior)

    coeffs = beta.reshape(n_rules, per_rule)
    corr = coeffs[:, 1:].copy() if n_terms > 0 else np.zeros((n_rules, 0))
    return corr, coeffs[:, 0].copy()


# --------------------------------------------------------------------------
# Stage 3: compact support
#
# Stage 2's finding was that per-bucket consequent solving makes every rule a
# much better approximator of its own region while making the blended model
# worse -- the blend mixes each rule in where it is not competent. Gaussian
# membership functions are why it can: they are strictly positive everywhere, so
# *every* rule fires, however faintly, at *every* point of the input space. A
# local model blended with a nonzero weight a long way from its own data is a
# local model being asked a question it was never fitted to answer.
#
# Compact support removes that by construction. Three ways to get it:
#
#   trapezoid / triangle  the library already fits these
#                         (`trapz_math.create_trapz_membership_dict`), and they
#                         are exactly zero outside [a, d] / [a, c].
#   clamped Gaussian      keep the Gaussian fit and zero it past k sigma. The
#                         library already uses `mu +/- 3 sigma` as a Gaussian's
#                         "effective support" in `gauss_data.mf_interval`, so
#                         this makes an existing convention literal.
#
# The cost is the mirror image of the benefit, and it has to be measured rather
# than assumed: compact support can leave points with NO rule covering them.
# `_normalize_firing_strengths` returns an all-zero row there and the model
# predicts exactly 0 -- not NaN, so it does not show up as a dropped row, it
# shows up as a quietly terrible prediction. `coverage_report` measures it.
# --------------------------------------------------------------------------
VALID_MEMBERSHIPS = ("gaussian", "clamped", "trapezoid", "trapezoid-em",
                     "triangle-em", "ruspini")


class ClampedGaussianMembership(typing.NamedTuple):
    """A Gaussian forced to exactly zero beyond ``k`` standard deviations.

    Deliberately **not** a subclass of `GaussianMembership`, and deliberately a
    distinct type: `kernel.compile_model` admits a model to its compiled fast path
    on ``isinstance(mf, GaussianMembership)``, and that path evaluates a plain
    Gaussian. A subclass would pass that check and the clamp would be silently
    dropped wherever the Cython extension is built -- which is not this
    environment, so the bug would not appear here and would appear in production.
    A separate type raises `NotCompilable` and falls back to the polymorphic
    Python loop, which calls `evaluate` and honours the clamp. Pinned as a test.

    ``smooth=False`` truncates: the membership steps from ``exp(-k^2/2)`` to 0 at
    the cutoff (0.011 at k=3, 0.023 at k=2.75 -- small, but a discontinuity).
    ``smooth=True`` subtracts that boundary value and rescales, so membership
    *reaches* zero continuously at exactly ``k`` sigma and the peak is still 1.
    The smooth form is the "non-linear clamp": it is the same Gaussian shape with
    its tail pulled down to meet the axis, not a Gaussian with a hole punched in
    it.
    """

    mu: float
    sigma: float
    k: float = 3.0
    smooth: bool = True
    id: typing.Optional[uuid.UUID] = None

    @staticmethod
    def create(mu: float, sigma: float, k: float = 3.0,
               smooth: bool = True) -> "ClampedGaussianMembership":
        return ClampedGaussianMembership(mu=mu, sigma=sigma, k=k, smooth=smooth,
                                         id=uuid.uuid4())

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        sigma = max(self.sigma, 1e-6)
        z = (x - self.mu) / sigma
        inside = np.abs(z) <= self.k
        bell = np.exp(-0.5 * z ** 2)
        if not self.smooth:
            return np.where(inside, bell, 0.0)
        floor = float(np.exp(-0.5 * self.k ** 2))
        # (bell - floor) / (1 - floor): 1 at the peak, 0 at the cutoff. The
        # `inside` mask is still applied so floating point cannot leave a
        # negative sliver just outside k sigma.
        return np.where(inside, np.maximum(0.0, (bell - floor) / (1.0 - floor)), 0.0)


def clamp_model(model: GaussianMixtureModel, k: float = 3.0,
                smooth: bool = True) -> GaussianMixtureModel:
    """Rebuild `model` with every Gaussian replaced by a clamped one.

    Non-Gaussian memberships pass through untouched, so this is a no-op on a
    trapezoid or triangle model rather than an error -- the arm matrix in
    `run_support.py` would otherwise need a special case per membership type.
    """
    feature_models = {}
    for name, fmodel in model.feature_models.items():
        label_models = {}
        for label, lmodel in fmodel.label_models.items():
            label_models[label] = LabelModel(memberships=[
                ClampedGaussianMembership.create(mf.mu, mf.sigma, k=k, smooth=smooth)
                if isinstance(mf, GaussianMembership) else mf
                for mf in lmodel.memberships
            ])
        feature_models[name] = FeatureModel(label_models=label_models)
    return GaussianMixtureModel(feature_models=feature_models)


def ruspinize_features(model: GaussianMixtureModel, merge_tol: float = 0.02
                       ) -> GaussianMixtureModel:
    """Re-express each feature as a shared Ruspini partition of triangular terms.

    Compact support with *guaranteed* coverage, which is the combination the
    trapezoid fitters do not deliver. Per feature: collect every bucket's fitted
    membership centres as apex landmarks, merge near-duplicates
    (`ruspini._merge_close`), build the shared triangular partition
    (`ruspini.build_triangular_partition` -- terms sum to exactly 1 at every point
    of the axis, with the first and last shouldered to +/-inf), then give each
    bucket the term whose apex is nearest each of its own centres.

    Two properties matter and they are the reason this arm exists:

    * every interior term is compactly supported, so a rule goes silent away from
      its own region -- what a per-bucket local consequent wants;
    * the terms tile the axis by construction, so no point is left uncovered --
      what the trapezoid arms fail at, in 1-D before any dimensionality effect.

    This reuses the library's own construction and its "nearest apex" matching
    heuristic (`ruspini.ruspinize_model` documents it) but emits a
    `GaussianMixtureModel` rather than the explicit rule layout, so the rest of
    this pipeline -- firing strengths, consequent solve, prediction -- is unchanged
    and the arm is comparable to the others.

    ``merge_tol`` is in the feature's own units; inputs here are unit-scaled to
    [0, 1], so 0.02 merges centres within 2% of the range.
    """
    from tribblefis.ruspini import _merge_close, build_triangular_partition

    feature_models = {}
    for name, fmodel in model.feature_models.items():
        centres = [mf.mu for lmodel in fmodel.label_models.values()
                   for mf in lmodel.memberships if hasattr(mf, "mu")]
        if not centres:
            feature_models[name] = fmodel
            continue
        apexes = _merge_close(centres, merge_tol)
        terms = build_triangular_partition(apexes)
        apex_arr = np.asarray(apexes, dtype=float)

        label_models = {}
        for label, lmodel in fmodel.label_models.items():
            picked, seen = [], set()
            for mf in lmodel.memberships:
                if not hasattr(mf, "mu"):
                    continue
                j = int(np.argmin(np.abs(apex_arr - mf.mu)))
                if j not in seen:
                    seen.add(j)
                    picked.append(terms[j])
            label_models[label] = LabelModel(memberships=picked)
        feature_models[name] = FeatureModel(label_models=label_models)
    return GaussianMixtureModel(feature_models=feature_models)


def pad_trapezoids(model: GaussianMixtureModel, X: pd.DataFrame,
                   pad: float = 0.25) -> GaussianMixtureModel:
    """Re-seat each fitted trapezoid so its observed data range is the *plateau*.

    Fixes a defect in the histogram fitter that makes compactly supported
    antecedents unusable on any feature with a mass point at its minimum.

    `TrapezoidMembership.evaluate` rises with a strict inequality (``x > a``), so
    membership is exactly **0 at x == a**, which is correct for an open trapezoid.
    But `trapz_math_fast.fit_trapezoids_fast` sets ``a = bin_edges[0]``, i.e. the
    minimum of the data it was fitted to -- so the smallest observed value, and
    every value tied with it, receives zero membership from the very term fitted
    to describe it. On concrete's scaled features 55% of rows sit exactly at
    FlyAsh's minimum, 44% at Slag's and 36% at Superplasticizer's; under the
    ``min`` t-norm one dead feature zeroes the whole rule, and 77% of test rows
    end up covered by no rule at all.

    The library already handles the identical hazard one module over:
    `regression.partition_output` nudges ``edges[0] -= 1e-9`` so "the smallest
    value lands in bucket 0 rather than becoming NaN -- `include_lowest` alone is
    not enough once the edges are supplied explicitly". This is that nudge, sized
    to matter rather than to break a tie: the fitted ``[a, d]`` becomes the plateau
    ``[b, c]``, and the support extends ``pad`` times the region width beyond it on
    each side. So every point the term was fitted to gets membership 1, points
    just outside ramp down, and the term is still compactly supported -- which is
    the whole reason for using trapezoids here.

    ``pad`` is relative to each region's own width, with a floor taken from the
    feature's observed range so that a degenerate region (a single repeated value,
    ``a == d``) still gets a usable support instead of a delta function.
    """
    padded = {}
    for name, fmodel in model.feature_models.items():
        col = X[name].to_numpy(dtype=float) if name in X else None
        span = float(np.nanmax(col) - np.nanmin(col)) if col is not None and len(col) else 1.0
        floor = pad * (span if span > 0 else 1.0)
        label_models = {}
        for label, lmodel in fmodel.label_models.items():
            out = []
            for mf in lmodel.memberships:
                if not isinstance(mf, TrapezoidMembership):
                    out.append(mf)
                    continue
                width = float(mf.d - mf.a)
                margin = max(pad * width, floor if width <= 0 else 0.0)
                if margin <= 0:
                    margin = floor
                out.append(TrapezoidMembership.create(
                    a=float(mf.a) - margin, b=float(mf.a),
                    c=float(mf.d), d=float(mf.d) + margin))
            label_models[label] = LabelModel(memberships=out)
        padded[name] = FeatureModel(label_models=label_models)
    return GaussianMixtureModel(feature_models=padded)


def coverage_report(firing_strengths: np.ndarray, threshold: float = 1e-6) -> dict:
    """How much of the input space each rule set actually covers.

    The quantity compact support trades against accuracy, and the one that does
    not announce itself: a row no rule covers gets an all-zero normalized firing
    row and a prediction of exactly 0. That is a finite number, so it survives
    every NaN filter in the pipeline and lands in the R2 as a large error with no
    diagnostic attached.

    Returns
    -------
    uncovered : fraction of rows where total firing is at or below the floor
        `_normalize_firing_strengths` uses, i.e. rows the model answers with 0.
    mean_active : mean number of rules firing above `threshold` per row -- 1.0
        would mean a genuine partition, `n_rules` means every rule everywhere
        (which is what an unclamped Gaussian model gives).
    active_frac : `mean_active` as a fraction of the rule count, so it is
        comparable across bucket counts.
    """
    active = (firing_strengths > threshold).sum(axis=1)
    n_rules = firing_strengths.shape[1]
    return dict(
        uncovered=float(np.mean(firing_strengths.sum(axis=1) <= 1e-6)),
        mean_active=float(np.mean(active)),
        active_frac=float(np.mean(active) / n_rules) if n_rules else float("nan"),
    )
