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

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from tribblefis.gauss_data import (
    FeatureModel,
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
    build_consequent_features,
    partition_output,
    predict_tsk,
    solve_tsk_consequents_from_firing,
)

VALID_SHAPES = ("flat", "ramp")
VALID_CONSEQUENT_FITS = ("global", "local")

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
) -> tuple[np.ndarray, np.ndarray]:
    """Fit each rule's consequent on its own (optionally overlapping) slice.

    Rule ``r`` solves ``min sum_i W[i, b_r] * (y_i - (mean_r + phi_i . c_r))^2``
    plus the same ridge penalty the global solver uses (intercept unpenalized).
    With ``W`` the hard indicator this is the classical per-bucket local TSK fit
    -- a ridge-regularized `compute_*_order_corrections` -- and with an overlap
    band it is that fit with neighbouring data blended in.

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
        pin = pin_extremes and n_rules >= 2 and r in (0, n_rules - 1) \
            and ybm.size > bucket and np.isfinite(ybm[bucket])

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
    consequent_fit : {"global", "local"}, default "global"
        ``"global"`` is the library's exact firing-weighted stacked ridge solve;
        ``"local"`` fits each rule's polynomial on its own overlapped slice.
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

    def _norms(self):
        return resolve_norm_pair(
            self.norm_conorm, self.t_norm, self.t_conorm, self.allow_mixed_norms)

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
        use_overlap = self.overlap > 0.0
        if use_overlap and self.overlap_antecedents:
            self.model_ = build_overlap_membership_model(
                X_df, self.overlap_weights_, self.top_features_,
                n_gaussians=self.n_gaussians, max_samples=self.max_samples,
                random_state=self.random_state, shape=self.overlap_shape)
        else:
            self.model_ = create_gaussian_membership_dict(
                X_df, y_partitioned["y_bucket"], top_n_var_names=self.top_features_,
                n_gaussians=self.n_gaussians, max_samples=self.max_samples,
                random_state=self.random_state)

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
        self.rule_labels_ = list(labels)

        if self.consequent_fit == "local":
            self.corr_terms_, self.y_bucket_mean_ = solve_consequents_local(
                firing, labels, X_df, self.top_features_, means_in, y_partitioned,
                self.overlap_weights_, order=self.tsk_order, l2_reg=self.l2_reg,
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

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=self.feature_names_in_)
        return predict_tsk(
            X_df, self.model_, self.top_features_, self.y_bucket_mean_,
            self.corr_terms_, order=self.tsk_order, basis=self.consequent_basis,
            norms=self._norms())

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
                mfs = feature_model.label_models[k].memberships
                if mfs:
                    grids.extend([m.mu - 4 * m.sigma for m in mfs])
                    grids.extend([m.mu + 4 * m.sigma for m in mfs])
            if not grids:
                continue
            grid = np.linspace(min(grids), max(grids), 512)
            for k in keys:
                mfs = feature_model.label_models[k].memberships
                envelopes[k] = (
                    np.max([np.exp(-0.5 * ((grid - m.mu) / max(m.sigma, 1e-12)) ** 2)
                            for m in mfs], axis=0)
                    if mfs else np.zeros_like(grid))
            for a, b in zip(keys[:-1], keys[1:]):
                lo, hi = envelopes[a], envelopes[b]
                denom = np.sum(np.maximum(lo, hi))
                if denom > 0:
                    scores.append(float(np.sum(np.minimum(lo, hi)) / denom))
        return float(np.mean(scores)) if scores else float("nan")
