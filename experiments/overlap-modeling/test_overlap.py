"""Property and degeneracy tests for overlap modeling.

The load-bearing test is `test_defaults_reproduce_the_library_regressor`: every
number this experiment reports is a difference against `TribbleRegressor`, and
that difference only means something if the zero-overlap arm *is* the baseline
rather than a near-miss re-implementation of it.

Run: python -m pytest experiments/overlap-modeling/test_overlap.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from overlap import (  # noqa: E402
    OverlapTribbleRegressor,
    build_overlap_membership_model,
    overlap_bucket_means,
    overlap_weights,
    solve_consequents_fused,
)
from tribblefis.gauss_math import (  # noqa: E402
    create_gaussian_membership_dict,
    tsk_firing_strengths,
)
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402
from tribblefis.regression import (  # noqa: E402
    partition_output,
    solve_tsk_consequents_from_firing,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({
        "a": rng.uniform(0, 1, n),
        "b": rng.uniform(0, 1, n),
        "c": rng.uniform(0, 1, n),
    })
    y = pd.Series(
        2.0 * X["a"] + 0.5 * X["b"] ** 2 - X["a"] * X["c"] + 0.05 * rng.normal(size=n),
        name="y_value",
    )
    return X, y


# --------------------------------------------------------------------------
# overlap_weights
# --------------------------------------------------------------------------
def test_zero_overlap_is_the_hard_indicator():
    y = np.arange(20, dtype=float)
    labels = np.repeat([0, 1, 2, 3], 5)
    W = overlap_weights(y, labels, 4, fraction=0.0)
    assert W.shape == (20, 4)
    np.testing.assert_array_equal(W.sum(axis=1), np.ones(20))
    for b in range(4):
        assert np.array_equal(np.flatnonzero(W[:, b]), np.flatnonzero(labels == b))


def test_band_takes_the_requested_fraction_of_the_neighbour():
    y = np.arange(30, dtype=float)
    labels = np.repeat([0, 1, 2], 10)
    W = overlap_weights(y, labels, 3, fraction=0.4)
    # Bucket 1 owns 10..19 and borrows 4 from each side: 6..9 and 20..23.
    assert set(np.flatnonzero(W[:, 1])) == set(range(6, 24))
    # Bucket 0 is an endpoint: it borrows upward only, never below its own floor.
    assert set(np.flatnonzero(W[:, 0])) == set(range(0, 14))
    assert set(np.flatnonzero(W[:, 2])) == set(range(16, 30))


def test_endpoints_are_single_sided_at_every_width():
    y = np.arange(30, dtype=float)
    labels = np.repeat([0, 1, 2], 10)
    for fraction in (0.1, 0.5, 1.0):
        W = overlap_weights(y, labels, 3, fraction=fraction)
        assert W[:, 0].nonzero()[0].min() == 0, "bucket 0 gained a point below the minimum"
        assert W[:, -1].nonzero()[0].max() == 29, "last bucket gained a point above the maximum"


def test_flat_band_weights_are_one_and_ramp_weights_decay_outward():
    y = np.arange(30, dtype=float)
    labels = np.repeat([0, 1, 2], 10)
    flat = overlap_weights(y, labels, 3, fraction=0.5, shape="flat")
    assert set(np.unique(flat[flat > 0])) == {1.0}

    ramp = overlap_weights(y, labels, 3, fraction=0.5, shape="ramp")
    np.testing.assert_allclose(ramp[10:20, 1], 1.0)          # own points keep full weight
    upper = ramp[20:25, 1]                                    # borrowed from above
    assert np.all(np.diff(upper) < 0), "ramp must decay away from the shared edge"
    assert upper[0] == pytest.approx(1.0)
    lower = ramp[5:10, 1]                                     # borrowed from below
    assert np.all(np.diff(lower) > 0), "ramp must rise toward the shared edge"


def test_overlap_is_monotone_in_the_fraction():
    y = np.arange(60, dtype=float)
    labels = np.repeat([0, 1, 2], 20)
    prev = overlap_weights(y, labels, 3, fraction=0.0)
    for fraction in (0.1, 0.25, 0.5):
        cur = overlap_weights(y, labels, 3, fraction=fraction, shape="flat")
        assert (cur > 0).sum() > (prev > 0).sum()
        assert np.all((cur > 0) >= (prev > 0)), "a wider band must not drop a row"
        prev = cur


def test_a_shared_point_counts_fully_in_both_rules():
    """Rows are fitting weights, not a probability -- they must not be normalized."""
    y = np.arange(20, dtype=float)
    labels = np.repeat([0, 1], 10)
    W = overlap_weights(y, labels, 2, fraction=0.5, shape="flat")
    shared = np.flatnonzero((W[:, 0] > 0) & (W[:, 1] > 0))
    assert len(shared) == 10
    np.testing.assert_allclose(W[shared].sum(axis=1), 2.0)


def test_fraction_is_validated():
    y, labels = np.arange(10, dtype=float), np.repeat([0, 1], 5)
    with pytest.raises(ValueError):
        overlap_weights(y, labels, 2, fraction=1.5)
    with pytest.raises(ValueError):
        overlap_weights(y, labels, 2, fraction=0.2, shape="gaussian")


def test_weighted_means_move_toward_the_neighbours():
    y = np.arange(30, dtype=float)
    labels = np.repeat([0, 1, 2], 10)
    hard = np.array([4.5, 14.5, 24.5])
    W = overlap_weights(y, labels, 3, fraction=0.5, shape="flat")
    soft = overlap_bucket_means(y, W, hard, pin_extremes=False)
    assert soft[1] == pytest.approx(14.5)          # symmetric overlap: unchanged
    assert soft[0] > hard[0]                       # pulled up toward bucket 1
    assert soft[2] < hard[2]                       # pulled down toward bucket 1
    pinned = overlap_bucket_means(y, W, hard, pin_extremes=True)
    assert pinned[0] == hard[0] and pinned[2] == hard[2]


# --------------------------------------------------------------------------
# Degeneracy against the library
# --------------------------------------------------------------------------
@pytest.mark.parametrize("order", ["1st", "2nd", "full-2nd"])
@pytest.mark.parametrize("partition", ["uniform", "quantile"])
def test_defaults_reproduce_the_library_regressor(data, order, partition):
    X, y = data
    kwargs = dict(n_output_buckets=3, output_partition=partition, tsk_order=order,
                  l2_reg=1e-2, pin_extremes=True, random_state=7)
    base = TribbleRegressor(**kwargs).fit(X, y)
    ours = OverlapTribbleRegressor(overlap=0.0, **kwargs).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), base.predict(X), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ours.y_bucket_mean_, base.y_bucket_mean_,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ours.corr_terms_, base.corr_terms_,
                               rtol=1e-8, atol=1e-8)


def test_zero_overlap_membership_model_matches_the_library_fit(data):
    X, y = data
    y_part, _ = partition_output(4, y, method="quantile")
    labels = y_part["y_bucket"].to_numpy()
    W = overlap_weights(y.to_numpy(), labels, 4, fraction=0.0)
    features = ["a", "b", "c"]
    ours = build_overlap_membership_model(X, W, features, n_gaussians=0, random_state=3)
    base = create_gaussian_membership_dict(X, y_part["y_bucket"], features,
                                           n_gaussians=0, random_state=3)
    # Sorted, not as-listed: `rule_ids` reports dict insertion order, and the
    # library keys its labels off `y.unique()` (order of first appearance) while
    # this builder inserts them in bucket order. Nothing downstream reads that
    # order -- `FeatureModel.ordered_keys`, which `tsk_firing_strengths` uses,
    # sorts -- so the sets are what must agree.
    assert sorted(ours.rule_ids) == sorted(base.rule_ids)
    for name in features:
        for label in base.feature_models[name].ordered_keys:
            a = ours.feature_models[name].label_models[label].memberships
            b = base.feature_models[name].label_models[label].memberships
            assert len(a) == len(b)
            for ma, mb in zip(a, b):
                assert ma.mu == pytest.approx(mb.mu)
                assert ma.sigma == pytest.approx(mb.sigma)


@pytest.mark.parametrize("pin", [False, True])
def test_fused_solver_reduces_to_the_library_solver_at_zero(data, pin):
    X, y = data
    features = ["a", "b", "c"]
    y_part, means = partition_output(3, y, method="quantile")
    model = create_gaussian_membership_dict(X, y_part["y_bucket"], features,
                                            n_gaussians=0, random_state=3)
    firing, labels = tsk_firing_strengths(X[features], model)
    args = (firing, labels, X, features, means, y_part)
    kw = dict(order="2nd", l2_reg=1e-3, basis="raw", pin_extremes=pin)
    ours = solve_consequents_fused(*args, fusion_reg=0.0, **kw)
    base = solve_tsk_consequents_from_firing(*args, verbose=False, **kw)
    np.testing.assert_allclose(ours[0], base[0], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(ours[1], base[1], rtol=1e-8, atol=1e-8)


def test_fusion_shrinks_the_gap_between_adjacent_consequents(data):
    X, y = data
    features = ["a", "b", "c"]
    y_part, means = partition_output(4, y, method="quantile")
    model = create_gaussian_membership_dict(X, y_part["y_bucket"], features,
                                            n_gaussians=0, random_state=3)
    firing, labels = tsk_firing_strengths(X[features], model)

    def spread(fusion_reg):
        corr, _ = solve_consequents_fused(
            firing, labels, X, features, means, y_part, order="2nd", l2_reg=1e-3,
            fusion_reg=fusion_reg)
        return float(np.abs(np.diff(corr, axis=0)).mean())

    gaps = [spread(f) for f in (0.0, 1.0, 100.0)]
    assert gaps[0] > gaps[1] > gaps[2], f"fusion did not pull neighbours together: {gaps}"


# --------------------------------------------------------------------------
# The mechanism does what it says
# --------------------------------------------------------------------------
def test_overlap_widens_the_membership_functions_at_a_fixed_component_count(data):
    """One membership function per (feature, bucket), so width is the only free thing.

    The component count has to be pinned for this to be a statement about width.
    Left to BIC the wider slice often supports *more* components -- 18 -> 22 on
    this fixture at ``overlap=0.4`` -- and the mean sigma then goes *down* while
    the functions still cover more of the axis between them. That is why the
    mechanism check the sweep relies on is the adjacent-overlap area below, not
    a sigma average.
    """
    X, y = data

    def mean_sigma(fraction):
        m = OverlapTribbleRegressor(
            n_output_buckets=4, output_partition="quantile", tsk_order="1st",
            n_gaussians=1, overlap=fraction, overlap_shape="flat",
            random_state=7).fit(X, y)
        return float(np.mean([g.sigma for g in m.model_.all_membership_fcns]))
    assert mean_sigma(0.4) > mean_sigma(0.0)


def test_bic_may_spend_a_wider_slice_on_more_components_instead_of_wider_ones(data):
    """Documents the confound above, so a later reader does not rediscover it."""
    X, y = data

    def counts(fraction):
        m = OverlapTribbleRegressor(
            n_output_buckets=4, output_partition="quantile", tsk_order="1st",
            n_gaussians=0, overlap=fraction, random_state=7).fit(X, y)
        return len(m.model_.all_membership_fcns)
    assert counts(0.4) > counts(0.0)


def test_overlap_raises_adjacent_membership_overlap(data):
    X, y = data

    def area(fraction):
        return OverlapTribbleRegressor(
            n_output_buckets=4, output_partition="quantile", tsk_order="1st",
            overlap=fraction, random_state=7).fit(X, y).membership_overlap_area()
    assert area(0.4) > area(0.0)


def test_only_the_antecedent_switch_moves_the_firing_strengths(data):
    """Firing strengths come from the membership functions and nothing else."""
    X, y = data
    kwargs = dict(n_output_buckets=4, output_partition="quantile", tsk_order="1st",
                  random_state=7)
    hard = OverlapTribbleRegressor(overlap=0.0, **kwargs).fit(X, y)
    for extra in (dict(overlap_antecedents=False),
                  dict(overlap_antecedents=False, consequent_fit="local")):
        soft = OverlapTribbleRegressor(overlap=0.3, **extra, **kwargs).fit(X, y)
        f_hard, _ = tsk_firing_strengths(X[hard.top_features_], hard.model_)
        f_soft, _ = tsk_firing_strengths(X[soft.top_features_], soft.model_)
        np.testing.assert_allclose(f_soft, f_hard, rtol=1e-12, atol=1e-12)


def test_overlap_means_is_inert_under_the_global_solve_without_pinning(data):
    """A finding, pinned as a test: the global solver discards unpinned centroids.

    `solve_tsk_consequents_from_firing` re-derives every rule's intercept as part
    of the exact firing-weighted optimum, so the centroid handed to it only
    survives where it is *pinned*. With ``pin_extremes=False`` there is no such
    column and ``overlap_means`` cannot reach the model at all -- it is not a
    weak effect, it is arithmetically zero. It becomes visible with pinning (the
    two end rungs), and under ``consequent_fit="local"`` through the weighted
    slices regardless.
    """
    X, y = data
    base = dict(n_output_buckets=4, output_partition="quantile", tsk_order="1st",
                random_state=7)
    soft = dict(overlap=0.3, overlap_antecedents=False)

    inert = dict(consequent_fit="global", pin_extremes=False, **base)
    np.testing.assert_allclose(
        OverlapTribbleRegressor(overlap=0.0, **inert).fit(X, y).predict(X),
        OverlapTribbleRegressor(**soft, **inert).fit(X, y).predict(X),
        rtol=1e-12, atol=1e-12)

    for live in (dict(consequent_fit="global", pin_extremes=True, **base),
                 dict(consequent_fit="local", pin_extremes=False, **base)):
        assert not np.allclose(
            OverlapTribbleRegressor(overlap=0.0, **live).fit(X, y).predict(X),
            OverlapTribbleRegressor(**soft, **live).fit(X, y).predict(X))


@pytest.mark.parametrize("shape", ["flat", "ramp"])
@pytest.mark.parametrize("fit", ["global", "local"])
def test_every_arm_fits_and_predicts_finitely(data, shape, fit):
    X, y = data
    m = OverlapTribbleRegressor(
        n_output_buckets=5, output_partition="quantile", tsk_order="full-2nd",
        l2_reg=1e-2, overlap=0.25, overlap_shape=shape, consequent_fit=fit,
        pin_extremes=True, random_state=7).fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),)
    assert np.all(np.isfinite(p))


def test_local_fit_at_zero_overlap_is_the_hard_per_bucket_fit(data):
    """The control that separates 'local instead of global' from 'soft instead of hard'."""
    X, y = data
    kwargs = dict(n_output_buckets=4, output_partition="quantile", tsk_order="2nd",
                  l2_reg=1e-3, random_state=7)
    local_hard = OverlapTribbleRegressor(consequent_fit="local", overlap=0.0,
                                         **kwargs).fit(X, y)
    local_soft = OverlapTribbleRegressor(consequent_fit="local", overlap=0.3,
                                         overlap_antecedents=False, overlap_means=False,
                                         **kwargs).fit(X, y)
    # Same antecedents and same centroid inputs; only the consequent slices differ.
    f_a, _ = tsk_firing_strengths(X[local_hard.top_features_], local_hard.model_)
    f_b, _ = tsk_firing_strengths(X[local_soft.top_features_], local_soft.model_)
    np.testing.assert_allclose(f_a, f_b, rtol=1e-12, atol=1e-12)
    assert not np.allclose(local_hard.corr_terms_, local_soft.corr_terms_)


def test_ramp_and_flat_differ_only_through_the_weights(data):
    X, y = data
    kwargs = dict(n_output_buckets=4, output_partition="quantile", tsk_order="2nd",
                  overlap=0.3, consequent_fit="local", random_state=7)
    flat = OverlapTribbleRegressor(overlap_shape="flat", **kwargs).fit(X, y)
    ramp = OverlapTribbleRegressor(overlap_shape="ramp", **kwargs).fit(X, y)
    W_flat, W_ramp = flat.overlap_weights_, ramp.overlap_weights_
    np.testing.assert_array_equal(W_flat > 0, W_ramp > 0)   # same support
    assert np.all(W_ramp <= W_flat + 1e-12)                 # ramp only ever lighter
    assert not np.allclose(flat.predict(X), ramp.predict(X))
