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
from tribblefis.gauss_data import (  # noqa: E402
    FeatureModel,
    GaussianMixtureModel,
    LabelModel,
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
    X = pd.DataFrame(
        {
            "a": rng.uniform(0, 1, n),
            "b": rng.uniform(0, 1, n),
            "c": rng.uniform(0, 1, n),
        }
    )
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
        assert (
            W[:, 0].nonzero()[0].min() == 0
        ), "bucket 0 gained a point below the minimum"
        assert (
            W[:, -1].nonzero()[0].max() == 29
        ), "last bucket gained a point above the maximum"


def test_flat_band_weights_are_one_and_ramp_weights_decay_outward():
    y = np.arange(30, dtype=float)
    labels = np.repeat([0, 1, 2], 10)
    flat = overlap_weights(y, labels, 3, fraction=0.5, shape="flat")
    assert set(np.unique(flat[flat > 0])) == {1.0}

    ramp = overlap_weights(y, labels, 3, fraction=0.5, shape="ramp")
    np.testing.assert_allclose(ramp[10:20, 1], 1.0)  # own points keep full weight
    upper = ramp[20:25, 1]  # borrowed from above
    assert np.all(np.diff(upper) < 0), "ramp must decay away from the shared edge"
    assert upper[0] == pytest.approx(1.0)
    lower = ramp[5:10, 1]  # borrowed from below
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


def test_random_band_matches_the_adjacent_band_in_everything_but_row_identity():
    """The control's whole job: same counts, same weights, structure destroyed."""
    y = np.arange(60, dtype=float)
    labels = np.repeat([0, 1, 2], 20)
    for shape in ("flat", "ramp"):
        adj = overlap_weights(y, labels, 3, 0.4, shape=shape, band="adjacent")
        rnd = overlap_weights(
            y, labels, 3, 0.4, shape=shape, band="random", random_state=11
        )
        np.testing.assert_array_equal((adj > 0).sum(axis=0), (rnd > 0).sum(axis=0))
        for b in range(3):
            np.testing.assert_allclose(
                np.sort(adj[adj[:, b] > 0, b]), np.sort(rnd[rnd[:, b] > 0, b])
            )
        assert not np.array_equal(adj > 0, rnd > 0), "the control did not move any row"
    # The ends stay single-sided: a random band still only reaches into a real
    # neighbour, so bucket 0 cannot gain a point from bucket 2.
    rnd = overlap_weights(y, labels, 3, 1.0, band="random", random_state=11)
    assert set(np.flatnonzero(rnd[:, 0])) <= set(range(40))


def test_random_band_is_reproducible_and_seed_dependent():
    y = np.arange(60, dtype=float)
    labels = np.repeat([0, 1, 2], 20)
    kw = dict(shape="flat", band="random")
    a = overlap_weights(y, labels, 3, 0.4, random_state=5, **kw)
    np.testing.assert_array_equal(
        a, overlap_weights(y, labels, 3, 0.4, random_state=5, **kw)
    )
    assert not np.array_equal(
        a, overlap_weights(y, labels, 3, 0.4, random_state=6, **kw)
    )


def test_fraction_is_validated():
    y, labels = np.arange(10, dtype=float), np.repeat([0, 1], 5)
    with pytest.raises(ValueError):
        overlap_weights(y, labels, 2, fraction=1.5)
    with pytest.raises(ValueError):
        overlap_weights(y, labels, 2, fraction=0.2, shape="gaussian")
    with pytest.raises(ValueError):
        overlap_weights(y, labels, 2, fraction=0.2, band="nearby")


def test_weighted_means_move_toward_the_neighbours():
    y = np.arange(30, dtype=float)
    labels = np.repeat([0, 1, 2], 10)
    hard = np.array([4.5, 14.5, 24.5])
    W = overlap_weights(y, labels, 3, fraction=0.5, shape="flat")
    soft = overlap_bucket_means(y, W, hard, pin_extremes=False)
    assert soft[1] == pytest.approx(14.5)  # symmetric overlap: unchanged
    assert soft[0] > hard[0]  # pulled up toward bucket 1
    assert soft[2] < hard[2]  # pulled down toward bucket 1
    pinned = overlap_bucket_means(y, W, hard, pin_extremes=True)
    assert pinned[0] == hard[0] and pinned[2] == hard[2]


# --------------------------------------------------------------------------
# Degeneracy against the library
# --------------------------------------------------------------------------
@pytest.mark.parametrize("order", ["1st", "2nd", "full-2nd"])
@pytest.mark.parametrize("partition", ["uniform", "quantile"])
def test_defaults_reproduce_the_library_regressor(data, order, partition):
    X, y = data
    kwargs = dict(
        n_output_buckets=3,
        output_partition=partition,
        tsk_order=order,
        l2_reg=1e-2,
        pin_extremes=True,
        random_state=7,
    )
    base = TribbleRegressor(**kwargs).fit(X, y)
    ours = OverlapTribbleRegressor(overlap=0.0, **kwargs).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), base.predict(X), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        ours.y_bucket_mean_, base.y_bucket_mean_, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(ours.corr_terms_, base.corr_terms_, rtol=1e-8, atol=1e-8)


def test_zero_overlap_membership_model_matches_the_library_fit(data):
    X, y = data
    y_part, _ = partition_output(4, y, method="quantile")
    labels = y_part["y_bucket"].to_numpy()
    W = overlap_weights(y.to_numpy(), labels, 4, fraction=0.0)
    features = ["a", "b", "c"]
    ours = build_overlap_membership_model(X, W, features, n_gaussians=0, random_state=3)
    base = create_gaussian_membership_dict(
        X, y_part["y_bucket"], features, n_gaussians=0, random_state=3
    )
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
    model = create_gaussian_membership_dict(
        X, y_part["y_bucket"], features, n_gaussians=0, random_state=3
    )
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
    model = create_gaussian_membership_dict(
        X, y_part["y_bucket"], features, n_gaussians=0, random_state=3
    )
    firing, labels = tsk_firing_strengths(X[features], model)

    def spread(fusion_reg):
        corr, _ = solve_consequents_fused(
            firing,
            labels,
            X,
            features,
            means,
            y_part,
            order="2nd",
            l2_reg=1e-3,
            fusion_reg=fusion_reg,
        )
        return float(np.abs(np.diff(corr, axis=0)).mean())

    gaps = [spread(f) for f in (0.0, 1.0, 100.0)]
    assert (
        gaps[0] > gaps[1] > gaps[2]
    ), f"fusion did not pull neighbours together: {gaps}"


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
            n_output_buckets=4,
            output_partition="quantile",
            tsk_order="1st",
            n_gaussians=1,
            overlap=fraction,
            overlap_shape="flat",
            random_state=7,
        ).fit(X, y)
        return float(np.mean([g.sigma for g in m.model_.all_membership_fcns]))

    assert mean_sigma(0.4) > mean_sigma(0.0)


def test_bic_may_spend_a_wider_slice_on_more_components_instead_of_wider_ones(data):
    """Documents the confound above, so a later reader does not rediscover it."""
    X, y = data

    def counts(fraction):
        m = OverlapTribbleRegressor(
            n_output_buckets=4,
            output_partition="quantile",
            tsk_order="1st",
            n_gaussians=0,
            overlap=fraction,
            random_state=7,
        ).fit(X, y)
        return len(m.model_.all_membership_fcns)

    assert counts(0.4) > counts(0.0)


def test_overlap_raises_adjacent_membership_overlap(data):
    X, y = data

    def area(fraction):
        return (
            OverlapTribbleRegressor(
                n_output_buckets=4,
                output_partition="quantile",
                tsk_order="1st",
                overlap=fraction,
                random_state=7,
            )
            .fit(X, y)
            .membership_overlap_area()
        )

    assert area(0.4) > area(0.0)


def test_only_the_antecedent_switch_moves_the_firing_strengths(data):
    """Firing strengths come from the membership functions and nothing else."""
    X, y = data
    kwargs = dict(
        n_output_buckets=4, output_partition="quantile", tsk_order="1st", random_state=7
    )
    hard = OverlapTribbleRegressor(overlap=0.0, **kwargs).fit(X, y)
    for extra in (
        dict(overlap_antecedents=False),
        dict(overlap_antecedents=False, consequent_fit="local"),
    ):
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
    base = dict(
        n_output_buckets=4, output_partition="quantile", tsk_order="1st", random_state=7
    )
    soft = dict(overlap=0.3, overlap_antecedents=False)

    inert = dict(consequent_fit="global", pin_extremes=False, **base)
    np.testing.assert_allclose(
        OverlapTribbleRegressor(overlap=0.0, **inert).fit(X, y).predict(X),
        OverlapTribbleRegressor(**soft, **inert).fit(X, y).predict(X),
        rtol=1e-12,
        atol=1e-12,
    )

    for live in (
        dict(consequent_fit="global", pin_extremes=True, **base),
        dict(consequent_fit="local", pin_extremes=False, **base),
    ):
        assert not np.allclose(
            OverlapTribbleRegressor(overlap=0.0, **live).fit(X, y).predict(X),
            OverlapTribbleRegressor(**soft, **live).fit(X, y).predict(X),
        )


@pytest.mark.parametrize("shape", ["flat", "ramp"])
@pytest.mark.parametrize("fit", ["global", "local"])
def test_every_arm_fits_and_predicts_finitely(data, shape, fit):
    X, y = data
    m = OverlapTribbleRegressor(
        n_output_buckets=5,
        output_partition="quantile",
        tsk_order="full-2nd",
        l2_reg=1e-2,
        overlap=0.25,
        overlap_shape=shape,
        consequent_fit=fit,
        pin_extremes=True,
        random_state=7,
    ).fit(X, y)
    p = m.predict(X)
    assert p.shape == (len(X),)
    assert np.all(np.isfinite(p))


def test_local_fit_at_zero_overlap_is_the_hard_per_bucket_fit(data):
    """The control that separates 'local instead of global' from 'soft instead of hard'."""
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-3,
        random_state=7,
    )
    local_hard = OverlapTribbleRegressor(
        consequent_fit="local", overlap=0.0, **kwargs
    ).fit(X, y)
    local_soft = OverlapTribbleRegressor(
        consequent_fit="local",
        overlap=0.3,
        overlap_antecedents=False,
        overlap_means=False,
        **kwargs,
    ).fit(X, y)
    # Same antecedents and same centroid inputs; only the consequent slices differ.
    f_a, _ = tsk_firing_strengths(X[local_hard.top_features_], local_hard.model_)
    f_b, _ = tsk_firing_strengths(X[local_soft.top_features_], local_soft.model_)
    np.testing.assert_allclose(f_a, f_b, rtol=1e-12, atol=1e-12)
    assert not np.allclose(local_hard.corr_terms_, local_soft.corr_terms_)


def test_ramp_and_flat_differ_only_through_the_weights(data):
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        overlap=0.3,
        consequent_fit="local",
        random_state=7,
    )
    flat = OverlapTribbleRegressor(overlap_shape="flat", **kwargs).fit(X, y)
    ramp = OverlapTribbleRegressor(overlap_shape="ramp", **kwargs).fit(X, y)
    W_flat, W_ramp = flat.overlap_weights_, ramp.overlap_weights_
    np.testing.assert_array_equal(W_flat > 0, W_ramp > 0)  # same support
    assert np.all(W_ramp <= W_flat + 1e-12)  # ramp only ever lighter
    assert not np.allclose(flat.predict(X), ramp.predict(X))


# --------------------------------------------------------------------------
# Follow-up arms: fit quality vs aggregation
# --------------------------------------------------------------------------
def test_shrink_local_reduces_to_the_baseline_at_zero_ridge(data):
    """With no ridge there is no prior, so the shrunk solve IS the global solve.

    The antecedents have to be held hard for this to be a statement about the
    consequent solver: `overlap_antecedents` moves the firing strengths, and two
    solves of different designs have no reason to agree. That is also why the
    driver runs the `shrink-local` arm with the antecedent overlap off -- otherwise
    the arm would confound the consequent prior with a different forward pass, the
    exact confound `soft-random` was added to break.
    """
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=0.0,
        random_state=7,
    )
    base = OverlapTribbleRegressor(consequent_fit="global", **kwargs).fit(X, y)
    shrunk = OverlapTribbleRegressor(
        consequent_fit="shrink-local",
        overlap=0.3,
        overlap_antecedents=False,
        overlap_means=False,
        **kwargs,
    ).fit(X, y)
    np.testing.assert_allclose(shrunk.predict(X), base.predict(X), rtol=1e-6, atol=1e-6)


def test_shrink_local_with_soft_antecedents_is_a_different_model(data):
    """Guards the confound the test above avoids, so nobody re-introduces it."""
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=0.0,
        consequent_fit="shrink-local",
        overlap=0.3,
        random_state=7,
    )
    hard_ante = OverlapTribbleRegressor(
        overlap_antecedents=False, overlap_means=False, **kwargs
    ).fit(X, y)
    soft_ante = OverlapTribbleRegressor(overlap_antecedents=True, **kwargs).fit(X, y)
    assert not np.allclose(hard_ante.predict(X), soft_ante.predict(X))


def test_shrink_local_lands_between_its_two_limits(data):
    """A big enough ridge pulls the corrections onto the local prior."""
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        overlap=0.3,
        overlap_antecedents=False,
        overlap_means=False,
        random_state=7,
    )
    far = OverlapTribbleRegressor(
        consequent_fit="shrink-local", l2_reg=1e6, **kwargs
    ).fit(X, y)
    prior_corr, _ = far.local_prior_
    np.testing.assert_allclose(far.corr_terms_, prior_corr, rtol=1e-3, atol=1e-3)

    near = OverlapTribbleRegressor(
        consequent_fit="shrink-local", l2_reg=1e-8, **kwargs
    ).fit(X, y)
    assert (
        np.abs(near.corr_terms_ - prior_corr).mean()
        > np.abs(far.corr_terms_ - prior_corr).mean()
    )


def test_residual_form_pins_every_constant_to_the_bucket_centroid(data):
    """The legacy formulation fixes all the constants, not just the extremes."""
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-3,
        overlap=0.2,
        random_state=7,
    )
    res = OverlapTribbleRegressor(consequent_fit="local-residual", **kwargs).fit(X, y)
    free = OverlapTribbleRegressor(consequent_fit="local", **kwargs).fit(X, y)
    # The residual form's means are the centroids it was handed, untouched.
    np.testing.assert_allclose(
        res.y_bucket_mean_, res.y_bucket_mean_in_, rtol=1e-10, atol=1e-10
    )
    assert not np.allclose(
        free.y_bucket_mean_, free.y_bucket_mean_in_
    ), "the free-intercept form should have moved at least one constant"
    # A free intercept is a superset of a fixed one, so it cannot fit its own
    # slices worse. Compared on the training fold, which is what it optimized.
    hard = res.hard_labels_
    assert (
        free.local_approximation_r2(X, y, hard)
        >= res.local_approximation_r2(X, y, hard) - 1e-9
    )


def test_winner_take_all_uses_exactly_one_rule_per_row(data):
    X, y = data
    m = OverlapTribbleRegressor(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-3,
        consequent_fit="local",
        overlap=0.3,
        predict_mode="wta",
        random_state=7,
    ).fit(X, y)
    norm_fs, rule_vals, _ = m._rule_values_and_weights(X)
    winner = np.argmax(norm_fs, axis=1)
    np.testing.assert_allclose(
        m.predict(X), rule_vals[np.arange(len(X)), winner], rtol=1e-12, atol=1e-12
    )
    assert not np.allclose(
        m.predict(X),
        OverlapTribbleRegressor(
            n_output_buckets=4,
            output_partition="quantile",
            tsk_order="2nd",
            l2_reg=1e-3,
            consequent_fit="local",
            overlap=0.3,
            random_state=7,
        )
        .fit(X, y)
        .predict(X),
    )


def test_blend_recalibration_is_the_identity_under_an_infinite_prior(data):
    """lambda -> inf pins (a_r, b_r) at (0, 1), which is the plain blend."""
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-3,
        consequent_fit="local",
        overlap=0.3,
        random_state=7,
    )
    plain = OverlapTribbleRegressor(**kwargs).fit(X, y)
    pinned = OverlapTribbleRegressor(
        blend_recalibrate=True, blend_recal_l2=1e12, **kwargs
    ).fit(X, y)
    np.testing.assert_allclose(pinned.blend_a_, 0.0, atol=1e-6)
    np.testing.assert_allclose(pinned.blend_b_, 1.0, atol=1e-6)
    np.testing.assert_allclose(
        pinned.predict(X), plain.predict(X), rtol=1e-5, atol=1e-5
    )

    free = OverlapTribbleRegressor(
        blend_recalibrate=True, blend_recal_l2=0.0, **kwargs
    ).fit(X, y)
    assert not np.allclose(free.predict(X), plain.predict(X))


def test_local_r2_measures_the_rule_not_the_blend(data):
    """The diagnostic must be insensitive to how rules are combined."""
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-3,
        consequent_fit="local",
        overlap=0.3,
        random_state=7,
    )
    blended = OverlapTribbleRegressor(**kwargs).fit(X, y)
    wta = OverlapTribbleRegressor(predict_mode="wta", **kwargs).fit(X, y)
    hard = blended.hard_labels_
    assert blended.local_approximation_r2(X, y, hard) == pytest.approx(
        wta.local_approximation_r2(X, y, hard)
    )
    assert not np.allclose(blended.predict(X), wta.predict(X))


def test_local_fits_approximate_their_own_buckets_better_than_the_global_solve(data):
    """The premise of the follow-up, stated as a test rather than assumed.

    A per-bucket solve optimizes exactly this quantity on the training fold, so if
    it does not win here the implementation is wrong, not the idea.
    """
    X, y = data
    kwargs = dict(
        n_output_buckets=5,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-3,
        random_state=7,
    )
    glob = OverlapTribbleRegressor(consequent_fit="global", **kwargs).fit(X, y)
    loc = OverlapTribbleRegressor(consequent_fit="local", overlap=0.0, **kwargs).fit(
        X, y
    )
    hard = glob.hard_labels_
    assert loc.local_approximation_r2(X, y, hard) > glob.local_approximation_r2(
        X, y, hard
    )


def test_predict_mode_is_validated(data):
    X, y = data
    m = OverlapTribbleRegressor(n_output_buckets=3, random_state=7).fit(X, y)
    m.predict_mode = "argmax"
    with pytest.raises(ValueError):
        m.predict(X)


def test_sharpening_is_the_identity_at_gamma_one(data):
    """gamma=1 must take the library's own predict path, byte for byte."""
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-3,
        consequent_fit="local",
        overlap=0.3,
        random_state=7,
    )
    plain = OverlapTribbleRegressor(**kwargs).fit(X, y)
    unity = OverlapTribbleRegressor(blend_sharpen=1.0, **kwargs).fit(X, y)
    np.testing.assert_allclose(
        unity.predict(X), plain.predict(X), rtol=1e-12, atol=1e-12
    )


def test_large_sharpening_converges_to_winner_take_all(data):
    """gamma -> inf is WTA, which is what makes the exponent a blend-width knob.

    Coefficients are held fixed and only the exponent varies, because raising
    gamma also re-solves the consequents -- so comparing two *fitted* models would
    conflate the aggregation with the fit.
    """
    from overlap import sharpen_firing
    from tribblefis.regression import _normalize_firing_strengths

    X, y = data
    m = OverlapTribbleRegressor(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-3,
        consequent_fit="local",
        overlap=0.3,
        random_state=7,
    ).fit(X, y)
    raw, _, rule_vals = (
        *tsk_firing_strengths(X[m.top_features_], m.model_),
        m._rule_values_and_weights(X)[1],
    )

    gaps, concentration = [], []
    for gamma in (1.0, 16.0, 256.0, 4096.0):
        norm = _normalize_firing_strengths(sharpen_firing(raw, gamma))
        winner = rule_vals[np.arange(len(X)), np.argmax(norm, axis=1)]
        gaps.append(float(np.abs(np.sum(norm * rule_vals, axis=1) - winner).max()))
        concentration.append(float(norm.max(axis=1).mean()))

    assert gaps == sorted(gaps, reverse=True), f"blend did not converge to WTA: {gaps}"
    assert concentration == sorted(concentration), concentration
    assert concentration[-1] > 0.999
    # The rate is set by each row's top-two firing ratio: a row whose two
    # strongest rules are within 0.2% of each other (the worst on this fixture)
    # needs gamma in the thousands before the runner-up's weight collapses. Those
    # are exactly the rows sitting on a bucket boundary -- the ones this whole
    # experiment is about -- so the slow tail is the interesting part, not noise.
    assert gaps[-1] < 1e-3, f"still {gaps[-1]:.2e} from WTA at gamma=4096"


def test_sharpening_does_not_underflow_a_row_to_zero(data):
    """The guard on `sharpen_firing`, as a test rather than a comment.

    Without the per-row rescale, a large exponent drives whole rows under
    `_normalize_firing_strengths`' 1e-6 floor, the row is returned as all-zeros,
    and the model predicts 0 there -- which looks like sharpening making the model
    worse when it is really an underflow.
    """
    from overlap import sharpen_firing
    from tribblefis.regression import _normalize_firing_strengths

    X, y = data
    m = OverlapTribbleRegressor(
        n_output_buckets=5, output_partition="quantile", tsk_order="1st", random_state=7
    ).fit(X, y)
    raw, _ = tsk_firing_strengths(X[m.top_features_], m.model_)
    live = _normalize_firing_strengths(raw).sum(axis=1) > 0
    for gamma in (2.0, 8.0, 64.0, 256.0):
        norm = _normalize_firing_strengths(sharpen_firing(raw, gamma))
        np.testing.assert_array_equal(
            norm.sum(axis=1) > 0, live, err_msg=f"row set changed at gamma={gamma}"
        )


def test_sharpening_is_applied_in_the_solve_as_well_as_the_prediction(data):
    """Fit/predict must share one weighting; this codebase has been bitten before."""
    X, y = data
    kwargs = dict(
        n_output_buckets=4,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-3,
        consequent_fit="global",
        random_state=7,
    )
    a = OverlapTribbleRegressor(blend_sharpen=1.0, **kwargs).fit(X, y)
    b = OverlapTribbleRegressor(blend_sharpen=3.0, **kwargs).fit(X, y)
    assert not np.allclose(
        a.corr_terms_, b.corr_terms_
    ), "the exponent did not reach the consequent solve"
    assert not np.allclose(a.predict(X), b.predict(X))


# --------------------------------------------------------------------------
# Stage 3: compact support
# --------------------------------------------------------------------------
def test_clamped_gaussian_is_not_admitted_to_the_compiled_kernel():
    """The trap this type exists to avoid, pinned so a rebuild cannot re-open it.

    `kernel.compile_model` admits a model on `isinstance(mf, GaussianMembership)`
    and then evaluates a *plain* Gaussian. If `ClampedGaussianMembership` were a
    subclass it would pass that check and the clamp would be silently dropped
    wherever the Cython extension is built -- which is not this environment, so the
    failure would never show up here and always show up in production.
    """
    from tribblefis import kernel
    from tribblefis.gauss_data import GaussianMembership as G

    from overlap import ClampedGaussianMembership as C

    assert not isinstance(C.create(0.0, 1.0), G)
    assert not issubclass(C, G)

    model = GaussianMixtureModel(
        feature_models={
            "a": FeatureModel(
                label_models={0: LabelModel(memberships=[C.create(0.0, 1.0)])}
            )
        }
    )
    with pytest.raises(kernel.NotCompilable):
        kernel.compile_model(model, ["a"])


@pytest.mark.parametrize("k", [2.0, 2.75, 3.0])
def test_clamped_gaussian_reaches_exactly_zero_at_the_cutoff(k):
    from overlap import ClampedGaussianMembership as C

    grid = np.linspace(-6, 6, 2001)
    smooth = C.create(0.0, 1.0, k=k, smooth=True)
    hard = C.create(0.0, 1.0, k=k, smooth=False)

    for mf in (smooth, hard):
        vals = mf.evaluate(grid)
        assert np.all(vals >= 0.0)
        assert mf.evaluate(np.array([0.0]))[0] == pytest.approx(1.0)
        assert np.all(vals[np.abs(grid) > k] == 0.0), "support is not compact"

    # The smooth form meets the axis continuously; the hard one steps down by
    # exp(-k^2/2), which is what "non-linear clamp" is there to avoid.
    just_inside = np.array([k - 1e-6])
    assert smooth.evaluate(just_inside)[0] == pytest.approx(0.0, abs=1e-5)
    assert hard.evaluate(just_inside)[0] == pytest.approx(np.exp(-0.5 * k**2), rel=1e-3)


def test_clamped_gaussian_matches_a_plain_gaussian_well_inside_the_cutoff():
    """A large k must leave the shape alone where it matters."""
    from tribblefis.gauss_data import GaussianMembership

    from overlap import ClampedGaussianMembership as C

    grid = np.linspace(-2.0, 2.0, 401)
    plain = GaussianMembership.create(0.3, 0.8).evaluate(grid)
    clamped = C.create(0.3, 0.8, k=12.0, smooth=True).evaluate(grid)
    np.testing.assert_allclose(clamped, plain, atol=1e-10)


def test_ruspini_terms_partition_the_axis_but_one_bucket_does_not(data):
    """The correction to a wrong prediction of mine, kept as a test.

    A Ruspini partition sums to exactly 1 at every point -- so it was tempting to
    expect a ruspini model to have full *rule* coverage. It does not, and the two
    statements are different: the partition tiles the axis, but each bucket is
    given only the term(s) nearest its own centres, so a bucket's own membership
    covers a fraction of the axis and the AND over features covers less again.
    Coverage of the axis by the term set does not imply coverage by any one rule.
    """
    from tribblefis.ruspini import verify_partition_of_unity

    X, y = data
    m = OverlapTribbleRegressor(
        n_output_buckets=5,
        output_partition="quantile",
        tsk_order="1st",
        membership="ruspini",
        random_state=7,
    ).fit(X, y)

    grid = np.linspace(-0.5, 1.5, 1001)
    for feature_model in m.model_.feature_models.values():
        terms = {}
        for lmodel in feature_model.label_models.values():
            for mf in lmodel.memberships:
                terms[mf.id] = mf
        assert verify_partition_of_unity(
            list(terms.values()), grid
        ), "the shared term set is not a partition of unity"

    assert (
        m.coverage(X)["uncovered"] > 0.0
    ), "a per-bucket selection from a partition should still leave gaps"


@pytest.mark.parametrize("membership", ["gaussian", "clamped", "trapezoid", "ruspini"])
def test_every_membership_shape_fits_predicts_and_reports_coverage(data, membership):
    X, y = data
    m = OverlapTribbleRegressor(
        n_output_buckets=5,
        output_partition="quantile",
        tsk_order="2nd",
        l2_reg=1e-2,
        membership=membership,
        overlap=0.25,
        consequent_fit="local",
        random_state=7,
    ).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (len(X),) and np.all(np.isfinite(pred))
    cov = m.coverage(X)
    assert 0.0 <= cov["uncovered"] <= 1.0
    assert 0.0 <= cov["active_frac"] <= 1.0


def test_gaussian_membership_covers_everything_and_clamping_reduces_it(data):
    """The premise of stage 3: an unclamped Gaussian model fires everywhere."""
    X, y = data

    def cov(**kw):
        return (
            OverlapTribbleRegressor(
                n_output_buckets=5,
                output_partition="quantile",
                tsk_order="1st",
                random_state=7,
                **kw,
            )
            .fit(X, y)
            .coverage(X)
        )

    plain = cov(membership="gaussian")
    # No row is ever left unanswered: with infinite support the total firing is
    # strictly positive everywhere.
    assert plain["uncovered"] == 0.0
    # `active_frac` counts rules above 1e-6, so it is high but not exactly 1 --
    # a Gaussian past about 5 sigma is nonzero and still below that floor. The
    # claim being tested is that clamping *reduces* it, not that it starts at 1.
    assert plain["active_frac"] > 0.8

    for k in (3.0, 2.0):
        tight = cov(membership="clamped", clamp_k=k)
        assert tight["active_frac"] < plain["active_frac"], f"k={k} did not localize"
    assert (
        cov(membership="clamped", clamp_k=2.0)["active_frac"]
        < cov(membership="clamped", clamp_k=3.0)["active_frac"]
    )


def test_coverage_report_counts_uncovered_rows():
    from overlap import coverage_report

    firing = np.array([[0.5, 0.2], [0.0, 0.0], [1e-9, 0.0], [0.9, 0.0]])
    rep = coverage_report(firing)
    assert rep["uncovered"] == pytest.approx(0.5)  # rows 1 and 2
    assert rep["mean_active"] == pytest.approx((2 + 0 + 0 + 1) / 4)
    assert rep["active_frac"] == pytest.approx(rep["mean_active"] / 2)


def test_membership_is_validated(data):
    X, y = data
    with pytest.raises(ValueError):
        OverlapTribbleRegressor(membership="bell").fit(X, y)


# --------------------------------------------------------------------------
# Stage 4: the trapezoid fitter's endpoint defect
# --------------------------------------------------------------------------
def test_trapezoid_membership_is_zero_at_its_own_endpoints():
    """The library behaviour the padding fix exists for, pinned as a fact.

    Correct for an open trapezoid, and a defect once the fitter sets ``a`` to the
    data minimum: the smallest observed value then gets zero membership from the
    term fitted to describe it.
    """
    from tribblefis.gauss_data import TrapezoidMembership

    t = TrapezoidMembership.create(a=0.0, b=0.25, c=0.75, d=1.0)
    assert t.evaluate(np.array([0.0]))[0] == 0.0
    assert t.evaluate(np.array([1.0]))[0] == 0.0
    assert t.evaluate(np.array([0.5]))[0] == pytest.approx(1.0)


def test_padding_gives_the_data_range_full_membership(data):
    """After padding, every point the term was fitted to has membership 1."""
    from tribblefis.gauss_data import TrapezoidMembership

    from overlap import pad_trapezoids

    X = pd.DataFrame({"a": np.linspace(0.0, 1.0, 50)})
    model = GaussianMixtureModel(
        feature_models={
            "a": FeatureModel(
                label_models={
                    0: LabelModel(
                        memberships=[TrapezoidMembership.create(0.2, 0.3, 0.7, 0.8)]
                    )
                }
            )
        }
    )
    padded = pad_trapezoids(model, X, pad=0.25)
    mf = padded.feature_models["a"].label_models[0].memberships[0]

    # The old support becomes the plateau...
    assert (mf.b, mf.c) == pytest.approx((0.2, 0.8))
    assert mf.evaluate(np.array([0.2, 0.5, 0.8]))[0] == pytest.approx(1.0)
    np.testing.assert_allclose(mf.evaluate(np.array([0.2, 0.5, 0.8])), 1.0)
    # ...and the support extends pad * width beyond it, still compact.
    assert mf.a == pytest.approx(0.2 - 0.25 * 0.6)
    assert mf.d == pytest.approx(0.8 + 0.25 * 0.6)
    assert mf.evaluate(np.array([mf.a - 1e-9, mf.d + 1e-9]))[0] == 0.0


def test_padding_rescues_a_degenerate_single_value_region():
    """a == d would otherwise be a delta function that fires nowhere."""
    from tribblefis.gauss_data import TrapezoidMembership

    from overlap import pad_trapezoids

    X = pd.DataFrame({"a": np.linspace(0.0, 1.0, 50)})
    model = GaussianMixtureModel(
        feature_models={
            "a": FeatureModel(
                label_models={
                    0: LabelModel(
                        memberships=[TrapezoidMembership.create(0.5, 0.5, 0.5, 0.5)]
                    )
                }
            )
        }
    )
    mf = (
        pad_trapezoids(model, X, pad=0.2)
        .feature_models["a"]
        .label_models[0]
        .memberships[0]
    )
    assert mf.d > mf.a, "a degenerate region was left as a delta"
    assert mf.evaluate(np.array([0.5]))[0] == pytest.approx(1.0)
    assert mf.evaluate(np.array([0.45]))[0] > 0.0


def test_upstream_fitter_covers_a_zero_inflated_column_without_padding(data):
    """Guards the upstream fix, and is why this test changed shape.

    Until `tribble-fis` #170 this asserted the *defect*: at ``trapz_pad=0`` a
    zero-inflated column left >20% of rows covered by no rule, and `pad_trapezoids`
    was what closed the hole. #170 fixed the fitter itself -- the plateau now spans
    the data region -- so at the pinned SHA `141596e` and later there is no hole to
    close, and ``uncovered`` is 0.0 with no padding at all.

    Kept as a guard rather than deleted: if the upstream geometry ever regresses,
    this fails, and `pad_trapezoids` remains available as the local remedy. Stage
    4's ``pad=0`` measurements in RESULTS.md were taken at the *pre-fix* pin
    (`058501f`) and cannot be reproduced at the current one -- see the note there.
    """
    X, y = data
    # A zero-inflated column, which is what triggered the defect on real data
    # (55% of concrete's scaled FlyAsh sits exactly at its minimum).
    X = X.copy()
    X["zeros"] = np.where(np.arange(len(X)) % 2 == 0, 0.0, X["a"].to_numpy())

    def cov(pad):
        return (
            OverlapTribbleRegressor(
                n_output_buckets=5,
                output_partition="quantile",
                tsk_order="1st",
                membership="trapezoid",
                trapz_bins=1,
                trapz_pad=pad,
                random_state=7,
            )
            .fit(X, y)
            .coverage(X)
        )

    assert cov(0.0)["uncovered"] == 0.0, "upstream fitter left rows uncovered"
    assert cov(0.1)["uncovered"] == 0.0
    # Padding on top of a fixed fitter is now a widening knob, not a repair.
    assert cov(0.3)["active_frac"] > cov(0.0)["active_frac"]


def test_fewer_bins_widen_the_support(data):
    """The n_bins mechanism, once padding has removed the endpoint confound."""
    X, y = data

    def frac(bins):
        return (
            OverlapTribbleRegressor(
                n_output_buckets=5,
                output_partition="quantile",
                tsk_order="1st",
                membership="trapezoid",
                trapz_bins=bins,
                trapz_pad=0.05,
                random_state=7,
            )
            .fit(X, y)
            .coverage(X)["active_frac"]
        )

    assert frac(1) >= frac(50), "coarser bins should not narrow the support"


def test_trapz_ramp_now_widens_the_support(data):
    """The other half of what `tribble-fis` #170 changed.

    Before the fix the ramps were inset *inside* the fitted region, so
    `ramp_width_ratio` moved the plateau and could not widen ``[a, d]`` at all --
    which is why sweeping it saturated in stage 3, and this test asserted the
    bounds were identical across ramp widths. After the fix the ramps are the
    outward shoulders, so the parameter is a genuine support knob.
    """
    X, y = data

    def bounds(ramp):
        m = OverlapTribbleRegressor(
            n_output_buckets=3,
            output_partition="quantile",
            tsk_order="1st",
            membership="trapezoid",
            trapz_bins=20,
            trapz_ramp=ramp,
            random_state=7,
        ).fit(X, y)
        return np.array(sorted((mf.a, mf.d) for mf in m.model_.all_membership_fcns))

    narrow, wide = bounds(0.1), bounds(0.4)
    assert narrow.shape == wide.shape
    # Wider ramp: every support's left edge moves left and right edge moves right.
    assert np.all(wide[:, 0] <= narrow[:, 0] + 1e-12)
    assert np.all(wide[:, 1] >= narrow[:, 1] - 1e-12)
    assert np.any(wide[:, 0] < narrow[:, 0] - 1e-9), "ramp width did not widen [a, d]"
