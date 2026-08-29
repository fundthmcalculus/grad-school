"""Network-free tests for `feature_expansion.AgglomerativeFeatureExpansion`.

All synthetic, so these run anywhere the `tribble-fis` submodule is importable:
    uv run --project tribble-fis python -m pytest \\
        reproduce/regression_scale/test_feature_expansion.py -v

They assert the two properties the module exists to guarantee -- that the
searched answer equals the full-scan answer (correctness) while touching fewer
k's (efficiency) -- plus agglomeration and cache-reuse behaviour.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_expansion as fe  # noqa: E402


def _synth_classification(n=1500, seed=0):
    """3 informative features (two near-duplicate) buried in 20."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    cols = {f"info{i}": base[:, i] for i in range(3)}
    cols["info0_dup"] = base[:, 0] + 0.01 * rng.normal(size=n)
    for j in range(16):
        cols[f"noise{j}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    y = (base[:, 0] * 1.5 + base[:, 1] - 0.8 * base[:, 2] > 0).astype(int)
    return X, pd.Series(y)


def _fitted(**kw):
    X, y = _synth_classification()
    sel = fe.AgglomerativeFeatureExpansion(
        task="classification", random_state=0, verbose=False, **kw
    )
    return sel.fit(X, y)


def test_agglomeration_merges_the_duplicate():
    sel = _fitted(decorrelate=True)
    # info0_dup is ~1.0 correlated with info0, so exactly one of the 20 merges
    # away and the more-differentiating name (info0) survives.
    assert sel.n_features_survived_ == 19
    assert "info0" in sel.rank_
    assert "info0_dup" not in sel.rank_
    assert "info0_dup" in sel.dropped_["info0"]


def test_ranking_puts_informative_features_first():
    sel = _fitted()
    assert set(sel.rank_[:3]) == {"info0", "info1", "info2"}


def test_target_bisection_matches_full_scan_but_cheaper():
    sel = _fitted()
    # A target reachable at small k, so there is a real crossing to find.
    target = 0.78
    res = sel.select(target=target)
    # Full scan on the same (cached) object: smallest k that clears the bar.
    for k in range(1, sel.n_features_survived_ + 1):
        sel._evaluate(k)
    passing = [
        k
        for k in range(1, sel.n_features_survived_ + 1)
        if sel._cache[k]["score"] >= target
    ]
    assert res.reached_target
    assert res.k == min(passing)  # correctness
    assert res.n_evaluations < sel.n_features_survived_  # efficiency


def test_target_unreached_reports_true_best():
    sel = _fitted()
    res = sel.select(target=1.01, verify_scan=True)  # impossible target
    assert res.reached_target is False
    best = max(r["score"] for r in sel._cache.values())
    assert res.score == pytest.approx(best)
    # Ties break toward the smaller model.
    assert res.k == min(
        k for k, r in sel._cache.items() if r["score"] == pytest.approx(best)
    )


def test_plateau_finds_the_knee():
    sel = _fitted()
    res = sel.select(plateau_tol=0.005, patience=2)
    # The knee sits at the informative-feature count, not the full set.
    assert res.k <= 4
    assert res.k < sel.n_features_survived_
    assert res.reached_target


def test_cache_is_reused_across_calls():
    sel = _fitted()
    sel.select(target=0.78)
    before = dict(sel._cache)  # {k: record object}
    res = sel.select(plateau_tol=0.005, patience=2)
    # A k already evaluated is returned as the *same* record object, never re-fit.
    for k, rec in before.items():
        if k in sel._cache:
            assert sel._cache[k] is rec
    # And n_evaluations counts exactly the k's this call newly fitted.
    assert res.n_evaluations == len(set(sel._cache) - set(before))


def test_verify_scan_raises_on_target_disagreement():
    """A non-monotone spike the gallop skips: k=3 clears the bar but the
    power-of-two probes (1,2,4,8) put bisection at k=8. verify_scan must raise,
    not warn, because the full scan's smallest passing k (3) differs."""
    sel = _fitted()
    M = sel.n_features_survived_
    curve = {k: 0.50 for k in range(1, M + 1)}
    curve[3] = 0.99  # the spike the gallop cannot see
    curve[8] = 0.99  # a power-of-two pass so the gallop brackets high
    sel._cache = {
        k: {"k": k, "features": sel.rank_[:k], "score": s, "fit_s": 0.0}
        for k, s in curve.items()
    }
    with pytest.raises(AssertionError, match="smallest passing k is 3"):
        sel.select(target=0.97, verify_scan=True)


def test_verify_scan_plateau_agrees_and_does_not_raise():
    """Plateau-mode verify_scan is no longer a no-op: it recomputes the knee over
    the full cache and agrees on the clean synthetic curve."""
    sel = _fitted()
    res = sel.select(plateau_tol=0.005, patience=2, verify_scan=True)
    assert res.reached_target
    assert res.k <= 4


def test_decorrelation_off_keeps_all_features():
    sel = _fitted(decorrelate=False)
    assert sel.n_features_survived_ == 20
    assert "info0_dup" in sel.rank_


def test_regression_path_ranks_and_selects():
    rng = np.random.default_rng(1)
    n = 1200
    base = rng.normal(size=(n, 3))
    cols = {f"info{i}": base[:, i] for i in range(3)}
    for j in range(10):
        cols[f"noise{j}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    y = 2.0 * base[:, 0] - 1.3 * base[:, 1] + 0.7 * base[:, 2]
    sel = fe.AgglomerativeFeatureExpansion(
        task="regression", random_state=0, n_output_buckets=4, verbose=False
    )
    sel.fit(X, pd.Series(y))
    assert set(sel.rank_[:3]) == {"info0", "info1", "info2"}
    res = sel.select(plateau_tol=0.01, patience=2)
    assert res.k <= 4
