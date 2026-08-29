#!/usr/bin/env python3
"""Agglomerative feature *expansion*: the smallest correlation-ranked feature
set that is good enough, found in O(log k) fits instead of O(M).

Motivation
----------
`mog_top_p_sweep.py` sweeps a *threshold* (`top_p`) and reads the feature count
back out; `table_a1_feature_scoring.py` sweeps a *fixed grid* of feature counts
(1, 2, 3, 5, 7, ...) and fits at every one. Both spend a fit on every point they
report. When the question is only "what is the smallest model that still clears
the bar?", most of those fits are wasted -- on PhiUSIIL the answer is *one*
feature (Table A.2), and a linear scan pays for twenty fits to discover it.

This module answers that question directly, reusing the two pieces the rest of
`regression_scale/` already relies on:

  1. **Agglomerative decorrelation** (same construction as
     `mog_top_p_sweep.decorrelate`): `sklearn.cluster.FeatureAgglomeration`
     collapses each cluster of mutually redundant features to a single named
     representative -- here the one with the highest differentiation score, so
     the survivor is the most *useful* member of its cluster, not an arbitrary
     one. This is the "agglomerative" half.

  2. **Differentiation-score ranking** (`gauss_math.calculate_gaussian_correlation`,
     the "correlation selection approach" the classifier and regressor both use
     internally): orders the survivors best-first.

The "expansion" half then walks the *nested* prefixes of that ranking --
top-1 features subset of top-2 subset of top-3 ... -- and, because the ranking
is fixed once and the prefixes are nested, the accuracy-vs-k curve is monotone
enough to search rather than scan:

  * **target mode** (`select(target=...)`): galloping bracket + bisection finds
    the smallest k whose score reaches the target. ~log2(k*) + log2(bracket)
    fits instead of k* of them.
  * **plateau mode** (`select()` with no target): expand one feature at a time,
    stop once `patience` consecutive additions each buy less than `plateau_tol`,
    then report the *knee* -- the smallest k already within `plateau_tol` of the
    best score seen. Early-stops as soon as the curve flattens.

Every evaluated k is cached, so the two search strategies -- and repeated
`select()` calls at different targets -- never re-fit a k they have already seen.

Nesting is what makes the search valid, so the ranking must be fixed *before*
the search: it is computed once, on the training split only, and the same
ranking scores every candidate k. That keeps the selection honest (no test-set
leakage into the ranking) and the prefixes genuinely nested (no re-ranking per
k). The curve is not guaranteed strictly monotone -- a later feature can add
noise -- so galloping/bisection find the smallest k under a near-monotone
assumption; `select(..., verify_scan=True)` falls back to a full scan and
raises `AssertionError` if the searched answer disagrees with it -- the
smallest passing k in target mode, the knee in plateau mode -- for the datasets
where that assumption needs checking.

Run the built-in demonstration from the repo root (needs the `tribble-fis`
submodule checked out, or `PILOT_TRIBBLE_FIS` pointed at a clone):
    uv run --project tribble-fis python reproduce/regression_scale/feature_expansion.py

Knobs (demonstration only; the class itself is configured by argument):
    FE_SEED=0            train/test split + model seed
    FE_PHIUSIIL_N=20000  PhiUSIIL sample size for the classification demo
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

FIS_ROOT = os.environ.get("PILOT_TRIBBLE_FIS", os.path.join(REPO_ROOT, "tribble-fis"))
sys.path.insert(0, os.path.join(FIS_ROOT, "src"))


# --------------------------------------------------------------------------- #
# preprocessing -- unit scaling + agglomerative decorrelation                 #
# --------------------------------------------------------------------------- #
def normalize(X, log_dynamic_range=2):
    """Unit-scale (with auto log1p on wide-dynamic-range columns), identical in
    construction to `mog_top_p_sweep.normalize`. Kept as a small local copy
    rather than an import so this module does not drag in that file's
    dataset-fetching side effects at import time."""
    from tribblefis.scaling import UnitScalar

    sc = UnitScalar(log_dynamic_range=log_dynamic_range)
    return pd.DataFrame(sc.fit_transform(X.copy()), index=X.index, columns=X.columns)


def agglomerate(X, scores, corr_threshold=0.9):
    """Collapse each cluster of mutually correlated features to one named
    representative: the member with the highest differentiation `score`.

    Squared Euclidean distance between mean-centered, unit-L2-norm columns is
    exactly `2 * (1 - corr)`, so a correlation threshold converts directly to a
    `FeatureAgglomeration` `distance_threshold` -- the same identity, and the
    same unit-*norm* (not unit-variance) normalization bug fix, documented in
    `mog_top_p_sweep.decorrelate`. The difference here is only the survivor
    rule: this keeps the most *differentiating* member so the downstream
    ranking is unchanged by the merge, rather than the one most correlated with
    a (bucketed) target.

    Returns
    -------
    (kept, dropped) : (list[str], dict[str, list[str]])
        Surviving representatives, and for each a list of the features it
        absorbed (empty when it stood alone).
    """
    if X.shape[1] < 2:
        return list(X.columns), {c: [] for c in X.columns}
    Xc = X - X.mean()
    norms = np.linalg.norm(Xc.values, axis=0)
    # A constant column has zero norm; it carries no correlation to anything, so
    # leave it on its own axis rather than dividing by zero.
    norms[norms == 0] = 1.0
    Xu = Xc / norms
    agg = FeatureAgglomeration(
        n_clusters=None,
        distance_threshold=np.sqrt(2 * (1 - corr_threshold)),
        metric="euclidean",
        linkage="average",
    )
    agg.fit(Xu.values)

    kept, dropped = [], {}
    for cluster_id in np.unique(agg.labels_):
        members = list(X.columns[agg.labels_ == cluster_id])
        rep = max(members, key=lambda c: scores.get(c, -np.inf))
        kept.append(rep)
        dropped[rep] = [m for m in members if m != rep]
    # Preserve differentiation-score order among the survivors.
    kept.sort(key=lambda c: scores.get(c, -np.inf), reverse=True)
    return kept, dropped


# --------------------------------------------------------------------------- #
# result container                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class ExpansionResult:
    """Outcome of one `select()` call."""

    k: int
    features: list
    score: float
    mode: str  # "target" | "plateau"
    reached_target: bool
    trace: list = field(default_factory=list)  # per-k dicts, in evaluation order
    n_evaluations: int = 0  # distinct k's actually fitted
    n_candidates: int = 0  # k's a full scan would have fitted

    @property
    def savings(self) -> str:
        if not self.n_candidates:
            return "n/a"
        return f"{self.n_evaluations}/{self.n_candidates} fits"


# --------------------------------------------------------------------------- #
# the selector                                                                #
# --------------------------------------------------------------------------- #
class AgglomerativeFeatureExpansion:
    """Rank features by differentiation score, optionally decorrelate first,
    then find the smallest top-k prefix that is good enough.

    Parameters
    ----------
    task : {"classification", "regression"}
        Chooses the default model, the default metric (accuracy / R^2), and
        whether the target is bucketed before ranking (regression buckets y with
        `regression.partition_output`, mirroring `TribbleRegressor.fit`).
    model_factory : callable(seed) -> estimator, optional
        Builds a fresh estimator configured to use *all* the columns it is
        handed (the selector controls which columns those are). Defaults to a
        Tribble classifier/regressor with `top_n=-1, top_p=1.0`.
    method : str
        Differentiation metric passed to `calculate_gaussian_correlation`.
    decorrelate : bool
        Run agglomerative decorrelation before ranking.
    corr_threshold : float
        Correlation above which two features are merged (see `agglomerate`).
    normalize_X : bool
        Unit-scale features before ranking/fitting.
    test_size, random_state : holdout split controls.
    n_repeats : int
        Refits per k, averaged, with the *ranking held fixed* (varying only the
        model seed). >1 smooths a noisy metric without breaking prefix nesting.
    n_output_buckets : int
        Regression-only: bucket count for ranking and for the default regressor.
    max_features : int, optional
        Cap on k (defaults to the number of features surviving decorrelation).
    """

    def __init__(
        self,
        task="classification",
        model_factory=None,
        method="wasserstein",
        decorrelate=True,
        corr_threshold=0.9,
        normalize_X=True,
        log_dynamic_range=2,
        test_size=0.2,
        random_state=0,
        n_repeats=1,
        n_output_buckets=3,
        max_features=None,
        metric=None,
        greater_is_better=None,
        verbose=True,
    ):
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be classification|regression, got {task!r}")
        self.task = task
        self.model_factory = model_factory
        self.method = method
        self.decorrelate = decorrelate
        self.corr_threshold = corr_threshold
        self.normalize_X = normalize_X
        self.log_dynamic_range = log_dynamic_range
        self.test_size = test_size
        self.random_state = random_state
        self.n_repeats = n_repeats
        self.n_output_buckets = n_output_buckets
        self.max_features = max_features
        self.metric = metric or (
            accuracy_score if task == "classification" else r2_score
        )
        self.greater_is_better = (
            greater_is_better if greater_is_better is not None else True
        )
        self.verbose = verbose

    # -- model + metric ---------------------------------------------------- #
    def _default_model(self, seed):
        if self.task == "classification":
            from tribblefis.gaussian_classifier import TribbleClassifier

            return TribbleClassifier(top_n=-1, top_p=1.0, random_state=seed)
        import tribblefis.gaussian_regressor as _gr

        Reg = (
            getattr(_gr, "TribbleRegressor", None)
            or _gr.MixtureOfGaussiansFuzzyRegressor
        )
        return Reg(
            n_output_buckets=self.n_output_buckets,
            tsk_order="1st",
            top_n=-1,
            top_p=1.0,
            random_state=seed,
        )

    def _make_model(self, seed):
        return (
            self.model_factory(seed)
            if self.model_factory
            else self._default_model(seed)
        )

    def _better(self, a, b):
        return a > b if self.greater_is_better else a < b

    def _passes(self, score, target):
        """Whether `score` clears `target`, respecting the optimization sense."""
        return score >= target if self.greater_is_better else score <= target

    def _knee(self, plateau_tol):
        """Smallest cached k whose score is within `plateau_tol` of the best
        score in the cache. Shared by the plateau search and its verify, so the
        searched knee and the full-scan knee are computed by identical logic."""
        best = None
        for r in self._cache.values():
            if best is None or self._better(r["score"], best):
                best = r["score"]
        threshold = best - plateau_tol if self.greater_is_better else best + plateau_tol
        for kk in sorted(self._cache):
            if not self._better(threshold, self._cache[kk]["score"]):
                return kk
        return sorted(self._cache)[-1]

    # -- ranking ----------------------------------------------------------- #
    def _rank(self, X, y):
        import tribblefis.gauss_math as gm

        if self.task == "regression":
            from tribblefis.regression import partition_output

            y_part, _ = partition_output(self.n_output_buckets, y)
            y_for_rank = y_part["y_bucket"]
        else:
            y_for_rank = pd.Series(np.asarray(y).ravel(), index=X.index)
        with contextlib.redirect_stdout(io.StringIO()):
            ranked = gm.calculate_gaussian_correlation(
                X, y_for_rank, method=self.method
            )
        return ranked  # list[(feature, score)] descending

    # -- prepare: split, rank, decorrelate --------------------------------- #
    def fit(self, X, y):
        """Split, rank on the training half, and (optionally) decorrelate.
        Populates the ranking; does not search -- call `select()` for that."""
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y).ravel())
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.task == "classification" else None,
        )
        if self.normalize_X:
            Xtr = normalize(Xtr, log_dynamic_range=self.log_dynamic_range)
            # Apply the *training* scaler's construction to test independently is
            # overkill for a per-column min-max used only to rank+fit a holdout
            # estimate; scale the test split on its own statistics, matching the
            # pilot's per-split treatment.
            Xte = normalize(Xte, log_dynamic_range=self.log_dynamic_range)

        ranked_full = self._rank(Xtr, ytr)
        scores = {f: s for f, s in ranked_full}

        if self.decorrelate:
            kept, dropped = agglomerate(
                Xtr[[f for f, _ in ranked_full]], scores, self.corr_threshold
            )
        else:
            kept = [f for f, _ in ranked_full]
            dropped = {f: [] for f in kept}

        self.Xtr_, self.Xte_, self.ytr_, self.yte_ = Xtr, Xte, ytr, yte
        self.rank_ = kept
        self.rank_scores_ = scores
        self.dropped_ = dropped
        self.n_features_in_ = X.shape[1]
        self.n_features_survived_ = len(kept)
        self._cache = {}
        if self.verbose:
            merged = sum(len(v) for v in dropped.values())
            print(
                f"  ranked {X.shape[1]} features; agglomeration "
                f"{'on' if self.decorrelate else 'off'} kept {len(kept)} "
                f"(merged away {merged}). top-5: {kept[:5]}"
            )
        return self

    # -- evaluate one k (cached) ------------------------------------------- #
    def _evaluate(self, k):
        k = int(k)
        if k in self._cache:
            return self._cache[k]
        feats = self.rank_[:k]
        scores, secs = [], []
        for r in range(self.n_repeats):
            seed = self.random_state + r
            model = self._make_model(seed)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                model.fit(self.Xtr_[feats], self.ytr_)
                pred = np.asarray(model.predict(self.Xte_[feats]))
            secs.append(time.perf_counter() - t0)
            scores.append(self.metric(self.yte_, pred))
        rec = {
            "k": k,
            "features": feats,
            "score": float(np.mean(scores)),
            "fit_s": float(np.mean(secs)),
        }
        self._cache[k] = rec
        if self.verbose:
            print(f"    k={k:<3} score={rec['score']:.4f}  fit={rec['fit_s']:.2f}s")
        return rec

    def _max_k(self):
        m = self.n_features_survived_
        return min(self.max_features, m) if self.max_features else m

    def _best_cached_k(self):
        """Smallest k achieving the best score seen so far -- ties broken toward
        the smaller model, since the whole point is the smallest good model."""
        best = max(
            r["score"] if self.greater_is_better else -r["score"]
            for r in self._cache.values()
        )
        target = best if self.greater_is_better else -best
        return min(
            k
            for k, r in self._cache.items()
            if (r["score"] if self.greater_is_better else -r["score"]) == target
        )

    # -- target mode: galloping bracket + bisection ------------------------ #
    def _search_target(self, target):
        M = self._max_k()

        def passes(k):
            return self._passes(self._evaluate(k)["score"], target)

        # Gallop: 1, 2, 4, ... until one passes or we run out of features.
        hi = 1
        while hi < M and not passes(hi):
            hi = min(hi * 2, M)
        if not passes(hi):
            # Never reaches the target, even with every feature: caller picks the
            # best k we saw (recomputed after any verify_scan), flagged as unmet.
            return self._best_cached_k(), False
        # Bisect the (lo, hi] bracket for the smallest passing k. lo is the last
        # power-of-two probe that failed (0 if k=1 already passed).
        lo = 0 if hi == 1 else hi // 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if passes(mid):
                hi = mid
            else:
                lo = mid
        return hi, True

    # -- plateau mode: expand until flat, report the knee ------------------ #
    def _search_plateau(self, plateau_tol, patience):
        M = self._max_k()
        best_score = -np.inf if self.greater_is_better else np.inf
        stale = 0
        k = 1
        while k <= M:
            s = self._evaluate(k)["score"]
            improved = self._better(
                s,
                best_score + (plateau_tol if self.greater_is_better else -plateau_tol),
            )
            if self._better(s, best_score):
                best_score = s
            stale = 0 if improved else stale + 1
            if stale >= patience:
                break
            k += 1
        # Knee: smallest evaluated k already within tol of the best score seen.
        return self._knee(plateau_tol), True

    # -- public search ----------------------------------------------------- #
    def select(self, target=None, plateau_tol=1e-3, patience=2, verify_scan=False):
        """Find the smallest good-enough k.

        target given  -> smallest k reaching it (galloping + bisection).
        target None   -> knee of the plateau (expand-until-flat).

        `verify_scan=True` additionally fits every k up to the cap and raises
        `AssertionError` if the searched answer disagrees with the full scan --
        the smallest passing k in target mode, the knee in plateau mode -- a
        guard for datasets whose score-vs-k curve may not be monotone enough for
        the search. It verifies both modes; an unreachable target has no passing
        k to check and is reported (reached=False) rather than raising.
        """
        if not hasattr(self, "rank_"):
            raise RuntimeError("call fit(X, y) before select()")
        mode = "target" if target is not None else "plateau"
        if self.verbose:
            print(
                f"  [{mode}] search"
                + (
                    f" for score {'>=' if self.greater_is_better else '<='} {target}"
                    if target is not None
                    else f" (plateau_tol={plateau_tol}, patience={patience})"
                )
            )

        n_before = len(self._cache)
        if target is not None:
            k, reached = self._search_target(target)
        else:
            k, reached = self._search_plateau(plateau_tol, patience)
        # Fits actually performed *this call*: new cache entries only. A k that
        # was already cached (from a prior select() on the same object) is a
        # free hit, so a fully-cached search honestly reports 0 new fits.
        n_eval_search = len(self._cache) - n_before

        if verify_scan:
            self._verify_against_scan(k, target, plateau_tol, patience)
        if not reached:
            # Target unmet: report the genuine best across everything evaluated
            # (including verify_scan), not just the points the gallop happened
            # to probe.
            k = self._best_cached_k()

        rec = self._cache[k]
        trace = [self._cache[kk] for kk in sorted(self._cache)]
        result = ExpansionResult(
            k=k,
            features=rec["features"],
            score=rec["score"],
            mode=mode,
            reached_target=reached,
            trace=trace,
            n_evaluations=n_eval_search,
            n_candidates=self._max_k(),
        )
        if self.verbose:
            tail = "" if reached else "  (target NOT reached; best k reported)"
            print(
                f"  -> k={result.k}  score={result.score:.4f}  "
                f"[{result.savings}]{tail}\n     features: {result.features}"
            )
        return result

    def _verify_against_scan(self, k_found, target, plateau_tol, patience):
        """Fit every k and confirm the searched answer equals the full-scan
        answer, raising on disagreement. Covers both modes: the smallest passing
        k in target mode, the knee in plateau mode. `AssertionError` is the
        contract `verify_scan=True` advertises -- an opt-in hard guarantee for
        datasets whose score-vs-k curve may not be monotone enough for the
        search, so a mismatch must fail loudly rather than warn."""
        for kk in range(1, self._max_k() + 1):
            self._evaluate(kk)
        if target is not None:
            passing = [
                kk
                for kk in range(1, self._max_k() + 1)
                if self._passes(self._cache[kk]["score"], target)
            ]
            # No passing k means the target is unreachable; the search already
            # reports reached=False and the best k, so there is nothing to check.
            true_k = min(passing) if passing else None
            if true_k is not None and true_k != k_found:
                raise AssertionError(
                    f"verify_scan: bisection returned k={k_found} but the full "
                    f"scan's smallest passing k is {true_k} (non-monotone curve)"
                )
            if self.verbose:
                print(f"  [verify] full scan agrees: smallest passing k = {true_k}")
        else:
            true_k = self._knee(plateau_tol)
            if true_k != k_found:
                raise AssertionError(
                    f"verify_scan: plateau search returned knee k={k_found} but "
                    f"the full-scan knee is k={true_k} (non-monotone curve)"
                )
            if self.verbose:
                print(f"  [verify] full scan agrees: plateau knee k = {true_k}")


# --------------------------------------------------------------------------- #
# demonstration                                                               #
# --------------------------------------------------------------------------- #
def _demo_synthetic():
    """Runs anywhere -- no external data. Three informative features buried in
    twenty, two of them near-duplicates so agglomeration has something to do."""
    print("\n" + "#" * 78)
    print("# Synthetic: 3 informative features (2 near-duplicate) among 20")
    print("#" * 78)
    rng = np.random.default_rng(0)
    n = 1500
    base = rng.normal(size=(n, 3))
    cols = {f"info{i}": base[:, i] for i in range(3)}
    cols["info0_dup"] = base[:, 0] + 0.01 * rng.normal(size=n)  # ~1.0 correlated
    for j in range(16):
        cols[f"noise{j}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    y = (base[:, 0] * 1.5 + base[:, 1] - 0.8 * base[:, 2] > 0).astype(int)

    sel = AgglomerativeFeatureExpansion(task="classification", random_state=0)
    sel.fit(X, y)
    sel.select(target=0.90, verify_scan=True)
    print("  (plateau mode, same ranking, cache reused:)")
    sel.select(plateau_tol=0.005, patience=2)


def _demo_synthetic_regression():
    """Regression twin of `_demo_synthetic`, so both task paths are exercised
    with no network: y is a smooth function of three features (two near-
    duplicate) buried in twenty, and plateau detection should stop at the knee."""
    print("\n" + "#" * 78)
    print("# Synthetic regression: 3 informative features (2 near-duplicate) among 20")
    print("#" * 78)
    rng = np.random.default_rng(1)
    n = 1500
    base = rng.normal(size=(n, 3))
    cols = {f"info{i}": base[:, i] for i in range(3)}
    cols["info0_dup"] = base[:, 0] + 0.01 * rng.normal(size=n)
    for j in range(16):
        cols[f"noise{j}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    y = (
        2.0 * base[:, 0]
        - 1.3 * base[:, 1]
        + 0.7 * base[:, 2]
        + 0.1 * rng.normal(size=n)
    )

    sel = AgglomerativeFeatureExpansion(
        task="regression", random_state=0, n_output_buckets=4
    )
    sel.fit(X, pd.Series(y))
    sel.select(plateau_tol=0.01, patience=2, verify_scan=True)


def _demo_phiusiil():
    print("\n" + "#" * 78)
    print("# PhiUSIIL (classification): target-mode bisection")
    print("#" * 78)
    sys.path.insert(0, os.path.join(REPO_ROOT, "reproduce", "tables"))
    try:
        import _fuzzy_models as F  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        print(f"  [skipped: cannot import table helpers: {exc}]")
        return
    n = int(os.environ.get("FE_PHIUSIIL_N", "20000"))
    data = F.load_phiusiil(sample_size=n)
    if data is None:
        print("  [skipped: PhiUSIIL data unavailable]")
        return
    X, y = data
    # PhiUSIIL ships label-leaking features -- URLSimilarityIndex is a URL's
    # similarity to a whitelist of known-legit URLs, i.e. the answer in disguise
    # (plus two legitimacy-derived probabilities). Drop them before selecting so
    # this demo does not showcase finding a leak; with them in, the "smallest
    # good-enough" set is just the leak. Once the loader drops these on load
    # (issue #215) this local guard becomes redundant.
    leak = ["URLSimilarityIndex", "TLDLegitimateProb", "URLCharProb"]
    dropped = [c for c in leak if c in X.columns]
    if dropped:
        print(f"  dropping {len(dropped)} label-leaking feature(s): {dropped}")
        X = X.drop(columns=dropped)
    seed = int(os.environ.get("FE_SEED", "0"))
    sel = AgglomerativeFeatureExpansion(task="classification", random_state=seed)
    sel.fit(X, y)
    sel.select(target=0.97, verify_scan=True)


def _demo_regression():
    print("\n" + "#" * 78)
    print("# California Housing (regression): plateau detection")
    print("#" * 78)
    import _datasets as D  # noqa: E402

    data = None
    for name, loader in (
        ("California Housing", D.load_housing),
        ("Superconductivity", D.load_superconduct),
    ):
        try:
            data = loader()
            print(f"  loaded {name}: X{data[0].shape}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name} unavailable: {type(exc).__name__}]")
    if data is None:
        print("  [skipped: no regression dataset reachable]")
        return
    X, y = data
    seed = int(os.environ.get("FE_SEED", "0"))
    sel = AgglomerativeFeatureExpansion(
        task="regression", random_state=seed, n_output_buckets=3
    )
    sel.fit(X, y)
    sel.select(plateau_tol=0.01, patience=2)
    sel.select(target=0.60)  # reuses the cache from the plateau scan


if __name__ == "__main__":
    print(
        f"tribble-fis commit: "
        f"{os.popen(f'git -C {FIS_ROOT} rev-parse --short HEAD').read().strip()}"
    )
    _demo_synthetic()
    _demo_synthetic_regression()
    _demo_phiusiil()
    _demo_regression()
