"""The hot-start problem for a *classifier*: PhiUSIIL, and the search around it.

The sibling of `problem.py`. Same contract — `x0`, `bounds`, `fitness`, `score`
— so `arms.py` and `budget.py` need no changes, and the same three questions get
asked of a classification problem that `problem.py` asks of a regression one.

Two things differ, and both come from the library rather than from a choice made
here.

**The objective is cross-entropy, not k-fold MSE.** `refine.py` ships two
refinement paths, and the classifier one optimizes
`_make_classifier_fitness` = training cross-entropy plus a ridge shrink toward
`x0`, scaled by each parameter's box width. That is imported, not reimplemented,
for the same reason the regression study imports `_make_kfold_fitness`: an
optimizer measured against a target the shipped code does not use is measuring
nothing anybody runs. The defaults come from `refine_classifier_antecedents`
(`l2_shrink=0.05`, `sigma_min_frac=0.02`, `sigma_max_frac=1.0`), including
`guard="none"`, under which the search sees all the training rows.

Note what that means for reading the results: the objective is a **training**
loss. It has a shrinkage term rather than a held-out fold to keep it honest, so
the gap between "drove the objective down" and "generalized" is if anything
wider here than on Concrete, and held-out accuracy is the only outcome to quote.

**The rule count is not a free parameter.** The MoG construction gives one rule
per class, so a binary problem has two rules and there is nothing to sweep. What
is free is the number of components per (feature, class), which the construction
chooses by BIC.

## Train and test sizes are set independently, on purpose

PhiUSIIL is 235,795 rows and it is **saturated** — the construction makes about
two or three test errors in ten thousand. That puts opposite pressures on the two
splits:

* **Training** rows set the cost of an objective evaluation, and the study spends
  thousands of them per arm per seed. Large is unaffordable.
* **Test** rows set the resolution of the accuracy column. On a 4,000-row test
  set one error moves accuracy by 0.00025 and the construction makes one or two,
  so the column cannot tell two good models apart at all.

So `n_train` and `n_test` are separate arguments over one stratified split of a
pool of `n_train + n_test` rows: a small, cheap training set and a test set large
enough that the accuracy column means something. Both are recorded in the
archive, and the accompanying table reports **error counts** as well as rates,
because a rate with two events behind it invites over-reading.

This is not the invisible 20,000-row cap that `fit_gaussians` used to apply —
that lived inside the fit and was unreported. This is the study's own sampling,
applied identically to every arm, and it bounds what the numbers mean rather than
distorting a comparison between them.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "reproduce"))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))

L2_SHRINK = 0.05  # refine_classifier_antecedents default
SIGMA_MIN_FRAC = 0.02  # ditto
SIGMA_MAX_FRAC = 1.0  # ditto


@dataclass
class ClassifierHotStart:
    """One (sample size, seed) instance, shaped like `problem.HotStartProblem`."""

    dataset: str
    seed: int
    x0: np.ndarray
    bounds: list
    fitness: object
    score: object  # vec -> (accuracy, error_rate)
    n_params: int
    radius: float
    init: str = "hot"
    heuristic_x: object = None
    heuristic_score: float = float("nan")
    heuristic_obj: float = float("nan")
    meta: dict = field(default_factory=dict)

    # `problem.py`'s names, so `arms.py` and `run_*` need no special-casing.
    @property
    def heuristic_r2(self):
        return self.heuristic_score

    @property
    def heuristic_cv(self):
        return self.heuristic_obj

    @property
    def lower(self):
        return np.array([b[0] for b in self.bounds])

    @property
    def upper(self):
        return np.array([b[1] for b in self.bounds])


def _trust_region(bounds, x0, radius):
    """Shrink each interval toward x0. Identical to `problem._trust_region`."""
    if radius >= 1.0:
        return list(bounds)
    out = []
    for (lo, hi), c in zip(bounds, x0):
        half = 0.5 * radius * (hi - lo)
        out.append((max(lo, c - half), min(hi, c + half)))
    return out


def _scorer(model, X_te, y_te, norms):
    """vec -> (test accuracy, test error rate).

    A factory for the same reason `problem._scorer` is one: the reference score
    must stay bound to the construction's own model even when an arm's init
    replaces the model with one of a different length.
    """
    from tribblefis.refine import _classifier_accuracy, apply_gaussian_params

    def score(vec):
        candidate = apply_gaussian_params(model, np.asarray(vec, dtype=float))
        try:
            acc = float(_classifier_accuracy(X_te, y_te, candidate, norms))
        except Exception:  # noqa: BLE001
            return float("nan"), float("nan")
        return acc, 1.0 - acc

    return score


_CACHE = {}


def build(
    seed=0,
    radius=1.0,
    n_train=16_000,
    n_test=48_000,
    top_n=10,
    init="hot",
    components=None,
):
    """Fit the construction on PhiUSIIL and return everything the arms share.

    `init`:
      "hot"       the construction's own antecedents — the tribble-fis result;
      "cold"      a uniform random draw inside the same box;
      "classical-kmeans" / "classical-fcm"
                  the pre-construction route for classification: cluster within
                  each class, one Gaussian per cluster per feature. This
                  REPLACES the construction's placement rather than perturbing
                  it, which is what makes it a fair venue for a timing
                  comparison. See `phishing.classical`.

    Everything else — the features, the split, the box, the objective — is held
    identical, so the only difference is where the search starts.
    """
    key = (seed, radius, n_train, n_test, top_n, init, components)
    if key in _CACHE:
        return _CACHE[key]

    from sklearn.model_selection import train_test_split
    from tribblefis.refine import (
        _make_classifier_fitness,
        build_param_bounds,
        extract_gaussian_params,
        resolve_norm_pair,
    )

    import phishing as P

    X, y = P.load(n_train + n_test)
    # An exact count, not a fraction: the two splits are sized for different
    # reasons (see the module docstring) and a fraction would couple them.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=min(n_train, len(X) - 1), random_state=seed, stratify=y
    )

    # Feature engineering, timed separately and charged to nobody: every init
    # below consumes the same selected feature set, so folding the screen into
    # one of them would compare a pipeline against a training step. Addendum 2
    # of RESULTS_2026-08-02.md is why this is a separate number.
    features, screen_seconds = P.screen(Xtr, ytr, top_n)
    Xtr, Xte = Xtr[features], Xte[features]

    # Timed: the construction every init is measured against. This is the number
    # the "how much faster" claim rests on, so it is measured on its own and
    # reported on its own.
    t0 = time.perf_counter()
    model, _inner = P.construction(Xtr, ytr, features, n_gaussians=components or -1)
    construction_seconds = time.perf_counter() - t0

    norms = resolve_norm_pair()
    construction_bounds = build_param_bounds(model, Xtr, SIGMA_MIN_FRAC, SIGMA_MAX_FRAC)
    c_lo = np.array([b[0] for b in construction_bounds])
    c_hi = np.array([b[1] for b in construction_bounds])
    heuristic = np.clip(extract_gaussian_params(model), c_lo, c_hi)

    # The construction's own reference numbers, captured against the
    # CONSTRUCTION's model before any init can replace it.
    reference = _scorer(model, Xte, yte, norms)
    heuristic_acc, _err = reference(heuristic)
    heuristic_fitness = _make_classifier_fitness(
        model, Xtr, ytr, L2_SHRINK, heuristic, c_lo, c_hi, norms
    )
    heuristic_obj = float(heuristic_fitness(heuristic))

    init_seconds = 0.0
    if init.startswith("classical-"):
        method = init.split("-", 1)[1]
        t0 = time.perf_counter()
        model, _inner = P.classical(Xtr, ytr, features, components or 3, method, seed)
        init_seconds = time.perf_counter() - t0

    bounds_full = build_param_bounds(model, Xtr, SIGMA_MIN_FRAC, SIGMA_MAX_FRAC)
    lo = np.array([b[0] for b in bounds_full])
    hi = np.array([b[1] for b in bounds_full])
    x0 = np.clip(extract_gaussian_params(model), lo, hi)

    if init == "cold":
        rng = np.random.default_rng(seed)
        x0 = lo + rng.random(len(lo)) * (hi - lo)

    bounds = _trust_region(list(zip(lo, hi)), x0, radius)
    b_lo = np.array([b[0] for b in bounds])
    b_hi = np.array([b[1] for b in bounds])

    # The objective the arm optimizes: the shipped classifier fitness, shrunk
    # toward this arm's OWN x0 -- which is what the shipped path does, and which
    # keeps `cold` from being penalised for starting somewhere the construction
    # did not.
    fitness = _make_classifier_fitness(
        model, Xtr, ytr, L2_SHRINK, x0, b_lo, b_hi, norms
    )
    score = _scorer(model, Xte, yte, norms)

    prob = ClassifierHotStart(
        dataset="phiusiil",
        seed=seed,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        fitness=fitness,
        score=score,
        n_params=len(x0),
        radius=radius,
        init=init,
        heuristic_x=heuristic,
        heuristic_score=heuristic_acc,
        heuristic_obj=heuristic_obj,
        meta={
            "construction_seconds": construction_seconds,
            "init_seconds": init_seconds,
            "screen_seconds": screen_seconds,
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "n_features": len(features),
            "features": list(features),
            "n_mfs": P.n_membership_fns(model),
        },
    )
    _CACHE[key] = prob
    return prob
