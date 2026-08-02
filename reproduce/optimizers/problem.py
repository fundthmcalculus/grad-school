"""The hot-start problem: a fitted MoG-TSK model, and the search around it.

Every arm in this study optimizes the *same* thing, and this module is what
guarantees it. One call to `build()` produces:

  * `x0`      -- the antecedent parameters of a heuristically-fitted model, i.e.
                 the tribble-fis result, which is the hot start;
  * `bounds`  -- the box the search may move in;
  * `fitness` -- k-fold held-out MSE, with the consequents re-solved in closed
                 form at every candidate;
  * `score`   -- test-set R^2 and RMSE for a parameter vector, which is the
                 number the chapters actually quote.

None of those are written here. They are `tribblefis.refine`'s own
`extract_gaussian_params`, `build_param_bounds` and `_make_kfold_fitness`, and
the model is built by the same three calls as
`reproduce/tables/table_concrete_reconciliation.py`. A study that reimplemented
the objective would be measuring optimizers against a target the shipped
refinement does not use.

**The trust region is the warm-start knob.** `radius=1.0` gives the full box
`build_param_bounds` returns, which is a cold start with a good initial guess
handed to whichever arms read one. Smaller radii shrink the box around `x0`, so
that *every* arm -- including the population methods, which sample their initial
population uniformly inside the bounds -- begins near the incumbent. It is the
most consequential hyperparameter in the study and it is swept, not chosen.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "reproduce"))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))


# Problems, smallest first. The study is meant to be walked up this list: get an
# answer on the cheap one, then spend the budget on the next.
#
# `n_params` is filled in at build time; the comment is the observed size, which
# is what makes Concrete the right first rung -- a hundred-odd free parameters is
# small enough to run ten seeds x seven arms in minutes.
DATASETS = {
    "concrete": {
        "kind": "reg",
        "note": "UCI Concrete, 1030 x 8 -> compressive strength. ~100 antecedent "
                "parameters. The rung §6.3.5's existing claim is measured on.",
    },
}


@dataclass
class HotStartProblem:
    """One (dataset, seed) instance, with everything the arms share."""
    dataset: str
    seed: int
    x0: np.ndarray
    bounds: list
    fitness: object
    score: object                      # vec -> (r2, rmse)
    n_params: int
    order: str
    radius: float
    meta: dict = field(default_factory=dict)

    @property
    def lower(self):
        return np.array([b[0] for b in self.bounds])

    @property
    def upper(self):
        return np.array([b[1] for b in self.bounds])


def _trust_region(bounds, x0, radius):
    """Shrink each interval toward x0, keeping it inside the original box.

    radius = 1.0 returns the original bounds unchanged. Smaller values keep x0
    strictly interior, so an arm that samples uniformly inside the box starts
    near the incumbent rather than anywhere in the feature's range.
    """
    if radius >= 1.0:
        return list(bounds)
    out = []
    for (lo, hi), c in zip(bounds, x0):
        half = 0.5 * radius * (hi - lo)
        out.append((max(lo, c - half), min(hi, c + half)))
    return out


_CACHE = {}


def build(dataset="concrete", seed=0, order="2nd", radius=1.0, n_folds=3,
          l2_reg=1e-2, test_size=0.2):
    """Fit the heuristic model and return everything the arms need.

    Cached on the full key. Every arm at a given seed must face the *identical*
    problem, and while the construction is deterministic anyway, refitting it
    once per arm costs a few seconds x arms x seeds for no benefit and leaves
    open the question of whether two arms really did get the same start.
    """
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; have {sorted(DATASETS)}")
    key = (dataset, seed, order, radius, n_folds, l2_reg, test_size)
    if key in _CACHE:
        return _CACHE[key]

    import table_concrete_reconciliation as TCR
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        take_top_features,
    )
    from tribblefis.refine import (
        _make_folds,
        _make_kfold_fitness,
        apply_gaussian_params,
        build_param_bounds,
        extract_gaussian_params,
    )
    from tribblefis.regression import predict_tsk, solve_tsk_consequents

    import _fuzzy_models as FM
    data = FM.load_concrete()
    if data is None:
        raise RuntimeError("Concrete is unavailable: no data/Concrete_Data.csv and "
                           "no reachable UCI mirror.")
    prep = TCR.prepare(*data)
    Xt, y = prep["Xt"], prep["y"]
    y_bucket_mean, span = prep["y_bucket_mean"], prep["span"]
    n_buckets = TCR.N_BUCKETS

    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=test_size,
                                          random_state=seed)

    diffs = calculate_gaussian_correlation(Xtr, ytr["y_bucket"])
    _, top_vars = take_top_features(diffs, top_n=len(Xt.columns))
    model = create_gaussian_membership_dict(Xtr, ytr["y_bucket"],
                                            top_n_var_names=top_vars,
                                            n_gaussians=-1)

    folds = _make_folds(len(Xtr), n_folds, test_size, seed)
    fitness = _make_kfold_fitness(model, Xtr, ytr, folds, top_vars, n_buckets,
                                  order, l2_reg, "raw", None)

    full_bounds = build_param_bounds(model, Xtr)
    x0 = np.clip(extract_gaussian_params(model),
                 [b[0] for b in full_bounds], [b[1] for b in full_bounds])
    bounds = _trust_region(full_bounds, x0, radius)

    def score(vec):
        """Test-set R^2 and RMSE (in MPa) for one antecedent vector.

        Consequents are re-solved on the training split for the candidate
        antecedents, exactly as `mog_arm` does -- an antecedent vector is not a
        model until its consequents are solved, and scoring it against stale
        consequents would flatter whichever arm moved least.
        """
        candidate = apply_gaussian_params(model, np.asarray(vec, dtype=float))
        corr, ybm = solve_tsk_consequents(
            Xtr, candidate, top_vars, y_bucket_mean, ytr,
            n_output_buckets=n_buckets, order=order, l2_reg=l2_reg,
            basis="raw", cross_pairs=None)
        pred = predict_tsk(Xte, candidate, top_vars, ybm, corr, order=order,
                           basis="raw", cross_pairs=None)
        truth = np.asarray(yte["y_value"], dtype=float).ravel()
        pred = np.asarray(pred, dtype=float).ravel()
        keep = ~np.isnan(pred)
        if not np.any(keep):
            return float("nan"), float("nan")
        rmse = float(np.sqrt(np.mean((truth[keep] - pred[keep]) ** 2))) * span
        return float(r2_score(truth[keep], pred[keep])), rmse

    problem = HotStartProblem(
        dataset=dataset, seed=seed, x0=x0, bounds=bounds, fitness=fitness,
        score=score, n_params=len(x0), order=order, radius=radius,
        meta={"n_train": len(Xtr), "n_test": len(Xte), "n_folds": n_folds,
              "n_buckets": n_buckets, "l2_reg": l2_reg,
              "logged": list(prep.get("logged") or [])},
    )
    _CACHE[key] = problem
    return problem
