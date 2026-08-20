"""A deterministic stand-in for wall-clock time, calibrated against wall-clock time.

The tuner's objective has to include runtime — the whole point of the strategy engine
is to spend less of it. But timing is a terrible thing to optimise against directly:

* it is noisy, so the search chases measurement error;
* it is not reproducible, so a tuning run cannot be repeated;
* and worst, it cannot be measured under parallel evaluation, because concurrent
  workers contend for the same cores. Timing the objective forces ``n_jobs=1`` and
  throws away most of the machine.

So instead the solver's own work counters — candidate evaluations, city scans, chain
levels entered, accepted moves, rule-base evaluations — are combined into a predicted
cost by a linear model fitted to real measured times. The fit is non-negative least
squares, because a counter cannot make the solver faster and a model that says it does
is fitting noise.

The model is calibrated once over many (instance, configuration) pairs, its quality is
reported rather than assumed, and the final benchmark still reports real wall clock.
This module is what the objective uses; it is not what the results are measured with.

Run:  python costmodel.py [--out costmodel.npz]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import fis
from core import build_candidates, greedy_edge_tour
from fis_lk import local_search as fis_ls
from lk import (
    STAT_CHAIN_CALLS,
    STAT_REV_WORK,
    STAT_DEPTH,
    STAT_EVALS,
    STAT_MOVES,
    STAT_SCANS,
    lk_solve,
)
from fis_lk import STAT_FIS_CALLS
from tsplib import load

import paths

# The counters the cost model is allowed to use, and the intercept. ``n`` is included
# because both arms pay an O(n) setup (pos array, queue seeding, final tour length)
# that no per-search counter sees.
FEATURES = (
    "scans",
    "evals",
    "chain_levels",
    "moves",
    "chain_fis",
    "effort_fis",
    "rev_work",
    "n",
)
N_FEAT = len(FEATURES)


def features_from_stats(stats, n):
    """The design-matrix row for one solver run.

    Accepts both arms' stats arrays. The baseline's is shorter — it has no
    EFFORT rule base to count — so that column reads zero for it, which is exactly
    the truth rather than a missing value.
    """
    effort_fis = (
        float(stats[STAT_FIS_CALLS]) if stats.shape[0] > STAT_FIS_CALLS else 0.0
    )
    return np.array(
        [
            float(stats[STAT_SCANS]),
            float(stats[STAT_EVALS]),
            float(stats[STAT_DEPTH]),
            float(stats[STAT_MOVES]),
            float(stats[STAT_CHAIN_CALLS]),
            effort_fis,
            float(stats[STAT_REV_WORK]),
            float(n),
        ],
        dtype=np.float64,
    )


# Instances used for calibration. A spread of sizes, because the intercept and the
# per-n term can only be separated by seeing both small and large instances.
CALIB = [
    "eil101",
    "ch150",
    "d198",
    "lin318",
    "pcb442",
    "rat575",
    "u724",
    "pr1002",
    "d1291",
    "u1817",
    "pr2392",
    "fnl4461",
    "rl5915",
    "rl11849",
]

# Configurations, chosen to move the counters as independently as the solver allows:
# depth mostly drives chain levels, deep breadth mostly drives evaluations.
CALIB_CONFIGS = [
    (32, 2, 4),
    (32, 4, 4),
    (32, 4, 32),
    (32, 6, 8),
    (32, 6, 32),
    (32, 10, 12),
    (32, 10, 32),
    (32, 16, 32),
    (16, 6, 16),
    (48, 10, 32),
]


def collect(reps=3, verbose=True):
    """(X, y) over baseline configurations and a couple of fuzzy ones.

    Both arms are sampled: the model has to predict the fuzzy arm's cost, and the
    fuzzy arm evaluates rule bases that the baseline never does (the ``fis_calls``
    column is identically zero on baseline rows, so without fuzzy rows its
    coefficient would be unidentifiable).
    """
    X = []
    y = []
    for name in CALIB:
        inst = load(name)
        for k in sorted({c[0] for c in CALIB_CONFIGS}):
            cand, cand_d = build_candidates(inst.coords, k, inst.ceil)
            start = greedy_edge_tour(inst.coords, cand, inst.ceil)
            for kk, depth, deep in CALIB_CONFIGS:
                if kk != k:
                    continue
                best = float("inf")
                for _ in range(reps):
                    t0 = time.perf_counter()
                    _, _, st = lk_solve(
                        inst.coords, cand, cand_d, inst.ceil, start, k, depth, deep, 3
                    )
                    best = min(best, time.perf_counter() - t0)
                X.append(features_from_stats(st, inst.n))
                y.append(best)
            # fuzzy rows, with and without the chain rule base
            cand, cand_d = build_candidates(inst.coords, 32, inst.ceil)
            start = greedy_edge_tour(inst.coords, cand, inst.ceil)
            for use_chain in (False, True):
                best = float("inf")
                for _ in range(reps):
                    t0 = time.perf_counter()
                    _, _, st = fis_ls(
                        inst,
                        cand,
                        cand_d,
                        start,
                        fis.EFFORT_CONS,
                        fis.CHAIN_CONS,
                        10,
                        3,
                        False,
                        use_chain,
                    )
                    best = min(best, time.perf_counter() - t0)
                X.append(features_from_stats(st, inst.n))
                y.append(best)
        if verbose:
            print(f"  calibrated on {name} (n={inst.n})", flush=True)
    return np.array(X), np.array(y)


def fit(X, y):
    """Non-negative least squares coefficients, and the fit's honest error stats.

    Fitted in *relative* space — each row divided by its own measured time, against a
    target of 1.0. Plain least squares here is actively wrong: the times span four
    orders of magnitude, so the residual is dominated by the largest instance and the
    fit buys accuracy there by over-predicting every small instance by more than 100%.
    Since the tuner sums predicted cost across a training set of mixed sizes, that
    would make the small instances' contribution meaningless.
    """
    w = 1.0 / np.maximum(y, 1e-12)
    coef, _ = nnls(X * w[:, None], np.ones_like(y))
    pred = X @ coef
    rel = (pred - y) / np.maximum(y, 1e-12)
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    stats = {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mean_abs_rel": float(np.mean(np.abs(rel))),
        "p90_abs_rel": float(np.percentile(np.abs(rel), 90)),
        "max_abs_rel": float(np.max(np.abs(rel))),
        "spearman": _spearman(pred, y),
        "n_samples": int(X.shape[0]),
    }
    return coef, stats


def _spearman(a, b):
    """Rank correlation — the property the tuner actually needs, since it only ever
    compares one candidate's cost against another's."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    return float(np.sum(ra * rb) / denom) if denom > 0 else float("nan")


def predict(coef, stats, n):
    return float(features_from_stats(stats, n) @ coef)


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=str(paths.COSTMODEL))
    args = ap.parse_args()

    # warm the JIT so compilation never lands inside a timed sample
    inst = load("berlin52")
    cand, cand_d = build_candidates(inst.coords, 32, inst.ceil)
    g = greedy_edge_tour(inst.coords, cand, inst.ceil)
    lk_solve(inst.coords, cand, cand_d, inst.ceil, g, 32, 10, 32, 3)
    for uc in (False, True):
        fis_ls(inst, cand, cand_d, g, fis.EFFORT_CONS, fis.CHAIN_CONS, 10, 3, False, uc)

    print("calibrating the cost model against measured wall clock")
    X, y = collect(args.reps)
    coef, stats = fit(X, y)
    print(f"\n{stats['n_samples']} samples")
    print(f"  R^2                 {stats['r2']:.5f}")
    print(f"  rank correlation    {stats['spearman']:.5f}")
    print(f"  mean |rel err|      {100 * stats['mean_abs_rel']:.2f}%")
    print(f"  p90  |rel err|      {100 * stats['p90_abs_rel']:.2f}%")
    print(f"  max  |rel err|      {100 * stats['max_abs_rel']:.2f}%")
    print("\n  ns per unit:")
    for f, c in zip(FEATURES, coef):
        print(f"    {f:>13s} {1e9 * c:9.2f}")
    np.savez(args.out, coef=coef, features=np.array(FEATURES), **stats)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
