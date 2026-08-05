#!/usr/bin/env python3
"""Pilot: does `MixtureOfGaussiansFuzzyRegressor` hold up past Concrete's 1,030
rows? Single seed throughout -- this is deliberately cheaper than the ten-seed
floor `reproduce/tables/` generators hold to, because it exists to find out
whether either candidate dataset is even worth measuring properly before
paying for that. See RESULTS_2026-08-05.md for the full write-up, findings,
and what is NOT yet settled by this pilot.

Pipeline, in order, mirroring `reproduce/tables/table_4_1_mog_baselines.py`'s
own convention where it overlaps:
  1. `_fuzzy_models.normalize`-equivalent: log1p (auto-detected, >= N decades
     dynamic range) + min-max to [0,1].
  2. `sklearn.cluster.FeatureAgglomeration`: removes highly-correlated
     features -- pairwise redundancy, which `top_p` cannot see on its own,
     since it scores each feature independently.
  3. `top_p`: the library's own per-feature differentiation-score threshold,
     applied to whatever survives step 2.

Run from repo root (needs the `tribble-fis` submodule checked out, or
`PILOT_TRIBBLE_FIS` pointed at a clone of it):
    PILOT_TRIBBLE_FIS=/path/to/tribble-fis \\
        uv run --project tribble-fis python reproduce/regression_scale/mog_top_p_sweep.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import _datasets as D  # noqa: E402

FIS_ROOT = os.environ.get("PILOT_TRIBBLE_FIS", os.path.join(REPO_ROOT, "tribble-fis"))
sys.path.insert(0, os.path.join(FIS_ROOT, "src"))

from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor  # noqa: E402
from tribblefis.scaling import UnitScalar  # noqa: E402

SEED = 0


def normalize(X, log_dynamic_range=2, report=False):
    sc = UnitScalar(log_dynamic_range=log_dynamic_range)
    Xt = pd.DataFrame(sc.fit_transform(X.copy()), index=X.index, columns=X.columns)
    if report:
        print(f"  log1p-transformed ({len(sc.log_features_)}/{X.shape[1]} features, "
              f"dynamic range >= {log_dynamic_range} decades): {sc.log_features_}")
    return Xt


def decorrelate(X, y, corr_threshold=0.9):
    """One named feature survives per correlated cluster, picked by |corr
    with y| -- not averaged into a synthetic feature, which would defeat the
    reason this construction exists (a reader has to be able to read the rule
    base). Squared Euclidean distance between mean-centered, unit-L2-norm
    columns equals `2 * (1 - corr)` exactly, so a correlation threshold
    converts directly to `FeatureAgglomeration`'s `distance_threshold`.
    (An earlier version of this function z-scored to unit *variance* instead
    of unit *norm* -- off by a factor of ~sqrt(N-1), so it never merged
    anything. Fixed here; flagged in RESULTS_2026-08-05.md as a caught bug,
    not a silent correction.)
    """
    Xc = X - X.mean()
    Xu = Xc / np.linalg.norm(Xc.values, axis=0)
    dist_threshold = np.sqrt(2 * (1 - corr_threshold))
    agg = FeatureAgglomeration(n_clusters=None, distance_threshold=dist_threshold,
                                metric="euclidean", linkage="average")
    agg.fit(Xu.values)

    corr_y = X.corrwith(y).abs()
    kept = []
    for cluster_id in np.unique(agg.labels_):
        members = X.columns[agg.labels_ == cluster_id]
        kept.append(corr_y[members].idxmax())
    return X[kept]


def prepare(name, loader, corr_threshold=0.9, log_dynamic_range=2):
    print(f"\n=== {name} ===")
    X, y = loader()
    print(f"  raw shape: {X.shape}, y range [{y.min():.3g}, {y.max():.3g}]")
    X = normalize(X, log_dynamic_range=log_dynamic_range, report=True)

    before = X.shape[1]
    X = decorrelate(X, y, corr_threshold=corr_threshold)
    print(f"  decorrelated (sklearn FeatureAgglomeration, |corr| <= {corr_threshold}): "
          f"{before} -> {X.shape[1]} features -- {list(X.columns)}")
    return X, y


def fit_at(X, y, top_p, warm_up=False):
    """One fit at one top_p. The library prints a differentiation-score
    ranking on every fit with no verbose switch to silence it; suppressed
    here except on the warm-up call, once per dataset."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    def new_model():
        return MixtureOfGaussiansFuzzyRegressor(
            n_output_buckets=3, tsk_order="1st", top_n=-1, top_p=top_p,
            random_state=SEED)

    ctx = contextlib.nullcontext() if warm_up else contextlib.redirect_stdout(io.StringIO())
    with ctx:
        if warm_up:
            # Discarded warm-up fit, same split -- table_4_1's own convention,
            # so seed 0 doesn't pay import/JIT/BLAS spin-up cost as a measurement.
            _rng_state = np.random.get_state()
            try:
                new_model().fit(Xtr, ytr).predict(Xte)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warm-up failed: {exc.__class__.__name__}: {exc}]")
            finally:
                np.random.set_state(_rng_state)

        model = new_model()
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        p = np.asarray(model.predict(Xte))

    r2 = r2_score(yte, p)
    n_mf = getattr(model.model_, "n_membership_functions", None)
    n_kept = getattr(model, "top_n_actual_", None)
    return {"top_p": top_p, "n_features": n_kept, "fit_s": fit_s, "r2": r2, "n_mf": n_mf}


def sweep(name, loader, top_p_grid, corr_threshold=0.9, log_dynamic_range=2):
    X, y = prepare(name, loader, corr_threshold=corr_threshold,
                   log_dynamic_range=log_dynamic_range)
    rows = []
    for i, top_p in enumerate(top_p_grid):
        row = fit_at(X, y, top_p, warm_up=(i == 0))
        rows.append(row)
        print(f"  top_p={top_p:<6} features={row['n_features']:<4} "
              f"fit={row['fit_s']:.2f}s  R2={row['r2']:.4f}  MF={row['n_mf']}")
    return rows


if __name__ == "__main__":
    print(f"tribble-fis commit: {os.popen(f'git -C {FIS_ROOT} rev-parse HEAD').read().strip()}")

    print("\n" + "#" * 80)
    print("# California Housing: coarse top_p sweep")
    print("#" * 80)
    sweep("California Housing, decorrelated, top_p sweep",
          D.load_housing, [0.1, 0.25, 0.5, 0.75, 0.9, 1.0])

    print("\n" + "#" * 80)
    print("# Superconductivity: fine top_p sweep around the coarse-sweep peak")
    print("#" * 80)
    FINE_GRID = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28,
                 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.45, 0.50]
    sweep("Superconductivity, decorrelated, fine top_p sweep (log_dynamic_range=2)",
          D.load_superconduct, FINE_GRID, log_dynamic_range=2)
    sweep("Superconductivity, decorrelated, fine top_p sweep (log_dynamic_range=1)",
          D.load_superconduct, FINE_GRID, log_dynamic_range=1)
