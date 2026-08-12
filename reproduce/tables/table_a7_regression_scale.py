"""Appendix A.7.1 -- large-scale regression: Table 6.1's model family on
California Housing and UCI Superconductivity, ten seeds.

Promotes `reproduce/regression_scale/model_family_pilot.py` (single seed,
`RESULTS_2026-08-05.md`) to the `reproduce/tables/` ten-seed floor
(`common.SEEDS`), answering CHECKLIST C13's open question of whether either
candidate is worth the investment: yes, both are now measured properly.

Both loaders in `_datasets.py` pull from CANONICAL sources as of 2026-08-11
(re-verified reachable; the pilot's GitHub mirrors are no longer used):
  - California Housing: `sklearn.datasets.fetch_california_housing()` --
    built in, no mirror. This is the *derived* 8-feature version
    (`MedInc`/`AveRooms`/`AveBedrms`/`AveOccup`/...), not the raw StatLib
    file the old mirror served -- see `_datasets.py`'s docstring. Numbers
    here are therefore not bit-for-bit comparable to RESULTS_2026-08-05.md's
    California Housing row, though they land in the same range.
  - Superconductivity: `archive.ics.uci.edu/static/public/464/...zip`
    (UCI dataset id 464), downloaded directly. Same 21,263 x 81 shape as the
    mirror, so this dataset's numbers ARE the direct ten-seed successor to
    the pilot's single-seed row.

Same convention as `table_6_1_model_family.py`: model family = tribblefis +
fuzzytree, baselines = sklearn CART / Random Forest, optional M5. The "flat"
row is MoG at the library's UNTUNED default (top_p=0.95, top_n=-1) -- not a
tuned arm; mixing a tuned MoG into an otherwise-untuned table would repeat
the exact mismatch Table 4.1 was written to avoid.

California Housing fits on its 8 raw features (works fine untreated, per the
pilot: R2 = 0.660 at one seed). Superconductivity's 81 raw features are
heavily collinear and break the flat MoG's closed-form solve outright (pilot:
R2 = -0.644). Every model here instead fits Superconductivity AFTER
`sklearn.cluster.FeatureAgglomeration` decorrelation (corr threshold 0.9, one
named survivor per cluster -- same recipe as `mog_top_p_sweep.decorrelate`,
inlined below rather than imported since that module also imports a class
upstream has since renamed; see this file's `decorrelate()` docstring), the
fix the pilot found necessary (RESULTS_2026-08-05.md section 4). Decorrelation is fit ONCE on the full
81-feature frame (deterministic: `FeatureAgglomeration` + argmax have no
randomness), not per seed and not per training fold -- inherited unchanged
from the pilot's own `prepare()`. This means the surviving-feature choice per
cluster (`corr_y[members].idxmax()`) is informed by every row's label,
including whatever ends up in a given seed's test fold: a mild, structural
leak in *which column is kept*, not in what the model is fit on. Flagged
here, not fixed here -- fixing it (per-fold refitting of the cluster choice)
is a separate methodological change from "run the pilot's own recipe at ten
seeds," which is this table's job.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_a7_regression_scale.py

Knobs:
    REPRO_SEEDS="0"   quick one-seed smoke run (labeled loudly in the note)
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
_REPRODUCE = os.path.dirname(_TABLES)
sys.path.insert(0, _REPRODUCE)  # reproduce/  -> import common
sys.path.insert(0, _TABLES)  # reproduce/tables -> import _fuzzy_models
sys.path.insert(0, os.path.join(_REPRODUCE, "regression_scale"))  # -> _datasets

import common as C  # noqa: E402
import _fuzzy_models as _fm  # noqa: E402
import _datasets as RD  # noqa: E402

(M5Prime,) = C.optional_import("m5py", ["M5Prime"])


def decorrelate(X, y, corr_threshold=0.9):
    """One named feature survives per correlated cluster, picked by |corr
    with y| -- not averaged into a synthetic feature, so surviving columns
    stay meaningful in a rule base. Squared Euclidean distance between
    mean-centered, unit-L2-norm columns equals `2 * (1 - corr)` exactly, so a
    correlation threshold converts directly to `FeatureAgglomeration`'s
    `distance_threshold`.

    Inlined from `reproduce/regression_scale/mog_top_p_sweep.py`'s function
    of the same name (not imported: that module also imports
    `tribblefis.gaussian_regressor.MixtureOfGaussiansFuzzyRegressor` at
    import time, a class upstream renamed to `TribbleRegressor` since the
    pilot was written -- importing it here would tie this table's import to
    that pilot script's staleness rather than just its logic).
    """
    Xc = X - X.mean()
    Xu = Xc / np.linalg.norm(Xc.values, axis=0)
    dist_threshold = np.sqrt(2 * (1 - corr_threshold))
    agg = FeatureAgglomeration(
        n_clusters=None,
        distance_threshold=dist_threshold,
        metric="euclidean",
        linkage="average",
    )
    agg.fit(Xu.values)

    corr_y = X.corrwith(y).abs()
    kept = []
    for cluster_id in np.unique(agg.labels_):
        members = X.columns[agg.labels_ == cluster_id]
        kept.append(corr_y[members].idxmax())
    return X[kept]

MODEL_ORDER = ["flat", "tree", "hme", "cart", "rf", "m5"]
MODEL_LABELS = ["flat", "fuzzy tree", "mixture (HME)", "CART", "Random Forest", "M5"]


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def _fit_all(seed, X, y):
    """One seed's split + fit for every model in the family. Returns
    {model_key: (r2, rmse, fit_seconds) or None}."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    factories = {
        "flat": lambda: _fm.mog_regressor(seed),
        "tree": lambda: _fm.tree_regressor(seed),
        "hme": lambda: _fm.hme_regressor(seed),
        "cart": lambda: DecisionTreeRegressor(random_state=seed),
        "rf": lambda: RandomForestRegressor(n_estimators=200, random_state=seed),
        "m5": (lambda: M5Prime()) if M5Prime else None,
    }
    out = {}
    for key, factory in factories.items():
        if factory is None:
            out[key] = None
            continue
        model = factory()
        if model is None:
            out[key] = None
            continue
        t0 = time.perf_counter()
        try:
            if key == "m5":
                p = model.fit(np.asarray(Xtr), ytr).predict(np.asarray(Xte))
            else:
                p = model.fit(Xtr, ytr).predict(Xte)
            fit_s = time.perf_counter() - t0
            out[key] = (r2_score(yte, p), _rmse(yte, p), fit_s)
        except Exception as exc:  # noqa: BLE001
            print(f"    [{key}] FAILED: {exc.__class__.__name__}: {exc}")
            out[key] = None
    return out


def run_dataset(label, X, y, seeds):
    print(f"\n=== {label}: {X.shape[0]} rows x {X.shape[1]} features ===")
    r2 = {k: [] for k in MODEL_ORDER}
    rmse = {k: [] for k in MODEL_ORDER}
    fit_s = {k: [] for k in MODEL_ORDER}
    for seed in seeds:
        t_seed = time.perf_counter()
        results = _fit_all(seed, X, y)
        for k, res in results.items():
            if res is not None:
                r2[k].append(res[0])
                rmse[k].append(res[1])
                fit_s[k].append(res[2])
        print(f"  seed {seed}: {time.perf_counter() - t_seed:.1f}s")
    r2_row = [label, "R2", *[C.cell(r2[k]) for k in MODEL_ORDER]]
    rmse_row = [label, "RMSE", *[C.cell(rmse[k]) for k in MODEL_ORDER]]
    fit_row = [label, "fit (s)", *[C.cell(fit_s[k], fmt="{:.2f}") for k in MODEL_ORDER]]
    return [r2_row, rmse_row, fit_row]


def main():
    seeds = C.SEEDS
    print("Table A.7.1 -- large-scale regression model family")
    print(f"  seeds: {seeds}")

    rows = []

    X_house, y_house = RD.load_housing()
    rows += run_dataset("California Housing", X_house, y_house, seeds)

    X_sc, y_sc = RD.load_superconduct()
    before = X_sc.shape[1]
    X_sc_dec = decorrelate(X_sc, y_sc, corr_threshold=0.9)
    print(
        f"\n  [superconductivity] decorrelated (FeatureAgglomeration, |corr|<=0.9): "
        f"{before} -> {X_sc_dec.shape[1]} features"
    )
    rows += run_dataset("Superconductivity (decorrelated)", X_sc_dec, y_sc, seeds)

    header = ["Dataset", "metric", *MODEL_LABELS]
    C.emit(
        "table_a7_regression_scale",
        "Table A.7.1 -- Model family on large-scale regression datasets",
        header,
        rows,
        note=(
            "California Housing: sklearn.fetch_california_housing() (canonical, "
            "derived 8-feature version), raw features. Superconductivity: UCI id "
            "464 direct download, decorrelated to remove collinear features "
            "(FeatureAgglomeration, corr threshold 0.9) before every model in this "
            "row -- raw features break the flat MoG's closed-form solve (see "
            "reproduce/regression_scale/RESULTS_2026-08-05.md). Decorrelation is "
            "fit once on the full frame, not per seed/fold -- see this file's "
            "module docstring for the resulting leak in which column survives "
            "per cluster. 'flat' is MoG at the library's UNTUNED default "
            "(top_p=0.95); 'M5' is N/A where m5py is unavailable."
        ),
    )


if __name__ == "__main__":
    main()
