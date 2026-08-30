"""TribbleTree TSK order sweep — quality vs. training time at order 0/1/2.

`research/proposal-defense/prose/02-background.md` draws the Mamdani/TSK line:
a Mamdani rule aggregates fuzzy output sets and defuzzifies; a TSK rule's
consequent is instead a function of the inputs -- constant (order 0), linear
(order 1), or polynomial (order 2+). TRIBBLE uses TSK throughout because, for
fixed firing strengths, the output is LINEAR in the consequent coefficients, so
the consequents solve in closed form. Order 0 collapses that consequent to a
constant per leaf/rule -- the one point where a TSK model degenerates to
something a Mamdani system already computes (a weighted constant output), which
is why it is the natural surrogate for "how would a Mamdani-style FIS have done
here." Orders 1 and 2 add a local linear/quadratic correction on top.

This sweeps `tsk_order` in {0th, 1st, 2nd} across the fuzzy-tree family --
`FuzzyRegressionTree` (the flat "TribbleTree" hierarchy) and
`HierarchicalFuzzyExpertsRegressor` (HME, whose LEAF EXPERTS take the same
`tsk_order`; the gate itself is a partition-of-unity product and is not a
sweepable axis -- see `hme.py`'s `compute_responsibilities` docstring) -- plus
the flat `TribbleRegressor` as a non-hierarchical reference row. Existing
tables fix this at "1st" (`table_6_1_model_family.py`,
`table_hyperparam_normalization.py`'s tree/HME arms) or sweep it without timing
it (`table_concrete_reconciliation.py`'s flat-MoG arms) -- see
`PROVENANCE_MAP.md` note 14, where a flat-MoG order-timing cell was left
`*pending*` for exactly this reason. Nothing here reuses those cells; this is a
new, separate measurement for the tree/HME family specifically.

WARM-UP. The same note documents a ±60% timing artifact: the first fit in a
process absorbs import, JIT and BLAS thread-pool spin-up, and looks like ±60%
seed noise until a throwaway fit is discarded before timing starts. This script
discards one warm-up fit per dataset before its seed loop for exactly that
reason -- see `_warm_up()`.

QUALITY is R2/RMSE on a held-out 20% split, identical splits/seeds shared by
every model+order cell for a given dataset. TRAINING TIME is wall-clock seconds
for `.fit()` alone (construction and `.predict()` are not timed). Hyperparameters
other than `tsk_order` are taken verbatim from `tribble-tree/demo_concrete.py`
(tree: max_depth=3, n_terms=2, top_n=4, min_soft_count=20; HME: max_depth=2,
n_gate_terms=2, top_n=4, min_soft_count=40, min_expert_samples=60), the same
provenance `table_hyperparam_normalization.py` documents for its own tree/HME
arms.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_tribbletree_tsk_order.py

Knobs:
    REPRO_SEEDS="0,1,2,3,4"           seeds (default: common.SEEDS, 0..9)
    REPRO_TSK_ORDERS="0th,1st,2nd"    which TSK orders to sweep
    REPRO_DATASETS="concrete,bodyfat" which regression datasets to include;
                                      add "bikeshare" for a ~17k-row arm (slow)
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))  # reproduce/  -> import common
sys.path.insert(0, _TABLES)  # reproduce/tables -> import _fuzzy_models
import common as C  # noqa: E402
import _fuzzy_models as _fm  # noqa: E402

ORDERS = [
    o.strip() for o in os.environ.get("REPRO_TSK_ORDERS", "0th,1st,2nd").split(",")
]
DATASET_NAMES = [
    d.strip().lower()
    for d in os.environ.get("REPRO_DATASETS", "concrete,bodyfat").split(",")
]

# Demo hyperparameters, taken verbatim from `tribble-tree/demo_concrete.py` /
# `table_hyperparam_normalization.py` (see that file's ARMS comment). Only
# `tsk_order` varies across cells; everything else is held fixed so a moving
# cell can only be attributed to the order.
TREE_KWARGS = dict(criterion="variance", max_depth=3, n_terms=2, top_n=4, min_soft_count=20)
HME_KWARGS = dict(
    criterion="variance", max_depth=2, n_gate_terms=2, top_n=4,
    min_soft_count=40, min_expert_samples=60,
)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def _load_datasets():
    """-> [(label, X, y)] for every requested, available dataset."""
    loaders = {
        "concrete": ("Concrete", _fm.load_concrete),
        "bodyfat": ("BodyFat", _fm.load_bodyfat),
        "bikeshare": ("BikeShare", _fm.load_bikeshare),
    }
    out = []
    for name in DATASET_NAMES:
        if name not in loaders:
            print(f"  [skip] unknown dataset {name!r}; choose from {list(loaders)}")
            continue
        label, load = loaders[name]
        data = load()
        if data is None:
            print(f"  [{name}] unavailable; its rows -> N/A")
            out.append((label, None, None))
            continue
        X, y = data
        out.append((label, X, np.asarray(y, dtype=float)))
    return out


# --- model builders, one per (family, order, seed) ----------------------------


def tree_regressor(order, seed):
    import fuzzytree

    cls = getattr(fuzzytree, "FuzzyRegressionTree", None)
    return _fm._try(lambda: cls(tsk_order=order, random_state=seed, **TREE_KWARGS)) if cls else None


def hme_regressor(order, seed):
    import fuzzytree

    cls = getattr(fuzzytree, "HierarchicalFuzzyExpertsRegressor", None)
    if cls is None:
        return None
    return _fm._try(
        lambda: cls(
            random_state=seed,
            expert_kwargs={"n_output_buckets": 3, "tsk_order": order},
            **HME_KWARGS,
        )
    )


def flat_regressor(order, seed):
    return _fm.mog_regressor(seed, tsk_order=order)


MODELS = [
    ("flat TribbleRegressor (reference)", flat_regressor),
    ("FuzzyRegressionTree (TribbleTree)", tree_regressor),
    ("HierarchicalFuzzyExpertsRegressor (HME)", hme_regressor),
]


def _warm_up(X, y):
    """One discarded fit so import/JIT/BLAS spin-up lands here, not on seed 0.

    PROVENANCE_MAP.md note 14: the first fit in a process ran at 3.68x the mean
    of the other nine seeds on the same arm, and looked like +/-60% seed spread
    until this exact throwaway fit was added upstream. Timed cells below would
    otherwise inherit that artifact on whichever (model, order) happens to run
    first for a given dataset.
    """
    try:
        est = tree_regressor(ORDERS[0], seed=0)
        if est is not None:
            est.fit(X, y)
    except Exception:  # noqa: BLE001 - best-effort warm-up only
        pass


def sweep(label, X, y):
    """-> {(model_name, order): {"r2": [...], "rmse": [...], "time": [...]}}"""
    store = {(m, o): {"r2": [], "rmse": [], "time": []} for m, _ in MODELS for o in ORDERS}
    complained = set()
    _warm_up(X, y)
    for seed in C.SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        for order in ORDERS:
            for model_name, build in MODELS:
                try:
                    est = build(order, seed)
                    if est is None:
                        raise RuntimeError("estimator class not found")
                    with C.timed() as t:
                        est.fit(Xtr, ytr)
                    pred = np.asarray(est.predict(Xte), dtype=float).ravel()
                except Exception as exc:  # noqa: BLE001 - report once, cell -> N/A
                    key = (model_name, order)
                    if key not in complained:
                        complained.add(key)
                        print(
                            f"  [skip] {label} / {model_name} / {order}: "
                            f"{exc.__class__.__name__}: {exc}"
                        )
                    continue
                store[(model_name, order)]["r2"].append(r2_score(yte, pred))
                store[(model_name, order)]["rmse"].append(_rmse(yte, pred))
                store[(model_name, order)]["time"].append(t.seconds)
        print(f"  {label}: seed {seed} done")
    return store


def rows_from(label, store):
    """Rows for the emitted table, plus a per-model relative-time view.

    The relative-time column reads training time as a MULTIPLE of that model's
    own 0th-order time (`common.normalized`'s "growth-shape" reading, not
    `normalized_worst`'s cross-model one): the question this table asks is "what
    does going from constant to linear to quadratic consequents cost *this*
    model," not "which model is fastest," so each model is its own baseline.
    """
    rows = []
    for model_name, _ in MODELS:
        times = [C.agg(store[(model_name, o)]["time"])[0] for o in ORDERS]
        rel = C.normalized(times)
        for order, rel_i in zip(ORDERS, rel):
            cell = store[(model_name, order)]
            rows.append(
                [
                    label,
                    model_name,
                    order,
                    C.cell(cell["r2"]),
                    C.cell(cell["rmse"]),
                    C.cell(cell["time"], fmt="{:.3f}"),
                    rel_i,
                ]
            )
    return rows


def main():
    print("TribbleTree TSK order sweep -- quality vs. training time")
    print(f"  orders:   {ORDERS}")
    print(f"  seeds:    {C.SEEDS}")
    print(f"  datasets: {DATASET_NAMES}")

    header = ["Dataset", "Model", "TSK order", "R2", "RMSE", "Train time (s)", "Time vs 0th-order"]
    rows = []
    for label, X, y in _load_datasets():
        if X is None:
            for model_name, _ in MODELS:
                for order in ORDERS:
                    rows.append([label, model_name, order, C.NA, C.NA, C.NA, C.NA])
            continue
        print(f"  [{label}] N={len(X)}  M={X.shape[1]}")
        rows += rows_from(label, sweep(label, X, y))

    C.emit(
        "table_tribbletree_tsk_order",
        "TribbleTree TSK order sweep — order 0/1/2, quality and training time",
        header,
        rows,
        note=(
            "Order 0 (constant leaf/rule consequents) is the TSK degenerate case "
            "closest to a Mamdani-style FIS's weighted-constant output; orders 1/2 "
            "add a local linear/quadratic correction. 'flat TribbleRegressor' is "
            "the non-hierarchical reference (tribblefis.gaussian_regressor); "
            "'FuzzyRegressionTree' is the plain hierarchical TribbleTree; the HME "
            "row sweeps its LEAF EXPERTS only -- the gate is a partition-of-unity "
            "product fixed by the model's semantics, not a free axis (see "
            "hme.py). Every non-order hyperparameter is held fixed at "
            "tribble-tree/demo_concrete.py's settings, taken verbatim, so a moving "
            "cell is attributable to the order alone. One warm-up fit is "
            "discarded per dataset before timing starts (PROVENANCE_MAP.md note "
            "14: an undiscarded first fit absorbs import/JIT/BLAS spin-up and "
            "reads as +/-60% seed noise). 'Time vs 0th-order' is each model's own "
            "training time as a multiple of ITS OWN 0th-order time -- a "
            "within-model growth-shape reading, not a cross-model speed "
            "comparison. Higher R2 and lower RMSE are better."
        ),
    )


if __name__ == "__main__":
    main()
