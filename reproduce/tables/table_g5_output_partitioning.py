"""Goal G5 -- uniform vs. quantile output partitioning, settled empirically.

Chapter 4 §4.3.2 presents the choice of output partition as genuinely open. This
settles it. Reading `tribblefis.regression.partition_output` first changes the
question, though, and worth stating up front: the shipped implementation is
already a HYBRID. It takes equal-frequency (`qcut`) bucket boundaries but then
overwrites the two extreme bucket centroids with the true min and max --

    y_bucket_mean[0]  = float(y_raw.min())
    y_bucket_mean[-1] = float(y_raw.max())

-- which is precisely the "quantile interior, pinned extremes" compromise that
the open question hypothesised might dominate. So the three arms are:

  uniform         equal-WIDTH bucket boundaries; centroids are bucket means.
                  Even coverage of the output range; starves buckets on skew.
  quantile        equal-FREQUENCY boundaries; centroids are bucket means.
                  Every bucket well-supported; under-resolves the extremes.
  hybrid          equal-frequency boundaries, extreme centroids pinned to
                  min/max. The default UP TO 2026-08-03; uniform is now shipped,
                  and `partition_output(..., method="quantile")` reproduces this.

CONSEQUENT ORDER IS THE AXIS THIS STUDY ORIGINALLY MISSED. It ran 1st and 2nd
order only, found every separation smaller than the seed spread producing it, and
left G5 open with no scheme recommended. That was a correct reading of the wrong
regime. `solve_tsk_consequents` pins the first and last rules' CONSTANT terms to
whatever centroids it is handed, so at 0th order -- where the constant is a rule's
entire output -- a pinned centroid becomes prediction error with nothing to absorb
it, while 1st and 2nd order pay for it through their linear terms and hide it
inside the seed spread. Concrete, three buckets:

  0th   uniform +0.394 +/- 0.065   quantile +0.242   hybrid -0.434 +/- 0.241
  1st   uniform  0.796 +/- 0.018   quantile  0.789   hybrid  0.787 +/- 0.026
  2nd   uniform  0.841 +/- 0.021   quantile  0.836   hybrid  0.832 +/- 0.027

0.828 across the arms at 0th order, 0.009 at 1st. The partition binds hardest
exactly where the study was not looking, so 0th order is now in ORDERS by default.
The decomposition matters too: of that 0.828, the boundary scheme is worth 0.152
and the PINNING is worth 0.676 -- the hybrid was a real third scheme all along,
and the worst of the three, not the inert no-op §4.3.2 once called it.

METRICS -- aggregate error alone cannot answer this, because the failure mode is
localised. A starved bucket or a mangled tail barely moves a global average. So
alongside R² and RMSE this reports tail RMSE (the true-value bottom and top
deciles, where uniform is predicted to win and quantile to lose), max absolute
error, and the minimum bucket occupancy that explains WHY a scheme fails when it
does.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_g5_output_partitioning.py

Knobs:
    REPRO_BUCKETS="3,4,6"      bucket counts to sweep
    REPRO_ORDERS="1st,2nd"     TSK orders
    REPRO_SEEDS="0,1,2"
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C            # noqa: E402
import _fuzzy_models as F     # noqa: E402

BUCKETS = [int(b) for b in os.environ.get("REPRO_BUCKETS", "3,4,6").split(",")]
ORDERS = [o.strip() for o in os.environ.get("REPRO_ORDERS", "0th,1st,2nd").split(",")]
L2 = 1e-2


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# --------------------------------------------------------------------------- #
# the three partition schemes
# --------------------------------------------------------------------------- #
def partition(y_raw, n, scheme):
    """Return (y_frame, bucket_centroids, min_occupancy) for one scheme."""
    y_raw = pd.Series(np.asarray(y_raw, dtype=float), name="y_value")

    if scheme == "uniform":
        edges = np.linspace(y_raw.min(), y_raw.max(), n + 1)
        edges[0] -= 1e-9
        lab = pd.cut(y_raw, bins=edges, labels=False, include_lowest=True)
    else:  # quantile boundaries for both "quantile" and "hybrid"
        lab = pd.qcut(y_raw, q=n, labels=False, duplicates="drop")

    lab = pd.Series(np.asarray(lab, dtype=float), name="y_bucket").fillna(0).astype(int)

    grouped = y_raw.groupby(lab).mean()
    cent = np.full(n, np.nan)
    for k, v in grouped.items():
        if 0 <= int(k) < n:
            cent[int(k)] = v
    cent = pd.Series(cent).interpolate(method="linear", limit_direction="both").values.copy()

    if scheme == "hybrid":                    # what the shipped code does
        cent[0] = float(y_raw.min())
        cent[-1] = float(y_raw.max())

    occ = lab.value_counts().reindex(range(n), fill_value=0).min()
    return pd.concat([lab, y_raw], axis=1), cent, int(occ)


def _drop_empty_buckets(lab, cent, n):
    """Renumber the occupied buckets to 0..k-1 and keep only their centroids.

    Needed because the partition is now fit on the training fold. At high skew an
    equal-width bucket can be empty in the train fold, and then the rule base built from
    the labels has fewer rules than `cent` has entries: `solve_tsk_consequents` returns
    one mean per rule while `predict_tsk` indexes by rule id, which raised IndexError.
    Starvation to the point of an empty bucket is the phenomenon this sweep exists to
    measure, so it has to degrade to a smaller rule base rather than crash.

    The caller keeps the occupancy reported by `partition` for the REQUESTED n, which is
    the starvation diagnostic and is allowed to be zero. This only changes what the model
    is fit on.
    """
    present = np.unique(np.asarray(lab))
    if len(present) == n:
        return np.asarray(lab), cent, n
    remap = {int(o): i for i, o in enumerate(present)}
    return (np.array([remap[int(v)] for v in np.asarray(lab)]),
            np.asarray(cent)[present.astype(int)], int(len(present)))


def run_one(X, y_raw, seed, n, order, scheme):
    from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                       create_gaussian_membership_dict,
                                       take_top_features)
    from tribblefis.regression import predict_tsk, solve_tsk_consequents

    # Leak-free since 2026-08-04: split first, then derive the target scale, the partition
    # and the feature scaler from the training fold alone. The partition is the variable
    # under study here, so fitting it on all 1,030 rows would have let every arm see the
    # test fold's target distribution through its own bin edges and centroids -- and the
    # centroids reach inference through `solve_tsk_consequents`. The leak was symmetric
    # across the three arms, so it did not bias the comparison, which is why this table's
    # conclusion is unchanged; it did make the absolute R2 values non-comparable with any
    # honest out-of-sample number. `occ` is the TRAIN-fold occupancy, which is the quantity
    # the starvation argument is about.
    idx = np.arange(len(X))
    itr, _ite = train_test_split(idx, test_size=0.2, random_state=seed)
    tr = np.zeros(len(X), dtype=bool); tr[itr] = True

    yr = pd.Series(np.asarray(y_raw, float), index=X.index, name="y_value")
    yt = F.unit_scale_with(float(yr[tr].min()), float(yr[tr].max()), yr)
    y_tr_part, cent, occ = partition(yt[tr], n, scheme)
    lab_tr, cent, n_eff = _drop_empty_buckets(y_tr_part["y_bucket"].values, cent, n)
    buckets = pd.Series(np.nan, index=yt.index, name="y_bucket")
    buckets[tr] = lab_tr
    buckets[~tr] = 0          # cosmetic: only ytr["y_bucket"] is ever read
    y = pd.concat([buckets.astype(int), yt.rename("y_value")], axis=1)

    # log + min-max, i.e. what this table has always measured (see F.normalize).
    sc, _lg = F.fit_scaler(X[tr], scaler="unit", log_dynamic_range=2)
    Xt = F.apply_scaler(sc, X)

    Xtr, Xte, ytr, yte = Xt[tr], Xt[~tr], y[tr], y[~tr]
    diffs = calculate_gaussian_correlation(Xtr, ytr["y_bucket"])
    _, top = take_top_features(diffs, top_n=len(Xt.columns))
    memb = create_gaussian_membership_dict(Xtr, ytr["y_bucket"], top_n_var_names=top)
    corr, cent2 = solve_tsk_consequents(Xtr, memb, top, cent, ytr,
                                        n_output_buckets=n_eff, order=order,
                                        l2_reg=L2, basis="raw", cross_pairs=None)
    pred = np.asarray(predict_tsk(Xte, memb, top, cent2, corr, order=order,
                                  basis="raw", cross_pairs=None), float).ravel()
    truth = np.asarray(yte["y_value"], float).ravel()
    keep = ~np.isnan(pred)
    pred, truth = pred[keep], truth[keep]
    if len(truth) < 5:
        return None

    span = float(np.asarray(y_raw, float).max() - np.asarray(y_raw, float).min())
    lo, hi = np.quantile(truth, 0.10), np.quantile(truth, 0.90)
    tail = (truth <= lo) | (truth >= hi)      # the extremes, where schemes differ
    return {
        "r2": r2_score(truth, pred),
        "rmse": _rmse(truth, pred) * span,
        "tail_rmse": (_rmse(truth[tail], pred[tail]) * span) if tail.any() else np.nan,
        "max_err": float(np.max(np.abs(truth - pred))) * span,
        "occ": occ,
    }


def main():
    print("Goal G5 -- output partitioning: uniform vs quantile vs hybrid")
    data = F.load_concrete()
    if data is None:
        print("  dataset unavailable")
        return
    X, y = data
    skew = float(pd.Series(np.asarray(y, float)).skew())
    print(f"  Concrete N={len(X)} target skew={skew:+.3f} "
          f"buckets={BUCKETS} orders={ORDERS} seeds={C.SEEDS}")

    store: dict = {}
    for n in BUCKETS:
        for order in ORDERS:
            for scheme in ("uniform", "quantile", "hybrid"):
                for seed in C.SEEDS:
                    try:
                        r = run_one(X, y, seed, n, order, scheme)
                    except Exception as exc:  # noqa: BLE001
                        print(f"    [{scheme} n={n} {order}] seed {seed}: "
                              f"{exc.__class__.__name__}")
                        continue
                    if r is None:
                        continue
                    k = (n, order, scheme)
                    store.setdefault(k, {m: [] for m in r})
                    for m, v in r.items():
                        store[k][m].append(v)
            print(f"  done: n={n} {order}")

    rows = []
    for (n, order, scheme), v in sorted(store.items()):
        rows.append([
            str(n), order,
            scheme + (" *(shipped)*" if scheme == "hybrid" else ""),
            C.cell(v["r2"]),
            C.cell(v["rmse"], fmt="{:.2f}"),
            C.cell(v["tail_rmse"], fmt="{:.2f}"),
            C.cell(v["max_err"], fmt="{:.1f}"),
            str(int(np.min(v["occ"]))),
        ])

    C.emit("table_g5_output_partitioning",
           "Goal G5 — output partitioning: uniform vs. quantile vs. hybrid (Concrete)",
           ["buckets", "order", "scheme", "R²", "RMSE (MPa)", "tail RMSE (MPa)",
            "max err (MPa)", "min bucket n"], rows,
           note=("Tail RMSE is computed over the true-value bottom and top deciles, which is "
                 "where the schemes are predicted to diverge: uniform covers the range evenly "
                 "and should hold the extremes, quantile crowds its boundaries where the data "
                 "is dense and should lose them. 'min bucket n' is the smallest training-bucket "
                 "occupancy and is the diagnostic that explains *why* a scheme fails when it "
                 "does. The 0th-order rows are the ones that separate the arms: "
                 "`solve_tsk_consequents` holds the first and last rules' constant terms at the "
                 "centroids it is handed, and at 0th order that constant is a rule's entire "
                 "output, so the hybrid's pinned extremes become prediction error directly. At "
                 "1st and 2nd order the linear terms absorb them and every separation falls "
                 "inside the seed spread — which is why running those two orders alone left G5 "
                 "looking undecidable. `hybrid` was the shipped default up to 2026-08-03; "
                 "`uniform` is now the default and `partition_output(..., method=\"quantile\") "
                 "reproduces the hybrid. Concrete target skew "
                 f"= {skew:+.3f}."))


if __name__ == "__main__":
    main()
