"""Goal G5, part two -- does quantile's advantage actually grow with skew?

The Concrete study found a crossover near four buckets but could not test the
hypothesis that motivated the whole question, because Concrete's target skew is
only +0.42. No collection of real datasets fixes that cleanly either: they differ
in dimensionality, noise, and sample size all at once, so a difference between
two of them is not attributable to skew.

So skew is isolated here as a single controlled parameter. A fixed linear signal
is generated once per seed, then pushed through a monotone transform whose
strength sets the target's skewness while leaving the *information content*
untouched -- the same X predicts the same ordering of y throughout. Any
difference between partitioning schemes across the sweep is therefore
attributable to the shape of the target distribution and to nothing else.

    y(lambda) = expm1(lambda * z) / lambda        (-> z as lambda -> 0)

with z the standardized signal-plus-noise. Sweeping lambda walks the target from
near-symmetric to heavily right-skewed. Because the map is strictly monotone, a
perfect learner would score identically at every lambda; the degradation that
shows up is exactly what the partitioning scheme fails to absorb.

Real datasets spanning a skew range are run alongside as corroboration.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_g5b_skew_sweep.py

Knobs:
    REPRO_LAMBDAS="0.01,0.5,1.0,1.5,2.0,2.5"
    REPRO_BUCKETS="4"        bucket count for the sweep
    REPRO_ORDERS="2nd"
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

LAMBDAS = [float(x) for x in os.environ.get(
    "REPRO_LAMBDAS", "0.01,0.5,1.0,1.5,2.0,2.5").split(",")]
BUCKETS = int(os.environ.get("REPRO_BUCKETS", "4"))
ORDER = os.environ.get("REPRO_ORDERS", "2nd").split(",")[0].strip()
N_SYNTH = int(os.environ.get("REPRO_N_SYNTH", "800"))
M_SYNTH = int(os.environ.get("REPRO_M_SYNTH", "6"))
L2 = 1e-2


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def synthetic(lam, seed):
    """Fixed signal, monotone-skewed target. Only the shape of y changes with lam."""
    rng = np.random.RandomState(1000 + seed)
    X = rng.randn(N_SYNTH, M_SYNTH)
    beta = rng.randn(M_SYNTH)
    z = X @ beta
    z = (z - z.mean()) / z.std()
    z = z + 0.30 * rng.randn(N_SYNTH)          # noise, fixed across lam
    y = z if abs(lam) < 1e-6 else np.expm1(lam * z) / lam
    cols = [f"x{i}" for i in range(M_SYNTH)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y_value")


def partition(y_raw, n, scheme):
    """uniform = equal width; quantile = equal frequency. Extremes pinned either way
    (the shipped solve now honours that -- see fix/pin-extreme-bucket-means)."""
    y_raw = pd.Series(np.asarray(y_raw, float), name="y_value")
    if scheme == "uniform":
        edges = np.linspace(y_raw.min(), y_raw.max(), n + 1)
        edges[0] -= 1e-9
        lab = pd.cut(y_raw, bins=edges, labels=False, include_lowest=True)
    else:
        lab = pd.qcut(y_raw, q=n, labels=False, duplicates="drop")
    lab = pd.Series(np.asarray(lab, float), name="y_bucket").fillna(0).astype(int)

    grouped = y_raw.groupby(lab).mean()
    cent = np.full(n, np.nan)
    for k, v in grouped.items():
        if 0 <= int(k) < n:
            cent[int(k)] = v
    cent = pd.Series(cent).interpolate(method="linear", limit_direction="both").values.copy()
    cent[0], cent[-1] = float(y_raw.min()), float(y_raw.max())
    occ = int(lab.value_counts().reindex(range(n), fill_value=0).min())
    return pd.concat([lab, y_raw], axis=1), cent, occ


def evaluate(X, y_raw, seed, scheme, n=BUCKETS, order=ORDER):
    from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                       create_gaussian_membership_dict,
                                       take_top_features)
    from tribblefis.regression import predict_tsk, solve_tsk_consequents

    yt = F.unit_scale(pd.Series(np.asarray(y_raw, float), name="y_value"))
    y, cent, occ = partition(yt, n, scheme)
    # min-max only -- this sweep deliberately applies NO log step.
    Xt = F.unit_scale(X)

    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.2, random_state=seed)
    d = calculate_gaussian_correlation(Xtr, ytr["y_bucket"])
    _, top = take_top_features(d, top_n=len(Xt.columns))
    memb = create_gaussian_membership_dict(Xtr, ytr["y_bucket"], top_n_var_names=top)
    corr, cent2 = solve_tsk_consequents(Xtr, memb, top, cent, ytr, n_output_buckets=n,
                                        order=order, l2_reg=L2, basis="raw",
                                        cross_pairs=None, verbose=False)
    pred = np.asarray(predict_tsk(Xte, memb, top, cent2, corr, order=order,
                                  basis="raw", cross_pairs=None), float).ravel()
    truth = np.asarray(yte["y_value"], float).ravel()
    k = ~np.isnan(pred)
    if k.sum() < 5:
        return None
    pred, truth = pred[k], truth[k]
    lo, hi = np.quantile(truth, 0.10), np.quantile(truth, 0.90)
    tail = (truth <= lo) | (truth >= hi)
    return {"r2": r2_score(truth, pred),
            "tail_rmse": _rmse(truth[tail], pred[tail]) if tail.any() else np.nan,
            "occ": occ}


def main():
    print("Goal G5b -- does quantile's advantage grow with target skew?")
    print(f"  synthetic: n={N_SYNTH} m={M_SYNTH} buckets={BUCKETS} order={ORDER} "
          f"seeds={C.SEEDS}")

    rows = []
    print(f"\n  {'lambda':>7} {'skew':>7} {'uniform R2':>11} {'quantile R2':>12} "
          f"{'Q-U':>8} {'unif min-n':>11}")
    for lam in LAMBDAS:
        got: dict = {"uniform": [], "quantile": []}
        tails: dict = {"uniform": [], "quantile": []}
        occs: dict = {"uniform": [], "quantile": []}
        skews = []
        for seed in C.SEEDS:
            X, y = synthetic(lam, seed)
            skews.append(float(pd.Series(y).skew()))
            for scheme in ("uniform", "quantile"):
                try:
                    r = evaluate(X, y, seed, scheme)
                except Exception as exc:  # noqa: BLE001
                    print(f"    [{scheme} lam={lam}] seed {seed}: {exc.__class__.__name__}")
                    continue
                if r:
                    got[scheme].append(r["r2"])
                    tails[scheme].append(r["tail_rmse"])
                    occs[scheme].append(r["occ"])
        um, _ = C.agg(got["uniform"])
        qm, _ = C.agg(got["quantile"])
        sk, _ = C.agg(skews)
        if um is None or qm is None:
            continue
        print(f"  {lam:7.2f} {sk:+7.2f} {um:11.3f} {qm:12.3f} {qm-um:+8.3f} "
              f"{int(np.min(occs['uniform'])):11d}")
        rows.append([f"{lam:.2f}", f"{sk:+.2f}",
                     C.cell(got["uniform"]), C.cell(got["quantile"]),
                     f"{qm-um:+.3f}",
                     C.cell(tails["uniform"], fmt="{:.3f}"),
                     C.cell(tails["quantile"], fmt="{:.3f}"),
                     str(int(np.min(occs["uniform"])))])

    C.emit("table_g5b_skew_sweep",
           "Goal G5b — partitioning vs. target skew (synthetic, skew isolated)",
           ["λ", "target skew", "uniform R²", "quantile R²", "Q − U",
            "uniform tail RMSE", "quantile tail RMSE", "uniform min bucket n"],
           rows,
           note=(f"A fixed linear signal is pushed through y = expm1(λz)/λ, a strictly "
                 f"monotone map, so the information in X is identical at every λ and only "
                 f"the SHAPE of the target changes. A perfect learner would score the same "
                 f"across the whole row; what degrades is what the partitioning fails to "
                 f"absorb. 'Q − U' positive means quantile wins. {BUCKETS} buckets, "
                 f"{ORDER}-order consequents, extremes pinned (the shipped behaviour after "
                 f"fix/pin-extreme-bucket-means). n={N_SYNTH}, m={M_SYNTH}."))


if __name__ == "__main__":
    main()
