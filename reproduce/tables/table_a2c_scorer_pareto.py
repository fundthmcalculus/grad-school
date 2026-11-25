"""Appendix A.4 -- Table A.2c: the runtime / accuracy trade-off of each feature scorer,
across datasets.

Table A.2 measures the eight library scorers on PhiUSIIL alone. This generator
runs the same fits -- `TribbleClassifier(top_n=k, scoring_method=<scorer>)`,
library defaults otherwise (0.85 redundancy filter on), ten 80/20 splits -- on
three classification datasets in one process, so that every timing comes from
the same host state, and reports which (scorer, k) configurations are on the
Pareto front of accuracy (higher is better) against fit time (lower is better).

Fit time includes scoring: the scorer runs inside `fit`, so the seconds are the
cost a user of the library actually pays for choosing that scorer.

Datasets:
  PhiUSIIL       repro_data.load_phiusiil (leak-free, 20,000-row sample) --
                 the same rows and splits as Table A.2, so its accuracies must
                 match Table A.2's exactly (a check, not a coincidence)
  Breast Cancer  scikit-learn's bundled copy (569 x 30, binary)
  Wine           scikit-learn's bundled UCI wine *recognition* set (178 x 13,
                 three cultivars) -- NOT the wine-quality data in data/

Runtime is compared across datasets by normalizing: each configuration's mean fit
seconds is divided by its dataset's mean over a common grid (every scorer at every
k in K_COMMON, the budgets all three datasets can take). A bigger dataset then
cannot look worse merely for being bigger, and 1.0 means "typical cost on that
dataset". The common grid, not each dataset's full grid, is the denominator, so
PhiUSIIL's k=15/20 fits do not inflate its average relative to Wine's.

Outputs:
  table_a2c_scorer_runs.csv     one line per fit (dataset, scorer, k, seed,
                                accuracy, fit seconds, features kept)
  table_a2c_scorer_pareto       per dataset: every configuration with its
                                normalized runtime and whether it is on that
                                dataset's front (Markdown shows the fronts only)
  table_a2c_pooled_pareto       per (scorer, k) in K_COMMON: accuracy averaged
                                over the datasets, normalized runtime as the
                                geometric mean over the datasets, and the pooled
                                front (Markdown shows the front only)

The front ignores seed spread: a configuration is dominated if another is at
least as accurate (mean) and at least as fast (mean), and strictly better on one.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_a2c_scorer_pareto.py

Knobs: REPRO_SEEDS, REPRO_PHIUSIIL_N, REPRO_SCORERS -- as Table A.2.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import _fuzzy_models as F  # noqa: E402
import table_a1_feature_scoring as A  # noqa: E402

K_COMMON = [1, 2, 3, 4, 5, 7, 10]


def _datasets():
    out = []
    data = F.load_phiusiil(sample_size=A.SAMPLE_N)
    if data is None:
        print("  [PhiUSIIL] unavailable -- its rows are absent, not substituted")
    else:
        X, y = data
        out.append(("PhiUSIIL", X, pd.Series(np.asarray(y).ravel(), index=X.index)))
    for name, loader in (("Breast Cancer", load_breast_cancer), ("Wine", load_wine)):
        Xd, yd = loader(return_X_y=True, as_frame=True)
        out.append((name, Xd.reset_index(drop=True), yd.reset_index(drop=True)))
    return out


def _front(points):
    """Indices of the non-dominated points; points are (accuracy, seconds)."""
    keep = []
    for i, (a, t) in enumerate(points):
        dominated = any(
            (b >= a and u <= t) and (b > a or u < t)
            for j, (b, u) in enumerate(points)
            if j != i
        )
        if not dominated:
            keep.append(i)
    return set(keep)


def main():
    print("Appendix A.4 -- Table A.2c: scorer runtime / accuracy trade-off")
    print(f"  seeds={C.SEEDS}  scorers={A.SCORERS}")
    runs = []  # dataset, scorer, k, seed, acc, secs, n_kept, features
    cells = {}  # (dataset, scorer, k) -> (acc_mean, acc_sd, sec_mean, sec_sd)
    names = []
    for ds, X, y in _datasets():
        names.append(ds)
        k_grid = [k for k in A.K_GRID if k <= X.shape[1]]
        print(f"  [{ds}] N={len(X)}  M={X.shape[1]}  k={k_grid}")
        for scorer in A.SCORERS:
            sub = []
            A._sweep(scorer, X, y, sub, k_grid=k_grid)
            runs += [[ds] + r for r in sub]
            for k in k_grid:
                acc = [float(r[3]) for r in sub if r[1] == k]
                sec = [float(r[4]) for r in sub if r[1] == k]
                cells[(ds, scorer, k)] = (
                    float(np.mean(acc)),
                    float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
                    float(np.mean(sec)),
                    float(np.std(sec, ddof=1)) if len(sec) > 1 else 0.0,
                )
            print(f"    {scorer}: done", flush=True)

    C.write_csv(
        os.path.join(C.OUTPUT_DIR, "table_a2c_scorer_runs.csv"),
        [
            "dataset",
            "scorer",
            "k",
            "seed",
            "accuracy",
            "fit_seconds",
            "n_kept",
            "features",
        ],
        runs,
    )

    # ---- per-dataset fronts ----
    denom = {
        ds: float(np.mean([cells[(ds, s, k)][2] for s in A.SCORERS for k in K_COMMON]))
        for ds in names
    }
    header = [
        "Dataset",
        "Scorer",
        "Features kept",
        "Accuracy (mean)",
        "Accuracy (s.d.)",
        "Fit s (mean)",
        "Fit s (s.d.)",
        "Runtime / dataset mean",
        "On front",
    ]
    rows, md_rows = [], []
    for ds in names:
        keys = [key for key in cells if key[0] == ds]
        front = _front([(cells[key][0], cells[key][2]) for key in keys])
        for key in sorted(keys, key=lambda kk: cells[kk][2]):
            a, sd, t, tsd = cells[key]
            on = keys.index(key) in front
            row = [
                ds,
                key[1],
                str(key[2]),
                f"{a:.4f}",
                f"{sd:.4f}",
                f"{t:.4f}",
                f"{tsd:.4f}",
                f"{t / denom[ds]:.3f}",
                "yes" if on else "",
            ]
            rows.append(row)
            if on:
                md_rows.append(
                    [
                        ds,
                        key[1],
                        str(key[2]),
                        f"{a:.4f} ± {sd:.4f}",
                        f"{t / denom[ds]:.2f}×",
                    ]
                )
    C.emit(
        "table_a2c_scorer_pareto",
        "Table A.2c -- Runtime / accuracy front of each dataset (eight scorers)",
        header,
        rows,
        md_header=["Dataset", "Scorer", "k", "Accuracy", "Runtime / dataset mean"],
        md_rows=md_rows,
        note=(
            "Fronts only, cheapest first; the CSV has every configuration. Runtime "
            "is mean fit seconds (scoring included) divided by the dataset's mean "
            "over every scorer at k in "
            f"{K_COMMON}: "
            + ", ".join(f"{ds} {denom[ds]:.3f} s" for ds in names)
            + ". The front ignores seed spread."
        ),
    )

    # ---- pooled front over the common grid ----
    pooled = {}
    for s in A.SCORERS:
        for k in K_COMMON:
            if not all((ds, s, k) in cells for ds in names):
                continue
            accs = [cells[(ds, s, k)][0] for ds in names]
            sds = [cells[(ds, s, k)][1] for ds in names]
            norm = [cells[(ds, s, k)][2] / denom[ds] for ds in names]
            pooled[(s, k)] = (
                float(np.mean(accs)),
                math.sqrt(sum(v * v for v in sds)) / len(sds),
                float(np.exp(np.mean(np.log(norm)))),
                norm,
            )
    pkeys = list(pooled)
    pfront = _front([(pooled[key][0], pooled[key][2]) for key in pkeys])
    header2 = (
        ["Scorer", "Features kept", "Accuracy (mean over datasets)", "± (pooled s.d.)"]
        + ["Runtime / dataset mean (geometric mean)"]
        + [f"{ds} runtime / mean" for ds in names]
        + ["On front"]
    )
    rows2, md2 = [], []
    for key in sorted(pkeys, key=lambda kk: pooled[kk][2]):
        a, sd, g, norm = pooled[key]
        on = pkeys.index(key) in pfront
        rows2.append(
            [key[0], str(key[1]), f"{a:.4f}", f"{sd:.4f}", f"{g:.3f}"]
            + [f"{v:.3f}" for v in norm]
            + ["yes" if on else ""]
        )
        if on:
            md2.append(
                [key[0], str(key[1]), f"{a:.4f} ± {sd:.4f}", f"{g:.2f}×"]
                + [f"{v:.2f}×" for v in norm]
            )
    C.emit(
        "table_a2c_pooled_pareto",
        "Table A.2c (pooled) -- Runtime / accuracy front over " + ", ".join(names),
        header2,
        rows2,
        md_header=["Scorer", "k", "Accuracy (mean of datasets)", "Runtime (geo. mean)"]
        + names,
        md_rows=md2,
        note=(
            "Front only, cheapest first; the CSV has every (scorer, k) in "
            f"{K_COMMON}. Accuracy is the plain mean of the per-dataset means, "
            "± sqrt(sum sd^2)/n. Runtime is the geometric mean over the datasets "
            "of runtime / dataset mean, so no one dataset's scale dominates."
        ),
    )


if __name__ == "__main__":
    main()
