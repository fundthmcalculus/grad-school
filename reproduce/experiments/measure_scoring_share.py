"""How much of a small-top_n fit is feature SCORING? (supports tribble-fis #242)

Every `fit` calls `calculate_gaussian_correlation`, which re-scores all features, so a
top_n sweep pays for scoring once per size. This times that call alone on the PhiUSIIL
seed-0 training split and sets it against the measured k = 1 fit time from a Table 4.1c
archive's per-run CSV, per scorer.

Expectation registered before running: scoring is the majority of a k = 1 fit (a k = 1
fit has almost nothing else to do).

    uv run --project tribble-fis python reproduce/experiments/measure_scoring_share.py \
        [--runs reproduce/outputs/topn-sweep-2026-11-21/table_4_1c_topn_runs.csv]

One dataset, one split, three repeats per scorer: a measurement, not a distribution.
Times are from the host that runs this; the share transfers across machines, the seconds
do not.
"""

import argparse
import contextlib
import csv
import io
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tables"))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import _fuzzy_models as fm  # noqa: E402
import tribblefis.gauss_math as gm  # noqa: E402

SCORERS = ("wasserstein", "bhattacharyya", "composite", "higher_order")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        default=os.path.join(
            os.path.dirname(HERE),
            "outputs",
            "topn-sweep-2026-11-21",
            "table_4_1c_topn_runs.csv",
        ),
    )
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    data = fm.load_phiusiil()
    assert (
        data is not None
    ), "PhiUSIIL did not load -- this would silently measure nothing"
    X, y = data
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    ytr = pd.Series(np.asarray(ytr).ravel(), index=Xtr.index)

    def score(sc):  # the function prints a ranking; swallow it, time only the call
        with contextlib.redirect_stdout(io.StringIO()):
            t = time.perf_counter()
            gm.calculate_gaussian_correlation(Xtr, ytr, method=sc)
            return time.perf_counter() - t

    score("wasserstein")  # warm-up, discarded
    runs = list(csv.DictReader(open(args.runs)))
    n_k = len({r["top_n"] for r in runs if r["dataset"].startswith("PhiUSIIL")})
    print(f"PhiUSIIL train split {Xtr.shape}; {n_k} top_n values in the sweep")
    print(
        f"{'scorer':14s} {'scoring':>9s} {'fit@k=1':>9s} {'share':>6s} {'sweep/seed':>11s} {'saved if scored once':>21s}"
    )
    for sc in SCORERS:
        s = st.mean(score(sc) for _ in range(args.repeats))
        fit = [
            float(r["fit_seconds"])
            for r in runs
            if r["dataset"].startswith("PhiUSIIL")
            and r["scorer"] == sc
            and r["top_n"] == "1"
            and r["fit_seconds"]
        ]
        per_seed = sum(
            float(r["fit_seconds"])
            for r in runs
            if r["dataset"].startswith("PhiUSIIL")
            and r["scorer"] == sc
            and r["fit_seconds"]
        ) / len({r["seed"] for r in runs})
        saved = s * (n_k - 1)
        print(
            f"{sc:14s} {s:8.3f}s {st.mean(fit):8.3f}s {100*s/st.mean(fit):5.0f}% {per_seed:10.2f}s {saved:8.2f}s ({100*saved/per_seed:.0f}%)"
        )


if __name__ == "__main__":
    main()
