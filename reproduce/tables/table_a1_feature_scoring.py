"""Appendix A.4 -- feature ranking by scorer, and what the ranking costs.

Two tables:

  A.1  the top features each scorer selects, side by side
  A.2  accuracy and fit time against the number of features kept

The point is that these are the same model and the same data; only the feature
*ranking* differs. On PhiUSIIL one scorer puts the single most informative
feature first and another leaves it out of the top twenty, and the downstream
difference is 0.9969 against 0.4251 at one feature.

The four-metric composite that used to be the default was removed upstream
(tribble-fis #34). If it is ever restored as `method="composite"` this script
picks it up automatically; until then that column reports N/A with the reason,
rather than being quietly dropped.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_a1_feature_scoring.py

Knobs:
    REPRO_SEEDS="0,1"          seeds (2 is plenty; the effect is enormous)
    REPRO_PHIUSIIL_N="20000"   sample size
    REPRO_SCORERS="wasserstein,bhattacharyya,composite"
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C            # noqa: E402
import _fuzzy_models as F     # noqa: E402

SAMPLE_N = int(os.environ.get("REPRO_PHIUSIIL_N", "20000"))
SCORERS = [s.strip() for s in
           os.environ.get("REPRO_SCORERS", "wasserstein,bhattacharyya,composite").split(",")]
K_GRID = [1, 2, 3, 4, 5, 7, 10, 15, 20]
TOP_SHOWN = 5


def _rank(scorer, X, y_series):
    """Ranked (feature, score) list under one scorer, or None if unsupported."""
    import tribblefis.gauss_math as gm
    try:
        return gm.calculate_gaussian_correlation(X, y_series, method=scorer)
    except (ValueError, TypeError) as exc:
        # ValueError: the library rejects the name. TypeError: no `method` kwarg
        # at all, i.e. an older build. Either way the column is unavailable, and
        # saying so beats substituting a different scorer's ranking.
        print(f"  [{scorer}] unavailable ({exc.__class__.__name__}: {exc})")
        return None


def _sweep(scorer, X, y):
    """(accuracy, fit_seconds) per k, with the scorer forced for every fit."""
    import tribblefis.gauss_math as gm
    import tribblefis.gaussian_classifier as gc

    original = gm.calculate_gaussian_correlation

    def forced(Xa, ya, method=scorer, **kw):
        return original(Xa, ya, method=scorer, **kw)

    gm.calculate_gaussian_correlation = forced
    gc.calculate_gaussian_correlation = forced
    try:
        out = {}
        for k in K_GRID:
            accs, secs = [], []
            for seed in C.SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=seed)
                model = gc.MixtureOfGaussiansFuzzyClassifier(top_n=k, random_state=seed)
                start = time.perf_counter()
                model.fit(Xtr, ytr)
                secs.append(time.perf_counter() - start)
                accs.append(accuracy_score(yte, np.asarray(model.predict(Xte))))
            out[k] = (float(np.mean(accs)), float(np.mean(secs)))
        return out
    finally:
        # Restore even on failure; a leaked monkeypatch would silently change
        # every table run afterwards in the same process.
        gm.calculate_gaussian_correlation = original
        gc.calculate_gaussian_correlation = original


def main():
    print("Appendix A.4 -- feature scoring")
    data = F.load_phiusiil(sample_size=SAMPLE_N)
    if data is None:
        C.emit("table_a1_feature_ranking", "Table A.1 -- Feature ranking by scorer",
               ["Rank"], [["(PhiUSIIL unavailable)"]])
        return
    X, y = data
    y_series = pd.Series(np.asarray(y).ravel(), index=X.index)
    print(f"  N={len(X)}  M={X.shape[1]}  seeds={C.SEEDS}  scorers={SCORERS}")

    rankings = {s: _rank(s, X, y_series) for s in SCORERS}
    available = [s for s, r in rankings.items() if r]

    # ---- Table A.1: who ranks what ----
    header = ["Rank"] + [s for s in SCORERS]
    rows = []
    for i in range(TOP_SHOWN):
        row = [str(i + 1)]
        for s in SCORERS:
            r = rankings[s]
            row.append(f"{r[i][0]} ({r[i][1]:.3f})" if r and i < len(r) else C.NA)
        rows.append(row)
    C.emit("table_a1_feature_ranking",
           "Table A.1 -- Feature ranking depends on the scorer (PhiUSIIL)",
           header, rows,
           note=("Same data, same model -- only the feature ranking differs. The "
                 "composite column is the four-metric consensus rule removed by "
                 "tribble-fis #34; it reports N/A until that is restored as "
                 "method='composite'."))

    # ---- Table A.2: what the ranking costs ----
    sweeps = {s: _sweep(s, X, y) for s in available}
    header2 = ["Features kept"] + [f"{s} (acc / fit s)" for s in available]
    rows2 = []
    for k in K_GRID:
        row = [str(k)]
        for s in available:
            acc, secs = sweeps[s][k]
            row.append(f"{acc:.4f} / {secs:.2f}")
        rows2.append(row)
    C.emit("table_a2_feature_count",
           "Table A.2 -- Accuracy and fit time vs features kept (PhiUSIIL)",
           header2, rows2,
           note=("A ranking that works reaches ~0.997 on ONE feature; a ranking "
                 "that does not never gets there at any size tested. This is the "
                 "evidence that interpretability is a property of the feature "
                 "ranking rather than of the model architecture -- see Appendix "
                 "A.4 and tribble-fis #49."))


if __name__ == "__main__":
    main()
