"""Appendix A.4 -- feature ranking by scorer, and what the ranking costs.

Tables:

  A.1  the top features each scorer selects, side by side
  A.2  accuracy and fit time against the number of features kept
  A.2b selection consistency: how many features every one of the ten seeds'
       fits agrees on, per scorer and budget (`table_a2b_selection_consistency`)
  table_a2_feature_runs.csv  one line per fit (scorer, k, seed, accuracy,
       fit seconds, features kept) -- the evidence A.2b and Figure A.1 read

The point is that these are the same model and the same data; only the feature
*ranking* differs. On PhiUSIIL one scorer puts the single most informative
feature first and another leaves it out of the top twenty, and the downstream
difference is 0.9969 against 0.4251 at one feature.

A scorer the library does not know reports N/A with the reason rather than
being quietly dropped (the four-metric composite was removed upstream in
tribble-fis #34 and is accepted again by the pinned build; `higher_order` -- a
moment distance through skewness and kurtosis -- arrived with tribble-fis #239;
the rank/bin scorers `hellinger`, `jensen_shannon`, `ks` and the kNN
`mutual_info` arrived with tribble-fis #253).

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_a1_feature_scoring.py

Knobs:
    REPRO_SEEDS="0,1"          seeds (2 is plenty; the effect is enormous)
    REPRO_PHIUSIIL_N="20000"   sample size
    REPRO_SCORERS="wasserstein,bhattacharyya,composite,higher_order,
                   hellinger,jensen_shannon,ks,mutual_info"
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
import common as C  # noqa: E402
import _fuzzy_models as F  # noqa: E402

SAMPLE_N = int(os.environ.get("REPRO_PHIUSIIL_N", "20000"))
SCORERS = [
    s.strip()
    for s in os.environ.get(
        "REPRO_SCORERS",
        "wasserstein,bhattacharyya,composite,higher_order,"
        "hellinger,jensen_shannon,ks,mutual_info",
    ).split(",")
]
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


def _sweep(scorer, X, y, runs):
    """(accuracy, fit_seconds) per k, with `scoring_method=scorer` for every fit.

    Appends one row per fit to `runs`: the features the fitted model actually
    kept (`top_features_`, after the 0.85 redundancy filter), so selection
    consistency across seeds is measured on the model, not on a re-ranking.

    Uses the estimator's own `scoring_method` (tribble-fis #239) instead of the
    monkeypatch of `calculate_gaussian_correlation` this used to need, which
    could leak into later tables in the same process if it was not restored.
    """
    import tribblefis.gaussian_classifier as gc

    out = {}
    for k in K_GRID:
        accs, secs = [], []
        for seed in C.SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            model = gc.TribbleClassifier(
                top_n=k, random_state=seed, scoring_method=scorer
            )
            start = time.perf_counter()
            model.fit(Xtr, ytr)
            secs.append(time.perf_counter() - start)
            acc = accuracy_score(yte, np.asarray(model.predict(Xte)))
            accs.append(acc)
            kept = [str(f) for f in (model.top_features_ or [])]
            runs.append(
                [scorer, k, seed, f"{acc:.6f}", f"{secs[-1]:.4f}", len(kept)]
                + ["|".join(kept)]
            )
        out[k] = (float(np.mean(accs)), float(np.mean(secs)))
    return out


def main():
    print("Appendix A.4 -- feature scoring")
    data = F.load_phiusiil(sample_size=SAMPLE_N)
    if data is None:
        C.emit(
            "table_a1_feature_ranking",
            "Table A.1 -- Feature ranking by scorer",
            ["Rank"],
            [["(PhiUSIIL unavailable)"]],
        )
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
    C.emit(
        "table_a1_feature_ranking",
        "Table A.1 -- Feature ranking depends on the scorer (PhiUSIIL)",
        header,
        rows,
        note=(
            "Same data, same model -- only the feature ranking differs. A "
            "column reads N/A when the pinned library rejects that scorer name."
        ),
    )

    # ---- Table A.2: what the ranking costs ----
    runs = []
    sweeps = {s: _sweep(s, X, y, runs) for s in available}
    header2 = ["Features kept"] + [f"{s} (acc / fit s)" for s in available]
    rows2 = []
    for k in K_GRID:
        row = [str(k)]
        for s in available:
            acc, secs = sweeps[s][k]
            row.append(f"{acc:.4f} / {secs:.2f}")
        rows2.append(row)
    C.emit(
        "table_a2_feature_count",
        "Table A.2 -- Accuracy and fit time vs features kept (PhiUSIIL)",
        header2,
        rows2,
        note=(
            # This note used to read "A ranking that works reaches ~0.997 on ONE
            # feature". That was true only while URLSimilarityIndex -- a URL's
            # similarity to a whitelist of known-legitimate URLs, i.e. the label
            # in disguise -- was in the feature set. grad-school #215 drops it on
            # load, and leak-free NO ranking is accurate at a single clause: the
            # best one-feature cell is 0.79, and the ordering of the scorers
            # INVERTS (wasserstein was the strongest at one feature and is now
            # the weakest). A generator that keeps asserting the old conclusion
            # in its own footnote is worse than one that asserts nothing, so the
            # note now states the measurement and leaves the conclusion to A.4.
            "Feature counts are AFTER the leak-free drop (grad-school #215): "
            "URLSimilarityIndex, TLDLegitimateProb and URLCharProb are not in "
            "the ranked set. Read the one-feature row against the twenty-feature "
            "row -- the gap between scorers at a fixed budget is the effect this "
            "table measures. See Appendix A.4, PROVENANCE_MAP note 31, and "
            "tribble-fis #49."
        ),
    )

    # ---- per-fit evidence, and selection consistency across seeds ----
    C.write_csv(
        os.path.join(C.OUTPUT_DIR, "table_a2_feature_runs.csv"),
        ["scorer", "k", "seed", "accuracy", "fit_seconds", "n_kept", "features"],
        runs,
    )
    header3 = ["Features kept"] + [f"{s} (all seeds / any seed)" for s in available]
    rows3 = []
    for k in K_GRID:
        row = [str(k)]
        for s in available:
            sets = [
                set(r[6].split("|")) - {""} for r in runs if r[0] == s and r[1] == k
            ]
            row.append(f"{len(set.intersection(*sets))} / {len(set.union(*sets))}")
        rows3.append(row)
    C.emit(
        "table_a2b_selection_consistency",
        "Table A.2b -- Features every seed's fit keeps, vs features any seed keeps (PhiUSIIL)",
        header3,
        rows3,
        note=(
            "Same fits as Table A.2. 'all seeds' counts the features present in "
            "every one of the ten fitted models' top_features_ (after the 0.85 "
            "redundancy filter); 'any seed' counts the features used by at least "
            "one. A scorer whose selection depends only on the data, not on which "
            "80% split it saw, reads k / k."
        ),
    )


if __name__ == "__main__":
    main()
