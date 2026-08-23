#!/usr/bin/env python3
"""Bisect the OTHER drift in Table 4.1 -- the one B14 does not explain.

Table 4.1's three regression rows moved between the archived run of record and the
current pin, and they move IDENTICALLY with and without the `wasserstein_distance`
correction, so they have a separate cause:

    Concrete R2             0.795 +/- 0.025  ->  0.808 +/- 0.030
    Concrete full-2nd R2    0.852 +/- 0.030  ->  0.867 +/- 0.031
    Bike Sharing R2         0.939 +/- 0.004  ->  0.960 +/- 0.003

Checklist D8. This is the same exercise `diagnose_wasserstein_regression.py` ran,
with two changes that make it cheaper the second time:

  * the FEATURES are frozen, not the raw data. X and y are written once, already
    normalized by the current pin, so the probe isolates the *model* from the
    scaler. If no commit in the range explains the move, the cause is in
    `tribblefis.scaling` rather than in the regressor, and that is a result too.
  * the probe reports mean R2 over ten seeds, which is what the table reports.

    # freeze the features once, from the current pin
    uv run --project tribble-fis python \
        reproduce/experiments/diagnose_regression_drift.py --freeze

    # then probe any commit
    uv run --no-project --python 3.13 \
        --with "tribble-fis @ git+https://github.com/fundthmcalculus/tribble-fis@<sha>" \
        --with scikit-learn \
        python reproduce/experiments/diagnose_regression_drift.py --probe
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FROZEN = os.path.join(os.environ.get("TEMP", "/tmp"), "concrete_frozen.npz")


def freeze() -> int:
    sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))
    import _fuzzy_models as FM

    data = FM.load_concrete()
    if data is None:
        print("Concrete unavailable", file=sys.stderr)
        return 2
    X, y = data
    Xt, logged = FM.normalize(X)
    np.savez(
        FROZEN,
        X=np.asarray(Xt, dtype=float),
        y=np.asarray(y, dtype=float),
        cols=np.array(list(Xt.columns), dtype=object),
    )
    print(f"froze {Xt.shape} -> {FROZEN}   logged={logged}")
    return 0


def probe() -> int:
    import pandas as pd
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    import tribblefis
    from tribblefis.gaussian_regressor import TribbleRegressor

    d = np.load(FROZEN, allow_pickle=True)
    X = pd.DataFrame(d["X"], columns=list(d["cols"]))
    y = pd.Series(d["y"])

    print(f"tribblefis at {tribblefis.__file__}")
    for order in ("1st", "full-2nd"):
        scores = []
        for seed in range(10):
            xtr, xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            model = TribbleRegressor(
                n_output_buckets=3, tsk_order=order, top_n=-1, random_state=seed
            )
            model.fit(xtr, ytr)
            scores.append(r2_score(yte, np.asarray(model.predict(xte))))
        a = np.asarray(scores)
        print(f"RESULT {order:9s} R2 = {a.mean():.4f} +/- {a.std():.4f}")
    return 0


def isolate() -> int:
    """Restore each function #95 replaced, one at a time, at the current pin."""
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import r2_score
    from sklearn.metrics import silhouette_score as sk_sil
    from sklearn.model_selection import train_test_split
    from scipy import stats as sp_stats
    from scipy.spatial.distance import jensenshannon as sp_js
    from scipy.stats import wasserstein_distance as sp_wass

    from tribblefis import gauss_math
    from tribblefis.gaussian_regressor import TribbleRegressor

    d = np.load(FROZEN, allow_pickle=True)
    X = pd.DataFrame(d["X"], columns=list(d["cols"]))
    y = pd.Series(d["y"])

    def _sk_km(data, k, random_state):
        if k <= 1:
            return np.zeros(len(data), dtype=int)
        return KMeans(n_clusters=k, random_state=random_state).fit_predict(
            np.asarray(data).reshape(-1, 1)
        )

    repl = {
        "norm_fit": lambda a: sp_stats.norm.fit(a),
        "norm_pdf": lambda x, mu, sd: sp_stats.norm.pdf(x, mu, sd),
        "jensenshannon_distance": sp_js,
        "wasserstein_distance": sp_wass,
        "silhouette_score": sk_sil,
        "_kmeans_labels_1d": _sk_km,
    }
    orig = {k: getattr(gauss_math, k) for k in repl}

    def measure(label):
        cols = []
        for order in ("1st", "full-2nd"):
            s = []
            for seed in range(10):
                xtr, xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=seed
                )
                m = TribbleRegressor(
                    n_output_buckets=3, tsk_order=order, top_n=-1, random_state=seed
                )
                m.fit(xtr, ytr)
                s.append(r2_score(yte, np.asarray(m.predict(xte))))
            a = np.asarray(s)
            cols.append(f"{order}={a.mean():.4f}+/-{a.std():.4f}")
        print(f"  {label:36s} " + "   ".join(cols))

    print(
        "  archive (tribble-fis 80e98d7)        1st=0.7950+/-0.0249   full-2nd=0.8517+/-0.0297"
    )
    measure("current pin, unmodified")
    for k in repl:
        setattr(gauss_math, k, repl[k])
        measure(f"restore {k}")
        setattr(gauss_math, k, orig[k])
    for k in repl:
        setattr(gauss_math, k, repl[k])
    measure("restore ALL")
    return 0


def main() -> int:
    if "--freeze" in sys.argv:
        return freeze()
    if "--probe" in sys.argv:
        return probe()
    if "--isolate" in sys.argv:
        return isolate()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
