"""Is the NumPy trainer the limiting factor? Three off-the-shelf models say no.

Every network number in this benchmark comes from the ~60-line Adam loop in
`fis2nn.train_adam`, which exists so that no arm can win on a framework default.
The obvious objection is that it might be *bad* -- that the network results
understate what a network does on DS02, or overstate it.

sklearn's `MLPRegressor` (a different optimizer, different init, different
stopping rule), a gradient-boosted tree, and a random forest answer that. They
are selected on the same validation engines and scored on the same test
engines, and they are not arms in the comparison -- they are a check on the
instrument.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np

import cmapss_data
import models
import metrics

OUT = os.path.join(cmapss_data.REPO, "outputs", "nn-cmapss")


def run(which: str, seeds=(0, 1, 2)) -> list:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor

    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which])
    names = b.feature_names
    fis_fit, _ = models.fit_fis(b.fit.X, b.fit.y, names, **models.FIS_CONFIGS[which])
    fis_all, _ = models.fit_fis(
        b.train.X, b.train.y, names, **models.FIS_CONFIGS[which]
    )
    sel_cols = np.array([names.index(f) for f in fis_fit.top_features_], dtype=int)
    fin_cols = np.array([names.index(f) for f in fis_all.top_features_], dtype=int)
    every = np.arange(len(names))

    grids = {
        "sklearn-MLP": [
            dict(
                hidden_layer_sizes=h,
                alpha=a,
                learning_rate_init=lr,
                max_iter=1500,
                early_stopping=False,
            )
            for h in ((8,), (32,), (64,), (32, 32))
            for a in (1e-4, 1e-2, 1.0)
            for lr in (1e-3, 1e-2)
        ],
        "hist-gbm": [
            dict(max_iter=n, learning_rate=lr, max_leaf_nodes=leaves)
            for n in (100, 300)
            for lr in (0.05, 0.1)
            for leaves in (7, 31)
        ],
        "random-forest": [
            dict(n_estimators=n, min_samples_leaf=leaf, n_jobs=-1)
            for n in (200,)
            for leaf in (1, 5, 20)
        ],
    }
    makers = {
        "sklearn-MLP": lambda p, s: MLPRegressor(random_state=s, **p),
        "hist-gbm": lambda p, s: HistGradientBoostingRegressor(random_state=s, **p),
        "random-forest": lambda p, s: RandomForestRegressor(random_state=s, **p),
    }

    rows = []
    for name, grid in grids.items():
        for space, sc, fc in (("fis", sel_cols, fin_cols), ("all", every, every)):
            best_p, best_v = None, np.inf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for p in grid:
                    m = makers[name](p, seeds[0]).fit(b.fit.X[:, sc], b.fit.y)
                    v = float(
                        np.sqrt(
                            np.mean((b.val.y_true - m.predict(b.val.X[:, sc])) ** 2)
                        )
                    )
                    if v < best_v:
                        best_p, best_v = p, v
                per_seed = []
                for s in seeds:
                    t0 = time.perf_counter()
                    m = makers[name](best_p, s).fit(b.train.X[:, fc], b.train.y)
                    fit_s = time.perf_counter() - t0
                    per_seed.append(
                        (fit_s, metrics.evaluate(b.test, m.predict(b.test.X[:, fc])))
                    )
            row = dict(
                model=f"{name} ({space})",
                bundle=which,
                params=str(best_p),
                val_rmse=best_v,
                fit_seconds=float(np.median([s for s, _ in per_seed])),
                test=dict(
                    rmse=float(np.median([e["rmse"] for _, e in per_seed])),
                    mae=float(np.median([e["mae"] for _, e in per_seed])),
                    rmse_endpoint=float(
                        np.median([e["rmse_endpoint"] for _, e in per_seed])
                    ),
                    nasa=float(np.median([e["nasa"] for _, e in per_seed])),
                ),
            )
            rows.append(row)
            print(
                f"  {row['model']:26s} val {best_v:6.2f}  test {row['test']['rmse']:6.2f}  "
                f"fit {row['fit_seconds']:7.3f}s"
            )
    return rows


if __name__ == "__main__":
    import sys

    out = []
    for which in sys.argv[1:] or ["honest", "best"]:
        print(f"=== external baselines: {which} ===")
        out.extend(run(which))
    path = os.path.join(OUT, "external_baselines.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.relpath(path, cmapss_data.REPO)}")
