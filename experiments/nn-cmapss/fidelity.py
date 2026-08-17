"""Why the hot start does not fire on DS02: measure the additivity boundary.

The fis-to-neural-net study's closing claim is that the axis-aligned conversion
is exact exactly when the FIS is additive, and that fidelity therefore degrades
with input dimension (it reports 0.030 relative at one input, 0.294 at eight,
>1.0 at twelve). On DS02 the conversion lands at 2.2 relative on the 21-feature
`honest` FIS and 17.7 on the `best` one -- far past anything that study saw.

A number that large is either the claim continuing to hold, or a broken
conversion. This script decides which, by forcing the FIS down to a handful of
features (`top_n`) and watching fidelity as dimension falls. If the machinery
is sound, fidelity must approach zero as the FIS approaches one input, because
in one dimension the partial-dependence profile *is* the FIS and the
decomposition is the equivalence itself.

It also separates the two ways an additive seed can be wrong:

  interaction  the FIS is genuinely not a sum of one-dimensional functions --
               irreducible, and what the study is about.
  clipping     the seed's ReLU knots do not span the data, so rows outside the
               knot range are served by a linear extrapolation. Fixable in
               principle, so it must be ruled out before blaming interaction.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import cmapss_data
import models

OUT = os.path.join(cmapss_data.REPO, "outputs", "nn-cmapss")

# `top_n` caps the FIS's feature count directly (top_p is a cumulative-
# importance threshold and cannot be pushed to exactly 1 or 2 features).
TOP_N = (1, 2, 3, 4, 6, 8, 12, 16, 24, -1)


def additive_reference(fis_pred_fn, X, feature_index, background_rows):
    """The best additive approximation to the FIS, evaluated by its own ANOVA
    projection on a *dense* grid rather than at the FIS's knots.

    If the seed is much worse than this, the loss is the knot grid (clipping or
    resolution). If they agree, the loss is interaction, and no axis-aligned
    initialization of any width can recover it.
    """
    n_grid = 33
    Xb = X[background_rows]
    base = float(np.mean(fis_pred_fn(Xb)))
    profiles = {}
    for f in feature_index:
        grid = np.quantile(X[:, f], np.linspace(0.0, 1.0, n_grid))
        vals = np.empty(n_grid)
        for i, t in enumerate(grid):
            Xt = Xb.copy()
            Xt[:, f] = t
            vals[i] = float(np.mean(fis_pred_fn(Xt)))
        profiles[f] = (grid, vals)

    def predict(Xq):
        out = np.full(len(Xq), base)
        for f, (grid, vals) in profiles.items():
            out += np.interp(Xq[:, f], grid, vals) - base
        return out

    return predict


def run(which: str, seeds=(0,)) -> None:
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which])
    names = b.feature_names
    base_kwargs = dict(models.FIS_CONFIGS[which])
    base_kwargs.pop("top_p", None)

    y_center = float(np.mean(b.train.y))
    y_scale = float(np.std(b.train.y)) or 1.0
    rng = np.random.default_rng(0)
    bg = rng.choice(len(b.train), min(256, len(b.train)), replace=False)

    rows = []
    for top_n in TOP_N:
        kwargs = dict(base_kwargs)
        if top_n > 0:
            kwargs["top_n"] = top_n
        else:
            kwargs["top_p"] = models.FIS_CONFIGS[which]["top_p"]
        fis, fis_s = models.fit_fis(b.train.X, b.train.y, names, **kwargs)
        conv = models.Conversion(fis, names, b.train.X, y_center, y_scale)

        pred_fis = models.fis_predict(fis, b.test.X, names)
        seed_pred = conv.net.predict(conv.subspace(b.test.X)) * y_scale + y_center
        sd = float(np.std(pred_fis)) or 1.0

        def fis_fn(X):
            return models.fis_predict(fis, X, names)

        add = additive_reference(fis_fn, b.train.X, conv.index, bg)
        add_pred = add(b.test.X)

        # Fraction of test rows whose value on some FIS feature falls outside
        # that feature's knot range -- rows the seed can only extrapolate to.
        outside = np.zeros(len(b.test), dtype=bool)
        for j, f in enumerate(conv.features):
            k = conv.knots[f]
            if k.size:
                col = b.test.X[:, conv.index[j]]
                outside |= (col < k.min()) | (col > k.max())

        row = dict(
            top_n=top_n,
            n_features=len(conv.features),
            n_hidden=conv.n_hidden,
            fis_seconds=fis_s,
            fis_test_rmse=models.evaluate(b.test, pred_fis)["rmse"],
            seed_test_rmse=models.evaluate(b.test, seed_pred)["rmse"],
            fidelity_rmse=float(np.sqrt(np.mean((seed_pred - pred_fis) ** 2))),
            fidelity_relative=float(np.sqrt(np.mean((seed_pred - pred_fis) ** 2)) / sd),
            additive_rmse=float(np.sqrt(np.mean((add_pred - pred_fis) ** 2))),
            additive_relative=float(np.sqrt(np.mean((add_pred - pred_fis) ** 2)) / sd),
            frac_rows_outside_knots=float(outside.mean()),
        )
        rows.append(row)
        print(
            f"  top_n={top_n:3d} -> {row['n_features']:3d} features, "
            f"{row['n_hidden']:4d} knots | seed-vs-FIS {row['fidelity_relative']:7.3f} rel "
            f"({row['fidelity_rmse']:8.2f} cyc) | best-additive {row['additive_relative']:7.3f} rel "
            f"| outside knots {row['frac_rows_outside_knots']:.1%}"
        )

    path = os.path.join(OUT, f"fidelity_{which}.json")
    with open(path, "w") as f:
        json.dump(dict(bundle=which, rows=rows), f, indent=1)
    print(f"wrote {os.path.relpath(path, cmapss_data.REPO)}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    for which in sys.argv[1:] or ["honest", "best"]:
        print(f"=== conversion fidelity vs FIS dimension: {which} ===")
        run(which)
