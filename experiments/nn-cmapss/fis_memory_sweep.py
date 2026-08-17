"""Tune the FIS's memory window for the accuracy/smoothness trade-off.

`fis_quality.py` established the mechanism: on the real 18-sensor set, replacing
per-cycle aggregates with `tribblefis`'s `MemoryWindowFeatureExtractor` cuts
per-engine RMSE 38% and up-cycle noise 63% at once -- both axes, still a genuine
FIS. This asks how far that goes: the window's `window_size` / `memory_size` set
how many cycles the FIS averages over, so they *are* the accuracy/smoothness
knob. Bigger windows should keep smoothing the trajectory; the question is where
accuracy starts to pay for it.

The 2.4 GB HDF5 is loaded and condition-corrected once; only the (cheap) memory
extraction and FIS fit re-run per setting.
"""

from __future__ import annotations

import json
import os
import warnings

import pandas as pd

import cmapss_data as D
import models
import monotone as M
import report

OUT = report.OUT


def corrected_frames(feature_set="real"):
    """Load DS02 and condition-correct once; return dev/test frames + columns."""
    data, var = D.load_h5(D.DEFAULT_H5)
    df_dev, df_test = D.to_frame(data, var, "dev"), D.to_frame(data, var, "test")
    del data
    w = [f"W_{n}" for n in var["W"]]
    xs = [f"Xs_{n}" for n in var["X_s"]]
    n_xv = D.FEATURE_SET_XV[feature_set]
    xv = (
        []
        if n_xv == 0
        else (
            [f"Xv_{n}" for n in var["X_v"]]
            if n_xv is None
            else [f"Xv_{n}" for n in var["X_v"][:n_xv]]
        )
    )
    models_cc = D.fit_condition_correction(df_dev, xs + xv, w)
    df_dev = D.apply_condition_correction(df_dev, xs + xv, w, models_cc)
    df_test = D.apply_condition_correction(df_test, xs + xv, w, models_cc)
    return df_dev, df_test, w + xs + xv


def memory_tables(df, feat_cols, window, memory, stride):
    """One row per subsampled sample, with short/long-term memory features."""
    from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor

    ext = MemoryWindowFeatureExtractor(window_size=window, memory_size=memory)
    frames = []
    for unit, sub in df.groupby("unit", sort=True):
        sub = sub.iloc[::stride].reset_index(drop=True)
        mem = ext.prepare_sequences(sub, feat_cols, include_time=False)
        mem["unit"] = unit
        mem["cycle"] = sub["cycle"].values
        mem["RUL"] = sub["RUL"].values
        mem["hs"] = sub["hs"].values
        frames.append(mem)
    out = pd.concat(frames, ignore_index=True)
    cols = [c for c in out.columns if c not in ("unit", "cycle", "RUL", "hs")]
    out[cols] = out[cols].bfill().ffill()
    return out, cols


def evaluate(df_dev, df_test, feat_cols, window, memory, stride):
    from sklearn.preprocessing import StandardScaler

    train_tab, agg_cols = memory_tables(df_dev, feat_cols, window, memory, stride)
    test_tab, _ = memory_tables(df_test, feat_cols, window, memory, stride)

    caps = D.physical_rul_cap(train_tab)  # training units only
    sc = StandardScaler().fit(train_tab[agg_cols].to_numpy(float))
    Xtr = sc.transform(train_tab[agg_cols].to_numpy(float))
    Xte = sc.transform(test_tab[agg_cols].to_numpy(float))
    ytr = D.capped_rul(train_tab, caps)

    fis, fit_s = models.fit_fis(Xtr, ytr, agg_cols, **models.FIS_CONFIGS["best"])
    pred = models.fis_predict(fis, Xte, agg_cols)

    g = M.per_cycle(
        test_tab["unit"].to_numpy(),
        test_tab["cycle"].to_numpy(),
        test_tab["RUL"].astype(float).to_numpy(),
        pred,
    )
    agg = M.aggregate(
        [
            M.score_engine(s.true.to_numpy(), s.pred.to_numpy())
            for _, s in g.groupby("unit")
        ]
    )
    agg.update(
        window=window,
        memory=memory,
        stride=stride,
        fit_seconds=fit_s,
        n_features=Xtr.shape[1],
        n_train=len(train_tab),
    )
    return agg


# (window, memory, stride). window=5/memory=2/stride=200 is the shipped `best`.
GRID = [
    (5, 2, 200),
    (10, 5, 200),
    (20, 10, 200),
    (40, 20, 200),
    (20, 10, 100),
    (40, 20, 100),
    (80, 40, 100),
]


def main(feature_set="real") -> None:
    warnings.simplefilter("ignore")
    print(f"Loading + condition-correcting DS02 ({feature_set}) once ...")
    df_dev, df_test, feat_cols = corrected_frames(feature_set)
    print(
        f"  {len(df_dev):,} dev + {len(df_test):,} test rows, {len(feat_cols)} channels"
    )

    rows = []
    for window, memory, stride in GRID:
        r = evaluate(df_dev, df_test, feat_cols, window, memory, stride)
        rows.append(r)
        span = window + memory
        print(
            f"  w={window:3d} m={memory:3d} stride={stride:3d}  "
            f"(~{span} samples/window)  n_train={r['n_train']:6d}  "
            f"rmse={r['rmse']:6.2f}  up%={r['up_frac']*100:3.0f}  "
            f"pos_tv={r['pos_tv']:6.1f}  fit={r['fit_seconds']:.2f}s"
        )

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fis_memory_sweep.json")
    with open(path, "w") as f:
        json.dump({"feature_set": feature_set, "rows": rows}, f, indent=1)
    print(f"\nwrote {os.path.relpath(path, D.REPO)}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "real")
