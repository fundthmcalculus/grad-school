"""Best-known N-CMAPSS DS02 RUL pipeline using TribbleRegressor.

Standalone and self-contained -- this is meant to be copied out and shared
or run on its own, without the rest of the grad-school DOE repo. It only
needs: h5py, numpy, pandas, scikit-learn, and tribble-fis (`pip install
tribble-fis`, or `pip install -e /path/to/tribble-fis` for the dev version).

Two pipelines, both training in single-digit seconds:

  --pipeline honest  (default)  Real sensors only (W + X_s). RMSE ~11.2 on
                                 the official held-out test units (11, 14,
                                 15). This is the number to trust for "could
                                 this actually fly on an aircraft."

  --pipeline best                Adds X_v, the dataset's "virtual sensors" --
                                 simulator-internal quantities the source
                                 paper explicitly excludes from condition-
                                 monitoring signals. RMSE ~6.5, better than
                                 the published CNN baseline (7.22) on this
                                 dataset -- but NOT reproducible with a real
                                 aircraft's actual instrumentation. A
                                 sensitivity/upper-bound result, not a
                                 deployment claim.

Both pipelines depend on one preprocessing step that turned out to matter
far more than any model hyperparameter: regressing each real sensor channel
against the W operating-condition channels (altitude, Mach, throttle,
inlet temp) using only each training engine's own early "healthy" cycles,
then using the *residual* -- not the raw reading -- as the model's input.
Raw per-cycle sensor means are dominated by flight-to-flight operating-
condition swings, not the (much smaller) degradation trend; this removes
that confound before the regressor ever sees the data. Full derivation and
the negative result that led here (a naive raw-mean detector that never
fired) are in cmapss_rul.py / the grad-school PR history.

Usage:
    python cmapss_rul_best.py --h5 /path/to/N-CMAPSS_DS02-006.h5
    python cmapss_rul_best.py --h5 /path/to/N-CMAPSS_DS02-006.h5 --pipeline best
"""

import argparse
import contextlib
import io
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor

TRAIN_UNITS = (2, 5, 10, 16, 18, 20)
TEST_UNITS = (11, 14, 15)

PIPELINES = {
    # aggregation: "whole_cycle" (one row/cycle, mean/std/min/max/last stats)
    # or "raw_memory" (subsampled raw stream through MemoryWindowFeatureExtractor)
    "honest": dict(
        feature_set="real", aggregation="whole_cycle",
        tribble_kwargs=dict(
            tsk_order="1st", n_gaussians=0, top_p=0.9, detect_interactions=False,
            norm_conorm="hamacher", l2_reg=0.01,
        ),
        expected_rmse=11.23,
    ),
    "best": dict(
        feature_set="all", aggregation="raw_memory",
        tribble_kwargs=dict(
            tsk_order="full-2nd", n_gaussians=0, top_p=0.95, detect_interactions=False,
            norm_conorm="hamacher", l2_reg=0.01,
        ),
        expected_rmse=6.54,
    ),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_h5(path: str) -> tuple[dict, dict]:
    data, var = {}, {}
    with h5py.File(path, "r") as f:
        for split in ("dev", "test"):
            data[split] = {
                "W": f[f"W_{split}"][:],
                "X_s": f[f"X_s_{split}"][:],
                "X_v": f[f"X_v_{split}"][:],
                "A": f[f"A_{split}"][:],
                "Y": f[f"Y_{split}"][:, 0].astype(np.float64),
            }
        for group in ("W", "X_s", "X_v"):
            var[group] = [v.decode() for v in f[f"{group}_var"][:]]
    return data, var


def to_frame(data: dict, var: dict, split: str) -> pd.DataFrame:
    d = data[split]
    df = pd.DataFrame(
        {
            "unit": d["A"][:, 0].astype(int),
            "cycle": d["A"][:, 1].astype(int),
            "hs": d["A"][:, 3],
            "RUL": d["Y"],
        }
    )
    for i, name in enumerate(var["W"]):
        df[f"W_{name}"] = d["W"][:, i]
    for i, name in enumerate(var["X_s"]):
        df[f"Xs_{name}"] = d["X_s"][:, i]
    for i, name in enumerate(var["X_v"]):
        df[f"Xv_{name}"] = d["X_v"][:, i]
    return df


# --------------------------------------------------------------------------
# Condition correction: the preprocessing step that matters most
# --------------------------------------------------------------------------
def fit_condition_correction(df: pd.DataFrame, sensor_cols, condition_cols, baseline_cycles=15):
    order = df.groupby("unit").cumcount()
    baseline = df[order < baseline_cycles]
    X_base = baseline[condition_cols].to_numpy(dtype=np.float64)
    return {
        col: LinearRegression().fit(X_base, baseline[col].to_numpy(dtype=np.float64))
        for col in sensor_cols
    }


def apply_condition_correction(df: pd.DataFrame, sensor_cols, condition_cols, models) -> pd.DataFrame:
    df = df.copy()
    X_all = df[condition_cols].to_numpy(dtype=np.float64)
    for col in sensor_cols:
        df[col] = df[col].to_numpy(dtype=np.float64) - models[col].predict(X_all)
    return df


# --------------------------------------------------------------------------
# RUL target: per-unit physical cap, derived from the `hs` health-state flag
# --------------------------------------------------------------------------
def physical_rul_cap(table: pd.DataFrame) -> dict:
    caps = {}
    for unit, sub in table.groupby("unit"):
        sub = sub.sort_values("cycle")
        unhealthy = sub[sub["hs"] == 0]
        onset = unhealthy["cycle"].min() if len(unhealthy) else sub["cycle"].max()
        at_or_after = sub[sub["cycle"] >= onset]
        caps[unit] = float(at_or_after["RUL"].max() if len(at_or_after) else sub["RUL"].max())
    return caps


def capped_rul(table: pd.DataFrame, caps: dict) -> np.ndarray:
    cap_series = table["unit"].map(caps)
    return np.minimum(table["RUL"].astype(float), cap_series).to_numpy()


# --------------------------------------------------------------------------
# Aggregation: one row per (unit, cycle), or a subsampled raw-memory stream
# --------------------------------------------------------------------------
AGG_FUNCS = ["mean", "std", "min", "max", "last"]


def aggregate_whole_cycle(df: pd.DataFrame, feat_cols) -> pd.DataFrame:
    g = df.groupby(["unit", "cycle"], sort=True)
    feat = g[feat_cols].agg(AGG_FUNCS)
    feat.columns = ["_".join(c) for c in feat.columns]
    meta = g.agg(RUL=("RUL", "first"), hs=("hs", "min"))
    return feat.join(meta).reset_index()


def aggregate_raw_memory(df: pd.DataFrame, feat_cols, stride: int = 200) -> pd.DataFrame:
    extractor = MemoryWindowFeatureExtractor(window_size=5, memory_size=2)
    frames = []
    for unit, sub in df.groupby("unit", sort=True):
        sub = sub.iloc[::stride].reset_index(drop=True)
        mem = extractor.prepare_sequences(sub, feat_cols, include_time=False)
        mem["unit"] = unit
        mem["cycle"] = sub["cycle"].values
        mem["RUL"] = sub["RUL"].values
        mem["hs"] = sub["hs"].values
        frames.append(mem)
    result = pd.concat(frames, ignore_index=True)
    mem_cols = [c for c in result.columns if c not in ("unit", "cycle", "RUL", "hs")]
    result[mem_cols] = result[mem_cols].bfill().ffill()
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def nasa_score(y_true, y_pred) -> float:
    delta = np.asarray(y_true) - np.asarray(y_pred)
    alpha = np.where(delta > 0, 1.0 / 13.0, 1.0 / 10.0)
    return float(np.sum(np.exp(alpha * np.abs(delta))))


def run(h5_path: str, pipeline_name: str):
    cfg = PIPELINES[pipeline_name]
    print(f"Pipeline: {pipeline_name!r} ({cfg['feature_set']} sensors, "
          f"{cfg['aggregation']} aggregation) -- expected RMSE ~{cfg['expected_rmse']}")

    print(f"Loading {h5_path} ...")
    t0 = time.perf_counter()
    data, var = load_h5(h5_path)
    df_dev = to_frame(data, var, "dev")
    df_test = to_frame(data, var, "test")
    print(f"  {len(df_dev):,} dev rows, {len(df_test):,} test rows ({time.perf_counter()-t0:.1f}s)")

    w_cols = [f"W_{n}" for n in var["W"]]
    xs_cols = [f"Xs_{n}" for n in var["X_s"]]
    xv_cols = [f"Xv_{n}" for n in var["X_v"]]
    correct_cols = xs_cols + (xv_cols if cfg["feature_set"] == "all" else [])

    print("Fitting condition correction on dev-unit early cycles ...")
    models = fit_condition_correction(df_dev, correct_cols, w_cols)
    df_dev = apply_condition_correction(df_dev, correct_cols, w_cols, models)
    df_test = apply_condition_correction(df_test, correct_cols, w_cols, models)

    feat_cols = w_cols + xs_cols + (xv_cols if cfg["feature_set"] == "all" else [])
    agg_fn = aggregate_whole_cycle if cfg["aggregation"] == "whole_cycle" else aggregate_raw_memory

    print(f"Aggregating ({cfg['aggregation']}) ...")
    t0 = time.perf_counter()
    train_tab = agg_fn(df_dev, feat_cols)
    test_tab = agg_fn(df_test, feat_cols)
    print(f"  {len(train_tab)} train rows, {len(test_tab)} test rows ({time.perf_counter()-t0:.1f}s)")

    agg_feat_cols = [c for c in train_tab.columns if c not in ("unit", "cycle", "RUL", "hs")]
    caps = physical_rul_cap(pd.concat([train_tab, test_tab], ignore_index=True))

    X_train = train_tab[agg_feat_cols].to_numpy(dtype=np.float64)
    X_test = test_tab[agg_feat_cols].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    y_train = capped_rul(train_tab, caps)
    y_test_true = test_tab["RUL"].astype(float).to_numpy()

    model = TribbleRegressor(random_state=42, max_samples=2000, **cfg["tribble_kwargs"])
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    pred_test = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test_true, pred_test)))
    score = nasa_score(y_test_true, pred_test)

    print(f"\n=== {pipeline_name} pipeline ===")
    print(f"fit_seconds={fit_seconds:.2f}  rmse_test_true={rmse:.2f}  nasa_score={score:.1f}")
    for unit in sorted(test_tab["unit"].unique()):
        m = (test_tab["unit"] == unit).to_numpy()
        u_rmse = float(np.sqrt(mean_squared_error(y_test_true[m], pred_test[m])))
        print(f"  unit {unit}: n={m.sum():4d}  rmse={u_rmse:.2f}")
    return rmse, fit_seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", required=True, help="Path to N-CMAPSS_DS02-006.h5")
    parser.add_argument("--pipeline", choices=list(PIPELINES), default="honest")
    args = parser.parse_args()
    run(args.h5, args.pipeline)
