"""N-CMAPSS DS02 feature pipeline, shared verbatim by the FIS and the network.

The point of this experiment is a *like-for-like* comparison, so every
preprocessing decision here is copied from `FuzzySystemsExperiments/
cmapss_rul_best.py` -- condition correction against the W channels fit on
training units' early cycles, the same two aggregations, the same per-unit
`hs`-derived RUL cap, the same StandardScaler. If the network wins or loses it
must be because of the model, not because it was handed different columns.

Two things are *deliberately different* from the DOE, and both tighten it:

1. **No test-unit information is ever computed.** The DOE's
   `physical_rul_cap(pd.concat([train_tab, test_tab]))` builds caps for the test
   engines too. They are never applied to the reported `rmse_test_true`, so the
   headline numbers are not leaky -- but the caps exist, and `cmapss_rul.py`'s
   `rmse_test_shaped` does use them. Here the cap dictionary is built from
   training units only and there is nothing to misuse.

2. **There is a validation split.** The DOE selects its Factor-D grid on
   `rmse_test_true` -- the official held-out test units. Every configuration
   quoted from it is therefore selected on the test set. This module carves two
   of the six dev units out as a validation fold so hyperparameters (for the
   network *and* for the FIS) can be chosen without touching units 11/14/15.

Loading + condition-correcting DS02 costs ~60 s and ~6 GB of peak RAM, so the
aggregated tables are cached to disk; everything downstream reads the cache.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_H5 = os.path.join(REPO, "data", "nasa-cmapps2", "N-CMAPSS_DS02-006.h5")
CACHE_DIR = os.path.join(REPO, "outputs", "nn-cmapss", "cache")

# DS02's official split, as published (and as `cmapss_rul_best.py` hardcodes).
TRAIN_UNITS = (2, 5, 10, 16, 18, 20)
TEST_UNITS = (11, 14, 15)
# Two of the six training engines, held out for model selection. Picked as the
# last two by unit number rather than by any score, so the choice cannot have
# been made after seeing which split flatters which arm.
VAL_UNITS = (18, 20)
FIT_UNITS = tuple(u for u in TRAIN_UNITS if u not in VAL_UNITS)

AGG_FUNCS = ["mean", "std", "min", "max", "last"]

# feature_set -> how many leading X_v channels to include. X_v is ordered
# [T40, P30, P45, W21, ...]; the first two are the 20-channel "condition
# monitoring signal" set the published DS02 CNN/MLP baselines use.
FEATURE_SET_XV = {"real": 0, "literature": 2, "all": None}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Condition correction (the preprocessing step the DOE found mattered most)
# ---------------------------------------------------------------------------
def fit_condition_correction(df, sensor_cols, condition_cols, baseline_cycles=15):
    order = df.groupby("unit").cumcount()
    baseline = df[order < baseline_cycles]
    X_base = baseline[condition_cols].to_numpy(dtype=np.float64)
    return {
        col: LinearRegression().fit(X_base, baseline[col].to_numpy(dtype=np.float64))
        for col in sensor_cols
    }


def apply_condition_correction(df, sensor_cols, condition_cols, models):
    df = df.copy()
    X_all = df[condition_cols].to_numpy(dtype=np.float64)
    for col in sensor_cols:
        df[col] = df[col].to_numpy(dtype=np.float64) - models[col].predict(X_all)
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_whole_cycle(df, feat_cols) -> pd.DataFrame:
    g = df.groupby(["unit", "cycle"], sort=True)
    feat = g[feat_cols].agg(AGG_FUNCS)
    feat.columns = ["_".join(c) for c in feat.columns]
    meta = g.agg(RUL=("RUL", "first"), hs=("hs", "min"))
    return feat.join(meta).reset_index()


def aggregate_raw_memory(df, feat_cols, stride: int = 200) -> pd.DataFrame:
    from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor

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


AGGREGATIONS = {
    "whole_cycle": aggregate_whole_cycle,
    "raw_memory": aggregate_raw_memory,
}


# ---------------------------------------------------------------------------
# RUL cap
# ---------------------------------------------------------------------------
def physical_rul_cap(table: pd.DataFrame) -> dict:
    """Per-unit RUL cap: the RUL at that unit's degradation onset, where onset is
    the first cycle its `hs` health flag drops to 0.

    `hs` is a simulator latent, not a sensor -- Stage 5 of the DOE shows a
    sensor-only moving-average detector can stand in for it at a cost. This
    experiment keeps the oracle cap because it is what the FIS numbers being
    compared against were produced with; swapping it would change both arms
    equally and confound the comparison under test.
    """
    caps = {}
    for unit, sub in table.groupby("unit"):
        sub = sub.sort_values("cycle")
        unhealthy = sub[sub["hs"] == 0]
        onset = unhealthy["cycle"].min() if len(unhealthy) else sub["cycle"].max()
        at_or_after = sub[sub["cycle"] >= onset]
        caps[unit] = float(
            at_or_after["RUL"].max() if len(at_or_after) else sub["RUL"].max()
        )
    return caps


def capped_rul(table: pd.DataFrame, caps: dict) -> np.ndarray:
    """Cap where a cap is known; leave the raw RUL where it is not.

    Only training-unit caps are ever built, so this is a no-op on any table of
    held-out engines -- which is the point: nothing about the test units'
    degradation onset can reach a target.
    """
    cap_series = table["unit"].map(caps).astype(float)
    raw = table["RUL"].astype(float)
    return np.where(cap_series.isna(), raw, np.minimum(raw, cap_series.fillna(raw)))


# ---------------------------------------------------------------------------
# The prepared bundle
# ---------------------------------------------------------------------------
@dataclass
class Split:
    """One (X, y, meta) block, all in the same scaled frame."""

    X: np.ndarray
    y: np.ndarray  # capped RUL where a cap exists (training target)
    y_true: np.ndarray  # uncapped ground-truth RUL (what test is scored on)
    unit: np.ndarray
    cycle: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Bundle:
    feature_names: list[str]
    fit: Split  # 4 dev engines -- used to train when selecting
    val: Split  # 2 dev engines -- used to select
    train: Split  # all 6 dev engines -- used for the final refit
    test: Split  # the 3 official held-out engines
    build_seconds: dict


def _scale(scaler, tab, feat_cols, caps):
    return Split(
        X=scaler.transform(tab[feat_cols].to_numpy(dtype=np.float64)),
        y=capped_rul(tab, caps),
        y_true=tab["RUL"].astype(float).to_numpy(),
        unit=tab["unit"].to_numpy(),
        cycle=tab["cycle"].to_numpy(),
    )


def build(
    h5_path: str = DEFAULT_H5,
    feature_set: str = "real",
    aggregation: str = "whole_cycle",
    stride: int = 200,
    verbose: bool = True,
) -> Bundle:
    p = print if verbose else (lambda *a, **k: None)
    seconds = {}

    t0 = time.perf_counter()
    data, var = load_h5(h5_path)
    df_dev = to_frame(data, var, "dev")
    df_test = to_frame(data, var, "test")
    del data
    seconds["load"] = time.perf_counter() - t0
    p(
        f"  load: {len(df_dev):,} dev + {len(df_test):,} test rows ({seconds['load']:.1f}s)"
    )

    w_cols = [f"W_{n}" for n in var["W"]]
    xs_cols = [f"Xs_{n}" for n in var["X_s"]]
    n_xv = FEATURE_SET_XV[feature_set]
    xv_cols = (
        [f"Xv_{n}" for n in var["X_v"]]
        if n_xv is None
        else [f"Xv_{n}" for n in var["X_v"][:n_xv]]
    )

    t0 = time.perf_counter()
    models = fit_condition_correction(df_dev, xs_cols + xv_cols, w_cols)
    df_dev = apply_condition_correction(df_dev, xs_cols + xv_cols, w_cols, models)
    df_test = apply_condition_correction(df_test, xs_cols + xv_cols, w_cols, models)
    seconds["condition_correction"] = time.perf_counter() - t0
    p(f"  condition correction: {seconds['condition_correction']:.1f}s")

    feat_cols = w_cols + xs_cols + xv_cols
    agg_fn = AGGREGATIONS[aggregation]
    kwargs = {"stride": stride} if aggregation == "raw_memory" else {}

    t0 = time.perf_counter()
    train_tab = agg_fn(df_dev, feat_cols, **kwargs)
    test_tab = agg_fn(df_test, feat_cols, **kwargs)
    seconds["aggregate"] = time.perf_counter() - t0
    p(
        f"  aggregate ({aggregation}): {len(train_tab):,} train / "
        f"{len(test_tab):,} test rows ({seconds['aggregate']:.1f}s)"
    )

    agg_feat_cols = [
        c for c in train_tab.columns if c not in ("unit", "cycle", "RUL", "hs")
    ]
    fit_tab = train_tab[train_tab["unit"].isin(FIT_UNITS)].reset_index(drop=True)
    val_tab = train_tab[train_tab["unit"].isin(VAL_UNITS)].reset_index(drop=True)

    # Two scalers and two cap dictionaries, each fit on exactly the rows its own
    # arm is allowed to see: the selection scaler never sees the validation
    # engines, the final scaler never sees the test engines.
    sc_fit = StandardScaler().fit(fit_tab[agg_feat_cols].to_numpy(dtype=np.float64))
    sc_all = StandardScaler().fit(train_tab[agg_feat_cols].to_numpy(dtype=np.float64))
    caps_fit = physical_rul_cap(fit_tab)
    caps_all = physical_rul_cap(train_tab)

    bundle = Bundle(
        feature_names=agg_feat_cols,
        fit=_scale(sc_fit, fit_tab, agg_feat_cols, caps_fit),
        val=_scale(sc_fit, val_tab, agg_feat_cols, caps_fit),
        train=_scale(sc_all, train_tab, agg_feat_cols, caps_all),
        test=_scale(sc_all, test_tab, agg_feat_cols, caps_all),
        build_seconds=seconds,
    )
    p(
        f"  splits: fit={len(bundle.fit)} val={len(bundle.val)} "
        f"train={len(bundle.train)} test={len(bundle.test)}  "
        f"features={len(agg_feat_cols)}"
    )
    return bundle


def cache_path(feature_set: str, aggregation: str, stride: int) -> str:
    tag = f"{feature_set}_{aggregation}"
    if aggregation == "raw_memory":
        tag += f"_s{stride}"
    return os.path.join(CACHE_DIR, f"ds02_{tag}.pkl")


def load_or_build(
    feature_set: str = "real",
    aggregation: str = "whole_cycle",
    stride: int = 200,
    h5_path: str = DEFAULT_H5,
    verbose: bool = True,
) -> Bundle:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(feature_set, aggregation, stride)
    if os.path.exists(path):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        if verbose:
            print(f"  (cached: {os.path.relpath(path, REPO)})")
        return bundle
    bundle = build(h5_path, feature_set, aggregation, stride, verbose)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


# The DS02 configurations this experiment runs. `honest` and `best` are named
# as `cmapss_rul_best.py` names them -- real sensors, one row per flight cycle;
# and the literature's 20-channel set through the memory-window extractor.
# `memory18` is the FIS-quality recommendation (see FIS_QUALITY.md): the strict
# 18 real sensors *with* memory features, which matches `best`'s accuracy and
# smoothness without the two virtual channels.
BUNDLES = {
    "honest": dict(feature_set="real", aggregation="whole_cycle"),
    "best": dict(feature_set="literature", aggregation="raw_memory"),
    "memory18": dict(feature_set="real", aggregation="raw_memory"),
}


if __name__ == "__main__":
    import sys

    # Import self rather than using the definitions in `__main__`: a Bundle
    # pickled from `__main__` cannot be unpickled by any other script.
    import cmapss_data

    which = sys.argv[1] if len(sys.argv) > 1 else "honest"
    cfg = cmapss_data.BUNDLES[which]
    print(f"Building DS02 bundle: {which} {cfg}")
    b = cmapss_data.load_or_build(**cfg)
    print(f"  build_seconds={b.build_seconds}")
    for name in ("fit", "val", "train", "test"):
        s = getattr(b, name)
        print(
            f"  {name:6s} n={len(s):6d}  units={sorted(set(s.unit.tolist()))}  "
            f"y_true[{s.y_true.min():.0f}, {s.y_true.max():.0f}]  "
            f"y_capped[{s.y.min():.0f}, {s.y.max():.0f}]"
        )
