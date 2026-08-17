"""The processing-engine steps, as plain functions.

`TribblePredictiveHealth` (in `pipeline.py`) is the class that composes these;
they are kept separate and importable so a step can be reused or tested on its
own. Each operates on a tidy per-sample DataFrame with, at minimum, an engine
column, a cycle column, a health flag, and the sensor / operating-condition
columns -- nothing here is specific to a single dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

AGG_FUNCS = ["mean", "std", "min", "max", "last"]


# ---------------------------------------------------------------------------
# Condition correction
# ---------------------------------------------------------------------------
def fit_condition_correction(
    dev, sensor_cols, condition_cols, unit_col="unit", baseline_cycles=15
):
    """Learn, per sensor, its dependence on the operating condition -- from each
    training engine's first `baseline_cycles` (presumed-healthy) cycles only."""
    early = dev[dev.groupby(unit_col).cumcount() < baseline_cycles]
    X = early[condition_cols].to_numpy(float)
    return {c: LinearRegression().fit(X, early[c].to_numpy(float)) for c in sensor_cols}


def apply_condition_correction(df, sensor_cols, condition_cols, models):
    """Replace each sensor with its residual after removing the fitted condition
    dependence. Fit on training data only, applied to both splits."""
    df = df.copy()
    X = df[condition_cols].to_numpy(float)
    for c in sensor_cols:
        df[c] = df[c].to_numpy(float) - models[c].predict(X)
    return df


# ---------------------------------------------------------------------------
# Feature aggregation (the two strategies)
# ---------------------------------------------------------------------------
def build_memory_features(
    df,
    sensor_cols,
    unit_col="unit",
    cycle_col="cycle",
    health_col="health",
    rul_col="rul",
    stride=200,
    window_size=5,
    memory_size=2,
):
    """Subsample every `stride`-th sample within each engine, then attach a
    short- and long-term rolling average of every sensor (one row per subsampled
    sample). The rolling memory is what lets a per-cycle-independent model track
    a slow trend smoothly. Returns (frame, feature_cols)."""
    from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor

    extractor = MemoryWindowFeatureExtractor(
        window_size=window_size, memory_size=memory_size
    )
    carry = [c for c in (unit_col, cycle_col, health_col, rul_col) if c in df.columns]
    frames, feature_cols = [], None
    for _, sub in df.groupby(unit_col, sort=True):
        sub = sub.iloc[::stride].reset_index(drop=True)
        mem = extractor.prepare_sequences(sub, sensor_cols, include_time=False)
        if feature_cols is None:
            feature_cols = list(mem.columns)  # the memory features themselves
        for c in carry:
            mem[c] = sub[c].values
        frames.append(mem)
    out = pd.concat(frames, ignore_index=True)
    out[feature_cols] = out[feature_cols].bfill().ffill()
    return out, feature_cols


def build_whole_cycle_features(
    df,
    sensor_cols,
    unit_col="unit",
    cycle_col="cycle",
    health_col="health",
    rul_col="rul",
):
    """One summary row per (engine, cycle): mean/std/min/max/last of each sensor,
    with the cycle's RUL and worst health flag carried through. Returns
    (frame, feature_cols)."""
    g = df.groupby([unit_col, cycle_col], sort=True)
    feat = g[sensor_cols].agg(AGG_FUNCS)
    feat.columns = ["_".join(c) for c in feat.columns]
    agg = {}
    if rul_col in df.columns:
        agg[rul_col] = (rul_col, "first")
    if health_col in df.columns:
        agg[health_col] = (health_col, "min")
    meta = g.agg(**agg) if agg else None
    out = feat.join(meta).reset_index() if meta is not None else feat.reset_index()
    return out, list(feat.columns)


# ---------------------------------------------------------------------------
# RUL target: cap the healthy plateau
# ---------------------------------------------------------------------------
def onset_caps(
    table, unit_col="unit", cycle_col="cycle", health_col="health", rul_col="rul"
):
    """Per training engine, the RUL at its first unhealthy cycle: nothing is
    learnable while the engine is healthy, so the target is capped there."""
    caps = {}
    for unit, sub in table.groupby(unit_col):
        sub = sub.sort_values(cycle_col)
        unhealthy = sub[sub[health_col] == 0]
        onset = unhealthy[cycle_col].min() if len(unhealthy) else sub[cycle_col].max()
        after = sub[sub[cycle_col] >= onset]
        caps[unit] = float(after[rul_col].max() if len(after) else sub[rul_col].max())
    return caps


def cap_rul(table, caps, unit_col="unit", rul_col="rul"):
    """Cap where a cap is known; pass through uncapped otherwise. `caps` is built
    from training engines only, so held-out engines are never touched."""
    cap = table[unit_col].map(caps).astype(float)
    raw = table[rul_col].astype(float)
    return np.where(cap.isna(), raw, np.minimum(raw, cap.fillna(raw)))


# ---------------------------------------------------------------------------
# Per-cycle collapse and the monotone clamp
# ---------------------------------------------------------------------------
def per_cycle(unit, cycle, pred, true=None):
    """Collapse predictions to one row per (engine, cycle); RUL is per cycle."""
    cols = {"unit": unit, "cycle": cycle, "pred": pred}
    if true is not None:
        cols["true"] = true
    return (
        pd.DataFrame(cols)
        .groupby(["unit", "cycle"], as_index=False)
        .mean()
        .sort_values(["unit", "cycle"])
    )


def clamp_monotone(per_cycle_df):
    """Hold each engine's prediction at its running minimum in cycle order --
    RUL only falls, so this turns a noisy curve into a staircase that never
    rises, using only the past (deployable online)."""
    df = per_cycle_df.copy()
    df["pred"] = df.groupby("unit")["pred"].transform(
        lambda p: np.minimum.accumulate(p.to_numpy())
    )
    return df
