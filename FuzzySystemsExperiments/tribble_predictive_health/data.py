"""Dataset adapters -- turn a raw file into the tidy per-sample frame the engine
expects. Only N-CMAPSS is needed here; the pipeline itself is dataset-agnostic,
so a new dataset is a new loader, not a new pipeline."""

from __future__ import annotations

import h5py
import pandas as pd


def load_ncmapss(h5_path, split):
    """Read one split ('dev' or 'test') of an N-CMAPSS HDF5 file into a tidy
    per-sample DataFrame, and return (df, condition_cols, sensor_cols).

    Columns: `unit`, `cycle`, `health` (1 while healthy, 0 once degradation
    begins), `rul`, the operating-condition channels `W_*`, and the measured
    sensors `Xs_*`.
    """
    with h5py.File(h5_path, "r") as f:
        aux = f[f"A_{split}"][:]  # unit, cycle, flight-class, health-state
        df = pd.DataFrame(
            {
                "unit": aux[:, 0].astype(int),
                "cycle": aux[:, 1].astype(int),
                "health": aux[:, 3],
                "rul": f[f"Y_{split}"][:, 0].astype(float),
            }
        )
        w_names = [v.decode() for v in f["W_var"][:]]
        xs_names = [v.decode() for v in f["X_s_var"][:]]
        w = f[f"W_{split}"][:]
        xs = f[f"X_s_{split}"][:]
    for i, name in enumerate(w_names):
        df[f"W_{name}"] = w[:, i]
    for i, name in enumerate(xs_names):
        df[f"Xs_{name}"] = xs[:, i]
    condition_cols = [f"W_{n}" for n in w_names]
    sensor_cols = [f"Xs_{n}" for n in xs_names]
    return df, condition_cols, sensor_cols
