"""Every way this experiment scores a prediction, in one place.

Two scoring conventions run through the whole study and used to live in two
different modules (`models` and `monotone`), which is the main reason the code
was hard to follow. They are unified here:

* **Whole-split, both C-MAPSS conventions** -- :func:`evaluate` takes a data
  `Split` and returns pooled per-sample RMSE/MAE/NASA *and* the canonical
  one-prediction-per-engine "endpoint" versions. This is what the benchmark
  quotes.
* **Per-engine trajectory** -- :func:`per_cycle` collapses to one row per
  (unit, cycle); :func:`score_engine` scores a single engine's RUL curve for
  both accuracy (RMSE/MAE) and monotonicity (:func:`monotonicity`); and
  :func:`aggregate` averages those over engines. This is the convention the
  smoothing / monotonicity work uses, where each trajectory weighs equally.

Nothing here fits a model or touches TRIBBLE -- it is pure measurement, so it
imports nothing from the rest of the package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Whole-split scoring (pooled and per-engine-endpoint)
# ---------------------------------------------------------------------------
def nasa_score(y_true, y_pred) -> float:
    """The PHM08 asymmetric penalty, summed. Under-prediction (late warning is
    the dangerous direction) is penalized at exp(|d|/10), over-prediction at
    exp(|d|/13). It is a *sum*, so it scales with the number of scored rows --
    only compare it between arms scored on identical rows."""
    delta = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    alpha = np.where(delta > 0, 1.0 / 13.0, 1.0 / 10.0)
    return float(np.sum(np.exp(alpha * np.abs(delta))))


def endpoint_rows(unit: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """Index of each engine's last recorded cycle -- the canonical C-MAPSS
    protocol scores one RUL per test engine, at the end of its trajectory."""
    df = pd.DataFrame({"unit": unit, "cycle": cycle}).reset_index()
    return df.sort_values(["unit", "cycle"]).groupby("unit")["index"].last().to_numpy()


def evaluate(split, y_pred: np.ndarray) -> dict:
    """Both scoring conventions the DOE reports, on one prediction vector."""
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = split.y_true
    err = y_true - y_pred
    idx = endpoint_rows(split.unit, split.cycle)
    e_err = y_true[idx] - y_pred[idx]
    return dict(
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        nasa=nasa_score(y_true, y_pred),
        rmse_endpoint=float(np.sqrt(np.mean(e_err**2))),
        mae_endpoint=float(np.mean(np.abs(e_err))),
        nasa_endpoint=nasa_score(y_true[idx], y_pred[idx]),
        n=int(len(y_true)),
        n_engines=int(len(idx)),
    )


# ---------------------------------------------------------------------------
# Per-engine trajectory scoring (accuracy + monotonicity)
# ---------------------------------------------------------------------------
def per_cycle(unit, cycle, true, pred) -> pd.DataFrame:
    """One row per (unit, cycle); RUL is constant within a cycle."""
    df = pd.DataFrame({"unit": unit, "cycle": cycle, "true": true, "pred": pred})
    return (
        df.groupby(["unit", "cycle"], as_index=False)
        .mean()
        .sort_values(["unit", "cycle"])
    )


def monotonicity(pred: np.ndarray) -> dict:
    """How far a per-engine prediction sequence is from monotone-decreasing.

    `pred` is ordered by ascending cycle. A perfect predictor of a quantity
    that only ever falls has every one of these at zero.
    """
    d = np.diff(np.asarray(pred, dtype=float))
    up = d[d > 0.0]
    return dict(
        up_frac=float((d > 0.0).mean()) if d.size else 0.0,
        pos_tv=float(up.sum()),  # total cycles of "RUL went up"
        max_up=float(up.max()) if up.size else 0.0,
    )


def score_engine(true: np.ndarray, pred: np.ndarray) -> dict:
    """Accuracy and monotonicity for one engine's RUL trajectory."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = monotonicity(pred)
    m["rmse"] = float(np.sqrt(np.mean((pred - true) ** 2)))
    m["mae"] = float(np.mean(np.abs(pred - true)))
    return m


def aggregate(per_engine: list[dict]) -> dict:
    """Mean over engines -- each engine is one trajectory, weighted equally."""
    keys = ("up_frac", "pos_tv", "max_up", "rmse", "mae")
    return {k: float(np.mean([e[k] for e in per_engine])) for k in keys}
