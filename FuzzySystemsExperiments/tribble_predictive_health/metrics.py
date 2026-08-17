"""How a remaining-useful-life prediction is scored.

RUL admits two conventions that disagree about which model is best, so both are
computed and neither is hidden:

* **per-sample** -- RMSE over every scored row, the continuous-prognostics
  metric the N-CMAPSS literature uses;
* **per-engine (canonical)** -- one RUL per engine at its last cycle, the
  classic C-MAPSS / PHM protocol comparable to published RMSE / NASA scores.

Plus a monotonicity read: true RUL only falls, so any cycle it rises is noise.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from .preprocessing import clamp_monotone, per_cycle


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(y_true, y_pred):
    """PHM08 asymmetric penalty (late warning punished harder), summed."""
    d = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sum(np.exp(np.where(d > 0, 1 / 13.0, 1 / 10.0) * np.abs(d))))


def rising_fraction(per_cycle_df):
    """Fraction of cycle-to-cycle steps on which the prediction rose (noise)."""
    rises = total = 0
    for _, sub in per_cycle_df.groupby("unit"):
        d = np.diff(sub["pred"].to_numpy())
        rises += int((d > 0).sum())
        total += len(d)
    return rises / total if total else 0.0


def per_engine_canonical(per_cycle_df):
    """One RUL per engine at its last cycle -- the standard C-MAPSS protocol.
    `per_cycle_df` must carry `true`. Returns (rmse, nasa)."""
    last = per_cycle_df.sort_values("cycle").groupby("unit").last()
    return rmse(last["true"], last["pred"]), nasa_score(last["true"], last["pred"])


def score(units, cycles, y_true, pred):
    """Every number that matters, from a set of per-row predictions, as one flat
    dict:

      per_sample_rmse       RMSE over every scored row (continuous-prognostics)
      raw_cycle_rmse        RMSE per (engine, cycle), before any clamp
      raw_rising            fraction of cycles the raw curve rose
      monotone_cycle_rmse   RMSE per cycle after the running-min clamp
      monotone_rising       fraction the clamped curve rose (0 by construction)
      per_engine_rmse       canonical C-MAPSS: one RUL per engine at its last
      per_engine_nasa       cycle, scored by RMSE and the PHM08 NASA score

    Callers print whichever convention they report; nothing is hidden.
    """
    cyc = per_cycle(units, cycles, pred, true=y_true)
    mono = clamp_monotone(cyc)
    eng_rmse, eng_nasa = per_engine_canonical(cyc)
    return {
        "per_sample_rmse": rmse(y_true, pred),
        "raw_cycle_rmse": rmse(cyc["true"], cyc["pred"]),
        "raw_rising": rising_fraction(cyc),
        "monotone_cycle_rmse": rmse(mono["true"], mono["pred"]),
        "monotone_rising": rising_fraction(mono),
        "per_engine_rmse": eng_rmse,
        "per_engine_nasa": eng_nasa,
    }
