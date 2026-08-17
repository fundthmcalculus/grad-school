"""The monotone post-processing operators -- turning a noisy RUL prediction
into a non-increasing one.

Split out of the old `monotone.py` (which was both this library and a driver)
so that importing an operator no longer means importing an experiment. Every
`out_*` takes one engine's prediction sequence, ordered by ascending cycle, and
returns the same length. `apply_output` maps one over every engine in a
per-cycle frame, scoring each with `metrics.score_engine`.

Causal operators use only cycles <= t (deployable); `out_iso_offline` uses the
whole trajectory and is the offline bound the causal ones are judged against.
See MONOTONE.md for which wins where.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import metrics


def _antitonic(y: np.ndarray) -> np.ndarray:
    """L2-optimal monotone *non-increasing* fit (pool-adjacent-violators)."""
    from sklearn.isotonic import IsotonicRegression

    x = np.arange(len(y))
    return IsotonicRegression(increasing=False).fit(x, y).predict(x)


def out_raw(p):
    return np.asarray(p, dtype=float)


def out_mean(p, k):
    return pd.Series(p).rolling(k, min_periods=1).mean().to_numpy()


def out_ewma(p, alpha):
    return pd.Series(p).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def out_cummin(p):
    """RUL revised only downward: the running minimum. Hard-monotone, causal --
    but a single early low outlier pins the whole trajectory under it."""
    return np.minimum.accumulate(np.asarray(p, dtype=float))


def out_ewma_cummin(p, alpha=0.3):
    """Smooth first, then clamp to non-increasing. The causal method that gets
    hard monotonicity without letting one raw outlier set the floor."""
    return np.minimum.accumulate(out_ewma(p, alpha))


def out_mean_cummin(p, k=5):
    """Trailing mean, then clamp to non-increasing. The recommended causal
    hard-monotone estimator: on the noisy `honest` pipeline it costs only
    +0.2 RMSE over raw (against +0.8 for the ewma variant and +7 for a bare
    running min), because a short symmetric-within-window average knocks down
    the spikes a running min would otherwise adopt as its floor."""
    return np.minimum.accumulate(out_mean(p, k))


def out_iso_causal(p):
    """Antitonic regression re-fit on cycles 0..t at each t, reported at t.

    Deployable: every fit sees only the past. More robust than a running min
    because pooling averages an outlier against its neighbours instead of
    adopting it as the floor.
    """
    p = np.asarray(p, dtype=float)
    return np.array([_antitonic(p[: t + 1])[-1] for t in range(len(p))])


def out_iso_offline(p):
    """Antitonic regression over the whole trajectory. NOT causal -- the L2-best
    monotone fit, and thus the bound every causal method above is chasing."""
    return _antitonic(np.asarray(p, dtype=float))


OUTPUT_METHODS = {
    "raw": (out_raw, "causal", "output"),
    "mean_k5": (lambda p: out_mean(p, 5), "causal", "output"),
    "ewma_0.3": (lambda p: out_ewma(p, 0.3), "causal", "output"),
    "cummin": (out_cummin, "causal", "output"),
    "ewma_cummin": (out_ewma_cummin, "causal", "output"),
    "mean5_cummin": (lambda p: out_mean_cummin(p, 5), "causal", "output"),
    "iso_causal": (out_iso_causal, "causal", "output"),
    "iso_offline": (out_iso_offline, "oracle", "output"),
}

# The one recommended for deployment, referenced by the plot and report.
RECOMMENDED = "mean5_cummin"


def apply_output(g: pd.DataFrame, fn) -> list[dict]:
    out = []
    for _, sub in g.groupby("unit"):
        out.append(
            metrics.score_engine(sub["true"].to_numpy(), fn(sub["pred"].to_numpy()))
        )
    return out
