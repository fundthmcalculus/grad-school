#!/usr/bin/env python3
"""Why does bounded normalization beat centred normalization on Concrete?

Checklist **E9**. The choice itself is settled -- `UnitScalar` (log + min-max to
[0,1]) is what the samples and every table use, and it is best-or-tied in 8 of 9
rows. What is *not* settled is the explanation, which Chapter 4 currently
asserts rather than shows: the working account is that Gaussian membership
functions and the `[0,1]`-pinned extreme bucket means assume a **bounded,
non-negative** domain, so an unbounded centred transform breaks an assumption
the construction relies on.

Two innocent explanations are already ruled out upstream of this script: ridge
scale (sweeping `l2_reg` 1e-2 -> 0 moves the gap by 0.001) and the
scale-dependent BIC membership count (pinning `n_gaussians` for an identical
rule base still gives -0.407/-0.524/-0.634). See `NORMALIZATION_THREE_ARM.md`.

E9 names three cheap experiments, none of which needs new data. This runs all
three on one shared set of splits, so the arms are directly comparable:

  (a) **`UnitScalar(feature_range=(-1, 1))`** -- centred but still bounded. This
      is the decisive arm, and it separates the two candidate causes cleanly:

        * if it behaves like [0,1], centring is harmless and it is
          UNBOUNDEDNESS that hurts;
        * if it collapses toward the z-score arm, centring alone is enough to
          do the damage, and the real culprit is the pin on 0.0/1.0 extreme
          bucket means rather than the tails.

  (b) **z-score clipped to a fixed range** (+/-2 and +/-3 sigma). Bounds the
      transform without un-centring it. How much of the loss comes back is a
      direct measure of how much of it the tails own.

  (c) **Damage localization.** Apply z-score to ONLY the log-detected features
      (Slag, Age on Concrete) with min-max elsewhere, and the reverse. If the
      loss follows the logged columns, the interaction is with the log step
      rather than with centring as such.

Two controls run alongside, because an explanation that does not reproduce the
thing it explains is not an explanation: `raw` (no transform) and the two arms
Table 4.1 already reports (`unit`, `standard`).

**Train MSE is reported beside test R2 on purpose.** The working account
predicts the model UNDERFITS rather than overfits under z-score -- the
archived evidence is train MSE 0.030 vs 0.009 -- so a z-score arm that fits
train well and generalizes badly would refute it. That is the falsification
condition, and it is registered here before the run rather than after it.

Ten seeds, shared splits, flat MoG-TSK at 1st order (the arm the -0.407 figure
comes from), `n_output_buckets=3`, `top_n=-1`, `l2_reg=1e-2`.

Run (from repo root):

    uv run --project tribble-fis python \
        reproduce/experiments/diagnose_bounded_normalization.py

Knobs:
    REPRO_SEEDS="0,1,2"         quick smoke run
    REPRO_ORDERS="1st,2nd"      consequent orders to include
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPRO = os.path.dirname(_HERE)
sys.path.insert(0, _REPRO)
sys.path.insert(0, os.path.join(_REPRO, "tables"))

import common as C  # noqa: E402
import _fuzzy_models as F  # noqa: E402

ORDERS = [o.strip() for o in os.environ.get("REPRO_ORDERS", "1st").split(",")]
N_BUCKETS = 3
LOG_DYNAMIC_RANGE = 2

# The second axis, and the one that turned out to matter. `pin_extremes` fixes
# the first and last output-bucket means to the observed min and max. It
# defaulted to True when E9's numbers were taken and defaults to False now
# (tribble-fis #102, `69e0bab`), which is the whole reason those numbers do not
# reproduce. Sweeping it makes the mechanism switchable instead of historical.
#
# Added AFTER a first pass over the transform arms alone, which is worth saying
# plainly: that pass found every normalized arm identical to three decimals and
# so had no signal at all. The pin axis was not a registered prediction; it was
# the hypothesis that first pass forced. The 2x2 below is what tests it.
PINS = [False, True]


def _regressor(seed, tsk_order, pin_extremes):
    """The flat MoG-TSK regressor, with `pin_extremes` exposed.

    Deliberately not `_fuzzy_models.mog_regressor`: that helper is shared with
    the shipped tables and must keep taking the library default, or this
    experiment would silently redefine what those tables measure.
    """
    try:
        from tribblefis.gaussian_regressor import TribbleRegressor

        return TribbleRegressor(
            n_output_buckets=N_BUCKETS,
            tsk_order=tsk_order,
            top_n=-1,
            random_state=seed,
            pin_extremes=pin_extremes,
        )
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# The arms. Each returns a transformed frame plus the logged column names, so
# every arm goes through the same downstream code path and only the transform
# differs.
# --------------------------------------------------------------------------
def _fit_transform(scaler, X):
    Xt = pd.DataFrame(scaler.fit_transform(X.copy()), index=X.index, columns=X.columns)
    return Xt, list(getattr(scaler, "log_features_", []))


def arm_raw(X):
    """No transform at all. The floor every other arm must beat."""
    return X.copy(), []


def arm_unit(X, lo=0.0, hi=1.0):
    """log + min-max to [lo, hi]. `(0, 1)` is the shipped default."""
    from tribblefis.scaling import UnitScalar

    return _fit_transform(
        UnitScalar(feature_range=(lo, hi), log_dynamic_range=LOG_DYNAMIC_RANGE), X
    )


def arm_standard(X, clip=None):
    """log + z-score, optionally clipped to +/- `clip` sigma."""
    from tribblefis.scaling import StandardScalar

    Xt, logged = _fit_transform(StandardScalar(log_dynamic_range=LOG_DYNAMIC_RANGE), X)
    if clip is not None:
        Xt = Xt.clip(lower=-clip, upper=clip)
    return Xt, logged


def arm_split(X, standard_on_logged):
    """z-score one group of columns, min-max the other.

    `standard_on_logged=True` puts z-score on the auto-detected log columns and
    min-max on the rest; `False` is the reverse. Between them they say whether
    the damage follows the logged features or the un-logged ones.

    The log detection itself is taken from a `UnitScalar` fit on the full frame,
    so both directions partition on the SAME column set -- otherwise the two
    arms would not be answering the same question.
    """
    from tribblefis.scaling import StandardScalar, UnitScalar

    probe = UnitScalar(log_dynamic_range=LOG_DYNAMIC_RANGE)
    probe.fit(X.copy())
    logged = list(probe.log_features_)
    if not logged:
        return None, []

    rest = [c for c in X.columns if c not in logged]
    group_std = logged if standard_on_logged else rest
    group_unit = rest if standard_on_logged else logged

    out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    if group_std:
        s = StandardScalar(log_dynamic_range=LOG_DYNAMIC_RANGE)
        out[group_std] = s.fit_transform(X[group_std].copy())
    if group_unit:
        u = UnitScalar(log_dynamic_range=LOG_DYNAMIC_RANGE)
        out[group_unit] = u.fit_transform(X[group_unit].copy())
    return out, logged


ARMS = [
    ("raw (no transform)", lambda X: arm_raw(X)),
    ("log + min-max [0,1]  <- shipped", lambda X: arm_unit(X, 0.0, 1.0)),
    ("log + min-max [-1,1]  (a)", lambda X: arm_unit(X, -1.0, 1.0)),
    ("log + z-score", lambda X: arm_standard(X)),
    ("log + z-score, clip +/-3  (b)", lambda X: arm_standard(X, clip=3.0)),
    ("log + z-score, clip +/-2  (b)", lambda X: arm_standard(X, clip=2.0)),
    ("z-score on logged cols only  (c)", lambda X: arm_split(X, True)),
    ("z-score on unlogged cols only  (c)", lambda X: arm_split(X, False)),
]


def main() -> int:
    data = F.load_concrete()
    if data is None:
        print("Concrete not available; nothing to do.")
        return 1
    X, y = data

    rows = []
    logged_note = ""
    for order in ORDERS:
        for label, make in ARMS:
            for pin in PINS:
                test_r2, train_mse, test_mse = [], [], []
                skipped = None
                for seed in C.SEEDS:
                    Xt, logged = make(X)
                    if Xt is None:
                        skipped = "no log-detected columns"
                        break
                    if logged and not logged_note:
                        logged_note = ", ".join(logged)
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xt, y, test_size=0.2, random_state=seed
                    )
                    model = _regressor(seed, order, pin)
                    if model is None:
                        skipped = "model unavailable"
                        break
                    try:
                        model.fit(Xtr, ytr)
                        p_te = np.asarray(model.predict(Xte), dtype=float)
                        p_tr = np.asarray(model.predict(Xtr), dtype=float)
                    except Exception as exc:  # noqa: BLE001
                        skipped = f"{type(exc).__name__}"
                        break
                    if not np.all(np.isfinite(p_te)) or not np.all(np.isfinite(p_tr)):
                        skipped = "non-finite predictions"
                        break
                    test_r2.append(r2_score(yte, p_te))
                    # Normalized so the two normalization arms are comparable at all:
                    # an MSE in target units is identical across arms, but the claim
                    # under test is about FIT quality, so scale by target variance.
                    var = float(np.var(np.asarray(ytr, dtype=float)))
                    train_mse.append(mean_squared_error(ytr, p_tr) / var)
                    test_mse.append(mean_squared_error(yte, p_te) / var)

                if skipped or not test_r2:
                    rows.append([order, label, str(pin), C.NA, C.NA, C.NA])
                    print(f"  {order:8s} {label:36s} pin={pin!s:5s} SKIP ({skipped})")
                    continue
                rows.append(
                    [
                        order,
                        label,
                        str(pin),
                        f"{np.mean(test_r2):.3f} ± {np.std(test_r2):.3f}",
                        f"{np.mean(train_mse):.3f}",
                        f"{np.mean(test_mse):.3f}",
                    ]
                )
                print(
                    f"  {order:8s} {label:36s} pin={pin!s:5s} "
                    f"test R2 {np.mean(test_r2):+.3f} ± {np.std(test_r2):.3f}   "
                    f"train MSE/var {np.mean(train_mse):.3f}"
                )

    C.emit(
        "diagnose_bounded_normalization",
        header=[
            "order",
            "arm",
            "pin_extremes",
            "test R²",
            "train MSE/var",
            "test MSE/var",
        ],
        rows=rows,
        title="E9 — bounded vs centred normalization on Concrete (flat MoG-TSK)",
        note=(
            "Ten seeds, shared splits, n_output_buckets=3, top_n=-1. "
            f"Auto-detected log columns: {logged_note or 'none'}. "
            "Arm (a) is the decisive one: if [-1,1] tracks [0,1] the cost is "
            "unboundedness, and if it tracks z-score the cost is centring and "
            "the [0,1]-pinned extreme bucket means are the mechanism. "
            "train MSE is normalized by training-target variance so the arms "
            "are comparable; the working account predicts UNDERFITTING under "
            "z-score, so a z-score arm that fits train well would refute it."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
