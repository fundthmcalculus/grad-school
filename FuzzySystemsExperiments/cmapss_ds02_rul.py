"""Remaining-useful-life on N-CMAPSS DS02 with a TRIBBLE fuzzy system.

The best-case pipeline, written to be read top to bottom. It is the distillation
of a long design-of-experiments (which lived in a dozen scripts and is kept only
as history): every knob below is the one that won, and nothing else is tried.

The five steps, in order, are:

  1. Load DS02.  The file gives one row per 1 Hz sample: the flight condition
     `W` (altitude, Mach, throttle, inlet temp), the measured sensors `X_s`
     (temperatures, pressures, speeds), and per-row bookkeeping (engine unit,
     flight cycle, health flag, true RUL).

  2. Condition-correct the sensors.  A sensor reading depends far more on what
     the aircraft is doing this flight than on how worn the engine is. Regress
     each sensor on `W` using each engine's first, presumed-healthy cycles, and
     keep the residual. The degradation signal is what's left once the
     operating condition is subtracted out -- this single step matters more
     than any model hyperparameter.

  3. Build memory features.  RUL changes once per flight cycle, not once per
     second, so collapse the stream to a manageable rate and give the model a
     short- and long-term rolling average of each channel. The rolling memory
     is what makes a per-cycle-independent model track a slow trend smoothly.

  4. Fit the fuzzy system.  TRIBBLE builds its rules consequent-first, in about
     a second, no gradient descent. Train on the six development engines,
     predict on the three the file holds out.

  5. Enforce monotonicity.  RUL only ever falls, so a prediction that rises is
     noise. Clamping each engine's trajectory to its running minimum removes
     every up-tick at essentially no cost in accuracy, and needs only the past,
     so it is deployable online.

Result on the official held-out engines (11, 14, 15): per-sample RMSE ~6.5
cycles, which beats the published DS02 CNN (7.22) and MLP (8.34); after the
monotone clamp, ~6.1 with zero rising cycles. Uses the 18 real sensors only --
adding the two "virtual" channels the literature also allows (T40, P30) does not
change the result.

Needs: h5py, numpy, pandas, scikit-learn, tribble-fis.  Run:

    python cmapss_ds02_rul.py --h5 NASA-CMAPSS/N-CMAPSS_DS02-006.h5
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

# DS02's own train/test split, as published (the engines the file holds out).
TEST_UNITS = (11, 14, 15)

# The winning fuzzy-system configuration. `n_gaussians=0` means "choose
# automatically"; `full-2nd` gives the consequents quadratic terms; the hamacher
# norm and a light ridge were the DOE's best-scoring pair.
FIS_KWARGS = dict(
    tsk_order="full-2nd",
    n_gaussians=0,
    top_p=0.95,
    detect_interactions=False,
    norm_conorm="hamacher",
    l2_reg=0.01,
)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_dataframe(h5_path, split):
    """Read one split ('dev' or 'test') into a tidy per-sample DataFrame."""
    with h5py.File(h5_path, "r") as f:
        aux = f[f"A_{split}"][:]  # unit, cycle, flight-class, health-state
        cols = {
            "unit": aux[:, 0].astype(int),
            "cycle": aux[:, 1].astype(int),
            "health": aux[:, 3],  # 1 while healthy, 0 once degradation begins
            "rul": f[f"Y_{split}"][:, 0].astype(float),
        }
        w_names = [v.decode() for v in f["W_var"][:]]
        xs_names = [v.decode() for v in f["X_s_var"][:]]
        w = f[f"W_{split}"][:]
        xs = f[f"X_s_{split}"][:]
    df = pd.DataFrame(cols)
    for i, name in enumerate(w_names):
        df[f"W_{name}"] = w[:, i]
    for i, name in enumerate(xs_names):
        df[f"Xs_{name}"] = xs[:, i]
    condition_cols = [f"W_{n}" for n in w_names]
    sensor_cols = [f"Xs_{n}" for n in xs_names]
    return df, condition_cols, sensor_cols


# ---------------------------------------------------------------------------
# 2. Condition correction
# ---------------------------------------------------------------------------
def fit_condition_correction(dev, sensor_cols, condition_cols, baseline_cycles=15):
    """Learn, per sensor, its dependence on the flight condition -- from each
    training engine's first `baseline_cycles` (still healthy) cycles only."""
    early = dev[dev.groupby("unit").cumcount() < baseline_cycles]
    X = early[condition_cols].to_numpy(float)
    return {c: LinearRegression().fit(X, early[c].to_numpy(float)) for c in sensor_cols}


def apply_condition_correction(df, sensor_cols, condition_cols, models):
    """Replace each sensor with its residual after removing the fitted
    condition dependence. Fit on dev only, applied to both splits."""
    df = df.copy()
    X = df[condition_cols].to_numpy(float)
    for c in sensor_cols:
        df[c] = df[c].to_numpy(float) - models[c].predict(X)
    return df


# ---------------------------------------------------------------------------
# 3. Memory features (per engine, in cycle order)
# ---------------------------------------------------------------------------
def build_memory_features(df, sensor_cols, stride=200):
    """Subsample every `stride`-th sample within each engine, then attach a
    short- and long-term rolling average of every sensor. One row out per
    subsampled sample, carrying its own (unit, cycle, health, rul)."""
    extractor = MemoryWindowFeatureExtractor(window_size=5, memory_size=2)
    frames = []
    for unit, sub in df.groupby("unit", sort=True):
        sub = sub.iloc[::stride].reset_index(drop=True)
        mem = extractor.prepare_sequences(sub, sensor_cols, include_time=False)
        for extra in ("unit", "cycle", "health", "rul"):
            mem[extra] = sub[extra].values
        frames.append(mem)
    out = pd.concat(frames, ignore_index=True)
    feature_cols = [
        c for c in out.columns if c not in ("unit", "cycle", "health", "rul")
    ]
    out[feature_cols] = out[feature_cols].bfill().ffill()
    return out, feature_cols


# ---------------------------------------------------------------------------
# The RUL target: cap the healthy plateau
# ---------------------------------------------------------------------------
def cap_rul(table, caps):
    """An engine's RUL is flat-then-declining: nothing is learnable while it is
    healthy, so cap the target at the RUL it had when degradation began. `caps`
    is built from training engines only; test engines pass through uncapped."""
    cap = table["unit"].map(caps).astype(float)
    raw = table["rul"].astype(float)
    return np.where(cap.isna(), raw, np.minimum(raw, cap.fillna(raw)))


def onset_caps(table):
    """Per training engine, the RUL at its first unhealthy cycle."""
    caps = {}
    for unit, sub in table.groupby("unit"):
        sub = sub.sort_values("cycle")
        unhealthy = sub[sub["health"] == 0]
        onset = unhealthy["cycle"].min() if len(unhealthy) else sub["cycle"].max()
        after = sub[sub["cycle"] >= onset]
        caps[unit] = float(after["rul"].max() if len(after) else sub["rul"].max())
    return caps


# ---------------------------------------------------------------------------
# 5. Monotone clamp + scoring
# ---------------------------------------------------------------------------
# RUL is a per-cycle quantity, but the memory pipeline emits several rows per
# cycle, so everything about monotonicity is done on one prediction per cycle.
def per_cycle(units, cycles, y_true, pred):
    """Collapse to one row per (unit, cycle): predictions averaged, true RUL
    taken as the cycle's value. Sorted by unit then cycle."""
    df = pd.DataFrame({"unit": units, "cycle": cycles, "true": y_true, "pred": pred})
    return (
        df.groupby(["unit", "cycle"], as_index=False)
        .mean()
        .sort_values(["unit", "cycle"])
    )


def clamp_monotone(per_cycle_df):
    """Hold each engine's per-cycle prediction at its running minimum -- RUL only
    falls, so this turns a noisy up-and-down curve into a staircase that never
    rises, using only the past (deployable online)."""
    df = per_cycle_df.copy()
    df["pred"] = df.groupby("unit")["pred"].transform(
        lambda p: np.minimum.accumulate(p.to_numpy())
    )
    return df


def rising_fraction(per_cycle_df):
    """Fraction of cycle-to-cycle steps on which the prediction rose (noise)."""
    rises = total = 0
    for _, sub in per_cycle_df.groupby("unit"):
        d = np.diff(sub["pred"].to_numpy())
        rises += int((d > 0).sum())
        total += len(d)
    return rises / total if total else 0.0


# ---------------------------------------------------------------------------
# The pipeline, end to end
# ---------------------------------------------------------------------------
def cyc_rmse(df):
    return float(np.sqrt(mean_squared_error(df["true"], df["pred"])))


def fit_and_score(dev, test, cond_cols, sensor_cols, verbose=True):
    """Steps 2-5 on an already-loaded dev/test pair. Returns a results dict.

    Shared with the all-datasets script so the two files never disagree about
    what "the pipeline" is: this function *is* the pipeline.
    """
    say = print if verbose else (lambda *a, **k: None)

    say("Condition-correcting sensors against flight condition ...")
    models = fit_condition_correction(dev, sensor_cols, cond_cols)
    dev = apply_condition_correction(dev, sensor_cols, cond_cols, models)
    test = apply_condition_correction(test, sensor_cols, cond_cols, models)

    say("Building memory features ...")
    train_tab, feature_cols = build_memory_features(dev, sensor_cols)
    test_tab, _ = build_memory_features(test, sensor_cols)
    say(
        f"  {len(train_tab):,} train rows, {len(test_tab):,} test rows,"
        f" {len(feature_cols)} features"
    )

    caps = onset_caps(train_tab)
    scaler = StandardScaler().fit(train_tab[feature_cols].to_numpy(float))
    X_train = scaler.transform(train_tab[feature_cols].to_numpy(float))
    X_test = scaler.transform(test_tab[feature_cols].to_numpy(float))
    y_train = cap_rul(train_tab, caps)

    say("Fitting the fuzzy system ...")
    model = TribbleRegressor(random_state=42, max_samples=2000, **FIS_KWARGS)
    t_fit = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # TRIBBLE is chatty
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t_fit

    units = test_tab["unit"].to_numpy()
    cycles = test_tab["cycle"].to_numpy()
    y_true = test_tab["rul"].to_numpy(float)
    pred = model.predict(X_test)

    cyc = per_cycle(units, cycles, y_true, pred)
    cyc_mono = clamp_monotone(cyc)
    return dict(
        n_rules=int(model.model_.n_rules),
        fit_seconds=fit_seconds,
        # Per-sample RMSE over every test row -- the figure the published
        # CNN/MLP baselines report, computed the same way.
        per_sample_rmse=float(np.sqrt(mean_squared_error(y_true, pred))),
        raw_rmse=cyc_rmse(cyc),  # per cycle, before clamping
        raw_rising=rising_fraction(cyc),
        monotone_rmse=cyc_rmse(cyc_mono),  # per cycle, recommended
        monotone_rising=rising_fraction(cyc_mono),
        per_cycle_monotone=cyc_mono,
    )


def main(h5_path):
    t0 = time.perf_counter()
    print(f"Loading {h5_path} ...")
    dev, cond_cols, sensor_cols = load_dataframe(h5_path, "dev")
    test, _, _ = load_dataframe(h5_path, "test")
    print(
        f"  {len(dev):,} dev rows, {len(test):,} test rows,"
        f" {len(sensor_cols)} sensors"
    )

    r = fit_and_score(dev, test, cond_cols, sensor_cols)

    print(
        f"\n=== DS02 remaining useful life ({r['n_rules']} rules,"
        f" fit in {r['fit_seconds']:.2f}s) ==="
    )
    print(
        f"  per-sample RMSE {r['per_sample_rmse']:5.2f}   "
        f"(published baselines: CNN 7.22, MLP 8.34)\n"
    )
    print(
        f"  per cycle, raw       RMSE {r['raw_rmse']:5.2f}   "
        f"rising cycles {r['raw_rising']:5.1%}"
    )
    print(
        f"  per cycle, monotone  RMSE {r['monotone_rmse']:5.2f}   "
        f"rising cycles {r['monotone_rising']:5.1%}   <- recommended\n"
    )
    for unit in TEST_UNITS:
        sub = r["per_cycle_monotone"].query("unit == @unit")
        print(f"  engine {unit}: {len(sub):3d} cycles   RMSE {cyc_rmse(sub):.2f}")
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default="NASA-CMAPSS/N-CMAPSS_DS02-006.h5")
    main(parser.parse_args().h5)
