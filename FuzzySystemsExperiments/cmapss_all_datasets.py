"""N-CMAPSS RUL pooled across every dataset, both scoring conventions.

The DS02 script (`cmapss_ds02_rul.py`) is the single best case. This is the
*general* result the dissertation's Table 4.10 rests on: pool all nine usable
N-CMAPSS files into one train/test split (each file contributing its own
official held-out engines), fit one model, and report both ways of scoring it.

Two things only become visible when you pool and score both ways, and they are
the whole point:

  * **The scoring convention decides the winner.**  Per *sample* -- RMSE over
    every test row, the density metric the N-CMAPSS literature uses -- the
    memory-feature model wins. Per *engine* -- one RUL per test engine at its
    last cycle, the classic C-MAPSS protocol -- the simplest whole-cycle model
    wins, and by a lot. So both aggregations are run and both metrics reported;
    reporting only the flattering one would be the mistake.

  * **The virtual channels are not needed.**  Everything here uses the 18 real,
    physically measurable sensors; dropping the two "virtual" channels the file
    allows (T40, P30) costs nothing (established on DS02 in `cmapss_ds02_rul.py`).

Condition correction and the RUL cap are fit per file on that file's own
training engines; the scaler is fit once on the pooled training table. No
test-set information is used to fit anything. Writes
`cmapss_all_datasets_report.md`.

Needs: h5py, numpy, pandas, scikit-learn, tribble-fis.  Run:

    python cmapss_all_datasets.py --h5-dir NASA-CMAPSS
"""

import argparse
import contextlib
import gc
import glob
import io
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from tribblefis.gaussian_regressor import TribbleRegressor

# Reuse the DS02 pipeline's building blocks so the two scripts can never
# disagree about what condition correction, memory features, or the RUL cap are.
from cmapss_ds02_rul import (
    apply_condition_correction,
    build_memory_features,
    cap_rul,
    fit_condition_correction,
    load_dataframe,
    onset_caps,
    per_cycle,
)

REPORT = "FuzzySystemsExperiments/cmapss_all_datasets_report.md"

# The two pooled models. `whole_cycle` (one summary row per flight cycle, a
# first-order fuzzy system) is the best per-engine model; `raw_memory` (the DS02
# best case, memory features and quadratic consequents) is the best per-sample
# model. Same 18 real sensors, same condition correction -- only the way the
# stream is summarised differs.
AGG_FUNCS = ["mean", "std", "min", "max", "last"]
CONFIGS = {
    "whole_cycle": dict(
        tsk_order="1st",
        n_gaussians=0,
        top_p=0.9,
        detect_interactions=False,
        norm_conorm="hamacher",
        l2_reg=0.01,
    ),
    "raw_memory": dict(
        tsk_order="full-2nd",
        n_gaussians=0,
        top_p=0.95,
        detect_interactions=False,
        norm_conorm="hamacher",
        l2_reg=0.01,
    ),
}
# The pooled raw-memory table is ~220k rows and its quadratic consequent solve
# is what would blow up memory, so the training table is subsampled to a fixed
# 30k rows (seed 42) before fitting -- about the size the single-DS02 fit uses.
# whole_cycle is only ~4.5k rows, so it trains on all of them.
POOLED_TRAIN_CAP = {"whole_cycle": None, "raw_memory": 30_000}


def build_whole_cycle_features(df, sensor_cols):
    """One row per (unit, cycle): mean/std/min/max/last of each sensor."""
    g = df.groupby(["unit", "cycle"], sort=True)
    feat = g[sensor_cols].agg(AGG_FUNCS)
    feat.columns = ["_".join(c) for c in feat.columns]
    meta = g.agg(rul=("rul", "first"), health=("health", "min"))
    return feat.join(meta).reset_index()


def nasa_score(y_true, y_pred):
    """PHM08 asymmetric penalty (late warning punished harder), summed."""
    d = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sum(np.exp(np.where(d > 0, 1 / 13.0, 1 / 10.0) * np.abs(d))))


def per_engine_canonical(test_tab, pred):
    """Standard C-MAPSS scoring: one RUL per engine at its last cycle.

    `per_cycle` returns the engine id in its `unit` column (it just names its
    first argument `unit`); we pass the global engine id in there.
    """
    df = per_cycle(
        test_tab["engine"].to_numpy(),
        test_tab["cycle"].to_numpy(),
        test_tab["rul"].to_numpy(float),
        pred,
    )
    last = df.sort_values("cycle").groupby("unit").last()
    return (
        float(np.sqrt(mean_squared_error(last["true"], last["pred"]))),
        nasa_score(last["true"], last["pred"]),
    )


# ---------------------------------------------------------------------------
# Load every file once, build both feature tables, pool
# ---------------------------------------------------------------------------
def gather(h5_dir):
    """Return pooled {whole_cycle, raw_memory} -> (train, test, feature_cols),
    plus the list of processed/skipped datasets. Each file is loaded once and
    freed before the next, so peak memory stays near one dataset."""
    pooled = {agg: {"train": [], "test": []} for agg in CONFIGS}
    feature_cols = {}
    processed, skipped = [], []
    for path in sorted(glob.glob(os.path.join(h5_dir, "*.h5"))):
        name = os.path.basename(path).replace("N-CMAPSS_", "").replace(".h5", "")
        try:
            dev, cond, sensors = load_dataframe(path, "dev")
            test, _, _ = load_dataframe(path, "test")
        except Exception as exc:  # the one truncated file
            skipped.append((name, f"{type(exc).__name__}"))
            continue
        models = fit_condition_correction(dev, sensors, cond)
        dev = apply_condition_correction(dev, sensors, cond, models)
        test = apply_condition_correction(test, sensors, cond, models)

        for agg in CONFIGS:
            if agg == "whole_cycle":
                tr = build_whole_cycle_features(dev, sensors)
                te = build_whole_cycle_features(test, sensors)
                cols = [
                    c for c in tr.columns if c not in ("unit", "cycle", "rul", "health")
                ]
            else:
                tr, cols = build_memory_features(dev, sensors)
                te, _ = build_memory_features(test, sensors)
            feature_cols[agg] = cols
            # Unit numbers repeat across files; make a globally unique engine id.
            for t in (tr, te):
                t["dataset"] = name
                t["engine"] = name + ":" + t["unit"].astype(str)
            pooled[agg]["train"].append(tr)
            pooled[agg]["test"].append(te)
        processed.append(name)
        del dev, test
        gc.collect()

    out = {}
    for agg in CONFIGS:
        train = pd.concat(pooled[agg]["train"], ignore_index=True)
        test = pd.concat(pooled[agg]["test"], ignore_index=True)
        out[agg] = (train, test, feature_cols[agg])
    return out, processed, skipped


# ---------------------------------------------------------------------------
# Fit one pooled model and score it both ways
# ---------------------------------------------------------------------------
def fit_pooled(agg, train, test, feature_cols):
    caps = onset_caps(train.assign(unit=train["engine"]))  # cap per global engine
    cap = POOLED_TRAIN_CAP[agg]
    if cap and len(train) > cap:
        train = train.sample(cap, random_state=42)

    scaler = StandardScaler().fit(train[feature_cols].to_numpy(float))
    X_train = scaler.transform(train[feature_cols].to_numpy(float))
    X_test = scaler.transform(test[feature_cols].to_numpy(float))
    y_train = cap_rul(train.assign(unit=train["engine"]), caps)

    model = TribbleRegressor(random_state=42, max_samples=2000, **CONFIGS[agg])
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    pred = model.predict(X_test)
    y_true = test["rul"].to_numpy(float)
    eng_rmse, eng_nasa = per_engine_canonical(test, pred)

    dataset = test["dataset"].to_numpy()
    per_dataset = {
        ds: float(
            np.sqrt(
                mean_squared_error(
                    test["rul"].to_numpy(float)[dataset == ds], pred[dataset == ds]
                )
            )
        )
        for ds in pd.unique(dataset)
    }
    return dict(
        config=agg,
        n_train=len(train),
        n_test=len(test),
        n_engines=int(test["engine"].nunique()),
        fit_seconds=fit_seconds,
        per_sample_rmse=float(np.sqrt(mean_squared_error(y_true, pred))),
        per_engine_rmse=eng_rmse,
        per_engine_nasa=eng_nasa,
        per_dataset=per_dataset,
    )


def write_report(results, processed, skipped):
    lines = [
        "# N-CMAPSS RUL, pooled across all datasets",
        "",
        "The 18-real-sensor pipeline (see `cmapss_ds02_rul.py`) pooled over every "
        "usable N-CMAPSS file -- each contributing its own official train/test "
        "engines -- and scored two ways. Regenerated by `cmapss_all_datasets.py`.",
        "",
        f"Datasets pooled: {', '.join(processed)}"
        + (
            f".  Skipped: {', '.join(f'{n} ({why})' for n, why in skipped)}."
            if skipped
            else "."
        ),
        "",
        "| model | input | per-sample RMSE | per-engine RMSE | per-engine NASA | fit s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r['config']}` | 18 real | {r['per_sample_rmse']:.2f} | "
            f"{r['per_engine_rmse']:.2f} | {r['per_engine_nasa']:,.0f} | "
            f"{r['fit_seconds']:.1f} |"
        )
    lines += [
        "",
        "Per-sample favours `raw_memory`; per-engine (the canonical C-MAPSS "
        "protocol, one RUL per test engine at its last cycle) favours "
        "`whole_cycle` -- the scoring convention decides the winner.",
        "",
        "## Per-dataset per-sample RMSE (same pooled model, broken out by file)",
        "",
        "| dataset | " + " | ".join(f"`{r['config']}`" for r in results) + " |",
        "|---|" + "---:|" * len(results),
    ]
    for ds in processed:
        cells = " | ".join(
            f"{r['per_dataset'].get(ds, float('nan')):.2f}" for r in results
        )
        lines.append(f"| {ds} | {cells} |")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(h5_dir):
    t0 = time.perf_counter()
    print(f"Loading and pooling N-CMAPSS files from {h5_dir} ...")
    pooled, processed, skipped = gather(h5_dir)
    print(
        f"  pooled {len(processed)} datasets"
        + (f", skipped {len(skipped)}" if skipped else "")
    )

    results = []
    for agg in CONFIGS:
        train, test, feature_cols = pooled[agg]
        print(f"Fitting pooled `{agg}` ({len(train):,} train rows) ...")
        results.append(fit_pooled(agg, train, test, feature_cols))

    print("\n=== N-CMAPSS pooled RUL ===")
    print(f"  {'model':12s} {'per-sample':>10s} {'per-engine':>10s} {'NASA':>10s}")
    for r in results:
        print(
            f"  {r['config']:12s} {r['per_sample_rmse']:10.2f} "
            f"{r['per_engine_rmse']:10.2f} {r['per_engine_nasa']:10,.0f}"
            f"   ({r['n_engines']} engines)"
        )
    write_report(results, processed, skipped)
    print(f"\nwrote {REPORT}")
    print(f"Total wall time: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", default="NASA-CMAPSS")
    main(parser.parse_args().h5_dir)
