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

Same engine as the DS02 script -- the reusable `TribblePredictiveHealth`. The
only twist pooling needs is that condition correction and the RUL cap are fit
per file on that file's own training engines: so each file is streamed, corrected
and featurised one at a time (keeping peak memory near a single dataset), then
the small feature tables are pooled and handed to the estimator's
`fit_featurized` entry point. No test-set information is used to fit anything.
Writes `cmapss_all_datasets_report.md`.

Needs: h5py, numpy, pandas, scikit-learn, tribble-fis.  Run:

    python cmapss_all_datasets.py --h5-dir NASA-CMAPSS
"""

import argparse
import gc
import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss
from tribble_predictive_health.preprocessing import (
    apply_condition_correction,
    build_memory_features,
    build_whole_cycle_features,
    fit_condition_correction,
)

REPORT = "FuzzySystemsExperiments/cmapss_all_datasets_report.md"

# The two pooled models, as estimator keyword arguments. `whole_cycle` (one
# summary row per flight cycle, a first-order fuzzy system) is the best
# per-engine model; `raw_memory` (the DS02 best case, memory features and
# quadratic consequents) is the best per-sample model. Same 18 real sensors, same
# condition correction -- only the way the stream is summarised differs.
#
# The pooled raw-memory table is ~220k rows and its quadratic consequent solve is
# what would blow up memory, so `max_train_rows` subsamples it to a fixed 30k
# (seed 42) before fitting -- about the size the single-DS02 fit uses. whole_cycle
# is only ~4.5k rows, so it trains on all of them.
CONFIGS = {
    "whole_cycle": dict(
        aggregation="whole_cycle", tsk_order="1st", top_p=0.9, max_train_rows=None
    ),
    "raw_memory": dict(
        aggregation="raw_memory",
        tsk_order="full-2nd",
        top_p=0.95,
        max_train_rows=30_000,
    ),
}


def _featurize(agg, df, sensors):
    """Per-file feature table for one aggregation. Returns (table, feature_cols)."""
    if agg == "whole_cycle":
        return build_whole_cycle_features(df, sensors)
    return build_memory_features(df, sensors)


# ---------------------------------------------------------------------------
# Load every file once, correct and featurise it, pool the small tables
# ---------------------------------------------------------------------------
def gather(h5_dir):
    """Return pooled {agg -> (train, test, feature_cols)}, plus the lists of
    processed/skipped datasets. Each file is loaded, corrected against its own
    baseline, featurised, and freed before the next, so peak memory stays near
    one dataset."""
    pooled = {agg: {"train": [], "test": []} for agg in CONFIGS}
    feature_cols = {}
    processed, skipped = [], []
    for path in sorted(glob.glob(os.path.join(h5_dir, "*.h5"))):
        name = os.path.basename(path).replace("N-CMAPSS_", "").replace(".h5", "")
        try:
            dev, cond, sensors = load_ncmapss(path, "dev")
            test, _, _ = load_ncmapss(path, "test")
        except Exception as exc:  # the one truncated file
            skipped.append((name, f"{type(exc).__name__}"))
            continue
        models = fit_condition_correction(dev, sensors, cond)
        dev = apply_condition_correction(dev, sensors, cond, models)
        test = apply_condition_correction(test, sensors, cond, models)

        for agg in CONFIGS:
            tr, cols = _featurize(agg, dev, sensors)
            te, _ = _featurize(agg, test, sensors)
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
    # `engine` is the globally-unique id; condition correction is already done,
    # so the estimator only caps, scales, fits, and scores.
    engine = TribblePredictiveHealth(
        condition_correction=False, unit_col="engine", **CONFIGS[agg]
    )
    t0 = time.perf_counter()
    engine.fit_featurized(train, feature_cols)
    fit_seconds = time.perf_counter() - t0

    m = engine.score_featurized(test)
    scored = engine.predict_samples_featurized(test)  # per-row, for the breakdown
    per_dataset = {
        ds: float(np.sqrt(mean_squared_error(sub["rul"], sub["predicted_rul"])))
        for ds, sub in scored.groupby("dataset")
    }
    return dict(
        config=agg,
        n_train=min(len(train), CONFIGS[agg]["max_train_rows"] or len(train)),
        n_test=len(test),
        n_engines=int(test["engine"].nunique()),
        n_rules=engine.n_rules_,
        fit_seconds=fit_seconds,
        per_sample_rmse=m["per_sample_rmse"],
        per_engine_rmse=m["per_engine_rmse"],
        per_engine_nasa=m["per_engine_nasa"],
        per_dataset=per_dataset,
    )


def write_report(results, processed, skipped):
    lines = [
        "# N-CMAPSS RUL, pooled across all datasets",
        "",
        "The 18-real-sensor pipeline (the `TribblePredictiveHealth` engine, see "
        "`cmapss_ds02_rul.py`) pooled over every usable N-CMAPSS file -- each "
        "contributing its own official train/test engines -- and scored two ways. "
        "Regenerated by `cmapss_all_datasets.py`.",
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
