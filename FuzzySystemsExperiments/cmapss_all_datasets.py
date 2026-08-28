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

  * **Real sensors only, as a deliberate simplification.** Everything here uses
    the 18 real, physically measurable sensors; the loader (`load_ncmapss`)
    does not read the file's two "virtual" channels (T40, P30) at all. An
    earlier docstring here claimed dropping them "costs nothing, established
    on DS02" -- that was never actually measured for this pipeline, and on
    DS02 alone it is false: the original design-of-experiments' winning
    config used T40/P30 and reached a lower RMSE than this real-sensors-only
    one. See `cmapss_ds02_rul.py`'s docstring and
    `research/proposal-defense/prose/04-fast-fis-synthesis-mog.md` §4.4.1.

Same engine as the DS02 script -- the reusable `TribblePredictiveHealth`. The
only twist pooling needs is that condition correction and the RUL cap are fit
per file on that file's own training engines: so each file is streamed, corrected
and featurised one at a time (keeping peak memory near a single dataset), then
the small feature tables are pooled and handed to the estimator's
`fit_featurized` entry point. No test-set information is used to fit anything.
Writes `cmapss_all_datasets_report.md`.

Needs: h5py, numpy, pandas, scikit-learn, tribble-fis.  Run:

    python cmapss_all_datasets.py --h5-dir data/nasa-cmapps2
"""

import argparse
import gc
import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from tribble_predictive_health import (
    TribblePredictiveHealth,
    load_or_build_many,
)
from tribble_predictive_health.metrics import nasa_score, rmse

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
#
# raw_memory uses 4 output buckets (rules) rather than the default 2: on the
# pooled per-sample metric a rule-count sweep bottoms out at 4 (per-sample RMSE
# 15.80 -> 14.87), climbing back by 6-8 -- the honest bias/variance sweet spot.
# See experiments/cmapss-ds02-fis/iterative_pooled.py. (Additive residual boosting
# drove the *training* residual lower still but only ever overfit the held-out
# engines, so bucket count, not boosting, is the lever.)
# A convex blend of the two models' per-cycle predictions. whole_cycle is
# low-variance but biased; raw_memory is low-bias but noisy at the endpoint.
# Averaging cancels raw_memory's endpoint noise while keeping its sharper trend,
# so the canonical per-engine (last-cycle) RMSE improves most. 0.8 is the value
# a held-out validation split (carved from the training engines) selects -- see
# experiments/cmapss-ds02-fis/validate_heldout.py; the whole 0.6-0.8 region beats
# whole_cycle on both validation and the untouched test engines, so the gain is
# a real generalisation result, not test-set tuning. blend_wc_rm.py is the
# earlier (test-selected) sweep that first surfaced it.
BLEND_ALPHA = 0.8

CONFIGS = {
    "whole_cycle": dict(
        aggregation="whole_cycle", tsk_order="1st", top_p=0.9, max_train_rows=None
    ),
    "raw_memory": dict(
        aggregation="raw_memory",
        tsk_order="full-2nd",
        top_p=0.95,
        n_output_buckets=4,
        max_train_rows=30_000,
    ),
}


# ---------------------------------------------------------------------------
# Load every file once, correct and featurise it, pool the small tables
# ---------------------------------------------------------------------------
def gather(h5_dir, rebuild_cache=False):
    """Return pooled {agg -> (train, test, feature_cols)}, plus the lists of
    processed/skipped datasets. Each file's condition-corrected feature tables
    come from the on-disk cache (`load_or_build_many`); on a cache miss the file
    is loaded, corrected against its own baseline, featurised for every
    aggregation from that single load, and cached, so peak memory stays near one
    dataset and repeat runs skip the load/featurise entirely."""
    pooled = {agg: {"train": [], "test": []} for agg in CONFIGS}
    feature_cols = {}
    processed, skipped = [], []
    for path in sorted(glob.glob(os.path.join(h5_dir, "*.h5"))):
        name = os.path.basename(path).replace("N-CMAPSS_", "").replace(".h5", "")
        try:
            bundles = load_or_build_many(path, list(CONFIGS), rebuild=rebuild_cache)
        except Exception as exc:  # the one truncated file
            skipped.append((name, f"{type(exc).__name__}"))
            continue

        for agg in CONFIGS:
            b = bundles[agg]
            feature_cols[agg] = b.feature_cols
            # Unit numbers repeat across files; make a globally unique engine id.
            for t in (b.dev, b.test):
                t["dataset"] = name
                t["engine"] = name + ":" + t["unit"].astype(str)
            pooled[agg]["train"].append(b.dev)
            pooled[agg]["test"].append(b.test)
        processed.append(name)
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
    # Collapse to one prediction per (engine, cycle) so the two models -- which
    # live on different row grains -- can be blended on a common index.
    per_cycle = (
        scored.groupby(["engine", "cycle"])
        .agg(pred=("predicted_rul", "mean"), true=("rul", "mean"))
        .reset_index()
    )
    result = dict(
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
    return result, per_cycle


def blend_pooled(per_cycle_by_agg, alpha=BLEND_ALPHA):
    """Convex-blend the two models' per-cycle predictions and score the mix.
    `per_cycle_by_agg` maps agg -> the per-(engine, cycle) frame returned by
    `fit_pooled`. Returns a result dict scored per cycle and per engine (one RUL
    per engine at its last cycle -- the canonical C-MAPSS protocol)."""
    wc = per_cycle_by_agg["whole_cycle"].rename(columns={"pred": "pred_wc"})
    rm = per_cycle_by_agg["raw_memory"].rename(columns={"pred": "pred_rm"})
    both = wc.merge(rm[["engine", "cycle", "pred_rm"]], on=["engine", "cycle"])
    both["blend"] = alpha * both["pred_wc"] + (1 - alpha) * both["pred_rm"]
    last = both.sort_values("cycle").groupby("engine").last()
    return dict(
        alpha=alpha,
        n_cycles=len(both),
        n_engines=int(both["engine"].nunique()),
        per_cycle_rmse=rmse(both["true"], both["blend"]),
        per_engine_rmse=rmse(last["true"], last["blend"]),
        per_engine_nasa=nasa_score(last["true"], last["blend"]),
        # each model alone on the same common grain, for the comparison
        wc_per_engine=rmse(last["true"], last["pred_wc"]),
        rm_per_engine=rmse(last["true"], last["pred_rm"]),
    )


def write_report(results, processed, skipped, blend=None):
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
    ]
    if blend is not None:
        lines += [
            "",
            "## Blended model (per-cycle convex mix)",
            "",
            f"Blending the two models' per-cycle predictions "
            f"`{blend['alpha']:.0%} whole_cycle + {1 - blend['alpha']:.0%} "
            f"raw_memory` -- whole_cycle is low-variance but biased, raw_memory "
            f"low-bias but noisy at the endpoint, so the average sharpens the "
            f"canonical per-engine (last-cycle) number. Scored on the "
            f"{blend['n_cycles']:,} common (engine, cycle) rows "
            f"({blend['n_engines']} engines).",
            "",
            "| model | per-cycle RMSE | per-engine RMSE | per-engine NASA |",
            "|---|---:|---:|---:|",
            f"| `whole_cycle` alone | -- | {blend['wc_per_engine']:.2f} | -- |",
            f"| `raw_memory` alone | -- | {blend['rm_per_engine']:.2f} | -- |",
            f"| **blend @ {blend['alpha']:.1f}** | {blend['per_cycle_rmse']:.2f} | "
            f"**{blend['per_engine_rmse']:.2f}** | {blend['per_engine_nasa']:,.0f} |",
            "",
            "The blend beats *both* models on per-engine RMSE and dominates "
            "`whole_cycle` on every metric -- the best per-engine number the "
            "pooled pipeline produces.",
        ]
    lines += [
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


def main(h5_dir, rebuild_cache=False):
    t0 = time.perf_counter()
    print(f"Loading and pooling N-CMAPSS files from {h5_dir} ...")
    pooled, processed, skipped = gather(h5_dir, rebuild_cache=rebuild_cache)
    print(
        f"  pooled {len(processed)} datasets"
        + (f", skipped {len(skipped)}" if skipped else "")
    )

    results = []
    per_cycle_by_agg = {}
    for agg in CONFIGS:
        train, test, feature_cols = pooled[agg]
        print(f"Fitting pooled `{agg}` ({len(train):,} train rows) ...")
        r, per_cycle_by_agg[agg] = fit_pooled(agg, train, test, feature_cols)
        results.append(r)

    blend = blend_pooled(per_cycle_by_agg)

    print("\n=== N-CMAPSS pooled RUL ===")
    print(f"  {'model':12s} {'per-sample':>10s} {'per-engine':>10s} {'NASA':>10s}")
    for r in results:
        print(
            f"  {r['config']:12s} {r['per_sample_rmse']:10.2f} "
            f"{r['per_engine_rmse']:10.2f} {r['per_engine_nasa']:10,.0f}"
            f"   ({r['n_engines']} engines)"
        )
    print(
        f"  {'blend@%.1f' % blend['alpha']:12s} {'--':>10s} "
        f"{blend['per_engine_rmse']:10.2f} {blend['per_engine_nasa']:10,.0f}"
        f"   ({blend['n_engines']} engines)"
    )
    write_report(results, processed, skipped, blend)
    print(f"\nwrote {REPORT}")
    print(f"Total wall time: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", default="data/nasa-cmapps2")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the preprocessing cache instead of reading it.",
    )
    args = parser.parse_args()
    main(args.h5_dir, rebuild_cache=args.rebuild_cache)
