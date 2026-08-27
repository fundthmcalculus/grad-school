"""#8 Blend the two pooled models -- whole_cycle (per-engine winner) and
raw_memory (per-sample winner) -- to see if a mix beats either on the metric it
loses.

The two live on different grains: whole_cycle is one prediction per (engine,
cycle); raw_memory is per subsampled sample (stride 200), so it does NOT cover
every cycle. We collapse raw_memory to per-cycle (mean over its samples in the
cycle), inner-join on (engine, cycle), and blend
    pred(alpha) = alpha * whole_cycle + (1 - alpha) * raw_memory
sweeping alpha. Coverage of the join is reported (the blend is only defined
where raw_memory has a sample). Per-cycle RMSE and the canonical per-engine
RMSE/NASA (one RUL per engine at its last *common* cycle) are scored for each
alpha. Run from the repo root (needs data/nasa-cmapps2/):

    python experiments/cmapss-ds02-fis/blend_wc_rm.py
"""

import contextlib
import csv
import io
import os

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _ds02_harness import bootstrap, rmse  # noqa: E402

bootstrap("FuzzySystemsExperiments", os.path.dirname(__file__))
import cmapss_all_datasets as cad  # noqa: E402
from tribble_predictive_health import TribblePredictiveHealth  # noqa: E402
from tribble_predictive_health.metrics import nasa_score  # noqa: E402

OUT = "outputs/ds02-iterative"
os.makedirs(OUT, exist_ok=True)


def per_cycle_pred(engine, table):
    """(engine, cycle) -> mean predicted_rul, mean true rul."""
    scored = engine.predict_samples_featurized(table)
    g = scored.groupby(["engine", "cycle"])
    return g.agg(pred=("predicted_rul", "mean"), true=("rul", "mean")).reset_index()


def per_engine_last(df, pred_col="pred"):
    last = df.sort_values("cycle").groupby("engine").last()
    return rmse(last["true"], last[pred_col]), nasa_score(last["true"], last[pred_col])


def main(h5_dir):
    print(f"Loading + pooling all datasets from {h5_dir} ...")
    pooled, processed, skipped = cad.gather(h5_dir)  # both aggregations
    rows = []
    frames = {}
    for agg in ("whole_cycle", "raw_memory"):
        train, test, cols = pooled[agg]
        eng = TribblePredictiveHealth(
            condition_correction=False, unit_col="engine", **cad.CONFIGS[agg]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            eng.fit_featurized(train, cols)
        frames[agg] = per_cycle_pred(eng, test)
        print(f"  {agg}: {len(frames[agg]):,} per-cycle rows")

    wc = frames["whole_cycle"].rename(columns={"pred": "pred_wc", "true": "true_wc"})
    rm = frames["raw_memory"].rename(columns={"pred": "pred_rm", "true": "true_rm"})
    both = wc.merge(
        rm[["engine", "cycle", "pred_rm"]], on=["engine", "cycle"], how="inner"
    )
    both = both.rename(columns={"true_wc": "true"})
    cov = len(both) / len(wc)
    print(
        f"\njoined {len(both):,} common (engine,cycle) rows "
        f"({cov:.0%} of whole_cycle's cycles, {both['engine'].nunique()} engines)\n"
    )

    print("  alpha  per-cycle RMSE   per-engine RMSE   NASA   (alpha=1 -> whole_cycle)")
    for alpha in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        both["blend"] = alpha * both["pred_wc"] + (1 - alpha) * both["pred_rm"]
        pc = rmse(both["true"], both["blend"])
        pe, na = per_engine_last(both, "blend")
        print(f"  {alpha:4.1f}   {pc:12.2f}   {pe:14.2f}   {na:8,.0f}")
        rows.append((alpha, pc, pe, na))

    # reference: each model's own per-cycle / per-engine on the COMMON grain
    for tag, col in (("whole_cycle", "pred_wc"), ("raw_memory", "pred_rm")):
        pc = rmse(both["true"], both[col])
        pe, na = per_engine_last(both, col)
        print(f"  [{tag:11s}]         {pc:12.2f}   {pe:14.2f}   {na:8,.0f}")

    with open(os.path.join(OUT, "blend_wc_rm.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "per_cycle_rmse", "per_engine_rmse", "per_engine_nasa"])
        w.writerows(rows)
    print(f"\nwrote {os.path.join(OUT, 'blend_wc_rm.csv')}  (coverage {cov:.0%})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", default="data/nasa-cmapps2")
    main(ap.parse_args().h5_dir)
