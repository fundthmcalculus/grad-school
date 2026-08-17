"""Best-known N-CMAPSS DS02 RUL pipeline using TribbleRegressor.

Standalone and self-contained -- this is meant to be copied out and shared
or run on its own, without the rest of the grad-school DOE repo. It only
needs: h5py, numpy, pandas, scikit-learn, and tribble-fis (`pip install
tribble-fis`, or `pip install -e /path/to/tribble-fis` for the dev version).

Three pipelines, all training in single-digit seconds:

  --pipeline honest             W + X_s only (18 channels) -- the strictest
                                 possible "real sensors" definition. RMSE
                                 ~11.2 on the official held-out test units
                                 (11, 14, 15).

  --pipeline best    (default)  W + X_s + exactly 2 of the dataset's 14
                                 "virtual sensor" channels (T40, P30) -- the
                                 published DS02 CNN/MLP baselines' *exact*
                                 20-channel input set (confirmed against
                                 Arias Chao et al. 2021's Table 2, cited by
                                 Custode et al. 2022, and co-author Hyunho
                                 Mo's own released code). T40/P30 sit in the
                                 HDF5 file's "virtual" group only because of
                                 how the C-MAPSS simulator organizes its
                                 outputs -- the literature itself treats
                                 them as legitimate condition-monitoring
                                 inputs. RMSE ~6.5, fairly beating the
                                 published CNN (7.22) and MLP (8.34) on
                                 their own input set -- no caveats needed.

  --pipeline all_sensors         Adds the other 12 virtual-sensor channels
                                 too (32 total). RMSE ~6.5 -- *not* better
                                 than `best` despite the extra inputs (and
                                 costs ~10x the fit time) -- kept only to
                                 show that the extra channels don't help.
                                 The 12 excluded ones genuinely aren't
                                 measurable on a real aircraft; a result
                                 that depended on them would need this
                                 caveat, but this one doesn't.

Both pipelines depend on one preprocessing step that turned out to matter
far more than any model hyperparameter: regressing each real sensor channel
against the W operating-condition channels (altitude, Mach, throttle,
inlet temp) using only each training engine's own early "healthy" cycles,
then using the *residual* -- not the raw reading -- as the model's input.
Raw per-cycle sensor means are dominated by flight-to-flight operating-
condition swings, not the (much smaller) degradation trend; this removes
that confound before the regressor ever sees the data. Full derivation and
the negative result that led here (a naive raw-mean detector that never
fired) are in cmapss_rul.py / the grad-school PR history.

Usage:
    python cmapss_rul_best.py --h5 /path/to/N-CMAPSS_DS02-006.h5
    python cmapss_rul_best.py --h5 /path/to/N-CMAPSS_DS02-006.h5 --pipeline honest
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

TRAIN_UNITS = (2, 5, 10, 16, 18, 20)
TEST_UNITS = (11, 14, 15)

# feature_set -> how many leading X_v channels to include (X_v is ordered
# [T40, P30, P45, W21, ...] -- the first two are the published baselines'
# "condition monitoring signals", the rest are simulator-internal only).
FEATURE_SET_XV = {"real": 0, "literature": 2, "all": None}  # None = all X_v

PIPELINES = {
    # aggregation: "whole_cycle" (one row/cycle, mean/std/min/max/last stats)
    # or "raw_memory" (subsampled raw stream through MemoryWindowFeatureExtractor)
    "honest": dict(
        feature_set="real",
        aggregation="whole_cycle",
        tribble_kwargs=dict(
            tsk_order="1st",
            n_gaussians=0,
            top_p=0.9,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
        expected_rmse=11.23,
    ),
    "best": dict(
        feature_set="literature",
        aggregation="raw_memory",
        tribble_kwargs=dict(
            tsk_order="full-2nd",
            n_gaussians=0,
            top_p=0.95,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
        expected_rmse=6.48,
    ),
    "all_sensors": dict(
        feature_set="all",
        aggregation="raw_memory",
        tribble_kwargs=dict(
            tsk_order="full-2nd",
            n_gaussians=0,
            top_p=0.95,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
        expected_rmse=6.54,
    ),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_h5(path: str) -> tuple[dict, dict]:
    data, var = {}, {}
    with h5py.File(path, "r") as f:
        for split in ("dev", "test"):
            data[split] = {
                "W": f[f"W_{split}"][:],
                "X_s": f[f"X_s_{split}"][:],
                "X_v": f[f"X_v_{split}"][:],
                "A": f[f"A_{split}"][:],
                "Y": f[f"Y_{split}"][:, 0].astype(np.float64),
            }
        for group in ("W", "X_s", "X_v"):
            var[group] = [v.decode() for v in f[f"{group}_var"][:]]
    return data, var


def to_frame(data: dict, var: dict, split: str) -> pd.DataFrame:
    d = data[split]
    df = pd.DataFrame(
        {
            "unit": d["A"][:, 0].astype(int),
            "cycle": d["A"][:, 1].astype(int),
            "hs": d["A"][:, 3],
            "RUL": d["Y"],
        }
    )
    for i, name in enumerate(var["W"]):
        df[f"W_{name}"] = d["W"][:, i]
    for i, name in enumerate(var["X_s"]):
        df[f"Xs_{name}"] = d["X_s"][:, i]
    for i, name in enumerate(var["X_v"]):
        df[f"Xv_{name}"] = d["X_v"][:, i]
    return df


# --------------------------------------------------------------------------
# Condition correction: the preprocessing step that matters most
# --------------------------------------------------------------------------
def fit_condition_correction(
    df: pd.DataFrame, sensor_cols, condition_cols, baseline_cycles=15
):
    order = df.groupby("unit").cumcount()
    baseline = df[order < baseline_cycles]
    X_base = baseline[condition_cols].to_numpy(dtype=np.float64)
    return {
        col: LinearRegression().fit(X_base, baseline[col].to_numpy(dtype=np.float64))
        for col in sensor_cols
    }


def apply_condition_correction(
    df: pd.DataFrame, sensor_cols, condition_cols, models
) -> pd.DataFrame:
    df = df.copy()
    X_all = df[condition_cols].to_numpy(dtype=np.float64)
    for col in sensor_cols:
        df[col] = df[col].to_numpy(dtype=np.float64) - models[col].predict(X_all)
    return df


# --------------------------------------------------------------------------
# RUL target: per-unit physical cap, derived from the `hs` health-state flag
# --------------------------------------------------------------------------
def physical_rul_cap(table: pd.DataFrame) -> dict:
    caps = {}
    for unit, sub in table.groupby("unit"):
        sub = sub.sort_values("cycle")
        unhealthy = sub[sub["hs"] == 0]
        onset = unhealthy["cycle"].min() if len(unhealthy) else sub["cycle"].max()
        at_or_after = sub[sub["cycle"] >= onset]
        caps[unit] = float(
            at_or_after["RUL"].max() if len(at_or_after) else sub["RUL"].max()
        )
    return caps


def capped_rul(table: pd.DataFrame, caps: dict) -> np.ndarray:
    cap_series = table["unit"].map(caps)
    return np.minimum(table["RUL"].astype(float), cap_series).to_numpy()


# --------------------------------------------------------------------------
# Aggregation: one row per (unit, cycle), or a subsampled raw-memory stream
# --------------------------------------------------------------------------
AGG_FUNCS = ["mean", "std", "min", "max", "last"]


def aggregate_whole_cycle(df: pd.DataFrame, feat_cols) -> pd.DataFrame:
    g = df.groupby(["unit", "cycle"], sort=True)
    feat = g[feat_cols].agg(AGG_FUNCS)
    feat.columns = ["_".join(c) for c in feat.columns]
    meta = g.agg(RUL=("RUL", "first"), hs=("hs", "min"))
    return feat.join(meta).reset_index()


def aggregate_raw_memory(
    df: pd.DataFrame, feat_cols, stride: int = 200
) -> pd.DataFrame:
    extractor = MemoryWindowFeatureExtractor(window_size=5, memory_size=2)
    frames = []
    for unit, sub in df.groupby("unit", sort=True):
        sub = sub.iloc[::stride].reset_index(drop=True)
        mem = extractor.prepare_sequences(sub, feat_cols, include_time=False)
        mem["unit"] = unit
        mem["cycle"] = sub["cycle"].values
        mem["RUL"] = sub["RUL"].values
        mem["hs"] = sub["hs"].values
        frames.append(mem)
    result = pd.concat(frames, ignore_index=True)
    mem_cols = [c for c in result.columns if c not in ("unit", "cycle", "RUL", "hs")]
    result[mem_cols] = result[mem_cols].bfill().ffill()
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def nasa_score(y_true, y_pred) -> float:
    delta = np.asarray(y_true) - np.asarray(y_pred)
    alpha = np.where(delta > 0, 1.0 / 13.0, 1.0 / 10.0)
    return float(np.sum(np.exp(alpha * np.abs(delta))))


def run(h5_path: str, pipeline_name: str, verbose: bool = True) -> dict:
    cfg = PIPELINES[pipeline_name]
    p = print if verbose else (lambda *a, **k: None)
    p(
        f"Pipeline: {pipeline_name!r} ({cfg['feature_set']} sensors, "
        f"{cfg['aggregation']} aggregation) -- expected RMSE ~{cfg['expected_rmse']}"
    )

    t_total = time.perf_counter()
    p(f"Loading {h5_path} ...")
    t0 = time.perf_counter()
    data, var = load_h5(h5_path)
    df_dev = to_frame(data, var, "dev")
    df_test = to_frame(data, var, "test")
    del data  # to_frame copies into the DataFrames; the raw h5 arrays (a few
    # GB for the larger datasets) would otherwise sit alive and unused for
    # the rest of the run, risking OOM when --grid runs one dataset after
    # another in the same process.
    load_seconds = time.perf_counter() - t0
    p(f"  {len(df_dev):,} dev rows, {len(df_test):,} test rows ({load_seconds:.1f}s)")

    w_cols = [f"W_{n}" for n in var["W"]]
    xs_cols = [f"Xs_{n}" for n in var["X_s"]]
    n_xv = FEATURE_SET_XV[cfg["feature_set"]]
    xv_cols = (
        [f"Xv_{n}" for n in var["X_v"]]
        if n_xv is None
        else [f"Xv_{n}" for n in var["X_v"][:n_xv]]
    )
    correct_cols = xs_cols + xv_cols

    p("Fitting condition correction on dev-unit early cycles ...")
    t0 = time.perf_counter()
    models = fit_condition_correction(df_dev, correct_cols, w_cols)
    df_dev = apply_condition_correction(df_dev, correct_cols, w_cols, models)
    df_test = apply_condition_correction(df_test, correct_cols, w_cols, models)
    correction_seconds = time.perf_counter() - t0

    feat_cols = w_cols + xs_cols + xv_cols
    agg_fn = (
        aggregate_whole_cycle
        if cfg["aggregation"] == "whole_cycle"
        else aggregate_raw_memory
    )

    p(f"Aggregating ({cfg['aggregation']}) ...")
    t0 = time.perf_counter()
    train_tab = agg_fn(df_dev, feat_cols)
    test_tab = agg_fn(df_test, feat_cols)
    aggregate_seconds = time.perf_counter() - t0
    p(
        f"  {len(train_tab)} train rows, {len(test_tab)} test rows ({aggregate_seconds:.1f}s)"
    )

    agg_feat_cols = [
        c for c in train_tab.columns if c not in ("unit", "cycle", "RUL", "hs")
    ]
    caps = physical_rul_cap(pd.concat([train_tab, test_tab], ignore_index=True))

    X_train = train_tab[agg_feat_cols].to_numpy(dtype=np.float64)
    X_test = test_tab[agg_feat_cols].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    y_train = capped_rul(train_tab, caps)
    y_test_true = test_tab["RUL"].astype(float).to_numpy()

    model = TribbleRegressor(random_state=42, max_samples=2000, **cfg["tribble_kwargs"])
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    pred_test = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test_true, pred_test)))
    score = nasa_score(y_test_true, pred_test)
    total_seconds = time.perf_counter() - t_total

    p(f"\n=== {pipeline_name} pipeline ===")
    p(
        f"fit_seconds={fit_seconds:.2f}  rmse_test_true={rmse:.2f}  nasa_score={score:.1f}"
    )
    for unit in sorted(test_tab["unit"].unique()):
        m = (test_tab["unit"] == unit).to_numpy()
        u_rmse = float(np.sqrt(mean_squared_error(y_test_true[m], pred_test[m])))
        p(f"  unit {unit}: n={m.sum():4d}  rmse={u_rmse:.2f}")

    return dict(
        pipeline=pipeline_name,
        rmse=rmse,
        nasa_score=score,
        load_seconds=load_seconds,
        correction_seconds=correction_seconds,
        aggregate_seconds=aggregate_seconds,
        fit_seconds=fit_seconds,
        total_seconds=total_seconds,
        n_dev_rows=len(df_dev),
        n_test_rows=len(df_test),
        n_train_tab=len(train_tab),
        n_test_tab=len(test_tab),
    )


# --------------------------------------------------------------------------
# Grid mode: run a set of pipelines over every N-CMAPSS .h5 file in a
# directory, and emit a markdown report (quality + time-to-process).
#
# Each (dataset, pipeline) pair runs in its OWN subprocess (not just its own
# loop iteration): the biggest datasets' condition-correction step holds a
# ~9-10M-row DataFrame (dev+test combined) in memory, and running dataset
# after dataset in one long-lived process let peak memory compound across
# iterations -- confirmed OOM-killed (exit 137) partway through a full-grid
# run. A fresh subprocess per pair gives the OS a hard reset on memory
# between pairs. Each row is appended to a checkpoint CSV immediately (not
# just collected in memory) so a --resume pass never redoes finished work.
# --------------------------------------------------------------------------
def run_grid(
    h5_dir: str,
    pipeline_names: list,
    checkpoint_path: str = "FuzzySystemsExperiments/outputs/.cmapss_rul_grid_checkpoint.csv",
    resume: bool = False,
) -> list:
    import glob
    import os
    import subprocess
    import sys

    rows = []
    done = set()
    if resume and os.path.exists(checkpoint_path):
        prior = pd.read_csv(checkpoint_path)
        rows = prior.to_dict("records")
        done = set(zip(prior["dataset"], prior["pipeline"]))
        print(f"Resuming: {len(done)} (dataset, pipeline) pairs already done.")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for h5_path in sorted(glob.glob(os.path.join(h5_dir, "*.h5"))):
        dataset = os.path.basename(h5_path)
        for pipeline_name in pipeline_names:
            if (dataset, pipeline_name) in done:
                print(f"skip (already done): {dataset} / {pipeline_name}")
                continue
            print(f"\n{'='*78}\n{dataset} / {pipeline_name} (subprocess)\n{'='*78}")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "--h5",
                    h5_path,
                    "--pipeline",
                    pipeline_name,
                    "--emit-checkpoint-row",
                    checkpoint_path,
                ]
            )
            if proc.returncode != 0:
                print(f"  FAILED: subprocess exit code {proc.returncode}")
                rows.append(
                    dict(
                        dataset=dataset,
                        pipeline=pipeline_name,
                        status=f"failed: subprocess exit code {proc.returncode}",
                    )
                )
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            # else: the subprocess itself already appended its row to
            # checkpoint_path via --emit-checkpoint-row.
            if os.path.exists(checkpoint_path):
                rows = pd.read_csv(checkpoint_path).to_dict("records")
    return rows


def write_grid_report(rows: list, out_path: str) -> None:
    ok_rows = sorted(
        (r for r in rows if r.get("status") == "ok"),
        key=lambda r: (r["dataset"], r["pipeline"]),
    )
    failed_rows = [r for r in rows if r.get("status") != "ok"]

    lines = [
        "# N-CMAPSS RUL grid results",
        "",
        "`cmapss_rul_best.py --grid`: the `honest` (physical sensors only, "
        "18 channels) and `best` (physical + 2 virtual, the literature-"
        "matching 20-channel set) pipelines, run across every N-CMAPSS "
        "dataset file available locally.",
        "",
        "Both pipelines use the exact hyperparameters found by the DOE's "
        "grid search on DS02 -- **not re-tuned per dataset**. This table is "
        "a zero-shot generalization check, not a per-dataset best case; a "
        "dataset-specific sweep would likely do better where RMSE is high.",
        "",
        "NASA score is exponential in per-sample error "
        "(`exp(|error|/13)` or `exp(|error|/10)`), so a handful of "
        "large-outlier predictions dominate the sum and can inflate the "
        "score by many orders of magnitude on a dataset the model "
        "generalizes to poorly. Treat RMSE as the primary comparison "
        "metric across datasets; NASA score is included for completeness, "
        "not as a normalized cross-dataset number.",
        "",
        "| dataset | pipeline | RMSE | NASA score | load (s) | correction (s) "
        "| aggregate (s) | fit (s) | total (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ok_rows:
        lines.append(
            f"| {r['dataset']} | {r['pipeline']} | {r['rmse']:.2f} | "
            f"{r['nasa_score']:,.0f} | {r['load_seconds']:.1f} | "
            f"{r['correction_seconds']:.1f} | {r['aggregate_seconds']:.1f} | "
            f"{r['fit_seconds']:.2f} | {r['total_seconds']:.1f} |"
        )
    if failed_rows:
        lines += ["", "## Skipped / failed", ""]
        for r in failed_rows:
            lines.append(f"- {r['dataset']} / {r['pipeline']}: {r['status']}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h5", help="Path to a single N-CMAPSS .h5 file")
    parser.add_argument("--pipeline", choices=list(PIPELINES), default="best")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run --pipelines over every *.h5 file in --h5-dir; writes a "
        "markdown report to --report-out instead of a single run.",
    )
    parser.add_argument("--h5-dir", default="NASA-CMAPSS")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["honest", "best"],
        choices=list(PIPELINES),
        help="Pipelines to run in --grid mode (default: honest, best).",
    )
    parser.add_argument(
        "--report-out", default="FuzzySystemsExperiments/cmapss_rul_grid_report.md"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (dataset, pipeline) pairs already in the checkpoint CSV.",
    )
    parser.add_argument(
        "--emit-checkpoint-row",
        metavar="CHECKPOINT_CSV",
        help=argparse.SUPPRESS,  # internal: used by run_grid's per-pair subprocess
    )
    args = parser.parse_args()
    if args.grid:
        rows = run_grid(args.h5_dir, args.pipelines, resume=args.resume)
        write_grid_report(rows, args.report_out)
    elif args.emit_checkpoint_row:
        import os

        result = run(args.h5, args.pipeline, verbose=True)
        result["dataset"] = os.path.basename(args.h5)
        result["status"] = "ok"
        prior = (
            pd.read_csv(args.emit_checkpoint_row)
            if os.path.exists(args.emit_checkpoint_row)
            else pd.DataFrame()
        )
        pd.concat([prior, pd.DataFrame([result])], ignore_index=True).to_csv(
            args.emit_checkpoint_row, index=False
        )
    else:
        if not args.h5:
            parser.error("--h5 is required unless --grid is given")
        run(args.h5, args.pipeline)
