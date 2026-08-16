"""Full N-CMAPSS RUL analysis: one combined train/test split, pooled across
every available dataset file.

Every N-CMAPSS_DS0*.h5 file ships its own official train/test unit split
(the `A_dev`/`A_test` groups -- the same split documented in Arias Chao et
al. 2021's "Aircraft Engine Run-to-Failure Dataset under Real Flight
Conditions" paper). This script respects that split *per file*, pools all
files' training units into one combined training set and all files' test
units into one combined held-out test set, and trains a single model on
the combined data -- rather than the DOE's earlier per-file zero-shot check
(tune on DS02, apply elsewhere unchanged). This is the "what happens if you
legitimately train across the whole dataset" experiment.

Single, sequential, dependency-light script (h5py, numpy, pandas,
scikit-learn, tribble-fis only) -- read top to bottom, rerun end to end:

    python cmapss_rul_full_analysis.py

Machine-learning hygiene (why the test set isn't poisoned):
  - Condition-correction regressions (each sensor channel regressed against
    the W operating-condition channels) are fit per file using ONLY that
    file's own training units' early "healthy" cycles, then applied
    (never re-fit) to both that file's training AND test rows.
  - The StandardScaler and the per-unit RUL cap are fit ONLY on the pooled
    TRAINING table, after all files are combined, then applied (never
    re-fit) to the pooled test table.
  - TribbleRegressor's internal feature selection happens inside
    model.fit(X_train, y_train) on training data only, by construction.
  - Raw unit numbers repeat across files (DS01's unit 2 and DS02's unit 2
    are different physical engines) -- every groupby in this script keys on
    the (dataset, unit) pair, never on unit alone, so no two files' engines
    are ever accidentally merged.

Memory note: each file's raw per-sample arrays (millions of rows) are
loaded, corrected, and aggregated down to a few hundred/thousand rows
*before* the next file is loaded -- only the small aggregated tables are
pooled across files, so peak memory is bounded by one file at a time, not
by the whole dataset's raw size.
"""

import glob
import os
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor

H5_DIR = "NASA-CMAPSS"
REPORT_PATH = "FuzzySystemsExperiments/cmapss_rul_full_analysis_report.md"

# Pooling every file's training data can exceed what this DOE has ever fit
# in one shot -- cap it via a fixed-seed random subsample rather than let
# memory grow unboundedly. Test data is NEVER subsampled or capped.
TRAIN_CAP = 50_000

# X_v (the HDF5 file's "virtual sensor" group) is ordered [T40, P30, P45,
# W21, ...] -- only the first two are the published baselines' "condition
# monitoring signals" (Arias Chao et al. 2021, Table 2); the rest are
# simulator-internal quantities not measurable on a real aircraft. We
# condition-correct both up front (cheap) and let each pipeline pick how
# many of them to actually use as model input.
CORRECT_N_XV = 2

PIPELINES = {
    # aggregation: "whole_cycle" (one row/cycle, mean/std/min/max/last
    # stats) or "raw_memory" (subsampled raw stream through
    # MemoryWindowFeatureExtractor). Hyperparameters are this DOE's already-
    # confirmed best config for each -- this script pools data across files,
    # it does not re-tune.
    "honest": dict(  # physical sensors only (18ch): W + X_s
        n_xv=0,
        aggregation="whole_cycle",
        tribble_kwargs=dict(
            tsk_order="1st",
            n_gaussians=0,
            top_p=0.9,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
    ),
    "best": dict(  # + T40, P30 (20ch) -- the published CNN/MLP baselines' exact input set
        n_xv=2,
        aggregation="raw_memory",
        tribble_kwargs=dict(
            tsk_order="full-2nd",
            n_gaussians=0,
            top_p=0.95,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
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


def to_frame(data: dict, var: dict, split: str, dataset: str) -> pd.DataFrame:
    d = data[split]
    df = pd.DataFrame(
        {
            "dataset": dataset,
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
# Condition correction: fit per file, on that file's own training units only
# --------------------------------------------------------------------------
def fit_condition_correction(
    df: pd.DataFrame, sensor_cols, condition_cols, baseline_cycles=15
):
    order = df.groupby(["dataset", "unit"]).cumcount()
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
# RUL target: per-(dataset, unit) physical cap from the `hs` health flag.
# Fit on the pooled TRAINING table only -- see module docstring.
# --------------------------------------------------------------------------
def physical_rul_cap(table: pd.DataFrame) -> dict:
    caps = {}
    for key, sub in table.groupby(["dataset", "unit"]):
        sub = sub.sort_values("cycle")
        unhealthy = sub[sub["hs"] == 0]
        onset = unhealthy["cycle"].min() if len(unhealthy) else sub["cycle"].max()
        at_or_after = sub[sub["cycle"] >= onset]
        caps[key] = float(
            at_or_after["RUL"].max() if len(at_or_after) else sub["RUL"].max()
        )
    return caps


def capped_rul(table: pd.DataFrame, caps: dict) -> np.ndarray:
    keys = list(zip(table["dataset"], table["unit"]))
    cap_arr = np.array([caps[k] for k in keys])
    return np.minimum(table["RUL"].astype(float).to_numpy(), cap_arr)


# --------------------------------------------------------------------------
# Aggregation: one row per (dataset, unit, cycle), or a subsampled raw-
# memory stream, one row per (dataset, unit) subsample.
# --------------------------------------------------------------------------
AGG_FUNCS = ["mean", "std", "min", "max", "last"]


def aggregate_whole_cycle(df: pd.DataFrame, feat_cols) -> pd.DataFrame:
    g = df.groupby(["dataset", "unit", "cycle"], sort=True)
    feat = g[feat_cols].agg(AGG_FUNCS)
    feat.columns = ["_".join(c) for c in feat.columns]
    meta = g.agg(RUL=("RUL", "first"), hs=("hs", "min"))
    return feat.join(meta).reset_index()


def aggregate_raw_memory(
    df: pd.DataFrame, feat_cols, stride: int = 200
) -> pd.DataFrame:
    extractor = MemoryWindowFeatureExtractor(window_size=5, memory_size=2)
    frames = []
    for (dataset, unit), sub in df.groupby(["dataset", "unit"], sort=True):
        sub = sub.iloc[::stride].reset_index(drop=True)
        mem = extractor.prepare_sequences(sub, feat_cols, include_time=False)
        mem["dataset"] = dataset
        mem["unit"] = unit
        mem["cycle"] = sub["cycle"].values
        mem["RUL"] = sub["RUL"].values
        mem["hs"] = sub["hs"].values
        frames.append(mem)
    result = pd.concat(frames, ignore_index=True)
    mem_cols = [
        c for c in result.columns if c not in ("dataset", "unit", "cycle", "RUL", "hs")
    ]
    result[mem_cols] = result[mem_cols].bfill().ffill()
    return result


def nasa_score(y_true, y_pred) -> float:
    delta = np.asarray(y_true) - np.asarray(y_pred)
    alpha = np.where(delta > 0, 1.0 / 13.0, 1.0 / 10.0)
    return float(np.sum(np.exp(alpha * np.abs(delta))))


# --------------------------------------------------------------------------
# Per-file processing: load -> correct -> aggregate (both pipelines) ->
# discard the raw per-sample data before returning. Called once per file,
# sequentially -- this is what keeps peak memory bounded.
# --------------------------------------------------------------------------
def process_file(h5_path: str, dataset_name: str) -> dict:
    data, var = load_h5(h5_path)
    df_dev = to_frame(data, var, "dev", dataset_name)
    df_test = to_frame(data, var, "test", dataset_name)
    del data

    w_cols = [f"W_{n}" for n in var["W"]]
    xs_cols = [f"Xs_{n}" for n in var["X_s"]]
    xv_cols = [f"Xv_{n}" for n in var["X_v"]]
    correct_cols = xs_cols + xv_cols[:CORRECT_N_XV]

    models = fit_condition_correction(df_dev, correct_cols, w_cols)
    df_dev = apply_condition_correction(df_dev, correct_cols, w_cols, models)
    df_test = apply_condition_correction(df_test, correct_cols, w_cols, models)

    out = {}
    for name, cfg in PIPELINES.items():
        feat_cols = w_cols + xs_cols + xv_cols[: cfg["n_xv"]]
        agg_fn = (
            aggregate_whole_cycle
            if cfg["aggregation"] == "whole_cycle"
            else aggregate_raw_memory
        )
        out[name] = (agg_fn(df_dev, feat_cols), agg_fn(df_test, feat_cols))

    del df_dev, df_test
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(results: dict, file_log: list, total_seconds: float, out_path: str):
    lines = [
        "# N-CMAPSS full-dataset RUL analysis",
        "",
        "One combined training set and one combined held-out test set, "
        "pooled from every N-CMAPSS file's own official train/test unit "
        "split. Condition-correction is fit per file on that file's own "
        "training units only; the scaler and per-unit RUL cap are fit once "
        "on the pooled training table only. No test-set information is "
        "used to fit anything.",
        "",
        "NASA score is exponential in per-sample error, so a handful of "
        "large-outlier predictions dominate the sum -- on a large, "
        "heterogeneous pooled test set this can inflate the score by many "
        "orders of magnitude. Treat RMSE as the primary metric; NASA score "
        "is included for completeness, not as a normalized number.",
        "",
        "## Files processed",
        "",
        "| dataset | status | seconds |",
        "|---|---|---:|",
    ]
    for r in file_log:
        lines.append(
            f"| {r['dataset']} | {r['status']} | {r.get('seconds', float('nan')):.1f} |"
        )

    for name, r in results.items():
        subsample_note = (
            f" (subsampled from {r['n_pooled_train']:,} pooled rows, seed=42)"
            if r["n_pooled_train"] > r["n_train"]
            else ""
        )
        lines += [
            "",
            f"## Pipeline: `{name}`",
            "",
            f"- Training rows: {r['n_train']:,}{subsample_note}  |  pooled test rows: {r['n_test']:,}",
            f"- **RMSE (combined test set): {r['rmse']:.2f}**  |  NASA score: {r['nasa_score']:,.0f}",
            f"- Fit time: {r['fit_seconds']:.2f}s",
            "",
            "Per-dataset test RMSE (same trained model, broken out by source file):",
            "",
            "| dataset | RMSE | n |",
            "|---|---:|---:|",
        ]
        for dataset, (rmse_d, n_d) in sorted(r["per_dataset"].items()):
            lines.append(f"| {dataset} | {rmse_d:.2f} | {n_d} |")

    lines += ["", f"Total wall time: {total_seconds:.1f}s"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    files = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))
    if not files:
        raise SystemExit(f"No .h5 files found under {H5_DIR}/")

    pooled = {name: dict(train=[], test=[]) for name in PIPELINES}
    file_log = []

    for h5_path in files:
        dataset_name = (
            os.path.basename(h5_path).replace("N-CMAPSS_", "").replace(".h5", "")
        )
        dataset_name = dataset_name.split("-")[0]  # "DS08a-009" -> "DS08a"
        print(f"\n{'=' * 78}\n{dataset_name}  ({h5_path})\n{'=' * 78}")
        t0 = time.perf_counter()
        try:
            per_pipeline = process_file(h5_path, dataset_name)
        except Exception as e:
            print(f"  SKIPPED: {e!r}")
            file_log.append(dict(dataset=dataset_name, status=f"skipped: {e!r}"))
            continue
        for name, (train_tab, test_tab) in per_pipeline.items():
            pooled[name]["train"].append(train_tab)
            pooled[name]["test"].append(test_tab)
        dt = time.perf_counter() - t0
        print(
            f"  ok in {dt:.1f}s  "
            f"(honest: {len(per_pipeline['honest'][0])} train / {len(per_pipeline['honest'][1])} test rows; "
            f"best: {len(per_pipeline['best'][0])} train / {len(per_pipeline['best'][1])} test rows)"
        )
        file_log.append(dict(dataset=dataset_name, status="ok", seconds=dt))

    results = {}
    for name, cfg in PIPELINES.items():
        print(f"\n{'=' * 78}\nPooled pipeline: {name}\n{'=' * 78}")
        train_tab = pd.concat(pooled[name]["train"], ignore_index=True)
        test_tab = pd.concat(pooled[name]["test"], ignore_index=True)
        n_pooled = len(train_tab)
        if n_pooled > TRAIN_CAP:
            # Pooling every file's training data can produce a training set
            # far larger than any single file this DOE has tuned/tested
            # against (confirmed OOM at ~221k pooled rows at 'full-2nd'
            # order). A fixed-seed random subsample keeps the pipeline
            # tractable -- the same principle TribbleRegressor's own
            # `max_samples` already applies internally to construction,
            # just extended to the whole fit. Test data is never subsampled.
            train_tab = train_tab.sample(n=TRAIN_CAP, random_state=42).reset_index(
                drop=True
            )
            print(
                f"  subsampled pooled training set: {n_pooled:,} -> {TRAIN_CAP:,} rows"
            )
        feat_cols = [
            c
            for c in train_tab.columns
            if c not in ("dataset", "unit", "cycle", "RUL", "hs")
        ]

        # Fit ONLY on pooled training data: RUL cap, scaler, and (inside
        # TribbleRegressor.fit) feature selection and antecedent
        # construction. Test data is only ever transformed/predicted on.
        caps = physical_rul_cap(train_tab)
        y_train = capped_rul(train_tab, caps)
        y_test_true = test_tab["RUL"].astype(float).to_numpy()

        X_train = train_tab[feat_cols].to_numpy(dtype=np.float64)
        X_test = test_tab[feat_cols].to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = TribbleRegressor(
            random_state=42, max_samples=2000, **cfg["tribble_kwargs"]
        )
        t0 = time.perf_counter()
        model.fit(X_train_s, y_train)
        fit_seconds = time.perf_counter() - t0

        pred = model.predict(X_test_s)
        rmse = float(np.sqrt(mean_squared_error(y_test_true, pred)))
        score = nasa_score(y_test_true, pred)

        per_dataset = {}
        for dataset, sub_idx in test_tab.groupby("dataset").groups.items():
            idx = test_tab.index.get_indexer(sub_idx)
            rmse_d = float(np.sqrt(mean_squared_error(y_test_true[idx], pred[idx])))
            per_dataset[dataset] = (rmse_d, len(idx))

        print(
            f"pooled train={len(train_tab):,}  test={len(test_tab):,}  "
            f"fit={fit_seconds:.2f}s  RMSE={rmse:.2f}  NASA score={score:,.0f}"
        )
        for dataset, (rmse_d, n_d) in sorted(per_dataset.items()):
            print(f"  {dataset}: rmse={rmse_d:.2f}  n={n_d}")

        results[name] = dict(
            rmse=rmse,
            nasa_score=score,
            fit_seconds=fit_seconds,
            n_train=len(train_tab),
            n_pooled_train=n_pooled,
            n_test=len(test_tab),
            per_dataset=per_dataset,
        )

    total_seconds = time.perf_counter() - t_start
    print(f"\nTotal wall time: {total_seconds:.1f}s")
    write_report(results, file_log, total_seconds, REPORT_PATH)


if __name__ == "__main__":
    main()
