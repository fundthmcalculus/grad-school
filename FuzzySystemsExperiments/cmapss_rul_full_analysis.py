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

    python cmapss_rul_full_analysis.py            # fits/evaluates PIPELINES
    python cmapss_rul_full_analysis.py --tune      # + re-runs the ~10-minute
                                                    # search that produced the
                                                    # *_full_tuned entries

PIPELINES has four entries: `honest`/`best` (this DOE's DS02-only-tuned
configs, reused unchanged) and `honest_full_tuned`/`best_full_tuned` (found
by this script's own hyperparameter search on the pooled dataset -- see
PIPELINES' comments for the discovered numbers). All four are hardcoded, so
a normal run just fits and evaluates them; `--tune` re-runs the search that
found the tuned pair, for reproducibility, and warns if the search's
current winner has drifted from what's hardcoded.

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
  - The `--tune` search (see above) validates against a group-held-out
    slice of the pooled TRAINING data (whole (dataset, unit) engines set
    aside, never split across train/validation) -- the real held-out test
    set is only ever touched once, for the final evaluation of each
    pipeline's winning config.

Memory note: each file's raw per-sample arrays (millions of rows) are
loaded, corrected, and aggregated down to a few hundred/thousand rows
*before* the next file is loaded -- only the small aggregated tables are
pooled across files, so peak memory is bounded by one file at a time, not
by the whole dataset's raw size.
"""

import glob
import itertools
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

# Hyperparameter search for a config tuned on the *pooled* dataset itself,
# rather than reused from DS02-only tuning. Kept small and targeted (the
# construction axes this DOE has already found to matter most) since each
# candidate costs a real model fit -- a full Cartesian grid over every
# possible TribbleRegressor knob would not stay "trains in seconds".
TUNE_GRID = dict(
    tsk_order=["1st", "full-2nd"],
    top_p=[0.90, 0.95],
    norm_conorm=["probability", "hamacher"],
    l2_reg=[1e-6, 0.01],
)
TUNE_VAL_FRACTION = 0.2  # fraction of (dataset, unit) groups held out for validation
TUNE_SUBSAMPLE_CAP = 15_000  # rows per search candidate -- smaller than TRAIN_CAP
# so a 16-candidate grid search stays a couple of minutes, not tens of
# minutes; the winning config is refit on the full TRAIN_CAP-sized pool
# afterward for the number that's actually reported.
TUNE_SEED = 123

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
    # MemoryWindowFeatureExtractor). Hyperparameters are this DOE's
    # DS02-only-tuned config for each -- reused unchanged here, not re-tuned
    # on the pooled dataset.
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
    # Found by this script's own --tune search (TUNE_GRID below), validated
    # on a group-held-out slice of pooled TRAINING units, then confirmed on
    # the real held-out test set. "honest_full_tuned" converged to exactly
    # the same config as "honest" above -- the DS02-tuned default already
    # generalizes best across the pooled dataset for this pipeline.
    # "best_full_tuned" differs by one knob (top_p 0.95 -> 0.9, i.e.
    # slightly more permissive feature selection) and measurably improves
    # on "best": RMSE 17.70 -> 16.18 on the full pooled test set (DS06, the
    # hardest single dataset, improves from 32.75 -> 24.35). Hardcoded here
    # so a normal run doesn't have to redo a ~10-minute search to get it;
    # rerun with --tune to reproduce/refresh this discovery.
    "honest_full_tuned": dict(
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
    "best_full_tuned": dict(
        n_xv=2,
        aggregation="raw_memory",
        tribble_kwargs=dict(
            tsk_order="full-2nd",
            n_gaussians=0,
            top_p=0.9,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
    ),
}
# Maps each *_full_tuned pipeline back to the base pipeline it shares
# pooled data with (honest_full_tuned reads the same pooled tables as
# honest, just with different hyperparameters) -- see main().
TUNED_TO_BASE = {"honest_full_tuned": "honest", "best_full_tuned": "best"}
BASE_PIPELINES = {k: v for k, v in PIPELINES.items() if k not in TUNED_TO_BASE}


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
    for name, cfg in BASE_PIPELINES.items():
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
# Hyperparameter search on the pooled dataset itself (not reused from
# DS02-only tuning). Uses a group-held-out validation split carved out of
# the TRAINING pool only -- the real test set is never touched during the
# search, only for the one final evaluation after the winning config is
# chosen.
# --------------------------------------------------------------------------
def group_train_val_split(table: pd.DataFrame, val_fraction: float, seed: int):
    """Split by (dataset, unit) group, not by row -- a unit's cycles/samples
    must land entirely on one side, or validation RMSE would be optimistic
    (the model would see that engine's own data on both sides)."""
    groups = table[["dataset", "unit"]].drop_duplicates().to_numpy()
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(groups))
    n_val = max(1, int(len(groups) * val_fraction))
    val_groups = pd.MultiIndex.from_tuples([tuple(g) for g in groups[idx[:n_val]]])
    row_keys = pd.MultiIndex.from_arrays([table["dataset"], table["unit"]])
    is_val = row_keys.isin(val_groups)
    return table[~is_val].reset_index(drop=True), table[is_val].reset_index(drop=True)


def tune_hyperparameters(
    train_tab: pd.DataFrame,
    feat_cols: list,
    grid: dict,
    subsample_cap: int,
    val_fraction: float,
    seed: int,
) -> tuple[dict, list]:
    tune_train, tune_val = group_train_val_split(train_tab, val_fraction, seed)

    caps = physical_rul_cap(tune_train)
    y_tune_train = capped_rul(tune_train, caps)
    y_tune_val = tune_val["RUL"].astype(float).to_numpy()

    X_tune_train = tune_train[feat_cols].to_numpy(dtype=np.float64)
    X_tune_val = tune_val[feat_cols].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X_tune_train)
    X_tune_train_s = scaler.transform(X_tune_train)
    X_tune_val_s = scaler.transform(X_tune_val)

    if len(X_tune_train_s) > subsample_cap:
        sub_idx = np.random.RandomState(seed).choice(
            len(X_tune_train_s), size=subsample_cap, replace=False
        )
        X_tune_train_s = X_tune_train_s[sub_idx]
        y_tune_train_search = y_tune_train[sub_idx]
    else:
        y_tune_train_search = y_tune_train

    keys = list(grid.keys())
    search_log = []
    for values in itertools.product(*grid.values()):
        kwargs = dict(zip(keys, values), n_gaussians=0, detect_interactions=False)
        t0 = time.perf_counter()
        model = TribbleRegressor(random_state=42, max_samples=2000, **kwargs)
        model.fit(X_tune_train_s, y_tune_train_search)
        pred_val = model.predict(X_tune_val_s)
        val_rmse = float(np.sqrt(mean_squared_error(y_tune_val, pred_val)))
        search_log.append(
            dict(kwargs=kwargs, val_rmse=val_rmse, seconds=time.perf_counter() - t0)
        )
        print(f"    {kwargs} -> val_rmse={val_rmse:.2f}")

    best = min(search_log, key=lambda r: r["val_rmse"])
    return best["kwargs"], search_log


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(
    results: dict,
    file_log: list,
    search_logs: dict,
    total_seconds: float,
    out_path: str,
):
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

        if name in search_logs:
            lines += [
                "",
                f"Hyperparameter search that produced this config (validation "
                f"RMSE on a group-held-out {TUNE_VAL_FRACTION:.0%} of pooled "
                "training units -- the real test set was not used for this "
                "search):",
                "",
                "| tsk_order | top_p | norm_conorm | l2_reg | val RMSE |",
                "|---|---:|---|---:|---:|",
            ]
            for entry in sorted(search_logs[name], key=lambda e: e["val_rmse"]):
                k = entry["kwargs"]
                lines.append(
                    f"| {k['tsk_order']} | {k['top_p']} | {k['norm_conorm']} | "
                    f"{k['l2_reg']} | {entry['val_rmse']:.2f} |"
                )

    lines += ["", f"Total wall time: {total_seconds:.1f}s"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(tune: bool = False):
    t_start = time.perf_counter()
    files = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))
    if not files:
        raise SystemExit(f"No .h5 files found under {H5_DIR}/")

    pooled = {name: dict(train=[], test=[]) for name in BASE_PIPELINES}
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

    # --tune re-runs the search that originally produced the *_full_tuned
    # entries in PIPELINES, purely to reproduce/refresh that discovery --
    # the winning configs are already hardcoded above, so a normal run
    # doesn't pay for a ~10-minute search. Validation is a group-held-out
    # split of the pooled TRAINING data only; the real test set is never
    # touched during the search.
    search_logs = {}
    if tune:
        for name, cfg in BASE_PIPELINES.items():
            print(f"\n{'=' * 78}\nTuning on pooled dataset: {name}\n{'=' * 78}")
            full_train_tab = pd.concat(pooled[name]["train"], ignore_index=True)
            feat_cols = [
                c
                for c in full_train_tab.columns
                if c not in ("dataset", "unit", "cycle", "RUL", "hs")
            ]
            best_kwargs, search_log = tune_hyperparameters(
                full_train_tab,
                feat_cols,
                TUNE_GRID,
                TUNE_SUBSAMPLE_CAP,
                TUNE_VAL_FRACTION,
                TUNE_SEED,
            )
            tuned_name = f"{name}_full_tuned"
            search_logs[tuned_name] = search_log
            print(f"  winner: {best_kwargs}")
            hardcoded = PIPELINES[tuned_name]["tribble_kwargs"]
            if best_kwargs != hardcoded:
                print(
                    f"  NOTE: this differs from the hardcoded {tuned_name} config "
                    f"({hardcoded}) -- update PIPELINES if you want to keep it."
                )

    results = {}
    for name, cfg in PIPELINES.items():
        print(f"\n{'=' * 78}\nPooled pipeline: {name}\n{'=' * 78}")
        pooled_key = TUNED_TO_BASE.get(name, name)
        train_tab = pd.concat(pooled[pooled_key]["train"], ignore_index=True)
        test_tab = pd.concat(pooled[pooled_key]["test"], ignore_index=True)
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
    write_report(results, file_log, search_logs, total_seconds, REPORT_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Re-run the hyperparameter search that produced the "
        "*_full_tuned entries in PIPELINES (~10 minutes extra). Without "
        "this flag, those already-discovered configs are just fit and "
        "evaluated directly, like every other pipeline.",
    )
    args = parser.parse_args()
    main(tune=args.tune)
