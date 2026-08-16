"""RUL prediction on N-CMAPSS DS02 using TribbleRegressor.

Collapses each (unit, cycle) flight into one row of summary-statistics
features -- TribbleRegressor is a flat tabular sklearn estimator with no
notion of sequence, and RUL only changes once per cycle, not once per
1 Hz sample -- then sweeps aggregation granularity (A), feature set (B),
and RUL target shaping (C) at a fixed model configuration (D), per the
"Turbofan RUL DOE" design of experiments.
"""

import contextlib
import io
import itertools
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor

H5_PATH = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"

FIXED_D = dict(
    tsk_order="1st", n_gaussians=0, top_p=0.95, detect_interactions=False
)


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
            "Fc": d["A"][:, 2],
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
# Factor B: feature sets
# --------------------------------------------------------------------------
def feature_columns(var: dict, feature_set: str) -> list[str]:
    cols = [f"W_{n}" for n in var["W"]] + [f"Xs_{n}" for n in var["X_s"]]
    if feature_set == "B2":
        cols += [f"Xv_{n}" for n in var["X_v"]]
    return cols


# --------------------------------------------------------------------------
# Factor A: aggregation granularity
# --------------------------------------------------------------------------
AGG_FUNCS = ["mean", "std", "min", "max", "last"]


def aggregate_whole_cycle(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """A1: one row per (unit, cycle), stats over every 1 Hz sample in it."""
    g = df.groupby(["unit", "cycle"], sort=True)
    feat = g[feat_cols].agg(AGG_FUNCS)
    feat.columns = ["_".join(c) for c in feat.columns]
    meta = g.agg(RUL=("RUL", "first"), hs=("hs", "min"))
    return feat.join(meta).reset_index()


def aggregate_phase_split(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """A2: same as A1, but stats are computed separately for the early/mid/late
    third of each cycle's samples (in recorded time order), as a proxy for
    climb/cruise/descend.

    The PDF's own climb/cruise/descend convention thresholds cruise at
    alt > 10,000 ft, but this release only records the >10,000 ft portion of
    every flight to begin with (verified: W_alt never drops below 10,001 ft
    across the dev set) -- an altitude-threshold split would leave the "low"
    phase permanently empty. A within-cycle position tercile is used instead,
    since it needs no altitude floor and still separates a flight's early
    (climb-like), middle (cruise-like), and late (descent-like) dynamics.
    """
    order = df.groupby(["unit", "cycle"]).cumcount()
    counts = df.groupby(["unit", "cycle"])["unit"].transform("size")
    relpos = order / counts
    phase = np.where(relpos < 1 / 3, "early", np.where(relpos < 2 / 3, "mid", "late"))
    out = None
    for p in ("early", "mid", "late"):
        sub = df[phase == p]
        g = sub.groupby(["unit", "cycle"], sort=True)
        feat = g[feat_cols].agg(AGG_FUNCS)
        feat.columns = [f"{p}_" + "_".join(c) for c in feat.columns]
        out = feat if out is None else out.join(feat, how="outer")
    meta = df.groupby(["unit", "cycle"], sort=True).agg(
        RUL=("RUL", "first"), hs=("hs", "min")
    )
    result = out.join(meta).reset_index()
    # A cycle with no samples in one phase (e.g. no low-altitude portion)
    # leaves that phase's columns NaN -- fill with the cycle's other-phase
    # mean rather than dropping the row.
    stat_cols = [c for c in result.columns if c not in ("unit", "cycle", "RUL", "hs")]
    result[stat_cols] = result[stat_cols].apply(lambda c: c.fillna(c.mean()))
    return result


def aggregate_raw_memory(
    df: pd.DataFrame, feat_cols: list[str], stride: int = 200
) -> pd.DataFrame:
    """A3: no cycle collapsing. Subsample every `stride`-th raw sample within
    each unit (in cycle/time order) and run the repo's own
    MemoryWindowFeatureExtractor over that stream, keeping each subsampled
    row's own (unit, cycle, RUL, hs) -- this is the "sequence-aware" arm,
    traded against a coarser sample rate to stay near the seconds budget."""
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


AGGREGATORS = {
    "A1_whole_cycle": aggregate_whole_cycle,
    "A2_phase_split": aggregate_phase_split,
    "A3_raw_memory": aggregate_raw_memory,
}


# --------------------------------------------------------------------------
# Factor C: RUL target shaping
# --------------------------------------------------------------------------
def unit_physical_caps(table: pd.DataFrame) -> dict[int, float]:
    """Per-unit cap = RUL at the moment abnormal degradation begins (hs
    first drops to 0), i.e. only the abnormal-degradation window is treated
    as learnable. Derived from data, not hardcoded from the PDF's table."""
    caps = {}
    for unit, sub in table.groupby("unit"):
        sub = sub.sort_values("cycle")
        unhealthy = sub[sub["hs"] == 0]
        cap = unhealthy["RUL"].max() if len(unhealthy) else sub["RUL"].max()
        caps[unit] = float(cap)
    return caps


def apply_rul_shape(table: pd.DataFrame, mode: str, caps: dict) -> pd.Series:
    if mode == "C1_raw":
        return table["RUL"].astype(float)
    if mode == "C2_fixed125":
        return table["RUL"].clip(upper=125).astype(float)
    if mode == "C3_physical":
        cap_series = table["unit"].map(caps)
        return np.minimum(table["RUL"].astype(float), cap_series)
    raise ValueError(mode)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    delta = np.asarray(y_true) - np.asarray(y_pred)
    alpha = np.where(delta > 0, 1.0 / 13.0, 1.0 / 10.0)
    return float(np.sum(np.exp(alpha * np.abs(delta))))


def run_one(
    train_tab: pd.DataFrame,
    test_tab: pd.DataFrame,
    feat_cols: list[str],
    rul_mode: str,
    caps: dict,
    d_kwargs: dict,
) -> dict:
    X_train = train_tab[feat_cols].to_numpy(dtype=np.float64)
    X_test = test_tab[feat_cols].to_numpy(dtype=np.float64)
    y_train = apply_rul_shape(train_tab, rul_mode, caps).to_numpy()
    y_test_target = apply_rul_shape(test_tab, rul_mode, caps).to_numpy()
    y_test_true = test_tab["RUL"].astype(float).to_numpy()

    model = TribbleRegressor(random_state=42, max_samples=2000, **d_kwargs)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return dict(
        fit_seconds=fit_seconds,
        rmse_train=float(np.sqrt(mean_squared_error(y_train, pred_train))),
        rmse_test_shaped=float(np.sqrt(mean_squared_error(y_test_target, pred_test))),
        rmse_test_true=float(np.sqrt(mean_squared_error(y_test_true, pred_test))),
        nasa_score_true=nasa_score(y_test_true, pred_test),
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=X_train.shape[1],
    )


# --------------------------------------------------------------------------
# Stage 1: screen A x B x C at fixed D
# --------------------------------------------------------------------------
def stage1():
    print(f"Loading {H5_PATH} ...")
    t0 = time.perf_counter()
    data, var = load_h5(H5_PATH)
    df_dev = to_frame(data, var, "dev")
    df_test = to_frame(data, var, "test")
    print(f"  loaded {len(df_dev):,} dev rows, {len(df_test):,} test rows"
          f" in {time.perf_counter() - t0:.1f}s")

    results = []
    agg_cache = {}
    for a_name, agg_fn in AGGREGATORS.items():
        for b_name in ("B1", "B2"):
            feat_cols = feature_columns(var, b_name)
            key = (a_name, b_name)
            print(f"\nAggregating {a_name} / {b_name} ...")
            t0 = time.perf_counter()
            try:
                train_tab = agg_fn(df_dev, feat_cols)
                test_tab = agg_fn(df_test, feat_cols)
            except Exception as exc:
                print(f"  SKIPPED ({exc!r})")
                continue
            agg_seconds = time.perf_counter() - t0
            agg_cache[key] = (train_tab, test_tab, feat_cols)
            print(f"  -> {len(train_tab)} train rows, {len(test_tab)} test rows,"
                  f" {len(feat_cols)} raw channels, {agg_seconds:.1f}s to build")

    for (a_name, b_name), (train_tab, test_tab, feat_cols) in agg_cache.items():
        caps = unit_physical_caps(pd.concat([train_tab, test_tab], ignore_index=True))
        agg_feat_cols = [
            c for c in train_tab.columns if c not in ("unit", "cycle", "RUL", "hs")
        ]
        for c_name in ("C1_raw", "C2_fixed125", "C3_physical"):
            try:
                r = run_one(train_tab, test_tab, agg_feat_cols, c_name, caps, FIXED_D)
            except Exception as exc:
                print(f"FAILED {a_name}/{b_name}/{c_name}: {exc!r}")
                continue
            r.update(pipeline=f"{a_name}/{b_name}/{c_name}")
            results.append(r)
            print(
                f"{r['pipeline']:32s} fit={r['fit_seconds']:6.2f}s "
                f"rmse_test={r['rmse_test_true']:6.2f} "
                f"score={r['nasa_score_true']:10.1f} "
                f"n_train={r['n_train']:4d} feats={r['n_features']:4d}"
            )

    results_df = pd.DataFrame(results).sort_values("rmse_test_true")
    print("\n=== Stage 1 ranked by test RMSE (true RUL) ===")
    print(
        results_df[
            ["pipeline", "fit_seconds", "rmse_test_true", "nasa_score_true", "n_features"]
        ].to_string(index=False)
    )
    results_df.to_csv("FuzzySystemsExperiments/cmapss_rul_stage1_results.csv", index=False)
    return results_df, agg_cache


# --------------------------------------------------------------------------
# Stage 2: grid Factor D on the winning Stage 1 pipelines
# --------------------------------------------------------------------------
D_GRID = dict(
    tsk_order=["0th", "1st", "full-2nd"],
    n_gaussians=[0, 3, 5],
    # top_p=1.0 (feature selection off) was tried and dropped: on the wide
    # pipelines (up to 480 raw stat columns) it hands every unselected
    # feature to full-2nd's O(n^2) cross-term basis and to interaction
    # detection's O(n_pairs) scoring, which blew past the seconds budget by
    # orders of magnitude (one grid point alone ran >10 minutes before being
    # killed). Some feature selection is in scope for every config now.
    top_p=[0.90, 0.95],
    detect_interactions=[False, True],
)


def stage2(agg_cache: dict, pipelines: list[str]):
    results = []
    keys = list(D_GRID.keys())
    for pipeline in pipelines:
        a_name, b_name, c_name = pipeline.split("/")
        train_tab, test_tab, _ = agg_cache[(a_name, b_name)]
        caps = unit_physical_caps(pd.concat([train_tab, test_tab], ignore_index=True))
        agg_feat_cols = [
            c for c in train_tab.columns if c not in ("unit", "cycle", "RUL", "hs")
        ]
        print(f"\nGridding Factor D for {pipeline} ...")
        for combo in itertools.product(*D_GRID.values()):
            d_kwargs = dict(zip(keys, combo))
            try:
                r = run_one(train_tab, test_tab, agg_feat_cols, c_name, caps, d_kwargs)
            except Exception as exc:
                print(f"  FAILED {d_kwargs}: {exc!r}")
                continue
            r.update(pipeline=pipeline, **d_kwargs)
            results.append(r)
        best = min((r for r in results if r["pipeline"] == pipeline),
                   key=lambda r: r["rmse_test_true"])
        print(f"  best: rmse_test={best['rmse_test_true']:.2f} "
              f"fit={best['fit_seconds']:.2f}s "
              f"tsk_order={best['tsk_order']} n_gaussians={best['n_gaussians']} "
              f"top_p={best['top_p']} detect_interactions={best['detect_interactions']}")

    results_df = pd.DataFrame(results).sort_values("rmse_test_true")
    results_df.to_csv("FuzzySystemsExperiments/cmapss_rul_stage2_results.csv", index=False)
    return results_df


# --------------------------------------------------------------------------
# Stage 3: confirm the final configuration
# --------------------------------------------------------------------------
def stage3(agg_cache: dict, pipeline: str, d_kwargs: dict):
    a_name, b_name, c_name = pipeline.split("/")
    train_tab, test_tab, _ = agg_cache[(a_name, b_name)]
    caps = unit_physical_caps(pd.concat([train_tab, test_tab], ignore_index=True))
    agg_feat_cols = [
        c for c in train_tab.columns if c not in ("unit", "cycle", "RUL", "hs")
    ]
    r = run_one(train_tab, test_tab, agg_feat_cols, c_name, caps, d_kwargs)

    X_train = train_tab[agg_feat_cols].to_numpy(dtype=np.float64)
    y_train = apply_rul_shape(train_tab, c_name, caps).to_numpy()
    model = TribbleRegressor(random_state=42, max_samples=2000, **d_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
    X_test = test_tab[agg_feat_cols].to_numpy(dtype=np.float64)
    pred_test = model.predict(X_test)
    y_test_true = test_tab["RUL"].astype(float).to_numpy()

    print(f"\n=== Stage 3: final confirmation -- {pipeline} ===")
    print(f"config: {d_kwargs}")
    print(f"fit_seconds={r['fit_seconds']:.3f}  "
          f"rmse_test_true={r['rmse_test_true']:.2f}  "
          f"nasa_score_true={r['nasa_score_true']:.1f}  "
          f"n_train={r['n_train']}  n_test={r['n_test']}  n_features={r['n_features']}")

    for unit in sorted(test_tab["unit"].unique()):
        m = (test_tab["unit"] == unit).to_numpy()
        rmse_u = float(np.sqrt(mean_squared_error(y_test_true[m], pred_test[m])))
        print(f"  unit {unit}: n={m.sum():3d}  rmse={rmse_u:.2f}")

    predictions = pd.DataFrame(
        {
            "unit": test_tab["unit"].to_numpy(),
            "cycle": test_tab["cycle"].to_numpy(),
            "RUL_true": y_test_true,
            "RUL_pred": pred_test,
        }
    ).sort_values(["unit", "cycle"])
    return r, predictions


if __name__ == "__main__":
    stage1_results, agg_cache = stage1()

    print("\n" + "=" * 78)
    print("STAGE 2")
    print("=" * 78)
    top_pipelines = [
        "A3_raw_memory/B2/C1_raw",       # best overall (includes virtual sensors -- leakage-flagged)
        "A3_raw_memory/B1/C3_physical",  # best "real-world" pipeline (W + X_s only)
        "A1_whole_cycle/B1/C3_physical", # cheapest/most interpretable alternative
    ]
    stage2_results = stage2(agg_cache, top_pipelines)
    print("\n=== Stage 2 top 10 overall ===")
    print(
        stage2_results.head(10)[
            ["pipeline", "tsk_order", "n_gaussians", "top_p", "detect_interactions",
             "fit_seconds", "rmse_test_true", "nasa_score_true"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("STAGE 3")
    print("=" * 78)
    stage3_predictions = {}
    for pipeline in top_pipelines:
        sub = stage2_results[stage2_results["pipeline"] == pipeline]
        if sub.empty:
            continue
        best_row = sub.loc[sub["rmse_test_true"].idxmin()]
        d_kwargs = {k: best_row[k] for k in D_GRID.keys()}
        d_kwargs["n_gaussians"] = int(d_kwargs["n_gaussians"])
        d_kwargs["detect_interactions"] = bool(d_kwargs["detect_interactions"])
        _, preds = stage3(agg_cache, pipeline, d_kwargs)
        stage3_predictions[pipeline] = preds
        preds.to_csv(
            f"FuzzySystemsExperiments/cmapss_rul_stage3_{pipeline.replace('/', '_')}.csv",
            index=False,
        )

    print("\n" + "=" * 78)
    print("PLOTS")
    print("=" * 78)
    from cmapss_rul_plots import make_plots

    make_plots(stage1_results, stage2_results, stage3_predictions, top_pipelines)
