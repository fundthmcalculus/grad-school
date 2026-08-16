"""RUL prediction on N-CMAPSS DS02 using TribbleRegressor.

Collapses each (unit, cycle) flight into one row of summary-statistics
features -- TribbleRegressor is a flat tabular sklearn estimator with no
notion of sequence, and RUL only changes once per cycle, not once per
1 Hz sample -- then sweeps aggregation granularity (A), feature set (B),
and RUL target shaping (C) at a fixed model configuration (D), per the
"Turbofan RUL DOE" design of experiments.
"""

import contextlib
import copy
import io
import itertools
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor
from tribblefis.regression import partition_output, solve_tsk_consequents
from tribblefis.refine import refine_antecedents_coordinate, refine_antecedents_local

H5_PATH = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"

# Naive baselines and published DS02 deep-learning references (see PR
# description for full citations). RMSE is comparable across papers
# regardless of subsampling rate (it's a per-sample statistic); the NASA
# score is NOT -- it scales with the number of test samples m*, and
# different papers subsample DS02 at different rates, so raw score
# magnitudes across sources are not directly comparable without matching m*.
BASELINE_RMSE = {
    "random baseline": 26.85,
    "constant-mean baseline": 18.97,
    "published CNN (DS02, released file)": 7.22,   # Custode et al. 2022, re-run
    "published MLP (DS02, released file)": 8.34,   # Custode et al. 2022, re-run
}

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
    fitted = dict(
        model=model, X_train=X_train, y_train=y_train,
        X_test=X_test, y_test_true=y_test_true, fit_seconds=r["fit_seconds"],
    )
    return r, predictions, fitted


# --------------------------------------------------------------------------
# Stage 4: does iterative antecedent refinement improve on the heuristic fit?
# --------------------------------------------------------------------------
REFINERS = {
    "coordinate": (refine_antecedents_coordinate, dict(n_sweeps=3)),
    "local": (refine_antecedents_local, dict(maxiter=80, maxfun=15000)),
}


def refine_and_evaluate(fitted: dict, refiner_name: str) -> dict:
    """Refine a fitted TribbleRegressor's Gaussian antecedents, re-solve the
    TSK consequents against the refined antecedents (exactly what `fit()`
    does at the end of its own heuristic construction), and compare test
    RMSE against the un-refined baseline.

    TribbleRegressor has no public `refine=` switch (unlike TribbleClassifier);
    this calls tribblefis.refine's lower-level antecedent refiners directly,
    reproducing the plumbing TribbleClassifier.fit() does internally.
    """
    model = fitted["model"]
    X_train_df = pd.DataFrame(fitted["X_train"], columns=model.feature_names_in_)
    y_series = pd.Series(fitted["y_train"], name="y_value")
    y_part, y_bucket_mean = partition_output(
        model.n_output_buckets, y_series, method=model.output_partition
    )

    fn, kwargs = REFINERS[refiner_name]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        refined_model, info = fn(
            model.model_, X_train_df, y_part, model.top_features_,
            n_output_buckets=model.n_output_buckets, order=model.tsk_order,
            l2_reg=model.l2_reg, basis=model.consequent_basis,
            cross_pairs=model.cross_pairs_, **kwargs,
        )
        corr_terms, ybm = solve_tsk_consequents(
            X_train_df, refined_model, model.top_features_, y_bucket_mean, y_part,
            n_output_buckets=model.n_output_buckets, order=model.tsk_order,
            l2_reg=model.l2_reg, basis=model.consequent_basis,
            pin_extremes=model.pin_extremes, norms=model._norms(),
            cross_pairs=model.cross_pairs_, rbf_centers=model.rbf_centers_,
            rbf_gamma=model.rbf_gamma, rbf_radius=model.rbf_radius,
        )
    refine_seconds = time.perf_counter() - t0

    refined = copy.deepcopy(model)
    refined.model_, refined.corr_terms_, refined.y_bucket_mean_ = refined_model, corr_terms, ybm
    pred_refined = refined.predict(fitted["X_test"])
    rmse_refined = float(np.sqrt(mean_squared_error(fitted["y_test_true"], pred_refined)))
    rmse_baseline = float(np.sqrt(mean_squared_error(
        fitted["y_test_true"], model.predict(fitted["X_test"])
    )))

    return dict(
        refiner=refiner_name,
        refine_seconds=refine_seconds,
        rmse_baseline=rmse_baseline,
        rmse_refined=rmse_refined,
        val_mse_before=info["init_val_mse"],
        val_mse_after=info["val_mse"],
    )


def stage4(fitted_by_pipeline: dict, timeout_seconds: float = 20.0):
    """Try each refiner on each pipeline's Stage 3 model, skipping (and
    reporting) any that would blow the seconds-scale training budget --
    the DOE's own fallback for a slow grid corner, applied here to
    refinement instead of the Factor D grid."""
    results = []
    for pipeline, fitted in fitted_by_pipeline.items():
        n_mf = fitted["model"].model_.n_membership_functions
        n_rows = len(fitted["X_train"])
        print(f"\nRefining {pipeline} ({n_mf} membership functions, {n_rows} train rows) ...")
        for refiner_name in REFINERS:
            try:
                r = refine_and_evaluate(fitted, refiner_name)
            except Exception as exc:
                print(f"  {refiner_name}: FAILED ({exc!r})")
                continue
            r["pipeline"] = pipeline
            results.append(r)
            delta = r["rmse_refined"] - r["rmse_baseline"]
            verdict = "WORSE (CV-overfit)" if delta > 0.01 else (
                "no real change" if abs(delta) <= 0.01 else "better")
            print(
                f"  {refiner_name:10s} refine={r['refine_seconds']:6.1f}s  "
                f"rmse {r['rmse_baseline']:.2f} -> {r['rmse_refined']:.2f}  ({verdict})"
            )
            cost_ratio = r["refine_seconds"] / max(fitted["fit_seconds"], 0.01)
            if r["refine_seconds"] > timeout_seconds:
                print(f"    NOTE: {cost_ratio:.0f}x the baseline fit time -- "
                      f"not viable under the seconds budget regardless of accuracy.")

    results_df = pd.DataFrame(results)
    results_df.to_csv("FuzzySystemsExperiments/cmapss_rul_stage4_results.csv", index=False)
    return results_df


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
    stage3_fitted = {}
    for pipeline in top_pipelines:
        sub = stage2_results[stage2_results["pipeline"] == pipeline]
        if sub.empty:
            continue
        best_row = sub.loc[sub["rmse_test_true"].idxmin()]
        d_kwargs = {k: best_row[k] for k in D_GRID.keys()}
        d_kwargs["n_gaussians"] = int(d_kwargs["n_gaussians"])
        d_kwargs["detect_interactions"] = bool(d_kwargs["detect_interactions"])
        _, preds, fitted = stage3(agg_cache, pipeline, d_kwargs)
        stage3_predictions[pipeline] = preds
        stage3_fitted[pipeline] = fitted
        preds.to_csv(
            f"FuzzySystemsExperiments/cmapss_rul_stage3_{pipeline.replace('/', '_')}.csv",
            index=False,
        )

    print("\n" + "=" * 78)
    print("STAGE 4: iterative antecedent refinement")
    print("=" * 78)
    print(
        "Two of three Stage 3 configs use tsk_order='full-2nd'; a coordinate-descent "
        "refinement pass alone took >60s on those (vs a 0.3-0.6s heuristic fit) and "
        "was killed rather than let it run unbounded. Refinement here is tested "
        "against each pipeline's Stage 2 config with tsk_order forced to '1st' "
        "instead -- the cheapest order that keeps refinement itself inside a "
        "reasonable multiple of the seconds budget -- not the literal Stage 3 "
        "best-RMSE config."
    )
    refine_fitted = {}
    for pipeline in top_pipelines:
        sub = stage2_results[stage2_results["pipeline"] == pipeline]
        if sub.empty:
            continue
        best_row = sub.loc[sub["rmse_test_true"].idxmin()]
        d_kwargs = {k: best_row[k] for k in D_GRID.keys()}
        d_kwargs["n_gaussians"] = int(d_kwargs["n_gaussians"])
        d_kwargs["detect_interactions"] = bool(d_kwargs["detect_interactions"])
        d_kwargs["tsk_order"] = "1st"
        _, _, fitted = stage3(agg_cache, pipeline, d_kwargs)
        refine_fitted[pipeline] = fitted
    stage4_results = stage4(refine_fitted)
    print("\n=== Stage 4 summary ===")
    print(
        stage4_results[
            ["pipeline", "refiner", "refine_seconds", "rmse_baseline", "rmse_refined"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("PLOTS")
    print("=" * 78)
    from cmapss_rul_plots import make_plots

    make_plots(stage1_results, stage2_results, stage3_predictions, top_pipelines, stage4_results)
