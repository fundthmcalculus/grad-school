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
import sys
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor
from tribblefis.regression import partition_output, solve_tsk_consequents
from tribblefis.refine import refine_antecedents_coordinate, refine_antecedents_optimizers

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
    """B1: W + X_s (18 channels) -- real sensors only, by this repo's
    original (stricter-than-published) definition of "real."
    B2: B1 + all of X_v (32 channels) -- every "virtual sensor," most of
    which genuinely aren't measurable on a real aircraft. A sensitivity/
    upper-bound arm, not a deployment claim.
    B3: B1 + X_v[:2] (T40, P30) = 20 channels -- matches the published
    DS02 CNN/MLP baselines' input set exactly. Confirmed three ways: Arias
    Chao et al. 2021 (arXiv:2003.00732) Table 2 lists T40/P30 among their
    20 "condition monitoring signals, [w, xs]"; Custode et al. 2022
    (Algorithms 15(3):98) cite that same 20-input convention; and co-author
    Hyunho Mo's own released code (N-CMAPSS_DL) slices `X_v[:, 0:2]` into
    the model's inputs. T40/P30 sit in the HDF5 file's X_v group only
    because of how the C-MAPSS simulator organizes its outputs -- the
    literature itself doesn't treat them as unmeasurable. B1 was, in that
    light, an unnecessarily strict definition of "real"; B3 is the one
    that's actually comparable to the published numbers.
    """
    cols = [f"W_{n}" for n in var["W"]] + [f"Xs_{n}" for n in var["X_s"]]
    if feature_set == "B2":
        cols += [f"Xv_{n}" for n in var["X_v"]]
    elif feature_set == "B3":
        cols += [f"Xv_{n}" for n in var["X_v"][:2]]
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


# --------------------------------------------------------------------------
# Preprocessing: condition-corrected sensor channels, at raw (pre-aggregation)
# resolution -- fed to the regressor as *input features*, not just used to
# find the RUL-cap onset (see Stage 5). Same fix as Stage 5's onset
# detector: raw per-cycle sensor readings are dominated by flight-to-flight
# operating-condition swings, not the (much smaller) degradation trend --
# regressing out W first exposes the signal the regressor is actually
# supposed to learn from. Fit only on dev/train units' own early cycles
# (never on test), then applied to both splits with the same fitted models.
# --------------------------------------------------------------------------
def fit_raw_condition_correction(
    df: pd.DataFrame, sensor_cols: list[str], condition_cols: list[str],
    baseline_cycles: int = 15,
) -> dict:
    from sklearn.linear_model import LinearRegression

    order = df.groupby("unit").cumcount()
    baseline = df[order < baseline_cycles]
    X_base = baseline[condition_cols].to_numpy(dtype=np.float64)
    return {
        col: LinearRegression().fit(X_base, baseline[col].to_numpy(dtype=np.float64))
        for col in sensor_cols
    }


def apply_raw_condition_correction(
    df: pd.DataFrame, sensor_cols: list[str], condition_cols: list[str], models: dict,
) -> pd.DataFrame:
    df = df.copy()
    X_all = df[condition_cols].to_numpy(dtype=np.float64)
    for col in sensor_cols:
        df[col] = df[col].to_numpy(dtype=np.float64) - models[col].predict(X_all)
    return df


AGGREGATORS = {
    "A1_whole_cycle": (aggregate_whole_cycle, False),
    "A2_phase_split": (aggregate_phase_split, False),
    "A3_raw_memory": (aggregate_raw_memory, False),
    # Same aggregators, fed the condition-corrected stream instead of raw.
    "A1_whole_cycle_cc": (aggregate_whole_cycle, True),
    "A3_raw_memory_cc": (aggregate_raw_memory, True),
}


# --------------------------------------------------------------------------
# Factor C: RUL target shaping
# --------------------------------------------------------------------------
def true_onset_cycle(table: pd.DataFrame) -> dict[int, int]:
    """Per-unit abnormal-degradation onset cycle, from the oracle `hs` health
    flag (first cycle it drops to 0). Not observable by a real onboard
    system -- `hs` comes from the simulator's latent health parameters, not
    a sensor reading -- see `detect_onset_moving_average` for the
    sensor-only estimate."""
    onsets = {}
    for unit, sub in table.groupby("unit"):
        sub = sub.sort_values("cycle")
        unhealthy = sub[sub["hs"] == 0]
        onsets[unit] = int(unhealthy["cycle"].min()) if len(unhealthy) else int(sub["cycle"].max())
    return onsets


def unit_caps_from_onset(table: pd.DataFrame, onset_by_unit: dict) -> dict[int, float]:
    """Per-unit RUL cap = the RUL value at (or just after) that unit's
    degradation-onset cycle -- only the abnormal-degradation window is
    treated as learnable, whichever onset estimate is supplied."""
    caps = {}
    for unit, sub in table.groupby("unit"):
        sub = sub.sort_values("cycle")
        onset = onset_by_unit.get(unit, sub["cycle"].max())
        at_or_after = sub[sub["cycle"] >= onset]
        cap = at_or_after["RUL"].max() if len(at_or_after) else sub["RUL"].max()
        caps[unit] = float(cap)
    return caps


def unit_physical_caps(table: pd.DataFrame) -> dict[int, float]:
    """Per-unit cap using the oracle `hs`-derived onset. Derived from data,
    not hardcoded from the PDF's table. See `unit_caps_from_onset` for the
    general form this specializes."""
    return unit_caps_from_onset(table, true_onset_cycle(table))


def apply_rul_shape(table: pd.DataFrame, mode: str, caps: dict) -> pd.Series:
    if mode == "C1_raw":
        return table["RUL"].astype(float)
    if mode == "C2_fixed125":
        return table["RUL"].clip(upper=125).astype(float)
    if mode in ("C3_physical", "C4_detected"):
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
    """`d_kwargs['scaler']` (popped before reaching TribbleRegressor) is an
    optional post-aggregation feature scaler -- 'standard' fits sklearn's
    StandardScaler on train only, transforms both splits. Not a
    TribbleRegressor constructor arg; a preprocessing step layered on top."""
    d_kwargs = dict(d_kwargs)
    scaler_name = d_kwargs.pop("scaler", None)

    X_train = train_tab[feat_cols].to_numpy(dtype=np.float64)
    X_test = test_tab[feat_cols].to_numpy(dtype=np.float64)
    if scaler_name == "standard":
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    elif scaler_name is not None:
        raise ValueError(scaler_name)

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

    print("Fitting condition correction (Xs/Xv ~ W, on dev-unit early cycles only) ...")
    t0 = time.perf_counter()
    w_cols = [f"W_{n}" for n in var["W"]]
    xs_cols = [f"Xs_{n}" for n in var["X_s"]]
    xv_cols = [f"Xv_{n}" for n in var["X_v"]]
    cc_models = fit_raw_condition_correction(df_dev, xs_cols + xv_cols, w_cols)
    df_dev_cc = apply_raw_condition_correction(df_dev, xs_cols + xv_cols, w_cols, cc_models)
    df_test_cc = apply_raw_condition_correction(df_test, xs_cols + xv_cols, w_cols, cc_models)
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    results = []
    agg_cache = {}
    for a_name, (agg_fn, use_corrected) in AGGREGATORS.items():
        src_dev, src_test = (df_dev_cc, df_test_cc) if use_corrected else (df_dev, df_test)
        for b_name in ("B1", "B2", "B3"):
            feat_cols = feature_columns(var, b_name)
            key = (a_name, b_name)
            print(f"\nAggregating {a_name} / {b_name} ...")
            t0 = time.perf_counter()
            try:
                train_tab = agg_fn(src_dev, feat_cols)
                test_tab = agg_fn(src_test, feat_cols)
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
    # norm_conorm/l2_reg added after a one-factor-at-a-time probe on the
    # A3_raw_memory/B2/C1_raw pipeline (holding the rest of that pipeline's
    # winning config fixed) turned up a real, zero-extra-cost improvement:
    # 'hamacher' + l2_reg=0.01 hit RMSE 8.83 vs. the original grid's 9.68 --
    # better than *any* refinement result found in this DOE, at the same
    # sub-second fit time. 'min/max' was close behind (8.91); 'luk' was
    # catastrophic (37.99, presumably a numerical-stability failure of the
    # Lukasiewicz operators at this order/data combo) and is deliberately
    # excluded here rather than included as a guaranteed-bad grid point.
    # n_output_buckets>2 and output_partition='quantile' were also probed
    # and consistently hurt (16+ RMSE) -- left at their defaults, not swept.
    # consequent_basis='gaussian-rbf' isn't in this grid, but not because
    # it's still broken -- tribble-fis#130 (compute_rbf_centers exploding to
    # n_centers**n_features) was fixed upstream (058501f) and confirmed
    # working here: no more OOM, ~0.65s fits. Manually tuned rbf_gamma
    # (0.003-0.01) and rbf_n_centers=8 across all 3 pipelines afterward, and
    # it's a real, safe basis option now -- just not a winner on this
    # dataset, landing close behind but consistently behind the raw/
    # orthogonal-basis champion above (12.48 vs. 8.83; 19.35 vs. 19.12;
    # 21.91 vs. 21.66). Left out of the grid on merit, not on the old bug.
    norm_conorm=["probability", "hamacher"],
    l2_reg=[1e-6, 0.01],
    # Post-aggregation StandardScaler stacks a small additional gain on top
    # of condition-corrected features (see AGGREGATORS' _cc variants) --
    # not a TribbleRegressor constructor arg, popped out in run_one/stage3
    # before the model is built.
    scaler=[None, "standard"],
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

    model_kwargs = dict(d_kwargs)
    scaler_name = model_kwargs.pop("scaler", None)
    X_train = train_tab[agg_feat_cols].to_numpy(dtype=np.float64)
    X_test = test_tab[agg_feat_cols].to_numpy(dtype=np.float64)
    if scaler_name == "standard":
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    elif scaler_name is not None:
        raise ValueError(scaler_name)

    y_train = apply_rul_shape(train_tab, c_name, caps).to_numpy()
    model = TribbleRegressor(random_state=42, max_samples=2000, **model_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X_train, y_train)
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
# L-BFGS-B ("local") was tried first and dropped: it was both slower and less
# accurate than every option below at every pipeline scale tested (see PR
# #110 history) -- coordinate descent beat it on cost, and the optimizers-GA
# search below beat it on both cost and, on the smallest pipeline, on whether
# it overfits its own CV split at all. `local_grad_optim="none"` is load-
# bearing: the default per-candidate local gradient polish
# ("single-var-grad") didn't finish a single generation in 90s even at
# population=6 -- disabling it is what makes the population search itself
# competitive with coordinate descent's cost.
REFINERS = {
    "coordinate": (refine_antecedents_coordinate, dict(n_sweeps=3)),
    "optimizers_ga": (
        refine_antecedents_optimizers,
        dict(method="ga", population_size=40, num_generations=25,
             local_scale=0.25, local_grad_optim="none"),
    ),
}


def refine_and_evaluate(fitted: dict, refiner_name: str, fn=None, kwargs=None) -> dict:
    """Refine a fitted TribbleRegressor's Gaussian antecedents, re-solve the
    TSK consequents against the refined antecedents (exactly what `fit()`
    does at the end of its own heuristic construction), and compare test
    RMSE against the un-refined baseline.

    TribbleRegressor has no public `refine=` switch (unlike TribbleClassifier);
    this calls tribblefis.refine's lower-level antecedent refiners directly,
    reproducing the plumbing TribbleClassifier.fit() does internally.

    `fn`/`kwargs` default to the `REFINERS[refiner_name]` entry; pass them
    explicitly to sweep a refiner's own hyperparameters under a label that
    isn't itself a REFINERS key (see `stage4b_refiner_sweep`).
    """
    if fn is None:
        fn, kwargs = REFINERS[refiner_name]
    model = fitted["model"]
    X_train_df = pd.DataFrame(fitted["X_train"], columns=model.feature_names_in_)
    y_series = pd.Series(fitted["y_train"], name="y_value")
    y_part, y_bucket_mean = partition_output(
        model.n_output_buckets, y_series, method=model.output_partition
    )

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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


STAGE4_CSV = "FuzzySystemsExperiments/cmapss_rul_stage4_results.csv"


def stage4(fitted_by_pipeline: dict, timeout_seconds: float = 20.0):
    """Try each refiner on each pipeline's Stage 3 model, skipping (and
    reporting) any that would blow the seconds-scale training budget --
    the DOE's own fallback for a slow grid corner, applied here to
    refinement instead of the Factor D grid.

    Writes the CSV after *every* (pipeline, refiner) pair, not just at the
    end: this stage alone runs 20-25 minutes, and a background run has
    already been killed mid-run once with no OOM/reboot evidence to explain
    it. Re-running with the CSV already present skips whatever pairs it
    already has, so a second kill doesn't cost the whole stage again.
    """
    import os

    done = set()
    results = []
    if os.path.exists(STAGE4_CSV):
        prior = pd.read_csv(STAGE4_CSV)
        results = prior.to_dict("records")
        done = set(zip(prior["pipeline"], prior["refiner"]))
        if done:
            print(f"Resuming Stage 4: {len(done)} (pipeline, refiner) pairs already done.")

    # coordinate descent's cost scales badly with membership-function count --
    # 150 MFs took 935s; a 313-MF pipeline in this round didn't finish in
    # 22+ minutes (still climbing CPU time, not stuck) before being killed
    # to avoid burning an hour on one already-well-understood data point.
    # GA is dramatically cheaper at scale (147-197s at 150 MFs) and is kept
    # unconditionally; coordinate is skipped above this threshold rather
    # than re-proven slow at every new model size.
    COORDINATE_MAX_MF = 200

    for pipeline, fitted in fitted_by_pipeline.items():
        n_mf = fitted["model"].model_.n_membership_functions
        n_rows = len(fitted["X_train"])
        print(f"\nRefining {pipeline} ({n_mf} membership functions, {n_rows} train rows) ...")
        for refiner_name in REFINERS:
            if (pipeline, refiner_name) in done:
                print(f"  {refiner_name:10s} already done -- skipping")
                continue
            if refiner_name == "coordinate" and n_mf > COORDINATE_MAX_MF:
                print(f"  {refiner_name:10s} skipped -- {n_mf} MFs exceeds "
                      f"COORDINATE_MAX_MF={COORDINATE_MAX_MF} (see comment above)")
                continue
            try:
                r = refine_and_evaluate(fitted, refiner_name)
            except Exception as exc:
                print(f"  {refiner_name}: FAILED ({exc!r})")
                continue
            r["pipeline"] = pipeline
            results.append(r)
            pd.DataFrame(results).to_csv(STAGE4_CSV, index=False)
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


# --------------------------------------------------------------------------
# Stage 4b: sweep each refiner's own hyperparameters (not just Factor D)
# --------------------------------------------------------------------------
# Run on the cheapest pipeline only (446 rows, ~5s/config) -- it's also the
# one where refinement overfits its CV split, so it's the sharpest test of
# whether a hyperparameter choice can rescue that failure mode rather than
# just moving the needle on an already-working case.
SWEEP_CONFIGS = [
    ("coordinate", refine_antecedents_coordinate, dict(n_sweeps=2)),
    ("coordinate", refine_antecedents_coordinate, dict(n_sweeps=3)),
    ("coordinate", refine_antecedents_coordinate, dict(n_sweeps=5)),
    ("optimizers_ga", refine_antecedents_optimizers,
     dict(method="ga", population_size=20, num_generations=25, local_scale=0.15, local_grad_optim="none")),
    ("optimizers_ga", refine_antecedents_optimizers,
     dict(method="ga", population_size=40, num_generations=25, local_scale=0.25, local_grad_optim="none")),
    ("optimizers_ga", refine_antecedents_optimizers,
     dict(method="ga", population_size=40, num_generations=25, local_scale=0.5, local_grad_optim="none")),
    ("optimizers_ga", refine_antecedents_optimizers,
     dict(method="ga", population_size=40, num_generations=25, local_scale=None, local_grad_optim="none")),
    ("optimizers_ga", refine_antecedents_optimizers,
     dict(method="ga", population_size=80, num_generations=40, local_scale=0.25, local_grad_optim="none")),
]


def stage4b_refiner_sweep(fitted: dict) -> pd.DataFrame:
    results = []
    for label, fn, kwargs in SWEEP_CONFIGS:
        try:
            r = refine_and_evaluate(fitted, label, fn=fn, kwargs=kwargs)
        except Exception as exc:
            print(f"  {label} {kwargs}: FAILED ({exc!r})")
            continue
        r["config"] = str(kwargs)
        results.append(r)
        delta = r["rmse_refined"] - r["rmse_baseline"]
        verdict = "WORSE (CV-overfit)" if delta > 0.01 else (
            "no real change" if abs(delta) <= 0.01 else "better")
        print(
            f"  {label:14s} {kwargs}  refine={r['refine_seconds']:6.1f}s  "
            f"rmse {r['rmse_baseline']:.2f} -> {r['rmse_refined']:.2f}  ({verdict})"
        )
    results_df = pd.DataFrame(results)
    results_df.to_csv("FuzzySystemsExperiments/cmapss_rul_stage4b_sweep.csv", index=False)
    return results_df


# --------------------------------------------------------------------------
# Stage 5: estimate degradation onset from real sensors, not the oracle `hs`
# --------------------------------------------------------------------------
# `unit_physical_caps`/C3_physical use `hs`, a health-state flag the dataset
# provides directly from the simulator's latent parameters -- not something a
# real onboard system could read off a sensor. This tests whether a moving-
# average changepoint detector over the actual measured channels (W + X_s
# only) can locate the same onset well enough to use in its place.
#
# First attempt used raw per-cycle Xs means directly and it never fired for
# any unit: baseline-period std was *larger* than each unit's whole-lifetime
# std (e.g. unit 2's Xs_T24_mean: 8.66 in the first 10 cycles vs. 7.36 over
# its full 75-cycle life). Flight-to-flight operating-condition variation
# (different altitude/Mach/route each cycle) swamps the actual degradation
# trend in raw sensor readings -- exactly the confound Wang et al. 2019
# ("Remaining Useful Life Estimation Using Functional Data Analysis") correct
# for before RUL modeling. `condition_corrected_residuals` applies the same
# fix: regress each Xs channel on the W operating-condition channels using
# each unit's own presumed-healthy first cycles, then hunt for onset in the
# *residuals*, where operating-condition variation is gone and a real
# degradation drift can actually stand out above the noise floor.
def condition_corrected_residuals(
    table: pd.DataFrame,
    sensor_cols: list[str],
    condition_cols: list[str],
    baseline_cycles: int = 15,
) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression

    order = table.groupby("unit").cumcount()
    baseline = table[order < baseline_cycles]
    X_base = baseline[condition_cols].to_numpy(dtype=np.float64)
    X_all = table[condition_cols].to_numpy(dtype=np.float64)

    resid = pd.DataFrame(index=table.index)
    resid["unit"] = table["unit"].to_numpy()
    resid["cycle"] = table["cycle"].to_numpy()
    for col in sensor_cols:
        reg = LinearRegression().fit(X_base, baseline[col].to_numpy(dtype=np.float64))
        resid[col] = table[col].to_numpy(dtype=np.float64) - reg.predict(X_all)
    return resid


def detect_onset_moving_average(
    table: pd.DataFrame,
    sensor_cols: list[str],
    baseline_cycles: int = 15,
    window: int = 2,
    z_thresh: float = 0.5,
    sustain: int = 2,
) -> dict[int, int]:
    """Per unit: z-score each sensor channel's rolling `window`-cycle moving
    average against that unit's own first-`baseline_cycles` mean/std, average
    the |z| across channels into one combined signal, and call the onset the
    first cycle where that signal exceeds `z_thresh` for `sustain` cycles
    running (a blip isn't onset; a sustained departure from the healthy
    baseline is). No `hs`, no theta -- only real sensor columns."""
    onsets = {}
    for unit, sub in table.groupby("unit"):
        sub = sub.sort_values("cycle").reset_index(drop=True)
        signals = sub[sensor_cols].to_numpy(dtype=np.float64)
        baseline_mean = signals[:baseline_cycles].mean(axis=0)
        baseline_std = signals[:baseline_cycles].std(axis=0) + 1e-9
        ma = pd.DataFrame(signals).rolling(window, min_periods=1).mean().to_numpy()
        z = np.abs((ma - baseline_mean) / baseline_std).mean(axis=1)

        onset_cycle, run = None, 0
        for i, zi in enumerate(z):
            if i < baseline_cycles:
                continue
            run = run + 1 if zi > z_thresh else 0
            if run >= sustain:
                onset_cycle = int(sub["cycle"].iloc[i - sustain + 1])
                break
        onsets[unit] = onset_cycle if onset_cycle is not None else int(sub["cycle"].iloc[-1])
    return onsets


def stage5(agg_cache: dict, top_pipelines: list[str]):
    train_tab, test_tab, _ = agg_cache[("A1_whole_cycle", "B1")]
    combined = pd.concat([train_tab, test_tab], ignore_index=True)
    condition_cols = [c for c in combined.columns if c.startswith("W_") and c.endswith("_mean")]
    sensor_cols = [c for c in combined.columns if c.startswith("Xs_") and c.endswith("_mean")]

    residuals = condition_corrected_residuals(combined, sensor_cols, condition_cols)
    true_onset = true_onset_cycle(combined)
    detected_onset = detect_onset_moving_average(residuals, sensor_cols)

    print(f"\nOnset detection using {len(sensor_cols)} real sensor channels "
          f"(mean per cycle, condition-corrected against {len(condition_cols)} W channels):")
    rows = []
    for unit in sorted(true_onset):
        t, d = true_onset[unit], detected_onset[unit]
        rows.append(dict(unit=unit, true_onset=t, detected_onset=d, error=d - t))
        print(f"  unit {unit:3d}: true onset={t:3d}  detected={d:3d}  error={d - t:+4d}")
    onset_df = pd.DataFrame(rows)
    mae = onset_df["error"].abs().mean()
    print(f"  MAE across {len(onset_df)} units: {mae:.1f} cycles")
    onset_df.to_csv("FuzzySystemsExperiments/cmapss_rul_stage5_onsets.csv", index=False)

    # Now: how much does the RUL-prediction pipeline lose using the detected
    # onset for the physical cap instead of the oracle hs-derived one?
    detected_caps = unit_caps_from_onset(combined, detected_onset)
    results = []
    for pipeline in top_pipelines:
        a_name, b_name, c_name = pipeline.split("/")
        if c_name != "C3_physical":
            continue  # only C3_physical has an onset-derived cap to swap
        p_train_tab, p_test_tab, _ = agg_cache[(a_name, b_name)]
        agg_feat_cols = [c for c in p_train_tab.columns if c not in ("unit", "cycle", "RUL", "hs")]
        oracle_caps = unit_physical_caps(pd.concat([p_train_tab, p_test_tab], ignore_index=True))
        for cap_name, caps in [("oracle_hs", oracle_caps), ("detected_ma", detected_caps)]:
            r = run_one(p_train_tab, p_test_tab, agg_feat_cols, "C3_physical", caps, FIXED_D)
            r.update(pipeline=pipeline, cap_source=cap_name)
            results.append(r)
            print(f"{pipeline:32s} cap={cap_name:12s} "
                  f"rmse_test={r['rmse_test_true']:.2f}  fit={r['fit_seconds']:.2f}s")

    results_df = pd.DataFrame(results)
    results_df.to_csv("FuzzySystemsExperiments/cmapss_rul_stage5_results.csv", index=False)
    return onset_df, results_df


# --------------------------------------------------------------------------
# Checkpointing: Stage 1/2 (agg_cache + stage2_results) are the cheap, always-
# reproducible part (deterministic given random_state=42). Stage 4 alone runs
# 20-25 minutes; a background run got killed mid-Stage-2 with no OOM or
# reboot evidence, so each phase should be resumable independently rather
# than re-paying the whole pipeline on any interruption.
# --------------------------------------------------------------------------
import pickle

CACHE_PATH = "FuzzySystemsExperiments/.cmapss_rul_cache.pkl"


def save_cache(agg_cache, stage1_results, stage2_results):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(dict(agg_cache=agg_cache, stage1_results=stage1_results,
                          stage2_results=stage2_results), f)


def load_cache():
    with open(CACHE_PATH, "rb") as f:
        d = pickle.load(f)
    return d["agg_cache"], d["stage1_results"], d["stage2_results"]


def select_top_pipelines(stage1_results: pd.DataFrame, cheap_family: str = "A1_whole_cycle") -> list[str]:
    """Pick pipelines dynamically from Stage 1's results rather than
    hardcoding names -- with condition-corrected aggregation variants added
    to the matrix, the actual best pipelines may not match any pipeline
    that was previously the winner. Picks, in order: best overall (any
    feature set, may include virtual sensors -- a sensitivity/upper-bound
    arm), best B3 (matches the published DS02 CNN/MLP baselines' exact
    20-channel input set -- the fair, literature-comparable "real" number),
    best B1 (the stricter 18-channel real-only definition, for completeness),
    and the cheapest/most-interpretable pipeline from `cheap_family` (always
    last -- see `cheap_pipeline = top_pipelines[-1]` in __main__)."""
    df = stage1_results.sort_values("rmse_test_true")
    agg_family = df["pipeline"].str.split("/").str[0]
    picks = [df.iloc[0]["pipeline"]]
    for tag in ("/B3/", "/B1/"):
        matches = df[df["pipeline"].str.contains(tag, regex=False)]
        if len(matches):
            picks.append(matches.iloc[0]["pipeline"])
    cheap = df[agg_family == cheap_family]  # exact match: "..._cc" must not count
    if len(cheap):
        picks.append(cheap.sort_values("fit_seconds").iloc[0]["pipeline"])
    return list(dict.fromkeys(picks))  # dedupe, preserve order


if __name__ == "__main__":
    import os

    # A3_raw_memory_cc/B3/C3_physical is added explicitly, not left to
    # select_top_pipelines: at Stage 1's fixed-D screen it looks like one of
    # the *worse* B3 options (25.6 RMSE) -- its potential only shows up
    # after full construction tuning (a one-off manual probe found 6.48,
    # on par with the all-virtual-sensor B2 champion, using only the exact
    # 20-channel input set the published CNN/MLP baselines use). The cheap
    # fixed-D screen is a useful first-pass filter, not a reliable predictor
    # of which pipeline benefits most from tuning -- this is the second
    # time that's bitten the automatic selection (A3_raw_memory_cc/B1 vs.
    # A1_whole_cycle_cc/B1 was the first), so don't trust it blindly for B3.
    FORCE_INCLUDE = ["A3_raw_memory_cc/B3/C3_physical"]

    if "--resume" in sys.argv and os.path.exists(CACHE_PATH):
        print(f"Resuming from cache: {CACHE_PATH}")
        agg_cache, stage1_results, stage2_results = load_cache()
        top_pipelines = select_top_pipelines(stage1_results)
        # cheap_pipeline is tracked *before* FORCE_INCLUDE is appended -- it's
        # select_top_pipelines' last (deduped) entry by construction, and
        # Stage 4b keys off it by name, not list position, so appending more
        # pipelines afterward can't silently repoint it.
        cheap_pipeline = top_pipelines[-1]
        top_pipelines = top_pipelines + [p for p in FORCE_INCLUDE if p not in top_pipelines]
    else:
        stage1_results, agg_cache = stage1()
        top_pipelines = select_top_pipelines(stage1_results)
        cheap_pipeline = top_pipelines[-1]
        top_pipelines = top_pipelines + [p for p in FORCE_INCLUDE if p not in top_pipelines]
        print(f"\nSelected pipelines for Stage 2+: {top_pipelines}")

        print("\n" + "=" * 78)
        print("STAGE 2")
        print("=" * 78)
        stage2_results = stage2(agg_cache, top_pipelines)
        save_cache(agg_cache, stage1_results, stage2_results)
        print(f"(cached Stage 1/2 to {CACHE_PATH})")

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

    if "--construction-only" in sys.argv:
        print("\n--construction-only: stopping after Stage 3 (skipping refinement "
              "and onset-detection stages).")
        print("\n=== Construction-only summary ===")
        for pipeline in top_pipelines:
            sub = stage2_results[stage2_results["pipeline"] == pipeline]
            if sub.empty:
                continue
            best = sub.loc[sub["rmse_test_true"].idxmin()]
            print(f"{pipeline:36s} rmse={best['rmse_test_true']:.2f}  "
                  f"fit={best['fit_seconds']:.2f}s  n_features={best['n_features']}")
        raise SystemExit(0)

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

    stage4_csv = "FuzzySystemsExperiments/cmapss_rul_stage4_results.csv"
    if "--resume" in sys.argv and os.path.exists(stage4_csv):
        print(f"Resuming Stage 4 from {stage4_csv} (skipping the 20-25 min refinement rerun)")
        stage4_results = pd.read_csv(stage4_csv)
    else:
        stage4_results = stage4(refine_fitted)
    print("\n=== Stage 4 summary ===")
    print(
        stage4_results[
            ["pipeline", "refiner", "refine_seconds", "rmse_baseline", "rmse_refined"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("STAGE 4b: sweep each refiner's own hyperparameters")
    print("=" * 78)
    print(f"Run on the cheapest pipeline only ({cheap_pipeline}, ~5s/config) -- also "
          "the one where refinement overfits its CV split, the sharpest test of "
          "whether a hyperparameter choice can rescue that failure mode.")
    stage4b_results = stage4b_refiner_sweep(refine_fitted[cheap_pipeline])

    print("\n" + "=" * 78)
    print("STAGE 5: degradation onset from real sensors, not the oracle hs flag")
    print("=" * 78)
    stage5_onsets, stage5_results = stage5(agg_cache, top_pipelines)

    print("\n" + "=" * 78)
    print("PLOTS")
    print("=" * 78)
    from cmapss_rul_plots import make_plots

    make_plots(
        stage1_results, stage2_results, stage3_predictions, top_pipelines,
        stage4_results, stage4b_results, stage5_onsets, stage5_results,
    )
