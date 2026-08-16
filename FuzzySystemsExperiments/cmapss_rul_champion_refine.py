"""Targeted coordinate-descent refinement (+ optional smoothing) on the two
headline pipelines.

GA refinement (`refine_antecedents_optimizers`) was tried first but proved
far too expensive at 'full-2nd' order (see git history of this file) --
every GA candidate perturbs ALL antecedents at once, forcing a full
firing-degree recompute across all membership functions per fitness
evaluation. Coordinate descent (`refine_antecedents_coordinate`) instead
optimizes one membership function's (mu, sigma) at a time with everything
else held fixed, so each sub-problem is cheap regardless of tsk_order --
this is also this library's own documented "recommended default" refiner.

Also prints a model-complexity table: membership functions, this library's
notion of "rules" (which needs a caveat -- see below), and total tunable
parameters, for both champions.
"""

import contextlib
import io
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from cmapss_rul import (
    load_h5,
    to_frame,
    feature_columns,
    H5_PATH,
    fit_raw_condition_correction,
    apply_raw_condition_correction,
    aggregate_whole_cycle,
    aggregate_raw_memory,
    unit_physical_caps,
    apply_rul_shape,
)
from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.refine import refine_antecedents_coordinate
from tribblefis.regression import partition_output, solve_tsk_consequents

CONFIGS = {
    # refine_kwargs is per-pipeline, not global. B3 has 125 membership
    # functions at 'full-2nd' order -- prior DOE timing at a similar MF
    # count (150 MF, full-2nd) showed coordinate descent taking ~935s for
    # the library's default n_sweeps=3, so n_sweeps is cut to 2 here to keep
    # this a bounded, targeted operation (~10 min) rather than open-ended.
    # B1 has only 92 MF at '1st' order (cheap regardless), so it keeps the
    # library default n_sweeps=3.
    "A3_raw_memory_cc/B3/C3_physical": dict(
        agg=aggregate_raw_memory,
        fs="B3",
        kw=dict(
            tsk_order="full-2nd",
            n_gaussians=0,
            top_p=0.95,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
        refine_kwargs=dict(n_sweeps=2, block=2, sub_maxfun=25),
    ),
    "A1_whole_cycle_cc/B1/C3_physical": dict(
        agg=aggregate_whole_cycle,
        fs="B1",
        kw=dict(
            tsk_order="1st",
            n_gaussians=0,
            top_p=0.9,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
        refine_kwargs=dict(n_sweeps=3, block=2, sub_maxfun=25),
    ),
}


def smooth_predictions(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Rolling-mean smoothing of RUL_pred, per unit, in cycle order. A
    secondary operation on top of refinement -- the raw-memory pipeline
    predicts once per raw (subsampled) sample, not once per cycle, so
    consecutive predictions within a cycle are noisy around a slowly-
    varying true signal; averaging them out should help."""
    df = df.sort_values(["unit", "cycle"]).copy()
    df["RUL_pred_smoothed"] = df.groupby("unit")["RUL_pred"].transform(
        lambda s: s.rolling(window, center=True, min_periods=1).mean()
    )
    return df


def main():
    print(f"Loading {H5_PATH} ...")
    data, var = load_h5(H5_PATH)
    df_dev = to_frame(data, var, "dev")
    df_test = to_frame(data, var, "test")
    w_cols = [f"W_{n}" for n in var["W"]]
    xs_cols = [f"Xs_{n}" for n in var["X_s"]]
    xv_cols = [f"Xv_{n}" for n in var["X_v"]]
    cc_models = fit_raw_condition_correction(df_dev, xs_cols + xv_cols, w_cols)
    df_dev_cc = apply_raw_condition_correction(
        df_dev, xs_cols + xv_cols, w_cols, cc_models
    )
    df_test_cc = apply_raw_condition_correction(
        df_test, xs_cols + xv_cols, w_cols, cc_models
    )

    complexity_rows = []
    trajectory_predictions = {}

    for name, cfg in CONFIGS.items():
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        feat_cols = feature_columns(var, cfg["fs"])
        train_tab = cfg["agg"](df_dev_cc, feat_cols)
        test_tab = cfg["agg"](df_test_cc, feat_cols)
        agg_feat_cols = [
            c for c in train_tab.columns if c not in ("unit", "cycle", "RUL", "hs")
        ]
        caps = unit_physical_caps(pd.concat([train_tab, test_tab], ignore_index=True))

        X_train = train_tab[agg_feat_cols].to_numpy(dtype=np.float64)
        X_test = test_tab[agg_feat_cols].to_numpy(dtype=np.float64)
        sc = StandardScaler().fit(X_train)
        X_train, X_test = sc.transform(X_train), sc.transform(X_test)
        y_train = apply_rul_shape(train_tab, "C3_physical", caps).to_numpy()
        y_test_true = test_tab["RUL"].astype(float).to_numpy()

        model = TribbleRegressor(random_state=42, max_samples=2000, **cfg["kw"])
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        pred_baseline = model.predict(X_test)
        rmse_baseline = float(np.sqrt(mean_squared_error(y_test_true, pred_baseline)))

        # --- model complexity ---
        n_mf = model.model_.n_membership_functions
        n_output_buckets = model.n_rules_
        possible_rules = model.model_.possible_rules
        antecedent_params = 2 * n_mf
        consequent_params = int(np.prod(model.corr_terms_.shape)) + len(
            np.atleast_1d(model.y_bucket_mean_)
        )
        complexity_rows.append(
            dict(
                pipeline=name,
                n_membership_functions=n_mf,
                n_output_buckets=n_output_buckets,
                possible_rules=possible_rules,
                antecedent_params=antecedent_params,
                consequent_params=consequent_params,
                total_params=antecedent_params + consequent_params,
                n_selected_features=len(model.top_features_),
                fit_seconds=fit_seconds,
                rmse_baseline=rmse_baseline,
            )
        )
        print(
            f"baseline: rmse={rmse_baseline:.3f}  fit={fit_seconds:.2f}s  "
            f"n_mf={n_mf}  total_params={antecedent_params + consequent_params}"
        )

        # --- coordinate-descent refinement (one call: refine antecedents,
        # re-solve consequents, get both the summary numbers and the refined
        # model's predictions) ---
        y_series = pd.Series(y_train, name="y_value")
        y_part, y_bucket_mean = partition_output(
            model.n_output_buckets, y_series, method=model.output_partition
        )
        X_train_df = pd.DataFrame(X_train, columns=model.feature_names_in_)
        t0 = time.perf_counter()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            refined_model, _ = refine_antecedents_coordinate(
                model.model_,
                X_train_df,
                y_part,
                model.top_features_,
                n_output_buckets=model.n_output_buckets,
                order=model.tsk_order,
                l2_reg=model.l2_reg,
                basis=model.consequent_basis,
                cross_pairs=model.cross_pairs_,
                norms=model._norms(),
                **cfg["refine_kwargs"],
            )
            corr_terms, ybm = solve_tsk_consequents(
                X_train_df,
                refined_model,
                model.top_features_,
                y_bucket_mean,
                y_part,
                n_output_buckets=model.n_output_buckets,
                order=model.tsk_order,
                l2_reg=model.l2_reg,
                basis=model.consequent_basis,
                pin_extremes=model.pin_extremes,
                norms=model._norms(),
                cross_pairs=model.cross_pairs_,
                rbf_centers=model.rbf_centers_,
                rbf_gamma=model.rbf_gamma,
                rbf_radius=model.rbf_radius,
            )
        refine_seconds = time.perf_counter() - t0

        import copy

        refined = copy.deepcopy(model)
        refined.model_, refined.corr_terms_, refined.y_bucket_mean_ = (
            refined_model,
            corr_terms,
            ybm,
        )
        pred_refined = refined.predict(X_test)
        rmse_refined = float(np.sqrt(mean_squared_error(y_test_true, pred_refined)))
        print(
            f"coordinate refined: rmse={rmse_refined:.3f}  refine_seconds={refine_seconds:.2f}s  "
            f"(baseline was {rmse_baseline:.3f})"
        )

        # --- smoothing, on top of the refined predictions ---
        preds = pd.DataFrame(
            {
                "unit": test_tab["unit"].to_numpy(),
                "cycle": test_tab["cycle"].to_numpy(),
                "RUL_true": y_test_true,
                "RUL_pred": pred_refined,
            }
        )
        preds_smoothed = smooth_predictions(preds, window=7)
        rmse_smoothed = float(
            np.sqrt(
                mean_squared_error(
                    preds_smoothed["RUL_true"], preds_smoothed["RUL_pred_smoothed"]
                )
            )
        )
        print(f"+ smoothing (window=7): rmse={rmse_smoothed:.3f}")

        complexity_rows[-1].update(
            rmse_refined=rmse_refined,
            refine_seconds=refine_seconds,
            rmse_refined_smoothed=rmse_smoothed,
        )
        preds_smoothed["RUL_pred_baseline"] = pred_baseline
        trajectory_predictions[name] = preds_smoothed

    print("\n" + "=" * 78)
    print("MODEL COMPLEXITY")
    print("=" * 78)
    complexity_df = pd.DataFrame(complexity_rows)
    print(
        complexity_df[
            [
                "pipeline",
                "n_selected_features",
                "n_membership_functions",
                "antecedent_params",
                "consequent_params",
                "total_params",
            ]
        ].to_string(index=False)
    )
    complexity_df.to_csv(
        "FuzzySystemsExperiments/cmapss_rul_champion_complexity.csv", index=False
    )

    print("\n" + "=" * 78)
    print("REFINEMENT + SMOOTHING SUMMARY")
    print("=" * 78)
    print(
        complexity_df[
            [
                "pipeline",
                "rmse_baseline",
                "rmse_refined",
                "rmse_refined_smoothed",
                "refine_seconds",
            ]
        ].to_string(index=False)
    )

    return complexity_df, trajectory_predictions


if __name__ == "__main__":
    main()
