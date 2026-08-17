"""Plot predicted-vs-true RUL trajectories for the two headline pipelines:

- A3_raw_memory_cc/B3/C3_physical -- best literature-matching config (the
  exact 20-channel input set the published CNN/MLP baselines use). RMSE 6.48.
- A1_whole_cycle_cc/B1/C3_physical -- best real-sensors-only ("physical")
  config, the strictest 18-channel definition. RMSE 11.23.

Both configs are Stage 2's confirmed grid-search winners, reproduced here
directly (not re-searched) for one clean plot in the same layout as
cmapss_rul_plots.plot_stage3.
"""

import contextlib
import io
import os
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

PIPELINES = {
    "A3_raw_memory_cc/B3/C3_physical": dict(
        aggregate=aggregate_raw_memory,
        feature_set="B3",
        tribble_kwargs=dict(
            tsk_order="full-2nd",
            n_gaussians=0,
            top_p=0.95,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
    ),
    "A1_whole_cycle_cc/B1/C3_physical": dict(
        aggregate=aggregate_whole_cycle,
        feature_set="B1",
        tribble_kwargs=dict(
            tsk_order="1st",
            n_gaussians=0,
            top_p=0.9,
            detect_interactions=False,
            norm_conorm="hamacher",
            l2_reg=0.01,
        ),
    ),
}


def main():
    print(f"Loading {H5_PATH} ...")
    data, var = load_h5(H5_PATH)
    df_dev = to_frame(data, var, "dev")
    df_test = to_frame(data, var, "test")

    w_cols = [f"W_{n}" for n in var["W"]]
    xs_cols = [f"Xs_{n}" for n in var["X_s"]]
    xv_cols = [f"Xv_{n}" for n in var["X_v"]]
    print("Fitting condition correction on dev-unit early cycles ...")
    cc_models = fit_raw_condition_correction(df_dev, xs_cols + xv_cols, w_cols)
    df_dev_cc = apply_raw_condition_correction(
        df_dev, xs_cols + xv_cols, w_cols, cc_models
    )
    df_test_cc = apply_raw_condition_correction(
        df_test, xs_cols + xv_cols, w_cols, cc_models
    )

    predictions = {}
    for pipeline, cfg in PIPELINES.items():
        feat_cols = feature_columns(var, cfg["feature_set"])
        train_tab = cfg["aggregate"](df_dev_cc, feat_cols)
        test_tab = cfg["aggregate"](df_test_cc, feat_cols)
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

        model = TribbleRegressor(
            random_state=42, max_samples=2000, **cfg["tribble_kwargs"]
        )
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        pred_test = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test_true, pred_test)))
        print(f"{pipeline:36s} rmse={rmse:.2f}  fit={fit_seconds:.2f}s")

        preds = pd.DataFrame(
            {
                "unit": test_tab["unit"].to_numpy(),
                "cycle": test_tab["cycle"].to_numpy(),
                "RUL_true": y_test_true,
                "RUL_pred": pred_test,
            }
        ).sort_values(["unit", "cycle"])
        predictions[pipeline] = preds

    from cmapss_rul_plots import plot_stage3

    os.makedirs("FuzzySystemsExperiments/outputs", exist_ok=True)
    out_path = "FuzzySystemsExperiments/outputs/cmapss_rul_champion_trajectories.png"
    plot_stage3(
        predictions,
        out_path,
        title="Best literature-matching (RMSE 6.48) vs. best real-sensors-only (RMSE 11.23)",
        row_labels={
            "A3_raw_memory_cc/B3/C3_physical": "literature-matching\n(20 ch, RMSE 6.48)",
            "A1_whole_cycle_cc/B1/C3_physical": "real sensors only\n(18 ch, RMSE 11.23)",
        },
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
