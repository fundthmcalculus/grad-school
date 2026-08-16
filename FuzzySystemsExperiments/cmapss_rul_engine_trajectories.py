"""Per-engine RUL predictions: table + trajectory graphs, every engine of
every N-CMAPSS file, using PER-FILE fits.

Each .h5 file is fit independently: the per-engine champion config
(honest_full_tuned -- physical sensors only, whole-cycle aggregation,
StandardScaler) is trained on that file's own training units and evaluated
on that file's own test units. This per-file setup specializes each model
to its file's flight conditions and generalizes better than one pooled
global model (see cmapss_rul_full_analysis.py for the pooled comparison).

For every engine -- training and held-out test, across all files -- it
reports predicted vs. actual RUL. Whole-cycle aggregation gives one
prediction per flight cycle, so each engine's predicted-RUL curve overlays
cleanly on its true run-to-failure descent.

Outputs (all under FuzzySystemsExperiments/outputs/, gitignored):
  cmapss_rul_engine_predictions.csv        every engine, all files, metrics
  cmapss_rul_engine_file_summary.csv       per-file train/test RMSE summary
  cmapss_rul_engine_<DSxx>.png             trajectory grid per file (train+test)

Fits on each file's TRAINING data only (condition-correction, scaler, RUL
cap, model); that file's test engines are predicted on, never fit.
"""

import contextlib
import glob
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import cmapss_rul_full_analysis as m
from tribblefis.gaussian_regressor import TribbleRegressor

H5_DIR = "NASA-CMAPSS"
OUT_DIR = "FuzzySystemsExperiments/outputs"
CHAMPION_KW = dict(
    tsk_order="1st",
    n_gaussians=0,
    top_p=0.9,
    detect_interactions=False,
    norm_conorm="hamacher",
    l2_reg=0.01,
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
TRUE_C = "#52514e"  # true RUL
PRED_TRAIN_C = "#2a78d6"  # predicted RUL, training engine (blue)
PRED_TEST_C = "#eb6834"  # predicted RUL, test engine (orange)


def fit_one_file(h5_path, dataset):
    """Fit the honest champion on this file's train units; predict both
    train and test engines. Returns (train_tab, test_tab) with RUL_pred."""
    data, var = m.load_h5(h5_path)
    df_dev = m.to_frame(data, var, "dev", dataset)
    df_test = m.to_frame(data, var, "test", dataset)
    del data
    w = [f"W_{n}" for n in var["W"]]
    xs = [f"Xs_{n}" for n in var["X_s"]]
    xv = [f"Xv_{n}" for n in var["X_v"]]
    correct = xs + xv[: m.CORRECT_N_XV]
    models = m.fit_condition_correction(df_dev, correct, w)
    df_dev = m.apply_condition_correction(df_dev, correct, w, models)
    df_test = m.apply_condition_correction(df_test, correct, w, models)
    feat_cols = w + xs  # honest: physical sensors only
    train_tab = m.aggregate_whole_cycle(df_dev, feat_cols)
    test_tab = m.aggregate_whole_cycle(df_test, feat_cols)
    agg = [
        c
        for c in train_tab.columns
        if c not in ("dataset", "unit", "cycle", "RUL", "hs")
    ]
    caps = m.physical_rul_cap(train_tab)
    y_train = m.capped_rul(train_tab, caps)
    sc = StandardScaler().fit(train_tab[agg].to_numpy(np.float64))
    Xtr = sc.transform(train_tab[agg].to_numpy(np.float64))
    Xte = sc.transform(test_tab[agg].to_numpy(np.float64))
    mdl = TribbleRegressor(random_state=42, max_samples=2000, **CHAMPION_KW)
    with contextlib.redirect_stdout(io.StringIO()):
        mdl.fit(Xtr, y_train)
    return (
        train_tab.assign(RUL_pred=mdl.predict(Xtr)),
        test_tab.assign(RUL_pred=mdl.predict(Xte)),
    )


def engine_rows(tab, dataset, split):
    rows = []
    for unit, sub in tab.groupby("unit"):
        sub = sub.sort_values("cycle")
        true_f = float(sub["RUL"].iloc[-1])
        pred_f = float(sub["RUL_pred"].iloc[-1])
        rmse = float(np.sqrt(mean_squared_error(sub["RUL"], sub["RUL_pred"])))
        # endpoint NASA: the score's asymmetric penalty at this engine's
        # last cycle (the C-MAPSS truncation-point convention).
        endpoint_nasa = m.nasa_score([true_f], [pred_f])
        # average NASA: the mean per-cycle penalty over this engine's whole
        # trajectory (total score / n cycles) -- a length-independent
        # per-sample figure, unlike the raw summed score.
        avg_nasa = m.nasa_score(sub["RUL"], sub["RUL_pred"]) / len(sub)
        rows.append(
            dict(
                dataset=dataset,
                split=split,
                unit=int(unit),
                n_cycles=len(sub),
                final_true_RUL=round(true_f, 1),
                final_pred_RUL=round(pred_f, 1),
                final_abs_err=round(abs(true_f - pred_f), 1),
                trajectory_rmse=round(rmse, 2),
                endpoint_nasa=round(endpoint_nasa, 3),
                avg_nasa=round(avg_nasa, 3),
            )
        )
    return rows


def plot_file(dataset, train_tab, test_tab, out_path):
    engines = [("train", u) for u in sorted(train_tab["unit"].unique())] + [
        ("test", u) for u in sorted(test_tab["unit"].unique())
    ]
    ncol = min(5, len(engines))
    nrow = int(np.ceil(len(engines) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), facecolor=SURFACE, squeeze=False
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (split, unit) in enumerate(engines):
        ax = axes.flat[i]
        ax.set_visible(True)
        tab = train_tab if split == "train" else test_tab
        pred_c = PRED_TRAIN_C if split == "train" else PRED_TEST_C
        sub = tab[tab["unit"] == unit].sort_values("cycle")
        rmse = float(np.sqrt(mean_squared_error(sub["RUL"], sub["RUL_pred"])))
        ax.plot(sub["cycle"], sub["RUL"], color=TRUE_C, lw=1.8, zorder=3)
        ax.plot(sub["cycle"], sub["RUL_pred"], color=pred_c, lw=1.4, zorder=4)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_SECONDARY, labelsize=7)
        ax.grid(color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        tag = "TEST " if split == "test" else ""
        ax.set_title(
            f"{tag}unit {unit}  (RMSE {rmse:.1f})",
            fontsize=9,
            color=(PRED_TEST_C if split == "test" else INK),
            loc="left",
        )
    handles = [
        plt.Line2D([0], [0], color=TRUE_C, lw=2, label="true RUL"),
        plt.Line2D(
            [0], [0], color=PRED_TRAIN_C, lw=2, label="predicted (train engine)"
        ),
        plt.Line2D([0], [0], color=PRED_TEST_C, lw=2, label="predicted (test engine)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        f"{dataset}: per-engine predicted vs. true RUL (per-file honest fit)",
        fontsize=13,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    file_summary = []
    for h5_path in sorted(glob.glob(os.path.join(H5_DIR, "*.h5"))):
        dataset = (
            os.path.basename(h5_path).replace("N-CMAPSS_", "").replace(".h5", "")
        ).split("-")[0]
        try:
            train_tab, test_tab = fit_one_file(h5_path, dataset)
        except Exception as e:
            print(f"{dataset}: SKIPPED ({e!r})")
            file_summary.append(dict(dataset=dataset, status=f"skipped: {e!r}"))
            continue
        all_rows += engine_rows(train_tab, dataset, "train")
        all_rows += engine_rows(test_tab, dataset, "test")
        tr_rmse = float(
            np.sqrt(mean_squared_error(train_tab["RUL"], train_tab["RUL_pred"]))
        )
        te_rmse = float(
            np.sqrt(mean_squared_error(test_tab["RUL"], test_tab["RUL_pred"]))
        )
        # Test-set NASA reported two ways:
        #  - endpoint: one prediction per test engine at its last cycle
        #    (C-MAPSS truncation-point convention), summed and per-engine.
        #  - average: mean per-cycle penalty over all test rows (total score
        #    / n rows) -- a length-independent per-sample figure.
        te_last = test_tab.sort_values("cycle").groupby("unit").tail(1)
        endpoint_nasa_sum = m.nasa_score(te_last["RUL"], te_last["RUL_pred"])
        endpoint_nasa_mean = endpoint_nasa_sum / len(te_last)
        avg_nasa = m.nasa_score(test_tab["RUL"], test_tab["RUL_pred"]) / len(test_tab)
        file_summary.append(
            dict(
                dataset=dataset,
                status="ok",
                n_train_units=train_tab["unit"].nunique(),
                n_test_units=test_tab["unit"].nunique(),
                train_rmse=round(tr_rmse, 2),
                test_rmse=round(te_rmse, 2),
                test_endpoint_nasa_sum=round(endpoint_nasa_sum, 2),
                test_endpoint_nasa_mean=round(endpoint_nasa_mean, 3),
                test_avg_nasa=round(avg_nasa, 3),
            )
        )
        plot_file(
            dataset, train_tab, test_tab, f"{OUT_DIR}/cmapss_rul_engine_{dataset}.png"
        )
        print(
            f"{dataset}: test_rmse={te_rmse:.2f}  "
            f"NASA endpoint(sum={endpoint_nasa_sum:.1f}, mean={endpoint_nasa_mean:.2f})  "
            f"NASA avg/sample={avg_nasa:.2f}"
        )

    eng_df = pd.DataFrame(all_rows)
    eng_df.to_csv(f"{OUT_DIR}/cmapss_rul_engine_predictions.csv", index=False)
    sum_df = pd.DataFrame(file_summary)
    sum_df.to_csv(f"{OUT_DIR}/cmapss_rul_engine_file_summary.csv", index=False)

    # Fleet-wide test NASA, both conventions, over all files' test engines.
    test_eng = eng_df[eng_df["split"] == "test"]
    ok = sum_df[sum_df["status"] == "ok"]
    fleet_endpoint_sum = float(ok["test_endpoint_nasa_sum"].sum())
    fleet_endpoint_mean = fleet_endpoint_sum / int(ok["n_test_units"].sum())

    print("\n=== per-file summary ===")
    print(sum_df.to_string(index=False))
    print(
        f"\nFLEET test NASA -- endpoint: sum={fleet_endpoint_sum:.1f} over "
        f"{int(ok['n_test_units'].sum())} test engines "
        f"(mean {fleet_endpoint_mean:.2f}/engine);  "
        f"average per-sample: {test_eng['avg_nasa'].mean():.2f}"
    )
    print(f"{len(eng_df)} engines total across {eng_df['dataset'].nunique()} files")
    print(f"wrote {OUT_DIR}/cmapss_rul_engine_predictions.csv")
    print(f"wrote {OUT_DIR}/cmapss_rul_engine_file_summary.csv")


if __name__ == "__main__":
    main()
