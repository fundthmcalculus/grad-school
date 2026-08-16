"""Per-engine RUL predictions: table + trajectory graphs, train and test.

Fits the per-engine champion (honest_full_tuned: physical sensors only,
whole-cycle aggregation, StandardScaler) on DS02 -- the sample file -- and
reports predicted vs. actual RUL for EVERY engine, both the training units
(2, 5, 10, 16, 18, 20) and the held-out test units (11, 14, 15).

Whole-cycle aggregation gives exactly one prediction per flight cycle, so
each engine's predicted-RUL curve overlays cleanly on its true-RUL line.

Outputs (all under FuzzySystemsExperiments/outputs/, gitignored):
  cmapss_rul_engine_predictions.csv     per-engine summary table
  cmapss_rul_engine_train.png           trajectory grid, training engines
  cmapss_rul_engine_test.png            trajectory grid, test engines

Fits on TRAINING data only (condition-correction, scaler, RUL cap, model);
test engines are predicted on, never fit.
"""

import contextlib
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

H5 = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"
OUT_DIR = "FuzzySystemsExperiments/outputs"
CHAMPION_KW = dict(
    tsk_order="1st",
    n_gaussians=0,
    top_p=0.9,
    detect_interactions=False,
    norm_conorm="hamacher",
    l2_reg=0.01,
)

# Palette (matches cmapss_rul_plots conventions)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
TRUE_C = "#52514e"  # true RUL: neutral ink
PRED_C = "#2a78d6"  # predicted RUL: blue


def build():
    data, var = m.load_h5(H5)
    df_dev = m.to_frame(data, var, "dev", "DS02")
    df_test = m.to_frame(data, var, "test", "DS02")
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
    train_tab = train_tab.assign(RUL_pred=mdl.predict(Xtr))
    test_tab = test_tab.assign(RUL_pred=mdl.predict(Xte))
    return train_tab, test_tab


def summarize(tab, split):
    rows = []
    for unit, sub in tab.groupby("unit"):
        sub = sub.sort_values("cycle")
        true_f = float(sub["RUL"].iloc[-1])
        pred_f = float(sub["RUL_pred"].iloc[-1])
        rmse = float(np.sqrt(mean_squared_error(sub["RUL"], sub["RUL_pred"])))
        rows.append(
            dict(
                split=split,
                unit=int(unit),
                n_cycles=len(sub),
                final_true_RUL=round(true_f, 1),
                final_pred_RUL=round(pred_f, 1),
                final_abs_err=round(abs(true_f - pred_f), 1),
                trajectory_rmse=round(rmse, 2),
            )
        )
    return rows


def plot_grid(tab, split, out_path):
    units = sorted(tab["unit"].unique())
    ncol = min(3, len(units))
    nrow = int(np.ceil(len(units) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), facecolor=SURFACE, squeeze=False
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for i, unit in enumerate(units):
        ax = axes.flat[i]
        ax.set_visible(True)
        sub = tab[tab["unit"] == unit].sort_values("cycle")
        rmse = float(np.sqrt(mean_squared_error(sub["RUL"], sub["RUL_pred"])))
        ax.plot(
            sub["cycle"], sub["RUL"], color=TRUE_C, lw=2, label="true RUL", zorder=3
        )
        ax.plot(
            sub["cycle"],
            sub["RUL_pred"],
            color=PRED_C,
            lw=1.6,
            label="predicted",
            zorder=4,
        )
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(
            f"unit {unit}   (traj RMSE {rmse:.1f})",
            fontsize=10,
            color=INK,
            loc="left",
        )
        if i % ncol == 0:
            ax.set_ylabel("RUL (cycles)", fontsize=9, color=INK_SECONDARY)
        ax.set_xlabel("cycle", fontsize=9, color=INK_SECONDARY)
    axes.flat[0].legend(
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right"
    )
    fig.suptitle(
        f"DS02 {split} engines: predicted vs. true RUL (honest champion)",
        fontsize=13,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_tab, test_tab = build()

    table = summarize(train_tab, "train") + summarize(test_tab, "test")
    df = pd.DataFrame(table)
    csv_path = f"{OUT_DIR}/cmapss_rul_engine_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}")

    plot_grid(train_tab, "train", f"{OUT_DIR}/cmapss_rul_engine_train.png")
    plot_grid(test_tab, "test", f"{OUT_DIR}/cmapss_rul_engine_test.png")
    print(f"wrote {OUT_DIR}/cmapss_rul_engine_train.png")
    print(f"wrote {OUT_DIR}/cmapss_rul_engine_test.png")


if __name__ == "__main__":
    main()
