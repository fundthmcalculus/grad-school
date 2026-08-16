"""Plots for the N-CMAPSS RUL DOE (cmapss_rul.py).

Palette and chart-form choices follow the repo's dataviz conventions:
fixed-order categorical hues (never cycled), one axis per chart, a legend
whenever 2+ series are present, thin recessive gridlines, direct labels
over decoration.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed categorical order -- first three slots (validated all-pairs) --
# assigned to Factor A's three levels, used consistently across every chart.
CAT = {
    "A1_whole_cycle": "#2a78d6",  # blue
    "A3_raw_memory": "#eb6834",   # orange
    "A2_phase_split": "#1baf7a",  # aqua
}
# DS02-specific references (Custode, Mo, Ferigo & Iacca 2022, re-running Arias
# Chao's own baselines on the actual released N-CMAPSS_DS02-006.h5 -- the
# original paper's ~4-5 RMSE was measured on a lower-noise pre-release file
# and is not reproducible on the file this DOE uses).
LIT_BAND = (7.22, 8.34)  # published CNN / MLP RMSE on DS02-006
CONST_MEAN_RMSE = 18.97  # naive constant-mean-predictor baseline, DS02-006


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def plot_stage1(stage1_df: pd.DataFrame, out_path: str):
    df = stage1_df.sort_values("rmse_test_true", ascending=True).reset_index(drop=True)
    agg = df["pipeline"].str.split("/").str[0]
    colors = agg.map(CAT)

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=SURFACE)
    y = np.arange(len(df))
    ax.barh(y, df["rmse_test_true"], color=colors, height=0.62, zorder=3)
    ax.axvspan(LIT_BAND[0], LIT_BAND[1], color=INK_MUTED, alpha=0.12, zorder=1)
    ax.text(
        sum(LIT_BAND) / 2, len(df) - 0.3, "published CNN/MLP (DS02-006)",
        ha="center", va="top", fontsize=8.5, color=INK_MUTED, style="italic",
    )
    ax.axvline(CONST_MEAN_RMSE, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.text(
        CONST_MEAN_RMSE + 0.4, len(df) - 0.3, "constant-mean baseline",
        ha="left", va="top", fontsize=8, color=INK_MUTED, style="italic",
    )
    for yi, v in zip(y, df["rmse_test_true"]):
        ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=8, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(df["pipeline"].str.replace("_", " "), fontsize=8.5, color=INK)
    ax.set_xlabel("Test RMSE, true RUL (cycles)  —  lower is better")
    ax.set_title(
        "Stage 1 screen: RMSE across all 18 aggregation / feature / RUL-shaping pipelines",
        fontsize=12, pad=14, loc="left",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.invert_yaxis()

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in [CAT["A1_whole_cycle"], CAT["A2_phase_split"], CAT["A3_raw_memory"]]
    ]
    ax.legend(
        handles, ["A1 whole-cycle", "A2 phase-split", "A3 raw-memory"],
        loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_stage2(stage2_df: pd.DataFrame, pipelines: list[str], out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    for pipeline in pipelines:
        sub = stage2_df[stage2_df["pipeline"] == pipeline]
        a_name = pipeline.split("/")[0]
        color = CAT[a_name]
        ax.scatter(
            sub["fit_seconds"], sub["rmse_test_true"], s=22, color=color,
            alpha=0.55, edgecolors="none", zorder=3,
        )
        best = sub.loc[sub["rmse_test_true"].idxmin()]
        ax.scatter(
            [best["fit_seconds"]], [best["rmse_test_true"]], s=140, facecolors="none",
            edgecolors=color, linewidths=2, zorder=4,
        )
        ax.annotate(
            pipeline.replace("_", " "),
            (best["fit_seconds"], best["rmse_test_true"]),
            xytext=(8, 8), textcoords="offset points", fontsize=8.5, color=INK_SECONDARY,
        )

    ax.axhspan(LIT_BAND[0], LIT_BAND[1], color=INK_MUTED, alpha=0.10, zorder=1)
    ax.axvline(1.0, color=BASELINE, linewidth=1, linestyle="--", zorder=1)
    ax.text(1.05, ax.get_ylim()[1] * 0.97, "1 second", fontsize=8, color=INK_MUTED, va="top")

    ax.set_xscale("log")
    ax.set_xlabel("Fit time, log scale (seconds)")
    ax.set_ylabel("Test RMSE, true RUL (cycles)")
    ax.set_title(
        "Stage 2: accuracy vs. training cost across the Factor D grid (54 configs × 3 pipelines)",
        fontsize=12, pad=14, loc="left",
    )
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)

    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CAT[p.split("/")[0]],
                           markersize=8, label=p.replace("_", " ")) for p in pipelines]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_stage3(predictions: dict, out_path: str):
    pipelines = list(predictions.keys())
    fig, axes = plt.subplots(
        len(pipelines), 3, figsize=(13, 3.4 * len(pipelines)), facecolor=SURFACE, sharex=False
    )
    if len(pipelines) == 1:
        axes = axes[np.newaxis, :]

    for row, pipeline in enumerate(pipelines):
        df = predictions[pipeline]
        units = sorted(df["unit"].unique())
        a_name = pipeline.split("/")[0]
        color = CAT[a_name]
        for col, unit in enumerate(units):
            ax = axes[row, col]
            sub = df[df["unit"] == unit].sort_values("cycle")
            ax.plot(sub["cycle"], sub["RUL_true"], color=INK, linewidth=2, label="true RUL", zorder=3)
            ax.plot(sub["cycle"], sub["RUL_pred"], color=color, linewidth=2,
                     linestyle="--", label="predicted", zorder=3)
            ax.set_title(f"unit {unit}", fontsize=10, color=INK, loc="left")
            ax.grid(color=GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            _style_axes(ax)
            if col == 0:
                ax.set_ylabel(pipeline.split("/", 1)[0].replace("_", " ") + "\nRUL (cycles)", fontsize=8.5)
            if row == len(pipelines) - 1:
                ax.set_xlabel("cycle")
            if row == 0 and col == 2:
                ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    fig.suptitle(
        "Stage 3: predicted vs. true RUL trajectories on the held-out test units",
        fontsize=12.5, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


REFINE_COLORS = {"baseline": INK_MUTED, "coordinate": "#2a78d6", "local": "#4a3aa7"}


def plot_stage4(stage4_df: pd.DataFrame, out_path: str):
    pipelines = list(stage4_df["pipeline"].unique())
    refiners = list(stage4_df["refiner"].unique())
    series = ["baseline"] + refiners

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(len(pipelines))

    for i, name in enumerate(series):
        if name == "baseline":
            vals = [stage4_df[stage4_df["pipeline"] == p]["rmse_baseline"].iloc[0] for p in pipelines]
        else:
            vals = [
                stage4_df[(stage4_df["pipeline"] == p) & (stage4_df["refiner"] == name)]["rmse_refined"].iloc[0]
                if not stage4_df[(stage4_df["pipeline"] == p) & (stage4_df["refiner"] == name)].empty
                else np.nan
                for p in pipelines
            ]
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9, color=REFINE_COLORS[name], label=name, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in pipelines], fontsize=8.5)
    ax.set_ylabel("Test RMSE, true RUL (cycles)")
    ax.set_title(
        "Stage 4: refinement helps where there's enough data/parameters to support it --\n"
        "at 200-3,000x the baseline fit time, and it hurts the smallest pipeline (CV-overfit)",
        fontsize=11.5, pad=14, loc="left",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def make_plots(stage1_df, stage2_df, stage3_predictions, top_pipelines, stage4_df=None):
    plot_stage1(stage1_df, "FuzzySystemsExperiments/cmapss_rul_stage1.png")
    print("wrote FuzzySystemsExperiments/cmapss_rul_stage1.png")
    plot_stage2(stage2_df, top_pipelines, "FuzzySystemsExperiments/cmapss_rul_stage2.png")
    print("wrote FuzzySystemsExperiments/cmapss_rul_stage2.png")
    plot_stage3(stage3_predictions, "FuzzySystemsExperiments/cmapss_rul_stage3.png")
    print("wrote FuzzySystemsExperiments/cmapss_rul_stage3.png")
    if stage4_df is not None and not stage4_df.empty:
        plot_stage4(stage4_df, "FuzzySystemsExperiments/cmapss_rul_stage4.png")
        print("wrote FuzzySystemsExperiments/cmapss_rul_stage4.png")
