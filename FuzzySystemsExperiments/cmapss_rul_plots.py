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
    "A3_raw_memory": "#eb6834",  # orange
    "A2_phase_split": "#1baf7a",  # aqua
    # Condition-corrected variants keep their base family's color -- same
    # aggregation strategy, just fed corrected input.
    "A1_whole_cycle_cc": "#2a78d6",
    "A3_raw_memory_cc": "#eb6834",
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
        sum(LIT_BAND) / 2,
        len(df) - 0.3,
        "published CNN/MLP (DS02-006)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK_MUTED,
        style="italic",
    )
    ax.axvline(CONST_MEAN_RMSE, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.text(
        CONST_MEAN_RMSE + 0.4,
        len(df) - 0.3,
        "constant-mean baseline",
        ha="left",
        va="top",
        fontsize=8,
        color=INK_MUTED,
        style="italic",
    )
    for yi, v in zip(y, df["rmse_test_true"]):
        ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=8, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(df["pipeline"].str.replace("_", " "), fontsize=8.5, color=INK)
    ax.set_xlabel("Test RMSE, true RUL (cycles)  —  lower is better")
    ax.set_title(
        "Stage 1 screen: RMSE across all 18 aggregation / feature / RUL-shaping pipelines",
        fontsize=12,
        pad=14,
        loc="left",
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
        handles,
        ["A1 whole-cycle", "A2 phase-split", "A3 raw-memory"],
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
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
            sub["fit_seconds"],
            sub["rmse_test_true"],
            s=22,
            color=color,
            alpha=0.55,
            edgecolors="none",
            zorder=3,
        )
        best = sub.loc[sub["rmse_test_true"].idxmin()]
        ax.scatter(
            [best["fit_seconds"]],
            [best["rmse_test_true"]],
            s=140,
            facecolors="none",
            edgecolors=color,
            linewidths=2,
            zorder=4,
        )
        ax.annotate(
            pipeline.replace("_", " "),
            (best["fit_seconds"], best["rmse_test_true"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    ax.axhspan(LIT_BAND[0], LIT_BAND[1], color=INK_MUTED, alpha=0.10, zorder=1)
    ax.axvline(1.0, color=BASELINE, linewidth=1, linestyle="--", zorder=1)
    ax.text(
        1.05, ax.get_ylim()[1] * 0.97, "1 second", fontsize=8, color=INK_MUTED, va="top"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Fit time, log scale (seconds)")
    ax.set_ylabel("Test RMSE, true RUL (cycles)")
    ax.set_title(
        "Stage 2: accuracy vs. training cost across the Factor D grid (54 configs × 3 pipelines)",
        fontsize=12,
        pad=14,
        loc="left",
    )
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CAT[p.split("/")[0]],
            markersize=8,
            label=p.replace("_", " "),
        )
        for p in pipelines
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


GRID_PIPELINE_COLOR = {
    # Same hues as the aggregation strategy each pipeline actually uses
    # (CAT above), so this chart reads consistently with every other plot
    # in this DOE: honest = whole_cycle (blue), best = raw_memory (orange).
    "honest": CAT["A1_whole_cycle"],
    "best": CAT["A3_raw_memory"],
}
GRID_PIPELINE_LABEL = {
    "honest": "honest (physical sensors only, 18ch)",
    "best": "best (physical + 2 virtual, 20ch)",
}


def _short_dataset_label(dataset: str) -> str:
    name = dataset.replace("N-CMAPSS_", "").replace(".h5", "")
    return name.split("-")[0]  # "DS08a-009" -> "DS08a", "DS02-006" -> "DS02"


def plot_grid_results(
    grid_df: pd.DataFrame, out_path: str, tuned_dataset: str = "N-CMAPSS_DS02-006.h5"
):
    """Grouped bar chart of RMSE per dataset per pipeline (cmapss_rul_best.py
    --grid). Both pipelines use hyperparameters tuned only on `tuned_dataset`
    -- every other dataset is a zero-shot generalization check, not a
    per-dataset best case, so that dataset's bars are highlighted to make
    the trained-on/untrained-on split visually obvious."""
    datasets = sorted(grid_df["dataset"].unique())
    pipelines = ["honest", "best"]
    x = np.arange(len(datasets))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)

    for j, pipeline in enumerate(pipelines):
        offset = (j - 0.5) * width
        values = [
            grid_df.loc[
                (grid_df["dataset"] == d) & (grid_df["pipeline"] == pipeline), "rmse"
            ].iloc[0]
            for d in datasets
        ]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=GRID_PIPELINE_COLOR[pipeline],
            label=GRID_PIPELINE_LABEL[pipeline],
            zorder=3,
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=7.5, color=INK_SECONDARY, padding=2)

    if tuned_dataset in datasets:
        i = datasets.index(tuned_dataset)
        ax.axvspan(i - 0.5, i + 0.5, color=INK_MUTED, alpha=0.12, zorder=0)
        local_max = grid_df.loc[grid_df["dataset"] == tuned_dataset, "rmse"].max()
        ax.text(
            i,
            local_max + ax.get_ylim()[1] * 0.03,
            "tuned here",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_short_dataset_label(d) for d in datasets], fontsize=9.5)
    ax.set_ylabel("Test RMSE, true RUL (cycles)")
    ax.set_title(
        "Zero-shot generalization: RMSE across all N-CMAPSS datasets\n"
        "(config tuned on DS02 only, not re-tuned per dataset)",
        fontsize=12.5,
        pad=14,
        loc="left",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_stage3(
    predictions: dict, out_path: str, title: str = None, row_labels: dict = None
):
    pipelines = list(predictions.keys())
    fig, axes = plt.subplots(
        len(pipelines),
        3,
        figsize=(13, 3.4 * len(pipelines)),
        facecolor=SURFACE,
        sharex=False,
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
            ax.plot(
                sub["cycle"],
                sub["RUL_true"],
                color=INK,
                linewidth=2,
                label="true RUL",
                zorder=3,
            )
            ax.plot(
                sub["cycle"],
                sub["RUL_pred"],
                color=color,
                linewidth=2,
                linestyle="--",
                label="predicted",
                zorder=3,
            )
            ax.set_title(f"unit {unit}", fontsize=10, color=INK, loc="left")
            ax.grid(color=GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            _style_axes(ax)
            if col == 0:
                label = (row_labels or {}).get(
                    pipeline, pipeline.split("/", 1)[0].replace("_", " ")
                )
                ax.set_ylabel(label + "\nRUL (cycles)", fontsize=8.5)
            if row == len(pipelines) - 1:
                ax.set_xlabel("cycle")
            if row == 0 and col == 2:
                ax.legend(
                    loc="upper right",
                    frameon=False,
                    fontsize=8.5,
                    labelcolor=INK_SECONDARY,
                )

    fig.suptitle(
        title
        or "Stage 3: predicted vs. true RUL trajectories on the held-out test units",
        fontsize=12.5,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


REFINE_COLORS = {
    "baseline": INK_MUTED,
    "coordinate": "#2a78d6",
    "local": "#4a3aa7",
    "optimizers_ga": "#1baf7a",
}


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
            vals = [
                stage4_df[stage4_df["pipeline"] == p]["rmse_baseline"].iloc[0]
                for p in pipelines
            ]
        else:
            vals = [
                (
                    stage4_df[
                        (stage4_df["pipeline"] == p) & (stage4_df["refiner"] == name)
                    ]["rmse_refined"].iloc[0]
                    if not stage4_df[
                        (stage4_df["pipeline"] == p) & (stage4_df["refiner"] == name)
                    ].empty
                    else np.nan
                )
                for p in pipelines
            ]
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width * 0.9,
            color=REFINE_COLORS[name],
            label=name,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in pipelines], fontsize=8.5)
    ax.set_ylabel("Test RMSE, true RUL (cycles)")
    fastest = stage4_df.loc[stage4_df.groupby("refiner")["refine_seconds"].idxmax()]
    cost_note = " / ".join(
        f"{row.refiner} up to {row.refine_seconds:,.0f}s"
        for row in fastest.itertuples()
    )
    ax.set_title(
        f"Stage 4: refinement helps where there's enough data/parameters to support it --\n"
        f"cost varies a lot by method ({cost_note})",
        fontsize=11.5,
        pad=14,
        loc="left",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _compact_config_label(refiner: str, config_str: str) -> str:
    import ast

    cfg = ast.literal_eval(config_str)
    if refiner == "coordinate":
        return f"coordinate  sweeps={cfg['n_sweeps']}"
    scale = cfg.get("local_scale")
    scale_s = "global" if scale is None else str(scale)
    return f"ga  pop={cfg['population_size']} gens={cfg['num_generations']} scale={scale_s}"


def plot_stage4b(stage4b_df: pd.DataFrame, baseline_rmse: float, out_path: str):
    df = stage4b_df.reset_index(drop=True)
    labels = [_compact_config_label(row.refiner, row.config) for row in df.itertuples()]
    colors = [REFINE_COLORS.get(r, INK_MUTED) for r in df["refiner"]]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    y = np.arange(len(df))
    ax.barh(y, df["rmse_refined"], color=colors, height=0.6, zorder=3)
    ax.axvline(baseline_rmse, color=INK, linewidth=1.5, linestyle="--", zorder=2)
    ax.text(
        baseline_rmse + 0.15,
        0.15,
        "heuristic\nbaseline",
        fontsize=8,
        color=INK,
        ha="left",
        va="top",
        style="italic",
    )
    xmax = max(df["rmse_refined"].max(), baseline_rmse)
    ax.set_xlim(0, xmax * 1.18)
    for yi, v, s in zip(y, df["rmse_refined"], df["refine_seconds"]):
        ax.text(
            v + xmax * 0.01,
            yi,
            f"{v:.2f} ({s:.1f}s)",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK, fontfamily="monospace")
    ax.set_xlabel("Test RMSE, true RUL (cycles) -- A1_whole_cycle/B1/C3_physical")
    ax.set_title(
        "Stage 4b: sweeping each refiner's own hyperparameters on the small pipeline",
        fontsize=12,
        pad=14,
        loc="left",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_stage5(onset_df: pd.DataFrame, cap_results_df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)

    ax = axes[0]
    lo = min(onset_df["true_onset"].min(), onset_df["detected_onset"].min()) - 3
    hi = max(onset_df["true_onset"].max(), onset_df["detected_onset"].max()) + 3
    ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(
        onset_df["true_onset"],
        onset_df["detected_onset"],
        s=70,
        color=CAT["A3_raw_memory"],
        zorder=3,
    )
    for row in onset_df.itertuples():
        ax.annotate(
            str(row.unit),
            (row.true_onset, row.detected_onset),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True onset (oracle hs), cycle")
    ax.set_ylabel("Detected onset (moving avg.), cycle")
    mae = onset_df["error"].abs().mean()
    ax.set_title(
        f"Onset detection: MAE = {mae:.1f} cycles", fontsize=11.5, pad=12, loc="left"
    )
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)

    ax2 = axes[1]
    if cap_results_df is None or cap_results_df.empty:
        ax2.text(
            0.5,
            0.5,
            "No C3_physical pipeline\nin this run's top picks",
            ha="center",
            va="center",
            fontsize=11,
            color=INK_MUTED,
            style="italic",
            transform=ax2.transAxes,
        )
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_visible(False)
    else:
        pipelines = list(cap_results_df["pipeline"].unique())
        cap_sources = ["oracle_hs", "detected_ma"]
        cap_colors = {"oracle_hs": INK_MUTED, "detected_ma": CAT["A3_raw_memory"]}
        x = np.arange(len(pipelines))
        width = 0.35
        for i, cs in enumerate(cap_sources):
            vals = [
                cap_results_df[
                    (cap_results_df["pipeline"] == p)
                    & (cap_results_df["cap_source"] == cs)
                ]["rmse_test_true"].iloc[0]
                for p in pipelines
            ]
            ax2.bar(
                x + (i - 0.5) * width,
                vals,
                width=width * 0.9,
                color=cap_colors[cs],
                label=cs,
                zorder=3,
            )
        ax2.set_xticks(x)
        ax2.set_xticklabels([p.replace("_", " ") for p in pipelines], fontsize=8)
        ax2.set_ylabel("Test RMSE, true RUL (cycles)")
        ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax2.set_axisbelow(True)
        _style_axes(ax2)
        ax2.legend(
            loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY
        )
    ax2.set_title(
        "Cost of using the detected onset instead of the oracle",
        fontsize=11.5,
        pad=12,
        loc="left",
    )

    fig.suptitle(
        "Stage 5: can a moving-average detector replace the oracle hs flag?",
        fontsize=12.5,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# Series colors distinct from the Factor-A categorical hues used elsewhere,
# since this chart tracks *pipelines over milestones*, not aggregation family.
PROGRESSION_COLORS = {"overall": "#4a3aa7", "real_sensor": "#eb6834"}


def plot_progression(milestones: list[dict], out_path: str):
    """milestones: list of {label, overall, real_sensor} in chronological
    order. 'overall' may use virtual/unmeasurable sensors (leakage-flagged
    sensitivity arm); 'real_sensor' is the honest deployable number."""
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=SURFACE)
    x = np.arange(len(milestones))
    labels = [m["label"] for m in milestones]

    for key, name in [
        ("overall", "best overall"),
        ("real_sensor", "best real-sensor-only"),
    ]:
        vals = [m[key] for m in milestones]
        color = PROGRESSION_COLORS[key]
        ax.plot(
            x,
            vals,
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            zorder=4,
            label=name,
        )
        for xi, v in zip(x, vals):
            ax.annotate(
                f"{v:.2f}",
                (xi, v),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.axhspan(LIT_BAND[0], LIT_BAND[1], color=INK_MUTED, alpha=0.10, zorder=1)
    ax.text(
        x[-1] + 0.08,
        sum(LIT_BAND) / 2,
        "published\nCNN/MLP\n(DS02-006)",
        fontsize=8,
        color=INK_MUTED,
        style="italic",
        va="center",
    )
    ax.axhline(CONST_MEAN_RMSE, color=BASELINE, linewidth=1, linestyle="--", zorder=1)
    ax.text(
        x[0] - 0.08,
        CONST_MEAN_RMSE,
        "constant-mean baseline",
        fontsize=8,
        color=INK_MUTED,
        style="italic",
        ha="right",
        va="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlim(-0.4, len(milestones) - 1 + 0.6)
    ax.set_ylabel("Test RMSE, true RUL (cycles)")
    ax.set_title(
        "How we got here: RMSE across successive rounds of this DOE",
        fontsize=13,
        pad=16,
        loc="left",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def make_plots(
    stage1_df,
    stage2_df,
    stage3_predictions,
    top_pipelines,
    stage4_df=None,
    stage4b_df=None,
    stage5_onsets=None,
    stage5_results=None,
):
    plot_stage1(stage1_df, "FuzzySystemsExperiments/outputs/cmapss_rul_stage1.png")
    print("wrote FuzzySystemsExperiments/outputs/cmapss_rul_stage1.png")
    plot_stage2(
        stage2_df,
        top_pipelines,
        "FuzzySystemsExperiments/outputs/cmapss_rul_stage2.png",
    )
    print("wrote FuzzySystemsExperiments/outputs/cmapss_rul_stage2.png")
    plot_stage3(
        stage3_predictions, "FuzzySystemsExperiments/outputs/cmapss_rul_stage3.png"
    )
    print("wrote FuzzySystemsExperiments/outputs/cmapss_rul_stage3.png")
    if stage4_df is not None and not stage4_df.empty:
        plot_stage4(stage4_df, "FuzzySystemsExperiments/outputs/cmapss_rul_stage4.png")
        print("wrote FuzzySystemsExperiments/outputs/cmapss_rul_stage4.png")
    if stage4b_df is not None and not stage4b_df.empty:
        baseline_rmse = stage4b_df["rmse_baseline"].iloc[0]
        plot_stage4b(
            stage4b_df,
            baseline_rmse,
            "FuzzySystemsExperiments/outputs/cmapss_rul_stage4b.png",
        )
        print("wrote FuzzySystemsExperiments/outputs/cmapss_rul_stage4b.png")
    if (
        stage5_onsets is not None
        and stage5_results is not None
        and not stage5_onsets.empty
    ):
        plot_stage5(
            stage5_onsets,
            stage5_results,
            "FuzzySystemsExperiments/outputs/cmapss_rul_stage5.png",
        )
        print("wrote FuzzySystemsExperiments/outputs/cmapss_rul_stage5.png")
