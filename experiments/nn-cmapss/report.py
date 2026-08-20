"""Turn the run artifacts into the benchmark report and its figures.

Reads only what the runners wrote to `outputs/nn-cmapss/`; nothing here refits
anything, so the report cannot disagree with the runs it describes.
"""

from __future__ import annotations

import json
import os

import re

import numpy as np
import pandas as pd

import cmapss_data

OUT = os.path.join(cmapss_data.REPO, "outputs", "nn-cmapss")
FIG = os.path.join(OUT, "figures")

# Categorical slots 1-5 of the data-viz reference palette, in its fixed order.
# Used unchanged and in order, which is the only way the published adjacent-pair
# CVD separation still holds; scatter plots cap at three by that palette's
# all-pairs rule, so the cost/quality scatter uses one hue plus direct labels.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"

ARM_LABEL = {
    "he": "he (random)",
    "he-all": "he, all features",
    "quantile": "quantile knots",
    "quantile-all": "quantile, all features",
    "elm": "random features (ELM)",
    "hot-analytic": "hot (label-free seed)",
    "hot": "hot (seed + ridge)",
}


def slug(tag: str) -> str:
    """Filename-safe key for a run label. `tag.split()[0]` was not enough --
    "best" and "best (1st-order FIS)" both reduced to "best" and the second
    figure silently overwrote the first."""
    return re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def arm_table(res: dict) -> pd.DataFrame:
    """Median over seeds, one row per arm, with the reference models above it."""
    rows = []
    for tag, r in res["references"].items():
        rows.append(
            dict(
                arm=tag,
                kind="reference",
                n_hidden=np.nan,
                n_parameters=np.nan,
                epochs=np.nan,
                setup_s=r["setup_seconds"],
                train_s=0.0,
                total_s=r["setup_seconds"],
                start_rmse=np.nan,
                rmse=r["test"]["rmse"],
                mae=r["test"]["mae"],
                nasa=r["test"]["nasa"],
                rmse_endpoint=r["test"]["rmse_endpoint"],
            )
        )
    df = pd.DataFrame(res["arms"])
    for arm, g in df.groupby("arm", sort=False):
        rows.append(
            dict(
                arm=arm,
                kind="network",
                n_hidden=(
                    g["n_hidden_actual"].median()
                    if "n_hidden_actual" in g
                    else g["n_hidden"].iloc[0]
                ),
                n_hidden_asked=g["n_hidden"].iloc[0],
                n_parameters=g["n_parameters"].median(),
                epochs=g["selected_epoch"].median(),
                setup_s=g["setup_seconds"].median(),
                train_s=g["train_seconds"].median(),
                total_s=g["total_seconds"].median(),
                start_rmse=np.median([r["rmse"] for r in g["start_test"]]),
                rmse=np.median([r["rmse"] for r in g["final_test"]]),
                rmse_iqr=float(
                    np.subtract(
                        *np.percentile([r["rmse"] for r in g["final_test"]], [75, 25])
                    )
                ),
                mae=np.median([r["mae"] for r in g["final_test"]]),
                nasa=np.median([r["nasa"] for r in g["final_test"]]),
                rmse_endpoint=np.median([r["rmse_endpoint"] for r in g["final_test"]]),
            )
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols, headers, fmts) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for _, r in df.iterrows():
        cells = []
        for c, f in zip(cols, fmts):
            v = r.get(c, np.nan)
            cells.append(
                "--"
                if (isinstance(v, float) and not np.isfinite(v))
                else (f.format(v) if f else str(v))
            )
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_sweep(sweeps, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(sweeps), figsize=(5.6 * len(sweeps), 4.0), facecolor=SURFACE
    )
    axes = np.atleast_1d(axes)
    for ax, (tag, d) in zip(axes, sweeps.items()):
        df = pd.DataFrame(d["rows"])
        for i, space in enumerate(["fis", "all"]):
            sub = df[df.space == space]
            if not len(sub):
                continue
            # Best configuration at each width: what the width can achieve, not
            # what an arbitrary learning rate happens to give at that width.
            best = (
                sub.groupby(["n_hidden", "lr", "batch_size"])["best_val_rmse"]
                .median()
                .reset_index()
                .sort_values("best_val_rmse")
                .groupby("n_hidden")
                .head(1)
                .sort_values("n_hidden")
            )
            ax.plot(
                best["n_hidden"],
                best["best_val_rmse"],
                "-o",
                color=C[i],
                linewidth=2,
                markersize=7,
                label={"fis": "TRIBBLE's features", "all": "all features"}[space],
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
        ax.axhline(d["fis"]["val"]["rmse"], color=INK2, linestyle="--", linewidth=1.5)
        ax.text(
            ax.get_xlim()[1],
            d["fis"]["val"]["rmse"],
            " FIS ",
            color=INK2,
            fontsize=9,
            va="bottom",
            ha="right",
        )
        ax.axhline(d["baselines"]["ridge"], color=GRID, linestyle=":", linewidth=1.5)
        ax.set_xscale("log", base=2)
        _style(
            ax,
            "hidden units",
            "validation RMSE (cycles)",
            f"{tag}: width is not the binding constraint",
        )
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_time_to_quality(
    res, path, arms=("he", "quantile", "elm", "hot", "hot-analytic")
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(res["arms"])
    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
    for i, arm in enumerate(arms):
        g = df[(df.arm == arm) & (df.seed == df.seed.min())]
        if not len(g):
            continue
        cur = g.iloc[0]["curve"]
        setup = g.iloc[0]["setup_seconds"]
        # Wall clock charged honestly: the FIS fit and the conversion are part
        # of what a hot start costs, so the curve starts where setup ends.
        t = np.asarray(cur["seconds"]) + setup
        r = np.asarray(cur["test_rmse"])
        keep = np.isfinite(r) & (r < 60)
        ax.plot(
            t[keep],
            r[keep],
            "-",
            color=C[i % len(C)],
            linewidth=2,
            label=ARM_LABEL.get(arm, arm),
        )
        ax.plot(
            t[keep][:1],
            r[keep][:1],
            "o",
            color=C[i % len(C)],
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
    fis_r = res["references"]["fis"]["test"]["rmse"]
    ax.axhline(fis_r, color=INK2, linestyle="--", linewidth=1.5)
    ax.text(
        ax.get_xlim()[1],
        fis_r,
        f" FIS {fis_r:.2f} ",
        color=INK2,
        fontsize=9,
        va="bottom",
        ha="right",
    )
    ax.set_xscale("log")
    _style(
        ax,
        "wall clock from zero, including setup (s)",
        "test RMSE (cycles)",
        "Time to quality: setup charged to the arm that needs it",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_fidelity(fids, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(fids), figsize=(5.6 * len(fids), 4.0), facecolor=SURFACE
    )
    axes = np.atleast_1d(axes)
    for ax, (tag, d) in zip(axes, fids.items()):
        df = pd.DataFrame(d["rows"])
        df = df[df.top_n > 0].sort_values("n_features")
        ax.plot(
            df.n_features,
            df.fidelity_relative,
            "-o",
            color=C[0],
            linewidth=2,
            markersize=7,
            label="the conversion's seed",
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.plot(
            df.n_features,
            df.additive_relative,
            "--s",
            color=C[1],
            linewidth=2,
            markersize=6,
            label="best possible additive fit",
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.axhline(1.0, color=GRID, linestyle=":", linewidth=1.5)
        ax.set_yscale("log")
        _style(
            ax,
            "features the FIS kept",
            "seed-vs-FIS error / FIS std",
            f"{tag}: the seed is as good as additive gets",
        )
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_cost_quality(tables, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 2 x 2, not 1 x N: at four panels in a row the direct labels collided into
    # an unreadable stack, which is the failure the "render it and look at it"
    # step exists to catch.
    n = len(tables)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.6 * ncols, 5.2 * nrows),
        facecolor=SURFACE,
        squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    for ax, (tag, df) in zip(flat, tables.items()):
        d = df[np.isfinite(df.rmse) & (df.rmse < 40)].copy()
        d["total_s"] = d["total_s"].clip(lower=1e-3)
        d = d.sort_values("rmse").reset_index(drop=True)
        # One hue plus direct labels: the palette's all-pairs rule caps
        # categorical scatter at three series, and there are eleven points here.
        ax.scatter(
            d.total_s,
            d.rmse,
            s=70,
            color=C[0],
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        # De-collide the direct labels. Alternating left/right alone was not
        # enough: on `best`, four arms land inside 0.3 cycles *and* inside one
        # decade of wall clock, so labels have to be pushed apart vertically as
        # well. Walk the points in RMSE order and give each one the first free
        # vertical slot, measured as a fraction of the y-range.
        span = max(d.rmse.max() - d.rmse.min(), 1e-9)
        placed = []  # y positions already used, in data units
        min_gap = 0.045 * span
        offsets = []
        for _, r in d.iterrows():
            y = float(r["rmse"])
            # Push *upward* only. Pushing both ways sent the lowest cluster's
            # labels through the bottom spine and over the x-axis title; going
            # one direction keeps every label inside a range we can pad for.
            while any(abs(y - p) < min_gap for p in placed):
                y += min_gap
            placed.append(y)
            offsets.append(y)
        d = d.assign(_label_y=offsets)
        ax.set_ylim(
            d.rmse.min() - 0.06 * span, max(d.rmse.max(), max(offsets)) + 0.08 * span
        )
        for i, r in d.iterrows():
            y = float(r["_label_y"])
            right = i % 2 == 0
            ax.annotate(
                ARM_LABEL.get(r["arm"], r["arm"]),
                xy=(r["total_s"], r["rmse"]),
                xytext=(r["total_s"], y),
                textcoords="data",
                ha="left" if right else "right",
                va="center",
                fontsize=8,
                color=INK2,
                # A leader line only where the label had to move off its point.
                arrowprops=(
                    dict(
                        arrowstyle="-", color=GRID, linewidth=0.8, shrinkA=2, shrinkB=4
                    )
                    if abs(y - r["rmse"]) > 1e-9
                    else None
                ),
            )
            # `xytext` in data coords sits *on* the point; nudge it clear.
            ax.texts[-1].set_position((r["total_s"] * (1.35 if right else 0.74), y))
        ax.set_xscale("log")
        lo, hi = d.total_s.min(), d.total_s.max()
        ax.set_xlim(lo / 12, hi * 12)  # room for labels on both sides
        _style(
            ax,
            "total wall clock: setup + training (s)",
            "test RMSE (cycles)",
            f"{tag}: cost against quality (down and left is better)",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    sweeps = {
        k: v
        for k, v in (
            ("honest", load("sweep_honest_small.json")),
            ("best", load("sweep_best.json")),
        )
        if v
    }
    fids = {
        k: v
        for k, v in (
            ("honest", load("fidelity_honest.json")),
            ("best", load("fidelity_best.json")),
        )
        if v
    }
    arms = {
        k: v
        for k, v in (
            ("honest", load("arms_honest.json")),
            ("honest (FIS width)", load("arms_honest_convwidth.json")),
            ("best", load("arms_best.json")),
            ("best (1st-order FIS)", load("arms_best_1storder.json")),
        )
        if v
    }
    tables = {k: arm_table(v) for k, v in arms.items()}

    if sweeps:
        fig_sweep(sweeps, os.path.join(FIG, "sweep.png"))
    if fids:
        fig_fidelity(fids, os.path.join(FIG, "fidelity.png"))
    if tables:
        fig_cost_quality(tables, os.path.join(FIG, "cost_quality.png"))
    for tag, res in arms.items():
        fig_time_to_quality(res, os.path.join(FIG, f"time_to_quality_{slug(tag)}.png"))

    for tag, df in tables.items():
        print(f"\n=== {tag} ===")
        print(
            md_table(
                df.sort_values("rmse"),
                [
                    "arm",
                    "n_hidden",
                    "n_parameters",
                    "epochs",
                    "setup_s",
                    "train_s",
                    "total_s",
                    "rmse",
                    "mae",
                    "rmse_endpoint",
                ],
                [
                    "arm",
                    "hidden",
                    "params",
                    "epochs",
                    "setup s",
                    "train s",
                    "total s",
                    "test RMSE",
                    "MAE",
                    "endpoint RMSE",
                ],
                [
                    None,
                    "{:.0f}",
                    "{:.0f}",
                    "{:.0f}",
                    "{:.3f}",
                    "{:.3f}",
                    "{:.3f}",
                    "**{:.2f}**",
                    "{:.2f}",
                    "{:.2f}",
                ],
            )
        )
    print(f"\nfigures -> {os.path.relpath(FIG, cmapss_data.REPO)}")
    return tables, sweeps, fids


if __name__ == "__main__":
    main()
