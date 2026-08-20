"""Comparison figures for the arXiv:2504.13453 reproduction.

Mirrors the paper's own figure formats:

  fig_rmse_*           sorted bar chart, RMSE only, best on the left
  fig_r2_*             sorted bar chart, R^2 only, best on the left
  fig_angles_*         theta(t) truth vs prediction, the plot the reference
                       notebooks emit at the end of every model cell; the held-out
                       series runs to 20 s with the training-window edge marked
  fig_trajectory_*     bob paths in the plane, Figs. 14, 15, 16, 19, with the
                       past-the-window portion drawn faint
  fig_rmse_heatmap     the model x system RMSE heatmap of Fig. 22
  fig_capacity         not in the paper: held-out-IC score against rule count,
                       which is where the reproduction's main finding lives
  fig_error_vs_time    not in the paper: error against time for every dataset,
                       showing how far past the training window each survives
  fig_bracket          not in the paper: written by bracket_diagnostic.py

The paper draws RMSE and R^2 as one dual-axis grouped chart (its Figs. 11-13,
18B-D). That packs two scales, two sort orders, and rotated in-bar labels onto one
figure. Here they are two charts, each sorted best-first on its own axis, so the
ranking is readable straight off the bar heights.

No animations, by request.

Colors are categorical slots 1 and 2 of this repo's validated light-mode
palette (reproduce/figures/figstyle.py), inlined the way test_fuzzy_ode.py:64
does it rather than importing figstyle, which is coupled to the proposal figure
registry and its own save path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from n_pendulum_animation import chain_xy  # noqa: E402

import paper_results as pr  # noqa: E402

FIG_DIR = HERE / "figures"

# Validated light-mode palette, categorical slots 1-4 plus neutrals.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
RED = "#e34948"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#3d3d39"
GRID = "#e2e2dc"
AXIS = "#c9c9c1"
FAINT = "#9a9a92"

DPI = 200
FS_TITLE = 10
FS_LABEL = 9
FS_TICK = 8
FS_SMALL = 7

OURS = "FIS\n(ours)"


def _style(ax, title=None, xlabel=None, ylabel=None, grid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=FS_TICK, length=3, width=0.8)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL, color=INK)
    return ax


def tint(color, amount=0.55, toward=SURFACE):
    """Blend a palette colour toward the surface, returning a *solid* colour.

    Same approach as figstyle.tint: the blend happens here rather than via alpha,
    so a de-emphasised line stays a real colour rather than a transparency that
    flattens differently per backend.
    """

    def rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    a, b = rgb(color), rgb(toward)
    mix = tuple(round(x + (y - x) * amount) for x, y in zip(a, b))
    return "#{:02x}{:02x}{:02x}".format(*mix)


def regime_colors(labels):
    """Map dataset labels to palette slots, friction cool and frictionless warm.

    Keeps the two regimes visually separated however many chain lengths are
    present, and keeps a given dataset the same colour across every figure.

    The suffix test is on ``_frictionless``, not ``_friction``: every frictionless
    label also ends with the substring "friction"'s prefix, so testing the shorter
    suffix first would put both regimes in the same pool.
    """
    pools = {True: [BLUE, AQUA, VIOLET], False: [ORANGE, YELLOW, RED]}
    used = {True: 0, False: 0}
    out = {}
    for label in sorted(labels):
        has_friction = not label.endswith("_frictionless")
        pool = pools[has_friction]
        out[label] = pool[used[has_friction] % len(pool)]
        used[has_friction] += 1
    return out


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def metric_bars(system, friction, setting, metric, fis_value, baselines=None):
    """One sorted vertical bar chart for a single metric.

    Replaces the dual-axis chart, which packed RMSE and R^2 onto one figure with
    two y-scales and rotated in-bar labels. Splitting them lets each use its own
    axis and sort order, so the ranking is readable directly off the bar heights:
    RMSE ascending, R^2 descending, best on the left either way.

    Models the paper never ran carry no value to sort, so they are grouped at the
    right under a hatched "not run in paper" marker rather than dropped.
    """
    assert metric in ("rmse", "r2"), metric
    rows = pr.as_rows(system, friction, setting)

    idx = 0 if metric == "rmse" else 1
    scored, missing = [], []
    for name, rmse, r2 in rows:
        value = (rmse, r2)[idx]
        (missing if value is None else scored).append((name, value, "paper"))
    scored.append((OURS.replace("\n", " "), fis_value, "fis"))
    for blabel, pair in (baselines or {}).items():
        scored.append((blabel.replace("\n", " "), pair[idx], "baseline"))

    scored.sort(key=lambda e: e[1], reverse=(metric == "r2"))
    labels = [e[0] for e in scored] + [m[0] for m in missing]
    values = [e[1] for e in scored]
    kinds = [e[2] for e in scored]

    face = {
        "paper": BLUE if metric == "rmse" else ORANGE,
        "fis": VIOLET,
        "baseline": AQUA,
    }
    fig, ax = plt.subplots(figsize=(0.72 * len(labels) + 2.0, 4.2))
    fig.patch.set_facecolor(SURFACE)

    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
    pad = max(hi - lo, abs(hi), 1e-9) * 0.16
    if metric == "rmse":
        bottom, top = 0.0, hi + pad
    else:
        # R^2 clusters near 1 on the friction cells; a 0-1 axis would make every
        # bar look identical. Zoom to the data and keep 0 in view when it is close.
        bottom, top = min(lo - pad, 0.0) if lo < 0.25 else lo - pad, min(hi + pad, 1.04)

    x = np.arange(len(labels), dtype=float)
    for i, (v, k) in enumerate(zip(values, kinds)):
        ax.bar(
            x[i],
            v - bottom,
            0.68,
            bottom=bottom,
            color=face[k],
            edgecolor=INK if k != "paper" else "none",
            linewidth=1.0 if k != "paper" else 0,
            hatch="..." if k == "baseline" else None,
        )
        off = (top - bottom) * 0.015
        above = v + off if v >= bottom else bottom + off
        ax.text(
            x[i],
            above,
            f"{v:.4g}",
            ha="center",
            va="bottom",
            fontsize=FS_SMALL,
            color=INK_2,
            rotation=90,
        )
    for j in range(len(missing)):
        i = len(values) + j
        ax.bar(
            x[i],
            top - bottom,
            0.68,
            bottom=bottom,
            color="none",
            edgecolor=FAINT,
            linewidth=0.9,
            hatch="///",
        )
        ax.text(
            x[i],
            bottom + (top - bottom) * 0.5,
            "not run\nin paper",
            ha="center",
            va="center",
            fontsize=FS_SMALL,
            color=FAINT,
            rotation=90,
        )

    sysname = system.capitalize()
    fric = "with friction" if friction else "frictionless"
    which = (
        "trained initial condition"
        if setting == "trained"
        else 'unknown "in-between" initial condition'
    )
    what = (
        "RMSE — scaled units, lower is better"
        if metric == "rmse"
        else "$R^2$ — higher is better"
    )
    _style(ax, title=f"{sysname} pendulum, {fric}\n{which}", ylabel=what)
    ax.set_ylim(bottom, top)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK, color=INK_2, rotation=30, ha="right")
    if metric == "r2" and bottom < 0 < top:
        ax.axhline(0.0, color=AXIS, linewidth=0.9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=face["paper"]),
        plt.Rectangle((0, 0), 1, 1, facecolor=VIOLET, edgecolor=INK),
    ]
    names = ["paper", "FIS (ours)"]
    if baselines:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=AQUA, edgecolor=INK, hatch="...")
        )
        names.append("no-learning baseline")
    ax.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        fontsize=FS_SMALL,
        frameon=False,
        ncol=len(names),
    )

    fr = "friction" if friction else "frictionless"
    return _save(fig, f"fig_{metric}_{system}_{fr}_{setting}")


def compare_cell(system, friction, setting, fis_rmse, fis_r2, baselines=None):
    """Both single-metric charts for one cell. Returns (rmse path, r2 path)."""
    return (
        metric_bars(system, friction, setting, "rmse", fis_rmse, baselines),
        metric_bars(system, friction, setting, "r2", fis_r2, baselines),
    )


# ---------------------------------------------------------------------------
def angles_overlay(pred, system, friction, setting, extra=""):
    """theta_i(t) in degrees, ground truth vs FIS prediction.

    When the series runs past the training window, a vertical rule marks where the
    training data ends and the y-axis is scaled to the *truth*, not the prediction.
    Past the window the FIS diverges by orders of magnitude, so an axis sized to
    include it would compress the entire signal into a flat line; the divergence is
    annotated with its actual magnitude instead of being drawn to scale.
    """
    t = pred["t"]
    truth, fis = pred["truth_deg"], pred["pred_deg"]
    n = truth.shape[1]
    t_end = pred.get("train_t_end")

    fig, axes = plt.subplots(n, 1, figsize=(7.6, 2.05 * n), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        ax.plot(t, truth[:, j], color=BLUE, linewidth=1.5, label="actual (RK4)")
        ax.plot(
            t,
            fis[:, j],
            color=ORANGE,
            linewidth=1.2,
            linestyle="--",
            label="FIS prediction",
        )
        _style(ax, ylabel=rf"$\theta_{j + 1}$ (deg)")

        if t_end is not None:
            lo, hi = float(np.min(truth[:, j])), float(np.max(truth[:, j]))
            pad = max(hi - lo, 1.0) * 0.12
            ax.set_ylim(lo - pad, hi + pad)
            ax.axvline(t_end, color=INK, linewidth=1.1, linestyle=":")
            if j == 0:
                ax.text(
                    t_end,
                    hi + pad,
                    " training data ends",
                    ha="left",
                    va="top",
                    fontsize=FS_SMALL,
                    color=INK,
                )
            worst = float(np.max(np.abs(fis[:, j])))
            if worst > 3.0 * max(abs(lo), abs(hi), 1.0):
                # Solid background, not alpha: the diverging trace runs underneath
                # this label and would otherwise strike through it.
                ax.text(
                    0.99,
                    0.03,
                    f"prediction leaves the axis; peaks at {worst:.3g}°",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=FS_SMALL,
                    color=ORANGE,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
                )
        if j == 0:
            ax.legend(loc="upper left", fontsize=FS_SMALL, frameon=False, ncol=2)
    axes[-1].set_xlabel("t (seconds)", fontsize=FS_LABEL, color=INK)
    ic = ", ".join(f"{v:g}" for v in pred["ic_deg"])
    fric = "with friction" if friction else "frictionless"
    which = "trained IC" if setting == "trained" else "unknown IC"
    axes[0].set_title(
        f"{system.capitalize()} pendulum, {fric} — {which} [{ic}]°{extra}",
        fontsize=FS_TITLE,
        color=INK,
    )
    fig.tight_layout()
    return _save(
        fig,
        f"fig_angles_{system}_{'friction' if friction else 'frictionless'}_{setting}",
    )


# ---------------------------------------------------------------------------
def trajectory_overlay(pred, system, friction, setting):
    """Bob paths in the plane over 10 s: actual vs predicted.

    The paper's Figs. 14-16 and 19. Rods are drawn at the final instant so the
    configuration at t = 10 s is legible; the traces are the full 10 s of the
    outermost bob and every intermediate joint.
    """
    n = pred["truth_deg"].shape[1]
    lv = [1.0] * n
    xt, yt = chain_xy(np.deg2rad(pred["truth_deg"]), lv)
    xp, yp = chain_xy(np.deg2rad(pred["pred_deg"]), lv)
    t = pred["t"]
    t_end = pred.get("train_t_end")
    # Angles are periodic, so a prediction that has diverged to 1e5 degrees still
    # lands somewhere on the unit circle -- the bob path stays on-scale and looks
    # plausible while being meaningless. Draw only the in-window part solid, and
    # the extrapolated part faint, so the two are never confused.
    inw = np.ones(t.size, dtype=bool) if t_end is None else (t < t_end)
    t_last = t[inw][-1]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.patch.set_facecolor(SURFACE)
    for j in range(n):
        ax.plot(
            xt[inw, j],
            yt[inw, j],
            color=BLUE,
            linewidth=0.7,
            label="actual" if j == 0 else None,
        )
        ax.plot(
            xp[inw, j],
            yp[inw, j],
            color=ORANGE,
            linewidth=0.7,
            label="FIS prediction" if j == 0 else None,
        )
        if not inw.all():
            ax.plot(
                xt[~inw, j],
                yt[~inw, j],
                color=tint(BLUE, 0.55),
                linewidth=0.6,
                label="actual, past training window" if j == 0 else None,
            )
            ax.plot(
                xp[~inw, j],
                yp[~inw, j],
                color=tint(ORANGE, 0.55),
                linewidth=0.6,
                label="FIS, past training window" if j == 0 else None,
            )
    end = np.flatnonzero(inw)[-1]
    ax.plot(
        np.r_[0, xt[end]],
        np.r_[0, yt[end]],
        "-o",
        color=BLUE,
        linewidth=2.2,
        markersize=5,
        label=f"actual at t={t_last:.0f} s",
    )
    ax.plot(
        np.r_[0, xp[end]],
        np.r_[0, yp[end]],
        "-o",
        color=ORANGE,
        linewidth=2.2,
        markersize=5,
        label=f"predicted at t={t_last:.0f} s",
    )
    ax.plot([0], [0], marker="x", color=INK, markersize=7)

    lim = n + 0.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ic = ", ".join(f"{v:g}" for v in pred["ic_deg"])
    fric = "with friction" if friction else "frictionless"
    which = "trained IC" if setting == "trained" else "unknown IC"
    span = f"{t[-1] + (t[1] - t[0]):.0f} s"
    _style(
        ax,
        title=f"{system.capitalize()} pendulum, {fric}\n{which} [{ic}]°, {span} of trajectory",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    ax.legend(loc="upper left", fontsize=FS_SMALL, frameon=False)
    return _save(
        fig,
        f"fig_trajectory_{system}_{'friction' if friction else 'frictionless'}_{setting}",
    )


# ---------------------------------------------------------------------------
def error_vs_time(preds, t_end, threshold=0.10):
    """Absolute prediction error against time, for every dataset, with the
    training-window edge marked.

    This is the direct answer to "how far does it generalize": error is plotted on
    a log axis in the target's own scaled units, so the 10% band is a horizontal
    line and the point each curve crosses it is legible. preds maps dataset label
    -> the dict returned by fis_timestep.predictions_for.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.patch.set_facecolor(SURFACE)
    colors = regime_colors(preds)
    for label, p in sorted(preds.items()):
        err = np.max(np.abs(p["pred_scaled"] - p["truth_scaled"]), axis=1)
        ax.semilogy(
            p["t"],
            np.maximum(err, 1e-6),
            color=colors[label],
            linewidth=1.2,
            label=label.replace("_", " "),
        )
    ax.axvline(t_end, color=INK, linewidth=1.2, linestyle=":")
    ax.axhline(threshold, color=FAINT, linewidth=1.0, linestyle="--")
    ax.text(
        t_end,
        ax.get_ylim()[1],
        " training data ends",
        ha="left",
        va="top",
        fontsize=FS_SMALL,
        color=INK,
    )
    # Right edge with a solid background: at the left the in-window traces cross
    # this level repeatedly and strike the label through.
    ax.text(
        0.985,
        threshold,
        f"{threshold:.0%} of the training-window range ",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=FS_SMALL,
        color=FAINT,
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
    )
    _style(
        ax,
        title="Prediction error against time, held-out initial condition",
        xlabel="t (seconds)",
        ylabel="max abs error (scaled units, log)",
    )
    ax.legend(
        fontsize=FS_SMALL,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
    )
    return _save(fig, "fig_error_vs_time")


def rmse_heatmap(fis_rmse, setting="holdout", friction=True, systems=None):
    """Fig. 22-style model x system RMSE heatmap, with the FIS row appended.

    fis_rmse : {system name: value}. Columns follow `systems`, defaulting to the
    keys of fis_rmse, so adding a chain length adds a column without a code change.
    Cells the paper never ran are drawn "n/a" rather than left blank.
    """
    systems = list(systems if systems is not None else fis_rmse)
    labels = pr.MODEL_ORDER + ["FIS (ours)"]
    M = np.full((len(labels), len(systems)), np.nan)
    for cj, sysname in enumerate(systems):
        cell = pr.RESULTS[(sysname, friction, setting)]
        for ri, m in enumerate(pr.MODEL_ORDER):
            if cell.get(m):
                M[ri, cj] = cell[m][0]
        M[-1, cj] = fis_rmse.get(sysname, np.nan)

    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(M, cmap="magma_r", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize=FS_SMALL,
                    color=FAINT,
                )
                continue
            norm = (M[i, j] - np.nanmin(M)) / max(np.nanmax(M) - np.nanmin(M), 1e-12)
            ax.text(
                j,
                i,
                f"{M[i, j]:.4g}",
                ha="center",
                va="center",
                fontsize=FS_SMALL,
                color="white" if norm > 0.55 else INK,
            )
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([s.capitalize() for s in systems], fontsize=FS_TICK, color=INK_2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=FS_TICK, color=INK_2)
    ax.set_title(
        f"RMSE, time-step approach\n{'friction' if friction else 'frictionless'}, "
        f"{'unknown' if setting == 'holdout' else 'trained'} initial angle",
        fontsize=FS_TITLE,
        color=INK,
    )
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045)
    cb.set_label("RMSE (scaled units)", fontsize=FS_LABEL, color=INK)
    cb.ax.tick_params(labelsize=FS_TICK, colors=INK_2)
    return _save(
        fig, f"fig_rmse_heatmap_{'friction' if friction else 'frictionless'}_{setting}"
    )


# ---------------------------------------------------------------------------
def capacity_curve(curves, best_paper=None):
    """Held-out-IC R^2 against rule count, per dataset.

    Not a figure in the paper. It is here because it carries the reproduction's
    central result: on the friction problems more rules keep helping all the way
    to the grid ceiling, while on the frictionless problems held-out score
    saturates within a handful of rules and then drifts down as the fit to the
    training initial conditions tightens. curves maps label -> (rules, r2).
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    fig.patch.set_facecolor(SURFACE)
    colors = regime_colors(curves)
    for label, (rules, r2) in sorted(curves.items()):
        order = np.argsort(rules)
        ax.plot(
            np.asarray(rules)[order],
            np.asarray(r2)[order],
            "-o",
            color=colors.get(label, FAINT),
            markersize=4,
            linewidth=1.4,
            label=label.replace("_", " "),
        )
    if best_paper is not None:
        ax.axhline(best_paper, color=FAINT, linestyle=":", linewidth=1.2)
        # Name the cell: with three chain lengths on the axes, an unqualified
        # "paper best" reads as if it applied to all of them, and the paper has no
        # n=5 result at all.
        ax.text(
            ax.get_xlim()[1],
            best_paper,
            " paper best (n=2, friction)",
            va="center",
            fontsize=FS_SMALL,
            color=FAINT,
        )
    _style(
        ax,
        title="Held-out initial condition: score against FIS capacity",
        xlabel="rules per output (n_output_buckets)",
        ylabel="$R^2$ on unknown IC",
    )
    # Legend below the axes: with six curves the low-left and low-right corners are
    # both occupied, and an inset legend covered the n=5 frictionless trace.
    ax.legend(
        fontsize=FS_SMALL,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
    )
    return _save(fig, "fig_capacity_vs_holdout")
