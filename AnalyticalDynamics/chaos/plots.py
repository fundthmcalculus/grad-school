"""Comparison figures for the arXiv:2504.13453 reproduction.

Mirrors the paper's own figure formats:

  fig_compare_*        the dual-axis RMSE / R^2 grouped bar chart of Figs. 11,
                       12, 13, 18B-D, with our FIS added as a ninth model
  fig_angles_*         theta(t) truth vs prediction, the plot the reference
                       notebooks emit at the end of every model cell
  fig_trajectory_*     bob paths in the plane over 10 s, Figs. 14, 15, 16, 19
  fig_rmse_heatmap     the model x system RMSE heatmap of Fig. 22
  fig_capacity         not in the paper: held-out-IC score against rule count,
                       which is where the reproduction's main finding lives

No animations, by request.

Colours are categorical slots 1 and 2 of this repo's validated light-mode
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
def compare_bars(system, friction, setting, fis_rmse, fis_r2, name=None, baselines=None):
    """The paper's dual-axis grouped bar chart, with our FIS appended.

    Left axis / blue bars: RMSE (lower is better). Right axis / orange bars:
    R^2 (higher is better). Models the paper did not run in this cell are drawn
    as a hatched "not run" marker rather than omitted, so the gap is visible.

    baselines : optional {short label: (rmse, r2)} for the no-learning
        references. Drawn in a third colour so a reader can see at a glance
        whether the learned models beat "average the two nearest trajectories".
    """
    rows = pr.as_rows(system, friction, setting)
    labels = [m for m, _, _ in rows] + [OURS]
    rmse = [r for _, r, _ in rows] + [fis_rmse]
    r2 = [q for _, _, q in rows] + [fis_r2]
    n_learned = len(labels)
    for blabel, (brm, br2) in (baselines or {}).items():
        labels.append(blabel)
        rmse.append(brm)
        r2.append(br2)

    x = np.arange(len(labels), dtype=float)
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.6 + 0.55 * len(baselines or {}), 4.0))
    fig.patch.set_facecolor(SURFACE)
    ax2 = ax.twinx()

    finite_rmse = [v for v in rmse if v is not None]
    top = max(finite_rmse) * 1.28

    for i, (rv, qv) in enumerate(zip(rmse, r2)):
        is_ours = i == n_learned - 1
        is_base = i >= n_learned
        if rv is None:
            ax.bar(x[i] - w / 2, top * 0.97, w, color="none", edgecolor=FAINT,
                   linewidth=0.9, hatch="///")
            ax.text(x[i], top * 0.5, "not run\nin paper", ha="center", va="center",
                    fontsize=FS_SMALL, color=FAINT, rotation=90)
            continue
        rc = VIOLET if is_ours else (AQUA if is_base else BLUE)
        qc = YELLOW if is_ours else (AQUA if is_base else ORANGE)
        edge = INK if (is_ours or is_base) else "none"
        lw = 1.0 if (is_ours or is_base) else 0
        hatch = "..." if is_base else None
        ax.bar(x[i] - w / 2, rv, w, color=rc, edgecolor=edge, linewidth=lw, hatch=hatch)
        ax2.bar(x[i] + w / 2, qv, w, color=qc, edgecolor=edge, linewidth=lw, hatch=hatch)
        ax.text(x[i] - w / 2, rv + top * 0.015, f"{rv:.3g}", ha="center", va="bottom",
                fontsize=FS_SMALL, color=INK_2, rotation=90)
        # R^2 labels sit *inside* the bar: these bars nearly fill their axis, so
        # anything above the cap lands in the legend or off the figure.
        ax2.text(x[i] + w / 2, qv, f" {qv:.4f}", ha="center", va="top",
                 fontsize=FS_SMALL, color="white", rotation=90)

    sysname = system.capitalize()
    fric = "With Friction" if friction else "Frictionless"
    which = ("Trained Initial Condition" if setting == "trained"
             else 'Unknown "In-Between" Initial Condition')
    # No x label: the tick labels are the model names, and a label here would sit
    # under the legend.
    _style(ax, title=f"{sysname} Pendulum, {fric} — {which}",
           ylabel="RMSE (scaled units, lower is better)")
    ax.set_ylim(0, top)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK, color=INK_2)
    ax.yaxis.label.set_color(BLUE)

    ax2.set_ylabel("$R^2$ (higher is better)", fontsize=FS_LABEL, color=ORANGE)
    ax2.tick_params(colors=INK_2, labelsize=FS_TICK)
    lo = min([q for q in r2 if q is not None])
    ax2.set_ylim(min(0.0, lo - 0.02), 1.06)
    for side in ("top", "left"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color(AXIS)
    ax2.grid(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE),
        plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE),
        plt.Rectangle((0, 0), 1, 1, facecolor=VIOLET, edgecolor=INK),
        plt.Rectangle((0, 0), 1, 1, facecolor=YELLOW, edgecolor=INK),
    ]
    names = ["RMSE (paper)", "$R^2$ (paper)", "RMSE (FIS)", "$R^2$ (FIS)"]
    if baselines:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=AQUA, edgecolor=INK, hatch="..."))
        names.append("no-learning baseline")
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=FS_SMALL, frameon=False, ncol=len(names))

    name = name or f"fig_compare_{system}_{'friction' if friction else 'frictionless'}_{setting}"
    return _save(fig, name)


# ---------------------------------------------------------------------------
def angles_overlay(pred, system, friction, setting, extra=""):
    """theta_i(t) in degrees, ground truth vs FIS prediction."""
    t = pred["t"]
    truth, fis = pred["truth_deg"], pred["pred_deg"]
    n = truth.shape[1]

    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.05 * n), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        ax.plot(t, truth[:, j], color=BLUE, linewidth=1.5, label="actual (RK4)")
        ax.plot(t, fis[:, j], color=ORANGE, linewidth=1.2, linestyle="--",
                label="FIS prediction")
        _style(ax, ylabel=rf"$\theta_{j + 1}$ (deg)")
        if j == 0:
            ax.legend(loc="upper right", fontsize=FS_SMALL, frameon=False, ncol=2)
    axes[-1].set_xlabel("t (seconds)", fontsize=FS_LABEL, color=INK)
    ic = ", ".join(f"{v:g}" for v in pred["ic_deg"])
    fric = "with friction" if friction else "frictionless"
    which = "trained IC" if setting == "trained" else "unknown IC"
    axes[0].set_title(
        f"{system.capitalize()} pendulum, {fric} — {which} [{ic}]°{extra}",
        fontsize=FS_TITLE, color=INK,
    )
    fig.tight_layout()
    return _save(fig, f"fig_angles_{system}_{'friction' if friction else 'frictionless'}_{setting}")


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

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    fig.patch.set_facecolor(SURFACE)
    for j in range(n):
        ax.plot(xt[:, j], yt[:, j], color=BLUE, linewidth=0.7,
                label="actual" if j == 0 else None)
        ax.plot(xp[:, j], yp[:, j], color=ORANGE, linewidth=0.7,
                label="FIS prediction" if j == 0 else None)
    ax.plot(np.r_[0, xt[-1]], np.r_[0, yt[-1]], "-o", color=BLUE, linewidth=2.2,
            markersize=5, label="actual at t=10 s")
    ax.plot(np.r_[0, xp[-1]], np.r_[0, yp[-1]], "-o", color=ORANGE, linewidth=2.2,
            markersize=5, label="predicted at t=10 s")
    ax.plot([0], [0], marker="x", color=INK, markersize=7)

    lim = n + 0.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ic = ", ".join(f"{v:g}" for v in pred["ic_deg"])
    fric = "with friction" if friction else "frictionless"
    which = "trained IC" if setting == "trained" else "unknown IC"
    _style(ax, title=f"{system.capitalize()} pendulum, {fric}\n{which} [{ic}]°, 10 s of trajectory",
           xlabel="x (m)", ylabel="y (m)")
    ax.legend(loc="upper left", fontsize=FS_SMALL, frameon=False)
    return _save(fig, f"fig_trajectory_{system}_{'friction' if friction else 'frictionless'}_{setting}")


# ---------------------------------------------------------------------------
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
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=FS_SMALL, color=FAINT)
                continue
            norm = (M[i, j] - np.nanmin(M)) / max(np.nanmax(M) - np.nanmin(M), 1e-12)
            ax.text(j, i, f"{M[i, j]:.4g}", ha="center", va="center", fontsize=FS_SMALL,
                    color="white" if norm > 0.55 else INK)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([s.capitalize() for s in systems], fontsize=FS_TICK, color=INK_2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=FS_TICK, color=INK_2)
    ax.set_title(
        f"RMSE, time-step approach\n{'friction' if friction else 'frictionless'}, "
        f"{'unknown' if setting == 'holdout' else 'trained'} initial angle",
        fontsize=FS_TITLE, color=INK,
    )
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045)
    cb.set_label("RMSE (scaled units)", fontsize=FS_LABEL, color=INK)
    cb.ax.tick_params(labelsize=FS_TICK, colors=INK_2)
    return _save(fig, f"fig_rmse_heatmap_{'friction' if friction else 'frictionless'}_{setting}")


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
        ax.plot(np.asarray(rules)[order], np.asarray(r2)[order], "-o",
                color=colors.get(label, FAINT), markersize=4, linewidth=1.4,
                label=label.replace("_", " "))
    if best_paper is not None:
        ax.axhline(best_paper, color=FAINT, linestyle=":", linewidth=1.2)
        # Name the cell: with three chain lengths on the axes, an unqualified
        # "paper best" reads as if it applied to all of them, and the paper has no
        # n=5 result at all.
        ax.text(ax.get_xlim()[1], best_paper, " paper best (n=2, friction)", va="center",
                fontsize=FS_SMALL, color=FAINT)
    _style(ax, title="Held-out initial condition: score against FIS capacity",
           xlabel="rules per output (n_output_buckets)", ylabel="$R^2$ on unknown IC")
    # Legend below the axes: with six curves the low-left and low-right corners are
    # both occupied, and an inset legend covered the n=5 frictionless trace.
    ax.legend(fontsize=FS_SMALL, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=3)
    return _save(fig, "fig_capacity_vs_holdout")
