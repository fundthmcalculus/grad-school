"""Every arm and LKH as time-quality curves, plus how the picture scales with n.

The point the top row has to make is not "who is better" — at matched time above its floor LKH
wins outright and reaches the published optimum — but that the solvers occupy *different parts
of the time axis*, and that the split moves with instance size.

The gap axis is linear, not log, and that is not a stylistic choice: LKH's gap is exactly 0 on
these instances, log(0) is -inf, and matplotlib drops such points *silently* — producing a clean
figure with the competitor absent from it. That is the worst available failure mode for a
comparison plot, and worth a comment so it is not reintroduced by someone reaching for a log
axis to spread out our own curve.

LKH through ``elkai`` takes a run count rather than a time limit, so it has a hard floor: the
cost of one full run, below which it produces nothing at all. That floor is drawn as a shaded
region. Every point of ours inside it is non-dominated *by construction*, because there is no
LKH tour to compare against at that budget — which is a weaker thing than winning, and the
figure labels it as such.

The bottom row is the scaling question, which one panel per instance cannot answer:

* **left** — LKH's floor and our time-to-best against n, on log-log axes. Whether the window
  below LKH's floor is a footnote or the result depends entirely on whether it widens with n,
  and two lines diverging on a log-log plot is what that looks like.
* **right** — the iterated 2x2 read at one common budget per instance, which isolates what the
  fuzzy engine contributes once perturbation is in play: identical loop, identical seed and
  budgets, and the only two things that vary are whether ``EFFORT`` aims the kicks and whether
  ``CHAIN`` sets each re-optimisation's depth. The common budget is the control's dearest
  point, because comparing each arm at its *own* best budget compares different amounts of
  spending — which is the confound the factorial exists to remove.

Run:  python figures_lkh.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import paths  # noqa: E402
from lkh_compare import report, scaling  # noqa: E402

# One colour per arm, used identically in both rows so a reader can carry the legend down.
# The 2x2 shares a warm family so it reads as one factorial rather than four unrelated lines.
STYLE = {
    "sweep": ("tab:blue", "o", "fixed-parameter LK sweep"),
    "fis_ls": ("tab:purple", "D", "FIS local search (EFFORT+CHAIN)"),
    "iterated": ("0.35", "s", "iterated: neither (control)"),
    "iterated_aim": ("tab:orange", "v", "iterated: EFFORT aims the kicks"),
    "iterated_chain": ("tab:brown", "^", "iterated: CHAIN sets reopt depth"),
    "iterated_fis": ("tab:red", "P", "iterated: both"),
    "lkh": ("tab:green", "*", "LKH (elkai), by run count"),
}

IT_KEYS = ("iterated", "iterated_aim", "iterated_chain", "iterated_fis")


def _front(points):
    """Lower-left staircase of (time, gap), ascending in time."""
    out, best = [], float("inf")
    for t, g in sorted(points):
        if g < best - 1e-12:
            out.append((t, g))
            best = g
    return out


def _arm(d, key):
    if key == "fis_ls":
        p = d.get("fis_ls")
        return [(p["s"], p["gap"])] if p else []
    if key == "sweep":
        return [(r["s"], r["gap"]) for r in d.get("sweep", [])]
    if key == "lkh":
        return [
            (r["s"], r["gap"]) for r in d.get("lkh", []) if r.get("gap") is not None
        ]
    return [(r["s"], r["gap"]) for r in d.get(key, []) if r.get("gap") is not None]


def _panel(ax, name, d):
    colour, marker, label = STYLE["sweep"]
    sweep = _arm(d, "sweep")
    if sweep:
        ax.scatter(
            *zip(*sweep),
            marker=marker,
            s=45,
            facecolors="none",
            edgecolors=colour,
            label=label,
            zorder=3,
        )

    for key in IT_KEYS:
        pts = _arm(d, key)
        if pts:
            colour, marker, label = STYLE[key]
            ax.plot(
                *zip(*_front(pts)),
                marker + "-",
                color=colour,
                ms=5,
                lw=1.6,
                label=label,
                zorder=4,
            )

    pts = _arm(d, "fis_ls")
    if pts:
        colour, marker, label = STYLE["fis_ls"]
        ax.scatter(*zip(*pts), marker=marker, s=70, color=colour, label=label, zorder=6)

    lkh = _arm(d, "lkh")
    if lkh:
        colour, marker, label = STYLE["lkh"]
        ax.plot(
            *zip(*sorted(lkh)),
            marker + "-",
            color=colour,
            ms=8,
            lw=1.6,
            label=label,
            zorder=5,
        )
        floor = min(t for t, _ in lkh)
        left = ax.get_xlim()[0]
        # Deliberately faint. This region is most of the panel — every point of ours is inside
        # it — so shading it at any weight that reads as "highlighted" would say the opposite of
        # what it means. It marks where there is no opponent, not where we are winning. The
        # label sits at the bottom, away from the legend and away from the curves.
        ax.axvspan(
            left if left > 0 else 1e-3, floor, color=colour, alpha=0.045, zorder=0
        )
        ax.axvline(floor, color=colour, ls="--", lw=1.2, alpha=0.7)
        # Anchored in axes fraction on y, because ``set_ylim`` below moves the data limits and
        # a data-coordinate annotation placed here would silently fall off the panel.
        ax.annotate(
            f"← LKH returns nothing below {floor:.1f}s",
            xy=(floor, 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color=colour,
        )
        if min(g for _, g in lkh) <= 1e-9:
            ax.annotate(
                "LKH: exactly optimal",
                xy=(max(t for t, _ in lkh), 0.10),
                xycoords=("data", "axes fraction"),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=colour,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "LKH did not finish",
            transform=ax.transAxes,
            ha="center",
            color=STYLE["lkh"][0],
            fontsize=9,
        )

    ax.set_xscale("log")
    # Linear in gap, deliberately — see the module docstring.
    ax.set_ylim(bottom=-0.15)
    ax.set_xlabel("wall clock (s, log)")
    ax.set_ylabel("% over published optimum")
    ax.set_title(f"{name}  (n={d.get('n', '?')})", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper right")


def _scaling_panels(ax_time, ax_gap, rows):
    ns = [r["n"] for r in rows]

    floors = [(r["n"], r["lkh_floor"]) for r in rows if r.get("lkh_floor")]
    if floors:
        ax_time.plot(
            *zip(*floors),
            "^-",
            color=STYLE["lkh"][0],
            ms=8,
            lw=1.8,
            label="LKH's cheapest available budget",
        )
    for key in ("iterated", "iterated_fis"):
        pts = [
            (r["n"], r[f"{key}_s_at_best"]) for r in rows if r.get(f"{key}_s_at_best")
        ]
        if pts:
            colour, marker, label = STYLE[key]
            ax_time.plot(
                *zip(*pts),
                marker + "-",
                color=colour,
                ms=5,
                lw=1.6,
                label=f"time to best tour — {label.split(': ')[-1]}",
            )
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("n (log)")
    ax_time.set_ylabel("wall clock (s, log)")
    ax_time.set_title("The window below LKH's floor, against n", fontsize=11)
    ax_time.grid(alpha=0.3, which="both")
    ax_time.legend(fontsize=7, loc="upper left")

    # The 2x2 read at one common budget per instance — the control's dearest point. Comparing
    # each arm at its *own* best budget would compare different amounts of spending, which is
    # exactly the confound these arms exist to avoid.
    for key in IT_KEYS:
        pts = [
            (r["n"], r[f"{key}_at_budget"])
            for r in rows
            if r.get(f"{key}_at_budget") is not None
        ]
        if pts:
            colour, marker, label = STYLE[key]
            ax_gap.plot(
                *zip(*pts),
                marker + "-",
                color=colour,
                ms=5,
                lw=1.6,
                label=label.replace("iterated: ", ""),
            )
    ax_gap.set_xscale("log")
    ax_gap.set_xlabel("n (log)")
    ax_gap.set_ylabel("% over optimum at the matched budget")
    ax_gap.set_title("The iterated 2x2 at a matched budget, against n", fontsize=11)
    ax_gap.set_ylim(bottom=-0.05)
    ax_gap.grid(alpha=0.3, which="both")
    ax_gap.legend(fontsize=7, loc="upper left", title="FIS role", title_fontsize=7)
    return ns


def figure(data, out):
    rows = scaling(data)
    names = sorted(data, key=lambda k: data[k].get("n", 0))
    # Wrapped at four columns rather than one row per instance. A seven-instance ladder in a
    # single row is 4000 px wide, which every viewer scales down until the axes are unreadable —
    # the ladder is the point of the figure, so it has to survive being long.
    ncol = min(4, max(len(names), 2))
    nrow = -(-len(names) // ncol)  # ceiling division
    fig = plt.figure(figsize=(5.6 * ncol, 4.6 * nrow + 4.8))
    gs = fig.add_gridspec(
        nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.95], hspace=0.34, wspace=0.26
    )
    for i, name in enumerate(names):
        _panel(fig.add_subplot(gs[i // ncol, i % ncol]), name, data[name])
    half = max(ncol // 2, 1)
    _scaling_panels(
        fig.add_subplot(gs[nrow, :half]), fig.add_subplot(gs[nrow, half:]), rows
    )

    fig.suptitle(
        "The solvers occupy different parts of the time axis, and the split moves with n",
        fontsize=12,
    )
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(paths.LKH_COMPARE))
    ap.add_argument("--out", default=str(paths.FIGURES / "fis_tsp_vs_lkh.png"))
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text())
    paths.ensure()
    print(f"wrote {figure(data, args.out)}")
    report(data)


if __name__ == "__main__":
    main()
