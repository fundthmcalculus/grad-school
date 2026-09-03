#!/usr/bin/env python3
"""Figure 3.6 -- footprint against N for each memory scheme, and where the budgets bind.

Table 3.3 gives the reachable N per scheme and precision as a grid of six
numbers. The figure draws the curves those numbers sit on: F(N) = k s N^2 for a
scheme holding k matrices at s bytes per entry (Appendix A.10.5), against the
two budgets §3.4 quotes, so the ceilings are where each curve crosses a budget
line and the sqrt(3) and sqrt(2) gaps between them are visible as horizontal
offsets on the log axis.

The curves are arithmetic. The ceilings are read from the archive's own
`table_3_2_memory_precision.csv` and drawn as markers, so if the table and the
formula ever disagreed the marker would sit off its curve. The matrix-free
scheme has no N^2 term and is drawn as a note rather than a curve: its measured
flat footprint is a §3.3.2 result from a separate experiment
(`check_matrix_free_reorder.py`) and is not quoted here. The two demonstrations
§3.4 names -- the 58,000-point shuttle set and the 135,000-point float32 run --
are marked as problem sizes on the axis, not as measurements.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "03-memory-ceiling"

GB = 1e9
BUDGETS = ((64, "64 GB working cap"), (96, "96 GB machine"))
DEMOS = (
    (58_000, "shuttle, 58k\n(float64)"),
    (135_000, "135k run\n(float32, cap lifted)"),
)


def _k(scheme):
    if "classical" in scheme:
        return 3
    if "in-place" in scheme:
        return 1
    return None  # matrix-free: no N^2 term


def build():
    rows, label = H.table("table_3_2_memory_precision")
    N = np.geomspace(5_000, 250_000, 300)

    fig, ax = F.figure(width=F.W_COL + 1.2, height=4.2)
    styles = {"float64": "solid", "float32": (0, (4, 2))}
    colours = {3: F.ORANGE, 1: F.BLUE}
    for r in rows:
        k = _k(r["scheme"])
        if k is None:
            continue
        s = H.number(r["bytes/entry"])
        prec = r["precision"]
        ax.plot(
            N,
            k * s * N**2 / GB,
            lw=1.7,
            ls=styles[prec],
            color=colours[k],
            label=f"{'classical, 3 matrices' if k == 3 else 'in-place, 1 matrix'}, {prec}",
            zorder=3,
        )
        for (gb, _), col in zip(BUDGETS, ("largest N in 64 GB", "largest N in 96 GB")):
            n_max = H.number(r[col])
            if n_max:
                ax.plot(
                    [n_max],
                    [gb],
                    marker="o",
                    ms=4.5,
                    color=colours[k],
                    mec=F.SURFACE,
                    mew=0.6,
                    zorder=5,
                )

    for gb, text in BUDGETS:
        ax.axhline(gb, lw=0.9, ls=(0, (2, 2)), color=F.FAINT, zorder=2)
        ax.text(
            N[0] * 1.05,
            gb * 1.06,
            text,
            ha="left",
            va="bottom",
            fontsize=F.FS_SMALL,
            color=F.MUTED,
        )
    for n_demo, text in DEMOS:
        ax.axvline(n_demo, lw=0.8, color=F.GRID, zorder=1)
        ax.text(
            n_demo,
            0.62,
            text,
            ha="center",
            va="bottom",
            fontsize=F.FS_SMALL - 0.5,
            color=F.MUTED,
            linespacing=1.3,
        )

    F.style_axes(
        ax,
        title="Dense footprint $k\\,s\\,N^2$ against $N$; markers are Table 3.3's ceilings",
        xlabel="$N$ (points)",
        ylabel="gigabytes",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(N[0], N[-1])
    ax.set_ylim(0.5, 2500)
    F.legend(ax, loc="upper left", bbox_to_anchor=(0.0, 0.93))

    ax.text(
        0.0,
        -0.2,
        "Every curve is arithmetic; the markers are read from the archived table and sit where "
        "the formula puts them. Along a budget line the in-place\nscheme reaches $\\sqrt{3}$ "
        "further than the classical one and float32 another $\\sqrt{2}$ (Appendix A.10.5). The "
        "matrix-free path has no $N^2$ term at all and\nis off this chart — §3.3.2 reports its "
        f"flat, measured footprint separately. {H.provenance_note(label)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
