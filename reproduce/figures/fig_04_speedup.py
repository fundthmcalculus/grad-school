#!/usr/bin/env python3
"""Figure 4.5 -- training time per method and dataset, from Table 4.1b's own CSV.

Table 4.1b is the chapter's speed claim, and §4.4 insists it be read as a range
("14x to 194x, not a flat two orders") with two honesties attached: the random
forest is not slower than the construction on two of the rows, and Bike Sharing
is the weakest cell because the fuzzy baselines are cheapest there. A table
makes a reader compute those comparisons; grouped bars on a log axis show them.

One group per dataset, four arms: the construction, ANFIS, the GA-tuned FIS and
the random forest reference, mean seconds with the ten-seed spread as error bars,
and the table's own speedup column printed above each group. Read from
`table_4_1b_baseline_timing.csv` in the archive the prose cites for it, so the
bars and the table cannot disagree.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "04-speedup"

ARMS = (
    ("MoG train", "MoG FIS (this work)", F.BLUE),
    ("ANFIS train", "ANFIS", F.ORANGE),
    ("GA-FIS train", "GA-tuned FIS", F.AQUA),
    ("RF train", "Random Forest (reference)", F.FAINT),
)


def _short(name):
    """'Concrete (regression, full 2nd order)' -> 'Concrete\\nfull 2nd order'."""
    base, _, paren = name.partition(" (")
    paren = paren.rstrip(")")
    extras = [p.strip() for p in paren.split(",")[1:]]
    if "12-class" in paren:
        extras.append("12-class")
    return base + ("\n" + ", ".join(extras) if extras else "")


def build():
    rows, label = H.table("table_4_1b_baseline_timing")
    n = len(rows)
    width = 0.19
    x = np.arange(n)

    fig, ax = F.figure(width=F.W_WIDE, height=3.6)
    for k, (col, name, colour) in enumerate(ARMS):
        means = [H.number(r[col]) for r in rows]
        errs = [H.spread(r[col]) or 0.0 for r in rows]
        ax.bar(
            x + (k - 1.5) * width,
            means,
            width=width,
            yerr=errs,
            capsize=2,
            color=F.tint(colour, 0.45),
            edgecolor=colour,
            linewidth=0.9,
            error_kw=dict(lw=0.8, ecolor=F.INK_2),
            label=name,
            zorder=3,
        )
    for i, r in enumerate(rows):
        top = max(H.number(r[c]) + (H.spread(r[c]) or 0) for c, _, _ in ARMS)
        ax.text(
            i,
            top * 1.35,
            r["MoG speedup vs slowest fuzzy"],
            ha="center",
            va="bottom",
            fontsize=F.FS_ANNOT,
            color=F.shade(F.BLUE, 0.25),
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_short(r["Dataset (task)"]) for r in rows], fontsize=F.FS_TICK)
    ax.set_yscale("log")
    F.style_axes(
        ax,
        title="Training time by method, ten seeds, shared splits (Table 4.1b)",
        ylabel="seconds, mean ± s.d. (log)",
        grid_axis="y",
    )
    ax.set_ylim(0.05, 3000)
    F.legend(ax, loc="upper left", ncol=2)
    ax.text(
        0.0,
        -0.18,
        "Above each group: the construction's speedup over the slower of the two fuzzy baselines, the table's own "
        "column. The honest statement is the range,\none to two orders of magnitude against fuzzy-system induction; "
        "the random forest is not slower than the construction on Concrete or Bike Sharing,\nso the speed argument "
        f"is against ANFIS and GA-FIS, not against trees. {H.provenance_note(label)}",
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
