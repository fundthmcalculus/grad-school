#!/usr/bin/env python3
"""Figure 4.2 -- output partitioning: the zeroth-order cliff, and quantile's instability.

§4.3.2 settles Goal G5 with two tables. Table 4.2 shows the three partitioning
schemes are a null result at first and second order and decisive at zeroth,
where the pinned-extreme arm goes negative; Table 4.3 shows that under
increasing target skew quantile partitioning does not become *inaccurate* so
much as *unstable*, its seed spread exploding while uniform degrades smoothly.
Both are arguments about spreads as much as means, which a table of "mean ±
s.d." cells conveys slowly and a plot conveys at once.

Panel (a): Table 4.2's three-bucket block -- R² by consequent order for the
three schemes, error bars the ten-seed spread. The three arms sit on top of each
other at first and second order and fan apart at zeroth. Panel (b): Table 4.3
-- uniform against quantile as skew rises, on a symmetric-log axis because
quantile's mean falls to -12.6 with a spread of ±24 and a linear axis would
show either the interesting region or the failure, not both.

Read from the archive's own CSVs (`table_g5_output_partitioning.csv`,
`table_g5b_skew_sweep.csv`), so the figure and the tables cannot disagree.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "04-output-partitioning"

SCHEMES = (
    ("uniform", "uniform (equal width)", F.BLUE),
    ("quantile", "quantile (equal frequency)", F.ORANGE),
    ("hybrid", "quantile + pinned extremes", F.FAINT),
)
ORDERS = ("0th", "1st", "2nd")


def build():
    part, label = H.table("table_g5_output_partitioning")
    skew, _ = H.table("table_g5b_skew_sweep")

    fig, (pa, pb) = F.grid_figure(1, 2, width=F.W_WIDE, height=3.4)

    # -- (a) three buckets, R² by order and scheme -------------------------------
    offsets = (-0.18, 0.0, 0.18)
    for (key, name, colour), dx in zip(SCHEMES, offsets):
        xs, ys, es = [], [], []
        for i, order in enumerate(ORDERS):
            for r in part:
                if (
                    H.number(r["buckets"]) == 3
                    and r["order"] == order
                    and key in r["scheme"]
                ):
                    xs.append(i + dx)
                    ys.append(H.number(r["R²"]))
                    es.append(H.spread(r["R²"]) or 0.0)
        pa.errorbar(
            xs,
            ys,
            yerr=es,
            fmt="o",
            ms=5,
            capsize=2.5,
            lw=1.4,
            color=colour,
            label=name,
            zorder=4,
        )
    pa.axhline(0, lw=0.8, color=F.AXIS, zorder=1)
    pa.set_xticks(range(len(ORDERS)))
    pa.set_xticklabels([f"{o} order" for o in ORDERS], fontsize=F.FS_TICK)
    F.style_axes(
        pa,
        title="(a)  Concrete, 3 buckets, by consequent order",
        ylabel="test $R^2$, mean ± s.d.",
        grid_axis="y",
    )
    pa.set_xlim(-0.5, len(ORDERS) - 0.5)
    F.legend(pa, loc="lower right")
    pa.text(
        0.55,
        0.15,
        "pinning the ends to the observed\nmin and max is fatal when the\nconstant is the whole output\n(Appendix A.10.11)",
        transform=pa.transData,
        ha="left",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.4,
    )

    # -- (b) skew sweep: means with spreads, symlog ---------------------------------
    sk = [H.number(r["target skew"]) for r in skew]
    for col, name, colour in (
        ("uniform R²", "uniform", F.BLUE),
        ("quantile R²", "quantile", F.ORANGE),
    ):
        ys = [H.number(r[col]) for r in skew]
        es = [H.spread(r[col]) or 0.0 for r in skew]
        pb.errorbar(
            sk,
            ys,
            yerr=es,
            marker="o",
            ms=4.5,
            lw=1.6,
            capsize=2.5,
            color=colour,
            label=name,
            zorder=4,
        )
    pb.axhline(0, lw=0.8, color=F.AXIS, zorder=1)
    pb.set_yscale("symlog", linthresh=1.0)
    pb.set_yticks([-30, -10, -3, -1, 0, 1])
    pb.set_yticklabels(["−30", "−10", "−3", "−1", "0", "1"], fontsize=F.FS_TICK)
    F.style_axes(
        pb,
        title="(b)  synthetic skew sweep, 4 buckets, 2nd order",
        xlabel="target skew",
        ylabel="test $R^2$, mean ± s.d.  (symlog)",
    )
    F.legend(pb, loc="lower left")
    pb.text(
        0.98,
        0.95,
        "quantile's mean trails a little;\nits spread explodes — a few\ncatastrophic seeds, not a trend",
        transform=pb.transAxes,
        ha="right",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.4,
    )

    fig.text(
        0.01,
        -0.02,
        "Tables 4.2 and 4.3 plotted from their own CSVs; ten seeds throughout. "
        f"{H.provenance_note(label)}",
        ha="left",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
