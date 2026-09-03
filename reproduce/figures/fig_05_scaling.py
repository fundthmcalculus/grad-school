#!/usr/bin/env python3
"""Figure 5.6 -- the two-stage selector against a flat set-cover as n grows (Table 5.4).

§5.4's scaling paragraph is the one place Chapter 5 reports a ten-seed sweep,
and it says three different things about three families: `many_scale` is
recovered exactly at every n while the flat baseline finds only the coarsest
level; `single_scale` is where the ten-seed floor exposes an instability in the
*reported granularity* that the partition's ARI hides; `log_separated` climbs
gradually with n rather than crossing a threshold. Three panels, one per family,
from the archive's own `table_5_4_ch5_g1_scaling.csv`.

Each panel: adjusted Rand index against n for the two-stage selector (blue) and
the flat set-cover (orange), the ten-seed spread as error bars where the table
carries one. For `many_scale` the flat column is per-level -- one partition
scored against three granularities -- so its three values are drawn as three
faint lines, and the two-stage column's three values coincide at 1.00. Under
each x tick the table's granularity-agreement fraction (how many seeds of ten
returned the modal granularity vector) is printed, which is the instability the
paragraph is about and which no ARI axis can show.

Read from the archive, pinned to the run the prose quotes; nothing here is typed.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "05-scaling"

FAMILIES = (
    ("many_scale", "many_scale: [8, 4, 2] nested"),
    ("single_scale", "single_scale: one level"),
    ("log_separated", "log_separated: log-spaced scales"),
)
_NUM = re.compile(r"([-+]?\d*\.?\d+)(?:\s*±\s*([-+]?\d*\.?\d+))?")


def _levels(cell):
    """'[0.24, 0.49, 1.00]' or '[0.98 ± 0.07]' -> [(mean, spread), ...]."""
    return [(float(m), float(s) if s else 0.0) for m, s in _NUM.findall(cell)]


def _agreement(cell):
    """'[5, 2] (6/10)' -> '6/10'."""
    m = re.search(r"\((\d+/\d+)\)", cell)
    return m.group(1) if m else ""


def build():
    rows, label = H.table("table_5_4_ch5_g1_scaling")
    fig, axes = F.grid_figure(1, 3, width=F.W_WIDE, height=3.1)

    for ax, (fam, title) in zip(axes, FAMILIES):
        sub = [r for r in rows if r["family"] == fam]
        n = [H.number(r["n"]) for r in sub]
        flat = [_levels(r["flat ARI/level"]) for r in sub]
        two = [_levels(r["two-stage ARI/level"]) for r in sub]
        n_levels = max(len(f) for f in flat)

        for lvl in range(n_levels):
            fl = [(f[lvl] if lvl < len(f) else (None, 0)) for f in flat]
            ys = [v[0] for v in fl]
            es = [v[1] for v in fl]
            ax.errorbar(
                n,
                ys,
                yerr=es,
                marker="s",
                ms=3.5,
                lw=1.3,
                capsize=2,
                color=F.ORANGE if n_levels == 1 else F.tint(F.ORANGE, 0.15 + 0.3 * lvl),
                label=(
                    "flat set-cover"
                    if n_levels == 1
                    else f"flat set-cover, scored at level {lvl + 1}"
                ),
                zorder=3,
            )
        # The two-stage column: mean over its levels when there are several
        # (they are all 1.00 on many_scale, so the mean loses nothing there).
        ys = [sum(v[0] for v in t) / len(t) for t in two]
        es = [max(v[1] for v in t) for t in two]
        ax.errorbar(
            n,
            ys,
            yerr=es,
            marker="o",
            ms=4.5,
            lw=1.8,
            capsize=2,
            color=F.BLUE,
            label="two-stage (bands, then cover)",
            zorder=4,
        )
        for nn, r in zip(n, sub):
            ag = _agreement(r["two-stage granularity (mode; agreement)"])
            ax.text(
                nn,
                0.02,
                ag,
                ha="center",
                va="bottom",
                fontsize=F.FS_SMALL - 1,
                color=F.MUTED,
                transform=ax.get_xaxis_transform(),
            )
        F.style_axes(
            ax,
            title=title,
            xlabel="$n$",
            ylabel="adjusted Rand index" if ax is axes[0] else None,
        )
        ax.set_xscale("log")
        ax.set_ylim(0, 1.08)
        # Legends sit in the band each panel leaves empty.
        spot = {"many_scale": (0.02, 0.60), "single_scale": (0.02, 0.62)}.get(fam)
        if spot:
            F.legend(
                ax, loc="lower left", bbox_to_anchor=spot, fontsize=F.FS_SMALL - 0.5
            )
        else:
            F.legend(
                ax,
                loc="lower right",
                bbox_to_anchor=(1.0, 0.10),
                fontsize=F.FS_SMALL - 0.5,
            )

    fig.text(
        0.5,
        -0.02,
        "Along the bottom of each panel: the fraction of ten seeds returning the modal granularity vector, the table's agreement "
        "column. On many_scale the two-stage selector recovers [8, 4, 2] at ARI 1.00\nat every $n$ while the flat cover "
        "lands only the coarsest level; on single_scale the partition is good while the reported granularity agrees "
        f"on 5–7 seeds of ten; log_separated climbs with $n$. {H.provenance_note(label)}",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
