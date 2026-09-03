#!/usr/bin/env python3
"""Figure 4.7 -- the membership-function deduplication sweep, per problem.

Table 4.8 reports, for six problems, the reduction at the shipped tolerance and
the largest "lossless" multiplier -- the first tolerance at which the paired
accuracy or R² delta's 95% confidence interval excludes zero. Its footnote
warns that the tails are non-monotone and that the boundary is a property of
the problem, not a constant. Fourteen multipliers times six problems is too
much for the table and exactly right for six small panels.

Each panel: the paired delta (dedup minus raw, same seed and split) against the
tolerance multiplier on a log axis, mean as a line, the ten-seed spread as a
band, and the shipped 1x marked. The first multiplier whose CI excludes zero --
the table's own "CI excludes zero" column, not a re-computation -- is drawn as a
vertical rule; everything left of it is the lossless region Table 4.8's
"max-lossless x" names. Where a later multiplier dips back inside the band the
panel shows it, which is the non-monotonicity the caption is about.

Read from the archive's own `table_4_8_mf_dedup_sweep.csv`; pinned to the run
Table 4.8 quotes.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "04-mf-dedup-sweep"


def build():
    rows, label = H.table("table_4_8_mf_dedup_sweep")
    datasets = []
    for r in rows:
        if r["Dataset"] not in datasets:
            datasets.append(r["Dataset"])

    fig, axes = F.grid_figure(2, 3, width=F.W_WIDE, height=4.6)
    for ax, ds in zip(axes.ravel(), datasets):
        sub = [r for r in rows if r["Dataset"] == ds]
        mult = [H.number(r["Multiplier"]) for r in sub]
        mean = [H.number(r["Delta (mean±std)"]) for r in sub]
        sd = [H.spread(r["Delta (mean±std)"]) or 0.0 for r in sub]
        ax.fill_between(
            mult,
            [m - s for m, s in zip(mean, sd)],
            [m + s for m, s in zip(mean, sd)],
            color=F.tint(F.BLUE, 0.85),
            linewidth=0,
            zorder=1,
        )
        ax.plot(mult, mean, marker="o", ms=3, lw=1.4, color=F.BLUE, zorder=4)
        ax.axhline(0, lw=0.8, color=F.AXIS, zorder=2)
        ax.axvline(1.0, lw=0.8, ls=(0, (2, 2)), color=F.FAINT, zorder=2)
        breaks = [
            m
            for m, r in zip(mult, sub)
            if r["CI excludes zero"].strip().lower() == "yes"
        ]
        if breaks:
            first = min(breaks)
            ax.axvline(first, lw=1.2, color=F.ORANGE, zorder=3)
            ax.axvspan(min(mult), first, color=F.tint(F.AQUA, 0.93), zorder=0)
            note = f"first break at {first:g}×"
        else:
            ax.axvspan(min(mult), max(mult), color=F.tint(F.AQUA, 0.93), zorder=0)
            note = "no break in the sweep"
        raw = H.number(sub[0]["Raw MF"])
        at_one = next(
            (
                H.number(r["Dedup MF (mean±std)"])
                for r in sub
                if H.number(r["Multiplier"]) == 1.0
            ),
            None,
        )
        red = (
            f"{100 * (1 - at_one / raw):.0f}% fewer MFs at 1×" if at_one and raw else ""
        )
        ax.set_xscale("log")
        F.style_axes(
            ax,
            title=f"{ds} ({sub[0]['Task']})",
            xlabel="tolerance multiplier (1× = shipped)" if ax in axes[1] else None,
            ylabel="paired Δ, mean ± s.d." if ax in axes[:, 0] else None,
        )
        ax.text(
            0.03,
            0.05,
            f"{note}\n{red}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=F.FS_SMALL - 0.5,
            color=F.MUTED,
            linespacing=1.3,
        )

    fig.text(
        0.5,
        -0.01,
        "Shaded aqua is the lossless region: every multiplier before the first one whose 95% CI excludes zero "
        "(orange rule, the table's own column). The shipped 1× (dotted) never\ncosts anything measurable; where the "
        f"boundary sits is a property of the problem, from 2× to 10×, and some tails re-enter the band later. "
        f"{H.provenance_note(label)}",
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
