#!/usr/bin/env python3
"""Figure 3.5 -- the three reorder arms in absolute seconds, and stage two's margin.

Table 3.2 and Figure 3.4 normalize every arm to its own value at the smallest N,
which is the right way to read an exponent and the wrong way to read a cost: a
normalized curve cannot say that stage two at N = 3,000 takes single-digit
milliseconds, or that the classical arm was capped at N = 1,000 because the
next size would have taken most of an hour. Appendix A.2.4 carries the seconds
as a table; this figure draws them.

Panel (a): mean seconds per arm against N on log-log axes, with the per-seed
spread as error bars (mostly smaller than the marker) and the classical arm
ending where the harness caps it. Panel (b): the stage-one / stage-two ratio
the table's `s1/s2` column reports, size by size -- the margin §3.4 quotes as a
range rather than a number, because it depends on N and on the host.

Read from the archive's own `table_3_1_three_arm.csv`; the label printed in the
corner is the run the figure drew from. Cells that read "not run (> cap)" are
skipped, not interpolated.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "03-three-arm-seconds"

ARMS = (
    ("classical O(N³) (s)", "classical, $O(N^3)$", F.ORANGE),
    ("stage 1 O(N²logN) (s)", "stage one, $O(N^2 \\log N)$", F.BLUE),
    ("stage 2 O(N²) (s)", "stage two, $O(N^2)$", F.AQUA),
)


def build():
    rows, label = H.table("table_3_1_three_arm")
    n = [H.number(r["N"]) for r in rows]

    fig, (pa, pb) = F.grid_figure(
        1, 2, width=F.W_WIDE, height=3.3, gridspec_kw={"width_ratios": [1.5, 1]}
    )

    for col, name, colour in ARMS:
        xs, ys, es = [], [], []
        for nn, r in zip(n, rows):
            v = H.number(r[col]) if "not run" not in r[col] else None
            if v is None or v <= 0:
                continue
            xs.append(nn)
            ys.append(v)
            es.append(H.spread(r[col]) or 0.0)
        pa.errorbar(
            xs,
            ys,
            yerr=es,
            marker="o",
            ms=4,
            lw=1.6,
            capsize=2,
            color=colour,
            label=name,
            zorder=4,
        )
        if col.startswith("classical"):
            pa.axvline(max(xs), lw=0.8, ls=(0, (2, 2)), color=F.FAINT, zorder=1)
            pa.text(
                max(xs) * 1.04,
                min(ys) * 1.3,
                "classical arm capped here\n(REPRO_NAIVE_CAP): the next\nsize is cubic, hours of clock",
                ha="left",
                va="bottom",
                fontsize=F.FS_SMALL,
                color=F.MUTED,
                linespacing=1.4,
            )
    F.style_axes(
        pa,
        title="(a)  reorder time per arm",
        xlabel="$N$ (points)",
        ylabel="seconds, mean ± s.d.",
    )
    pa.set_xscale("log")
    pa.set_yscale("log")
    F.legend(pa, loc="upper left")

    ratio = [
        (nn, H.number(r["s1/s2"])) for nn, r in zip(n, rows) if H.number(r["s1/s2"])
    ]
    pb.plot(
        [a for a, _ in ratio],
        [b for _, b in ratio],
        marker="o",
        ms=4,
        lw=1.6,
        color=F.AQUA,
        zorder=4,
    )
    lo, hi = min(b for _, b in ratio), max(b for _, b in ratio)
    pb.axhspan(lo, hi, color=F.tint(F.AQUA, 0.9), zorder=0)
    pb.text(
        0.03,
        0.95,
        f"stage one / stage two:\n{lo:.1f}× to {hi:.1f}× across this grid",
        transform=pb.transAxes,
        ha="left",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.shade(F.AQUA, 0.3),
        linespacing=1.5,
    )
    F.style_axes(
        pb,
        title="(b)  stage two's margin over stage one",
        xlabel="$N$ (points)",
        ylabel="ratio of mean times",
    )
    pb.set_xscale("log")
    pb.set_ylim(0, hi * 1.35)

    fig.text(
        0.01,
        -0.02,
        "Same sweep as Table 3.2 and Figure 3.4, in seconds rather than normalized; ten seeds. "
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
