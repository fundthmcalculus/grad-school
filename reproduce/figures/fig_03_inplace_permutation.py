#!/usr/bin/env python3
"""Figure 3.2 -- the in-place permutation, and what it does to the memory budget.

§3.3.2's memory scheme rests on a 1977 trick (Cate and Twigg): a permutation
decomposes into disjoint cycles, and walking each cycle moves every element to
its final slot with one temporary, so the reordered matrix can overwrite the
original instead of being written beside it. The classical VAT keeps the
original, the reordered copy and a work matrix -- three N x N arrays -- and the
in-place scheme keeps one.

Panel (a) draws a small permutation as arrows between slots, one colour per
cycle, with the cycles listed underneath so the reader can trace one. Panel (b)
draws the consequence for a 64,000-point float64 problem: the three-matrix and
one-matrix footprints, each computed from k * s * N^2 (Appendix A.10.5) rather
than typed, against the two budgets §3.4 quotes. The sqrt(3) in the reachable
size is the ratio of those two bar heights, square-rooted.

The permutation is a fixed-seed random one for legibility; the footprints are
arithmetic, as Table 3.3 labels them. Nothing here is a measurement.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "03-inplace-permutation"

SEED = 4
N_SLOTS = 10
N_DEMO = 64_000  # the size Table 3.3 quotes its footprints at
BYTES = 8  # float64
GB = 1e9  # decimal gigabytes, as in Table 3.3
BUDGETS = ((64, "64 GB working cap"), (96, "96 GB machine"))


def _cycles(p):
    seen = set()
    out = []
    for start in range(len(p)):
        if start in seen:
            continue
        cyc, v = [], start
        while v not in seen:
            seen.add(v)
            cyc.append(v)
            v = int(p[v])
        if len(cyc) > 1:
            out.append(cyc)
    return out


def build():
    rng = np.random.RandomState(SEED)
    p = rng.permutation(N_SLOTS)
    while any(p == np.arange(N_SLOTS)) or len(_cycles(p)) < 2:
        p = rng.permutation(N_SLOTS)
    cycles = _cycles(p)

    fig, (pa, pb) = F.grid_figure(
        1, 2, width=F.W_WIDE, height=3.3, gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # -- (a) the permutation as cycles -------------------------------------------
    xs = np.arange(N_SLOTS) * 10.0
    colours = [F.BLUE, F.ORANGE, F.AQUA, F.VIOLET, F.MAGENTA]
    for k, cyc in enumerate(cycles):
        col = colours[k % len(colours)]
        for v in cyc:
            F.arrow(
                pa,
                (xs[v], 30),
                (xs[p[v]], 30),
                color=col,
                lw=1.3,
                connection=f"arc3,rad={0.35 if xs[p[v]] > xs[v] else -0.35}",
            )
    for v in range(N_SLOTS):
        F.box(
            pa,
            xs[v],
            30,
            7,
            9,
            str(v),
            color=F.FAINT,
            fill_amount=0.92,
            title_size=F.FS_ANNOT,
            title_weight="normal",
        )
    pa.text(-3, 30, "slot", ha="right", va="center", fontsize=F.FS_SMALL, color=F.MUTED)
    pa.text(
        xs.mean(),
        7,
        "walk each cycle once: save one element, shift the rest along the cycle, "
        "drop the saved one in;\n"
        + "   ".join(
            "(" + " → ".join(str(v) for v in cyc) + " → " + str(cyc[0]) + ")"
            for cyc in cycles
        )
        + f"   — {len(cycles)} cycles, one temporary, no second matrix",
        ha="center",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.6,
    )
    pa.set_xlim(-8, xs[-1] + 8)
    pa.set_ylim(0, 60)
    pa.axis("off")
    pa.set_title(
        "(a)  a permutation is disjoint cycles — the reorder can overwrite in place",
        fontsize=F.FS_LABEL,
        color=F.INK,
        pad=6,
    )

    # -- (b) footprints from k s N^2 -----------------------------------------------
    schemes = (
        ("classical\n$D$ + copy + work", 3, F.FAINT),
        ("in-place\n$D$ only", 1, F.BLUE),
    )
    for x, (label, k, col) in enumerate(schemes):
        gb = k * BYTES * N_DEMO**2 / GB
        pb.bar(
            x,
            gb,
            width=0.6,
            color=F.tint(col, 0.55),
            edgecolor=col,
            linewidth=1.0,
            zorder=3,
        )
        pb.text(
            x,
            gb + 2.5,
            f"{gb:.1f} GB",
            ha="center",
            va="bottom",
            fontsize=F.FS_ANNOT,
            color=F.INK,
        )
    for gb, label in BUDGETS:
        pb.axhline(gb, lw=1.0, ls=(0, (3, 2)), color=F.FAINT, zorder=2)
        above = gb > 80
        pb.text(
            1.42,
            gb + (1.5 if above else -1.5),
            label,
            ha="right",
            va="bottom" if above else "top",
            fontsize=F.FS_SMALL,
            color=F.MUTED,
        )
    pb.set_xticks([0, 1])
    pb.set_xticklabels([s[0] for s in schemes], fontsize=F.FS_SMALL)
    F.style_axes(
        pb,
        title=f"(b)  footprint at $N$ = {N_DEMO:,}, float64",
        ylabel="gigabytes  ($k \\cdot s \\cdot N^2$)",
        grid_axis="y",
    )
    pb.set_xlim(-0.5, 1.5)
    pb.set_ylim(0, 3 * BYTES * N_DEMO**2 / GB * 1.18)
    pb.text(
        1.0,
        BYTES * N_DEMO**2 / GB * 2.12,
        "reachable $N \\propto \\sqrt{1/k}$:\n$\\sqrt{3}$ more points\nfor the same bytes",
        ha="center",
        va="bottom",
        fontsize=F.FS_SMALL,
        color=F.shade(F.BLUE, 0.2),
        linespacing=1.4,
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
