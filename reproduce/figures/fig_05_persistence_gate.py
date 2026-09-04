#!/usr/bin/env python3
"""Figure 5.4 -- the persistence gate and the greedy cover, on two battery datasets.

§5.3.4 describes selection in two sentences: admit a block if its persistence is
a robust outlier above the bulk (median + gap_sigma * 1.4826 * MAD), then cover
the data greedily by uncovered-point gain. Table 5.1 reports the result as
numbers; Appendix A.10.14 proves the cover is an antichain. Neither shows the
gate *deciding*, which is the thing a reader needs to see to believe that "k is
an output": which blocks clear the bar, and what the cover does with them.

Two datasets from the battery, chosen because the gate behaves differently on
them. Concentric rings is the success case -- two long-lived blocks stand far
above the threshold and the cover takes exactly those two. Bridged Gaussians is
the recorded failure: the chain of bridge points gives the hierarchy no block
that is both persistent and covers a whole cluster, so the gate admits three
small pure pieces, discovers k = 3 against a true 2 and covers half the data.
Both are the numbers Table 5.1 prints, recomputed here through the same calls.

Top row: every dendrogram node as a persistence-diagram point (birth, death),
the MAD gate as the diagonal offset it is, admitted blocks in blue, the size
window greyed. Bottom row: the points, coloured by the block that covers them,
uncovered points hollow, with the discovered k and coverage in the title.

Computed: `selection.select_coverage_cover` and its `_all_blocks` enumeration on
`ivat_mf.minimax_transform` of the battery's own generators, at the driver's
default `gap_sigma`. Needs scipy and scikit-learn (the battery imports them).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "05-persistence-gate"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gated-minimax-selection"))

GAP_SIGMA = 2.0  # select_coverage_cover's default, the value the driver runs
MAX_SIZE_FRAC = 0.6


def build():
    import battery as B  # noqa: E402
    import ivat_mf as im  # noqa: E402
    import selection as S  # noqa: E402

    cases = [
        ("concentric rings", B.concentric_rings),
        ("bridged Gaussians", B.bridged_gaussians),
    ]
    fig, axes = F.grid_figure(
        2, 2, width=F.W_WIDE, height=6.4, gridspec_kw={"height_ratios": [1.15, 1]}
    )
    colours = [F.BLUE, F.ORANGE, F.AQUA, F.VIOLET, F.MAGENTA, F.GREEN]

    for col, (title, gen) in enumerate(cases):
        X, y = gen()
        Dstar = im.minimax_transform(im.dissimilarity(X))
        blocks, n = S._all_blocks(Dstar)
        sel = S.select_coverage_cover(
            Dstar, gap_sigma=GAP_SIGMA, max_size_frac=MAX_SIZE_FRAC
        )
        persist = np.array([b["persistence"] for b in blocks])
        med = np.median(persist)
        mad = np.median(np.abs(persist - med)) + 1e-12
        thr = med + GAP_SIGMA * 1.4826 * mad
        ceiling = MAX_SIZE_FRAC * n

        # -- top: persistence diagram with the gate ------------------------------
        pa = axes[0, col]
        births = np.array([b["birth"] for b in blocks])
        deaths = np.array([b["death"] for b in blocks])
        sizes = np.array([b["size"] for b in blocks])
        in_window = (sizes >= 3) & (sizes <= ceiling)
        admitted = in_window & (persist >= thr)
        chosen = {id(b) for b in sel}
        picked = np.array([id(b) in chosen for b in blocks])

        lim = deaths.max() * 1.08
        pa.plot([0, lim], [0, lim], lw=0.9, ls="--", color=F.FAINT, zorder=1)
        # The gate is a diagonal offset: death - birth >= thr.
        pa.plot([0, lim], [thr, lim + thr], lw=1.1, color=F.ORANGE, zorder=2)
        pa.text(
            lim * 0.42,
            lim * 0.42 + thr,
            f"gate: persistence ≥ med + {GAP_SIGMA:g}·1.4826·MAD = {thr:.2f}",
            rotation=45,
            rotation_mode="anchor",
            ha="center",
            va="bottom",
            transform_rotates_text=False,
            fontsize=F.FS_SMALL,
            color=F.shade(F.ORANGE, 0.25),
        )
        pa.scatter(
            births[~in_window],
            deaths[~in_window],
            s=14,
            marker="x",
            color=F.FAINT,
            linewidths=0.8,
            zorder=3,
            label="outside the size window",
        )
        pa.scatter(
            births[in_window & ~admitted],
            deaths[in_window & ~admitted],
            s=16,
            color=F.tint(F.INK_2, 0.55),
            linewidths=0,
            zorder=3,
            label="below the gate",
        )
        pa.scatter(
            births[admitted & ~picked],
            deaths[admitted & ~picked],
            s=34,
            facecolor=F.SURFACE,
            edgecolor=F.BLUE,
            linewidths=1.1,
            zorder=4,
            label="admitted, not taken by the cover",
        )
        pa.scatter(
            births[picked],
            deaths[picked],
            s=46,
            color=F.BLUE,
            linewidths=0,
            zorder=5,
            label="selected by the greedy cover",
        )
        F.style_axes(
            pa,
            title=f"({'ab'[col]})  {title}: {len(blocks)} candidate blocks",
            xlabel="birth height",
            ylabel="death height",
        )
        pa.set_xlim(0, lim)
        pa.set_ylim(0, lim)
        pa.set_aspect("equal")
        if col == 0:
            F.legend(pa, loc="upper left", handletextpad=0.4, fontsize=F.FS_SMALL - 0.5)

        # -- bottom: the partition the cover implies ---------------------------
        pb = axes[1, col]
        label = np.full(n, -1)
        for k, b in enumerate(sel):
            label[np.fromiter(b["members"], dtype=int)] = k
        unc = label < 0
        pb.scatter(
            X[unc, 0],
            X[unc, 1],
            s=16,
            facecolor=F.SURFACE,
            edgecolor=F.FAINT,
            linewidths=0.8,
            zorder=2,
            label="uncovered",
        )
        for k in range(len(sel)):
            m = label == k
            pb.scatter(
                X[m, 0],
                X[m, 1],
                s=16,
                color=colours[k % len(colours)],
                linewidths=0,
                zorder=3,
                label=f"block {k + 1}, {m.sum()} pts",
            )
        cov = S.coverage_of(sel, n)
        true_k = len({v for v in y if v >= 0})
        pb.set_aspect("equal")
        pb.axis("off")
        pb.set_title(
            f"({'cd'[col]})  discovered $k$ = {len(sel)} (true {true_k}), coverage {cov:.3f}",
            fontsize=F.FS_LABEL,
            color=F.INK,
            pad=6,
        )
        F.legend(
            pb,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            handletextpad=0.4,
            fontsize=F.FS_SMALL - 0.5,
        )

    fig.text(
        0.5,
        -0.005,
        "Same calls as Table 5.1's set-cover column. On the rings two blocks stand far off the diagonal and the cover "
        "takes exactly those; on the bridge no persistent block spans a whole cluster,\nso the gate admits three pure "
        "fragments, gets the count wrong and covers half the data — the failure §5.4 reports. The cover is always a "
        "disjoint antichain (Appendix A.10.14).",
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
