#!/usr/bin/env python3
"""Figure 1.2 -- the pipeline as a roadmap, with each stage's claim under it.

Load-bearing: this is the figure that orients a reader who has met none of the
chapters yet, and the review lists it as one of the two that carry an argument
rather than illustrating one.

Two decisions worth recording.

**Refinement is drawn as a detour, not a stage.** It is dashed and it is
labelled optional, because the whole thesis is that the search which everyone
else puts on the critical path is not on this one. Drawing it inline, the same
weight as the others, would quietly concede the argument the chapter is making.

**Every claim under a box is read from the harness**, not typed here. The
speedup comes from Table 3.1, the multi-scale ARI from Table 5.2, the accuracy
and training time from Table 4.1, and refinement's decay from the Concrete
reconciliation -- the same CSVs the corresponding chapters quote. A roadmap that
advertises numbers the tables no longer support is exactly the drift this
project has spent four rounds of retraction unwinding.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "01-pipeline-roadmap"


def _claims():
    """The per-stage claims, each read from the table that owns it."""
    scaling, label = H.table("table_3_1")
    largest = [r for r in scaling if r["speedup"] not in ("N/A", "")][-1]
    speedup = f"{int(H.number(largest['speedup'])):,}×"   # '1116x' -> '1,116×'
    n_at = largest["N (points)"]

    multiscale, _ = H.table("table_5_2_multiscale")
    deepest = max(multiscale, key=lambda r: len(r["granularities recovered"].split(",")))
    ms_ari = H.number(deepest["multi-scale (mean ARI)"])
    levels = len(deepest["granularities recovered"].split(","))

    mog, _ = H.table("table_4_1")
    phiusiil = next(r for r in mog if "PhiUSIIL" in r["Dataset (task)"])
    acc = H.number(phiusiil["MoG accuracy / R2"])
    secs = H.number(phiusiil["MoG train time"])

    # Refinement's decay: the same model at three consequent orders, closed-form
    # against refined. The shrinking gap is the structure-before-search evidence.
    concrete, _ = H.table("table_concrete_reconciliation")
    def r2(model, refinement):
        row = next(r for r in concrete
                   if r["Model"] == model and r["Refinement"] == refinement)
        return H.number(row["R²"])
    gain_0 = r2("flat MoG-TSK 0th", "refined") - r2("flat MoG-TSK 0th", "closed-form only")
    gain_2 = r2("flat MoG-TSK 2nd", "refined") - r2("flat MoG-TSK 2nd", "closed-form only")

    return {
        "structure": f"Exact reorder, {speedup} faster at $N$ = {n_at}, and\n"
                     f"feasible past $10^5$ points.   (Table 3.1)",
        "membership": f"$k$ and the number of scales are outputs, not\n"
                      f"inputs: ARI {ms_ari:.2f} across all {levels} levels.   (Table 5.2)",
        "synthesis": f"$K$ rules for a $K$-class problem — {acc:.3f}\n"
                     f"accuracy in {secs:.2f} s on PhiUSIIL.   (Table 4.1)",
        "refine": f"Worth {gain_0:+.2f} $R^2$ at consequent order 0 and\n"
                  f"only {gain_2:+.2f} at order 2 — the structure got there first.",
    }, label


def build():
    fig, ax = F.canvas(width=F.W_WIDE, height=5.4, xlim=(0, 100), ylim=(0, 100))
    claims, label = _claims()

    # Vertical, not left-to-right. Six stages with a sentence of evidence each
    # do not fit across a text block; stacked, every stage gets room for its
    # claim on the same line, which is what the figure is for.
    X, W, HBOX = 29, 42, 12
    ys = [92, 76, 60, 44, 28, 12]

    stages = [
        (F.FAINT, "Raw data", "feature table or dissimilarity matrix", None, False, None),
        (F.BLUE, "Structure discovery", "mergeVAT reorder · iVAT minimax transform",
         "Ch. 3", False, "structure"),
        (F.AQUA, "Membership generation", "persistence-gated set cover over the hierarchy",
         "Ch. 5", False, "membership"),
        (F.ORANGE, "FIS synthesis", "per-class Gaussian mixtures · ridge-TSK solve",
         "Ch. 4, 6", False, "synthesis"),
        (F.VIOLET, "Refinement — optional", "L-BFGS-B polish of the antecedents",
         "Ch. 6, A", True, "refine"),
        (F.FAINT, "Interpretable fuzzy model", "IF–THEN rules over named variables",
         None, False, None),
    ]

    for y, (color, title, body, chapter, dashed, claim) in zip(ys, stages):
        F.box(ax, X, y, W, HBOX, title, body, color=color, dashed=dashed,
              fill_amount=0.93 if dashed else 0.88)
        if chapter:
            F.badge(ax, 4.0, y, chapter, color=color)
        if claim:
            ax.text(X + W / 2 + 4, y, claims[claim], ha="left", va="center",
                    fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.6)

    for a, b, dashed in zip(ys, ys[1:], [False, False, False, True, True]):
        F.arrow(ax, (X, a - HBOX / 2), (X, b + HBOX / 2),
                color=F.FAINT if dashed else F.AXIS, lw=1.2 if dashed else 1.4)

    ax.text(50, 2, "Refinement is dashed on purpose: the search everyone else puts on "
                   "the critical path is not on this one.",
            ha="center", va="center", fontsize=F.FS_SMALL, color=F.MUTED, style="italic")
    ax.text(99, -3, H.provenance_note(label), ha="right", va="center",
            fontsize=F.FS_SMALL - 1, color=F.FAINT)

    ax.set_ylim(-5, 100)
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
