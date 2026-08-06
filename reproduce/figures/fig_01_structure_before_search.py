#!/usr/bin/env python3
"""Figure 1.1 -- the two routes to a trained FIS, side by side.

Chapter 1's thesis in one picture: on the left the conventional route (grid the
inputs, then grind a stochastic search against the whole model), on the right
the structure-first route (recover the structure, read the model off it, polish
locally). The caption asks for the rule count and the training time under each.

The rule count is arithmetic, not a measurement -- the grid product from §1.1
evaluated on PhiUSIIL's 54 features against the K rules the answer-first
construction produces on the same dataset. Both sides of that comparison are
exact.

The training time is not symmetric, and the figure says so rather than
balancing it. The structure-first side is measured and read from the harness.
The conventional side has **no measured baseline in this document**: Table 4.1's
ANFIS and GA-FIS columns are N/A because those adapters are not written, which
Chapter 7 carries as Goal G3. Putting a plausible number opposite a measured one
would be the single most misleading thing this figure could do, so the panel
prints the gap instead. When the adapters land, this figure gains a number and
loses a caveat.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "01-structure-before-search"

# PhiUSIIL, as Chapter 4 describes it: 235K rows x 54 features, binary. The grid
# count is 3^54 -- three fuzzy sets on each input, the example §1.1 and §2.1 both
# use -- and it is the number the answer-first construction refuses to pay.
N_FEATURES = 54
SETS_PER_FEATURE = 3
N_CLASSES = 2


def build():
    fig, ax = F.canvas(width=F.W_WIDE, height=5.8, xlim=(0, 100), ylim=(0, 100))

    rows, label = H.table("table_4_1")
    phiusiil = next(r for r in rows if "PhiUSIIL" in r["Dataset (task)"])
    train_s = phiusiil["MoG train time"].replace(" s", "").strip()
    accuracy = H.number(phiusiil["MoG accuracy / R2"])

    grid_rules = SETS_PER_FEATURE**N_FEATURES
    exponent = len(str(grid_rules)) - 1
    mantissa = grid_rules / 10**exponent

    LEFT, RIGHT = 32, 76
    W = 38

    ax.text(
        LEFT,
        97,
        "Conventional: search first",
        ha="center",
        va="center",
        fontsize=F.FS_TITLE,
        color=F.INK,
        fontweight="bold",
    )
    ax.text(
        RIGHT,
        97,
        "This work: structure first",
        ha="center",
        va="center",
        fontsize=F.FS_TITLE,
        color=F.INK,
        fontweight="bold",
    )

    # -- left column ---------------------------------------------------------
    # The search box is deliberately the tall one. On this route it is where the
    # time goes, and a diagram that drew it the same size as its neighbours
    # would be arguing the opposite of §1.1.
    left_stack = [
        (
            88,
            14,
            "Grid-partition every input",
            f"{SETS_PER_FEATURE} fuzzy sets on each of $M$ features",
        ),
        (
            64,
            22,
            "Search the whole model",
            "genetic algorithm · gradient descent\nANFIS-style hybrids",
        ),
        (39, 14, "Trained FIS", None),
    ]
    for y, h, title, body in left_stack:
        F.box(ax, LEFT, y, W, h, title, body, color=F.ORANGE)
    for a, b in zip(left_stack, left_stack[1:]):
        F.arrow(ax, (LEFT, a[0] - a[1] / 2), (LEFT, b[0] + b[1] / 2))

    # -- right column --------------------------------------------------------
    right_stack = [
        (
            88,
            14,
            "Recover the structure",
            "mergeVAT reorder, persistence\n(Ch. 3, Ch. 5)",
        ),
        (
            71,
            14,
            "Read the model off it",
            "per-class Gaussian mixtures,\nclosed-form consequents   (Ch. 4)",
        ),
        (
            55,
            14,
            "Polish locally — optional",
            "ridge re-solve, L-BFGS-B\n(Ch. 6, App. A)",
        ),
        (39, 14, "Trained FIS", None),
    ]
    for y, h, title, body in right_stack:
        F.box(ax, RIGHT, y, W, h, title, body, color=F.BLUE)
    for a, b in zip(right_stack, right_stack[1:]):
        F.arrow(ax, (RIGHT, a[0] - a[1] / 2), (RIGHT, b[0] + b[1] / 2))

    # -- the two readings, on a shared baseline ------------------------------
    # A divider rather than a third box: these are what the two columns are
    # read FOR, not another stage in either of them.
    ax.plot([1, 99], [29, 29], lw=0.8, color=F.AXIS, zorder=1)
    ax.text(
        1,
        23,
        "rule\nbase",
        ha="left",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.4,
    )
    ax.text(
        1,
        7,
        "training\ntime",
        ha="left",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.4,
    )

    ax.text(
        LEFT,
        23,
        f"${mantissa:.1f}\\times10^{{{exponent}}}$ rules",
        ha="center",
        va="center",
        fontsize=13,
        color=F.shade(F.ORANGE, 0.2),
        fontweight="bold",
    )
    ax.text(
        LEFT,
        16,
        f"$\\prod_i N_{{\\mu_i}} = {SETS_PER_FEATURE}^{{{N_FEATURES}}}$"
        f"  on PhiUSIIL's {N_FEATURES} features",
        ha="center",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
    )
    ax.text(
        RIGHT,
        23,
        f"{N_CLASSES} rules",
        ha="center",
        va="center",
        fontsize=13,
        color=F.shade(F.BLUE, 0.2),
        fontweight="bold",
    )
    ax.text(
        RIGHT,
        16,
        f"one per answer, $K = {N_CLASSES}$",
        ha="center",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
    )

    ax.plot([1, 99], [12, 12], lw=0.8, color=F.GRID, zorder=1)
    ax.text(
        LEFT,
        7,
        "no measured baseline — the ANFIS and GA-FIS\n"
        "adapters are Goal G3 (Table 4.1 reports N/A)",
        ha="center",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
        style="italic",
    )
    ax.text(
        RIGHT,
        7,
        f"{train_s} s   ·   accuracy {accuracy:.3f}",
        ha="center",
        va="center",
        fontsize=F.FS_ANNOT,
        color=F.shade(F.BLUE, 0.2),
    )

    ax.text(
        99,
        0.5,
        H.provenance_note(label),
        ha="right",
        va="center",
        fontsize=F.FS_SMALL - 1,
        color=F.FAINT,
    )
    ax.set_ylim(-1, 100)

    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
