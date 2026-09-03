#!/usr/bin/env python3
"""Figure 6.6 -- antecedent refinement against consequent order, from the reconciliation CSV.

§6.4's refinement table makes one argument: refinement helps most where the
consequent has least capacity, +0.326 at zeroth order against +0.038 and
+0.021, a factor of nine across the ladder, and that gradient is the
*structure before search* thesis in miniature. Three rows of mean ± s.d. carry
the argument; a plot of the two arms against order carries it faster, and it
shows the other thing the table says -- that at second order the two arms are
within a spread of each other.

One panel: test R² for the closed-form construction and for the refined arm at
each TSK order, error bars the ten-seed spread, and the paired gain printed
between them. Read from the archive's own `table_concrete_reconciliation.csv`,
pinned to the run the prose quotes; the flat rows are the `log+standardized`
preprocessing arm, which is the min-max transform §4.3 insists on naming
correctly (the generator's label predates the rename).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "06-refinement-ladder"

ORDERS = ("0th", "1st", "2nd")


def build():
    rows, label = H.table("table_concrete_reconciliation")

    def cell(order, refinement):
        for r in rows:
            if (
                r["Model"].endswith(order)
                and r["Refinement"] == refinement
                and "flat" in r["Model"]
            ):
                return H.number(r["R²"]), H.spread(r["R²"]) or 0.0
        raise KeyError((order, refinement))

    fig, ax = F.figure(width=F.W_COL + 0.6, height=3.8)
    xs = range(len(ORDERS))
    for refinement, name, colour, dx in (
        ("closed-form only", "closed-form construction", F.BLUE, -0.12),
        ("refined", "with antecedent refinement", F.ORANGE, 0.12),
    ):
        vals = [cell(o, refinement) for o in ORDERS]
        ax.errorbar(
            [i + dx for i in xs],
            [v[0] for v in vals],
            yerr=[v[1] for v in vals],
            fmt="o",
            ms=6,
            capsize=3,
            lw=1.5,
            color=colour,
            label=name,
            zorder=4,
        )
    for i, o in enumerate(ORDERS):
        a, b = cell(o, "closed-form only")[0], cell(o, "refined")[0]
        ax.annotate(
            "",
            xy=(i + 0.12, b),
            xytext=(i - 0.12, a),
            arrowprops=dict(arrowstyle="->", lw=1.0, color=F.AXIS),
        )
        ax.text(
            i,
            max(a, b) + 0.045,
            f"Δ = {b - a:+.3f}",
            ha="center",
            va="bottom",
            fontsize=F.FS_ANNOT,
            color=F.INK,
            fontweight="bold",
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{o} order" for o in ORDERS], fontsize=F.FS_TICK)
    F.style_axes(
        ax,
        title="Concrete: what refining the antecedents buys, by consequent order",
        ylabel="test $R^2$, mean ± s.d. (10 seeds)",
        grid_axis="y",
    )
    ax.set_xlim(-0.5, len(ORDERS) - 0.5)
    ax.set_ylim(0.3, 1.0)
    F.legend(ax, loc="lower right")
    ax.text(
        0.0,
        -0.2,
        "Refinement is a search over the membership-function centres and widths with the closed-form solve as its "
        "inner objective (§6.3.5).\nIt buys most where the consequent can do least — a constant per rule — and a "
        "spread's worth once the consequent carries linear and\nquadratic terms: the better the structure-derived "
        f"model, the less a subsequent search finds. {H.provenance_note(label)}",
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
