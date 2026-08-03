#!/usr/bin/env python3
"""What the Gaussian construction is worth as a starting point, in iterations.

Three panels, answering the study's three questions in order.

**(a) Held-out R² against the evaluation budget, hot against cold.** Solid lines
start from the Gaussian construction's antecedents; dashed from a uniform random
point in the same box. If the construction were carrying the accuracy, the solid
lines would start high and stay above. They do not.

**(b) The price of not having the construction.** For each cold run, the number
of evaluations before its objective matched what the construction supplies for
free. This is the study's headline number and the reason the x axis everywhere
is evaluations rather than seconds.

**(c) The paired gain over the heuristic, hot and cold side by side.** Mean ±1
s.d. of `R²(run) − R²(heuristic)` per seed. Pairing matters because every run at
a given seed faces the identical split and folds. The bars overlap each other
almost completely, which is the honest headline: something is reliably available
above the heuristic, and neither the optimizer nor the starting point determines
how much.

One caveat the figure cannot draw, and the caption states: a "cold" draw is
random only in *placement and width*. It inherits the structure the construction
discovered — which features carry signal and how many mixture components sit on
each — because those come from the fitted model that `build_param_bounds` is
built from. This isolates the value of the placement, not of the construction as
a whole.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "07-optimizer-hotstart"

ARM_ORDER = ["scipy-lbfgsb", "scipy-powell", "scipy-de",
             "opt-ga", "opt-pso", "opt-aco", "opt-gd"]
COLOUR = {a: F.SERIES[i % len(F.SERIES)] for i, a in enumerate(ARM_ORDER)}
STYLE = {"hot": "solid", "cold": (0, (4, 2))}


def _budget_curve():
    """{(arm, init): {budget: {seed: r2}}}, plus the heuristic reference.

    Keyed on (arm, init), not arm. The two inits share arm names, and collapsing
    them would average a hot run together with a cold one and quietly destroy
    the only comparison this figure exists to make.
    """
    rows, label = H.table("table_opt_hotstart_budget")
    curve = defaultdict(lambda: defaultdict(dict))
    heuristic = {}
    for r in rows:
        init = r.get("init", "hot")
        curve[(r["arm"], init)][int(r["budget"])][int(r["seed"])] = float(r["r2"])
        heuristic[int(r["seed"])] = float(r.get("heuristic_r2") or r["r2_0"])
    return curve, heuristic, label


def _seed_rows():
    rows, _ = H.table("table_opt_hotstart_seeds")
    out = defaultdict(list)
    for r in rows:
        out[(r["arm"], r.get("init", "hot"))].append(r)
    return out


def build():
    curve, heuristic, label = _budget_curve()
    seeds_by = _seed_rows()
    present = [a for a in ARM_ORDER if (a, "hot") in curve]
    inits = [i for i in ("hot", "cold") if any((a, i) in curve for a in present)]
    if not present:
        raise RuntimeError("no budget curve; run reproduce/optimizers/run_study.py")

    fig, (rx, px, bx) = F.grid_figure(1, 3, width=F.W_WIDE + 1.6, height=3.8,
                                      gridspec_kw={"width_ratios": [1.35, 1, 1.1]})

    # -- (a) R^2 against budget, hot vs cold ---------------------------------
    # Two colours, not seven. Fourteen coloured lines (seven arms x two inits)
    # made this panel unreadable and buried its one message. Arm identity lives
    # in (b) and (c); here the only question is whether the solid family sits
    # above the dashed one, so the arms are drawn thin in the init's colour and
    # the median across arms is drawn bold on top.
    HOT_C, COLD_C = F.BLUE, F.ORANGE
    ref = float(np.median(list(heuristic.values())))
    rx.axhline(ref, lw=1.2, ls=(0, (2, 2)), color=F.FAINT, zorder=2)

    budgets = sorted(curve[(present[0], "hot")])
    for init, colour in (("hot", HOT_C), ("cold", COLD_C)):
        per_arm = []
        for arm in present:
            data = curve.get((arm, init))
            if not data:
                continue
            med = [float(np.median(list(data[b].values()))) for b in budgets]
            per_arm.append(med)
            rx.plot(budgets, med, lw=0.9, color=F.tint(colour, 0.55), zorder=3)
        if per_arm:
            rx.plot(budgets, np.median(np.vstack(per_arm), axis=0), lw=2.4,
                    color=colour, marker="o", ms=4.5, zorder=6,
                    label=f"{init} start (median over arms)")

    rx.set_xscale("log")
    rx.set_xticks(budgets)
    rx.set_xticklabels([str(b) for b in budgets], fontsize=F.FS_TICK)
    rx.minorticks_off()
    F.style_axes(rx, title="(a)  held-out $R^2$ against budget",
                 xlabel="objective evaluations (log)", ylabel="held-out $R^2$")
    rx.text(budgets[-1], ref, "Gaussian construction ", va="bottom", ha="right",
            fontsize=F.FS_SMALL, color=F.MUTED)
    F.legend(rx, loc="lower right")

    # -- (b) evaluations for a cold run to match the construction ------------
    reach = {}
    for arm in present:
        vals = [int(r["evals_to_heuristic"]) for r in seeds_by.get((arm, "cold"), [])
                if r.get("evals_to_heuristic") not in (None, "")]
        if vals:
            reach[arm] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    order = sorted(reach, key=lambda a: reach[a][0])
    y = np.arange(len(order))
    px.barh(y, [max(reach[a][0], 1.0) for a in order],
            color=[COLOUR[a] for a in order], height=0.6, zorder=3)
    for yi, a in zip(y, order):
        mean, _, n = reach[a]
        note = f"{mean:.0f}" + ("" if n == 10 else f"  ({n}/10 seeds)")
        px.text(mean * 1.35, yi, note, va="center", ha="left",
                fontsize=F.FS_SMALL, color=F.INK_2)
    px.set_yticks(y)
    px.set_yticklabels(order, fontsize=F.FS_SMALL)
    px.invert_yaxis()
    # Log axis: the answers span 2 to 272, and on a linear scale the population
    # methods' bars are invisible next to the local ones -- which is the
    # comparison the panel exists to show.
    px.set_xscale("log")
    px.set_xlim(1, max(v[0] for v in reach.values()) * 4)
    F.style_axes(px, title="(b)  cold start: evaluations to\nmatch the construction",
                 xlabel="objective evaluations (log)", grid_axis="x")

    # -- (c) paired gain over the heuristic, hot and cold --------------------
    width = 0.38
    for k, init in enumerate(inits):
        means, sds = [], []
        for arm in present:
            d = [float(r["r2"]) - float(r["heuristic_r2"])
                 for r in seeds_by.get((arm, init), []) if r.get("heuristic_r2")]
            means.append(float(np.mean(d)) if d else np.nan)
            sds.append(float(np.std(d)) if d else np.nan)
        y = np.arange(len(present)) + (k - 0.5) * width
        bx.barh(y, means, xerr=sds, height=width * 0.9,
                color=[F.tint(COLOUR[a], 0.0 if init == "hot" else 0.55)
                       for a in present],
                zorder=3, error_kw=dict(ecolor=F.INK_2, elinewidth=0.9, capsize=2))
    bx.axvline(0, lw=0.9, color=F.AXIS, zorder=4)
    bx.set_yticks(np.arange(len(present)))
    bx.set_yticklabels(present, fontsize=F.FS_SMALL)
    bx.invert_yaxis()
    F.style_axes(bx, title="(c)  paired gain over the\nconstruction, ±1 s.d.",
                 xlabel="$R^2$(run) $-$ $R^2$(heuristic)", grid_axis="x")
    bx.text(0.0, 1.005, "solid = hot start · pale = cold start",
            transform=bx.transAxes, ha="left", va="bottom",
            fontsize=F.FS_SMALL, color=F.MUTED)

    fig.text(0.5, -0.02,
             "Same objective, same box, same folds, same test split; the budget is "
             "evaluations, not time, and every arm is single-threaded. A \"cold\" "
             "draw is random in the\nplacement and width of the membership "
             "functions only — it inherits the structure the construction found "
             "(which features, how many components), because that is what\n"
             "`build_param_bounds` is built from. So this measures what the "
             "construction's *placement* is worth, not the construction as a whole. "
             f"{H.provenance_note(label)}",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
