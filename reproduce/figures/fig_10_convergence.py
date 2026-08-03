#!/usr/bin/env python3
"""Full convergence: how far each optimizer gets, and how much of it converts.

The budget-checkpoint figure (`fig_07`) samples five points. This one draws the
whole trace, because "is there headroom left" is a question about the *shape* of
the curve near the end of the budget, not about its endpoint.

The traces are event-based — the harness records a row only when the best-so-far
improves — so they are step functions on irregular grids and have to be
forward-filled onto a common axis before any median across seeds means anything.
That is done here rather than by the harness so the raw record stays raw.

**(a) Objective against evaluations.** Best-so-far cross-validated MSE, as a
fraction of the construction's own objective on that seed, so 1.0 is the
construction and lower is better. Median across ten seeds. Hot solid, cold
dashed. Read the right-hand end: a curve still descending at 2,000 has headroom
behind it, and a flat one does not.

**(b) The same, against generations.** This is the axis a population method
actually experiences, and the arms are nowhere near comparable on it. GA, PSO and
ACO carry 30 individuals, so 2,000 evaluations is ~67 generations. SciPy's
differential evolution takes `popsize` as a *multiplier of the dimension* — 8 x
136 parameters = 1,088 — so the same budget buys it **under two generations**.
Its curve is not converged; it is barely started, and that is a property of the
configuration §6.3.5 uses rather than of DE.

**(c) What the headroom is worth.** Objective removed on x, paired held-out R²
gain on y, one trajectory per arm walking through the five budget checkpoints,
hollow marker at the 2,000-evaluation end. Two things to read off it.

The headroom is **real and cheap**. Every arm ends between +0.017 and +0.025 R²
above the construction, and the population methods bank most of that in their
first 125 evaluations — DE is at +0.017 there, 70% of where it finishes.

The headroom is also **shallow**. Past roughly 15% of the objective removed the
trajectories go flat: DE finishes at 13.4% removed for +0.0242, L-BFGS-B at 41.9%
for +0.0254. Three times the objective progress, +0.0012 R². That is the study's
§4 finding drawn rather than tabulated — the objective and generalization stop
agreeing early, and driving the objective harder after that is work the test set
does not see.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "10-convergence"

ARM_ORDER = ["scipy-lbfgsb", "scipy-powell", "scipy-de",
             "opt-ga", "opt-pso", "opt-aco", "opt-gd"]
COLOUR = {a: F.SERIES[i % len(F.SERIES)] for i, a in enumerate(ARM_ORDER)}
LABEL = {"scipy-lbfgsb": "L-BFGS-B", "scipy-powell": "Powell",
         "scipy-de": "DE (scipy)", "opt-ga": "GA", "opt-pso": "PSO",
         "opt-aco": "ACO", "opt-gd": "GD"}

#: Individuals per generation, from `arms.py`'s own defaults. `scipy-de` is the
#: exception that makes this panel worth drawing: scipy multiplies `popsize` by
#: the problem dimension, so its population is `8 * n_params` rather than 8.
POPULATION = {"opt-ga": 30, "opt-pso": 30, "opt-aco": 30, "opt-gd": 10}
DE_POPSIZE_MULTIPLIER = 8
GENERATIONAL = ["scipy-de", "opt-ga", "opt-pso", "opt-aco", "opt-gd"]


def _n_params(label):
    """Antecedent parameter count, read from the archive rather than assumed.

    It is not a constant — it follows the model the construction produces, and
    the identification fix moved it from 144 to 136. Converting evaluations into
    generations for `scipy-de` depends on it, so a wrong guess would silently
    mislabel that panel's x axis.
    """
    for name, path, _ in H.archives():
        if name != label:
            continue
        with open(os.path.join(path, "PROVENANCE.txt")) as f:
            m = re.search(r"^params:\s*(\d+)", f.read(), re.M)
        if m:
            return int(m.group(1))
    raise RuntimeError(
        f"no `params:` line in {label}'s PROVENANCE.txt; re-run the study with "
        f"the harness that records it, or panel (b)'s generation axis is a guess")


def _traces():
    """{(arm, init): {seed: (evals, best_cv)}} plus the per-seed construction ref."""
    rows, label = H.table("table_opt_hotstart_traces")
    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        series[(r["arm"], r["init"])][int(r["seed"])].append(
            (int(r["eval"]), float(r["best_cv_mse"])))
    ref = {}
    for seed, pts in series.get(("none", "hot"), {}).items():
        ref[seed] = min(v for _e, v in pts)
    return series, ref, label


def _median_curve(per_seed, ref, grid):
    """Forward-fill each seed's step function onto `grid`, then take the median.

    Forward fill, not interpolation: the value between two recorded improvements
    is the earlier one, exactly, because that is what best-so-far means. Linear
    interpolation would invent progress the optimizer had not made yet.
    """
    stacked = []
    for seed, pts in per_seed.items():
        if seed not in ref or not pts:
            continue
        pts = sorted(pts)
        e = np.array([p[0] for p in pts], dtype=float)
        v = np.array([p[1] for p in pts], dtype=float)
        idx = np.searchsorted(e, grid, side="right") - 1
        idx = np.clip(idx, 0, len(v) - 1)
        stacked.append(v[idx] / ref[seed])
    if not stacked:
        return None
    return np.median(np.vstack(stacked), axis=0)


def _panel_objective(ax, series, ref, arms, budget, xscale="evals",
                     n_params=136):
    grid = np.unique(np.concatenate([[1], np.logspace(0, np.log10(budget), 200)]))
    for arm in arms:
        pop = (DE_POPSIZE_MULTIPLIER * n_params if arm == "scipy-de"
               else POPULATION.get(arm, 1))
        for init, ls in (("hot", "solid"), ("cold", (0, (4, 2)))):
            per_seed = series.get((arm, init))
            if not per_seed:
                continue
            med = _median_curve(per_seed, ref, grid)
            if med is None:
                continue
            x = grid / pop if xscale == "gens" else grid
            ax.plot(x, med, lw=1.7 if init == "hot" else 1.2, ls=ls,
                    color=COLOUR[arm], zorder=4,
                    label=LABEL[arm] if init == "hot" else None)
    ax.axhline(1.0, lw=1.1, ls=(0, (2, 2)), color=F.FAINT, zorder=2)
    ax.set_xscale("log")


def build():
    series, ref, label = _traces()
    if not ref:
        raise RuntimeError("no `none/hot` reference; run run_study.py")
    arms = [a for a in ARM_ORDER if (a, "hot") in series]
    budget = max(e for per_seed in series.values()
                 for pts in per_seed.values() for e, _v in pts) or 2000
    n_params = _n_params(label)

    fig, (ax, gx, cx) = F.grid_figure(1, 3, width=F.W_WIDE + 1.8, height=4.0,
                                      gridspec_kw={"width_ratios": [1.15, 1.15, 1]})

    # -- (a) objective against evaluations ---------------------------------- #
    _panel_objective(ax, series, ref, arms, budget, "evals", n_params)
    ax.text(1.05, 1.0, " the construction", va="bottom", ha="left",
            fontsize=F.FS_SMALL, color=F.MUTED)
    F.style_axes(ax, title="(a)  objective against evaluations",
                 xlabel="objective evaluations (log)",
                 ylabel="best CV MSE / construction's CV MSE")
    F.legend(ax, loc="lower left", ncol=2, handlelength=2.4)

    # -- (b) objective against generations ---------------------------------- #
    gen_arms = [a for a in arms if a in GENERATIONAL]
    _panel_objective(gx, series, ref, gen_arms, budget, "gens", n_params)
    de_gens = budget / (DE_POPSIZE_MULTIPLIER * n_params)
    gx.axvline(de_gens, lw=1.0, ls=(0, (2, 2)), color=F.FAINT, zorder=2)
    gx.annotate(f"DE's whole budget is\n{de_gens:.1f} generations\n"
                f"({DE_POPSIZE_MULTIPLIER}×{n_params} per population)",
                xy=(de_gens, 0.90), xytext=(0.04, 0.24),
                textcoords="axes fraction", fontsize=F.FS_SMALL,
                color=F.MUTED, linespacing=1.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS,
                                connectionstyle="arc3,rad=0.15"))
    gx.text(0.04, 0.06, "colours as in (a); the two local\nmethods have no "
            "generation and\nare left out", transform=gx.transAxes,
            fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.5)
    F.style_axes(gx, title="(b)  objective against generations",
                 xlabel="generations = evaluations / population (log)",
                 ylabel="best CV MSE / construction's CV MSE")

    # -- (c) objective removed against R² gained ---------------------------- #
    rows, _ = H.table("table_opt_hotstart_budget")
    at = defaultdict(lambda: defaultdict(dict))
    heur = {}
    for r in rows:
        b = int(r["budget"])
        at[(r["arm"], r.get("init", "hot"))][b][int(r["seed"])] = (
            float(r["cv_mse"]), float(r["r2"]))
        heur[int(r["seed"])] = float(r.get("heuristic_r2") or r["r2_0"])
    budgets = sorted({int(r["budget"]) for r in rows})
    for arm in arms:
        per_b = at.get((arm, "hot"), {})
        xs, ys = [], []
        for b in budgets:
            seeds = per_b.get(b, {})
            common = [s for s in seeds if s in ref and s in heur]
            if not common:
                continue
            xs.append(float(np.median([100.0 * (1.0 - seeds[s][0] / ref[s])
                                       for s in common])))
            ys.append(float(np.median([seeds[s][1] - heur[s] for s in common])))
        if len(xs) < 2:
            continue
        cx.plot(xs, ys, lw=1.6, marker="o", ms=3.6, color=COLOUR[arm],
                label=LABEL[arm], zorder=4)
        cx.plot(xs[-1:], ys[-1:], marker="o", ms=7.0, mfc=F.SURFACE,
                mec=COLOUR[arm], mew=1.6, zorder=5)
    cx.axhline(0.0, lw=1.0, color=F.AXIS, zorder=3)
    # The flattening is the point of the panel, so it is stated on the panel.
    # Endpoints are read out of the plotted data rather than typed in.
    ends = {}
    for line in cx.get_lines():
        lab = line.get_label()
        if lab and not lab.startswith("_") and len(line.get_xdata()) > 1:
            ends[lab] = (line.get_xdata()[-1], line.get_ydata()[-1])
    if "DE (scipy)" in ends and "L-BFGS-B" in ends:
        (dx, dy), (lx, ly) = ends["DE (scipy)"], ends["L-BFGS-B"]
        lo, hi = cx.get_ylim()
        cx.set_ylim(lo, hi + 0.28 * (hi - lo))
        cx.text(0.03, 0.93,
                f"DE ends at {dx:.0f}% removed for {dy:+.4f} $R^2$;\n"
                f"L-BFGS-B at {lx:.0f}% for {ly:+.4f}.\n"
                f"{lx / dx:.1f}× the objective progress, {ly - dy:+.4f} $R^2$.",
                transform=cx.transAxes, va="top", ha="left",
                fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.5)
    F.style_axes(cx, title="(c)  what the headroom converts to",
                 xlabel="% of the construction's objective removed",
                 ylabel="paired gain in held-out $R^2$")
    F.legend(cx, loc="lower right", ncol=1, handlelength=2.0)

    fig.text(0.5, -0.02,
             "Best-so-far traces, forward-filled onto a common grid and "
             "median-ed across ten seeds; each seed is normalized by its own "
             "construction objective, so 1.0 in (a) and (b) is the\nGaussian "
             "construction and the y axis is scale-free. Hot solid, cold dashed. "
             "(b) divides the same budget by each method's population, which is "
             "the only axis on which\n\"generations\" means the same thing for "
             "two different optimizers — and it shows they are not comparable at "
             "a fixed evaluation budget. Hollow markers in (c)\nare the full "
             f"2,000-evaluation endpoint. {H.provenance_note(label)}",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
