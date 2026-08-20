"""When does aiming the perturbation pay off? When uniform perturbation has stopped working.

The finding this draws is a *criterion*, not a trend, so the figure has to make the criterion
visible and let a reader see the counterexamples rather than a fitted line through them.

**Left — the criterion.** Each instance is one point: how much better the `EFFORT`-aimed arm is
than the uniform-kick control at a matched budget, against **how much the control still had left
to gain** — the improvement it made over the final quadrupling of its own kick budget. A control
at 0% has plateaued: four times the budget bought it nothing at all.

That is the variable that separates. Both fully-plateaued instances are the two clearest aiming
wins; as the control's headroom grows, aiming's advantage collapses through parity. Instance size
and kick-budget density were each tried first and neither separates — the retractions are in
FINDINGS §6.3, and the reason they looked convincing is that on this instance set both are
correlated with plateauing.

The y axis is a **ratio** (control gap ÷ aimed gap) on a log scale, not a difference, because the
gaps span 0.087% to 4.5% and a difference would let fl1577 decide the panel by itself. Log makes
"1.5x better" and "1.5x worse" the same distance from parity, which is the honest geometry for a
ratio.

**Right — the mechanism, on the instance where it is largest.** fl1577's uniform-kick control
returns *the same tour* at 102 400 and 409 600 kicks. The aimed arm keeps descending over the
same range. fl1577 is a clustered drilling instance whose difficulty lives in a few regions, so
uniformly-sited kicks land almost every time on tour that is already as good as this
neighbourhood allows.

Colour separates the LKH-hard instances from the easy ones and marker fill separates win from
loss, so neither distinction is carried by colour alone. The palette is the validated categorical
default (blue/orange, adjacent-pair ΔE 24.7 under protanopia, 33.6 for normal vision).

Run:  python figures_aim.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import paths  # noqa: E402
from lkh_compare import LADDER_HARD, scaling  # noqa: E402

# Validated categorical slots 1 and 2. Grey is a text token, used for the control series, which
# is a reference baseline rather than a peer category.
HARD_C = "#2a78d6"
EASY_C = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

MECHANISM_INSTANCE = "fl1577"


def _rows(data):
    """(name, n, hard?, control headroom %, control gap, aimed gap) at the matched budget.

    ``headroom`` is what the uniform-kick control still gained over the final quadrupling of its
    own budget, as a percentage of where it started that step. Zero means it returned the same
    tour for four times the kicks — it has plateaued, and no amount of further uniform
    perturbation is going to move it.
    """
    out = []
    for r in scaling(data):
        c = r.get("iterated_at_budget")
        a = r.get("iterated_aim_at_budget")
        if c is None or a is None:
            continue
        pts = sorted(
            (p["kicks"], p["gap"]) for p in data[r["name"]]["iterated"] if p["kicks"]
        )
        if len(pts) < 2:
            continue
        headroom = (pts[-2][1] - pts[-1][1]) / pts[-2][1] * 100.0
        out.append((r["name"], r["n"], r["name"] in LADDER_HARD, headroom, c, a))
    return out


def figure(data, out):
    rows = _rows(data)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.8))
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK_2, labelsize=9)
        a.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
        a.set_axisbelow(True)

    # ---- left: the criterion
    ax.axvspan(-1.2, 1.2, color=HARD_C, alpha=0.07, zorder=0)
    ax.axhline(1.0, color=INK_2, lw=1.4, ls="--", zorder=2)
    ax.annotate(
        "parity",
        xy=(0.995, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_2,
        va="bottom",
        ha="right",
    )

    for name, n, hard, hr, c, a in rows:
        ratio = c / a
        colour = HARD_C if hard else EASY_C
        ax.scatter(
            hr,
            ratio,
            s=130,
            zorder=5,
            marker="o",
            facecolors=colour if ratio > 1.0 else "none",
            edgecolors=colour,
            linewidths=2.0,
        )

    # Hand-placed offsets for the crowded band around parity: three hard instances land within
    # 4 points of x and 0.2 of y of each other, so a uniform offset rule collides every time.
    NUDGE = {
        "fl1577": (0, -30, "center"),
        "fl3795": (0, -30, "center"),
        "d2103": (0, 16, "center"),
        "fnl4461": (0, 16, "center"),
        "d18512": (-13, -20, "right"),
        "brd14051": (0, 17, "center"),
        "d15112": (13, -20, "left"),
        "rl5915": (0, -20, "center"),
        "pcb1173": (15, 0, "left"),
    }
    for name, n, hard, hr, c, a in rows:
        if name not in NUDGE:
            continue
        dx, dy, ha = NUDGE[name]
        ax.annotate(
            f"{name}\nn={n}",
            xy=(hr, c / a),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8,
            color=INK_2,
            linespacing=1.25,
        )

    ax.annotate(
        "control plateaued —\n4× the budget, same tour",
        xy=(1.2, 7.6),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.5,
        color=INK_2,
        linespacing=1.3,
    )

    ax.set_yscale("log")
    ax.set_xlim(-4, 52)
    ax.set_ylim(0.33, 11.0)
    ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax.set_yticklabels(["0.5×", "0.7×", "1×", "1.5×", "2×", "3×", "5×"])
    # A log axis relabels its minor ticks in scientific notation, which collides with the
    # multiplier labels above and reads as a second, contradictory scale.
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(
        "what the uniform control still had left to gain\n"
        "(% improvement over its final 4× of budget)",
        fontsize=10,
        color=INK,
    )
    ax.set_ylabel(
        "aimed kicks vs uniform control\n(control gap ÷ aimed gap, log)",
        fontsize=10,
        color=INK,
    )
    ax.set_title(
        "Aiming wins where uniform kicking has stopped working",
        fontsize=11.5,
        color=INK,
        pad=12,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=10,
            mfc=HARD_C,
            mec=HARD_C,
            mew=2,
            label="LKH-hard, aiming wins",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=10,
            mfc="none",
            mec=HARD_C,
            mew=2,
            label="LKH-hard, aiming loses",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=10,
            mfc=EASY_C,
            mec=EASY_C,
            mew=2,
            label="LKH-easy, aiming wins",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=10,
            mfc="none",
            mec=EASY_C,
            mew=2,
            label="LKH-easy, aiming loses",
        ),
    ]
    ax.legend(
        handles=handles,
        fontsize=8.5,
        loc="upper right",
        frameon=False,
        labelcolor=INK_2,
    )

    # ---- right: the mechanism
    d = data.get(MECHANISM_INSTANCE)
    if d:
        for arm, colour, label, lw in (
            ("iterated", INK_2, "uniform kicks (control)", 2.0),
            ("iterated_aim", HARD_C, "EFFORT-aimed kicks", 2.4),
        ):
            pts = sorted((p["kicks"], p["gap"]) for p in d[arm] if p["kicks"] > 0)
            bx.plot(
                [k for k, _ in pts],
                [g for _, g in pts],
                "-o",
                color=colour,
                lw=lw,
                ms=8,
                label=label,
                zorder=4,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )

        ctl = {p["kicks"]: p["gap"] for p in d["iterated"]}
        if 102400 in ctl and 409600 in ctl and abs(ctl[102400] - ctl[409600]) < 1e-9:
            bx.annotate(
                "identical tour at 4× the budget",
                xy=(409600, ctl[409600]),
                xytext=(-14, 42),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                color=INK_2,
                arrowprops=dict(
                    arrowstyle="-", color=INK_2, lw=1.0, shrinkA=0, shrinkB=7
                ),
            )
        aim = {p["kicks"]: p["gap"] for p in d["iterated_aim"]}
        if 409600 in aim:
            bx.annotate(
                f"{aim[409600]:.3f}%",
                xy=(409600, aim[409600]),
                xytext=(0, -19),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
                color=HARD_C,
            )

    bx.set_xscale("log")
    bx.set_ylim(bottom=0)
    bx.set_xlabel("double-bridge kicks (log)", fontsize=10, color=INK)
    bx.set_ylabel("% over the published optimum", fontsize=10, color=INK)
    bx.set_title(
        f"Why — {MECHANISM_INSTANCE}, a clustered drilling instance\n"
        f"where the difficulty lives in a few regions",
        fontsize=11.5,
        color=INK,
        pad=12,
    )
    bx.legend(fontsize=9.5, loc="lower left", frameon=False, labelcolor=INK_2)

    fig.suptitle(
        "EFFORT-aimed perturbation pays off exactly where uniform perturbation has plateaued",
        fontsize=12.5,
        color=INK,
        y=0.985,
    )
    fig.text(
        0.5,
        0.005,
        "Matched budget = the control's dearest point on that instance. 13 TSPLIB "
        "instances, one seed, one machine. Both plateaued instances are aiming wins; of "
        "the eleven with headroom left, aiming wins two and loses nine.",
        ha="center",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.93])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


# Diverging poles for the outcome regions: blue and red, gray midpoint. Validated as a pair
# (adjacent CVD dE 21.6 protan, 34.5 tritan; normal-vision 32.3) against the light surface.
WIN_C = "#2a78d6"
LOSE_C = "#e34948"
NEUTRAL = "#f0efec"


def regions_figure(data, out):
    """The same measurement as the left panel of :func:`figure`, drawn as outcome *regions*.

    The scatter shows where each instance fell; this shows what the plane is divided into, which
    is the part a reader has to carry away. Two encodings, deliberately kept separate:

    * **colour is the outcome** — a diverging blue/red pair around a neutral parity line, because
      win-versus-loss is a polarity with a meaningful zero, and a categorical pair would imply
      two unrelated kinds rather than two directions from a midpoint;
    * **shape is the instance class** — circle for the instances LKH cannot solve, square for the
      ones it can. So neither distinction is carried by colour alone, and the two questions
      ("did aiming win" and "is this a hard instance") stay readable independently.

    The plateau band is drawn on the axis rather than inferred, since "the control stopped
    improving" is the criterion the figure exists to state, and a reader should be able to see
    that both of its members are in the win region without measuring anything.
    """
    rows = _rows(data)
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5)

    XLO, XHI, YLO, YHI = -4.0, 52.0, 0.33, 11.0

    # the two outcome regions
    ax.axhspan(1.0, YHI, color=WIN_C, alpha=0.085, zorder=0)
    ax.axhspan(YLO, 1.0, color=LOSE_C, alpha=0.075, zorder=0)
    ax.axhline(1.0, color=INK_2, lw=1.6, zorder=3)

    ax.annotate(
        "AIMING WINS",
        xy=(0.985, 0.965),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=13,
        color=WIN_C,
        weight="bold",
    )
    ax.annotate(
        "a shorter tour than uniform kicking, at the same budget",
        xy=(0.985, 0.925),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color=INK_2,
    )
    ax.annotate(
        "AIMING LOSES",
        xy=(0.985, 0.045),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=13,
        color=LOSE_C,
        weight="bold",
    )
    ax.annotate(
        "the fixed cost of computing the aim is not repaid",
        xy=(0.985, 0.09),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        color=INK_2,
    )

    # the criterion, as a band on the x axis
    ax.axvspan(XLO, 1.6, color=INK_2, alpha=0.07, zorder=1)
    ax.annotate(
        "control has\nplateaued",
        xy=(-1.2, 8.4),
        fontsize=9.5,
        color=INK_2,
        ha="center",
        va="center",
        linespacing=1.35,
    )

    for name, n, hard, hr, c, a in rows:
        ratio = c / a
        won = ratio > 1.0
        colour = WIN_C if won else LOSE_C
        ax.scatter(
            hr,
            ratio,
            s=190 if hard else 150,
            zorder=6,
            marker="o" if hard else "s",
            facecolors=colour if hard else "none",
            edgecolors=colour,
            linewidths=2.2,
        )

    NUDGE = {
        "fl1577": (14, 0, "left"),
        "fl3795": (0, -23, "center"),
        "d2103": (0, 20, "center"),
        "fnl4461": (0, 21, "center"),
        "d18512": (0, -21, "center"),
        "brd14051": (0, 20, "center"),
        "d15112": (14, 7, "left"),
        "rl5915": (0, -21, "center"),
        "pcb1173": (16, 0, "left"),
        "pcb3038": (0, -21, "center"),
        "rat783": (0, 20, "center"),
        "rl1323": (-14, -4, "right"),
        "pr2392": (0, 20, "center"),
    }
    for name, n, hard, hr, c, a in rows:
        dx, dy, ha = NUDGE.get(name, (0, 18, "center"))
        ax.annotate(
            f"{name}\nn={n}",
            xy=(hr, c / a),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.5,
            color=INK_2,
            linespacing=1.25,
        )

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax.set_yticklabels(
        [
            "0.5\u00d7",
            "0.7\u00d7",
            "1\u00d7 parity",
            "1.5\u00d7",
            "2\u00d7",
            "3\u00d7",
            "5\u00d7",
        ]
    )
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(False)
    ax.set_xlabel(
        "what uniform kicking still had left to gain\n"
        "(% the control improved over its final 4\u00d7 of budget)",
        fontsize=10.5,
        color=INK,
    )
    ax.set_ylabel(
        "aimed kicks vs uniform kicks\n(control gap \u00f7 aimed gap, log)",
        fontsize=10.5,
        color=INK,
    )

    # Neutral ink, so the legend conveys *shape* only. Colour is the outcome axis here and a
    # coloured legend swatch would assert a link between instance class and outcome that the
    # figure exists to test.
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=11,
            mfc=INK_2,
            mec=INK_2,
            mew=2,
            label="filled circle: LKH cannot solve it (0/10 runs)",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=10,
            mfc="none",
            mec=INK_2,
            mew=2,
            label="open square: LKH solves it every run",
        ),
    ]
    ax.legend(
        handles=handles,
        fontsize=9.5,
        loc="lower left",
        frameon=False,
        labelcolor=INK_2,
        ncol=1,
        bbox_to_anchor=(0.012, 0.015),
    )

    fig.suptitle(
        "Aimed perturbation wins in one region: where uniform kicking has plateaued",
        fontsize=13.5,
        color=INK,
        y=0.975,
    )
    fig.text(
        0.5,
        0.028,
        "13 TSPLIB instances, matched budget, one seed. Both plateaued instances land in "
        "the win region;\nof the eleven with headroom left, two do and nine do not. "
        "Instance size and kick density were each\ntried as the criterion first and "
        "neither survived \u2014 see FINDINGS \u00a76.3.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=INK_2,
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.115, 1, 0.945])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(paths.LKH_COMPARE))
    ap.add_argument("--out", default=str(paths.FIGURES / "fis_tsp_aimed_kicks.png"))
    ap.add_argument(
        "--regions-out", default=str(paths.FIGURES / "fis_tsp_aim_regions.png")
    )
    args = ap.parse_args()
    paths.ensure()
    data = json.loads(Path(args.data).read_text())
    print(f"wrote {figure(data, args.out)}")
    print(f"wrote {regions_figure(data, args.regions_out)}")
    for name, n, hard, hr, c, a in sorted(_rows(data), key=lambda r: r[3]):
        print(
            f"  {name:>9s} n={n:6d} {'hard' if hard else 'easy':>5s} "
            f"headroom {hr:5.1f}%   control {c:6.3f}%  aimed {a:6.3f}%  "
            f"{c / a:5.2f}x  {'AIM' if c > a else 'ctl'}"
        )


if __name__ == "__main__":
    main()
