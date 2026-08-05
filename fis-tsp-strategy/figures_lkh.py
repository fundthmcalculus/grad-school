"""Both solvers as time-quality curves, with LKH's floor marked.

The point this figure has to make is not "who is better" — at matched time above its floor LKH
wins outright and reaches the published optimum — but that the two solvers occupy *different
parts of the time axis*, and that the split moves with instance size.

The gap axis is linear, not log, and that is not a stylistic choice: LKH's gap is exactly 0 on
these instances, log(0) is -inf, and matplotlib drops such points *silently* — producing a clean
figure with the competitor absent from it.

LKH through `elkai` takes a run count rather than a time limit, so it has a hard floor: the cost
of one full run, below which it produces nothing at all. That floor is drawn as a shaded region.
Every point of ours inside it is non-dominated by construction, because there is no LKH tour to
compare against at that budget. Whether that is a footnote or the result depends entirely on how
the floor scales, which is why the panels are ordered by n and share a log time axis.

Run:  python figures_lkh.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def _front(points):
    """Lower-left staircase of (time, gap), ascending in time."""
    pts = sorted(points)
    out = []
    best = float("inf")
    for t, g in pts:
        if g < best - 1e-12:
            out.append((t, g))
            best = g
    return out


def figure(data, out):
    names = list(data)
    fig, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 5.0), squeeze=False)
    for ax, name in zip(axes[0], names):
        d = data[name]
        n = None

        sweep = [(r["s"], r["gap"]) for r in d["sweep"]]
        ax.scatter(*zip(*sweep), marker="o", s=45, facecolors="none",
                   edgecolors="tab:blue", label="fixed-parameter LK sweep", zorder=3)

        it = [(r["s"], r["gap"]) for r in d["iterated"]]
        ax.plot(*zip(*_front(it)), "s-", color="tab:red", ms=5, lw=1.6,
                label="iterated (double-bridge kicks)", zorder=4)

        lkh = [(r["s"], r["gap"]) for r in d.get("lkh", []) if r.get("gap") is not None]
        if lkh:
            ax.plot(*zip(*sorted(lkh)), "^-", color="tab:green", ms=8, lw=1.6,
                    label="LKH (elkai), by run count", zorder=5)
            floor = min(t for t, _ in lkh)
            ax.axvspan(
                ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1e-3, floor,
                color="tab:green", alpha=0.07, zorder=0,
            )
            ax.axvline(floor, color="tab:green", ls="--", lw=1.2, alpha=0.7)
            ax.annotate(
                f"LKH cannot run\nbelow {floor:.1f}s",
                xy=(floor, ax.get_ylim()[1]), xytext=(-6, -6),
                textcoords="offset points", ha="right", va="top",
                fontsize=8, color="tab:green",
            )
            if min(g for _, g in lkh) <= 1e-9:
                ax.annotate(
                    "LKH: exactly optimal",
                    xy=(max(t for t, _ in lkh), 0.0), xytext=(-4, 8),
                    textcoords="offset points", ha="right", fontsize=8, color="tab:green",
                )
        else:
            ax.text(0.5, 0.5, "LKH did not finish", transform=ax.transAxes,
                    ha="center", color="tab:green", fontsize=9)

        n = int(name[-4:]) if name[-4:].isdigit() else None
        ax.set_xscale("log")
        # Linear in gap, deliberately. A log gap axis drops LKH entirely, because its gap is
        # exactly 0 on these instances and log(0) is -inf — matplotlib discards the points
        # silently, so the figure renders cleanly with the competitor missing. That is the worst
        # available failure mode for a comparison plot, and worth a comment so it is not
        # reintroduced by someone reaching for a log axis to spread out our own curve.
        ax.set_ylim(bottom=-0.15)
        ax.set_xlabel("wall clock (s, log)")
        ax.set_ylabel("% over published optimum")
        ax.set_title(f"{name}" + (f"  (n={n})" if n else ""), fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "The two solvers occupy different parts of the time axis, and the split moves with n",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "lkh_frontier.json"))
    ap.add_argument("--out", default=str(HERE / "figures" / "fis_tsp_vs_lkh.png"))
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text())
    Path(args.out).parent.mkdir(exist_ok=True)
    print(f"wrote {figure(data, args.out)}")

    print("\nLKH's floor against ours, per instance:")
    for name, d in data.items():
        lkh = [r for r in d.get("lkh", []) if r.get("gap") is not None]
        floor = min((r["s"] for r in lkh), default=None)
        ours = [r for r in d["iterated"] if floor is None or r["s"] < floor]
        best_under = min((r["gap"] for r in ours), default=None)
        f = f"{floor:8.1f}s" if floor else "  did not finish"
        b = f"{best_under:6.3f}%" if best_under is not None else "      —"
        print(f"  {name:>10s}  LKH floor {f}   our best strictly under it {b}")


if __name__ == "__main__":
    main()
