"""Figures for the FIS TSP strategy engine, drawn from ``results.json``.

Two panels, because there are two claims to check.

The first is the one that matters: time against tour quality, with the baseline LK
swept into a frontier and the FIS arm plotted as a single point. A point that sits
below and to the left of the whole frontier is a win on both axes; a point sitting
*on* it is a re-parameterisation, not an improvement. Drawing it this way means the
figure can refute the claim, which a bar chart against one chosen baseline cannot.

The second shows where the fuzzy effort actually goes — the distribution of chain
depth over cities — which is the mechanism the first panel is evidence for.

Run:  python figures.py [--results results.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import _instance_frontier as instance_frontier  # noqa: E402

HERE = Path(__file__).resolve().parent


def pareto_front(points):
    """Indices of the points not dominated on (time, gap), both minimised."""
    keep = []
    for i, (t, g) in enumerate(points):
        beaten = False
        for j, (tt, gg) in enumerate(points):
            if j != i and tt <= t and gg <= g and (tt < t or gg < g):
                beaten = True
                break
        if not beaten:
            keep.append(i)
    return sorted(keep, key=lambda i: points[i][0])


def figure(results, out):
    summary = results["summary"]
    rows = results["rows"]

    base = {}
    for key, v in summary.items():
        if key.startswith("lk_"):
            base[key] = (v["total_s"], v["mean_gap"])

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.4))

    # --- panel A: the time/quality plane
    ax = axes[0]
    pts = list(base.values())
    labels = list(base.keys())
    ax.scatter(
        [p[0] for p in pts],
        [p[1] for p in pts],
        s=42,
        facecolor="white",
        edgecolor="tab:blue",
        zorder=3,
        label="baseline LK (swept)",
    )
    front = pareto_front(pts)
    ax.plot(
        [pts[i][0] for i in front],
        [pts[i][1] for i in front],
        "-",
        color="tab:blue",
        alpha=0.55,
        lw=1.8,
        zorder=2,
        label="baseline LK frontier",
    )
    for i in front:
        k, depth, deep = labels[i].split("_")[1:4]
        ax.annotate(
            f"k{k}/d{depth}/b{deep}",
            pts[i],
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=7,
            color="tab:blue",
        )

    markers = {
        "fis_defer": ("*", "tab:red", 320, "FIS + deferred verification"),
        "fis_effort_chain_greedy": ("v", "tab:green", 85, "FIS effort+chain, GA-fitted"),
        "fis_chain_greedy_handwritten": ("D", "tab:orange", 70, "FIS effort+chain, hand-written"),
        "fis_effort_greedy": ("s", "tab:olive", 60, "FIS effort, GA-fitted"),
        "fis_effort_greedy_handwritten": ("P", "tab:gray", 55, "FIS effort, hand-written"),
        "fis_full": ("^", "tab:purple", 55, "FIS + fuzzy construction"),
        "fis_effort_nn": ("X", "tab:brown", 50, "FIS effort, NN start"),
    }
    for key, (mk, col, size, label) in markers.items():
        if key not in summary:
            continue
        v = summary[key]
        # The q value goes in the label because this plane is the *aggregate* view — mean
        # gap against total seconds — and that view flatters the fuzzy arms. q is the
        # per-instance frontier ratio from FINDINGS.md §1 and is the honest number; a
        # marker can sit below-left of this aggregate frontier while still having q > 1.
        q = v.get("mean_q")
        tag = f"{label}  (q={q:.4f})" if q else label
        ax.scatter(
            v["total_s"], v["mean_gap"], marker=mk, s=size, color=col, zorder=5, label=tag
        )

    # The movement fitting produced. Drawn as arrows from the hand-written rule base to
    # the GA-fitted one, for each of the two configurations, because the direction is the
    # result: on these held-out instances fitting moved both arms up and to the right.
    for hand, fitted, col in (
        ("fis_chain_greedy_handwritten", "fis_effort_chain_greedy", "tab:green"),
    ):
        if hand not in summary or fitted not in summary:
            continue
        a, b = summary[hand], summary[fitted]
        ax.annotate(
            "",
            xy=(b["total_s"], b["mean_gap"]),
            xytext=(a["total_s"], a["mean_gap"]),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, alpha=0.85,
                            shrinkA=7, shrinkB=7, linestyle=(0, (4, 2))),
            zorder=4,
        )
    ax.plot([], [], color="black", ls=(0, (4, 2)), lw=1.6,
            label="hand-written $\\rightarrow$ GA-fitted")

    # LKH is deliberately absent from this panel. It only finishes on a subset of the
    # test instances, and putting a point measured over a different subset on a shared
    # total-time axis would be a misleading comparison. It is reported in FINDINGS.md
    # against both arms restricted to the same instances.
    ax.set_xscale("log")
    ax.set_xlabel("total wall clock over the test instances (s, log scale)")
    ax.set_ylabel("mean % over published optimum")
    ax.set_title(
        "Swept LK effort baseline, and where adaptive effort lands\n"
        "aggregate view — q in the legend is the per-instance frontier ratio",
        fontsize=10,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper right")

    # --- panel B: where the effort went
    ax = axes[1]
    ns = [r["n"] for r in rows]
    order = np.argsort(ns)
    ns = np.array(ns)[order]
    key = "fis_defer"
    md = np.array([r.get(f"{key}_mean_depth", np.nan) for r in rows])[order]
    mb = np.array([r.get(f"{key}_mean_breadth", np.nan) for r in rows])[order]
    ax.plot(ns, md, "o-", color="tab:red", label="FIS mean chain depth (adaptive)")
    ax.axhline(
        10.0, color="tab:blue", ls="--", lw=1.5, label="baseline chain depth (fixed, 10)"
    )
    ax.plot(ns, mb, "s-", color="tab:green", alpha=0.8, label="FIS mean first-level breadth")
    ax.axhline(32.0, color="tab:blue", ls=":", lw=1.5, label="baseline breadth (fixed, 32)")
    ax.set_xscale("log")
    ax.set_xlabel("n (cities)")
    ax.set_ylabel("mean parameter value over city searches")
    ax.set_title(
        "What the rule base spends\n(depth is the parameter that costs time)", fontsize=10
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    # --- panel C: the size dependence, which is the actual finding.
    # Band means rather than per-instance points: per instance the ratio swings by a few
    # percent on the small instances, which compresses the region where every arm sits
    # within a fraction of a percent of the frontier and where the result actually is. The
    # error bars are the standard error within each band, and they are the reason the
    # write-up does not claim more than "among the best" at large n.
    ax = axes[2]
    bands = [(0, 1000), (1000, 2000), (2000, 5000), (5000, 30000)]
    centres = [np.sqrt(max(a, 60) * b) for a, b in bands]
    series = [
        ("fis_defer", "tab:red", "o", "FIS + deferred verification"),
        ("fis_effort_chain_greedy", "tab:green", "v", "FIS effort+chain, GA-fitted"),
        ("fis_chain_greedy_handwritten", "tab:orange", "D", "FIS effort+chain, hand-written"),
        ("lk_32_2_4", "tab:blue", "s", "LK k32/d2/b4 (best fixed overall)"),
        ("lk_48_10_32", "tab:cyan", "^", "LK k48/d10/b32 (best quality)"),
    ]
    for key, col, mk, label in series:
        mus, ses = [], []
        for a, b in bands:
            qs = []
            for r in rows:
                if not (a <= r["n"] < b):
                    continue
                t, g = instance_frontier(r)
                bar = float(np.interp(r[f"{key}_s"], t, g))
                qs.append((1.0 + r[f"{key}_gap"] / 100.0) / (1.0 + bar / 100.0))
            qs = np.array(qs)
            mus.append(qs.mean())
            ses.append(qs.std(ddof=1) / np.sqrt(len(qs)) if len(qs) > 1 else 0.0)
        ax.errorbar(
            centres, mus, yerr=ses, fmt=mk + "-", color=col, ms=6, lw=1.5,
            capsize=3, alpha=0.9, label=label,
        )
    ax.axhline(1.0, color="black", lw=1.6, ls="--", label="the baseline frontier")
    ax.set_xscale("log")
    ax.set_xticks(centres)
    ax.set_xticklabels([f"<1k\n({sum(1 for r in rows if r['n'] < 1000)})"]
                       + [f"{a // 1000}k-{b // 1000}k\n"
                          f"({sum(1 for r in rows if a <= r['n'] < b)})"
                          for a, b in bands[1:]], fontsize=8)
    ax.set_xlabel("instance size band (instances)")
    ax.set_ylabel("mean q  (tour length / frontier at equal wall clock)")
    ax.set_title(
        "Frontier ratio against size, with standard error\n"
        "(below the dashed line is outside LK's frontier)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results.json"))
    ap.add_argument("--out", default=str(HERE / "figures" / "fis_tsp_pareto.png"))
    args = ap.parse_args()
    results = json.loads(Path(args.results).read_text())
    Path(args.out).parent.mkdir(exist_ok=True)
    print(f"wrote {figure(results, args.out)}")


if __name__ == "__main__":
    main()
