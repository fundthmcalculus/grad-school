"""Figure 8: PhiUSIIL accuracy against feature budget, by scorer.

Reads the tracked archive reproduce/outputs/topn-sweep-2026-11-21/ and writes
fig/fig8-phiusiil-topn.pdf. Run before build.sh; fig/ is gitignored.

Run (from repo root):  python research/ieee-tfs-mog/plot_topn.py
"""

from __future__ import annotations

import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(
    REPO, "reproduce", "outputs", "topn-sweep-2026-11-21", "table_4_1c_topn_sweep.csv"
)
OUT = os.path.join(HERE, "fig", "fig8-phiusiil-topn.pdf")
MAJORITY = 0.5761
PATTERN = re.compile(r"([-\d.]+) ± ([-\d.]+)")
SCORERS = ["wasserstein", "bhattacharyya", "composite", "higher_order"]
LABELS = {
    "wasserstein": "Wasserstein",
    "bhattacharyya": "Bhattacharyya",
    "composite": "Composite",
    "higher_order": "Higher-order",
}
COLORS = {
    "wasserstein": "#d62728",
    "bhattacharyya": "#1f77b4",
    "composite": "#2ca02c",
    "higher_order": "#9467bd",
}


def main():
    with open(SRC, encoding="utf-8") as fh:
        rows = [
            r for r in csv.DictReader(fh) if r["Dataset (task)"].startswith("PhiUSIIL")
        ]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for sc in SCORERS:
        ks, mean, sd = [], [], []
        for r in rows:
            m = PATTERN.match(r[sc])
            ks.append(int(r["top_n (requested)"]))
            mean.append(float(m.group(1)))
            sd.append(float(m.group(2)))
        ax.errorbar(
            ks,
            mean,
            yerr=sd,
            marker="o",
            ms=3,
            capsize=1.5,
            lw=1.1,
            color=COLORS[sc],
            label=LABELS[sc],
        )
    ax.axhline(MAJORITY, ls="--", color="0.4", lw=0.8)
    ax.text(
        47, MAJORITY - 0.02, "majority", ha="right", va="top", fontsize=6, color="0.3"
    )
    ax.axvspan(4, 5, color="#d62728", alpha=0.08)
    ax.set_xlabel("features retained, $k$ (top\\_n)", fontsize=8)
    ax.set_ylabel("test accuracy", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
