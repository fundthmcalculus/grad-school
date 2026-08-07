"""Generate rollout_error_vs_n{,_lines}.png: Table 7's circular error, plotted.

Shows how different angle representations (no wrap, pointwise wrap, hysteresis
wrap, sin/cos) affect prediction error across chain lengths -- read from
results/representation.json (written by wrap_sweep.build_representation_rows,
via run_all.py's `representation` stage or `python wrap_sweep.py`), not from a
transcription of the paper's own table. Run the representation stage first if
that file doesn't exist yet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fis_timestep import RESULT_DIR

REPR_PATH = RESULT_DIR / "representation.json"

representations = ["no wrap", "±180°", "±360°", "hysteresis", "sin/cos"]

#: (representation, wrap_limit_deg) each column reads from the JSON rows.
#: "±180°" is scored on the pointwise wrap -- pointwise and hysteresis agree at
#: L=180 by construction (no overlap band yet), so either would read the same
#: value; pointwise is what the paper's table uses at that column.
COLUMNS = {
    "no wrap": ("pointwise", "none"),
    "±180°": ("pointwise", "180"),
    "±360°": ("pointwise", "360"),
    "hysteresis": ("hysteresis", "360"),
    "sin/cos": ("sin/cos", "n/a"),
}

_SYSTEM_N = {"double": 2, "triple": 3, "quintuple": 5}


def _display_label(dataset):
    """'quintuple_friction' -> '5-link, friction'."""
    system, _, regime = dataset.rpartition("_")
    n = _SYSTEM_N[system]
    fric = "no fric." if regime == "frictionless" else "friction"
    return f"{n}-link, {fric}"


def load_table7(path=REPR_PATH):
    """{display_label: {column: (in_window_deg, past_window_deg)}} from the JSON."""
    if not path.exists():
        sys.exit(
            f"{path} not found -- run `python wrap_sweep.py` or "
            f"`python run_all.py --stage representation` first"
        )
    # run_all.py's stage writes the cache envelope ({"payload": {...}}); wrap_sweep.py
    # run standalone writes the plain dict directly. Accept either.
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("payload", doc).get("representations")
    if rows is None:
        sys.exit(f"{path} has no 'representations' data (keys found: {list(doc.keys())})")
    by_key = {
        (r["dataset"], r["representation"], r["wrap_limit_deg"]): r for r in rows
    }
    data = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        label = _display_label(dataset)
        cols = {}
        for col, (repr_, limit) in COLUMNS.items():
            r = by_key.get((dataset, repr_, limit))
            if r is None:
                continue
            cols[col] = (r["inwindow_rmse_circ_deg"], r["extrap_rmse_circ_deg"])
        data[label] = cols
    return data


def main():
    data = load_table7()
    datasets = list(data.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        cols = [c for c in representations if c in data[dataset]]
        in_window = [data[dataset][c][0] for c in cols]
        past_window = [data[dataset][c][1] for c in cols]

        x = np.arange(len(cols))
        width = 0.35
        bars1 = ax.bar(x - width / 2, in_window, width, label="In-window (0-10s)", alpha=0.8)
        bars2 = ax.bar(x + width / 2, past_window, width, label="Past-window (10-20s)", alpha=0.8)
        for bar, color in zip(bars1, colors):
            bar.set_color(color)
        for bar, color in zip(bars2, colors):
            bar.set_color(color)
            bar.set_alpha(0.4)

        ax.set_xlabel("Representation", fontsize=11)
        ax.set_ylabel("Circular error (°)", fontsize=11)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, max(past_window + in_window) * 1.15])

        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 2, f"{height:.0f}°",
                ha="center", va="bottom", fontsize=8,
            )

    plt.suptitle(
        "Angle Representation Effects: Circular Error Across Representations",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    out1 = Path(__file__).parent / "figures" / "rollout_error_vs_n.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"wrote {out1}")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    for dataset in datasets:
        cols = [c for c in representations if c in data[dataset]]
        in_window = [data[dataset][c][0] for c in cols]
        linestyle = "-" if "no fric" in dataset else "--"
        marker = "o" if "5-link" not in dataset or "no fric" in dataset else "s"
        label_prefix = "Frictionless" if "no fric" in dataset else "Friction"
        chain = dataset.split("-")[0]
        ax.plot(
            np.arange(len(cols)), in_window,
            label=f"{label_prefix} n={chain} (in-window)",
            marker=marker, linestyle=linestyle, linewidth=2, markersize=7,
        )

    ax.set_xlabel("Representation Scheme", fontsize=12)
    ax.set_ylabel("Circular Error (°)", fontsize=12)
    ax.set_title(
        "In-Window Error by Representation: sin/cos Best Only on Some Datasets",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(np.arange(len(representations)))
    ax.set_xticklabels(representations, fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = Path(__file__).parent / "figures" / "rollout_error_vs_n_lines.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"wrote {out2}")
    plt.close()


if __name__ == "__main__":
    main()
