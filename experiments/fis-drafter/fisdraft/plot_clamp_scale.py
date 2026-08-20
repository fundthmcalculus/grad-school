"""Plot the clamp scaling study: causal centrality vs fractional depth per model,
and the k-sweep (effective dimensionality) at each model's peak layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed-order categorical palette (colourblind-safe), assigned by model size
COLORS = ["#8a8f98", "#4c78a8", "#f58518", "#54a24b", "#b279a2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[
            "runs/stream_smol360",
            "runs/stream_qwen1p5b",
            "runs/stream_qwen3b",
            "runs/stream_qwen7b",
            "runs/stream_qwen14b",
        ],
    )
    ap.add_argument("--out", default="runs/clamp_scale.png")
    a = ap.parse_args()
    runs = [r for r in a.runs if (Path(r) / "clamp_scan.json").exists()]
    data = [json.loads((Path(r) / "clamp_scan.json").read_text()) for r in runs]
    data.sort(key=lambda r: r["hidden"])  # small -> large

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for i, r in enumerate(data):
        nL = r["n_layers"] + 1
        c = {int(L): v for L, v in r["causal_centrality"].items()}
        xs = [L / nL for L in sorted(c)]
        ys = [c[L] for L in sorted(c)]
        lbl = f"{r['model'].split('/')[-1]}  (d={r['hidden']})"
        col = COLORS[i % len(COLORS)]
        ax1.plot(xs, ys, "-o", color=col, ms=4, lw=2, label=lbl)
        # k-sweep at peak
        ks = sorted(int(k) for k in r["k_sweep_at_peak"])
        kl = [
            (
                r["k_sweep_at_peak"][str(k)]["clamp"]["behav_KL"]
                if str(k) in r["k_sweep_at_peak"]
                else r["k_sweep_at_peak"][k]["clamp"]["behav_KL"]
            )
            for k in ks
        ]
        ax2.plot(ks, kl, "-o", color=col, ms=4, lw=2, label=lbl)

    ax1.axhline(0, color="#c00", lw=1, ls="--", alpha=0.6)
    ax1.set_xlabel("fractional depth (clamp layer / total layers)")
    ax1.set_ylabel("causal centrality  (clamp KL − random KL)")
    ax1.set_title("Where the exogenous-response subspace lives")
    ax1.text(
        0.02,
        -0.03,
        "≤ 0: no subspace (≈ random)",
        color="#c00",
        fontsize=8,
        transform=ax1.get_yaxis_transform() if False else ax1.transData,
    )
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.25)

    ax2.set_xscale("log", base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32])
    ax2.set_xticklabels([1, 2, 4, 8, 16, 32])
    ax2.set_xlabel("subspace rank k projected out")
    ax2.set_ylabel("behavioural KL vs unclamped (nats)")
    ax2.set_title("Effective dimensionality (k-sweep at peak layer)")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        "Scaling the causal clamp: the exogenous-response subspace across models",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print("wrote", a.out, "with", len(data), "models")


if __name__ == "__main__":
    main()
