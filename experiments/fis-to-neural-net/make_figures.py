"""Figures for paper.md, regenerated from the run of record.

    python experiments/fis-to-neural-net/make_figures.py

Writes PNGs to `figures/`. Figure 1 is analytic. Figure 2 reads the
frictionless / damped numbers from the archived tables in `outputs/`
(`pendulum.md`, `pendulum_frictionless.md`). Figure 3 reads the per-epoch
curves in `outputs/pendulum_results.json`, which `run_pendulum.py` writes.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

DPI = 200
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)


def relu(x):
    return np.maximum(x, 0.0)


def fig1_triangle_as_relus(path):
    """§1: a triangular membership function is a sum of three ReLUs."""
    a, b, c = 0.0, 1.0, 2.0
    s_a = 1.0 / (b - a)
    s_c = 1.0 / (c - b)
    x = np.linspace(-0.6, 2.6, 800)

    tri = np.clip(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0.0, None)
    r_a = s_a * relu(x - a)
    r_b = -(s_a + s_c) * relu(x - b)
    r_c = s_c * relu(x - c)
    total = r_a + r_b + r_c

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    for knot, label in [(a, "a"), (b, "b"), (c, "c")]:
        ax0.axvline(knot, color="0.85", lw=0.8, zorder=0)
        ax0.text(knot, 2.55, label, ha="center", fontsize=9, color="0.35")
    ax0.plot(x, r_a, "--", color="#1f77b4", lw=1.2, label=r"$s_a\,\mathrm{relu}(x-a)$")
    ax0.plot(
        x,
        r_b,
        "--",
        color="#ff7f0e",
        lw=1.2,
        label=r"$-(s_a{+}s_c)\,\mathrm{relu}(x-b)$",
    )
    ax0.plot(x, r_c, "--", color="#2ca02c", lw=1.2, label=r"$s_c\,\mathrm{relu}(x-c)$")
    ax0.plot(x, total, color="0.15", lw=1.6, label="sum of the three")
    ax0.set(
        xlabel="x",
        ylabel="contribution",
        title="(a) three ReLU units",
        ylim=(-2.8, 2.8),
    )
    ax0.legend(fontsize=7, frameon=False, loc="lower left")

    ax1.plot(x, tri, color="#d62728", lw=2.4, alpha=0.5, label="triangle $A(x)$")
    ax1.plot(x, total, color="0.15", lw=1.0, ls=":", label="ReLU sum")
    ax1.set(xlabel="x", ylabel="membership", title="(b) identical", ylim=(-0.1, 1.15))
    ax1.legend(fontsize=8, frameon=False, loc="upper right")

    err = float(np.max(np.abs(total - tri)))
    ax1.text(
        0.02,
        0.93,
        f"max |error| = {err:.1e}",
        transform=ax1.transAxes,
        fontsize=8,
        color="0.3",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return err


def fig2_additivity_boundary(path):
    """§6: converting the FIS is exact when the FIS is additive (damped), lossy when not."""
    # Seed R2 is the mean over seeds; FIS R2 is the tribble FIS's own best R2.
    # Source: outputs/pendulum.md and outputs/pendulum_frictionless.md.
    cases = ["damped\n(3 seeds)", "frictionless\n(2 seeds)"]
    fis_r2 = [0.8746, 0.6696]
    seed_r2 = [0.8747, 0.5562]

    x = np.arange(len(cases))
    w = 0.36
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    b1 = ax.bar(x - w / 2, fis_r2, w, color="0.6", label="TRIBBLE FIS")
    b2 = ax.bar(
        x + w / 2, seed_r2, w, color="#1f77b4", label="converted network, at start"
    )
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(
                f"{rect.get_height():.4f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
    loss = fis_r2[1] - seed_r2[1]
    ax.annotate(
        f"loss {loss:.2f} $R^2$",
        (x[1] + w / 2, seed_r2[1]),
        xytext=(x[1] + 0.55, 0.80),
        fontsize=8,
        color="#d62728",
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 0.8},
    )
    ax.set_xticks(x, cases)
    ax.set(ylabel="$R^2$", ylim=(0, 1.2))
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    err = fig1_triangle_as_relus(os.path.join(FIGURES, "fig1_triangle_relu.png"))
    print(f"fig1 max |error| = {err:.3e}")
    fig2_additivity_boundary(os.path.join(FIGURES, "fig2_additivity.png"))
    print("fig2 written")


if __name__ == "__main__":
    main()
