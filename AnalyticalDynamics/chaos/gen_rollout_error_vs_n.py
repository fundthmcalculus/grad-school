"""Generate rollout_error_vs_n.png: Circular error by representation and chain length.

Shows Table 7 results visually: how different angle representations (no wrap, pointwise wrap,
hysteresis wrap, sin/cos) affect prediction error across chain lengths.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Data from Table 7: Circular error, in-window / past-window, in degrees
# Format: {dataset: {representation: (in_window, past_window)}}
data = {
    "2-link, no fric.": {
        "no wrap": (81.9, 107.4),
        "±180°": (73.7, 106.9),
        "±360°": (68.5, 108.0),
        "hysteresis": (62.5, 108.9),
        "sin/cos": (74.1, 92.1),
    },
    "3-link, no fric.": {
        "no wrap": (59.0, 108.7),
        "±180°": (49.8, 106.9),
        "±360°": (54.9, 108.3),
        "hysteresis": (59.5, 109.1),
        "sin/cos": (48.0, 80.1),
    },
    "5-link, no fric.": {
        "no wrap": (51.9, 107.7),
        "±180°": (50.5, 111.4),
        "±360°": (59.0, 103.7),
        "hysteresis": (51.2, 104.2),
        "sin/cos": (43.6, 71.9),
    },
    "5-link, friction": {
        "no wrap": (12.3, 92.2),
        "±180°": (13.7, 107.2),
        "±360°": (38.3, 92.8),
        "hysteresis": (15.8, 95.8),
        "sin/cos": (12.1, 109.1),
    },
}

datasets = list(data.keys())
representations = ["no wrap", "±180°", "±360°", "hysteresis", "sin/cos"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, dataset in enumerate(datasets):
    ax = axes[idx]

    in_window = [data[dataset][rep][0] for rep in representations]
    past_window = [data[dataset][rep][1] for rep in representations]

    x = np.arange(len(representations))
    width = 0.35

    bars1 = ax.bar(x - width / 2, in_window, width, label="In-window (0-10s)", alpha=0.8)
    bars2 = ax.bar(x + width / 2, past_window, width, label="Past-window (10-20s)", alpha=0.8)

    # Color the bars
    for bar, color in zip(bars1, colors):
        bar.set_color(color)
    for bar, color in zip(bars2, colors):
        bar.set_color(color)
        bar.set_alpha(0.4)

    ax.set_xlabel("Representation", fontsize=11)
    ax.set_ylabel("Circular error (°)", fontsize=11)
    ax.set_title(dataset, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(representations, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 120])

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f"{height:.0f}°",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.suptitle(
    "Angle Representation Effects: Circular Error Across Representations",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
output_path = Path(__file__).parent / "figures" / "rollout_error_vs_n.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"✓ Created {output_path}")
plt.close()

# Also create a line plot version showing the trend more clearly
fig, ax = plt.subplots(figsize=(14, 7))

chain_lengths = [2, 3, 5, 5]  # Last 5 is friction
x_pos = np.arange(len(representations))

for i, dataset in enumerate(datasets):
    in_window = [data[dataset][rep][0] for rep in representations]
    past_window = [data[dataset][rep][1] for rep in representations]

    linestyle = "-" if "no fric" in dataset else "--"
    marker = "o" if i < 3 else "s"
    label_prefix = "Frictionless" if "no fric" in dataset else "Friction"
    chain = int(dataset.split("-")[0])

    ax.plot(
        x_pos,
        in_window,
        label=f"{label_prefix} n={chain} (in-window)",
        marker=marker,
        linestyle=linestyle,
        linewidth=2,
        markersize=7,
    )

ax.set_xlabel("Representation Scheme", fontsize=12)
ax.set_ylabel("Circular Error (°)", fontsize=12)
ax.set_title(
    "In-Window Error by Representation: sin/cos Best Only on Some Datasets",
    fontsize=13,
    fontweight="bold",
)
ax.set_xticks(x_pos)
ax.set_xticklabels(representations, fontsize=11)
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = Path(__file__).parent / "figures" / "rollout_error_vs_n_lines.png"
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
print(f"✓ Created {output_path2}")
plt.close()


if __name__ == "__main__":
    print("Generated angle representation comparison figures")
