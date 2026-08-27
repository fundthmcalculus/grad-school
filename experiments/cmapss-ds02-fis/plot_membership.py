import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

OUT = "outputs/hdbscan-ds02"
df = pd.read_csv(f"{OUT}/mf_ds02.csv")
viable = ["gaussian", "trap-fast"]
configs = ["full-2nd (top_p=0.95)", "2nd (top_p=0.99)"]
colors = {"full-2nd (top_p=0.95)": "#e8590c", "2nd (top_p=0.99)": "#2f9e44"}
metrics = [
    ("per_sample", "per-sample RMSE"),
    ("monotone", "monotone per-cycle RMSE"),
    ("per_engine", "per-engine canonical RMSE"),
]
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle(
    "DS02: Gaussian vs fast-histogram trapezoid membership functions "
    "(raw_memory, l2_reg=0.01)\n"
    "trapezoid ≈ Gaussian and edges it on the endpoint; EM-trapezoid & "
    "triangular blew up (off-scale: per-sample 10–22) — omitted",
    fontsize=11,
)
x = np.arange(len(viable))
w = 0.36
for ax, (key, ylab) in zip(axes, metrics):
    for i, c in enumerate(configs):
        s = df[df.config == c].set_index("mf").loc[viable]
        bars = ax.bar(x + (i - 0.5) * w, s[key].values, w, label=c, color=colors[c])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(viable)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", alpha=0.3)
    vals = df[df.mf.isin(viable)][key]
    ax.set_ylim(vals.min() - 0.5, vals.max() + 0.6)
    if key == "per_sample":
        ax.axhline(6.48, ls="--", color="0.5", lw=1)
        ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(f"{OUT}/mf_ds02.png", bbox_inches="tight")
print("wrote", f"{OUT}/mf_ds02.png")
