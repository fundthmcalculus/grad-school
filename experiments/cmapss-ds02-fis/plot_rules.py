import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "outputs/hdbscan-ds02"
df = pd.read_csv(f"{OUT}/sweep_rules_ds02.csv")
ORDERS = ["1st", "2nd", "full-2nd"]
colors = {"1st": "#2b6cb0", "2nd": "#2f9e44", "full-2nd": "#e8590c"}
FULL2ND_095, CNN = 6.48, 7.22
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
fig.suptitle(
    "DS02: adding rules (output buckets) vs consequent order  "
    "[top_p=0.99, l2_reg=0.01]\n"
    "extra simple rules substitute for full-2nd cross terms — most "
    "dramatically on the canonical per-engine endpoint",
    fontsize=12,
)
panels = [
    ("per_sample", "per-sample RMSE (all rows)"),
    ("monotone", "monotone per-cycle RMSE"),
    ("per_engine", "per-engine canonical RMSE (endpoint, 3 engines)"),
]
for ax, (key, ylab) in zip(axes, panels):
    for order in ORDERS:
        s = df[df.order == order].sort_values("n_rules")
        ax.plot(s.n_rules, s[key], "-o", ms=5, color=colors[order], label=order)
    if key == "per_sample":
        ax.axhline(FULL2ND_095, ls="--", color="#c0392b", lw=1)
        ax.axhline(CNN, ls=":", color="0.5", lw=1)
        ax.text(
            df.n_rules.max(),
            FULL2ND_095,
            " full-2nd best 6.48",
            color="#c0392b",
            fontsize=8,
            va="bottom",
            ha="right",
        )
        ax.text(
            df.n_rules.max(),
            CNN,
            " CNN 7.22",
            color="0.4",
            fontsize=8,
            va="bottom",
            ha="right",
        )
    ax.set_xlabel("number of rules (= output buckets)")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/sweep_rules_ds02.png", bbox_inches="tight")
print("wrote", f"{OUT}/sweep_rules_ds02.png", "from", len(df), "rows")
