import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
OUT="outputs/hdbscan-ds02"
df=pd.read_csv(f"{OUT}/em_widthreg.csv").sort_values("width_reg").drop_duplicates("width_reg")
plt.rcParams.update({"font.size":10,"figure.dpi":130})
fig,ax=plt.subplots(figsize=(8.6,5.4))
ax.plot(df.width_reg, df.per_sample,"-o",color="#7048e8",ms=6,label="EM trapezoid (width-regularized)")
ax.axhline(6.48,ls="--",color="#e8590c",lw=1.2,label="gaussian 6.48")
ax.axhline(6.42,ls="--",color="#2f9e44",lw=1.2,label="fast trapezoid 6.42")
ax.axhline(7.22,ls=":",color="0.5",lw=1,label="published CNN 7.22")
ax.annotate("pure MLE (density-optimal,\npartition-poor): 9.66",
            (0, df.per_sample.iloc[0]), xytext=(0.4, 9.2), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="0.5"))
ax.set_xlabel("WIDTH_REG  (support-width regularization in the EM M-step)")
ax.set_ylabel("per-sample RMSE (cycles)")
ax.set_title("Fixing EM-trapezoid antecedents on DS02 (full-2nd)\n"
             "density-MLE gives narrow, plateau-collapsed MFs; widening the support recovers most of the gap",
             fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/em_widthreg.png",bbox_inches="tight")
print("wrote", f"{OUT}/em_widthreg.png")
