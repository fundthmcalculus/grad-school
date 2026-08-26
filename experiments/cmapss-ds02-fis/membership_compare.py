"""DS02: does the antecedent membership-function shape matter? Compare Gaussian
vs trapezoid (fast histogram + EM) vs triangular, on the winning full-2nd config
and the lean 2nd-order config. Featurise once; write rows incrementally."""

import os, time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

from _ds02_harness import bootstrap

bootstrap("FuzzySystemsExperiments")
from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss
from tribble_predictive_health.preprocessing import (
    apply_condition_correction,
    build_memory_features,
    fit_condition_correction,
)

H5 = "data/nasa-cmapps2/N-CMAPSS_DS02-006.h5"
OUT = "outputs/hdbscan-ds02"
os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/mf_ds02.csv"
CONFIGS = [
    ("full-2nd (top_p=0.95)", dict(tsk_order="full-2nd", top_p=0.95)),
    ("2nd (top_p=0.99)", dict(tsk_order="2nd", top_p=0.99)),
]
MFS = [
    ("gaussian", dict(member_function="gaussian")),
    ("trap-fast", dict(member_function="trap", trapz_method="fast")),
    ("trap-em", dict(member_function="trap", trapz_method="em")),
    ("triangular", dict(member_function="triangular")),
]

print("Loading + featurising DS02 once ...", flush=True)
dev, cond, sensors = load_ncmapss(H5, "dev")
test, _, _ = load_ncmapss(H5, "test")
models = fit_condition_correction(dev, sensors, cond)
dev_c = apply_condition_correction(dev, sensors, cond, models)
test_c = apply_condition_correction(test, sensors, cond, models)
train_tab, feat_cols = build_memory_features(dev_c, sensors)
test_tab, _ = build_memory_features(test_c, sensors)

open(CSV, "w").write("config,mf,per_sample,monotone,per_engine,n_rules,fit_s\n")
rows = []
for cname, ckw in CONFIGS:
    for mfname, mfkw in MFS:
        t = time.perf_counter()
        eng = TribblePredictiveHealth(
            condition_correction=False,
            aggregation="raw_memory",
            l2_reg=0.01,
            **ckw,
            **mfkw,
        )
        eng.fit_featurized(train_tab, feat_cols)
        m = eng.score_featurized(test_tab)
        dt = time.perf_counter() - t
        r = dict(
            config=cname,
            mf=mfname,
            per_sample=m["per_sample_rmse"],
            monotone=m["monotone_cycle_rmse"],
            per_engine=m["per_engine_rmse"],
            n_rules=eng.n_rules_,
            fit_s=dt,
        )
        rows.append(r)
        open(CSV, "a").write(
            f"{cname},{mfname},{r['per_sample']:.4f},{r['monotone']:.4f},"
            f"{r['per_engine']:.4f},{r['n_rules']},{dt:.1f}\n"
        )
        print(
            f"  {cname:22s} {mfname:11s} per-sample={r['per_sample']:.3f} "
            f"monotone={r['monotone']:.3f} per-engine={r['per_engine']:.3f} ({dt:.1f}s)",
            flush=True,
        )

df = pd.DataFrame(rows)
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle(
    "DS02: antecedent membership-function shape (Gaussian vs trapezoid vs triangular)\n"
    "raw_memory, l2_reg=0.01",
    fontsize=12,
)
mf_order = [m[0] for m in MFS]
x = np.arange(len(mf_order))
w = 0.35
colors = {"full-2nd (top_p=0.95)": "#e8590c", "2nd (top_p=0.99)": "#2f9e44"}
for ax, key, ylab in [
    (axes[0], "per_sample", "per-sample RMSE"),
    (axes[1], "monotone", "monotone per-cycle RMSE"),
]:
    for i, (cname, _) in enumerate(CONFIGS):
        s = df[df.config == cname].set_index("mf").loc[mf_order]
        ax.bar(x + (i - 0.5) * w, s[key].values, w, label=cname, color=colors[cname])
    ax.set_xticks(x)
    ax.set_xticklabels(mf_order)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    lo = df[key].min()
    ax.set_ylim(lo - 0.4, df[key].max() + 0.3)
axes[0].axhline(6.48, ls="--", color="0.5", lw=1)
axes[0].text(
    len(mf_order) - 1,
    6.48,
    " gaussian full-2nd best 6.48",
    color="0.4",
    fontsize=8,
    va="bottom",
    ha="right",
)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/mf_ds02.png", bbox_inches="tight")
print(f"\nwrote {OUT}/mf_ds02.png", flush=True)
