"""DS02: does adding rules (n_output_buckets) let low-order consequents close the
gap to full-2nd? Each output bucket -> one rule (a piecewise model across RUL
bands). Swept for 1st, 2nd, full-2nd at top_p=0.99, l2_reg=0.01. Featurise once,
reuse via fit_featurized. Writes rows incrementally so progress is visible."""

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
CSV = f"{OUT}/sweep_rules_ds02.csv"
BUCKETS = [2, 3, 4, 6, 8, 12]
ORDERS = ["1st", "2nd", "full-2nd"]
FULL2ND, CNN = 6.48, 7.22

print("Loading + featurising DS02 once ...", flush=True)
t0 = time.perf_counter()
dev, cond, sensors = load_ncmapss(H5, "dev")
test, _, _ = load_ncmapss(H5, "test")
models = fit_condition_correction(dev, sensors, cond)
dev_c = apply_condition_correction(dev, sensors, cond, models)
test_c = apply_condition_correction(test, sensors, cond, models)
train_tab, feat_cols = build_memory_features(dev_c, sensors)
test_tab, _ = build_memory_features(test_c, sensors)
print(f"  featurised in {time.perf_counter()-t0:.0f}s", flush=True)

with open(CSV, "w") as f:
    f.write("order,buckets,n_rules,p,n_params,per_sample,monotone,per_engine,fit_s\n")
rows = []
for order in ORDERS:
    for nb in BUCKETS:
        t = time.perf_counter()
        eng = TribblePredictiveHealth(
            condition_correction=False,
            aggregation="raw_memory",
            tsk_order=order,
            top_p=0.99,
            l2_reg=0.01,
            n_output_buckets=nb,
        )
        eng.fit_featurized(train_tab, feat_cols)
        m = eng.score_featurized(test_tab)
        dt = time.perf_counter() - t
        r = dict(
            order=order,
            buckets=nb,
            n_rules=eng.n_rules_,
            p=len(eng.regressor_.top_features_),
            n_params=int(np.asarray(eng.regressor_.corr_terms_).size),
            per_sample=m["per_sample_rmse"],
            monotone=m["monotone_cycle_rmse"],
            per_engine=m["per_engine_rmse"],
            fit_s=dt,
        )
        rows.append(r)
        with open(CSV, "a") as f:
            f.write(
                f"{order},{nb},{r['n_rules']},{r['p']},{r['n_params']},"
                f"{r['per_sample']:.4f},{r['monotone']:.4f},{r['per_engine']:.4f},{dt:.1f}\n"
            )
        print(
            f"  {order:9s} buckets={nb:2d} rules={r['n_rules']:2d} "
            f"params={r['n_params']:5d} per-sample={r['per_sample']:.3f} "
            f"monotone={r['monotone']:.3f} ({dt:.1f}s)",
            flush=True,
        )

df = pd.DataFrame(rows)
for order in ORDERS:
    sub = df[df.order == order]
    b = sub.loc[sub.per_sample.idxmin()]
    print(
        f"\n=== {order} === best per-sample {b.per_sample:.3f} @ {int(b.n_rules)} rules "
        f"({int(b.n_params)} params, monotone {b.monotone:.2f})",
        flush=True,
    )

plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle(
    "DS02: more rules vs consequent order (top_p=0.99, l2_reg=0.01)\n"
    "can extra rules substitute for full-2nd cross terms?",
    fontsize=12,
)
colors = {"1st": "#2b6cb0", "2nd": "#2f9e44", "full-2nd": "#e8590c"}
for ax, key, ylab in [
    (axes[0], "per_sample", "per-sample RMSE"),
    (axes[1], "monotone", "monotone per-cycle RMSE"),
]:
    for order in ORDERS:
        s = df[df.order == order].sort_values("n_rules")
        ax.plot(s.n_rules, s[key], "-o", ms=5, color=colors[order], label=order)
    ax.axhline(FULL2ND, ls="--", color="#c0392b", lw=1)
    ax.axhline(CNN, ls=":", color="0.5", lw=1)
    ax.set_xlabel("number of rules (= output buckets)")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].text(
    df.n_rules.max(),
    FULL2ND,
    " full-2nd@2rules 6.48",
    color="#c0392b",
    fontsize=8,
    va="bottom",
    ha="right",
)
axes[0].text(
    df.n_rules.max(), CNN, " CNN 7.22", color="0.4", fontsize=8, va="bottom", ha="right"
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/sweep_rules_ds02.png", bbox_inches="tight")
print(f"\nwrote {OUT}/sweep_rules_ds02.png", flush=True)
