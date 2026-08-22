"""Hyperparameter sweep on DS02 for the low-order consequents (1st and 2nd),
explored independently. Question: with more ridge regularization than the
full-2nd default (l2_reg > 0.01), how close can a cheap linear or diagonal-
quadratic consequent get to the full-2nd best case (6.48 per-sample)?

Axes:
  * l2_reg  : detailed grid, all > 0.01 (the ridge penalty on correction coeffs)
  * top_p   : feature-selection threshold (also a complexity knob -> p features)
  * order   : '1st' and '2nd', swept and reported separately.
Fixed at the DS02 winners: raw_memory features, norm_conorm='hamacher',
n_gaussians=0, stride/window/memory defaults, random_state=42.

Condition correction + featurisation are done once and reused via
fit_featurized, so each of the ~80 fits is just cap/scale/solve.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "FuzzySystemsExperiments")
from tribble_predictive_health import (
    TribblePredictiveHealth,
    load_ncmapss,
)  # noqa: E402
from tribble_predictive_health.preprocessing import (  # noqa: E402
    apply_condition_correction,
    build_memory_features,
    fit_condition_correction,
)

H5 = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"
OUT = "outputs/hdbscan-ds02"
os.makedirs(OUT, exist_ok=True)

L2_GRID = [0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0]
TOP_P_GRID = [0.90, 0.95, 0.99]
ORDERS = ["1st", "2nd"]
FULL2ND = 6.48  # DS02 full-2nd per-sample best case, for reference
CNN = 7.22  # published DS02 CNN


# ---------------------------------------------------------------------------
# Featurise once (condition correction + memory features), then reuse
# ---------------------------------------------------------------------------
print("Loading + featurising DS02 once ...")
dev, cond, sensors = load_ncmapss(H5, "dev")
test, _, _ = load_ncmapss(H5, "test")
models = fit_condition_correction(dev, sensors, cond)
dev_c = apply_condition_correction(dev, sensors, cond, models)
test_c = apply_condition_correction(test, sensors, cond, models)
train_tab, feat_cols = build_memory_features(dev_c, sensors)
test_tab, _ = build_memory_features(test_c, sensors)
print(
    f"  {len(train_tab)} train rows, {len(test_tab)} test rows, "
    f"{len(feat_cols)} features"
)


def run(order, l2, top_p):
    eng = TribblePredictiveHealth(
        condition_correction=False,
        aggregation="raw_memory",
        tsk_order=order,
        l2_reg=l2,
        top_p=top_p,
    )
    eng.fit_featurized(train_tab, feat_cols)
    m = eng.score_featurized(test_tab)
    return dict(
        order=order,
        l2_reg=l2,
        top_p=top_p,
        p=len(eng.regressor_.top_features_),
        n_rules=eng.n_rules_,
        n_params=int(np.asarray(eng.regressor_.corr_terms_).size),
        per_sample=m["per_sample_rmse"],
        monotone=m["monotone_cycle_rmse"],
        per_engine=m["per_engine_rmse"],
    )


rows = []
for order in ORDERS:
    for top_p in TOP_P_GRID:
        for l2 in L2_GRID:
            rows.append(run(order, l2, top_p))
    print(f"  swept {order}: {len(L2_GRID) * len(TOP_P_GRID)} fits done")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "sweep_lowfit_ds02.csv"), index=False)


# ---------------------------------------------------------------------------
# Report: best per order, and the baseline default (l2=0.01) for contrast
# ---------------------------------------------------------------------------
for order in ORDERS:
    sub = df[df.order == order]
    b = sub.loc[sub.per_sample.idxmin()]
    bm = sub.loc[sub.monotone.idxmin()]
    print(f"\n=== {order} order ===")
    print(
        f"  best per-sample : RMSE {b.per_sample:.3f}  "
        f"(l2_reg={b.l2_reg}, top_p={b.top_p}, p={int(b.p)}, "
        f"{int(b.n_params)} params, monotone {b.monotone:.2f})"
    )
    print(
        f"  best monotone   : RMSE {bm.monotone:.3f}  "
        f"(l2_reg={bm.l2_reg}, top_p={bm.top_p}, per-sample {bm.per_sample:.2f})"
    )
    # full grid, compact
    piv = sub.pivot(index="l2_reg", columns="top_p", values="per_sample")
    print("  per-sample RMSE by (l2_reg rows x top_p cols):")
    print(piv.round(3).to_string())


# ---------------------------------------------------------------------------
# Figure: 2 rows (per-sample, monotone) x 2 cols (1st, 2nd), lines by top_p
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
fig.suptitle(
    "DS02 low-order consequent sweep — l2_reg (>0.01) × top_p, per order\n"
    "how close can 1st / 2nd order get to the full-2nd best case?",
    fontsize=12,
)
cmap = plt.get_cmap("viridis")
metrics = [("per_sample", "per-sample RMSE"), ("monotone", "monotone per-cycle RMSE")]
for c, order in enumerate(ORDERS):
    for r, (key, ylabel) in enumerate(metrics):
        ax = axes[r, c]
        sub = df[df.order == order]
        for i, tp in enumerate(TOP_P_GRID):
            s = sub[sub.top_p == tp].sort_values("l2_reg")
            ax.plot(
                s.l2_reg,
                s[key],
                "-o",
                ms=4,
                color=cmap(i / (len(TOP_P_GRID) - 1)),
                label=f"top_p={tp}",
            )
        ax.axhline(FULL2ND, ls="--", color="#c0392b", lw=1)
        ax.axhline(CNN, ls=":", color="0.5", lw=1)
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        if r == 0:
            ax.set_title(f"{order} order", fontsize=11)
        if c == 0:
            ax.set_ylabel(ylabel)
        if r == 1:
            ax.set_xlabel("l2_reg (log scale)")
        if r == 0 and c == 0:
            ax.legend(fontsize=8, loc="best")
axes[0, 1].text(
    L2_GRID[-1],
    FULL2ND,
    " full-2nd 6.48",
    color="#c0392b",
    fontsize=8,
    va="bottom",
    ha="right",
)
axes[0, 1].text(
    L2_GRID[-1], CNN, " CNN 7.22", color="0.4", fontsize=8, va="bottom", ha="right"
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
path = os.path.join(OUT, "sweep_lowfit_ds02.png")
fig.savefig(path, bbox_inches="tight")
print(f"\nwrote {path}")
