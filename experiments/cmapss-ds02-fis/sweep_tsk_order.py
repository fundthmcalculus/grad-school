"""Exploratory: consequent order vs complexity on DS02 (raw_memory best case).

The DS02 winner uses tsk_order='full-2nd' -- each rule's consequent is a full
quadratic: linear terms, squares, AND every cross term x_i*x_j. With p features
per rule that is 1 + 2p + p(p-1)/2 coefficients, quadratic in p. The cross terms
are the expensive part.

Question: do we need them? Compare
  '1st'      : 1 + p            (linear)
  '2nd'      : 1 + 2p           (linear + squares, NO interactions)
  'full-2nd' : 1 + 2p + p(p-1)/2 (adds every cross term)
Everything else is held at the DS02 winning configuration; only the consequent
order changes (feature selection is independent of it, so p is constant). If
'2nd' matches 'full-2nd' accuracy, we drop the p(p-1)/2 cross terms for free.
"""

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "FuzzySystemsExperiments")
from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss  # noqa: E402
from tribble_predictive_health.metrics import rmse  # noqa: E402

H5 = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"
OUT = "outputs/hdbscan-ds02"
os.makedirs(OUT, exist_ok=True)
TEST_UNITS = (11, 14, 15)
# (label, kwargs). "full-2nd +detect" keeps the squares but adds cross terms only
# for *detected* interacting pairs -- the middle ground between 2nd and full-2nd.
CONFIGS = [
    ("1st", dict(tsk_order="1st")),
    ("2nd", dict(tsk_order="2nd")),
    ("full-2nd +detect", dict(tsk_order="full-2nd", detect_interactions=True)),
    ("full-2nd +select", dict(tsk_order="full-2nd", select_interactions=True)),
    ("full-2nd", dict(tsk_order="full-2nd")),
]


print("Loading DS02 ...")
dev, _, _ = load_ncmapss(H5, "dev")
test, _, _ = load_ncmapss(H5, "test")

rows = []
for label, kw in CONFIGS:
    engine = TribblePredictiveHealth(**kw)  # raw_memory best case otherwise
    t0 = time.perf_counter()
    engine.fit(dev, dev["rul"])
    fit_s = time.perf_counter() - t0

    reg = engine.regressor_
    p = len(reg.top_features_)
    n_params = int(np.asarray(reg.corr_terms_).size)  # true consequent coeff count
    # terms/rule (incl. intercept) derived from the true param count -- robust to
    # detect_interactions changing how many cross terms survive.
    terms_per_rule = n_params // engine.n_rules_ + 1

    m = engine.score(test)
    frame = engine.predict_frame(test, include_true=True)
    per_engine = {u: rmse(frame[frame.unit == u]["true"], frame[frame.unit == u]["rul"])
                  for u in TEST_UNITS}
    rows.append(dict(order=label, p=p, n_rules=engine.n_rules_,
                     terms_per_rule=terms_per_rule, n_params=n_params, fit_s=fit_s,
                     per_sample=m["per_sample_rmse"], monotone=m["monotone_cycle_rmse"],
                     **{f"eng{u}": per_engine[u] for u in TEST_UNITS}))
    print(f"  {label:16s}: p={p}  terms/rule={terms_per_rule:4d}  "
          f"consequent params={n_params:5d}  per-sample RMSE={m['per_sample_rmse']:.2f}  "
          f"monotone={m['monotone_cycle_rmse']:.2f}  ({fit_s:.1f}s)")


# ---------------------------------------------------------------------------
# Report table + complexity/accuracy figure
# ---------------------------------------------------------------------------
base = next(r for r in rows if r["order"] == "full-2nd")
hdr = (f"{'order':9s} {'terms/rule':>10s} {'params':>7s} {'params vs full':>14s} "
       f"{'per-sample':>10s} {'monotone':>9s} {'eng11/14/15':>16s}")
print("\n" + hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['order']:9s} {r['terms_per_rule']:10d} {r['n_params']:7d} "
          f"{r['n_params'] / base['n_params']:13.1%} {r['per_sample']:10.2f} "
          f"{r['monotone']:9.2f} "
          f"{r['eng11']:.2f}/{r['eng14']:.2f}/{r['eng15']:.2f}")

plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
fig, ax = plt.subplots(figsize=(8.4, 5.4))

# The pruning variants collapse onto canonical orders (detect->2nd, select->
# full-2nd), so the distinct accuracy/complexity frontier is just the 3 orders.
frontier = [r for r in rows if r["order"] in ("1st", "2nd", "full-2nd")]
ax.plot([r["n_params"] for r in frontier], [r["per_sample"] for r in frontier],
        "-o", color="#2b6cb0", markersize=9, zorder=2)
for r in frontier:
    ax.annotate(f"  {r['order']}\n  ({r['terms_per_rule']} terms/rule, "
                f"{r['n_params']} params)",
                (r["n_params"], r["per_sample"]), fontsize=9, va="center")

ax.axhline(7.22, ls="--", color="0.5", lw=1)
ax.text(ax.get_xlim()[1], 7.22, " published CNN 7.22", color="0.4",
        fontsize=8, va="bottom", ha="right")
ax.text(0.02, 0.03,
        "pruning the cross terms fails on DS02:\n"
        "  detect_interactions → 0 pairs kept (= 2nd)\n"
        "  select_interactions (LassoCV) → all pairs kept (= full-2nd)",
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round", fc="#fff7e6", ec="0.7"))

ax.set_xscale("log")
ax.set_xlabel("consequent parameters (log scale)  →  model complexity")
ax.set_ylabel("per-sample RMSE (cycles)")
ax.set_title("DS02 raw_memory: consequent order vs accuracy\n"
             "the full-2nd cross terms earn their complexity — and don't prune", fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
path = os.path.join(OUT, "tsk_order_ds02.png")
fig.savefig(path, bbox_inches="tight")
print(f"\nwrote {path}")
