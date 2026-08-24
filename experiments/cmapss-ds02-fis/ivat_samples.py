"""Exploratory: iVAT on subsampled DS02 1 Hz samples, raw vs condition-corrected.

The HDBSCAN view showed the raw sensors clustering by flight regime and the
corrected sensors re-organising around RUL. iVAT (improved Visual Assessment of
cluster Tendency) is the complementary picture: reorder the samples along the VAT
(Prim-MST) path and display the minimax-path dissimilarity as an image. Dark
diagonal blocks = clusters; a smooth corner-to-corner gradient = one continuum.

To make the image *readable* we align two colour strips to the same ordering:
altitude (the operating regime) and true RUL. If the raw blocks line up with the
altitude strip, the tendency is regime; if the corrected ordering lines up with
the RUL strip, the tendency is degradation.

Reuses this repo's iVAT core (gated-minimax-selection/ivat_mf.py).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _ds02_harness import bootstrap  # noqa: E402

bootstrap("FuzzySystemsExperiments", "gated-minimax-selection")
from ivat_mf import dissimilarity, minimax_transform_fast  # noqa: E402
from tribble_predictive_health import load_ncmapss  # noqa: E402
from tribble_predictive_health.preprocessing import (  # noqa: E402
    apply_condition_correction,
    fit_condition_correction,
)

H5 = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"
OUT = "outputs/hdbscan-ds02"
os.makedirs(OUT, exist_ok=True)
N_SUB = 800  # iVAT is O(n^2); a few hundred points is the usual range
VAR_KEEP = 0.95


def vat_order(D):
    """VAT (Prim-MST) visitation order, O(n^2). Seeds at the global-max pair."""
    n = len(D)
    i0 = int(np.unravel_index(np.argmax(D), D.shape)[0])
    order = np.empty(n, dtype=int)
    order[0] = i0
    chosen = np.zeros(n, dtype=bool)
    chosen[i0] = True
    mind = D[i0].copy()
    mind[i0] = np.inf
    for t in range(1, n):
        j = int(np.argmin(mind))
        order[t] = j
        chosen[j] = True
        mind[j] = np.inf
        mind = np.minimum(mind, D[j])
        mind[chosen] = np.inf
    return order


# ---------------------------------------------------------------------------
# Load, condition-correct, subsample
# ---------------------------------------------------------------------------
print("Loading DS02 (dev + test) ...")
dev, cond, sensors = load_ncmapss(H5, "dev")
test, _, _ = load_ncmapss(H5, "test")
models = fit_condition_correction(dev, sensors, cond)
dev_c = apply_condition_correction(dev, sensors, cond, models)
test_c = apply_condition_correction(test, sensors, cond, models)

raw = pd.concat([dev, test], ignore_index=True)
cor = pd.concat([dev_c, test_c], ignore_index=True)
rng = np.random.default_rng(42)
idx = rng.choice(len(raw), size=N_SUB, replace=False)
raw, cor = raw.iloc[idx].reset_index(drop=True), cor.iloc[idx].reset_index(drop=True)
alt = raw["W_alt"].to_numpy(float)
rul = raw["rul"].to_numpy(float)
print(f"  {N_SUB} of {len(dev) + len(test):,} samples")


def ivat(tab):
    """PCA-reduced standardised sensors -> iVAT image + VAT order."""
    X = StandardScaler().fit_transform(tab[sensors].to_numpy(float))
    Xr = PCA(n_components=VAR_KEEP, svd_solver="full").fit_transform(X)
    D = dissimilarity(Xr, metric="euclidean")
    order = vat_order(D)
    img = minimax_transform_fast(D)[np.ix_(order, order)]
    return img, order, Xr.shape[1]


results = {}
for name, tab in (("raw sensors", raw), ("condition-corrected", cor)):
    img, order, npc = ivat(tab)
    # How monotonically does each covariate run along the VAT ordering? |rho|
    # near 1 means the path threads that covariate -- the tendency it reflects.
    rho_alt = abs(spearmanr(np.arange(N_SUB), alt[order]).statistic)
    rho_rul = abs(spearmanr(np.arange(N_SUB), rul[order]).statistic)
    results[name] = dict(
        img=img, order=order, npc=npc, rho_alt=rho_alt, rho_rul=rho_rul
    )
    print(
        f"{name}: PCA {npc} comps;  |rho|(order,altitude)={rho_alt:.2f}  "
        f"|rho|(order,RUL)={rho_rul:.2f}"
    )


# ---------------------------------------------------------------------------
# Figure: 2 columns (raw / corrected); each = altitude strip, RUL strip, iVAT
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
fig = plt.figure(figsize=(13.5, 7.6))
gs = fig.add_gridspec(
    3,
    3,
    width_ratios=[10, 10, 0.35],
    height_ratios=[0.55, 0.55, 11],
    hspace=0.08,
    wspace=0.14,
)
fig.suptitle(
    "iVAT on subsampled DS02 1 Hz samples — cluster tendency, raw vs corrected\n"
    f"({N_SUB} samples, {len(sensors)} sensors → PCA {VAR_KEEP:.0%} var; "
    "dark diagonal blocks = clusters, smooth gradient = one continuum)",
    fontsize=12,
)

alt_norm, rul_norm = Normalize(alt.min(), alt.max()), Normalize(rul.min(), rul.max())
strip_axes = {}
for col, name in enumerate(("raw sensors", "condition-corrected")):
    r = results[name]
    o = r["order"]
    ax_alt = fig.add_subplot(gs[0, col])
    ax_alt.imshow(alt[o][None, :], aspect="auto", cmap="plasma", norm=alt_norm)
    ax_alt.set_yticks([])
    ax_alt.set_xticks([])
    ax_alt.set_title(name, fontsize=11, pad=6)
    ax_alt.set_ylabel("alt", rotation=0, ha="right", va="center", fontsize=8)

    ax_rul = fig.add_subplot(gs[1, col])
    ax_rul.imshow(rul[o][None, :], aspect="auto", cmap="viridis", norm=rul_norm)
    ax_rul.set_yticks([])
    ax_rul.set_xticks([])
    ax_rul.set_ylabel("RUL", rotation=0, ha="right", va="center", fontsize=8)

    ax = fig.add_subplot(gs[2, col])
    # gray: 0 dissimilarity (diagonal / within-cluster) = black, so a cluster
    # reads as a dark block on the diagonal.
    ax.imshow(r["img"], cmap="gray", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        f"VAT order  ·  |ρ|(order, altitude)={r['rho_alt']:.2f}   "
        f"|ρ|(order, RUL)={r['rho_rul']:.2f}",
        fontsize=9,
    )
    strip_axes[name] = (ax_alt, ax_rul)

# shared colorbars for the two strips
cax_alt = fig.add_subplot(gs[0, 2])
fig.colorbar(
    plt.cm.ScalarMappable(norm=alt_norm, cmap="plasma"), cax=cax_alt, label="alt (ft)"
)
cax_rul = fig.add_subplot(gs[1, 2])
fig.colorbar(
    plt.cm.ScalarMappable(norm=rul_norm, cmap="viridis"), cax=cax_rul, label="RUL"
)

path = os.path.join(OUT, "ivat_ds02_samples.png")
fig.savefig(path, bbox_inches="tight")
print(f"\nwrote {path}")
