"""Exploratory: HDBSCAN on subsampled DS02 *raw 1 Hz samples*, raw vs corrected.

The per-cycle view blurred the operating regime; this clusters the actual time
series. Each row is a 1 Hz sample: operating condition W (altitude, Mach,
throttle, inlet temp) and the 14 sensors X_s. At this granularity the sensors are
dominated by *what the aircraft is doing*, so density-based HDBSCAN should carve
the data into flight/operating regimes -- and condition correction should be
exactly what dissolves them, leaving degradation as the residual.

We subsample ~9k samples (1 Hz is heavily autocorrelated; a random draw
decorrelates), cluster on raw sensors and on condition-corrected sensors, and
colour a t-SNE embedding by cluster, by altitude (the regime), and by RUL.
AMI(cluster, regime) and a k-NN RUL R^2 put numbers on the two pictures.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "FuzzySystemsExperiments")
from tribble_predictive_health import load_ncmapss  # noqa: E402
from tribble_predictive_health.preprocessing import (  # noqa: E402
    apply_condition_correction,
    fit_condition_correction,
)

H5 = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"
OUT = "outputs/hdbscan-ds02"
os.makedirs(OUT, exist_ok=True)
N_SUB = 9000       # samples kept for clustering / embedding
VAR_KEEP = 0.95
MIN_CLUSTER = 80   # HDBSCAN min_cluster_size at ~9k points
N_REGIME = 6       # KMeans-on-W regimes, as a per-sample regime label


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
print(f"  subsampled {len(raw):,} of {sum(map(len,(dev,test))):,} samples")

alt = raw["W_alt"].to_numpy(float)
rul = raw["rul"].to_numpy(float)
unit = raw["unit"].to_numpy()
# A per-sample "regime" label from the operating condition W itself.
regime = KMeans(n_clusters=N_REGIME, n_init=10, random_state=42).fit_predict(
    StandardScaler().fit_transform(raw[cond].to_numpy(float))
)


def cluster(tab):
    """Standardise sensors -> PCA -> HDBSCAN -> t-SNE embedding + metrics."""
    X = StandardScaler().fit_transform(tab[sensors].to_numpy(float))
    Xr = PCA(n_components=VAR_KEEP, svd_solver="full").fit_transform(X)
    labels = HDBSCAN(min_cluster_size=MIN_CLUSTER, copy=True).fit_predict(Xr)
    emb = TSNE(n_components=2, init="pca", perplexity=40,
               random_state=42).fit_transform(Xr)
    knn_r2 = float(cross_val_score(
        KNeighborsRegressor(n_neighbors=25), Xr, rul, cv=5, scoring="r2"
    ).mean())
    return dict(
        emb=emb, labels=labels, n_pca=Xr.shape[1],
        k=len(set(labels)) - (1 if -1 in labels else 0),
        noise=float(np.mean(labels == -1)), knn_r2=knn_r2,
        ami_regime=adjusted_mutual_info_score(regime, labels),
        ami_unit=adjusted_mutual_info_score(unit, labels),
    )


results = {}
for name, tab in (("raw sensors", raw), ("condition-corrected", cor)):
    print(f"\nHDBSCAN on {name} ...")
    r = cluster(tab)
    results[name] = r
    print(f"  PCA {r['n_pca']} comps; {r['k']} clusters, {r['noise']:.0%} noise; "
          f"AMI(regime)={r['ami_regime']:.3f}  AMI(engine)={r['ami_unit']:.3f}; "
          f"k-NN RUL R^2={r['knn_r2']:.3f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
fig.suptitle(
    "HDBSCAN on subsampled DS02 1 Hz samples — the raw sensors ARE the flight "
    "regime; correction dissolves it\n"
    f"({N_SUB:,} random samples, {len(sensors)} sensors → PCA {VAR_KEEP:.0%} var, "
    f"min_cluster_size={MIN_CLUSTER})",
    fontsize=12,
)


def scat(ax, xy, c, cat=False, cmap=None, cbar=None, legend=False):
    if cat:
        for i, v in enumerate(sorted(pd.unique(c))):
            m = c == v
            ax.scatter(xy[m, 0], xy[m, 1], s=6,
                       color="0.8" if v == -1 else plt.get_cmap("tab20")(i % 20),
                       label=("noise" if v == -1 else str(v)) if legend else None,
                       alpha=0.7, linewidths=0)
        if legend:
            ax.legend(markerscale=2.0, fontsize=6, ncol=2, loc="best", framealpha=0.9)
    else:
        sm = ax.scatter(xy[:, 0], xy[:, 1], s=6, c=c, cmap=cmap, alpha=0.75,
                        linewidths=0)
        if cbar:
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label=cbar)
    ax.set_xticks([]); ax.set_yticks([])


for row, name in enumerate(("raw sensors", "condition-corrected")):
    r = results[name]
    scat(axes[row, 0], r["emb"], r["labels"], cat=True, legend=True)
    axes[row, 0].set_title(
        f"{name}: HDBSCAN cluster\n{r['k']} clusters, {r['noise']:.0%} noise",
        fontsize=10)
    scat(axes[row, 1], r["emb"], alt, cmap="plasma", cbar="altitude (ft)")
    axes[row, 1].set_title(
        f"{name}: altitude (flight regime)\nAMI(cluster, W-regime) = "
        f"{r['ami_regime']:.2f}", fontsize=10)
    scat(axes[row, 2], r["emb"], rul, cmap="viridis", cbar="RUL (cycles)")
    axes[row, 2].set_title(
        f"{name}: true RUL\nk-NN RUL R² = {r['knn_r2']:.2f}", fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.94])
path = os.path.join(OUT, "hdbscan_ds02_samples.png")
fig.savefig(path, bbox_inches="tight")
print(f"\nwrote {path}")
