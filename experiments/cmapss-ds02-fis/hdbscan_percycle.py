"""Exploratory: HDBSCAN on N-CMAPSS DS02 per-cycle features, raw vs corrected.

The pipeline's first real step is condition correction: regress each sensor on the
operating condition W and keep the residual. This asks what that step does to the
*density structure* of the data. We cluster per-cycle features with HDBSCAN
(density-based -- it finds clumps and calls the rest noise) twice: on the raw
sensors, and on the condition-corrected sensors. Clustering is done in a PCA-
reduced space (HDBSCAN's density estimate is meaningless in raw 70-D), and a
t-SNE embedding is coloured by cluster, by true RUL, and by engine.

Story to look for: raw features cluster into operating/engine structure; once the
condition is subtracted out, the clumps dissolve into a single continuous
manifold whose only smooth axis is degradation (RUL) -- which is exactly why a
smooth regressor, not a bank of regime models, is the right tool.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from _ds02_harness import bootstrap  # noqa: E402

bootstrap("FuzzySystemsExperiments")
from tribble_predictive_health import load_ncmapss  # noqa: E402
from tribble_predictive_health.preprocessing import (  # noqa: E402
    apply_condition_correction,
    build_whole_cycle_features,
    fit_condition_correction,
)

H5 = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"
OUT = "outputs/hdbscan-ds02"
os.makedirs(OUT, exist_ok=True)
VAR_KEEP = 0.90  # PCA variance retained before clustering
MIN_CLUSTER = 20  # HDBSCAN min_cluster_size


def build(corrected):
    """Per-cycle whole-cycle features for all 9 DS02 engines, raw or corrected."""
    dev, cond, sensors = load_ncmapss(H5, "dev")
    test, _, _ = load_ncmapss(H5, "test")
    if corrected:
        models = fit_condition_correction(dev, sensors, cond)
        dev = apply_condition_correction(dev, sensors, cond, models)
        test = apply_condition_correction(test, sensors, cond, models)
    tr, feat_cols = build_whole_cycle_features(dev, sensors)
    te, _ = build_whole_cycle_features(test, sensors)
    tab = pd.concat([tr, te], ignore_index=True)
    return tab, feat_cols


def cluster(tab, feat_cols):
    """Standardise -> PCA(90% var) -> HDBSCAN -> t-SNE embedding. Returns a dict."""
    X = StandardScaler().fit_transform(tab[feat_cols].to_numpy(float))
    pca = PCA(n_components=VAR_KEEP, svd_solver="full").fit(X)
    Xr = pca.transform(X)
    labels = HDBSCAN(min_cluster_size=MIN_CLUSTER, copy=True).fit_predict(Xr)
    emb = TSNE(
        n_components=2, init="pca", perplexity=30, random_state=42
    ).fit_transform(Xr)
    health = (tab["health"].to_numpy() < 0.5).astype(int)  # 1 = degraded
    k = len(set(labels)) - (1 if -1 in labels else 0)
    # Is RUL a smooth function of the features? 5-fold k-NN RUL R^2 on the
    # reduced space -- high means neighbours share RUL, i.e. a clean gradient.
    rul = tab["rul"].to_numpy(float)
    knn_r2 = float(
        cross_val_score(
            KNeighborsRegressor(n_neighbors=15), Xr, rul, cv=5, scoring="r2"
        ).mean()
    )
    return dict(
        emb=emb,
        labels=labels,
        rul=rul,
        unit=tab["unit"].to_numpy(),
        n_pca=Xr.shape[1],
        k=k,
        noise=float(np.mean(labels == -1)),
        knn_r2=knn_r2,
        ami_health=adjusted_mutual_info_score(health, labels),
        ami_unit=adjusted_mutual_info_score(tab["unit"].to_numpy(), labels),
    )


print("Building raw and corrected per-cycle feature tables ...")
raw_tab, cols = build(corrected=False)
cor_tab, _ = build(corrected=True)
print(
    f"  {len(raw_tab)} engine-cycles, {len(cols)} features, "
    f"{raw_tab['unit'].nunique()} engines"
)

results = {}
for name, tab in (("raw sensors", raw_tab), ("condition-corrected", cor_tab)):
    print(f"\nHDBSCAN on {name} ...")
    r = cluster(tab, cols)
    results[name] = r
    print(
        f"  PCA kept {r['n_pca']} comps ({VAR_KEEP:.0%} var); "
        f"{r['k']} clusters, {r['noise']:.0%} noise; "
        f"AMI(health)={r['ami_health']:.3f}  AMI(engine)={r['ami_unit']:.3f}; "
        f"k-NN RUL R^2={r['knn_r2']:.3f}"
    )


# ---------------------------------------------------------------------------
# Figure: rows = raw / corrected, cols = cluster / RUL / engine
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
fig.suptitle(
    "HDBSCAN on DS02 per-cycle features — what condition correction does to "
    "density structure\n"
    f"({len(raw_tab)} engine-cycles, {len(cols)} features → PCA {VAR_KEEP:.0%} "
    f"var, min_cluster_size={MIN_CLUSTER})",
    fontsize=12,
)


def scat(ax, xy, c, cat=False, cmap=None, cbar=None, legend=False):
    if cat:
        for i, v in enumerate(sorted(pd.unique(c))):
            m = c == v
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=10,
                color="0.8" if v == -1 else plt.get_cmap("tab10")(i % 10),
                label=("noise" if v == -1 else str(v)) if legend else None,
                alpha=0.85,
                linewidths=0,
            )
        if legend:
            ax.legend(markerscale=1.5, fontsize=7, ncol=2, loc="best", framealpha=0.9)
        sm = None
    else:
        sm = ax.scatter(
            xy[:, 0], xy[:, 1], s=10, c=c, cmap=cmap, alpha=0.85, linewidths=0
        )
        if cbar:
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label=cbar)
    ax.set_xticks([])
    ax.set_yticks([])


for row, name in enumerate(("raw sensors", "condition-corrected")):
    r = results[name]
    scat(axes[row, 0], r["emb"], r["labels"], cat=True, legend=True)
    axes[row, 0].set_title(
        f"{name}: HDBSCAN cluster\n{r['k']} clusters, {r['noise']:.0%} noise",
        fontsize=10,
    )
    scat(axes[row, 1], r["emb"], r["rul"], cmap="viridis", cbar="RUL (cycles)")
    axes[row, 1].set_title(
        f"{name}: true RUL\nk-NN RUL R² = {r['knn_r2']:.2f}", fontsize=10
    )
    scat(axes[row, 2], r["emb"], r["unit"], cat=True, legend=True)
    axes[row, 2].set_title(
        f"{name}: engine unit\nAMI(engine)={r['ami_unit']:.2f}  "
        f"AMI(health)={r['ami_health']:.2f}",
        fontsize=10,
    )

fig.tight_layout(rect=[0, 0, 1, 0.94])
path = os.path.join(OUT, "hdbscan_ds02_raw_vs_corrected.png")
fig.savefig(path, bbox_inches="tight")
print(f"\nwrote {path}")
