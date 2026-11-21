"""PCA before clustering: k-means, FCM and E-means on the embedding corpora.

Question: is FCM's collapse on sentence embeddings a dimensionality effect? On unit-length
384-d vectors all pairwise distances are nearly equal, so FCM memberships are close to 1/k
and the centres collapse. Reducing dimension with PCA should restore contrast. The same
reduction is applied before every method, so the comparison stays fair.

Settings per corpus/embedding: original dimension, PCA to 32, PCA to 64. PCA is fitted on
the base set only (centred, no whitening). Methods: k-means (k-means++), FCM (m=2), E-means
(5 epochs, default init). Ten seeds. k = number of classes.

Diagnostics per run: mean largest membership (FCM only; uniform = 1/k), smallest and largest
cluster size, so collapse is visible in the output, not inferred.

Run (repo root), in the tribble-cluster environment built outside the submodule:
    UV_PROJECT_ENVIRONMENT=$HOME/.cache/grad-school/venv-tribble-cluster \\
      uv run --project tribble-cluster --with faiss-cpu==1.16.0 --with scikit-learn \\
        --with scipy --with matplotlib --with pandas \\
        python reproduce/experiments/ivat_embeddings/run_pca_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "reproduce"))
sys.path.insert(0, _HERE)
import common as C  # noqa: E402
import run_embedding_comparison as RE  # noqa: E402  (CACHE, CORPORA, EMBEDS)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from faiss.contrib.e_means import NumpyEmeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from tribbleclustering import FuzzyCMeans, KMeans  # noqa: E402

DIMS = [None, 32, 64]  # None = original dimension
SEEDS = list(C.SEEDS)
RAW_JSON = os.path.join(C.OUTPUT_DIR, "pca_compare_raw.json")


def reduce(Xb, Xq, dim):
    if dim is None or dim >= Xb.shape[1]:
        return Xb, Xq, Xb.shape[1]
    p = PCA(n_components=dim, random_state=0).fit(Xb)
    return p.transform(Xb), p.transform(Xq), dim


def run_method(name, Xb, k, s):
    if name == "k-means":
        m = KMeans(k, random_state=s).fit(Xb)
        return np.asarray(m.labels_), None
    if name == "FCM (m=2)":
        m = FuzzyCMeans(k, m=2.0, random_state=s).fit(Xb)
        return (
            np.asarray(m.labels_),
            np.asarray(m.membership_matrix_).max(axis=1).mean(),
        )
    if name == "E-means (5 ep)":
        _, lab = NumpyEmeans(Xb.shape[1], k, seed=s).train(
            np.ascontiguousarray(Xb, np.float32)
        )
        return np.asarray(lab).ravel(), None
    raise ValueError(name)


METHODS = ["k-means", "FCM (m=2)", "E-means (5 ep)"]


def main():
    raw, rows = {}, []
    for corp in RE.CORPORA:
        for emb in RE.EMBEDS:
            d = np.load(os.path.join(RE.CACHE, f"{corp}.npz"))
            Xb0 = d[f"{emb}_base"].astype(np.float64)
            Xq0 = d[f"{emb}_query"].astype(np.float64)
            yb = d["y_base"]
            k = int(d["n_classes"])
            key = f"{corp}|{emb}"
            raw[key] = {}
            print(f"== {corp} / {emb}: k={k}", flush=True)
            for dim in DIMS:
                Xb, _, dd = reduce(Xb0, Xq0, dim)
                dlabel = f"d={dd}" + (
                    " (PCA)"
                    if dim is not None and dim < Xb0.shape[1]
                    else " (original)"
                )
                for meth in METHODS:
                    ari, maxm, smin, smax, secs = [], [], [], [], []
                    for s in SEEDS:
                        t0 = time.perf_counter()
                        lab, mm = run_method(meth, Xb, k, s)
                        secs.append(time.perf_counter() - t0)
                        ari.append(adjusted_rand_score(yb, lab))
                        sizes = np.bincount(lab, minlength=k)
                        smin.append(int(sizes.min()))
                        smax.append(int(sizes.max()))
                        if mm is not None:
                            maxm.append(float(mm))
                    raw[key][f"{meth}|{dlabel}"] = {
                        "ari": ari,
                        "min_size": smin,
                        "max_size": smax,
                        "max_membership": maxm,
                        "sec": secs,
                    }
                    mm_txt = f"{np.mean(maxm):.3f}" if maxm else "—"
                    rows.append(
                        [
                            corp,
                            emb,
                            dlabel,
                            meth,
                            f"{np.mean(ari):.3f} ± {np.std(ari, ddof=1):.3f}",
                            f"{np.mean(smin):.0f} / {np.mean(smax):.0f}",
                            mm_txt,
                            f"{np.mean(secs):.2f}",
                        ]
                    )
                    print(
                        f"  {dlabel:16s} {meth:16s} ARI={np.mean(ari):.3f} sizes(min/max)={np.mean(smin):.0f}/{np.mean(smax):.0f} "
                        f"max-membership={mm_txt}",
                        flush=True,
                    )

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh)
    C.emit(
        "table_pca_before_clustering",
        "PCA before clustering: ARI and collapse diagnostics (mean over 10 seeds)",
        [
            "corpus",
            "embedding",
            "dimension",
            "method",
            "ARI (mean ± std)",
            "min / max cluster size",
            "mean max-membership (FCM)",
            "seconds",
        ],
        rows,
        note="PCA fitted on the base set, centred, no whitening. Uniform FCM membership = 1/k. A min cluster size of 0 means an empty cluster.",
        seeds=SEEDS,
    )

    # figure: ARI by dimension, per method, one panel per corpus/embedding
    combos = [(c, e) for c in RE.CORPORA for e in RE.EMBEDS]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    dim_labels = None
    for ax, (c, e) in zip(axes.ravel(), combos):
        keys = [kk for kk in raw[f"{c}|{e}"] if kk.startswith("k-means")]
        dim_labels = [kk.split("|")[1] for kk in keys]
        x = np.arange(len(dim_labels))
        w = 0.27
        for i, meth in enumerate(METHODS):
            mu = [np.mean(raw[f"{c}|{e}"][f"{meth}|{dl}"]["ari"]) for dl in dim_labels]
            sd = [
                np.std(raw[f"{c}|{e}"][f"{meth}|{dl}"]["ari"], ddof=1)
                for dl in dim_labels
            ]
            ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=2, label=meth)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                dl.replace(" (original)", "\n(orig.)").replace(" (PCA)", "\n(PCA)")
                for dl in dim_labels
            ],
            fontsize=7,
        )
        ax.set_title(f"{c} / {e}", fontsize=9)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("ARI vs labels (mean ± std, 10 seeds)")
    axes[1, 0].set_ylabel("ARI vs labels (mean ± std, 10 seeds)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "PCA before clustering: k-means, FCM and E-means by dimension", fontsize=11
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    C.save_figure(fig, "fig_pca_before_clustering")
    plt.close(fig)
    print("raw:", RAW_JSON, flush=True)


if __name__ == "__main__":
    main()
