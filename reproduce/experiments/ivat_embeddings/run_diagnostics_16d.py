"""Why the variations are so bad, and a direct 16-dimensional embedding for comparison.

Part A: diagnostics (figures)
  A1 distance contrast: how far apart nearest and farthest neighbours are, per embedding.
     Relative contrast = (dmax - dmin) / dmin per point; near-zero means all distances look alike.
  A2 FCM membership: histogram of the largest membership per document, against uniform 1/k.
  A3 IVAT profile: superdiagonal of the iVAT image (successive link distances in VAT order) on
     a 1,500-document subsample, with the k-1 largest jumps (where block cuts fall) marked,
     and the iVAT image for two embeddings. Subsample for display only.

Part B: direct 16-dimensional embedding
  TF-IDF is reduced directly to 16 dimensions (TruncatedSVD on the TF-IDF matrix), not by
  PCA of the 128-d SVD vectors. The TF-IDF matrix is rebuilt by replaying build_embeddings.py's
  selection (same seed, same call order) and checked against the cached labels. MiniLM has no
  native 16-d form, so its 16-d comparison is PCA-16. Methods: k-means, E-means (5 ep), FCM
  (m=2), IVATMeans. Ten seeds for baselines, one run for IVATMeans.

Run (repo root), in the tribble-cluster environment built outside the submodule:
    UV_PROJECT_ENVIRONMENT=$HOME/.cache/grad-school/venv-tribble-cluster \\
      uv run --project tribble-cluster --with faiss-cpu==1.16.0 --with scikit-learn \\
        --with scipy --with matplotlib --with pandas --with datasets \\
        python reproduce/experiments/ivat_embeddings/run_diagnostics_16d.py
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
import run_embedding_comparison as RE  # noqa: E402
import build_embeddings as BE  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.decomposition import PCA, TruncatedSVD  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from tribbleclustering import FuzzyCMeans, IVATMeans  # noqa: E402
from tribbleclustering.pvat import compute_ivat  # noqa: E402
from faiss.contrib.e_means import NumpyEmeans  # noqa: E402

SEEDS = list(C.SEEDS)
RAW_JSON = os.path.join(C.OUTPUT_DIR, "diag16_raw.json")
DIAG_SAMPLE = 2000
IVAT_SAMPLE = 1500


def replay_tfidf():
    """Rebuild each corpus's final base TF-IDF matrix by replaying build_embeddings.py's selection."""
    rng = np.random.default_rng(BE.SEED)
    out = {}
    for name, repo in BE.CORPORA.items():
        tr_text, tr_y = BE.load_split(repo, "train")
        te_text, te_y = BE.load_split(repo, "test")
        ib = [int(v) for v in rng.choice(len(tr_text), BE.N_BASE, replace=False)]
        iq = [int(v) for v in rng.choice(len(te_text), BE.N_QUERY, replace=False)]
        vec, tf_b, tf_q, ib, iq, _ = BE.select_nonempty(rng, tr_text, te_text, ib, iq)
        cache = np.load(os.path.join(RE.CACHE, f"{name}.npz"))
        assert np.array_equal(
            tr_y[ib], cache["y_base"]
        ), f"{name}: replayed labels do not match the cache"
        out[name] = tf_b
    return out


def relative_contrast(X, rng):
    idx = rng.choice(len(X), min(DIAG_SAMPLE, len(X)), replace=False)
    Z = X[idx]
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    dmin = D.min(axis=1)
    np.fill_diagonal(D, -np.inf)
    dmax = D.max(axis=1)
    rc = (dmax - dmin) / dmin
    pair = D[np.isfinite(D)]
    return float(np.median(rc)), pair / np.median(pair)


def main():
    rng = np.random.default_rng(1)
    tfidf_raw = replay_tfidf()
    raw, rows, diag = {}, [], {}

    for corp in RE.CORPORA:
        d = np.load(os.path.join(RE.CACHE, f"{corp}.npz"))
        yb = d["y_base"]
        k = int(d["n_classes"])
        st = d["st_base"].astype(np.float64)
        tf128 = d["tfidf_base"].astype(np.float64)
        tf16 = TruncatedSVD(16, random_state=0).fit_transform(tfidf_raw[corp])
        tf16 = tf16 / (np.linalg.norm(tf16, axis=1, keepdims=True) + 1e-12)
        embs = {
            "MiniLM 384 (original)": st,
            "MiniLM PCA-16": PCA(16, random_state=0).fit_transform(st),
            "TF-IDF 128 (original)": tf128,
            "TF-IDF PCA-16 (of 128)": PCA(16, random_state=0).fit_transform(tf128),
            "TF-IDF SVD-16 (direct)": tf16,
        }
        for ename, X in embs.items():
            key = f"{corp}|{ename}"
            raw[key] = {}
            rc, _ = relative_contrast(X, rng)
            raw[key]["median_relative_contrast"] = rc
            print(
                f"== {corp} / {ename}: d={X.shape[1]} median relative contrast={rc:.3f}",
                flush=True,
            )

            # baselines: ten seeds
            for meth in ["k-means", "E-means (5 ep)", "FCM (m=2)"]:
                ari, sizes_min, sizes_max, maxm = [], [], [], []
                for s in SEEDS:
                    if meth == "k-means":
                        lab = np.asarray(RE.m_kmeans(X, k, s)[0])
                    elif meth == "E-means (5 ep)":
                        _, lab = NumpyEmeans(X.shape[1], k, seed=s).train(
                            np.ascontiguousarray(X, np.float32)
                        )
                        lab = np.asarray(lab).ravel()
                    else:
                        f = FuzzyCMeans(k, m=2.0, random_state=s).fit(X)
                        lab = np.asarray(f.labels_)
                        if s == 0:
                            maxm = np.asarray(f.membership_matrix_).max(axis=1)
                    ari.append(adjusted_rand_score(yb, lab))
                    sz = np.bincount(lab, minlength=k)
                    sizes_min.append(int(sz.min()))
                    sizes_max.append(int(sz.max()))
                raw[key][meth] = {
                    "ari": ari,
                    "min_size": sizes_min,
                    "max_size": sizes_max,
                }
                print(
                    f"  {meth:16s} ARI={np.mean(ari):.3f} ± {np.std(ari, ddof=1):.3f} sizes(min/max)={np.mean(sizes_min):.0f}/{np.mean(sizes_max):.0f}",
                    flush=True,
                )
                if meth == "FCM (m=2)":
                    diag.setdefault("fcm", {})[key] = {"maxm": maxm, "k": k}

            # IVATMeans: one run
            t0 = time.perf_counter()
            ivm = IVATMeans(n_clusters=k, random_state=0).fit(X)
            sec = time.perf_counter() - t0
            lab = np.asarray(ivm.labels_)
            ari_i = adjusted_rand_score(yb, lab)
            sz = np.bincount(lab, minlength=k)
            raw[key]["IVATMeans (medoid)"] = {
                "ari": ari_i,
                "min_size": int(sz.min()),
                "max_size": int(sz.max()),
                "sec": sec,
            }
            print(
                f"  IVATMeans        ARI={ari_i:.3f} sizes(min/max)={int(sz.min())}/{int(sz.max())} sec={sec:.2f}",
                flush=True,
            )

            rows.append(
                [corp, ename, str(X.shape[1]), f"{rc:.2f}"]
                + [
                    f"{np.mean(raw[key][m]['ari']):.3f} ± {np.std(raw[key][m]['ari'], ddof=1):.3f}"
                    for m in ["k-means", "E-means (5 ep)", "FCM (m=2)"]
                ]
                + [f"{raw[key]['IVATMeans (medoid)']['ari']:.3f}"]
            )

            if corp in ("ag_news", "newsgroups20") and ename in (
                "MiniLM 384 (original)",
                "TF-IDF SVD-16 (direct)",
            ):
                idx = np.random.default_rng(2).choice(
                    len(X), IVAT_SAMPLE, replace=False
                )
                Z = X[idx]
                Dz = cdist(Z, Z)
                ivat_img, _, order = compute_ivat(Dz.copy(), inplace=False)
                diag.setdefault("ivat", {})[key] = {
                    "image": np.asarray(ivat_img),
                    "super": np.diag(np.asarray(ivat_img), 1).copy(),
                    "k": k,
                }

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh)

    C.emit(
        "table_direct16_comparison",
        "Direct 16-dimensional embedding and diagnostics: ARI and relative contrast",
        [
            "corpus",
            "embedding",
            "dim",
            "median relative contrast",
            "k-means",
            "E-means 5 ep",
            "FCM (m=2)",
            "IVATMeans",
        ],
        rows,
        note="Baselines: mean ± std over 10 seeds. IVATMeans: one run. Relative contrast = (dmax - dmin) / dmin, median over 2,000 sampled documents; near 0 means distances barely separate neighbours from strangers.",
        seeds=SEEDS,
    )

    # A1: distance contrast
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    labels_rc, vals_rc = [], []
    for corp in RE.CORPORA:
        for ename in ["MiniLM 384 (original)", "TF-IDF SVD-16 (direct)"]:
            labels_rc.append(f"{corp}\n{ename.split(' (')[0]}")
            vals_rc.append(raw[f"{corp}|{ename}"]["median_relative_contrast"])
    ax.bar(range(len(vals_rc)), vals_rc)
    ax.set_xticks(range(len(vals_rc)))
    ax.set_xticklabels(labels_rc, fontsize=7)
    ax.set_ylabel("median relative contrast (dmax - dmin)/dmin")
    ax.set_title(
        "Distance contrast: how much nearest and farthest neighbours differ", fontsize=9
    )
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    for corp in ["newsgroups20", "ag_news"]:
        X = np.load(os.path.join(RE.CACHE, f"{corp}.npz"))["st_base"].astype(np.float64)
        idx = np.random.default_rng(3).choice(len(X), 1000, replace=False)
        D = cdist(X[idx], X[idx])
        pair = D[np.triu_indices(len(idx), 1)]
        ax.hist(
            pair / np.median(pair),
            bins=80,
            density=True,
            alpha=0.5,
            label=f"{corp} (MiniLM)",
        )
    ax.set_xlabel("pairwise distance / median")
    ax.set_ylabel("density")
    ax.set_title("MiniLM pairwise distances concentrate near the median", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    C.save_figure(fig, "fig_diag_distance_contrast")
    plt.close(fig)

    # A2: FCM membership histograms
    keys = list(diag["fcm"].keys())
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, key in zip(axes.ravel(), keys):
        mm = diag["fcm"][key]["maxm"]
        k = diag["fcm"][key]["k"]
        ax.hist(mm, bins=60, range=(0, 1), color="tab:orange")
        ax.axvline(1.0 / k, color="k", ls="--", lw=1)
        ax.set_title(key.replace("|", "\n"), fontsize=7)
        ax.set_xlabel("largest membership", fontsize=7)
    for ax in axes.ravel()[len(keys) :]:
        ax.axis("off")
    fig.suptitle(
        "FCM (m=2): largest membership per document; dashed = uniform 1/k", fontsize=10
    )
    fig.tight_layout()
    C.save_figure(fig, "fig_diag_fcm_membership")
    plt.close(fig)

    # A3: iVAT profile and image
    ikeys = list(diag["ivat"].keys())
    fig, axes = plt.subplots(2, len(ikeys), figsize=(4.2 * len(ikeys), 7.5))
    for col, key in enumerate(ikeys):
        ivat = diag["ivat"][key]["image"]
        sup = diag["ivat"][key]["super"]
        k = diag["ivat"][key]["k"]
        ax = axes[0, col]
        ax.imshow(ivat, cmap="viridis", aspect="auto")
        ax.set_title(key.replace("|", "\n"), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1, col]
        ax.plot(sup, lw=0.8)
        top = np.argsort(-sup)[: k - 1]
        ax.scatter(
            top, sup[top], color="red", s=14, zorder=3, label=f"{k-1} largest jumps"
        )
        ax.set_xlabel("position in VAT order", fontsize=7)
        ax.set_ylabel("link distance", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "iVAT image (top) and link-distance profile along the VAT order (bottom); 1,500-document subsample",
        fontsize=10,
    )
    fig.tight_layout()
    C.save_figure(fig, "fig_diag_ivat_profile")
    plt.close(fig)

    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh)
    print("raw:", RAW_JSON, flush=True)


if __name__ == "__main__":
    main()
