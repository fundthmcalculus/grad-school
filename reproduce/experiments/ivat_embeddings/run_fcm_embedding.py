"""FCM membership vectors as cluster embeddings: clustering quality and similarity search.

Reading of "FCM for cluster embeddings": each document is represented by its fuzzy
membership vector across k clusters (k = number of classes), and that k-dimensional
vector is used both as a clustering (argmax) and as an embedding for similarity search.
The question is how well the membership space preserves the neighbourhood structure of
the original embedding, and how the fuzzifier m controls that trade-off.

For each corpus/embedding and fuzzifier m in M_VALUES, ten seeds:
  - fit FCM(k, m) on the base set (tribble FuzzyCMeans), take its centres;
  - membership of base and query documents from the standard FCM formula
    u_ij = 1 / sum_l (d_ij / d_il)^(2/(m-1)), Euclidean d, with the fitted centres;
  - ARI of the argmax labelling against the document labels (clustering quality);
  - the membership vector, L2-normalised, is the embedding. Exact cosine top-10 in the
    membership space is compared with exact cosine top-10 in the original embedding:
    recall@10 (overlap) and label precision@10.

Reference rows: exact search in the original embedding (recall 1 by definition).

Run (repo root), in the tribble-cluster environment built outside the submodule:
    UV_PROJECT_ENVIRONMENT=$HOME/.cache/grad-school/venv-tribble-cluster \\
      uv run --project tribble-cluster --with faiss-cpu==1.16.0 --with scikit-learn \\
        --with scipy --with matplotlib --with pandas \\
        python reproduce/experiments/ivat_embeddings/run_fcm_embedding.py
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
import run_embedding_comparison as RE  # noqa: E402  (CACHE, CORPORA, EMBEDS, purity)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from tribbleclustering import FuzzyCMeans  # noqa: E402

M_VALUES = [1.1, 1.5, 2.0, 3.0]
TOPK = 10
SEEDS = list(C.SEEDS)
RAW_JSON = os.path.join(C.OUTPUT_DIR, "fcm_embed_raw.json")


def membership(X, centres, m):
    d = cdist(X, centres) + 1e-12
    inv = d ** (-2.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)


def l2n(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def top_k(E, Q, k=TOPK):
    S = Q @ E.T
    return np.argpartition(-S, k, axis=1)[:, :k]


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if v.size > 1 else f"{v.mean():.3f}"


def main():
    raw, rows = {}, []
    for corp in RE.CORPORA:
        for emb in RE.EMBEDS:
            d = np.load(os.path.join(RE.CACHE, f"{corp}.npz"))
            Xb = l2n(d[f"{emb}_base"].astype(np.float64))
            Xq = l2n(d[f"{emb}_query"].astype(np.float64))
            yb, yq = d["y_base"], d["y_query"]
            k = int(d["n_classes"])
            key = f"{corp}|{emb}"
            raw[key] = {}
            print(f"== {corp} / {emb}: N={len(Xb)} d={Xb.shape[1]} k={k}", flush=True)

            # reference: exact search in the original embedding
            ref_top = top_k(Xb, Xq)
            ref_prec = float(
                np.mean([np.mean(yb[ref_top[i]] == yq[i]) for i in range(len(Xq))])
            )
            raw[key]["reference_exact_precision"] = ref_prec
            rows.append(
                [
                    corp,
                    emb,
                    "original embedding (exact)",
                    "—",
                    "1.000",
                    f"{ref_prec:.3f}",
                    "—",
                    "—",
                ]
            )

            for m in M_VALUES:
                ari, rec, prec, secs = [], [], [], []
                for s in SEEDS:
                    t0 = time.perf_counter()
                    fcm = FuzzyCMeans(k, m=m, random_state=s).fit(Xb)
                    secs.append(time.perf_counter() - t0)
                    cen = np.asarray(fcm.cluster_centers_)
                    Ub = membership(Xb, cen, m)
                    Uq = membership(Xq, cen, m)
                    ari.append(adjusted_rand_score(yb, Ub.argmax(axis=1)))
                    Eb, Eq = l2n(Ub), l2n(Uq)
                    top = top_k(Eb, Eq)
                    rec.append(
                        np.mean(
                            [
                                len(set(top[i]) & set(ref_top[i])) / TOPK
                                for i in range(len(Xq))
                            ]
                        )
                    )
                    prec.append(
                        np.mean([np.mean(yb[top[i]] == yq[i]) for i in range(len(Xq))])
                    )
                raw[key][f"m={m}"] = {
                    "ari": ari,
                    "recall": rec,
                    "precision": prec,
                    "sec": secs,
                }
                rows.append(
                    [
                        corp,
                        emb,
                        f"FCM membership, m = {m}",
                        ms(ari),
                        ms(rec),
                        ms(prec),
                        ms(secs),
                        str(k),
                    ]
                )
                print(
                    f"  m={m}: ARI={ms(ari)} recall@10={ms(rec)} precision@10={ms(prec)} sec={ms(secs)}",
                    flush=True,
                )

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh)

    header = [
        "corpus",
        "embedding",
        "representation",
        "ARI (argmax)",
        "recall@10 vs original",
        "precision@10 (labels)",
        "seconds",
        "k",
    ]
    C.emit(
        "table_fcm_membership_embedding",
        "FCM membership vectors as cluster embeddings: clustering and similarity search",
        header,
        rows,
        note="Representation: k-dim FCM membership, L2-normalised; search is exact cosine over the membership space. Recall is the overlap with the exact top 10 in the original embedding. Ten seeds.",
        seeds=SEEDS,
    )

    # figure: recall vs label precision by m
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    combos = [(c, e) for c in RE.CORPORA for e in RE.EMBEDS]
    for ax, (c, e) in zip(axes.ravel(), combos):
        xs = [np.mean(raw[f"{c}|{e}"][f"m={m}"]["recall"]) for m in M_VALUES]
        ys = [np.mean(raw[f"{c}|{e}"][f"m={m}"]["precision"]) for m in M_VALUES]
        ax.plot(xs, ys, marker="o")
        for m, x, y in zip(M_VALUES, xs, ys):
            ax.annotate(
                f"m={m}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points"
            )
        ax.axhline(
            raw[f"{c}|{e}"]["reference_exact_precision"], color="grey", ls=":", lw=1
        )
        ax.set_title(f"{c} / {e}", fontsize=9)
        ax.set_xlabel("recall@10 vs original space", fontsize=8)
        ax.set_ylabel("label precision@10", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "FCM membership embeddings: recall vs label precision as m varies (dotted: original-space precision)",
        fontsize=10,
    )
    fig.tight_layout()
    C.save_figure(fig, "fig_fcm_membership_tradeoff")
    plt.close(fig)
    print("raw:", RAW_JSON, flush=True)


if __name__ == "__main__":
    main()
