"""IVATMeans for embedding clustering and similarity search: k-means vs E-means vs FCM vs IVATMeans.

Inputs: the per-corpus .npz files from build_embeddings.py (base set N=10,000 labelled
documents, 500 held-out queries; st = MiniLM, tfidf = TF-IDF->SVD128).

Two questions, same data and k (= number of classes) for every method:

  A. Clustering quality and cost: ARI and NMI against the document labels, and wall clock.
     Baselines run over ten seeds (the seed sets the initialisation). IVATMeans has no
     random state in its construction, so it is run once per dataset and its labelling
     is checked for identity across seeds.

  B. IVF-style similarity search. Each method's centroids route a query to its nearest
     `nprobe` cells, the base documents in those cells are scored by cosine, and the top
     10 is returned. Reported against exact cosine search over the whole base set:
     recall@10 (overlap with the exact top 10), label precision@10 (fraction of the
     returned documents sharing the query's label), and scan fraction (share of the base
     set scored). An exact-search row is the reference.

Pre-stated expectations (before the run): none of these is a pass/fail gate. The aim is to
see where IVATMeans sits on the cost/recall curve, and whether it matches the baselines on
clustering quality for embedding geometry, which is closer to the prototype-friendly
case than the Table 3.5 constructions.

Run (repo root), in the tribble-cluster environment built outside the submodule:
    UV_PROJECT_ENVIRONMENT=$HOME/.cache/grad-school/venv-tribble-cluster \\
      uv run --project tribble-cluster --with faiss-cpu==1.16.0 --with scikit-learn \\
        --with scipy --with matplotlib --with pandas \\
        python reproduce/experiments/ivat_embeddings/run_embedding_comparison.py
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
import common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from faiss.contrib.e_means import NumpyEmeans  # noqa: E402
from sklearn.cluster import kmeans_plusplus  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402
from tribbleclustering import FuzzyCMeans, IVATMeans, KMeans  # noqa: E402

CACHE = os.environ.get("REPRO_EMBED_CACHE", os.path.expanduser("~/.cache/grad-school/embeddings"))
CORPORA = ["ag_news", "emotion", "newsgroups20"]
EMBEDS = ["st", "tfidf"]
SEEDS = list(C.SEEDS)
TOPK = 10
NPROBES = (1, 2, 3)
RAW_JSON = os.path.join(C.OUTPUT_DIR, "ivat_embed_raw.json")


# --------------------------------------------------------------------------- #
# methods: return (labels, centres); labels index the centres
# --------------------------------------------------------------------------- #
def m_kmeans(X, k, s):
    m = KMeans(k, random_state=s).fit(X)
    return np.asarray(m.labels_), np.asarray(m.cluster_centers_)


def m_fcm(X, k, s):
    m = FuzzyCMeans(k, random_state=s).fit(X)
    return np.asarray(m.labels_), np.asarray(m.cluster_centers_)


def m_ivat(X, k, s):
    m = IVATMeans(n_clusters=k, random_state=s).fit(X)
    return np.asarray(m.labels_), np.asarray(m.cluster_centers_)


def m_em_default(X, k, s):
    C_, lab = NumpyEmeans(X.shape[1], k, seed=s).train(np.ascontiguousarray(X, np.float32))
    return np.asarray(lab).ravel(), np.asarray(C_)


def m_em_kpp25(X, k, s):
    c0, _ = kmeans_plusplus(X, k, random_state=s)
    C_, lab = NumpyEmeans(X.shape[1], k, nepochs=25, seed=s).train(
        np.ascontiguousarray(X, np.float32), init_centroids=np.ascontiguousarray(c0, np.float32)
    )
    return np.asarray(lab).ravel(), np.asarray(C_)


METHODS = [
    ("k-means (k-means++)", m_kmeans, "seeds"),
    ("E-means (5 ep, default init)", m_em_default, "seeds"),
    ("E-means (25 ep, k-means++ init)", m_em_kpp25, "seeds"),
    ("FCM (m=2)", m_fcm, "seeds"),
    ("IVATMeans (medoid)", m_ivat, "once"),
]


def purity(y, lab):
    tot = 0
    for c in np.unique(lab):
        _, cnt = np.unique(y[lab == c], return_counts=True)
        tot += cnt.max()
    return tot / len(y)


# --------------------------------------------------------------------------- #
# similarity search
# --------------------------------------------------------------------------- #
def search_metrics(Xb, yb, Xq, yq, labels, centres):
    S = Xq @ Xb.T
    exact_top = np.argpartition(-S, TOPK, axis=1)[:, :TOPK]
    exact_prec = np.mean([np.mean(yb[exact_top[i]] == yq[i]) for i in range(len(Xq))])
    d2 = ((Xq[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    order = np.argsort(d2, axis=1)
    out = {"exact_precision": float(exact_prec)}
    for p in NPROBES:
        rec, prec, scan = [], [], []
        for i in range(len(Xq)):
            cand = np.flatnonzero(np.isin(labels, order[i, :p]))
            if cand.size == 0:
                rec.append(0.0); prec.append(0.0); scan.append(0.0)
                continue
            top = cand[np.argsort(-S[i, cand])[:TOPK]]
            rec.append(len(set(top.tolist()) & set(exact_top[i].tolist())) / TOPK)
            prec.append(float(np.mean(yb[top] == yq[i])))
            scan.append(cand.size / len(Xb))
        out[f"nprobe{p}"] = {"recall": float(np.mean(rec)), "precision": float(np.mean(prec)),
                             "scan": float(np.mean(scan))}
    return out


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if v.size > 1 else f"{v.mean():.3f}"


def main():
    raw = {}
    clus_rows, srch_rows = [], []
    for corp in CORPORA:
        for emb in EMBEDS:
            path = os.path.join(CACHE, f"{corp}.npz")
            d = np.load(path)
            Xb, Xq = d[f"{emb}_base"].astype(np.float64), d[f"{emb}_query"].astype(np.float64)
            yb, yq = d["y_base"], d["y_query"]
            k = int(d["n_classes"])
            key = f"{corp}|{emb}"
            raw[key] = {}
            print(f"== {corp} / {emb}: N={len(Xb)} d={Xb.shape[1]} k={k} queries={len(Xq)}", flush=True)
            for name, fn, mode in METHODS:
                seeds = SEEDS if mode == "seeds" else [0]
                ari, nmi, pur, secs, labs_all, search_runs = [], [], [], [], [], []
                for s in seeds:
                    fn(Xb[:200], k, s)  # JIT / warm-up, outside the timed call
                    t0 = time.perf_counter()
                    lab, cen = fn(Xb, k, s)
                    secs.append(time.perf_counter() - t0)
                    ari.append(adjusted_rand_score(yb, lab))
                    nmi.append(normalized_mutual_info_score(yb, lab))
                    pur.append(purity(yb, lab))
                    labs_all.append(lab.tobytes())
                    search_runs.append(search_metrics(Xb, yb, Xq, yq, lab, cen))
                distinct = len(set(labs_all))
                raw[key][name] = {"ari": ari, "nmi": nmi, "purity": pur, "sec": secs,
                                  "distinct_labelings": distinct, "search": search_runs}
                clus_rows.append([corp, emb, name, ms(ari), ms(nmi), ms(pur), ms(secs), str(distinct)])
                row = [corp, emb, name, f"{np.mean([r['exact_precision'] for r in search_runs]):.3f}"]
                for p in NPROBES:
                    rr = [r[f"nprobe{p}"] for r in search_runs]
                    row += [ms([x["recall"] for x in rr]), ms([x["precision"] for x in rr]),
                            ms([x["scan"] for x in rr])]
                srch_rows.append(row)
                print(f"  {name:32s} ARI={ms(ari)}  sec={ms(secs)}  distinct={distinct}  "
                      f"recall@10 nprobe1={ms([r['nprobe1']['recall'] for r in search_runs])}", flush=True)

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh)

    clus_header = ["corpus", "embedding", "method", "ARI", "NMI", "purity", "seconds", "distinct labelings"]
    C.emit("table_ivat_embed_clustering", "IVATMeans for embedding clustering: quality and cost (mean ± std)",
           clus_header, clus_rows,
           note="Baselines over 10 seeds; IVATMeans run once (no random state in its construction), its distinct-labelings count is over the single run.",
           seeds=SEEDS)
    srch_header = ["corpus", "embedding", "method", "exact precision@10"] + [
        h for p in NPROBES for h in (f"nprobe{p} recall@10", f"nprobe{p} precision@10", f"nprobe{p} scan")]
    C.emit("table_ivat_embed_search", "IVF-style similarity search on method cells (mean ± std over seeds)",
           srch_header, srch_rows,
           note="500 queries per corpus against 10,000 base documents, cosine. Exact cosine search is the reference (recall 1.0 by definition). Scan = share of base scored per query.",
           seeds=SEEDS)

    # figure: ARI by corpus x embedding
    fig, ax = plt.subplots(figsize=(10, 4.8))
    combos = [(c, e) for c in CORPORA for e in EMBEDS]
    names = [n for n, _, _ in METHODS]
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        mu = [np.mean(raw[f"{c}|{e}"][nm]["ari"]) for c, e in combos]
        sd = [np.std(raw[f"{c}|{e}"][nm]["ari"], ddof=1) if len(raw[f"{c}|{e}"][nm]["ari"]) > 1 else 0 for c, e in combos]
        ax.bar(np.arange(len(combos)) + (i - len(names) / 2 + 0.5) * w, mu, w, yerr=sd, capsize=2, label=nm)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([f"{c}\n{e}" for c, e in combos], fontsize=8)
    ax.set_ylabel("ARI vs document labels (mean ± std)"); ax.set_ylim(0, 1)
    ax.set_title("Embedding clustering: ARI by corpus and embedding (N = 10,000, k = classes)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    C.save_figure(fig, "fig_ivat_embed_ari")
    plt.close(fig)

    # figure: recall@10 vs scan fraction (search cost/quality trade-off)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=False, sharey=True)
    for ax, (c, e) in zip(axes.ravel(), combos):
        for nm in names:
            runs = raw[f"{c}|{e}"][nm]["search"]
            xs = [np.mean([r[f"nprobe{p}"]["scan"] for r in runs]) for p in NPROBES]
            ys = [np.mean([r[f"nprobe{p}"]["recall"] for r in runs]) for p in NPROBES]
            ax.plot(xs, ys, marker="o", label=nm)
        ax.set_xscale("log")
        ax.set_title(f"{c} / {e}", fontsize=9)
        ax.set_xlabel("share of base scanned (log)", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("recall@10 vs exact"); axes[1, 0].set_ylabel("recall@10 vs exact")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("IVF-style search: recall@10 vs scan cost (nprobe = 1, 2, 3)", fontsize=10)
    fig.tight_layout()
    C.save_figure(fig, "fig_ivat_embed_search")
    plt.close(fig)
    print("raw:", RAW_JSON, flush=True)


if __name__ == "__main__":
    main()
