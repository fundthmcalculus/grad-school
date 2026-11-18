"""IVATMeans under different distance metrics, on the embedding corpora.

`IVATMeans(metric=...)` only controls how new points are assigned to the medoids (the
predict step). The VAT/iVAT ordering and the minimax medoids are computed from the L2
distance matrix inside `fit`, and the constructor does not change that. To test a metric
properly, this sweep replaces the library's distance stage (the module-level
`ivatmeans._pairwise_distances`) with scipy's metric for the duration of one fit, and sets
the same name as the estimator's `metric` so assignment agrees with the ordering. The
patch is restored in a `finally` block.

Metrics: euclidean (the library's own L2 path, reproduced through the patch as a check),
manhattan (scipy cityblock), cosine. Each on the six corpus/embedding combinations from
build_embeddings.py, with N = 10,000 base documents and k = number of classes.

Manipulation checks, recorded in the output:
  C1  the library's unpatched default fit reproduces the ARI from run_embedding_comparison
      (same inputs, same estimator), so the sweep's baseline is the library as shipped;
  C2  the patch changes the result: the labelings must differ across metrics on at least
      one combination, or the knob did not bite;
  C3  the patched euclidean run agrees with the unpatched default (ARI between the two
      labelings), so the patch itself does not alter the L2 path.

IVATMeans has no random state, so each metric is run once per combination.

Run (repo root), in the tribble-cluster environment built outside the submodule:
    UV_PROJECT_ENVIRONMENT=$HOME/.cache/grad-school/venv-tribble-cluster \\
      uv run --project tribble-cluster --with faiss-cpu==1.16.0 --with scikit-learn \\
        --with scipy --with matplotlib --with pandas \\
        python reproduce/experiments/ivat_embeddings/run_metric_sweep.py
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
import run_embedding_comparison as RE  # noqa: E402  (search_metrics, purity, CACHE, corpora)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402
import tribbleclustering.ivatmeans as IVM  # noqa: E402
from tribbleclustering import IVATMeans  # noqa: E402

# estimator metric name -> scipy metric name
METRICS = {"euclidean": "euclidean", "manhattan": "cityblock", "cosine": "cosine"}
RAW_JSON = os.path.join(C.OUTPUT_DIR, "ivat_metric_raw.json")
NAN_COUNT = {"n": 0}  # NaN distance entries replaced in the most recent patched fit
BASELINE_JSON = os.path.join(C.OUTPUT_DIR, "ivat_embed_raw.json")


def fit_default(X, k):
    """The library as shipped: no patch, metric=None."""
    m = IVATMeans(n_clusters=k, random_state=0).fit(X)
    return np.asarray(m.labels_), np.asarray(m.cluster_centers_)


def fit_patched(X, k, metric_name):
    """Distance stage replaced by scipy's `metric`; the estimator's metric set to match."""
    scipy_name = METRICS[metric_name]
    original = IVM._pairwise_distances

    def patched(Z):
        D = cdist(Z, Z, metric=scipy_name)
        # Cosine distance is undefined when a row is all zeros (a document whose TF-IDF
        # vector is empty). Such entries are set to the maximal dissimilarity, 1, and
        # counted. Unhandled NaN crashes the compiled IVAT kernel (double free), so the
        # count is recorded per fit rather than silently absorbed.
        bad = np.isnan(D)
        NAN_COUNT["n"] = int(bad.sum())
        D[bad] = 1.0
        np.fill_diagonal(D, 0.0)
        return D.astype(np.float32)

    IVM._pairwise_distances = patched
    try:
        m = IVATMeans(n_clusters=k, metric=metric_name, random_state=0).fit(X)
    finally:
        IVM._pairwise_distances = original
    return np.asarray(m.labels_), np.asarray(m.cluster_centers_)


def main():
    baseline = json.load(open(BASELINE_JSON)) if os.path.exists(BASELINE_JSON) else {}
    raw, clus_rows, srch_rows = {}, [], []
    label_store = {}
    for corp in RE.CORPORA:
        for emb in RE.EMBEDS:
            d = np.load(os.path.join(RE.CACHE, f"{corp}.npz"))
            Xb = d[f"{emb}_base"].astype(np.float64)
            Xq = d[f"{emb}_query"].astype(np.float64)
            yb, yq = d["y_base"], d["y_query"]
            k = int(d["n_classes"])
            key = f"{corp}|{emb}"
            raw[key] = {}
            print(f"== {corp} / {emb}: N={len(Xb)} d={Xb.shape[1]} k={k}", flush=True)

            # C1: unpatched library default
            fit_default(Xb[:200], k)
            t0 = time.perf_counter()
            lab0, cen0 = fit_default(Xb, k)
            sec0 = time.perf_counter() - t0
            ari0 = adjusted_rand_score(yb, lab0)
            prev = baseline.get(key, {}).get("IVATMeans (medoid)", {}).get("ari", [None])
            prev = prev[0] if isinstance(prev, list) else prev
            print(f"  default (unpatched)  ARI={ari0:.3f}  previous run={prev}  sec={sec0:.2f}", flush=True)
            raw[key]["default_unpatched"] = {"ari": ari0, "sec": sec0, "labels": lab0.tolist()}
            label_store[(key, "default")] = lab0

            for mname in METRICS:
                fit_patched(Xb[:200], k, mname)
                t0 = time.perf_counter()
                lab, cen = fit_patched(Xb, k, mname)
                sec = time.perf_counter() - t0
                nan_n = NAN_COUNT["n"]
                ari = adjusted_rand_score(yb, lab)
                nmi = normalized_mutual_info_score(yb, lab)
                pur = RE.purity(yb, lab)
                search = RE.search_metrics(Xb, yb, Xq, yq, lab, cen)
                raw[key][mname] = {"ari": ari, "nmi": nmi, "purity": pur, "sec": sec,
                                   "nan_replaced": nan_n, "labels": lab.tolist(), "search": search}
                if nan_n:
                    print(f"  NOTE {mname}: {nan_n} undefined distance entries set to 1 (zero-vector rows)", flush=True)
                label_store[(key, mname)] = lab
                clus_rows.append([corp, emb, mname, f"{ari:.3f}", f"{nmi:.3f}", f"{pur:.3f}", f"{sec:.2f}"])
                srch_rows.append([corp, emb, mname, f"{search['exact_precision']:.3f}"] + [
                    f"{search[f'nprobe{p}']['recall']:.3f} / {search[f'nprobe{p}']['scan']:.3f}" for p in RE.NPROBES])
                print(f"  {mname:10s} ARI={ari:.3f} NMI={nmi:.3f} sec={sec:.2f} "
                      f"recall@10 nprobe1={search['nprobe1']['recall']:.3f} scan={search['nprobe1']['scan']:.3f}",
                      flush=True)

            # C3: patched euclidean vs unpatched default
            c3 = adjusted_rand_score(lab0, label_store[(key, "euclidean")])
            raw[key]["check_C3_ari_patched_euclid_vs_default"] = c3
            print(f"  C3 ARI(patched euclidean, default) = {c3:.4f}", flush=True)

    # C2: does the metric change the partition?
    changed = []
    for corp in RE.CORPORA:
        for emb in RE.EMBEDS:
            key = f"{corp}|{emb}"
            pairs = [("euclidean", "manhattan"), ("euclidean", "cosine"), ("manhattan", "cosine")]
            agree = {f"{a} vs {b}": adjusted_rand_score(label_store[(key, a)], label_store[(key, b)])
                     for a, b in pairs}
            raw[key]["check_C2_ari_between_metrics"] = agree
            changed.append((key, agree))

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh)

    C.emit("table_ivat_metric_clustering", "IVATMeans under different distance metrics: clustering quality (N = 10,000)",
           ["corpus", "embedding", "metric", "ARI", "NMI", "purity", "seconds"], clus_rows,
           note="Distance stage replaced by the named scipy metric for each fit; one run per metric (IVATMeans has no random state). Default unpatched library run in ivat_metric_raw.json.",
           seeds=[0])
    C.emit("table_ivat_metric_search", "IVATMeans under different distance metrics: IVF search (recall@10 / scan share, nprobe 1-3)",
           ["corpus", "embedding", "metric", "exact precision@10"] + [f"nprobe{p} recall / scan" for p in RE.NPROBES],
           srch_rows,
           note="Routing uses Euclidean distance to the medoids (as in run_embedding_comparison); only the cell construction varies with the metric.",
           seeds=[0])

    # figure: ARI by metric
    combos = [(c, e) for c in RE.CORPORA for e in RE.EMBEDS]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    names = list(METRICS)
    w = 0.8 / len(names)
    for i, mname in enumerate(names):
        vals = [raw[f"{c}|{e}"][mname]["ari"] for c, e in combos]
        ax.bar(np.arange(len(combos)) + (i - len(names) / 2 + 0.5) * w, vals, w, label=f"IVATMeans, {mname}")
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([f"{c}\n{e}" for c, e in combos], fontsize=8)
    ax.set_ylabel("ARI vs document labels"); ax.set_ylim(-0.05, 0.5)
    ax.set_title("IVATMeans by distance metric (N = 10,000, k = classes)")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    C.save_figure(fig, "fig_ivat_metric_ari")
    plt.close(fig)
    print("raw:", RAW_JSON, flush=True)


if __name__ == "__main__":
    main()
