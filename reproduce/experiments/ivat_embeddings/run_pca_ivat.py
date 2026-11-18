"""IVATMeans under PCA, alongside the PCA comparison of k-means, FCM and E-means.

Same settings as run_pca_comparison.py: original dimension, PCA-32, PCA-64, PCA fitted on the
base set only (centred, no whitening), k = number of classes. IVATMeans has no random state,
so each setting is one run. Output: table_pca_ivat.{md,csv} and pca_ivat_raw.json.

Run (repo root), in the tribble-cluster environment built outside the submodule:
    UV_PROJECT_ENVIRONMENT=$HOME/.cache/grad-school/venv-tribble-cluster \\
      uv run --project tribble-cluster --with faiss-cpu==1.16.0 --with scikit-learn \\
        --with scipy --with pandas python reproduce/experiments/ivat_embeddings/run_pca_ivat.py
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

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from tribbleclustering import IVATMeans  # noqa: E402

DIMS = [None, 32, 64]
RAW_JSON = os.path.join(C.OUTPUT_DIR, "pca_ivat_raw.json")


def main():
    raw, rows = {}, []
    for corp in RE.CORPORA:
        for emb in RE.EMBEDS:
            d = np.load(os.path.join(RE.CACHE, f"{corp}.npz"))
            Xb0 = d[f"{emb}_base"].astype(np.float64)
            yb = d["y_base"]
            k = int(d["n_classes"])
            key = f"{corp}|{emb}"
            raw[key] = {}
            for dim in DIMS:
                if dim is None or dim >= Xb0.shape[1]:
                    Xb, dd, tag = Xb0, Xb0.shape[1], "original"
                else:
                    Xb = PCA(n_components=dim, random_state=0).fit(Xb0).transform(Xb0)
                    dd, tag = dim, "PCA"
                IVATMeans(n_clusters=k, random_state=0).fit(Xb[:200])  # JIT warm-up
                t0 = time.perf_counter()
                lab = np.asarray(IVATMeans(n_clusters=k, random_state=0).fit(Xb).labels_)
                sec = time.perf_counter() - t0
                ari = adjusted_rand_score(yb, lab)
                sizes = np.bincount(lab, minlength=k)
                dlabel = f"d={dd} ({tag})"
                raw[key][dlabel] = {"ari": ari, "min_size": int(sizes.min()), "max_size": int(sizes.max()),
                                    "sec": sec, "labels": lab.tolist()}
                rows.append([corp, emb, dlabel, f"{ari:.3f}", f"{int(sizes.min())} / {int(sizes.max())}", f"{sec:.2f}"])
                print(f"{corp:14s} {emb:6s} {dlabel:18s} ARI={ari:.3f} sizes={int(sizes.min())}/{int(sizes.max())} sec={sec:.2f}", flush=True)

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh)
    C.emit("table_pca_ivat", "IVATMeans under PCA (one run per setting; no random state)",
           ["corpus", "embedding", "dimension", "ARI", "min / max cluster size", "seconds"], rows,
           note="Same PCA settings as table_pca_before_clustering. PCA fitted on the base set, centred, no whitening. IVATMeans has no random state, so each row is one run.",
           seeds=[0])
    print("raw:", RAW_JSON, flush=True)


if __name__ == "__main__":
    main()
