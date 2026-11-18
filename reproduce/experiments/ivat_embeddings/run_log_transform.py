"""Does a logarithm help? Log transforms on features and on distances.

Three transforms, each compared with its untransformed baseline on the same documents:

  F1  TF-IDF before SVD: log1p of the TF-IDF weights, then TruncatedSVD(16) and L2-normalise.
      Compressing the heavy tail of term weights before the projection.
  F2  MiniLM signed log: sign(x) * log1p(|x|) per coordinate. Compresses large components
      without changing sign (MiniLM values are signed, so plain log is undefined).
  D1  IVAT distances: log1p(distance) for IVATMeans. A monotone transform of distances leaves
      minimax (iVAT) order unchanged, so only the gap-based cut selection can move.

Baselines: the untransformed embeddings. Methods: k-means, E-means (5 ep), FCM (m=2), with ten
seeds; IVATMeans once. The TF-IDF matrix is replayed exactly as in run_diagnostics_16d.py, with
its labels checked against the cache.

Run (repo root), in the tribble-cluster environment built outside the submodule:
    UV_PROJECT_ENVIRONMENT=$HOME/.cache/grad-school/venv-tribble-cluster \\
      uv run --project tribble-cluster --with faiss-cpu==1.16.0 --with scikit-learn \\
        --with scipy --with matplotlib --with pandas --with datasets \\
        python reproduce/experiments/ivat_embeddings/run_log_transform.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "reproduce"))
sys.path.insert(0, _HERE)
import common as C  # noqa: E402
import run_embedding_comparison as RE  # noqa: E402
import run_diagnostics_16d as D16  # noqa: E402  (replay_tfidf)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from faiss.contrib.e_means import NumpyEmeans  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from tribbleclustering import FuzzyCMeans, IVATMeans  # noqa: E402
import tribbleclustering.ivatmeans as IVM  # noqa: E402

SEEDS = list(C.SEEDS)
RAW_JSON = os.path.join(C.OUTPUT_DIR, "log_transform_raw.json")


def l2n(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def run_baselines(X, k):
    out = {}
    for meth in ["k-means", "E-means (5 ep)", "FCM (m=2)"]:
        ari, smin, smax = [], [], []
        for s in SEEDS:
            if meth == "k-means":
                lab = np.asarray(RE.m_kmeans(X, k, s)[0])
            elif meth == "E-means (5 ep)":
                _, lab = NumpyEmeans(X.shape[1], k, seed=s).train(np.ascontiguousarray(X, np.float32))
                lab = np.asarray(lab).ravel()
            else:
                lab = np.asarray(FuzzyCMeans(k, m=2.0, random_state=s).fit(X).labels_)
            ari.append(adjusted_rand_score(RE_yb, lab))
            sz = np.bincount(lab, minlength=k)
            smin.append(int(sz.min())); smax.append(int(sz.max()))
        out[meth] = {"ari": ari, "min_size": smin, "max_size": smax}
    return out


def run_ivat(X, k, log_distance=False):
    orig = IVM._pairwise_distances

    def patched(Z):
        from scipy.spatial.distance import cdist as _cd
        Dm = _cd(Z, Z)
        if log_distance:
            Dm = np.log1p(Dm)
        return Dm.astype(np.float32)

    if log_distance:
        IVM._pairwise_distances = patched
    try:
        lab = np.asarray(IVATMeans(n_clusters=k, random_state=0).fit(X).labels_)
    finally:
        IVM._pairwise_distances = orig
    sz = np.bincount(lab, minlength=k)
    return {"ari": adjusted_rand_score(RE_yb, lab), "min_size": int(sz.min()), "max_size": int(sz.max())}


RE_yb = None


def main():
    global RE_yb
    tfidf_raw = D16.replay_tfidf()
    raw, rows = {}, []
    for corp in RE.CORPORA:
        d = np.load(os.path.join(RE.CACHE, f"{corp}.npz"))
        RE_yb = d["y_base"]
        k = int(d["n_classes"])
        st = d["st_base"].astype(np.float64)
        tf_raw = tfidf_raw[corp]

        tf_log = tf_raw.copy()
        tf_log.data = np.log1p(tf_log.data)
        plain = l2n(TruncatedSVD(16, random_state=0).fit_transform(tf_raw))
        logged = l2n(TruncatedSVD(16, random_state=0).fit_transform(tf_log))
        st_slog = signed_log(st)

        cases = {
            "MiniLM 384 (baseline)": (st, False),
            "MiniLM 384, signed log (F2)": (st_slog, False),
            "TF-IDF SVD-16 (baseline)": (plain, False),
            "TF-IDF SVD-16, log1p before SVD (F1)": (logged, False),
            "MiniLM 384, IVAT on log1p distances (D1)": (st, True),
            "TF-IDF SVD-16, IVAT on log1p distances (D1)": (plain, True),
        }
        for name, (X, logd) in cases.items():
            key = f"{corp}|{name}"
            if logd:
                ivat = run_ivat(X, k, log_distance=True)
                base = None
                raw[key] = {"IVATMeans (medoid)": ivat}
                rows.append([corp, name, "—", "—", "—", f"{ivat['ari']:.3f}", f"{ivat['min_size']} / {ivat['max_size']}"])
                print(f"{corp:14s} {name:46s} IVAT ARI={ivat['ari']:.3f} sizes={ivat['min_size']}/{ivat['max_size']}", flush=True)
                continue
            res = run_baselines(X, k)
            res["IVATMeans (medoid)"] = run_ivat(X, k)
            raw[key] = res
            a = {m: np.mean(res[m]["ari"]) for m in res}
            print(f"{corp:14s} {name:46s} km={a['k-means']:.3f} em={a['E-means (5 ep)']:.3f} "
                  f"fcm={a['FCM (m=2)']:.3f} ivat={a['IVATMeans (medoid)']['ari']:.3f}", flush=True)
            rows.append([corp, name,
                         f"{a['k-means']:.3f} ± {np.std(res['k-means']['ari'], ddof=1):.3f}",
                         f"{a['E-means (5 ep)']:.3f} ± {np.std(res['E-means (5 ep)']['ari'], ddof=1):.3f}",
                         f"{a['FCM (m=2)']:.3f} ± {np.std(res['FCM (m=2)']['ari'], ddof=1):.3f}",
                         f"{res['IVATMeans (medoid)']['ari']:.3f}",
                         f"{res['IVATMeans (medoid)']['min_size']} / {res['IVATMeans (medoid)']['max_size']}"])

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh)
    C.emit("table_log_transform",
           "Log transforms: ARI for k-means, E-means, FCM and IVATMeans against their baselines",
           ["corpus", "representation", "k-means", "E-means 5 ep", "FCM (m=2)", "IVATMeans", "IVATMeans min / max size"],
           rows,
           note="Baseline rows and transformed rows use the same documents. Baselines: mean ± std over 10 seeds. IVATMeans: one run. D1 rows are IVATMeans only.",
           seeds=SEEDS)
    print("raw:", RAW_JSON, flush=True)


if __name__ == "__main__":
    main()
