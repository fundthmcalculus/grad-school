"""Stage 4 -- PCA/SVD-free representations for the anomaly rule.

Motivation. PCA is fit unsupervised on the truthful split, so it keeps the
*highest-variance* directions -- which need not be the ones that separate
truthful from confabulated. §3.3 found no low-rank truthful subspace at all
(effective rank 174-248 of 960), so truncating to 32-64 components was
discarding signal on principle. This stage removes PCA and SVD entirely.

Protocol is unchanged and is the one asked for: fit on **accurate
(question, answer) pairs only**, then let the anomaly rule flag the wrong pairs.
No hallucination is seen during fitting; selection on val, reporting on test.

Four PCA-free feature families:

  stats      19 output-distribution statistics (already projection-free)
  rawdim     raw residual-stream coordinates at one layer, chosen by tribble's
             own discriminant ranking over the known-good modes
  layerstat  per-layer summary scalars over all 33 layers -- L2 norm, mean, std,
             max|a|, kurtosis. Interpretable: "layer 20's norm is unusual."
  centroid   per-layer cosine similarity + L2 distance to the truthful centroid
             (a mean, not a decomposition)

Every representation is scored two ways so the two failure modes can be told
apart:

  * Mahalanobis (full covariance)  -- is the REPRESENTATION any good?
  * tribble FIS anomaly rule       -- can the fuzzy rule exploit it?

If Mahalanobis improves but the FIS does not, the limit is the diagonal-Gaussian
antecedent structure, not the features.
"""

import argparse
import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
)

from analyze import DATA, SCALAR_COLS, SEED, fpr_at_tpr, make_splits
from norm_sweep import anomaly_score, membership_tensors

warnings.filterwarnings("ignore")

BEST_LAYER = 20          # from the §3.2 validation sweep
POOLING = "prompt"       # the winning pooling site


# --------------------------------------------------------------------------
# PCA-free feature builders
# --------------------------------------------------------------------------

def f_stats(meta, hidden, sp):
    return meta[SCALAR_COLS].reset_index(drop=True).astype(float)


def f_rawdim(meta, hidden, sp, layer=BEST_LAYER):
    """Raw residual-stream coordinates -- no projection at all."""
    X = hidden[:, layer, :].astype(np.float32)
    return pd.DataFrame(X, columns=[f"D{i:04d}" for i in range(X.shape[1])])


def f_layerstat(meta, hidden, sp):
    """Per-layer summary scalars. 33 layers x 5 = 165 features, no projection."""
    cols = {}
    for li in range(hidden.shape[1]):
        A = hidden[:, li, :].astype(np.float32)
        cols[f"L{li:02d}_norm"] = np.linalg.norm(A, axis=1)
        cols[f"L{li:02d}_mean"] = A.mean(1)
        cols[f"L{li:02d}_std"] = A.std(1)
        cols[f"L{li:02d}_absmax"] = np.abs(A).max(1)
        cols[f"L{li:02d}_kurt"] = sstats.kurtosis(A, axis=1)
    return pd.DataFrame(cols)


def f_centroid(meta, hidden, sp):
    """Per-layer geometry relative to the truthful centroid (a mean, not a basis)."""
    cols = {}
    for li in range(hidden.shape[1]):
        A = hidden[:, li, :].astype(np.float32)
        c = A[sp["fit"]].mean(0, keepdims=True)          # fit split only
        d = A - c
        cols[f"L{li:02d}_dist"] = np.linalg.norm(d, axis=1)
        cols[f"L{li:02d}_cos"] = ((A @ c.T).ravel()
                                  / (np.linalg.norm(A, axis=1) * np.linalg.norm(c) + 1e-8))
    return pd.DataFrame(cols)


BUILDERS = {"stats": f_stats, "rawdim": f_rawdim,
            "layerstat": f_layerstat, "centroid": f_centroid}


def drop_constant(F, fit_idx, tol=1e-8):
    """Drop features with no variance on the fit split.

    Needed because layer 0 of the `prompt` pooling site is constant by
    construction: the last prompt token is always the same chat-template token
    (`assistant\\n`), so its embedding — and hence its distance to any centroid —
    is identical for every example. `L00_dist` is exactly 0 and `L00_cos`
    exactly 1.

    Beyond carrying no signal, such a column crashes tribblefis's
    `calculate_gaussian_correlation`, which histograms each feature over
    (min, max) and cannot build bins on a degenerate range.
    """
    v = F.iloc[fit_idx].var()
    keep = list(v[v > tol].index)
    return F[keep]


# --------------------------------------------------------------------------

def rank_features(F, fit_idx, labels, top_n):
    """Pick antecedents. Uses tribble's discriminant ranking when K>=2.

    With a single known-good class the pairwise ranking is identically zero
    (see FINDINGS §1), so fall back to variance -- still unsupervised and still
    blind to the hallucination class.
    """
    if len(set(labels)) >= 2:
        with contextlib.redirect_stdout(io.StringIO()):
            diff = calculate_gaussian_correlation(
                F.iloc[fit_idx].reset_index(drop=True),
                pd.Series([f"mode{c}" for c in labels]))
        _, feats = take_top_features(diff, top_n=top_n)
        return feats
    v = F.iloc[fit_idx].var().sort_values(ascending=False)
    return list(v.index[:top_n])


def modes(F, feats, fit_idx, K):
    Xf = StandardScaler().fit_transform(F.iloc[fit_idx][feats].to_numpy())
    if K <= 1:
        return np.zeros(len(fit_idx), dtype=int), np.nan
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Xf)
    return km.labels_, silhouette_score(Xf, km.labels_)


def score_all(F, feats, sp, labels, pairs, theta):
    """Return {name: score_fn(idx)} for Mahalanobis + each FIS operator pair."""
    Xf = StandardScaler().fit(F.iloc[sp["fit"]][feats].to_numpy())
    lw = LedoitWolf().fit(Xf.transform(F.iloc[sp["fit"]][feats].to_numpy()))
    out = {"Mahalanobis": lambda ix: lw.mahalanobis(
        Xf.transform(F.iloc[ix][feats].to_numpy()))}

    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        model = create_gaussian_membership_dict(
            F.iloc[sp["fit"]][feats].reset_index(drop=True), y_modes,
            top_n_var_names=feats)

    def make(tn, sn):
        def fn(ix):
            classes, tens = membership_tensors(F, ix, model, feats)
            with np.errstate(all="ignore"):
                return anomaly_score(classes, tens, tn, sn, theta)
        return fn

    for tn, sn in pairs:
        out[f"FIS T={tn}/S={sn}"] = make(tn, sn)
    return out, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--layer", type=int, default=BEST_LAYER)
    args = ap.parse_args()

    meta = pd.read_parquet(DATA / "capture_meta.parquet")
    hidden = np.load(DATA / f"capture_hidden_{POOLING}.npy")
    sp = make_splits(meta)
    print(f"pooling={POOLING} layer={args.layer} | fit={len(sp['fit'])} "
          f"test_neg={len(sp['test_neg'])} test_tq={len(sp['test_tq'])} "
          f"test_fp={len(sp['test_fp'])}")

    # Best matched + best mismatched pair from the §3.5 sweep, plus product.
    PAIRS = [("product", "product"), ("hamacher", "hamacher"),
             ("dombi2", "einstein")]
    K_GRID = [1, 2, 4, 8]
    TOP_N = [8, 16, 32]

    built = {}
    for name, fn in BUILDERS.items():
        built[name] = fn(meta, hidden, sp)
        print(f"  built {name:<10} {built[name].shape[1]:>4} features")
    # combinations of the winning scalar family with each hidden-state family
    for name in ("rawdim", "layerstat", "centroid"):
        built[f"stats+{name}"] = pd.concat([built["stats"], built[name]], axis=1)

    rows = []
    for fname, F in built.items():
        F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for K in K_GRID:
            for tn in TOP_N:
                if tn > F.shape[1]:
                    continue
                # rank on a provisional single-mode pass, then re-rank per K
                feats0 = rank_features(F, sp["fit"], np.zeros(len(sp["fit"])), tn)
                labels, sil = modes(F, feats0, sp["fit"], K)
                feats = rank_features(F, sp["fit"], labels, tn)
                scorers, model = score_all(F, feats, sp, labels, PAIRS, args.theta)

                for det, fn in scorers.items():
                    for tag, pk in [("TriviaQA", "test_tq"),
                                    ("FalsePremise", "test_fp")]:
                        ix = np.concatenate([sp["test_neg"], sp[pk]])
                        y = np.concatenate([np.zeros(len(sp["test_neg"])),
                                            np.ones(len(sp[pk]))])
                        s = np.asarray(fn(ix), dtype=float)
                        rec = {"features": fname, "K": K, "top_n": tn,
                               "silhouette": sil, "detector": det, "family": tag,
                               "mfs": model.n_membership_functions}
                        if not np.isfinite(s).all() or np.ptp(s) == 0:
                            rows.append({**rec, "auroc": np.nan, "auprc": np.nan,
                                         "fpr@95tpr": np.nan})
                        else:
                            rows.append({**rec, "auroc": roc_auc_score(y, s),
                                         "auprc": average_precision_score(y, s),
                                         "fpr@95tpr": fpr_at_tpr(y, s)})
            print(f"  {fname:<16} K={K} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "nopca_results.csv", index=False)

    # ---------------- reporting ------------------------------------------
    df["kind"] = np.where(df.detector == "Mahalanobis", "Mahalanobis", "FIS")

    for tag in ("TriviaQA", "FalsePremise"):
        sub = df[df.family == tag].dropna(subset=["auroc"])
        piv = sub.pivot_table(index="features", columns="kind",
                              values="auroc", aggfunc="max")
        base = pd.read_csv(DATA / f"baselines_{tag.lower()}.csv")
        pca_fis = {"TriviaQA": 0.643, "FalsePremise": 0.819}[tag]

        print(f"\n{'='*74}\n{tag} -- best AUROC per PCA-free representation\n{'='*74}")
        print(piv.sort_values("Mahalanobis", ascending=False)
              .to_string(float_format=lambda v: f"{v:.3f}"))
        print(f"\n  reference | best baseline (with PCA where used): "
              f"{base.iloc[0]['detector']} {base.iloc[0]['auroc']:.3f}")
        print(f"  reference | best FIS WITH PCA:                    {pca_fis:.3f}")
        best = sub.nlargest(1, "auroc").iloc[0]
        print(f"  best PCA-free overall: {best.detector} on {best.features} "
              f"(K={best.K}, top_n={best.top_n}) = {best.auroc:.3f}")

        print(f"\n  top 8 PCA-free configurations:")
        print(sub.nlargest(8, "auroc")[["features", "detector", "K", "top_n",
                                        "auroc", "auprc", "fpr@95tpr"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
