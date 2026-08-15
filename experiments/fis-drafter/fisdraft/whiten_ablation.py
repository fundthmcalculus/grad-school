"""Ablation: is PCA-whitening necessary, and what replaces it?

The one-class detector decorrelates features with PCA-whitening before fitting
per-component Gaussians, because the product-t-norm rule assumes independence.
This ablates that choice, scoring with the trimmed log-domain score throughout:

  raw            no transform (correlated features) -- the null: does the rule
                 work at all without decorrelation?
  standardize    per-feature z-score only (diagonal covariance, NO rotation) --
                 removes scale but not correlation.
  pca_k          PCA-whitening keeping k components (the current method) --
                 decorrelate + unit-variance + rank reduction, swept over k.
  pca_full       PCA-whitening, all components -- decorrelation without the
                 rank cut, to separate "decorrelate" from "reduce".
  zca            ZCA (symmetric) whitening, full rank -- decorrelates while
                 staying in the original feature basis (no rotation to PCs).

Reports det@1%FP and within-length AUROC on deepset (easy) and safeguard (hard),
so the effect of removing / replacing PCA is visible where the method works and
where it struggles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .fmri_detect import layer_features
from .injection_detect_v2 import within_len_auc


def det_at(y, s, cap=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    h = tpr[fpr <= cap]
    return float(h[-1]) if len(h) else 0.0


def transform(kind, Xtr, Xall, k=32):
    """Return whitened/standardized Xall given the fit rows Xtr."""
    mu = Xtr.mean(0)
    if kind == "raw":
        return Xall - mu
    if kind == "standardize":
        sd = Xtr.std(0) + 1e-9
        return (Xall - mu) / sd
    Xc = Xtr - mu
    # eigdecomposition of the covariance
    cov = np.cov(Xc.T) + 1e-6 * np.eye(Xc.shape[1])
    w, V = np.linalg.eigh(cov)         # ascending
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    if kind.startswith("pca"):
        kk = Xc.shape[1] if kind == "pca_full" else min(k, Xc.shape[1])
        Wm = V[:, :kk] / np.sqrt(w[:kk] + 1e-9)      # PCA-whiten to kk comps
        return (Xall - mu) @ Wm
    if kind == "zca":
        Wm = V @ np.diag(1.0 / np.sqrt(w + 1e-9)) @ V.T   # ZCA (symmetric)
        return (Xall - mu) @ Wm
    raise ValueError(kind)


def trimmed_score(Z, fit_idx, trim=2):
    """Per-component surprisal under fitted Gaussians, trimmed sum."""
    mu = Z[fit_idx].mean(0)
    sd = Z[fit_idx].std(0) + 1e-9
    S = 0.5 * ((Z - mu) / sd) ** 2
    if trim > 0 and S.shape[1] > trim:
        S = np.sort(S, 1)[:, : S.shape[1] - trim]
    return S.sum(1)


def run(rundir: Path, seeds=6, k=32) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    mid = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]
    kinds = ["raw", "standardize", "pca_16", "pca_32", "pca_full", "zca"]
    acc = {kk: {"d1": [], "wl": []} for kk in kinds}

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = np.where(y == 0)[0]; rng.shuffle(ben); inj = np.where(y == 1)[0]
        fit = ben[: int(0.6 * len(ben))]; tb = ben[int(0.6 * len(ben)):]
        X, _ = layer_features(act, fit, 8)
        ti = np.r_[tb, inj]; yt = np.r_[np.zeros(len(tb)), np.ones(len(inj))]
        tlt = np.r_[tl[tb], tl[inj]]
        fit_pos = np.arange(len(fit))                 # fit rows are first in X? no
        for kk in kinds:
            kind = "pca" if kk == "pca_16" or kk == "pca_32" else kk
            kv = 16 if kk == "pca_16" else k
            Z = transform("pca" if kk in ("pca_16", "pca_32") else kk,
                          X[fit], X, k=kv)
            s = trimmed_score(Z, fit)  # fit indices are 0..len(fit)-1 in X order
            acc[kk]["d1"].append(det_at(yt, s[ti], 0.01))
            acc[kk]["wl"].append(within_len_auc(yt, s[ti], tlt))
    return {"model": mid, "dataset": rundir.name,
            "arms": {kk: {"det@1%FP": round(float(np.mean(v["d1"])), 3),
                          "wl_auroc": round(float(np.mean(v["wl"])), 3)}
                     for kk, v in acc.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["runs/injection_qwen3b", "runs/sg_qwen3b"])
    a = ap.parse_args()
    allout = []
    print("Whitening / PCA ablation (trimmed log-domain score, one-class):\n")
    print("%-22s %-14s %10s %10s" % ("transform", "dataset", "det@1%FP", "wl-AUROC"))
    for r in a.runs:
        res = run(Path(r))
        allout.append(res)
        ds = "deepset" if "sg_" not in r and "spml" not in r else r.split("_")[0].split("/")[-1]
        for kk, v in res["arms"].items():
            print("%-22s %-14s %10.3f %10.3f" % (kk, ds, v["det@1%FP"], v["wl_auroc"]))
        print()
    Path("runs/whiten_ablation.json").write_text(json.dumps(allout, indent=2))


if __name__ == "__main__":
    main()
