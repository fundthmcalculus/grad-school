"""Deep exploration of regularized-background-covariance (relative Mahalanobis).

Relative Mahalanobis (Ren et al. 2021): score = MD_foreground - MD_background,
where the background is a *more regularized* covariance. The idea is to subtract
"generic" variation so the score isolates deviations in the structured
directions. The naive diagonal-background version failed earlier; this sweeps the
design space properly on RAW per-layer features (not whitened — whitening sets
Sigma=I and the construction degenerates).

Covariances use shrinkage  Sigma(l) = (1-l) S + l * gamma * I,  gamma = tr(S)/d,
so they are always invertible on the ~279-dim features from ~200 samples.

Arms (all zero-shot, benign-only fit):
  trimmed_pca32   the current method (PCA-whiten 32 + trimmed surprisal) -- baseline
  maha_lw         Ledoit-Wolf-shrunk full-covariance Mahalanobis on raw features
                  (is a well-regularized plain Mahalanobis already better?)
  maha_shrink_l   full Mahalanobis at fixed shrinkage l (sweep)
  rmd_bg_l        relative MD: MD_fg(lw) - MD_bg(l), sweeping the BACKGROUND
                  shrinkage l -- the knob the user wants explored
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from .fmri_detect import layer_features
from .injection_detect_v2 import within_len_auc


def d1(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    h = tpr[fpr <= 0.01]
    return float(h[-1]) if len(h) else 0.0


def shrink_cov(S, lam):
    d = S.shape[0]
    gamma = np.trace(S) / d
    return (1 - lam) * S + lam * gamma * np.eye(d)


def md(X, mu, prec):
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)


def run(rundir: Path, seeds=6) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    mid = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]
    bg_lams = [0.3, 0.5, 0.7, 0.9, 0.99]
    fg_lams = [0.05, 0.2, 0.5]
    arms = ["trimmed_pca32", "maha_lw"] + [f"maha_shrink_{l}" for l in fg_lams] \
        + [f"rmd_bg_{l}" for l in bg_lams]
    acc = {a: {"d1": [], "wl": []} for a in arms}

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = np.where(y == 0)[0]; rng.shuffle(ben); inj = np.where(y == 1)[0]
        fit = ben[: int(0.6 * len(ben))]; tb = ben[int(0.6 * len(ben)):]
        ti = np.r_[tb, inj]; yt = np.r_[np.zeros(len(tb)), np.ones(len(inj))]
        tlt = np.r_[tl[tb], tl[inj]]
        X, _ = layer_features(act, fit, 8)
        mu = X[fit].mean(0)
        Xc = X - mu

        # baseline: PCA-whiten 32 + trimmed surprisal
        Z = PCA(n_components=32, whiten=True, random_state=seed).fit(X[fit]).transform(X)
        S2 = 0.5 * Z ** 2
        trim = np.sort(S2, 1)[:, : S2.shape[1] - 2].sum(1)
        acc["trimmed_pca32"]["d1"].append(d1(yt, trim[ti]))
        acc["trimmed_pca32"]["wl"].append(within_len_auc(yt, trim[ti], tlt))

        # Ledoit-Wolf foreground precision
        lw = LedoitWolf().fit(X[fit])
        prec_fg = lw.precision_
        md_fg = md(X, mu, prec_fg)
        acc["maha_lw"]["d1"].append(d1(yt, md_fg[ti]))
        acc["maha_lw"]["wl"].append(within_len_auc(yt, md_fg[ti], tlt))

        S = np.cov(Xc[fit].T)
        for l in fg_lams:
            prec = np.linalg.pinv(shrink_cov(S, l))
            m = md(X, mu, prec)
            acc[f"maha_shrink_{l}"]["d1"].append(d1(yt, m[ti]))
            acc[f"maha_shrink_{l}"]["wl"].append(within_len_auc(yt, m[ti], tlt))

        # relative MD: foreground = Ledoit-Wolf, background = shrink(l), sweep l.
        # scale-match the two MDs (both ~chi2_d) before subtracting.
        for l in bg_lams:
            prec_bg = np.linalg.pinv(shrink_cov(S, l))
            md_bg = md(X, mu, prec_bg)
            fg_n = md_fg / md_fg[fit].mean()
            bg_n = md_bg / md_bg[fit].mean()
            rmd = fg_n - bg_n
            acc[f"rmd_bg_{l}"]["d1"].append(d1(yt, rmd[ti]))
            acc[f"rmd_bg_{l}"]["wl"].append(within_len_auc(yt, rmd[ti], tlt))

    return {"model": mid, "dataset": rundir.name,
            "arms": {a: {"det@1%FP": round(float(np.mean(v["d1"])), 3),
                         "wl_auroc": round(float(np.mean(v["wl"])), 3)}
                     for a, v in acc.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["runs/injection_qwen3b", "runs/sg_qwen3b_big", "runs/spml_qwen3b"])
    a = ap.parse_args()
    allout = []
    print("Regularized-background-covariance exploration (zero-shot, det@1%FP | wl-AUROC):\n")
    for r in a.runs:
        res = run(Path(r))
        allout.append(res)
        ds = "deepset" if ("sg_" not in r and "spml" not in r) else r.split("/")[-1].split("_")[0]
        print(f"== {res['model'].split('/')[-1]} · {ds} ==")
        for aa, v in res["arms"].items():
            print("  %-18s %.3f | %.3f" % (aa, v["det@1%FP"], v["wl_auroc"]))
        print()
    Path("runs/relmaha_explore.json").write_text(json.dumps(allout, indent=2))


if __name__ == "__main__":
    main()
