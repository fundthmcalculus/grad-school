"""Refine the FIS anomaly detector after construction, and time it.

Construction (moment-matched Gaussians, uniform layer weights) is ~0.04 ms and
gives the one-class detector of Part 3/4. This asks whether a refinement stage
beats it, and which optimizer to use.

Two refinement regimes, kept separate because they make different promises:

  UNSUPERVISED (no attacks)   replace each feature's single Gaussian with a
      K-component mixture fitted by EM -- a richer 'normal' density. Keeps the
      no-attack-examples guarantee. If benign activations are multi-modal this
      should help; if they are effectively unimodal it will not.

  FEW-SHOT (a handful of attacks)  the FIS gives a per-layer anomaly vector
      S[i, l]; refinement learns a layer weighting w from a small validation set
      of known attacks and scores as w . S. This is the realistic deployment
      regime -- lots of benign traffic, a few known attacks -- and it is where
      an optimizer choice actually matters, because the AUROC-surrogate
      objective over w is non-convex.

The few-shot objective is a smooth Wilcoxon-Mann-Whitney surrogate for AUROC
(mean over benign/attack pairs of sigmoid((s_attack - s_benign)/tau)), optimised
several ways so the methods can be compared on the same objective, wall clock,
and held-out AUROC. The uniform-weight baseline is the pure one-class detector
and uses zero attacks; every refined number is reported against it.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")


def per_layer_scores(act, fit_idx, per_layer_pca=8, gmm_k=1, seed=0):
    """S[i, l] = anomaly contribution of layer l for prompt i.

    gmm_k=1 is the moment-matched single Gaussian (construction). gmm_k>1 fits a
    K-component mixture per layer by EM (the unsupervised refinement): the
    anomaly is the negative log-likelihood under that mixture, whitened.
    """
    n, Lp1, D = act.shape
    S = np.zeros((n, Lp1))
    t0 = time.time()
    for l in range(Lp1):
        A = act[:, l, :]
        k = min(per_layer_pca, D, len(fit_idx) - 1)
        p = PCA(n_components=k, whiten=True, random_state=seed).fit(A[fit_idx])
        Z = p.transform(A)
        if gmm_k <= 1:
            mu, sd = Z[fit_idx].mean(0), Z[fit_idx].std(0)
            sd = np.maximum(sd, 1e-3 * np.abs(mu).mean() + 1e-6)
            S[:, l] = 0.5 * (((Z - mu) / sd) ** 2).mean(1)
        else:
            gm = GaussianMixture(
                n_components=min(gmm_k, max(1, len(fit_idx) // 20)),
                covariance_type="diag", random_state=seed, reg_covar=1e-4,
            ).fit(Z[fit_idx])
            S[:, l] = -gm.score_samples(Z) / Z.shape[1]
    return S, time.time() - t0


# --------------------------------------------------------------------------
# Few-shot objective and optimizers over layer weights
# --------------------------------------------------------------------------


def soft_auc(w, Sb, Sa, tau=0.5):
    """Smooth AUROC surrogate for weight vector w (log-weights -> softplus)."""
    wpos = np.log1p(np.exp(w))
    sb, sa = Sb @ wpos, Sa @ wpos
    d = (sa[:, None] - sb[None, :]) / tau
    return float(1.0 / (1.0 + np.exp(-d)).mean())


def fit_weights(Sb_val, Sa_val, method, seed=0):
    """Optimise layer weights on a small val set. Returns (w_pos, seconds)."""
    L = Sb_val.shape[1]
    rng = np.random.default_rng(seed)
    neg = lambda w: -soft_auc(w, Sb_val, Sa_val)
    w0 = np.zeros(L)
    t0 = time.time()

    if method == "uniform":
        return np.ones(L), 0.0
    if method == "logistic":
        from sklearn.linear_model import LogisticRegression

        X = np.vstack([Sb_val, Sa_val])
        y = np.r_[np.zeros(len(Sb_val)), np.ones(len(Sa_val))]
        m = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
        return np.maximum(m.coef_[0], 0) + 1e-6, time.time() - t0
    if method == "lda":
        mb, ma = Sb_val.mean(0), Sa_val.mean(0)
        cov = np.cov(np.vstack([Sb_val, Sa_val]).T) + 1e-4 * np.eye(L)
        w = np.linalg.solve(cov, ma - mb)
        return np.maximum(w, 0) + 1e-6, time.time() - t0
    if method == "lbfgs":
        r = minimize(neg, w0, method="L-BFGS-B")
        return np.log1p(np.exp(r.x)), time.time() - t0
    if method == "neldermead":
        r = minimize(neg, w0, method="Nelder-Mead",
                     options={"maxiter": 2000, "xatol": 1e-3, "fatol": 1e-4})
        return np.log1p(np.exp(r.x)), time.time() - t0
    if method == "powell":
        r = minimize(neg, w0, method="Powell", options={"maxiter": 2000})
        return np.log1p(np.exp(r.x)), time.time() - t0
    if method == "diffevo":
        r = differential_evolution(
            neg, [(-4, 4)] * L, seed=seed, maxiter=60, tol=1e-4,
            polish=True, updating="deferred", workers=1,
        )
        return np.log1p(np.exp(r.x)), time.time() - t0
    if method == "coord":
        w = w0.copy()
        best = neg(w)
        for _ in range(15):
            improved = False
            for j in range(L):
                for step in (0.5, -0.5, 0.15, -0.15):
                    wt = w.copy(); wt[j] += step
                    v = neg(wt)
                    if v < best - 1e-6:
                        w, best, improved = wt, v, True
            if not improved:
                break
        return np.log1p(np.exp(w)), time.time() - t0
    raise ValueError(method)


def auc_within_len(y, s, tl, nbin=8):
    dec = pd.qcut(pd.Series(tl), nbin, labels=False, duplicates="drop").to_numpy()
    tot_w = tot = 0.0
    for v in np.unique(dec):
        m = dec == v
        if m.sum() < 15 or len(np.unique(y[m])) < 2:
            continue
        tot += roc_auc_score(y[m], s[m]) * m.sum(); tot_w += m.sum()
    return tot / tot_w if tot_w else float("nan")


def run(rundir: Path, variant="mean", seed=0, n_vals=(10, 25, 50)) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / f"act_{variant}.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    rng = np.random.default_rng(seed)
    ben = np.where(y == 0)[0]; rng.shuffle(ben)
    inj = np.where(y == 1)[0]; rng.shuffle(inj)

    fit = ben[: int(0.5 * len(ben))]           # density fit (benign only)
    ben_rest = ben[int(0.5 * len(ben)) :]
    ben_val = ben_rest[: len(ben_rest) // 2]
    ben_test = ben_rest[len(ben_rest) // 2 :]

    out: dict = {"model": json.loads((rundir / "meta.json").read_text())["config"]["model_id"],
                 "n_layers": act.shape[1], "unsupervised": {}, "few_shot": {}}

    # ---- construction + unsupervised refinement (no attacks) -------------
    for k in (1, 2, 3, 5):
        S, t_build = per_layer_scores(act, fit, gmm_k=k, seed=seed)
        s_uniform = S.mean(1)
        yt = np.r_[np.zeros(len(ben_test)), np.ones(len(inj))]
        st = np.r_[s_uniform[ben_test], s_uniform[inj]]
        out["unsupervised"][f"gmm_k={k}"] = {
            "test_auroc": round(roc_auc_score(yt, st), 3),
            "test_within_len": round(auc_within_len(yt, st, np.r_[tl[ben_test], tl[inj]]), 3),
            "build_ms": round(t_build * 1000, 1),
        }
    # keep the single-Gaussian per-layer scores for few-shot
    S, _ = per_layer_scores(act, fit, gmm_k=1, seed=seed)

    # ---- few-shot layer-weight refinement --------------------------------
    methods = ["uniform", "logistic", "lda", "lbfgs", "neldermead", "powell",
               "diffevo", "coord"]
    for nv in n_vals:
        inj_val, inj_test = inj[:nv], inj[nv:]
        if len(inj_test) < 20:
            continue
        Sb_val, Sa_val = S[ben_val], S[inj_val]
        yt = np.r_[np.zeros(len(ben_test)), np.ones(len(inj_test))]
        tlt = np.r_[tl[ben_test], tl[inj_test]]
        row = {}
        for m in methods:
            w, secs = fit_weights(Sb_val, Sa_val, m, seed=seed)
            st = np.r_[S[ben_test] @ w, S[inj_test] @ w]
            row[m] = {
                "test_auroc": round(roc_auc_score(yt, st), 3),
                "within_len": round(auc_within_len(yt, st, tlt), 3),
                "refine_ms": round(secs * 1000, 1),
            }
        out["few_shot"][f"n_val={nv}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/injection")
    ap.add_argument("--variant", default="mean")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    rundir = Path(a.run)
    # average few-shot over seeds (attack val sampling is noisy at small n)
    runs = [run(rundir, variant=a.variant, seed=s) for s in range(a.seeds)]
    base = runs[0]

    print(f"== {base['model']} ==")
    print("\nUNSUPERVISED refinement (no attacks), test within-len AUROC:")
    for k, v in base["unsupervised"].items():
        print(f"  {k:10s} auroc {v['test_auroc']:.3f}  within-len {v['test_within_len']:.3f}"
              f"  build {v['build_ms']:.0f} ms")

    print("\nFEW-SHOT layer-weight refinement (within-len AUROC, mean over "
          f"{a.seeds} seeds):")
    for nvkey in base["few_shot"]:
        print(f"  {nvkey}:")
        methods = list(base["few_shot"][nvkey])
        for m in methods:
            wl = np.mean([r["few_shot"][nvkey][m]["within_len"] for r in runs])
            au = np.mean([r["few_shot"][nvkey][m]["test_auroc"] for r in runs])
            ms = np.mean([r["few_shot"][nvkey][m]["refine_ms"] for r in runs])
            print(f"    {m:12s} within-len {wl:.3f}  auroc {au:.3f}  {ms:7.1f} ms")

    (rundir / f"refine_{a.variant}.json").write_text(
        json.dumps({"per_seed": runs}, indent=2))


if __name__ == "__main__":
    main()
