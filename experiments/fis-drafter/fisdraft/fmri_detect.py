"""One-class anomaly detection on the activation atlas.

Learn 'normal' from innocuous prompts only, then flag deviation. This is the
regime the FIS 'none of the above' rule was designed for, and the first place
in this project where the FIS has a task that suits it rather than a regression
it keeps losing.

Detectors, all trained on innocuous activations ONLY (true one-class):

  fis_anomaly   the tribble mechanism, reimplemented for one class: fit a
                Gaussian per feature on normal data, membership of a test value
                = exp(-0.5 z^2), aggregate across features with a product
                t-norm, anomaly = 1 - aggregate. This is exactly 'none of the
                learned patterns fit'.
  mahalanobis   distance in PCA-whitened activation space -- the standard
                one-class detector, and the one that beat the FIS in
                experiments/fuzzy-lm-anomaly.md.
  iforest       isolation forest.

The controls that make a positive result mean something:

  surface_*     the SAME detectors on token-only features (prompt length,
                unigram log-prob, fraction of tokens that are alphabetic). No
                activations. If the activation atlas does not beat this, the
                'fMRI' adds nothing over reading the input.
  per-layer     the detector restricted to one layer's activations, swept over
                layers, so we can say WHERE a malformation becomes visible --
                early (surface) vs late (semantic). This is the fMRI payoff.

Reported per malformation type, because the structural types (word_salad,
nonsense) are matched to normal token content and are the ones that test the
deep-representation claim; the surface types (gibberish) are expected to be
easy and are a sanity floor, not the result.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

STRUCTURAL = {"word_salad", "char_scramble", "truncated", "nonsense", "injection"}
SURFACE = {"char_gibberish", "token_gibberish", "repeated"}


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------


class FISAnomaly:
    """Per-feature Gaussian memberships aggregated by a product t-norm, then
    complemented. A faithful one-class reduction of the tribble anomaly rule."""

    def __init__(self, floor_sigma=1e-3):
        self.floor = floor_sigma

    def fit(self, X):
        self.mu = X.mean(0)
        self.sigma = X.std(0)
        self.sigma = np.maximum(self.sigma, self.floor * np.abs(self.mu).mean())
        return self

    def score(self, X):
        z = (X - self.mu) / self.sigma
        # log-membership per feature = -0.5 z^2; product t-norm = sum of logs;
        # normalise by feature count so dimensionality does not set the scale.
        log_mem = (-0.5 * z**2).mean(1)
        return -log_mem  # higher = more anomalous


class Mahalanobis:
    def __init__(self, n_pca=32):
        self.n_pca = n_pca

    def fit(self, X):
        self.n = min(self.n_pca, X.shape[1], X.shape[0] - 1)
        self.pca = PCA(n_components=self.n, whiten=True, random_state=0).fit(X)
        return self

    def score(self, X):
        Z = self.pca.transform(X)
        return (Z**2).sum(1)  # whitened -> squared Mahalanobis distance


class IForest:
    def fit(self, X):
        self.m = IsolationForest(n_estimators=300, random_state=0).fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


DETECTORS = {"fis_anomaly": FISAnomaly, "mahalanobis": Mahalanobis, "iforest": IForest}


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------


def layer_features(act, fit_idx, per_layer_pca=8):
    """Per-layer PCA coords (fit on normal) + per-layer norms.

    Returns X (n, n_layers*(pca+1)) and a list mapping columns -> layer.
    """
    n, Lp1, D = act.shape
    feats, layer_of = [], []
    for l in range(Lp1):
        A = act[:, l, :]
        p = PCA(n_components=min(per_layer_pca, D, len(fit_idx) - 1),
                random_state=0).fit(A[fit_idx])
        Z = p.transform(A)
        norm = np.linalg.norm(A, axis=1, keepdims=True)
        block = np.hstack([Z, norm])
        feats.append(block)
        layer_of += [l] * block.shape[1]
    return np.hstack(feats), np.array(layer_of)


def surface_features(df, fit_idx):
    """Token-only features -- no activations. The control."""
    from collections import Counter

    txt = df.text.tolist()
    toks = [t.lower().split() for t in txt]
    # unigram log-prob estimated on the normal training prompts
    c = Counter()
    for i in fit_idx:
        c.update(toks[i])
    tot = sum(c.values()) or 1
    logp = []
    for tk in toks:
        if not tk:
            logp.append(-20.0)
            continue
        logp.append(np.mean([np.log((c[w] + 1) / (tot + len(c))) for w in tk]))
    frac_alpha = [
        np.mean([any(ch.isalpha() for ch in w) for w in tk]) if tk else 0.0
        for tk in toks
    ]
    mean_wlen = [np.mean([len(w) for w in tk]) if tk else 0.0 for tk in toks]
    return np.column_stack(
        [df.tok_len.to_numpy(dtype=float), np.array(logp),
         np.array(frac_alpha), np.array(mean_wlen)]
    )


# --------------------------------------------------------------------------


def evaluate(score, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, score)
    return a


def run(rundir: Path, variant="mean", seed=0, per_layer_pca=8) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet")
    act = np.load(rundir / f"act_{variant}.npy")
    is_innoc = (df.label == "innocuous").to_numpy()

    rng = np.random.default_rng(seed)
    innoc_idx = np.where(is_innoc)[0]
    rng.shuffle(innoc_idx)
    cut = int(0.6 * len(innoc_idx))
    fit_idx = innoc_idx[:cut]           # train (normal only)
    test_innoc = innoc_idx[cut:]        # held-out normal

    X, layer_of = layer_features(act, fit_idx, per_layer_pca)
    Xs = surface_features(df, fit_idx)

    types = [t for t in df.label.unique() if t != "innocuous"]
    out: dict = {
        "variant": variant,
        "n_train_normal": len(fit_idx),
        "n_test_normal": len(test_innoc),
        "n_features": X.shape[1],
        "detectors": {},
        "surface_control": {},
        "per_layer_mahalanobis": {},
    }

    def eval_block(featmat, name, container):
        res = {}
        for dname, D in DETECTORS.items():
            det = D().fit(featmat[fit_idx])
            s_all = det.score(featmat)
            per_type, all_y, all_s = {}, [], []
            for t in types:
                ti = np.where(df.label.to_numpy() == t)[0]
                y = np.r_[np.zeros(len(test_innoc)), np.ones(len(ti))]
                s = np.r_[s_all[test_innoc], s_all[ti]]
                per_type[t] = evaluate(s, y)
                all_y.append(y[len(test_innoc):])
                all_s.append(s[len(test_innoc):])
            yv = np.r_[np.zeros(len(test_innoc)),
                       np.ones(sum(len(np.where(df.label.to_numpy() == t)[0]) for t in types))]
            sv = np.r_[s_all[test_innoc],
                       np.concatenate([s_all[np.where(df.label.to_numpy() == t)[0]] for t in types])]
            struct_types = [t for t in types if t in STRUCTURAL]
            surf_types = [t for t in types if t in SURFACE]
            res[dname] = {
                "per_type": {k: (None if np.isnan(v) else round(v, 3))
                             for k, v in per_type.items()},
                "auroc_all": round(evaluate(sv, yv), 3),
                "auroc_structural": round(
                    np.nanmean([per_type[t] for t in struct_types]), 3),
                "auroc_surface": round(
                    np.nanmean([per_type[t] for t in surf_types]), 3),
            }
        container.update(res)

    eval_block(X, "activation", out["detectors"])
    eval_block(Xs, "surface", out["surface_control"])

    # per-layer Mahalanobis: where does malformation become visible
    for l in sorted(set(layer_of)):
        cols = np.where(layer_of == l)[0]
        det = Mahalanobis().fit(X[np.ix_(fit_idx, cols)])
        s_all = det.score(X[:, cols])
        struct, surf = [], []
        for t in types:
            ti = np.where(df.label.to_numpy() == t)[0]
            y = np.r_[np.zeros(len(test_innoc)), np.ones(len(ti))]
            s = np.r_[s_all[test_innoc], s_all[ti]]
            a = evaluate(s, y)
            (struct if t in STRUCTURAL else surf).append(a)
        out["per_layer_mahalanobis"][int(l)] = {
            "structural": round(float(np.nanmean(struct)), 3),
            "surface": round(float(np.nanmean(surf)), 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/fmri")
    ap.add_argument("--variant", default="mean")
    ap.add_argument("--pca", type=int, default=8)
    a = ap.parse_args()
    rundir = Path(a.run)
    r = run(rundir, variant=a.variant, per_layer_pca=a.pca)
    (rundir / f"detect_{a.variant}.json").write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("per_layer_mahalanobis",)}, indent=2))
    print("\nper-layer Mahalanobis (structural | surface):")
    for l, v in r["per_layer_mahalanobis"].items():
        print(f"  layer {l:2d}: {v['structural']:.3f} | {v['surface']:.3f}")


if __name__ == "__main__":
    main()
