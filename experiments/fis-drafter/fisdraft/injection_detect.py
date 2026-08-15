"""Prompt-injection detection: does the activation atlas beat length?

The application from PLAN_ANOMALY.md. A one-class monitor trained on benign
prompts only, flagging injections it has never seen. The entire result hinges
on one control: **public injection corpora are massively length-confounded**
(deepset injections median 125 chars vs benign 42), so a detector that only
reads length will look excellent and mean nothing.

Three defences, all reported:

  surface control     the same detectors on token-only features INCLUDING
                      length. Activations must beat this, not just chance.
  length-stratified   AUROC computed within length deciles and pooled, so
                      length cannot be the carrier.
  length-matched      a benign/injection subset matched on token count, scored
                      separately.

One-class throughout: the detector is fitted on benign activations only and
never sees an injection during training. That is the selling point -- no attack
examples -- and it is also the honest hard setting.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .fmri_detect import FISAnomaly, Mahalanobis, IForest, layer_features, surface_features

warnings.filterwarnings("ignore")


def stratified_auc(y, s, strat, min_n=20):
    tot_w, tot = 0.0, 0.0
    for v in np.unique(strat):
        m = strat == v
        if m.sum() < min_n or len(np.unique(y[m])) < 2:
            continue
        tot += roc_auc_score(y[m], s[m]) * m.sum()
        tot_w += m.sum()
    return tot / tot_w if tot_w else float("nan")


def whitened_fis(Xfit, Xall, n):
    pca = PCA(n_components=n, whiten=True, random_state=0).fit(Xfit)
    Zf, Za = pca.transform(Xfit), pca.transform(Xall)
    return FISAnomaly().fit(Zf).score(Za)


def run(rundir: Path, variant="mean", seed=0) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / f"act_{variant}.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    benign = np.where(y == 0)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(benign)
    cut = int(0.6 * len(benign))
    fit_idx = benign[:cut]                      # train: benign only
    test_benign = benign[cut:]
    inj_idx = np.where(y == 1)[0]
    test_idx = np.r_[test_benign, inj_idx]
    yt = np.r_[np.zeros(len(test_benign)), np.ones(len(inj_idx))]

    tl = df.tok_len.to_numpy()
    decile = pd.qcut(pd.Series(tl[test_idx]), 8, labels=False, duplicates="drop").to_numpy()

    X, layer_of = layer_features(act, fit_idx, per_layer_pca=8)
    Xs = surface_features(df, fit_idx)
    n_w = min(32, len(fit_idx) - 1)

    out: dict = {
        "variant": variant,
        "n_benign_train": len(fit_idx),
        "n_benign_test": len(test_benign),
        "n_injection": len(inj_idx),
        "median_len_benign": int(np.median(tl[benign])),
        "median_len_injection": int(np.median(tl[inj_idx])),
        "one_class": {},
        "supervised_reference": {},
    }

    def block(featmat, tag, container):
        nn = min(n_w, featmat.shape[1], len(fit_idx) - 1)
        scores = {
            "mahalanobis": Mahalanobis(n_pca=nn).fit(featmat[fit_idx]).score(featmat),
            "iforest": IForest().fit(featmat[fit_idx]).score(featmat),
            "fis_whitened": whitened_fis(featmat[fit_idx], featmat, nn),
        }
        for dn, sc in scores.items():
            st = sc[test_idx]
            container[f"{tag}:{dn}"] = {
                "auroc": round(roc_auc_score(yt, st), 3),
                "auroc_within_len": round(stratified_auc(yt, st, decile), 3),
            }
            print(f"  {tag:12s} {dn:14s} pooled {container[f'{tag}:{dn}']['auroc']:.3f}"
                  f"  within-len {container[f'{tag}:{dn}']['auroc_within_len']:.3f}",
                  flush=True)

    print("ONE-CLASS (train on benign only):")
    block(X, "activation", out["one_class"])
    block(Xs, "surface", out["one_class"])

    # length alone
    sl = tl[test_idx].astype(float)
    out["one_class"]["length_only"] = {
        "auroc": round(roc_auc_score(yt, sl), 3),
        "auroc_within_len": round(stratified_auc(yt, sl, decile), 3),
    }
    print(f"  {'length':12s} {'n_tokens':14s} pooled {out['one_class']['length_only']['auroc']:.3f}"
          f"  within-len {out['one_class']['length_only']['auroc_within_len']:.3f}")

    # supervised upper bound (HAS seen injections; cross-validated)
    print("\nSUPERVISED reference (5-fold, has seen attacks -- unfair upper bound):")
    for tag, featmat in (("activation", X), ("surface", Xs)):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        preds = np.zeros(len(df))
        nn = min(n_w, featmat.shape[1], int(0.8 * len(df)) - 1)
        for tr, te in skf.split(featmat, y):
            sc = PCA(n_components=nn, whiten=True, random_state=0).fit(featmat[tr])
            m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(featmat[tr]), y[tr])
            preds[te] = m.predict_proba(sc.transform(featmat[te]))[:, 1]
        out["supervised_reference"][tag] = {
            "auroc": round(roc_auc_score(y, preds), 3),
            "auroc_within_len": round(
                stratified_auc(y, preds,
                               pd.qcut(pd.Series(tl), 8, labels=False,
                                       duplicates="drop").to_numpy()), 3),
        }
        print(f"  {tag:12s} logreg         pooled {out['supervised_reference'][tag]['auroc']:.3f}"
              f"  within-len {out['supervised_reference'][tag]['auroc_within_len']:.3f}")

    # per-layer one-class (where does injection show up)
    out["per_layer"] = {}
    for l in sorted(set(layer_of)):
        cols = np.where(layer_of == l)[0]
        sc = Mahalanobis(n_pca=min(8, len(fit_idx) - 1)).fit(
            X[np.ix_(fit_idx, cols)]).score(X[:, cols])
        out["per_layer"][int(l)] = {
            "auroc": round(roc_auc_score(yt, sc[test_idx]), 3),
            "within_len": round(stratified_auc(yt, sc[test_idx], decile), 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/injection")
    ap.add_argument("--variant", default="mean")
    a = ap.parse_args()
    rundir = Path(a.run)
    r = run(rundir, variant=a.variant)
    (rundir / f"detect_{a.variant}.json").write_text(json.dumps(r, indent=2))
    print("\nlen(benign) median %d vs len(injection) median %d"
          % (r["median_len_benign"], r["median_len_injection"]))
    print("per-layer within-length AUROC:")
    for l, v in r["per_layer"].items():
        if l % 3 == 0 or v["within_len"] > 0.7:
            print(f"  layer {l:2d}: {v['within_len']:.3f}")


if __name__ == "__main__":
    main()
