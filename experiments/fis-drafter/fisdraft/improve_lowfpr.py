"""Three ideas to make 1%-FPR injection detection usable, each tested.

Diagnosis (Part 7): the TribbleOneClassDetector's `1 - max firing` score is
`1 - exp(-0.5 * sum_j z_j^2)` over the whitened components. With ~32 components
it *saturates* -- benign points already push the product toward 0, so many
prompts tie at anomaly ~ 1, flattening the low-FPR tail. Mahalanobis (`sum z^2`,
no exp) keeps det@1%FP because it never saturates. The three ideas attack three
different facets of this.

All ideas reuse the genuine library: the per-component Gaussian memberships come
from a fitted `TribbleOneClassDetector` (whitened). Only the way those
memberships are *aggregated / calibrated / ensembled* changes.

IDEA 1 -- non-saturating & robust aggregation.
  The `max`/product-then-complement is the specific culprit. Replace it with
  log-domain aggregations that preserve ordering, and a robust variant that a
  single odd component cannot dominate:
    surprisal_sum   sum_j 0.5 z_j^2         (== Mahalanobis; the non-saturating ref)
    trimmed_sum     drop the top-m z_j^2 per prompt before summing -- a benign
                    prompt with one weird component should not be flagged
    topk_mean       mean of the k largest surprisals (attack signal is diffuse)
  Hypothesis: trimmed_sum beats plain Mahalanobis at 1%-FPR by removing the
  benign false positives caused by a single high-z component.

IDEA 2 -- ensemble / bagging.
  The heavy benign tail is partly variance in the density estimate: a benign
  prompt near the manifold edge scores high by chance. Bag detectors over
  bootstrap benign subsamples x random component subsets and average the scores.
  Variance reduction tightens the benign tail (the isolation-forest insight).

IDEA 3 -- conditional / local threshold calibration.
  A global 1%-FPR threshold is dragged up by the worst benign subgroup (e.g. a
  length band). Normalise each prompt's score by the *local* benign score level
  -- its z-score against the k nearest benign prompts in feature space -- so the
  FPR is equalised across regions and the global threshold is not hostage to one
  dense-but-high-scoring pocket of benign traffic.

Everything is scored one-class (benign-only fit) and reported as det@1%FP with
det@5%FP and AUROC alongside, against the unmodified detector as baseline.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from tribblefis.one_class import TribbleOneClassDetector
from tribblefis.refine import extract_gaussian_params
from .fmri_detect import layer_features


def det_at(y, s, cap):
    fpr, tpr, _ = roc_curve(y, s)
    h = tpr[fpr <= cap]
    return float(h[-1]) if len(h) else 0.0


def surprisals(det, Xdf):
    """Per-component surprisal 0.5 z^2 for every prompt, from the fitted model."""
    Z = det._transform(Xdf).to_numpy()
    params = extract_gaussian_params(det.model_)  # [mu0,sig0,mu1,sig1,...]
    mu = params[0::2]
    sig = np.maximum(params[1::2], 1e-6)
    z = (Z - mu) / sig
    return 0.5 * z**2  # (n, n_components)


# --------------------------------------------------------------------------
# Idea 1: aggregations
# --------------------------------------------------------------------------


def agg_scores(S):
    n_comp = S.shape[1]
    out = {
        "baseline_1-maxfiring": 1.0 - np.exp(-S.sum(1)),  # == library score
        "surprisal_sum(=Mahal)": S.sum(1),
        "trimmed_sum_top2": np.sort(S, 1)[:, : n_comp - 2].sum(1),
        "trimmed_sum_top4": np.sort(S, 1)[:, : n_comp - 4].sum(1),
        "topk_mean_16": np.sort(S, 1)[:, -16:].mean(1),
    }
    return out


# --------------------------------------------------------------------------
# Idea 2: bagging
# --------------------------------------------------------------------------


def bagged_score(act, fit, Xdf, n_pca, n_bags, seed):
    rng = np.random.default_rng(seed)
    scores = []
    for b in range(n_bags):
        sub = rng.choice(fit, size=len(fit), replace=True)  # bootstrap benign
        with contextlib.redirect_stdout(io.StringIO()):
            det = TribbleOneClassDetector(
                whiten=True,
                whiten_components=min(n_pca, len(np.unique(sub)) - 1),
                n_gaussians=1,
                norm_conorm="probability",
                random_state=seed + b,
            ).fit(Xdf.iloc[sub])
        S = surprisals(det, Xdf)
        # random component subset per bag
        keep = rng.choice(S.shape[1], size=max(8, S.shape[1] // 2), replace=False)
        scores.append(S[:, keep].sum(1))
    # rank-average across bags (scale-free)
    ranks = np.mean([pd.Series(s).rank().to_numpy() for s in scores], 0)
    return ranks


# --------------------------------------------------------------------------
# Idea 3: local calibration
# --------------------------------------------------------------------------


def local_calibrated(det, Xdf, fit_idx, base_score, k=20):
    """z-score each prompt's base score against its k nearest benign-train
    neighbours in whitened feature space."""
    Z = det._transform(Xdf).to_numpy()
    Zf = Z[fit_idx]
    nn = NearestNeighbors(n_neighbors=min(k, len(fit_idx))).fit(Zf)
    _, idx = nn.kneighbors(Z)
    # local benign score stats from the training scores
    tr_scores = base_score[fit_idx]
    loc_mu = tr_scores[idx].mean(1)
    loc_sd = tr_scores[idx].std(1) + 1e-9
    return (base_score - loc_mu) / loc_sd


# --------------------------------------------------------------------------


def run(rundir: Path, seeds=6, n_pca=32) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    model_id = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]

    arms: dict[str, list] = {}

    def rec(name, d1, d5, au):
        arms.setdefault(name, {"d1": [], "d5": [], "auc": []})
        arms[name]["d1"].append(d1)
        arms[name]["d5"].append(d5)
        arms[name]["auc"].append(au)

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = np.where(y == 0)[0]
        rng.shuffle(ben)
        inj = np.where(y == 1)[0]
        fit = ben[: int(0.6 * len(ben))]
        tb = ben[int(0.6 * len(ben)) :]
        ti = np.r_[tb, inj]
        yt = np.r_[np.zeros(len(tb)), np.ones(len(inj))]

        X, _ = layer_features(act, fit, 8)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        with contextlib.redirect_stdout(io.StringIO()):
            det = TribbleOneClassDetector(
                whiten=True,
                whiten_components=min(n_pca, len(fit) - 1),
                n_gaussians=1,
                norm_conorm="probability",
                random_state=seed,
            ).fit(Xdf.iloc[fit])
        S = surprisals(det, Xdf)

        # Idea 1
        for name, sc in agg_scores(S).items():
            rec(
                f"1:{name}",
                det_at(yt, sc[ti], 0.01),
                det_at(yt, sc[ti], 0.05),
                roc_auc_score(yt, sc[ti]),
            )

        # Idea 2
        bag = bagged_score(act, fit, Xdf, n_pca, n_bags=15, seed=seed)
        rec(
            "2:bagged_15",
            det_at(yt, bag[ti], 0.01),
            det_at(yt, bag[ti], 0.05),
            roc_auc_score(yt, bag[ti]),
        )

        # Idea 3 (on the best non-saturating base: surprisal_sum)
        base = S.sum(1)
        loc = local_calibrated(det, Xdf, fit, base, k=20)
        rec(
            "3:local_calib(Mahal)",
            det_at(yt, loc[ti], 0.01),
            det_at(yt, loc[ti], 0.05),
            roc_auc_score(yt, loc[ti]),
        )
        # Idea 1+3 combined: local-calibrated trimmed sum
        base_t = np.sort(S, 1)[:, : S.shape[1] - 4].sum(1)
        loc_t = local_calibrated(det, Xdf, fit, base_t, k=20)
        rec(
            "1+3:local_trimmed",
            det_at(yt, loc_t[ti], 0.01),
            det_at(yt, loc_t[ti], 0.05),
            roc_auc_score(yt, loc_t[ti]),
        )

    return {
        "model": model_id,
        "arms": {
            k: {
                "det@1%FP": round(float(np.mean(v["d1"])), 3),
                "det@5%FP": round(float(np.mean(v["d5"])), 3),
                "auroc": round(float(np.mean(v["auc"])), 3),
            }
            for k, v in arms.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/injection"])
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    order = [
        "1:baseline_1-maxfiring",
        "1:surprisal_sum(=Mahal)",
        "1:trimmed_sum_top2",
        "1:trimmed_sum_top4",
        "1:topk_mean_16",
        "2:bagged_15",
        "3:local_calib(Mahal)",
        "1+3:local_trimmed",
    ]
    allout = []
    print("Three ideas for better 1%-FPR (det@1%FP | det@5%FP | AUROC), one-class:\n")
    for r in a.runs:
        res = run(Path(r), seeds=a.seeds)
        allout.append(res)
        print(f"== {res['model'].split('/')[-1]} ==")
        for k in order:
            if k in res["arms"]:
                v = res["arms"][k]
                mark = (
                    " *"
                    if v["det@1%FP"]
                    > res["arms"]["1:baseline_1-maxfiring"]["det@1%FP"] + 0.03
                    else ""
                )
                print(
                    "  %-26s %.3f | %.3f | %.3f%s"
                    % (k, v["det@1%FP"], v["det@5%FP"], v["auroc"], mark)
                )
        print()
    Path("runs/improve_lowfpr.json").write_text(json.dumps(allout, indent=2))


if __name__ == "__main__":
    main()
