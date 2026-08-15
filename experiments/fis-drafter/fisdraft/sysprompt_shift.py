"""System-prompt-shift robustness: does the 'normal' manifold move?

A deployed monitor is fit on benign activations under one system prompt, but
production may wrap prompts in a different system prompt. The chat wrapper shifts
every activation, so the learned normal manifold may no longer describe live
benign traffic — the FPR could blow up, and detection degrade.

Setup: the same deepset battery captured under four system prompts (default /
helpful / support / terse), Qwen-3B. For every (train prompt A, test prompt B):

  * fit the one-class detector on benign under A (whitening + density);
  * set the 1%-FPR threshold on benign-A held-out;
  * apply to benign-B (measure realized FPR) and injection-B (detection).

Two failure modes to separate:
  realized_FPR(A->B)  benign-B flagged as anomalous because the manifold moved
                      (the operational risk).
  det@1%FP(A->B)      oracle detection on B using the A-fit score (ranking
                      transfer, threshold-free).

Reported as A×B matrices. The diagonal (A==B) is the matched baseline.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from tribblefis.one_class import TribbleOneClassDetector
from .fmri_detect import layer_features

PROMPTS = ["default", "helpful", "support", "terse"]


def det_at(y, s, cap=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    h = tpr[fpr <= cap]
    return float(h[-1]) if len(h) else 0.0


def load(tag):
    d = Path(f"runs/sp_{tag}")
    df = pd.read_parquet(d / "probes.parquet").reset_index(drop=True)
    import os

    act = np.load(d / os.environ.get("SP_VARIANT", "act_mean.npy"))
    y = (df.label == "injection").to_numpy().astype(int)
    return df, act, y


def run(seeds=6, n_pca=32, score="trimmed") -> dict:
    data = {t: load(t) for t in PROMPTS}
    fpr_mat = np.zeros((len(PROMPTS), len(PROMPTS)))
    det_mat = np.zeros((len(PROMPTS), len(PROMPTS)))

    for ai, A in enumerate(PROMPTS):
        dfA, actA, yA = data[A]
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            benA = np.where(yA == 0)[0]
            rng.shuffle(benA)
            fit = benA[: int(0.5 * len(benA))]
            calib = benA[int(0.5 * len(benA)) : int(0.7 * len(benA))]
            # fit detector + whitening on A-benign, get its score fn via a fitted det
            XA, _ = layer_features(actA, fit, 8)
            XAdf = pd.DataFrame(XA, columns=[f"f{i}" for i in range(XA.shape[1])])
            with contextlib.redirect_stdout(io.StringIO()):
                det = TribbleOneClassDetector(
                    whiten=True,
                    whiten_components=min(n_pca, len(fit) - 1),
                    cov="pca",
                    n_gaussians=1,
                    score=score,
                    random_state=seed,
                ).fit(XAdf.iloc[fit])
            # 1%-FPR threshold from A-benign calibration
            sA_cal = det.anomaly_score(XAdf.iloc[calib])
            tau = float(np.quantile(sA_cal, 0.99))
            for bi, B in enumerate(PROMPTS):
                dfB, actB, yB = data[B]
                # rebuild B features in the SAME per-layer PCA basis as A? No --
                # the detector's whitening is fit on A; we must feed B through the
                # SAME layer_features PCA. layer_features refits per call, so to
                # keep the A basis we transform B's raw layer features with A's
                # detector transform. Simplest faithful path: recompute B features
                # with the A fit indices' PCA by using the same fit rows is not
                # possible across corpora; instead use A's PCA basis via the model.
                # Practical equivalent: features are per-layer PCA(8) fit on A-fit;
                # apply to B by fitting layer_features on A-fit but transforming B.
                XB = layer_features_transform(actA, fit, actB)
                XBdf = pd.DataFrame(XB, columns=[f"f{i}" for i in range(XB.shape[1])])
                sB = det.anomaly_score(XBdf)
                benB = np.where(yB == 0)[0]
                injB = np.where(yB == 1)[0]
                # held-out benign-B (exclude the rows used as A-fit indices? A and
                # B are the same prompts under different wrappers, so exclude fit)
                test_benB = np.array([i for i in benB if i not in set(fit)])
                fpr_mat[ai, bi] += float((sB[test_benB] > tau).mean())
                yt = np.r_[np.zeros(len(test_benB)), np.ones(len(injB))]
                st = np.r_[sB[test_benB], sB[injB]]
                det_mat[ai, bi] += det_at(yt, st)
    fpr_mat /= seeds
    det_mat /= seeds
    return {
        "prompts": PROMPTS,
        "score": score,
        "realized_fpr": fpr_mat.round(4).tolist(),
        "det_at_1fp": det_mat.round(3).tolist(),
    }


def layer_features_transform(act_fit_src, fit_idx, act_target):
    """Per-layer PCA(8) fit on `act_fit_src[fit_idx]`, applied to `act_target`.

    A and B are the SAME prompts under different system prompts, so their arrays
    are row-aligned; the feature basis is learned from A's fit rows and used to
    transform B's rows, mirroring a deployed detector fit under A scoring traffic
    wrapped under B.
    """
    from sklearn.decomposition import PCA

    n, Lp1, D = act_fit_src.shape
    feats = []
    for l in range(Lp1):
        # match layer_features exactly: per-layer PCA(8) coords fit on A + norm
        p = PCA(n_components=min(8, D, len(fit_idx) - 1), random_state=0).fit(
            act_fit_src[fit_idx, l, :]
        )
        Z = p.transform(act_target[:, l, :])
        norm = np.linalg.norm(act_target[:, l, :], axis=1, keepdims=True)
        feats.append(np.hstack([Z, norm]))
    return np.hstack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="trimmed")
    a = ap.parse_args()
    r = run(score=a.score)
    Path("runs/sysprompt_shift.json").write_text(json.dumps(r, indent=2))
    P = r["prompts"]
    print(f"System-prompt shift, score={a.score}, Qwen-3B/deepset, 6 seeds.\n")
    print(
        "REALIZED FPR (threshold set at 1% on train-A benign; rows=train A, cols=test B):"
    )
    print("        " + "".join("%9s" % b for b in P))
    for i, A in enumerate(P):
        print(
            "%-8s" % A
            + "".join("%9.3f" % r["realized_fpr"][i][j] for j in range(len(P)))
        )
    print("\nDET@1%FP (oracle, rows=train A, cols=test B):")
    print("        " + "".join("%9s" % b for b in P))
    for i, A in enumerate(P):
        print(
            "%-8s" % A + "".join("%9.3f" % r["det_at_1fp"][i][j] for j in range(len(P)))
        )


if __name__ == "__main__":
    main()
