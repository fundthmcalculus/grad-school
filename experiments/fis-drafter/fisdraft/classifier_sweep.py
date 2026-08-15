"""Parameter sweep on the real TribbleClassifier for injection detection.

Supervised (benign vs injection), on captured activation features, to see how the
library's own knobs move detection. One-factor-at-a-time from a fixed baseline so
each parameter's marginal effect is legible, plus the best grid cell. 5-fold CV,
scored by within-length AUROC and det@1%FP so it is comparable to the one-class
results. Features are the per-layer-PCA activation atlas the detectors use.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from tribblefis.gaussian_classifier import TribbleClassifier
from .fmri_detect import layer_features
from .injection_detect_v2 import within_len_auc

BASE = dict(
    top_n=16,
    n_gaussians=2,
    norm_conorm="probability",
    member_function="gaussian",
    refine=False,
)

SWEEP = {
    "top_n": [4, 8, 16, 24, 32],
    "n_gaussians": [1, 2, 3, 4],
    "norm_conorm": ["probability", "einstein", "hamacher", "min/max"],
    "member_function": ["gaussian", "trap"],
    "refine": [False, True],
}


def det_at(y, s, cap=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    h = tpr[fpr <= cap]
    return float(h[-1]) if len(h) else 0.0


def cv_score(X, y, tl, params, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = np.zeros(len(X))
    fit_s = 0.0
    for tr, te in skf.split(X, y):
        Xtr = pd.DataFrame(X[tr], columns=[f"f{i}" for i in range(X.shape[1])])
        Xte = pd.DataFrame(X[te], columns=[f"f{i}" for i in range(X.shape[1])])
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            clf = TribbleClassifier(random_state=seed, **params).fit(
                Xtr, y[tr].astype(int)
            )
            p = clf.predict_proba(Xte)
        fit_s += time.time() - t0
        ic = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        pred[te] = p[:, ic]
    return (within_len_auc(y, pred, tl), det_at(y, pred), fit_s / 5)


def run(rundir: Path, seed=0) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    # features: per-layer PCA atlas, fit PCA on all (supervised CV handles leakage
    # only at the classifier; the shared PCA basis is unsupervised so it is fine)
    X, _ = layer_features(act, np.arange(len(y)), 8)
    mid = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]

    out = {"model": mid, "baseline": BASE, "one_factor": {}, "grid_best": None}

    a0, d0, t0 = cv_score(X, y, tl, BASE, seed)
    out["baseline_score"] = {
        "wl_auroc": round(a0, 3),
        "det@1%FP": round(d0, 3),
        "fit_s": round(t0, 3),
    }
    print(f"baseline {BASE}: wlAUC {a0:.3f} det@1%FP {d0:.3f} ({t0*1000:.0f} ms/fit)\n")

    for param, vals in SWEEP.items():
        rows = []
        for v in vals:
            p = dict(BASE)
            p[param] = v
            try:
                a, d, ft = cv_score(X, y, tl, p, seed)
                rows.append(
                    {
                        "value": v,
                        "wl_auroc": round(a, 3),
                        "det@1%FP": round(d, 3),
                        "fit_ms": round(ft * 1000),
                    }
                )
                print(
                    f"  {param}={v!s:14s} wlAUC {a:.3f}  det@1%FP {d:.3f}  {ft*1000:.0f} ms"
                )
            except Exception as e:
                rows.append({"value": v, "error": str(e)[:60]})
                print(f"  {param}={v!s:14s} ERROR {str(e)[:50]}")
        out["one_factor"][param] = rows
        print()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/injection_qwen3b")
    a = ap.parse_args()
    r = run(Path(a.run))
    Path(a.run + "_clf_sweep.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
