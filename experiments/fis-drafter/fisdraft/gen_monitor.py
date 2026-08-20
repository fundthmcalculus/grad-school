"""Does generation add signal? Prompt vs streaming-activation vs logit-shape.

Four monitors on the SAME probes, all one-class (benign-only fit), reported at
det@1%FP and within-length AUROC (length-confound-controlled), 6 seeds:

  prompt_readout  the existing prompt monitor -- per-layer activation at the final
                  prompt token (step-0 readout, == act_last). The baseline the
                  generation axis must beat to justify its cost.
  gen_mean        streaming hidden states, mean-pooled over generated tokens.
  gen_last        streaming hidden states at the final generated token.
  logit_shape     ONLY the output-distribution shape trajectory (entropy, top-1
                  mass, top-5 mass, top1-top2 margin, perplexity; per-step mean/std,
                  first-step value, entropy slope). No hidden states at all -- the
                  original 'the logit shape will tell us something' theory, isolated.
  combined        prompt_readout activation features + logit_shape features.

The scientific question is not 'does each work' but 'does generation add anything
the cheap prompt pass does not already have', and 'does logit shape carry
independent signal'. combined vs prompt_readout answers the first; logit_shape's
standalone number and its lift in combined answer the second.

few_shot=logistic variants use 10 labelled attacks for the strict gate, matching
the deployable recipe from the sensitivity study.
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
from .injection_detect_v2 import within_len_auc

SHAPE_STEP = ["entropy", "max_prob", "top5_mass", "margin", "ppl"]


def d1(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    h = tpr[fpr <= 0.01]
    return float(h[-1]) if len(h) else 0.0


def shape_features(shape_steps: np.ndarray) -> np.ndarray:
    """(n, K, 5) per-step scalars -> (n, F) per-probe shape features, nan-aware.

    Per scalar: nanmean, nanstd, first-step value. Plus a per-probe entropy slope
    (least-squares over valid steps). Trajectory shape, not just level.
    """
    n, K, C = shape_steps.shape
    feats = []
    for c in range(C):
        col = shape_steps[:, :, c]
        feats.append(np.nan_to_num(np.nanmean(col, axis=1)))
        feats.append(np.nan_to_num(np.nanstd(col, axis=1)))
        feats.append(np.nan_to_num(col[:, 0]))
    # entropy slope over valid steps
    ent = shape_steps[:, :, 0]
    slope = np.zeros(n)
    t = np.arange(K, dtype=float)
    for i in range(n):
        v = ~np.isnan(ent[i])
        if v.sum() >= 3:
            slope[i] = np.polyfit(t[v], ent[i, v], 1)[0]
    feats.append(slope)
    return np.column_stack(feats)


def _oneclass(Xfit, Xall, y, fit_idx, seed, n_pca, few_shot=False, y_fit=None):
    Xdf = pd.DataFrame(Xall, columns=[f"f{i}" for i in range(Xall.shape[1])])
    kw = dict(
        whiten=True,
        whiten_components=min(n_pca, len(fit_idx) - 1),
        cov="ledoit_wolf",
        n_gaussians=1,
        score="trimmed",
        random_state=seed,
    )
    if few_shot:
        kw.update(few_shot="logistic")
    with contextlib.redirect_stdout(io.StringIO()):
        if few_shot:
            det = TribbleOneClassDetector(**kw).fit(Xdf.iloc[fit_idx], y_fit)
        else:
            det = TribbleOneClassDetector(**kw).fit(Xdf.iloc[fit_idx])
    return det.anomaly_score(Xdf)


def run(rundir: Path, seeds=6, n_pca=32) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    mid = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]

    readout = np.load(rundir / "readout.npy")
    gen_mean = np.load(rundir / "gen_mean.npy")
    gen_last = np.load(rundir / "gen_last.npy")
    shp = shape_features(np.load(rundir / "shape_steps.npy"))

    arms = [
        "prompt_readout",
        "gen_mean",
        "gen_last",
        "logit_shape",
        "combined",
        "prompt_readout+fs",
        "logit_shape+fs",
        "combined+fs",
    ]
    acc = {a: {"d1": [], "wl": []} for a in arms}

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = np.where(y == 0)[0]
        rng.shuffle(ben)
        inj = np.where(y == 1)[0]
        fit = ben[: int(0.6 * len(ben))]
        tb = ben[int(0.6 * len(ben)) :]
        ti = np.r_[tb, inj]
        yt = np.r_[np.zeros(len(tb)), np.ones(len(inj))]
        tlt = np.r_[tl[tb], tl[inj]]

        # few-shot: fit rows = benign-fit + 10 labelled attacks
        rng.shuffle(inj)
        fs_att = inj[:10]
        fs_fit = np.r_[fit, fs_att]
        y_fs = np.r_[np.zeros(len(fit)), np.ones(len(fs_att))].astype(int)

        # activation feature blocks (per-layer PCA fit on benign-fit)
        Xr, _ = layer_features(readout, fit, 8)
        Xgm, _ = layer_features(gen_mean, fit, 8)
        Xgl, _ = layer_features(gen_last, fit, 8)
        Xcomb = np.hstack([Xr, shp])

        # few-shot feature blocks refit PCA on fs_fit's benign part (== fit here)
        Xr_fs, _ = layer_features(readout, fit, 8)
        Xcomb_fs = np.hstack([Xr_fs, shp])

        def score(X, fs=False, yf=None, fitidx=fit):
            return _oneclass(None, X, y, fitidx, seed, n_pca, few_shot=fs, y_fit=yf)

        S = {
            "prompt_readout": score(Xr),
            "gen_mean": score(Xgm),
            "gen_last": score(Xgl),
            "logit_shape": score(shp),
            "combined": score(Xcomb),
            "prompt_readout+fs": score(Xr_fs, fs=True, yf=y_fs, fitidx=fs_fit),
            "logit_shape+fs": score(shp, fs=True, yf=y_fs, fitidx=fs_fit),
            "combined+fs": score(Xcomb_fs, fs=True, yf=y_fs, fitidx=fs_fit),
        }
        for a in arms:
            acc[a]["d1"].append(d1(yt, S[a][ti]))
            acc[a]["wl"].append(within_len_auc(yt, S[a][ti], tlt))

    return {
        "model": mid,
        "dataset": rundir.name,
        "arms": {
            a: {
                "det@1%FP": round(float(np.mean(v["d1"])), 3),
                "wl_auroc": round(float(np.mean(v["wl"])), 3),
            }
            for a, v in acc.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/gen_qwen3b")
    a = ap.parse_args()
    r = run(Path(a.run))
    Path(a.run + "/gen_monitor.json").write_text(json.dumps(r, indent=2))
    print(
        f"Generation vs prompt vs logit-shape monitors -- {r['model']} / {r['dataset']}"
    )
    print(f"{'arm':<22}{'det@1%FP':>10}{'wl-AUROC':>10}")
    for a_, v in r["arms"].items():
        print(f"{a_:<22}{v['det@1%FP']:>10.3f}{v['wl_auroc']:>10.3f}")


if __name__ == "__main__":
    main()
