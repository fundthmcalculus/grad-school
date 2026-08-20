"""ROC sweep (FPR vs detection quality) and construction/inference timing.

Produces, for every captured instruct model:
  * a full ROC curve (FPR, TPR) for the genuine TribbleOneClassDetector and for
    Mahalanobis, one-class, within-length-aware evaluation on held-out data;
  * a timing row -- model config plus wall-clock construction (fit) and
    inference (per-prompt score) for the FIS detector.

Writes runs/sweep_roc.json (curves) and runs/timing.json (table). Deterministic
across a fixed seed set; timings are the median of repeats to damp noise.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from tribblefis.one_class import TribbleOneClassDetector
from .fmri_detect import layer_features, Mahalanobis
from .injection_detect_v2 import within_len_auc

RUNS = {
    "SmolLM2-135M": "runs/injection",
    "SmolLM2-360M": "runs/injection_360m",
    "gemma-3-270m-it": "runs/injection_gemma",
    "TinyLlama-1.1B": "runs/injection_tinyllama",
}


def mean_roc(curves, grid):
    """Interpolate each (fpr,tpr) onto a common FPR grid and average."""
    ts = np.stack([np.interp(grid, f, t) for f, t in curves])
    return ts.mean(0), ts.std(0)


def run(seeds=6, n_pca=32) -> tuple[dict, dict]:
    grid = np.linspace(0, 1, 201)
    roc_out, timing_out = {}, {}

    for name, rd in RUNS.items():
        rdp = Path(rd)
        if not (rdp / "act_mean.npy").exists():
            continue
        meta = json.loads((rdp / "meta.json").read_text())
        df = pd.read_parquet(rdp / "probes.parquet").reset_index(drop=True)
        act = np.load(rdp / "act_mean.npy")
        y = (df.label == "injection").to_numpy().astype(int)
        tl = df.tok_len.to_numpy()

        fis_curves, mah_curves = [], []
        fis_auc, mah_auc = [], []
        fit_ms, score_us = [], []

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

            X, _ = layer_features(act, fit, 8)
            Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

            # time construction (fit) -- median of a few repeats
            reps = []
            for _ in range(3):
                t0 = time.time()
                with contextlib.redirect_stdout(io.StringIO()):
                    det = TribbleOneClassDetector(
                        whiten=True,
                        whiten_components=min(n_pca, len(fit) - 1),
                        n_gaussians=1,
                        norm_conorm="probability",
                        random_state=seed,
                    ).fit(Xdf.iloc[fit])
                reps.append(time.time() - t0)
            fit_ms.append(1000 * np.median(reps))

            # time inference per prompt
            t0 = time.time()
            for _ in range(5):
                s_fis = det.anomaly_score(Xdf)
            score_us.append(1e6 * (time.time() - t0) / (5 * len(Xdf)))

            f, t, _ = roc_curve(yt, s_fis[ti])
            fis_curves.append((f, t))
            fis_auc.append(within_len_auc(yt, s_fis[ti], tlt))

            s_mah = Mahalanobis(n_pca=min(n_pca, len(fit) - 1)).fit(X[fit]).score(X)
            f, t, _ = roc_curve(yt, s_mah[ti])
            mah_curves.append((f, t))
            mah_auc.append(within_len_auc(yt, s_mah[ti], tlt))

        fm, fs = mean_roc(fis_curves, grid)
        mm, ms = mean_roc(mah_curves, grid)
        roc_out[name] = {
            "fpr_grid": grid.tolist(),
            "fis_tpr": fm.tolist(),
            "fis_tpr_std": fs.tolist(),
            "mah_tpr": mm.tolist(),
            "fis_auroc": float(np.mean(fis_auc)),
            "mah_auroc": float(np.mean(mah_auc)),
        }
        cfg = meta["config"]
        timing_out[name] = {
            "model_id": cfg["model_id"],
            "n_layers": meta["n_layers"],
            "hidden": meta["hidden"],
            "n_features": int(min(n_pca, int(0.6 * (y == 0).sum()) - 1)),
            "n_train_benign": int(0.6 * (y == 0).sum()),
            "construct_ms": round(float(np.median(fit_ms)), 1),
            "infer_us_per_prompt": round(float(np.median(score_us)), 1),
            "within_len_auroc": round(float(np.mean(fis_auc)), 3),
        }
        print(
            f"{name:18s} AUROC fis {np.mean(fis_auc):.3f} mah {np.mean(mah_auc):.3f}"
            f" | construct {np.median(fit_ms):.0f} ms"
            f" | infer {np.median(score_us):.1f} us/prompt",
            flush=True,
        )

    Path("runs/sweep_roc.json").write_text(json.dumps(roc_out))
    Path("runs/timing.json").write_text(json.dumps(timing_out, indent=2))
    return roc_out, timing_out


if __name__ == "__main__":
    run()
