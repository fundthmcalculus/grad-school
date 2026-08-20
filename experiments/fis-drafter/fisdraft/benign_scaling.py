"""Does more benign (baseline) training data improve one-class detection?

Sweeps the number of benign prompts used to fit the one-class detector, holding
the test set fixed, on a large benign pool (safeguard, 2000 benign captured).
Reports det@1%FP and within-length AUROC (trimmed log-domain score) vs training
size, so the data-scaling curve is visible.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tribblefis.one_class import TribbleOneClassDetector
from .fmri_detect import layer_features
from .improve_lowfpr import det_at, surprisals
from .injection_detect_v2 import within_len_auc

SIZES = [50, 100, 200, 400, 800, 1600]


def run(rundir: Path, seeds=6, n_pca=32) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    mid = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]
    ben_all = np.where(y == 0)[0]
    inj = np.where(y == 1)[0]

    curve = {n: {"d1": [], "wl": []} for n in SIZES}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = ben_all.copy()
        rng.shuffle(ben)
        # fixed held-out benign test (last 400), train pool is the rest
        test_ben = ben[-400:]
        pool = ben[:-400]
        ti = np.r_[test_ben, inj]
        yt = np.r_[np.zeros(len(test_ben)), np.ones(len(inj))]
        tlt = np.r_[tl[test_ben], tl[inj]]
        for n in SIZES:
            if n > len(pool):
                continue
            fit = pool[:n]
            X, _ = layer_features(act, fit, 8)
            Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            with contextlib.redirect_stdout(io.StringIO()):
                det = TribbleOneClassDetector(
                    whiten=True,
                    whiten_components=min(n_pca, n - 1),
                    n_gaussians=1,
                    norm_conorm="probability",
                    random_state=seed,
                ).fit(Xdf.iloc[fit])
            S = surprisals(det, Xdf)
            trim = np.sort(S, 1)[:, : S.shape[1] - 2].sum(1)
            curve[n]["d1"].append(det_at(yt, trim[ti], 0.01))
            curve[n]["wl"].append(within_len_auc(yt, trim[ti], tlt))

    out = {
        "model": mid,
        "n_benign_pool": len(ben_all),
        "n_test_benign": 400,
        "n_injection": len(inj),
        "curve": {},
    }
    print(f"{mid}: benign pool {len(ben_all)}, test 400 benign + {len(inj)} inj\n")
    print("%8s %10s %12s" % ("n_benign", "det@1%FP", "wl-AUROC"))
    for n in SIZES:
        if curve[n]["d1"]:
            d1 = float(np.mean(curve[n]["d1"]))
            wl = float(np.mean(curve[n]["wl"]))
            sd = float(np.std(curve[n]["d1"]))
            out["curve"][n] = {
                "det@1%FP": round(d1, 3),
                "det@1%FP_sd": round(sd, 3),
                "wl_auroc": round(wl, 3),
            }
            print("%8d %10.3f %12.3f  (±%.3f)" % (n, d1, wl, sd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/sg_qwen3b_big")
    a = ap.parse_args()
    r = run(Path(a.run))
    Path(a.run + "_benign_scaling.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
