"""Comprehensive cross-model, cross-dataset summary.

For every captured run, reports the one-class detector with the log-domain
*trimmed* score (the Part 8 fix), and -- because injection corpora have opposite
length confounds -- both the deployment metric (det@1%FP, pooled) and the
confound-controlled signal (within-length AUROC), with the surface-only baseline
alongside so the activation-vs-surface question is answerable per (model,
dataset).
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tribblefis.one_class import TribbleOneClassDetector
from .fmri_detect import layer_features, Mahalanobis, surface_features
from .improve_lowfpr import det_at, surprisals
from .injection_detect_v2 import within_len_auc
from .summarize_all import label

# run dir -> dataset tag
DATASET = {"jailbreak": "jailbreak", "sg_": "safeguard", "spml": "spml"}


def dataset_of(name):
    for k, v in DATASET.items():
        if k in name:
            return v
    return "deepset"


def evaluate(rundir: Path, seeds=6, n_pca=32) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    mid = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]

    fis_d1, fis_wl, mah_wl, srf_wl = [], [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = np.where(y == 0)[0]; rng.shuffle(ben); inj = np.where(y == 1)[0]
        fit = ben[: int(0.6 * len(ben))]; tb = ben[int(0.6 * len(ben)):]
        ti = np.r_[tb, inj]; yt = np.r_[np.zeros(len(tb)), np.ones(len(inj))]
        tlt = np.r_[tl[tb], tl[inj]]
        X, _ = layer_features(act, fit, 8)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        with contextlib.redirect_stdout(io.StringIO()):
            det = TribbleOneClassDetector(
                whiten=True, whiten_components=min(n_pca, len(fit) - 1),
                n_gaussians=1, norm_conorm="probability", random_state=seed).fit(Xdf.iloc[fit])
        S = surprisals(det, Xdf)
        trim = np.sort(S, 1)[:, : S.shape[1] - 2].sum(1)
        fis_d1.append(det_at(yt, trim[ti], 0.01))
        fis_wl.append(within_len_auc(yt, trim[ti], tlt))
        mah = Mahalanobis(n_pca=min(n_pca, len(fit) - 1)).fit(X[fit]).score(X)
        mah_wl.append(within_len_auc(yt, mah[ti], tlt))
        Xs = surface_features(df, fit)
        srf = Mahalanobis(n_pca=4).fit(Xs[fit]).score(Xs)
        srf_wl.append(within_len_auc(yt, srf[ti], tlt))
    name, params = label(mid)
    return {"model": name, "params": params, "dataset": dataset_of(rundir.name),
            "fis_trim_det1": round(float(np.mean(fis_d1)), 3),
            "fis_trim_wlAUC": round(float(np.mean(fis_wl)), 3),
            "mahal_wlAUC": round(float(np.mean(mah_wl)), 3),
            "surface_wlAUC": round(float(np.mean(srf_wl)), 3)}


def main():
    runs = [Path("runs/injection")] + sorted(Path("runs").glob("injection_*")) \
        + [Path("runs/jailbreak")] + sorted(Path("runs").glob("sg_*"))
    runs = [r for r in runs if (r / "act_mean.npy").exists()]
    res = []
    for r in runs:
        try:
            res.append(evaluate(r))
        except Exception as e:
            print(f"skip {r.name}: {e}")
    res.sort(key=lambda d: (d["dataset"], d["params"]))
    print("\nComprehensive summary -- FIS one-class (trimmed log-domain score), "
          "within-length AUROC is confound-controlled:\n")
    print("%-20s %-10s %5s %9s %9s %9s %9s  %s"
          % ("model", "dataset", "prm", "det@1FP", "FIS-wlAUC", "Mahal-wl", "surf-wl", "act>surf?"))
    for d in res:
        win = "YES" if d["fis_trim_wlAUC"] > d["surface_wlAUC"] else "no"
        print("%-20s %-10s %5.2f %9.3f %9.3f %9.3f %9.3f  %s"
              % (d["model"], d["dataset"], d["params"], d["fis_trim_det1"],
                 d["fis_trim_wlAUC"], d["mahal_wlAUC"], d["surface_wlAUC"], win))
    Path("runs/comprehensive.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
