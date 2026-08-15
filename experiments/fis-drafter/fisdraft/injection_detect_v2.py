"""Parts 3-5, re-run through the genuine tribblefis library.

The original injection experiments (`injection_detect.py`, `refine_anomaly.py`)
used `FISAnomaly`, a hand-written reimplementation. `CORRECTION.md` documents why
that was not a faithful stand-in. This version uses the real
`tribblefis.one_class.TribbleOneClassDetector` (whiten=True) as the FIS arm, so
the "one-class, no attack examples" claim holds *with the actual library*.

Everything else is identical to Part 3: one-class fit on benign only, the length
confound controlled three ways (surface-only baseline, within-length
stratification, operating points), and Mahalanobis + length reported alongside.
The FIS arm is now the library, not a reimplementation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .fmri_detect import Mahalanobis, layer_features, surface_features
from tribblefis.one_class import TribbleOneClassDetector

warnings.filterwarnings("ignore")


def within_len_auc(y, s, tl, nbin=8):
    dec = pd.qcut(pd.Series(tl), nbin, labels=False, duplicates="drop").to_numpy()
    tot_w = tot = 0.0
    for v in np.unique(dec):
        m = dec == v
        if m.sum() < 15 or len(np.unique(y[m])) < 2:
            continue
        tot += roc_auc_score(y[m], s[m]) * m.sum()
        tot_w += m.sum()
    return tot / tot_w if tot_w else float("nan")


def op_points(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    d1 = tpr[fpr <= 0.01][-1] if (fpr <= 0.01).any() else 0.0
    d5 = tpr[fpr <= 0.05][-1] if (fpr <= 0.05).any() else 0.0
    return float(d1), float(d5)


def run(rundir: Path, variant="mean", seeds=6, n_pca=32) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / f"act_{variant}.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    model_id = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]

    rows = {"tribble_oneclass": [], "mahalanobis": [], "surface": [], "length": []}
    ops = {"tribble_oneclass": [], "mahalanobis": []}

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

        # FIS arm: the real library, one-class, built-in whitening, benign only
        with contextlib.redirect_stdout(io.StringIO()):
            det = TribbleOneClassDetector(
                whiten=True,
                whiten_components=min(n_pca, len(fit) - 1),
                n_gaussians=1,
                norm_conorm="probability",
                random_state=seed,
            ).fit(Xdf.iloc[fit])
        s_fis = det.anomaly_score(Xdf)

        s_mah = Mahalanobis(n_pca=min(n_pca, len(fit) - 1)).fit(X[fit]).score(X)
        Xs = surface_features(df, fit)
        s_srf = Mahalanobis(n_pca=4).fit(Xs[fit]).score(Xs)

        rows["tribble_oneclass"].append(within_len_auc(yt, s_fis[ti], tlt))
        rows["mahalanobis"].append(within_len_auc(yt, s_mah[ti], tlt))
        rows["surface"].append(within_len_auc(yt, s_srf[ti], tlt))
        rows["length"].append(within_len_auc(yt, tl[ti].astype(float), tlt))
        ops["tribble_oneclass"].append(op_points(yt, s_fis[ti]))
        ops["mahalanobis"].append(op_points(yt, s_mah[ti]))

    out = {
        "model": model_id,
        "n_injection": int(y.sum()),
        "n_benign": int((~y.astype(bool)).sum()),
        "within_len_auroc": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in rows.items()
        },
        "operating_points": {
            k: {
                "det@1%FP": float(np.mean([o[0] for o in v])),
                "det@5%FP": float(np.mean([o[1] for o in v])),
            }
            for k, v in ops.items()
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/injection"])
    ap.add_argument("--variant", default="mean")
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    allout = []
    print(
        "Parts 3-5 through the REAL tribblefis (TribbleOneClassDetector, "
        "whiten=True, one-class):\n"
    )
    print(
        "%-26s %-18s %8s %8s %10s %10s"
        % ("model", "detector", "within-AUC", "±", "det@1%FP", "det@5%FP")
    )
    for r in a.runs:
        res = run(Path(r), variant=a.variant, seeds=a.seeds)
        allout.append(res)
        m = res["model"].split("/")[-1]
        for det in ("tribble_oneclass", "mahalanobis", "surface", "length"):
            wl = res["within_len_auroc"][det]
            op = res["operating_points"].get(det, {})
            print(
                "%-26s %-18s %8.3f %8.3f %10s %10s"
                % (
                    m,
                    det,
                    wl["mean"],
                    wl["std"],
                    ("%.2f" % op["det@1%FP"]) if op else "-",
                    ("%.2f" % op["det@5%FP"]) if op else "-",
                )
            )
        print()
    Path("runs/parts345_rerun.json").write_text(json.dumps(allout, indent=2))


if __name__ == "__main__":
    main()
