"""Few-shot injection detection: a handful of attacks solve the strict gate.

Synthesis of the tail-separation round. The one-class detector's strict-FPR
failure on hard corpora is a *ranking* problem: it scores distance-from-normal
(a magnitude), discarding the *direction* attacks point in. Unsupervised scores
(OCSVM, kNN) recover only a little of that direction (safeguard det@1%FP
0.06 -> 0.10). A few labelled attacks recover it fully.

The method (a score-level Deep-SAD, per the research shortlist): fit the benign
whitening as before; when N attack examples are available, add an L2-regularised
logistic-regression discriminant on the whitened features. Balanced classes;
benign-validation split as the negatives. The discriminant *direction* is what
the one-class magnitude score threw away.

Reported: the 0-shot trimmed baseline vs few-shot logistic at N in {5,10,25},
det@1%FP and within-length AUROC, across every captured model and corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from .fmri_detect import layer_features
from .injection_detect_v2 import within_len_auc
from .summarize_all import label


def det_at(y, s, cap=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    h = tpr[fpr <= cap]
    return float(h[-1]) if len(h) else 0.0


def run(rundir: Path, n_atts=(0, 5, 10, 25), seeds=8, n_pca=32) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    mid = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]

    acc = {n: {"d1": [], "wl": []} for n in n_atts}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = np.where(y == 0)[0]; rng.shuffle(ben)
        inj = np.where(y == 1)[0]; rng.shuffle(inj)
        fitb = ben[: int(0.5 * len(ben))]
        bval = ben[int(0.5 * len(ben)): int(0.65 * len(ben))]
        btest = ben[int(0.65 * len(ben)):]
        X, _ = layer_features(act, fitb, 8)
        Z = PCA(n_components=min(n_pca, len(fitb) - 1), whiten=True,
                random_state=seed).fit(X[fitb]).transform(X)
        for n in n_atts:
            aval, atest = inj[:n], inj[n:]
            ti = np.r_[btest, atest]
            yt = np.r_[np.zeros(len(btest)), np.ones(len(atest))]
            tlt = np.r_[tl[btest], tl[atest]]
            if n == 0:
                S = 0.5 * Z ** 2
                s = np.sort(S, 1)[:, : S.shape[1] - 2].sum(1)   # trimmed baseline
            else:
                Xtr = np.vstack([Z[bval], Z[aval]])
                ytr = np.r_[np.zeros(len(bval)), np.ones(len(aval))]
                m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, ytr)
                s = m.decision_function(Z)
            acc[n]["d1"].append(det_at(yt, s[ti]))
            acc[n]["wl"].append(within_len_auc(yt, s[ti], tlt))
    name, params = label(mid)
    return {"model": name, "params": params, "dataset": rundir.name,
            "curve": {n: {"det@1%FP": round(float(np.mean(v["d1"])), 3),
                          "det@1%FP_sd": round(float(np.std(v["d1"])), 3),
                          "wl_auroc": round(float(np.mean(v["wl"])), 3)}
                      for n, v in acc.items()}}


def dataset_of(name):
    for k, v in {"jailbreak": "jailbreak", "sg_": "safeguard", "spml": "spml"}.items():
        if k in name:
            return v
    return "deepset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None)
    a = ap.parse_args()
    if a.runs:
        runs = [Path(r) for r in a.runs]
    else:
        runs = ([Path("runs/injection")] + sorted(Path("runs").glob("injection_*"))
                + sorted(Path("runs").glob("sg_*")) + sorted(Path("runs").glob("spml_*")))
    runs = [r for r in runs if (r / "act_mean.npy").exists()]
    res = []
    for r in runs:
        try:
            res.append(run(r))
        except Exception as e:
            print(f"skip {r.name}: {e}")
    res.sort(key=lambda d: (dataset_of(d["dataset"]), d["params"]))
    print("\nFew-shot injection detection: det@1%FP by #attack examples "
          "(N=0 is the unsupervised trimmed baseline):\n")
    print("%-20s %-10s %8s %8s %8s %8s" % ("model", "dataset", "N=0", "N=5", "N=10", "N=25"))
    for d in res:
        c = d["curve"]
        print("%-20s %-10s %8.3f %8.3f %8.3f %8.3f"
              % (d["model"], dataset_of(d["dataset"]),
                 c[0]["det@1%FP"], c[5]["det@1%FP"], c[10]["det@1%FP"], c[25]["det@1%FP"]))
    Path("runs/fewshot.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
