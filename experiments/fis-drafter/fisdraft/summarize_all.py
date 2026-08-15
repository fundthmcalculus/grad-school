"""Master comparison across every captured model, with the log-domain fix.

For each runs/injection_* directory, reports the one-class detector at three
scores -- the current `1 - max firing`, the surprisal sum (= Mahalanobis), and
the robust trimmed sum -- plus the best of the three ideas, so the whole model
set reads as one table. Sorted by model size.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tribblefis.one_class import TribbleOneClassDetector
from tribblefis.refine import extract_gaussian_params
from .fmri_detect import layer_features
from .improve_lowfpr import det_at, surprisals, local_calibrated

# model_id substring -> (display name, approx params in B)
SIZE = {
    "SmolLM2-135M": ("SmolLM2-135M", 0.135), "SmolLM2-360M": ("SmolLM2-360M", 0.36),
    "gemma-3-270m": ("gemma-3-270m-it", 0.27), "TinyLlama": ("TinyLlama-1.1B", 1.1),
    "LFM2.5-1.2B": ("LFM2.5-1.2B", 1.2), "Qwen2.5-3B": ("Qwen2.5-3B", 3.0),
    "Qwen2.5-7B": ("Qwen2.5-7B", 7.0), "Qwen2.5-14B": ("Qwen2.5-14B", 14.0),
    "pythia": ("pythia-410m (base)", 0.41),
}


def label(model_id):
    for k, v in SIZE.items():
        if k in model_id:
            return v
    return (model_id.split("/")[-1], 99.0)


def evaluate(rundir: Path, seeds=6, n_pca=32) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    mid = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]
    tl = df.tok_len.to_numpy()

    rows = {k: [] for k in ["complement", "surprisal", "trimmed", "local", "auroc"]}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = np.where(y == 0)[0]; rng.shuffle(ben); inj = np.where(y == 1)[0]
        fit = ben[: int(0.6 * len(ben))]; tb = ben[int(0.6 * len(ben)):]
        ti = np.r_[tb, inj]; yt = np.r_[np.zeros(len(tb)), np.ones(len(inj))]
        X, _ = layer_features(act, fit, 8)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        with contextlib.redirect_stdout(io.StringIO()):
            det = TribbleOneClassDetector(
                whiten=True, whiten_components=min(n_pca, len(fit) - 1),
                n_gaussians=1, norm_conorm="probability", random_state=seed).fit(Xdf.iloc[fit])
        S = surprisals(det, Xdf)
        comp = 1.0 - np.exp(-S.sum(1))
        surp = S.sum(1)
        trim = np.sort(S, 1)[:, : S.shape[1] - 2].sum(1)
        loc = local_calibrated(det, Xdf, fit, surp, k=20)
        rows["complement"].append(det_at(yt, comp[ti], 0.01))
        rows["surprisal"].append(det_at(yt, surp[ti], 0.01))
        rows["trimmed"].append(det_at(yt, trim[ti], 0.01))
        rows["local"].append(det_at(yt, loc[ti], 0.01))
        from sklearn.metrics import roc_auc_score
        rows["auroc"].append(roc_auc_score(yt, surp[ti]))
    name, params = label(mid)
    return {"name": name, "params": params,
            **{k: round(float(np.mean(v)), 3) for k, v in rows.items()}}


def main():
    runs = sorted(Path("runs").glob("injection_*"))
    runs = [r for r in runs if (r / "act_mean.npy").exists()]
    # include the base deepset run (runs/injection) too
    if (Path("runs/injection") / "act_mean.npy").exists():
        runs = [Path("runs/injection")] + runs
    results = []
    for r in runs:
        try:
            results.append(evaluate(r))
        except Exception as e:
            print(f"skip {r}: {e}")
    results.sort(key=lambda d: d["params"])
    print("\nMaster det@1%FP comparison (deepset, one-class, 6 seeds), by model size:\n")
    print("%-22s %6s %11s %11s %10s %9s %8s"
          % ("model", "params", "complement", "surprisal", "trimmed", "local", "AUROC"))
    print("%-22s %6s %11s %11s %10s %9s %8s"
          % ("", "(B)", "(current)", "(=Mahal)", "(robust)", "calib", ""))
    for d in results:
        best = max(d["surprisal"], d["trimmed"], d["local"])
        bm = {d["surprisal"]: "S", d["trimmed"]: "T", d["local"]: "L"}[best]
        print("%-22s %6.2f %11.3f %11.3f %10.3f %9.3f %8.3f  best=%s"
              % (d["name"], d["params"], d["complement"], d["surprisal"],
                 d["trimmed"], d["local"], d["auroc"], bm))
    Path("runs/master_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
