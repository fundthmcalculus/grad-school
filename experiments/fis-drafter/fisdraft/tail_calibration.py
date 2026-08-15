"""Tail calibration for the strict (1% FPR) operating point.

`det@1%FP` as reported elsewhere is read off the test ROC -- it is the *oracle*
operating point. A deployed monitor cannot do that: it sets a threshold from a
benign *calibration* set and hopes it holds the target FPR on unseen benign
traffic. With a 1% target and only ~100-300 calibration prompts, the empirical
99th-percentile threshold is a high-variance estimate, so the realized FPR drifts
and detection suffers. That gap -- oracle vs deployable -- is what tail
calibration attacks.

Three thresholding methods, all fit on a *calibration* benign split (never the
test benign, never any injection):

  empirical   the (1 - alpha) empirical quantile of calibration benign scores.
  evt         Peaks-Over-Threshold: fit a Generalized Pareto to the exceedances
              over a high anchor (92nd pct), extrapolate the (1-alpha) quantile.
              The standard extreme-quantile estimator for limited data -- a
              smooth tail model instead of one noisy order statistic.
  length_cond empirical quantile computed *within* length deciles, so the FPR is
              equalised across the nuisance variable that differs most.

Reported per method: realized FPR on held-out benign (should hit the 1% target),
detection (TPR) on injections at that threshold, and the gap to the oracle
det@1%FP. On the easy corpus the question is whether EVT recovers the oracle;
on the hard corpus, whether any calibration helps a ranking that barely separates.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.metrics import roc_curve

from tribblefis.one_class import TribbleOneClassDetector
from .fmri_detect import layer_features
from .improve_lowfpr import surprisals


def oracle_det(y, s, cap=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    h = tpr[fpr <= cap]
    return float(h[-1]) if len(h) else 0.0


def thr_empirical(cal, alpha):
    return float(np.quantile(cal, 1 - alpha))


def thr_evt(cal, alpha, anchor_q=0.92):
    """Peaks-Over-Threshold GPD extreme-quantile estimate."""
    u = np.quantile(cal, anchor_q)
    exc = cal[cal > u] - u
    if len(exc) < 15:
        return thr_empirical(cal, alpha)
    xi, _, beta = genpareto.fit(exc, floc=0.0)
    n, nu = len(cal), len(exc)
    # tau = u + (beta/xi)[ (n/nu * alpha)^(-xi) - 1 ]  (alpha = tail prob)
    ratio = (n / nu) * alpha
    if abs(xi) < 1e-6:
        tau = u - beta * np.log(ratio)
    else:
        tau = u + (beta / xi) * (ratio ** (-xi) - 1)
    return float(tau)


def realized(score, y, tl, thr, decile=None):
    """FPR on benign and TPR on injections at threshold thr (or per-decile)."""
    if decile is None:
        flag = score > thr
    else:
        flag = np.zeros(len(score), bool)
        for v in np.unique(decile):
            m = decile == v
            flag[m] = score[m] > thr[v]
    fpr = float(flag[y == 0].mean())
    tpr = float(flag[y == 1].mean())
    return fpr, tpr


def run(rundir: Path, alpha=0.01, seeds=10, n_pca=32) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    mid = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]

    acc = {k: {"fpr": [], "tpr": []} for k in ("empirical", "evt", "length_cond")}
    oracle = []

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = np.where(y == 0)[0]; rng.shuffle(ben); inj = np.where(y == 1)[0]
        n = len(ben)
        fit = ben[: int(0.4 * n)]           # density fit
        cal = ben[int(0.4 * n): int(0.7 * n)]  # threshold calibration
        te_b = ben[int(0.7 * n):]           # held-out benign (measure FPR)
        X, _ = layer_features(act, fit, 8)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        with contextlib.redirect_stdout(io.StringIO()):
            det = TribbleOneClassDetector(
                whiten=True, whiten_components=min(n_pca, len(fit) - 1),
                n_gaussians=1, norm_conorm="probability", random_state=seed).fit(Xdf.iloc[fit])
        S = surprisals(det, Xdf)
        s = np.sort(S, 1)[:, : S.shape[1] - 2].sum(1)   # trimmed score

        ti = np.r_[te_b, inj]
        yt = np.r_[np.zeros(len(te_b)), np.ones(len(inj))]
        oracle.append(oracle_det(yt, s[ti], alpha))

        s_cal = s[cal]
        # empirical + evt (global)
        for name, tau in (("empirical", thr_empirical(s_cal, alpha)),
                          ("evt", thr_evt(s_cal, alpha))):
            fpr, tpr = realized(s[ti], yt, None, tau)
            acc[name]["fpr"].append(fpr); acc[name]["tpr"].append(tpr)
        # length-conditional empirical (deciles fit on calib, applied to test)
        edges = np.quantile(tl[cal], np.linspace(0, 1, 6))
        edges[0], edges[-1] = -np.inf, np.inf
        dcal = np.digitize(tl[cal], edges[1:-1])
        dte = np.digitize(tl[ti], edges[1:-1])
        thr_by = {}
        for v in np.unique(dcal):
            sv = s_cal[dcal == v]
            thr_by[v] = thr_empirical(sv, alpha) if len(sv) >= 10 else thr_empirical(s_cal, alpha)
        thr_vec = {v: thr_by.get(v, thr_empirical(s_cal, alpha)) for v in np.unique(dte)}
        fpr, tpr = realized(s[ti], yt, dte, thr_vec, decile=dte)
        acc["length_cond"]["fpr"].append(fpr); acc["length_cond"]["tpr"].append(tpr)

    out = {"model": mid, "dataset": rundir.name, "alpha": alpha,
           "oracle_det@1%FP": round(float(np.mean(oracle)), 3),
           "methods": {}}
    for k, v in acc.items():
        out["methods"][k] = {
            "realized_FPR": round(float(np.mean(v["fpr"])), 4),
            "realized_FPR_sd": round(float(np.std(v["fpr"])), 4),
            "detection_TPR": round(float(np.mean(v["tpr"])), 3),
            "detection_TPR_sd": round(float(np.std(v["tpr"])), 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["runs/injection_qwen3b", "runs/sg_qwen3b_big", "runs/spml_qwen3b"])
    ap.add_argument("--alpha", type=float, default=0.01)
    a = ap.parse_args()
    allout = []
    print(f"Tail calibration @ target FPR={a.alpha} (trimmed score, one-class, "
          f"threshold from benign calibration only):\n")
    for r in a.runs:
        res = run(Path(r), alpha=a.alpha)
        allout.append(res)
        print(f"== {res['model'].split('/')[-1]} · {Path(r).name} · oracle det@1%FP={res['oracle_det@1%FP']} ==")
        print("  %-13s %14s %16s" % ("method", "realized FPR", "detection TPR"))
        for k, v in res["methods"].items():
            print("  %-13s %7.4f±%.4f %8.3f±%.3f"
                  % (k, v["realized_FPR"], v["realized_FPR_sd"],
                     v["detection_TPR"], v["detection_TPR_sd"]))
        print()
    Path("runs/tail_calibration.json").write_text(json.dumps(allout, indent=2))


if __name__ == "__main__":
    main()
