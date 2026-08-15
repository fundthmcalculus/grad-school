"""Refine the fitted TribbleOneClassDetector with the `optimizers` package.

Part 6 left the one-class FIS detector competitive on AUROC but weak at the
strict operating point (det@1%FP collapsed to ~0.1 while Mahalanobis held ~0.55),
because `1 - max firing` gives benign a heavy upper tail. Moment-matched
Gaussians on the whitened components are already the max-likelihood fit to
benign, so an *unsupervised* refit cannot move that. The lever that can is a
small validation set of known attacks: reshape the antecedent Gaussians so
benign and attack separate at the operating point we actually deploy at.

This is the `refine=True, refine_method="optimizers"` idea from the classifier,
applied to the one-class detector: extract the model's Gaussian antecedent
parameters, and search them with the population + local-polish optimizers
(GA / PSO / ACO) from the `optimizers` package against a validation objective.

Two objectives are compared, because Part 5/6's lesson is that AUROC is the
wrong target for a low-FPR monitor:

  auroc     maximise within-fold AUROC on benign-val + attack-val.
  lowfpr    maximise detection at <=1% FPR (a partial-AUC in the left tail),
            i.e. directly the metric that was broken.

Everything is held to the project's discipline: the antecedents are searched on
a *validation* split and scored on a disjoint *test* split, the unrefined
detector is the baseline, and refinement is never allowed to be reported as an
improvement it did not make on held-out data.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from tribblefis.one_class import TribbleOneClassDetector
from tribblefis.refine import (
    extract_gaussian_params,
    apply_gaussian_params,
    build_param_bounds,
)

from .fmri_detect import layer_features
from .injection_detect_v2 import within_len_auc

warnings.filterwarnings("ignore")


# The installed `optimizers` package uses a keyword-only constructor, so
# tribblefis's `_run_optimizer_search` (positional) does not drive it -- the same
# incompatibility behind that repo's pre-existing optimizers-backend test
# failures. Call the package directly against its current API instead.
_OPT_CLASSES = {
    "ga": ("GeneticAlgorithmOptimizer", "GeneticAlgorithmOptimizerConfig"),
    "pso": ("ParticleSwarmOptimizer", "ParticleSwarmOptimizerConfig"),
    "aco": ("AntColonyOptimizer", "AntColonyOptimizerConfig"),
}


def _optimize(fitness, bounds, x0, method, seed, pop=30, gens=20):
    import optimizers as _o
    from optimizers.continuous.variables import InputContinuousVariable
    from optimizers.core.random import set_seed

    set_seed(seed)
    opt_cls = getattr(_o, _OPT_CLASSES[method][0])
    cfg_cls = getattr(_o, _OPT_CLASSES[method][1])
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0c = np.clip(x0, lo, hi)
    variables = [
        InputContinuousVariable(
            f"p{i}", float(lo[i]), float(hi[i]), initial_value=float(x0c[i])
        )
        for i in range(len(bounds))
    ]

    def fcn(x):
        try:
            return float(fitness(np.asarray(x, dtype=float)))
        except Exception:
            return 1e6

    cfg = cfg_cls(
        name=f"{method}-oneclass", num_generations=gens, population_size=pop, n_jobs=1
    )
    opt = opt_cls(config=cfg, fcn=fcn, variables=variables)
    res = opt.solve()
    best_x = np.clip(np.asarray(res.solution_vector, dtype=float), lo, hi)
    # Never worse than the heuristic start (the wrapper's guarantee, kept here).
    if fcn(best_x) > fcn(x0c):
        best_x = x0c
    return best_x


def det_at_fpr(y, s, fpr_cap=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    hit = tpr[fpr <= fpr_cap]
    return float(hit[-1]) if len(hit) else 0.0


def partial_auc_lowfpr(y, s, fpr_cap=0.01):
    """Area under ROC restricted to FPR <= cap, normalised to [0,1]. A smooth-ish
    left-tail objective; higher is better."""
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= fpr_cap
    if m.sum() < 2:
        return det_at_fpr(y, s, fpr_cap)
    return float(np.trapz(tpr[m], fpr[m]) / fpr_cap)


def refine_detector(det, Zval_ben, Zval_att, objective, method, seed):
    """Search the detector's Gaussian antecedents with the optimizers package.

    `Zval_*` are the *whitened* validation features (the space the model lives
    in), so the fitness can score directly without re-transforming.
    """
    model0 = det.model_
    x0 = extract_gaussian_params(model0)
    # bounds need a frame in the whitened feature space
    Zval = pd.DataFrame(
        np.vstack([Zval_ben, Zval_att]),
        columns=det._transform(
            pd.DataFrame(
                np.zeros((1, len(det.feature_names_in_))), columns=det.feature_names_in_
            )
        ).columns,
    )
    bounds = build_param_bounds(model0, Zval)
    yv = np.r_[np.zeros(len(Zval_ben)), np.ones(len(Zval_att))]

    from tribblefis.gauss_math import tsk_firing_strengths

    def score(model, Z):
        f, _ = tsk_firing_strengths(Z, model, det._norm_params())
        return 1.0 - np.clip(f.max(1) if f.size else np.zeros(len(Z)), 0, 1)

    Zval_df = Zval

    # L2 shrinkage toward the heuristic start (tribblefis's own overfitting
    # control for refinement): with 64 antecedent params and a small attack-val
    # set, an unregularised search games the validation objective and collapses
    # on held-out data. The penalty is scaled to the parameter magnitudes.
    x0_scale = np.maximum(np.abs(x0), 1e-6)
    l2 = 0.15

    def fitness(x):
        m = apply_gaussian_params(model0, x)
        s = score(m, Zval_df)
        shrink = l2 * float(np.mean(((x - x0) / x0_scale) ** 2))
        if objective == "auroc":
            return -roc_auc_score(yv, s) + shrink
        # blend: the low-FPR tail is degenerate (pAUC@1%FP starts at 0), so
        # anchor it with AUROC so the search has a gradient to follow.
        return (
            -(0.5 * roc_auc_score(yv, s) + 0.5 * partial_auc_lowfpr(yv, s, 0.02))
            + shrink
        )

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        best_x = _optimize(fitness, bounds, x0, method, seed)
    secs = time.time() - t0
    info = {}
    refined = TribbleOneClassDetector(**det.get_params())
    # copy fitted state, swap the refined antecedents in
    for attr in (
        "feature_names_in_",
        "top_features_",
        "_pca_",
        "offset_",
        "is_fitted_",
    ):
        if hasattr(det, attr):
            setattr(refined, attr, getattr(det, attr))
    refined.model_ = apply_gaussian_params(model0, best_x)
    return refined, secs, info


def run(
    rundir: Path, objective="lowfpr", methods=("ga", "pso"), n_att=40, seeds=4, n_pca=32
) -> dict:
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    act = np.load(rundir / "act_mean.npy")
    y = (df.label == "injection").to_numpy().astype(int)
    tl = df.tok_len.to_numpy()
    model_id = json.loads((rundir / "meta.json").read_text())["config"]["model_id"]

    acc = {"unrefined": {"auc": [], "d1": []}}
    for m in methods:
        acc[m] = {"auc": [], "d1": [], "secs": []}

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ben = np.where(y == 0)[0]
        rng.shuffle(ben)
        inj = np.where(y == 1)[0]
        rng.shuffle(inj)
        fit = ben[: int(0.5 * len(ben))]
        ben_val = ben[int(0.5 * len(ben)) : int(0.65 * len(ben))]
        ben_test = ben[int(0.65 * len(ben)) :]
        att_val, att_test = inj[:n_att], inj[n_att:]

        X, _ = layer_features(act, fit, 8)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        with contextlib.redirect_stdout(io.StringIO()):
            det = TribbleOneClassDetector(
                whiten=True,
                whiten_components=min(n_pca, len(fit) - 1),
                n_gaussians=1,
                norm_conorm="probability",
                random_state=seed,
            ).fit(Xdf.iloc[fit])

        Zall = det._transform(Xdf)
        yt = np.r_[np.zeros(len(ben_test)), np.ones(len(att_test))]
        tlt = np.r_[tl[ben_test], tl[att_test]]
        ti = np.r_[ben_test, att_test]

        s0 = det.anomaly_score(Xdf)
        acc["unrefined"]["auc"].append(within_len_auc(yt, s0[ti], tlt))
        acc["unrefined"]["d1"].append(det_at_fpr(yt, s0[ti], 0.01))

        Zvb = Zall.iloc[ben_val].to_numpy()
        Zva = Zall.iloc[att_val].to_numpy()
        for m in methods:
            refined, secs, _ = refine_detector(det, Zvb, Zva, objective, m, seed)
            s = refined.anomaly_score(Xdf)
            acc[m]["auc"].append(within_len_auc(yt, s[ti], tlt))
            acc[m]["d1"].append(det_at_fpr(yt, s[ti], 0.01))
            acc[m]["secs"].append(secs)

    def summ(d):
        out = {
            "within_len_auroc": round(float(np.mean(d["auc"])), 3),
            "det@1%FP": round(float(np.mean(d["d1"])), 3),
        }
        if "secs" in d:
            out["refine_s"] = round(float(np.mean(d["secs"])), 2)
        return out

    return {
        "model": model_id,
        "objective": objective,
        "n_attack_val": n_att,
        "arms": {k: summ(v) for k, v in acc.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/injection"])
    ap.add_argument("--objective", default="lowfpr", choices=["auroc", "lowfpr"])
    ap.add_argument("--methods", nargs="+", default=["ga", "pso"])
    ap.add_argument("--n-att", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    allout = []
    print(
        f"Optimizers-package refinement of TribbleOneClassDetector "
        f"(objective={a.objective}, {a.n_att} attack-val examples):\n"
    )
    print(
        "%-24s %-12s %12s %10s %9s"
        % ("model", "arm", "within-AUC", "det@1%FP", "refine_s")
    )
    for r in a.runs:
        res = run(
            Path(r),
            objective=a.objective,
            methods=tuple(a.methods),
            n_att=a.n_att,
            seeds=a.seeds,
        )
        allout.append(res)
        m = res["model"].split("/")[-1]
        for arm, v in res["arms"].items():
            print(
                "%-24s %-12s %12.3f %10.3f %9s"
                % (
                    m,
                    arm,
                    v["within_len_auroc"],
                    v["det@1%FP"],
                    ("%.1f" % v["refine_s"]) if "refine_s" in v else "-",
                )
            )
        print()
    Path("runs/optimizer_refine.json").write_text(json.dumps(allout, indent=2))


if __name__ == "__main__":
    main()
