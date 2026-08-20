"""Experiment 2 -- FIS against every baseline that could embarrass it.

Protocol fixed before running, because the failure mode this design exists to
avoid is the one documented in `experiments/fuzzy-lm-anomaly.md`: two of the
five confounds that study found were *unequal search budget* and an
*under-specified baseline*, and a single scalar statistic beat every learned
detector once the baselines were searched properly.

So:

* **Equal budget.** Every arm gets exactly `--budget` randomly sampled
  hyperparameter configurations. No arm is hand-tuned. The `constant` and
  `single_feature` arms are cheaper by nature and that is stated, not hidden --
  `single_feature` spends its budget searching *which* feature.
* **Three-way split on `prompt_id`, never on rows.** Train fits, validation
  selects the configuration, test is touched once. Steps within a generation
  are dependent, so a row split leaks near-duplicates across the boundary.
* **An MLP is in the baseline set**, because that is what the entropy-gating
  literature (AdaEDL, SpecKV) actually deploys. Beating ridge is not the bar.
* **Wall time is recorded per arm**, since the entire premise is that the FIS
  is cheap enough to run inside a decoding loop.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .exp1b_predictive import TIER_A, TIER_B, add_tier_b

warnings.filterwarnings("ignore")

NORMS = ["probability", "einstein", "hamacher", "min/max"]


# --------------------------------------------------------------------------
# Hyperparameter spaces. Each returns one sampled config.
# --------------------------------------------------------------------------


def space_ridge(r):
    return {"alpha": float(10 ** r.uniform(-4, 3))}


def space_mlp(r):
    return {
        "hidden_layer_sizes": tuple(
            int(r.choice([16, 32, 64, 128])) for _ in range(int(r.integers(1, 3)))
        ),
        "alpha": float(10 ** r.uniform(-6, -1)),
        "learning_rate_init": float(10 ** r.uniform(-4, -2)),
        "max_iter": int(r.choice([100, 200, 400])),
    }


def space_gbm(r):
    return {
        "max_iter": int(r.choice([100, 200, 400, 800])),
        "learning_rate": float(10 ** r.uniform(-2, -0.5)),
        "max_leaf_nodes": int(r.choice([15, 31, 63, 127])),
        "min_samples_leaf": int(r.choice([10, 20, 50, 100])),
        "l2_regularization": float(10 ** r.uniform(-6, 0)),
    }


def space_fis(r):
    return {
        "top_n": int(r.choice([4, 6, 8, 12, 16, -1])),
        "n_gaussians": int(r.choice([0, 2, 3, 4, 5])),
        "n_output_buckets": int(r.choice([3, 5, 7, 10, 15])),
        "tsk_order": str(r.choice(["0th", "1st", "2nd"])),
        "norm_conorm": str(r.choice(NORMS)),
        "output_partition": str(r.choice(["uniform", "quantile"])),
        "l2_reg": float(10 ** r.uniform(-8, -2)),
    }


def space_it2(r):
    c = {
        "top_n": int(r.choice([4, 6, 8, 12, 16])),
        "n_gaussians": int(r.choice([2, 3, 4, 5])),
        "n_output_buckets": int(r.choice([3, 5, 7, 10, 15])),
        "uncertainty_width": float(r.uniform(0.1, 1.0)),
        "km_iterations": int(r.choice([0, 5, 10, 20])),
    }
    if c["km_iterations"] == 0:
        c["km_iterations"] = None
    return c


# --------------------------------------------------------------------------


def make_fitter(arm):
    """Return (build, needs_frame, needs_scaling) for an arm name."""
    if arm == "ridge":
        return (lambda c: Ridge(**c)), False, True
    if arm == "mlp":
        return (lambda c: MLPRegressor(random_state=0, **c)), False, True
    if arm == "gbm":
        return (
            (lambda c: HistGradientBoostingRegressor(random_state=0, **c)),
            False,
            False,
        )
    if arm == "t1_fis":
        from tribblefis.gaussian_regressor import TribbleRegressor

        return (lambda c: TribbleRegressor(random_state=0, **c)), True, False
    if arm == "it2_fis":
        from tribblefis.it2_regressor import IntervalType2FuzzyRegressor

        return (lambda c: IntervalType2FuzzyRegressor(random_state=0, **c)), True, False
    raise ValueError(arm)


SPACES = {
    "ridge": space_ridge,
    "mlp": space_mlp,
    "gbm": space_gbm,
    "t1_fis": space_fis,
    "it2_fis": space_it2,
}


def run_arm(arm, Xtr, ytr, Xva, yva, Xte, yte, cols, budget, seed):
    build, needs_frame, needs_scale = make_fitter(arm)
    space = SPACES[arm]
    r = np.random.default_rng(seed)

    if needs_scale:
        sc = StandardScaler().fit(Xtr)
        Xtr_, Xva_, Xte_ = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    else:
        Xtr_, Xva_, Xte_ = Xtr, Xva, Xte

    if needs_frame:
        Xtr_ = pd.DataFrame(Xtr_, columns=cols)
        Xva_ = pd.DataFrame(Xva_, columns=cols)
        Xte_ = pd.DataFrame(Xte_, columns=cols)

    best = {"val_r2": -np.inf}
    n_failed = 0
    t0 = time.time()
    for i in range(budget):
        cfg = space(r)
        try:
            m = build(cfg).fit(Xtr_, ytr)
            v = r2_score(yva, m.predict(Xva_))
        except Exception as e:  # a config the library rejects still costs budget
            n_failed += 1
            continue
        if np.isfinite(v) and v > best["val_r2"]:
            best = {"val_r2": float(v), "cfg": cfg, "model": m}
    search_s = time.time() - t0

    if "model" not in best:
        return {"arm": arm, "error": "no config succeeded", "n_failed": n_failed}

    t0 = time.time()
    pred = best["model"].predict(Xte_)
    predict_s = time.time() - t0

    return {
        "arm": arm,
        "test_r2": float(r2_score(yte, pred)),
        "val_r2": best["val_r2"],
        "cfg": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in best["cfg"].items()
        },
        "budget": budget,
        "n_failed_cfgs": n_failed,
        "search_s": search_s,
        "predict_us_per_row": 1e6 * predict_s / len(yte),
    }


def run(rundir: Path, target="entropy", budget=32, seed=0) -> dict:
    meta = json.loads((rundir / "meta.json").read_text())
    df = pd.read_parquet(rundir / "steps.parquet")
    df = add_tier_b(df, meta["config"]["model_id"])
    df = df[df.step > 0]

    cols = TIER_A + TIER_B
    X = df[cols].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    if target == "nucleus_90":
        y = np.log(y)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    df, X, y = df[ok], X[ok], y[ok]

    r = np.random.default_rng(seed)
    pids = df.prompt_id.unique()
    r.shuffle(pids)
    a, b = int(0.6 * len(pids)), int(0.8 * len(pids))
    grp = df.prompt_id.to_numpy()
    m_tr = np.isin(grp, pids[:a])
    m_va = np.isin(grp, pids[a:b])
    m_te = np.isin(grp, pids[b:])

    out = {
        "target": target,
        "budget_per_arm": budget,
        "n_features": len(cols),
        "features": cols,
        "n_train": int(m_tr.sum()),
        "n_val": int(m_va.sum()),
        "n_test": int(m_te.sum()),
        "arms": {},
    }

    # --- zero/low-budget reference arms -------------------------------------
    d = DummyRegressor(strategy="mean").fit(X[m_tr], y[m_tr])
    out["arms"]["constant"] = {
        "arm": "constant",
        "test_r2": float(r2_score(y[m_te], d.predict(X[m_te]))),
        "budget": 0,
    }

    # single feature: budget spent choosing WHICH feature, using a fixed learner
    best_j, best_v = None, -np.inf
    for j in range(X.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(
            X[m_tr][:, [j]], y[m_tr]
        )
        v = r2_score(y[m_va], m.predict(X[m_va][:, [j]]))
        if v > best_v:
            best_j, best_v, best_m = j, v, m
    out["arms"]["single_feature"] = {
        "arm": "single_feature",
        "test_r2": float(r2_score(y[m_te], best_m.predict(X[m_te][:, [best_j]]))),
        "val_r2": float(best_v),
        "feature": cols[best_j],
        "budget": X.shape[1],
    }
    print(f"  constant       {out['arms']['constant']['test_r2']:+.4f}")
    print(
        f"  single_feature {out['arms']['single_feature']['test_r2']:+.4f} "
        f"({cols[best_j]})",
        flush=True,
    )

    for arm in ["ridge", "mlp", "gbm", "t1_fis", "it2_fis"]:
        res = run_arm(
            arm,
            X[m_tr],
            y[m_tr],
            X[m_va],
            y[m_va],
            X[m_te],
            y[m_te],
            cols,
            budget,
            seed,
        )
        out["arms"][arm] = res
        if "test_r2" in res:
            print(
                f"  {arm:14s} {res['test_r2']:+.4f}  "
                f"({res['search_s']:.0f}s search, "
                f"{res['predict_us_per_row']:.2f} us/row, "
                f"{res['n_failed_cfgs']} cfg failures)",
                flush=True,
            )
        else:
            print(f"  {arm:14s} FAILED: {res}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--target", default="entropy")
    ap.add_argument("--budget", type=int, default=32)
    # Ten seeds is the protocol in `reproduce/`, not a knob. A five-seed mean
    # in this project once certified as stable a model that diverged one time
    # in ten (WORKINGDOC.md section 3); the seed here redraws both the
    # prompt-level split and the hyperparameter sample, so it varies the two
    # things that could make an arm ordering look real when it is not.
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    rundir = Path(a.run)

    per_seed = []
    for s in range(a.seeds):
        print(f"== target={a.target} budget={a.budget}/arm seed={s} ==", flush=True)
        per_seed.append(run(rundir, a.target, a.budget, seed=s))

    arms = list(per_seed[0]["arms"])
    summary = {}
    for arm in arms:
        vals = [
            d["arms"][arm]["test_r2"]
            for d in per_seed
            if "test_r2" in d["arms"].get(arm, {})
        ]
        us = [
            d["arms"][arm]["predict_us_per_row"]
            for d in per_seed
            if "predict_us_per_row" in d["arms"].get(arm, {})
        ]
        summary[arm] = {
            "mean_test_r2": float(np.mean(vals)) if vals else float("nan"),
            "std_test_r2": (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            ),
            "min_test_r2": float(np.min(vals)) if vals else float("nan"),
            "n_seeds": len(vals),
            "mean_predict_us_per_row": float(np.mean(us)) if us else None,
        }

    # Paired comparisons against the FIS. Pairing on seed removes the split
    # variance, which is the dominant term -- an unpaired comparison of two
    # arms at this sample size cannot resolve the gaps involved.
    paired = {}
    for arm in arms:
        if arm == "t1_fis":
            continue
        d = [
            per_seed[i]["arms"]["t1_fis"]["test_r2"]
            - per_seed[i]["arms"][arm]["test_r2"]
            for i in range(len(per_seed))
            if "test_r2" in per_seed[i]["arms"].get(arm, {})
            and "test_r2" in per_seed[i]["arms"].get("t1_fis", {})
        ]
        if d:
            paired[f"t1_fis - {arm}"] = {
                "mean": float(np.mean(d)),
                "std": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
                "n_wins": int(sum(x > 0 for x in d)),
                "n": len(d),
            }

    out = {
        "target": a.target,
        "budget_per_arm": a.budget,
        "n_seeds": a.seeds,
        "summary": summary,
        "paired_vs_t1_fis": paired,
        "per_seed": per_seed,
    }
    (rundir / f"exp2_bakeoff_{a.target}.json").write_text(json.dumps(out, indent=2))
    print(f"\n== {a.target}: mean +/- sd over {a.seeds} seeds ==")
    for arm, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean_test_r2"]):
        print(
            f"  {arm:16s} {s['mean_test_r2']:+.4f} +/- {s['std_test_r2']:.4f} "
            f"(worst {s['min_test_r2']:+.4f})"
        )


if __name__ == "__main__":
    main()
