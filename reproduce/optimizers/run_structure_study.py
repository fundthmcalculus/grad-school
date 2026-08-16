#!/usr/bin/env python3
"""Structure search: let the optimizer decide how many rules and how many
membership functions the model needs.

    uv run --project tribble-fis --with-editable tribble-opt --with scikit-learn \
        python reproduce/optimizers/run_structure_study.py --smoke
    uv run --project tribble-fis --with-editable tribble-opt --with scikit-learn \
        python reproduce/optimizers/run_structure_study.py --archive <label>

The antecedent study holds the model's shape fixed and tunes what is inside it.
This one hands the shape over: output-bucket count (which *is* the rule count
for a regression MoG-TSK), Gaussians per feature, features retained, consequent
order, and ridge strength. Five decision variables, four of them discrete.

Three things worth knowing before reading a result.

**Evaluations here are not evaluations there.** One evaluation in this study
rebuilds the entire model — re-partitions the output, re-ranks the features,
re-fits every mixture — where one evaluation in the antecedent study only
re-solved consequents against a fixed structure. The two budgets are not
comparable and the tables never put them on the same axis.

**The reference is the configuration the pipeline ships with**, `DEFAULT` in
`structure.py`: three buckets, automatic Gaussian count, all eight features,
2nd-order consequents, ridge 1e-2. That is Chapter 6's setup, so "did the search
beat it" is a question about the shipped defaults rather than about an arbitrary
baseline.

**Rule count is reported, never optimized.** The objective is held-out error
alone. Penalising complexity would fix an exchange rate between rules and
accuracy that nobody in this document has justified, and the interpretability
argument depends on that rate being the reader's to choose. The table prints
what each answer costs in rules and membership functions so the trade is
visible.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "reproduce"))

import arms as A  # noqa: E402
import common as C  # noqa: E402
import structure as S  # noqa: E402
from budget import BudgetedObjective  # noqa: E402


class _StructureAdapter:
    """Give `arms.py` the shape it expects: `.x0`, `.bounds`, `.n_params`."""

    def __init__(self, x0, bounds):
        self.x0 = np.asarray(x0, dtype=float)
        self.bounds = bounds
        self.n_params = len(self.x0)


def run_one(arm, seed, budget, start):
    """One (arm, seed) structure search. Never raises."""
    problem = S.StructureProblem(seed=seed)
    x0 = S.encode(S.DEFAULT if start == "default" else _random_structure(seed))
    obj = BudgetedObjective(problem.objective(), max_evals=budget, x0=x0).start()

    error = None
    try:
        A.run(arm, obj, _StructureAdapter(x0, S.bounds()), seed)
    except Exception as exc:  # noqa: BLE001
        error = f"{exc.__class__.__name__}: {exc}"
    obj.finalize()

    found = S.decode(obj.best_x)
    r2, rmse, n_rules, n_mfs = problem.test_score(found)
    ref = S.DEFAULT
    ref_r2, ref_rmse, ref_rules, ref_mfs = problem.test_score(ref)
    return {
        "arm": arm,
        "seed": seed,
        "start": start,
        "structure": found,
        "cv_mse": obj.best_f,
        "cv_mse_0": obj.f0,
        "r2": r2,
        "rmse": rmse,
        "n_rules": n_rules,
        "n_mfs": n_mfs,
        "ref_r2": ref_r2,
        "ref_rmse": ref_rmse,
        "ref_rules": ref_rules,
        "ref_mfs": ref_mfs,
        "evals": obj.n_evals,
        "error": error,
    }


def _random_structure(seed):
    """A structure drawn uniformly from the space -- the cold start."""
    rng = np.random.default_rng([seed, 0x57C7])
    return {
        "n_buckets": int(rng.choice(S.SPACE["n_buckets"])),
        "n_gaussians": int(rng.choice(S.SPACE["n_gaussians"])),
        "top_n": int(rng.choice(S.SPACE["top_n"])),
        "order": str(rng.choice(S.SPACE["order"])),
        "log10_l2": float(rng.uniform(*S.SPACE["log10_l2"])),
    }


def _mode(values):
    """The most common choice across seeds, with how often it was chosen."""
    uniq, counts = np.unique(np.asarray(values, dtype=object), return_counts=True)
    i = int(np.argmax(counts))
    return uniq[i], int(counts[i])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arms", default="scipy-de,opt-ga,opt-pso,opt-aco")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--start", default="default", choices=["default", "random"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--archive", metavar="LABEL")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.budget = args.seeds or "0,1,2", min(args.budget, 30)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else C.SEEDS
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(
        f"structure search: budget={args.budget} model builds/arm/seed, "
        f"start={args.start}, seeds={seeds}"
    )

    rows, records = [], []
    for arm in arm_names:
        per_arm = []
        for seed in seeds:
            rec = run_one(arm, seed, args.budget, args.start)
            per_arm.append(rec)
            records.append(rec)
            st = rec["structure"]
            print(
                f"  {arm:<12} seed {seed}: {st['n_buckets']} rules, "
                f"g={st['n_gaussians']}, top_n={st['top_n']}, "
                f"{st['order']}, l2=1e{st['log10_l2']:.1f}  ->  "
                f"R2 {rec['r2']:.3f} (default {rec['ref_r2']:.3f})"
            )

        d_r2 = [
            r["r2"] - r["ref_r2"]
            for r in per_arm
            if np.isfinite(r["r2"]) and np.isfinite(r["ref_r2"])
        ]
        won = sum(1 for d in d_r2 if d > 0)
        buckets, bn = _mode([r["structure"]["n_buckets"] for r in per_arm])
        order, on = _mode([r["structure"]["order"] for r in per_arm])
        topn, tn = _mode([r["structure"]["top_n"] for r in per_arm])
        rows.append(
            [
                arm,
                f"{buckets} ({bn}/{len(per_arm)})",
                f"{order} ({on}/{len(per_arm)})",
                f"{topn} ({tn}/{len(per_arm)})",
                C.cell([r["n_mfs"] for r in per_arm], fmt="{:.0f}"),
                C.cell([r["r2"] for r in per_arm]),
                C.cell(d_r2, fmt="{:+.3f}") if d_r2 else C.NA,
                f"{won}/{len(d_r2)}",
            ]
        )

    ref = records[0] if records else None
    C.emit(
        "table_opt_structure",
        "Structure search — rules and membership functions as decision variables "
        "(concrete)",
        [
            "arm",
            "rules found (modal)",
            "order (modal)",
            "features (modal)",
            "membership fns",
            "test R²",
            "Δ R² vs shipped default",
            "wins",
        ],
        rows,
        note=(
            f"Five decision variables: output buckets (= rules), Gaussians per "
            f"feature (−1 = automatic), features retained, consequent order, and "
            f"ridge strength. Budget is {args.budget} **model builds** per arm "
            f"per seed — an evaluation here rebuilds the whole model, so these "
            f"are not comparable to the antecedent study's evaluations and the "
            f"two must not share an axis. Objective is k-fold held-out MSE on the "
            f"training split only, with the output partition recomputed on "
            f"training rows so the test target distribution cannot leak into the "
            f"rule centres. The reference is the configuration the pipeline "
            f"ships with — {S.DEFAULT['n_buckets']} rules, automatic Gaussian "
            f"count, all {S.DEFAULT['top_n']} features, "
            f"{S.DEFAULT['order']} consequents, ridge 1e{S.DEFAULT['log10_l2']:.0f} "
            f"— which scores R² {ref['ref_r2']:.3f} with {ref['ref_mfs']} "
            f"membership functions. **Rule count is reported, never optimized:** "
            f"the objective is accuracy alone, because a complexity penalty would "
            f"fix an exchange rate between rules and error that this document has "
            f"not justified."
            if ref
            else ""
        ),
    )

    path = os.path.join(C.OUTPUT_DIR, "table_opt_structure_seeds.csv")
    C.write_csv(
        path,
        [
            "arm",
            "seed",
            "start",
            "n_buckets",
            "n_gaussians",
            "top_n",
            "order",
            "log10_l2",
            "n_rules",
            "n_mfs",
            "cv_mse",
            "r2",
            "rmse",
            "ref_r2",
            "ref_rmse",
            "evals",
        ],
        [
            [
                r["arm"],
                r["seed"],
                r["start"],
                r["structure"]["n_buckets"],
                r["structure"]["n_gaussians"],
                r["structure"]["top_n"],
                r["structure"]["order"],
                f"{r['structure']['log10_l2']:.4f}",
                r["n_rules"],
                r["n_mfs"],
                f"{r['cv_mse']:.6f}",
                f"{r['r2']:.6f}",
                f"{r['rmse']:.4f}",
                f"{r['ref_r2']:.6f}",
                f"{r['ref_rmse']:.4f}",
                r["evals"],
            ]
            for r in records
        ],
    )
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
