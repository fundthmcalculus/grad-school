"""Driver: sweep datasets x k x selector, 10 seeds, stratified 70/30 splits.

    python run_experiment.py                 # full sweep
    python run_experiment.py --quick         # 3 seeds, iris+wine, k in {3,5}
    python run_experiment.py --rules iris:5:greedy

Seeds follow the repository standard in `reproduce/common.py`: ten of them,
overridable with REPRO_SEEDS for a smoke run. Every number reported is a mean
+/- population std across those seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier

import datasets as ds
import model as mdl
import selection as sel
from ruspini import partition_defect

SEEDS = [int(s) for s in os.environ.get("REPRO_SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
SELECTOR_ORDER = ["mass", "mst_mf", "mst_core", "greedy", "anneal", "exhaustive"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
NA = "N/A"


def agg(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    return (statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0)


def cell(values: list[float], fmt: str = "{:.3f}") -> str:
    if not values:
        return NA
    mean, std = agg(values)
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def split(data: ds.Dataset, seed: int) -> tuple[Any, Any, Any, Any]:
    return train_test_split(
        data.x, data.y, test_size=0.3, random_state=seed, stratify=data.y
    )


def run_reference(data: ds.Dataset, seeds: list[int]) -> dict[str, list[float]]:
    """Context, not competition: an unconstrained tree, and a C-prototype model."""
    out: dict[str, list[float]] = {"tree": [], "nearest_centroid": []}
    for seed in seeds:
        xtr, xte, ytr, yte = split(data, seed)
        tree = DecisionTreeClassifier(random_state=seed).fit(xtr, ytr)
        out["tree"].append(float(accuracy_score(yte, tree.predict(xte))))
        nc = NearestCentroid().fit(xtr, ytr)
        out["nearest_centroid"].append(float(accuracy_score(yte, nc.predict(xte))))
    return out


def run_cell(
    data: ds.Dataset,
    k: int,
    selector: str,
    seeds: list[int],
    tnorm: str,
    disjunction: str,
    lam: float,
    convex: bool = False,
) -> dict[str, Any] | None:
    rec: dict[str, list[float]] = {
        "acc": [], "acc_unweighted": [], "f1": [], "f1_unweighted": [], "train_obj": [],
        "seconds": [], "mfs_per_rule": [], "dontcare": [], "silent": [], "convex_frac": [],
    }
    for seed in seeds:
        xtr, xte, ytr, yte = split(data, seed)
        fit = mdl.fit(
            xtr, ytr, k, selector, data.n_classes,
            lam=lam, tnorm=tnorm, disjunction=disjunction, convex=convex,
            seed=seed, class_names=data.class_names,
        )
        if fit is None:
            return None
        mte = mdl.memberships(fit.scaler, xte, k)
        priors = np.bincount(ytr, minlength=data.n_classes).astype(np.float64)
        priors /= priors.sum()

        pred = fit.model.predict(mte, priors)
        rec["acc"].append(float(accuracy_score(yte, pred)))
        rec["f1"].append(float(f1_score(yte, pred, average="macro")))

        # Same fitted rule base, argmax without the inverse-mass weights: the
        # weighting is a decision-rule choice, not part of the selection.
        saved = fit.model.weights
        fit.model.weights = None
        raw = fit.model.predict(mte, priors)
        rec["acc_unweighted"].append(float(accuracy_score(yte, raw)))
        rec["f1_unweighted"].append(float(f1_score(yte, raw, average="macro")))
        fit.model.weights = saved

        cx = fit.model.complexity()
        rec["train_obj"].append(fit.train_objective)
        rec["seconds"].append(fit.fit_seconds)
        rec["mfs_per_rule"].append(cx["mfs_per_rule"])
        rec["dontcare"].append(cx["dontcare_vars_per_rule"])
        rec["convex_frac"].append(cx["convex_frac"])
        rec["silent"].append(fit.model.silent_rate(mte))
    return {key: list(vals) for key, vals in rec.items()}


def _table(per_k: dict[str, Any], ks: list[int], selectors: list[str],
           key: str, fmt: str) -> list[str]:
    """One selector-by-k table for a single recorded quantity."""
    head = "| selector | " + " | ".join(f"k={k}" for k in ks) + " |"
    lines = [head, "|---" * (len(ks) + 1) + "|"]
    for name in selectors:
        row = [name]
        for k in ks:
            got = per_k.get(str(k), {}).get(name)
            row.append(cell(got[key], fmt) if got else NA)
        lines.append("| " + " | ".join(row) + " |")
    return lines


def markdown_tables(results: dict[str, Any], ks: list[int], selectors: list[str]) -> str:
    panels = [
        ("acc", "{:.3f}", "**Test accuracy** (mean ± std over {seeds} seeds)"),
        ("mfs_per_rule", "{:.1f}", "**MFs selected per rule** (of {available})"),
        ("train_obj", "{:.3f}", "**Training objective** (sum of the C one-vs-rest "
                                "margins; this is what every selector optimises)"),
        ("convex_frac", "{:.2f}", "**Fraction of antecedents that are contiguous** "
                                  "(1.0 = every rule reads as a linguistic term)"),
        ("silent", "{:.3f}", "**Fraction of test samples where no rule fires**"),
        ("seconds", "{:.3f}", "**Fit seconds per model**"),
    ]
    lines: list[str] = []
    for dname, per_k in results["datasets"].items():
        meta = results["meta"]["datasets"][dname]
        lines.append(
            f"\n### {dname} — {meta['n']} samples, {meta['d']} features, "
            f"{meta['C']} classes → {meta['C']} rules\n"
        )
        for key, fmt, title in panels:
            available = " / ".join(f"{meta['d'] * k} available at k={k}" for k in ks)
            lines.append(title.format(seeds=len(results["meta"]["seeds"]),
                                      available=available) + "\n")
            lines.extend(_table(per_k, ks, selectors, key, fmt))
            if key == "acc":
                for name, vals in results["reference"][dname].items():
                    lines.append(f"| _{name}_ | " + " | ".join([cell(vals)] * len(ks)) + " |")
            lines.append("")
    return "\n".join(lines)


def dump_rules(spec: str, seed: int, tnorm: str, disjunction: str, lam: float,
               convex: bool = False) -> str:
    dname, k_s, selector = spec.split(":")
    k = int(k_s)
    data = ds.load(dname)
    xtr, xte, ytr, yte = split(data, seed)
    fit = mdl.fit(xtr, ytr, k, selector, data.n_classes, lam=lam, tnorm=tnorm,
                  disjunction=disjunction, convex=convex, seed=seed,
                  class_names=data.class_names)
    if fit is None:
        return f"{spec}: selector could not run (out of budget)"
    mte = mdl.memberships(fit.scaler, xte, k)
    priors = np.bincount(ytr, minlength=data.n_classes).astype(np.float64)
    priors /= priors.sum()
    acc = accuracy_score(yte, fit.model.predict(mte, priors))
    body = fit.model.describe(data.feature_names, data.class_names)
    return (f"{spec} (seed {seed}) — test accuracy {acc:.3f}, "
            f"{fit.model.complexity()['mfs_per_rule']:.1f} MFs/rule\n\n{body}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="iris,wine,glass")
    ap.add_argument("--ks", default="3,5,7")
    ap.add_argument("--selectors", default=",".join(SELECTOR_ORDER))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--tnorm", default="min", choices=["min", "product"])
    ap.add_argument("--disjunction", default="sum", choices=["sum", "max"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--convex", action="store_true",
                    help="restrict every antecedent to one contiguous run of MFs")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rules", default="", help="dataset:k:selector, dump the rule text")
    ap.add_argument("--tag", default="main", help="output filename tag")
    args = ap.parse_args()

    if args.quick:
        args.datasets, args.ks, args.seeds = "iris,wine", "3,5", "0,1,2"

    if args.rules:
        print(dump_rules(args.rules, int(args.seeds.split(",")[0]),
                         args.tnorm, args.disjunction, args.lam, args.convex))
        return

    names = args.datasets.split(",")
    ks = [int(k) for k in args.ks.split(",")]
    selectors = args.selectors.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    for k in ks:
        defect = partition_defect(k)
        if defect > 1e-12:
            raise AssertionError(f"k={k} is not a partition of unity (defect {defect:.2e})")
    print(f"Ruspini partition-of-unity check passed for k in {ks}")

    results: dict[str, Any] = {
        "meta": {
            "seeds": seeds, "ks": ks, "selectors": selectors, "tnorm": args.tnorm,
            "disjunction": args.disjunction, "lam": args.lam,
            "convex": args.convex, "datasets": {},
        },
        "datasets": {}, "reference": {},
    }
    t_start = time.perf_counter()
    for name in names:
        data = ds.load(name)
        n, d = data.shape
        results["meta"]["datasets"][name] = {"n": n, "d": d, "C": data.n_classes}
        results["reference"][name] = run_reference(data, seeds)
        results["datasets"][name] = {}
        for k in ks:
            per_var = sel.n_subsets(k, args.convex)
            space = per_var ** (d * data.n_classes)
            print(f"\n{name} k={k}: search space {per_var}^({d}*{data.n_classes}) "
                  f"= {space:.3e}"
                  f"{' [convex antecedents only]' if args.convex else ''}"
                  f", exhaustive per class "
                  f"{'feasible' if sel.exhaustive_feasible(d, k, convex=args.convex) else 'out of budget'}")
            results["datasets"][name][str(k)] = {}
            for s in selectors:
                t0 = time.perf_counter()
                got = run_cell(data, k, s, seeds, args.tnorm, args.disjunction,
                               args.lam, args.convex)
                if got is None:
                    print(f"  {s:<11} {NA} (out of budget)")
                    continue
                results["datasets"][name][str(k)][s] = got
                print(f"  {s:<11} acc {cell(got['acc'])}  obj {cell(got['train_obj'])}"
                      f"  mfs {cell(got['mfs_per_rule'], '{:.1f}')}"
                      f"  [{time.perf_counter() - t0:.1f}s]")

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, f"results-{args.tag}.json")
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=1)
    md_path = os.path.join(OUT_DIR, f"tables-{args.tag}.md")
    with open(md_path, "w") as handle:
        handle.write(f"# Results ({args.tag}) — t-norm `{args.tnorm}`, "
                     f"disjunction `{args.disjunction}`, lambda {args.lam}\n")
        handle.write(markdown_tables(results, ks, selectors))
        handle.write("\n")
    print(f"\nTotal {time.perf_counter() - t_start:.1f}s")
    print(f"Wrote {json_path}\n      {md_path}")


if __name__ == "__main__":
    main()
