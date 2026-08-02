"""Where do the heuristics sit in the full objective landscape?

Only meaningful where the per-class space ``(2^k - 1)^d`` is small enough to
enumerate, which for these datasets means iris at k in {3, 5}. For each class we
score every subset combination, then locate each selector's answer in that
distribution: its percentile, its gap to the optimum, and how many combinations
it had to evaluate to get there.

    python landscape.py [--dataset iris] [--ks 3,5] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

import datasets as ds
import selection as sel
from ruspini import UnitScaler, fuzzify

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def analyse(dataset: str, k: int, seed: int, lam: float = 1.0,
            convex: bool = False) -> dict[str, Any] | None:
    data = ds.load(dataset)
    xtr, _, ytr, _ = train_test_split(
        data.x, data.y, test_size=0.3, random_state=seed, stratify=data.y
    )
    d = int(xtr.shape[1])
    if not sel.exhaustive_feasible(d, k, convex=convex):
        return None
    scaler = UnitScaler.fit(xtr)
    m = fuzzify(scaler.transform(xtr), k)
    p = sel.n_subsets(k, convex)

    per_class: list[dict[str, Any]] = []
    for c in range(data.n_classes):
        prob = sel.Problem(m=m, in_class=(ytr == c), lam=lam, convex=convex,
                           seed=seed * 1000 + c)
        scores = sel.all_subset_scores(prob)
        assert scores is not None
        best = float(scores.max())
        entry: dict[str, Any] = {
            "class": data.class_names[c],
            "space": int(scores.size),
            "optimum": best,
            "mean": float(scores.mean()),
            "frac_within_1pct": float(np.mean(scores >= best - 0.01 * abs(best))),
            "frac_within_5pct": float(np.mean(scores >= best - 0.05 * abs(best))),
            "n_optimal": int(np.sum(scores >= best - 1e-12)),
            "selectors": {},
        }
        for name in ("mass", "mst_mf", "mst_core", "greedy", "anneal"):
            probe = sel.Problem(m=m, in_class=(ytr == c), lam=lam, convex=convex,
                                seed=seed * 1000 + c)
            s = sel.select(name, probe)
            assert s is not None
            j = probe.score(s)
            entry["selectors"][name] = {
                "objective": j,
                "gap_to_optimum": best - j,
                "percentile": float(np.mean(scores <= j + 1e-12) * 100.0),
                "n_evaluations": int(probe.ev.n_calls),
                "frac_of_space_seen": float(probe.ev.n_calls / scores.size),
                "mfs": int(s.sum()),
            }
        per_class.append(entry)
    return {
        "dataset": dataset, "k": k, "seed": seed, "d": d, "convex": convex,
        "per_class_space": p**d, "classes": per_class,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        f"\n### {report['dataset']}, k={report['k']}, seed {report['seed']}"
        f"{', convex antecedents' if report.get('convex') else ''} — "
        f"{report['per_class_space']:,} subset combinations per class\n",
        "| class | selector | objective | gap to optimum | percentile | evals | % of space |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in report["classes"]:
        for name, got in entry["selectors"].items():
            lines.append(
                f"| {entry['class']} | {name} | {got['objective']:.4f} | "
                f"{got['gap_to_optimum']:.4f} | {got['percentile']:.3f} | "
                f"{got['n_evaluations']} | {100 * got['frac_of_space_seen']:.4f}% |"
            )
        lines.append(
            f"| {entry['class']} | **optimum** | {entry['optimum']:.4f} | 0 | 100 | "
            f"{entry['space']} | 100% |"
        )
    lines.append("\n| class | mean objective | within 1% of optimum | within 5% | tied optima |")
    lines.append("|---|---|---|---|---|")
    for entry in report["classes"]:
        lines.append(
            f"| {entry['class']} | {entry['mean']:.4f} | "
            f"{100 * entry['frac_within_1pct']:.4f}% | "
            f"{100 * entry['frac_within_5pct']:.4f}% | {entry['n_optimal']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="iris")
    ap.add_argument("--ks", default="3,5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--convex", action="store_true")
    args = ap.parse_args()

    reports = []
    body = []
    for k in [int(x) for x in args.ks.split(",")]:
        report = analyse(args.dataset, k, args.seed, convex=args.convex)
        if report is None:
            print(f"k={k}: per-class space out of enumeration budget, skipped")
            continue
        reports.append(report)
        text = render(report)
        body.append(text)
        print(text)

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "-convex" if args.convex else ""
    with open(os.path.join(OUT_DIR, f"landscape{suffix}.json"), "w") as handle:
        json.dump(reports, handle, indent=1)
    with open(os.path.join(OUT_DIR, f"landscape{suffix}.md"), "w") as handle:
        handle.write("# Objective landscape, by exhaustive enumeration\n")
        handle.write("\n".join(body) + "\n")
    print(f"\nWrote {OUT_DIR}/landscape{suffix}.[json|md]")


if __name__ == "__main__":
    main()
