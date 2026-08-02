"""Cross-run comparisons quoted in RESULTS.md.

Reads the JSON written by `run_experiment.py` and answers the questions the
per-run tables cannot: how often greedy actually reached the enumerated
optimum, what the inverse-mass weighting costs or buys, what the convexity
constraint costs, and how the operator ablations compare.

    python summarize.py > outputs/summary.md
"""

from __future__ import annotations

import json
import os
import statistics as st

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SELECTORS = ["mass", "mst_mf", "mst_core", "greedy", "anneal", "exhaustive"]


def load(tag: str) -> dict | None:
    path = os.path.join(OUT_DIR, f"results-{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def cell(values: list[float], fmt: str = "{:.3f}") -> str:
    if not values:
        return "N/A"
    return f"{fmt.format(st.fmean(values))} ± {fmt.format(st.pstdev(values))}"


def get(run: dict, dataset: str, k: int, selector: str) -> dict | None:
    return run["datasets"].get(dataset, {}).get(str(k), {}).get(selector)


def section_optimality(run: dict) -> str:
    """Per seed: did the heuristic reach the enumerated optimum?"""
    lines = ["\n## How often does a heuristic find the optimum?\n",
             "Per-seed comparison of each selector's training objective against "
             "`exhaustive` on the same split. A seed counts as a match at 1e-9.\n",
             "| dataset | k | selector | seeds matching optimum | mean gap | "
             "mean fit seconds |", "|---|---|---|---|---|---|"]
    for dname, per_k in run["datasets"].items():
        for k_s, sels in per_k.items():
            opt = sels.get("exhaustive")
            if not opt:
                continue
            for name in SELECTORS:
                if name == "exhaustive" or name not in sels:
                    continue
                got = sels[name]
                gaps = [o - g for o, g in zip(opt["train_obj"], got["train_obj"])]
                hits = sum(1 for g in gaps if g <= 1e-9)
                lines.append(
                    f"| {dname} | {k_s} | {name} | {hits}/{len(gaps)} | "
                    f"{st.fmean(gaps):.4f} | {st.fmean(got['seconds']):.3f} |")
            lines.append(
                f"| {dname} | {k_s} | _exhaustive_ | {len(opt['train_obj'])}"
                f"/{len(opt['train_obj'])} | 0.0000 | "
                f"{st.fmean(opt['seconds']):.3f} |")
    return "\n".join(lines)


def section_weights(run: dict) -> str:
    lines = ["\n## Inverse-mass rule weighting: accuracy vs macro-F1\n",
             "Same fitted rule bases, two decision rules. `w` = weighted argmax, "
             "`raw` = unweighted.\n",
             "| dataset | k | selector | acc (w) | acc (raw) | F1 (w) | F1 (raw) |",
             "|---|---|---|---|---|---|---|"]
    for dname, per_k in run["datasets"].items():
        for k_s, sels in per_k.items():
            for name in SELECTORS:
                got = sels.get(name)
                if not got or "f1_unweighted" not in got:
                    continue
                lines.append(
                    f"| {dname} | {k_s} | {name} | {cell(got['acc'])} | "
                    f"{cell(got['acc_unweighted'])} | {cell(got['f1'])} | "
                    f"{cell(got['f1_unweighted'])} |")
    return "\n".join(lines)


def section_convex(free: dict, convex: dict) -> str:
    lines = ["\n## Convex (interval) antecedents vs free subsets\n",
             "| dataset | k | selector | acc free | acc convex | obj free | "
             "obj convex | convex frac (free run) |", "|---|---|---|---|---|---|---|"]
    for dname, per_k in free["datasets"].items():
        for k_s, sels in per_k.items():
            for name in SELECTORS:
                a = sels.get(name)
                b = get(convex, dname, int(k_s), name)
                if not a or not b:
                    continue
                frac = cell(a.get("convex_frac", []), "{:.2f}")
                lines.append(
                    f"| {dname} | {k_s} | {name} | {cell(a['acc'])} | "
                    f"{cell(b['acc'])} | {cell(a['train_obj'])} | "
                    f"{cell(b['train_obj'])} | {frac} |")
    return "\n".join(lines)


def section_operators(runs: dict[str, dict]) -> str:
    tags = [t for t in ("main", "tnorm-product", "disj-max") if t in runs]
    lines = ["\n## Operator ablation (test accuracy)\n",
             "| dataset | k | selector | " + " | ".join(tags) + " |",
             "|---" * (3 + len(tags)) + "|"]
    base = runs["main"]
    for dname, per_k in base["datasets"].items():
        for k_s, sels in per_k.items():
            for name in SELECTORS:
                if name not in sels:
                    continue
                row = [dname, k_s, name]
                any_other = False
                for tag in tags:
                    got = get(runs[tag], dname, int(k_s), name)
                    row.append(cell(got["acc"]) if got else "N/A")
                    any_other = any_other or (tag != "main" and got is not None)
                if any_other:
                    lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def section_silent(run: dict) -> str:
    lines = ["\n## Samples where no rule fires at all\n",
             "With C rules and no default rule, a test point can fall outside "
             "every antecedent; the argmax then picks on the tie-break alone.\n",
             "| dataset | k | selector | silent fraction |", "|---|---|---|---|"]
    for dname, per_k in run["datasets"].items():
        for k_s, sels in per_k.items():
            for name in SELECTORS:
                got = sels.get(name)
                if got and st.fmean(got["silent"]) > 0.0:
                    lines.append(f"| {dname} | {k_s} | {name} | "
                                 f"{cell(got['silent'], '{:.3f}')} |")
    return "\n".join(lines)


def main() -> None:
    runs = {tag: run for tag in ("main", "convex", "tnorm-product", "disj-max")
            if (run := load(tag)) is not None}
    if "main" not in runs:
        raise SystemExit("outputs/results-main.json missing; run run_experiment.py first")
    print("# Cross-run summary")
    print(section_optimality(runs["main"]))
    print(section_weights(runs["main"]))
    if "convex" in runs:
        print(section_convex(runs["main"], runs["convex"]))
        print(section_optimality(runs["convex"]).replace(
            "## How often", "## How often (convex antecedents)"))
    print(section_operators(runs))
    print(section_silent(runs["main"]))


if __name__ == "__main__":
    main()
