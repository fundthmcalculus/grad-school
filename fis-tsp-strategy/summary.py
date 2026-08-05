"""One flat size / quality / time table over everything that has been measured.

The results are spread over three files that answer different questions — the test-set
frontier comparison at two rule-base scales, and the LKH curve study — and each prints its own
shape. That is right for each of them and inconvenient for the ordinary question, which is
"what does this arm cost and what does it get, at this size".

So this joins them into one long table: one row per (instance, arm), carrying n, the gap over
the published optimum, and wall clock. Long rather than wide, because the arms differ between
sources and a wide table would be mostly empty cells.

``q_arm_mean`` is carried where it exists, and its name says what it is. It is the
frontier-relative number the test-set benchmark reports — tour length over what the swept LK
frontier reaches at the same wall clock — but *averaged over the whole test set*, so it is a
property of the arm and repeats down every row of that arm rather than describing the instance
beside it. It is the honest comparison within this project. It is deliberately blank for the
LKH study's arms, which are measured against LKH rather than against the sweep; filling it in
would silently mix two denominators in one column.

Run:  python summary.py                    # markdown to stdout, CSV to results/summary.csv
      python summary.py --sort n           # or: arm, gap, s
      python summary.py --arms iterated lkh
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import paths

#: How each source file names its arms, and what to call them here. Anything not listed is
#: taken through under its own name.
RENAME = {
    "lk_32_2_4": "LK k32/d2/b4",
    "lk_32_3_8": "LK k32/d3/b8",
    "lk_32_6_32": "LK k32/d6/b32",
    "lk_32_10_32": "LK k32/d10/b32",
    "lk_48_10_32": "LK k48/d10/b32",
    "fis_effort_greedy": "FIS effort",
    "fis_effort_chain_greedy": "FIS effort+chain",
    "fis_defer": "FIS deferred verification",
    "fis_full": "FIS full (fuzzy construction)",
    "construct_greedy": "greedy-edge construction",
    "construct_nn": "nearest-neighbour construction",
    "fis_ls": "FIS local search",
    "iterated": "iterated (control)",
    "iterated_aim": "iterated + EFFORT-aimed kicks",
    "iterated_chain": "iterated + CHAIN reopt depth",
    "iterated_fis": "iterated + both",
}


def _from_benchmark(path, scale):
    """Rows from a ``results_<scale>.json``: every arm it measured, on every test instance."""
    if not Path(path).exists():
        return []
    doc = json.loads(Path(path).read_text())
    rows = []
    for r in doc["rows"]:
        for key in [k[:-4] for k in r if k.endswith("_gap")]:
            if f"{key}_s" not in r:
                continue
            rows.append({
                "source": f"benchmark/{scale}",
                "instance": r["name"],
                "n": r["n"],
                "arm": RENAME.get(key, key),
                "gap_pct": r[f"{key}_gap"],
                "seconds": r[f"{key}_s"],
                "q_arm_mean": None,
                "detail": "",
            })
    # q lives in the summary block, keyed by arm rather than by instance, so it attaches to the
    # arm's aggregate rather than to any single row — carried on the row whose instance is the
    # whole set, which does not exist. It is joined per arm instead, as a constant column.
    for key, v in doc.get("summary", {}).items():
        name = RENAME.get(key, key)
        for row in rows:
            if row["arm"] == name and row["source"] == f"benchmark/{scale}":
                row["q_arm_mean"] = v.get("mean_q")
    return rows


def _from_lkh_compare(path):
    """Rows from ``lkh_compare.json``: the sweep, the FIS local search, the 2x2, and LKH."""
    if not Path(path).exists():
        return []
    doc = json.loads(Path(path).read_text())
    rows = []
    for name, d in doc.items():
        n = d.get("n")

        def add(arm, gap, s, detail=""):
            if gap is None or s is None:
                return
            rows.append({"source": "lkh_compare", "instance": name, "n": n,
                         "arm": RENAME.get(arm, arm), "gap_pct": gap, "seconds": s,
                         "q_arm_mean": None, "detail": detail})

        for r in d.get("sweep", []):
            add(f"LK {r['cfg']}", r["gap"], r["s"])
        if d.get("fis_ls"):
            add("fis_ls", d["fis_ls"]["gap"], d["fis_ls"]["s"])
        for arm in ("iterated", "iterated_aim", "iterated_chain", "iterated_fis"):
            for r in d.get(arm, []):
                add(arm, r["gap"], r["s"], f"{r['kicks']} kicks")
        for r in d.get("lkh", []):
            add("LKH", r.get("gap"), r.get("s"), f"{r['runs']} runs")
    return rows


def collect():
    rows = []
    for scale in ("small", "large"):
        rows += _from_benchmark(paths.benchmark(scale), scale)
    rows += _from_lkh_compare(paths.LKH_COMPARE)
    return rows


def markdown(rows, limit=None):
    head = ["source", "instance", "n", "arm", "detail", "gap %", "seconds", "q (arm mean)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows[:limit]:
        q = f"{r['q_arm_mean']:.4f}" if r["q_arm_mean"] is not None else ""
        out.append(
            f"| {r['source']} | {r['instance']} | {r['n']} | {r['arm']} | {r['detail']} "
            f"| {r['gap_pct']:.3f} | {r['seconds']:.4f} | {q} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default="n", choices=("n", "arm", "gap", "s", "instance"))
    ap.add_argument("--arms", nargs="*", default=None, help="substring filter on the arm name")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="rows printed; the CSV is complete")
    ap.add_argument("--csv", default=str(paths.RESULTS / "summary.csv"))
    args = ap.parse_args()
    paths.ensure()

    rows = collect()
    if not rows:
        raise SystemExit("nothing measured yet — run run_all.py first")

    if args.instances:
        rows = [r for r in rows if r["instance"] in args.instances]
    if args.arms:
        rows = [r for r in rows if any(a.lower() in r["arm"].lower() for a in args.arms)]

    key = {"n": lambda r: (r["n"], r["arm"]), "arm": lambda r: (r["arm"], r["n"]),
           "gap": lambda r: r["gap_pct"], "s": lambda r: r["seconds"],
           "instance": lambda r: (r["instance"], r["arm"])}[args.sort]
    rows.sort(key=key)

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(markdown(rows, args.limit))
    if args.limit and len(rows) > args.limit:
        print(f"\n... {len(rows) - args.limit} more rows; all {len(rows)} are in the CSV")
    print(f"\nwrote {args.csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
