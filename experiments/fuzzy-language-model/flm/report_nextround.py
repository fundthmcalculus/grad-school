"""Paired report for a screen (PLAN_NEXTROUND.md): per-arm mean bits per character, the change against a reference paired by seed, and the decision rule (a gain of at least 0.005 on at least four of five seeds).

``python -m flm.report_nextround outputs/v64-groups --ref outputs/v53-fuzzylr/lr6_wu600 [--arms a b ...] [--min-gain 0.005] [--min-seeds 4] [--out report/nextround_v64.md]``
Cells are ``<dir>/<arm>_s<seed>.json``; the reference is ``<prefix>_s<seed>.json``.  Prints a table, writes it (markdown and json) when ``--out`` is given.  A partial seed set is reported as such (the n column), never as a pass.
"""

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path


def load(prefix_dir, arm_re):
    cells = {}
    for f in Path(prefix_dir).glob("*_s[0-9].json"):
        m = re.match(r"(.+)_s(\d+)\.json$", f.name)
        if not m or not arm_re(m.group(1)):
            continue
        r = json.loads(f.read_text())
        if r.get("val_bpc") != r.get("val_bpc") or r.get("diverged_step"):
            continue
        cells.setdefault(m.group(1), {})[int(m.group(2))] = (
            r["val_bpc"],
            r["params_total"],
        )
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", required=True, help="path prefix of the reference cells")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--min-seeds", type=int, default=4)
    ap.add_argument("--of", type=int, default=5, help="seeds in a complete screen")
    ap.add_argument("--out")
    a = ap.parse_args()
    ref = {}
    for f in Path(a.ref).parent.glob(Path(a.ref).name + "_s[0-9].json"):
        ref[int(re.search(r"_s(\d+)\.json$", f.name).group(1))] = json.loads(
            f.read_text()
        )["val_bpc"]
    cells = load(a.dir, lambda n: not a.arms or n in a.arms)
    rows = []
    for arm in sorted(cells):
        d = cells[arm]
        seeds = sorted(s for s in d if s in ref)
        if not seeds:
            continue
        dv = [d[s][0] - ref[s] for s in seeds]
        gain = sum(x <= -a.min_gain for x in dv)
        complete = len(seeds) >= a.of
        rows.append(
            dict(
                arm=arm,
                n=len(seeds),
                params=d[seeds[0]][1],
                bpc=round(st.mean(d[s][0] for s in seeds), 4),
                delta=round(st.mean(dv), 4),
                se=round(st.stdev(dv) / math.sqrt(len(dv)), 4) if len(dv) > 1 else None,
                per_seed=[round(x, 4) for x in dv],
                gain_seeds=gain,
                worse_seeds=sum(x > 0 for x in dv),
                verdict=(
                    ("PASS" if gain >= a.min_seeds else "no")
                    if complete
                    else f"partial ({len(seeds)}/{a.of})"
                ),
            )
        )
    lines = [
        f"Reference `{a.ref}` ({len(ref)} seeds, mean {st.mean(ref.values()):.4f}); gain = at least {a.min_gain} BPC better on at least {a.min_seeds} of {a.of} seeds.",
        "",
        "| arm | seeds | params | BPC | change vs ref | se | per seed | gain seeds | worse seeds | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['n']} | {r['params']:,} | {r['bpc']:.4f} | {r['delta']:+.4f} | {r['se'] if r['se'] is not None else ''} | {' '.join(f'{x:+.4f}' for x in r['per_seed'])} | {r['gain_seeds']} | {r['worse_seeds']} | {r['verdict']} |"
        )
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
