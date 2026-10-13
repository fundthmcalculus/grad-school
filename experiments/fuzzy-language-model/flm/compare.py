"""Compare arms of one sweep directory by seed-replicated final val BPC.

    .venv/bin/python -m flm.compare v4-final10 [--ref D]

Arms are job names with the trailing ``_s<seed>`` removed (``D-lr0.03_30M``, ``ref1-..._30M``).
The reference arm is the one whose name starts with ``--ref`` (default ``D``). For every other
arm it reports the mean difference (arm - ref, so positive = the arm is worse than the ref),
the standard error of that difference, and a verdict:

    se = sqrt(var_ref / n_ref + var_arm / n_arm);  "real" if |delta| > 2 * se

Diverged runs (NaN) are excluded and counted. Fewer than 2 finite seeds in either arm gives no
verdict. Writes outputs/<grid>/compare.md. This is the decision rule the plan registers for
every multi-seed comparison, so use it rather than eyeballing means.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs"


def load_arms(grid: str) -> dict[str, list[tuple[int, float, int]]]:
    arms = defaultdict(list)
    for f in sorted((OUT / grid).glob("*.json")):
        m = re.match(r"^(?P<arm>.+)_s(?P<seed>\d+)$", f.stem)
        if m:
            r = json.loads(f.read_text())
            arms[m["arm"]].append((int(m["seed"]), r["val_bpc"], r["params_total"]))
    return arms


def verdict(ref: np.ndarray, arm: np.ndarray) -> tuple[float, float | None, str]:
    delta = float(arm.mean() - ref.mean())
    if len(ref) < 2 or len(arm) < 2:
        return delta, None, "no verdict (< 2 seeds)"
    se = math.sqrt(ref.var(ddof=1) / len(ref) + arm.var(ddof=1) / len(arm))
    if abs(delta) > 2 * se:
        return delta, se, "ref better, real" if delta > 0 else "arm better, real"
    return delta, se, "within noise"


def compare(grid: str, ref: str = "D") -> str:
    arms = load_arms(grid)
    refs = [a for a in arms if a.startswith(ref)]
    if len(refs) != 1:
        raise SystemExit(
            f"--ref {ref!r} must match exactly one arm, matched {refs}: {sorted(arms)}"
        )
    rname = refs[0]
    fin = {a: np.array([b for _, b, _ in v if b == b]) for a, v in arms.items()}
    lines = [
        f"# `{grid}`: arms against `{rname}` (positive delta = the arm is worse than the reference)\n",
        "| arm | params | seeds (finite/total) | mean ± std | delta | 2·se | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in sorted(arms, key=lambda a: fin[a].mean() if len(fin[a]) else 9e9):
        f, params = fin[a], arms[a][0][2]
        n = f"{len(f)}/{len(arms[a])}"
        if a == rname:
            lines.append(
                f"| **{a}** (ref) | {params:,} | {n} | {f.mean():.4f} ± {f.std(ddof=1) if len(f) > 1 else float('nan'):.4f} | | | |"
            )
            continue
        d, se, v = verdict(fin[rname], f)
        lines.append(
            f"| {a} | {params:,} | {n} | {f.mean():.4f} ± {f.std(ddof=1) if len(f) > 1 else float('nan'):.4f} "
            f"| {d:+.4f} | {'' if se is None else f'{2 * se:.4f}'} | {v} |"
        )
    text = "\n".join(lines) + "\n"
    (OUT / grid / "compare.md").write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("grid")
    p.add_argument("--ref", default="D")
    a = p.parse_args()
    print(compare(a.grid, a.ref))


if __name__ == "__main__":
    main()
