"""Report a fast architecture screen: final BPC per seed, early-curve speed, a noise-aware verdict.

    .venv/bin/python -m flm.screen_report screen [screen2 ...] [--base A-base] [--name merged]

Reads outputs/<screen>/<variant>_d16_s<seed>.json, merging every named sweep dir. Per variant:
params, starting BPC (step 0), steps to reach 3.0 / 2.5 / 2.2 BPC, the mean and per-seed final
BPC, and the difference of means against the base. The verdict compares that difference with
two standard errors, using the base's own seed-to-seed std (pooled over its seeds) as the noise:

    se = sd_base * sqrt(1/n_variant + 1/n_base),   "real" if |delta| > 2 * se

With fewer than 2 base seeds there is no noise estimate and no verdict is given.
Writes outputs/<name>/summary.md.
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
NAME = re.compile(r"^(?P<variant>.+?)(?:_d\d+)?(?:_L\d+)?_s(?P<seed>\d+)$")
THRESHOLDS = (3.0, 2.5, 2.2)


def steps_to(evals, level):
    """First evaluated step at which BPC <= level (None if never)."""
    for step, bpc in evals:
        if bpc <= level:
            return step
    return None


def load(screens):
    runs = defaultdict(dict)
    for screen in screens:
        for f in sorted((OUT / screen).glob("*.json")):
            m = NAME.match(f.stem)
            if m:
                runs[m["variant"]][int(m["seed"])] = json.loads(f.read_text())
    return runs


def report(screens, base: str, name: str) -> str:
    runs = load(screens)
    finals = {
        v: np.array([runs[v][s]["val_bpc"] for s in sorted(runs[v])]) for v in runs
    }
    b = finals.get(base, np.array([]))
    sd = float(b.std(ddof=1)) if len(b) > 1 else None
    lines = [
        f"# Screen {' + '.join(screens)}: final val BPC and early-curve speed\n",
        f"Base `{base}`: {len(b)} seeds, mean {b.mean():.4f}, std {sd if sd is None else round(sd, 4)}. "
        "Δ is the difference of means against the base; `real` means |Δ| > 2 standard errors "
        "(se = base std · √(1/n_variant + 1/n_base)). Nothing here is citable.\n",
        "| variant | params | seeds | start BPC | to 3.0 | to 2.5 | to 2.2 | final BPC (mean ± std) | Δ vs base | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in sorted(runs):
        seeds = runs[v]
        r = seeds[min(seeds)]
        ev = [tuple(e) for e in r["evals"]]
        start = f"{ev[0][1]:.3f}" if ev and ev[0][0] == 0 else "-"
        cells = [
            str(steps_to(ev, t)) if steps_to(ev, t) is not None else "never"
            for t in THRESHOLDS
        ]
        f = finals[v]
        spread = f"{f.std(ddof=1):.3f}" if len(f) > 1 else "n/a"
        delta, verdict = "", ""
        if v != base and len(b):
            d = float(f.mean() - b.mean())
            delta = f"{d:+.4f}"
            if sd is None:
                verdict = "no noise estimate"
            else:
                se = sd * math.sqrt(1 / len(f) + 1 / len(b))
                verdict = (
                    ("real, better" if d < 0 else "real, worse")
                    if abs(d) > 2 * se
                    else f"within noise (2se = {2 * se:.3f})"
                )
        lines.append(
            f"| {v} | {r['params_total']:,} | {len(f)} | {start} | {cells[0]} | {cells[1]} | {cells[2]} "
            f"| {f.mean():.4f} ± {spread} | {delta} | {verdict} |"
        )
    info = [(v, runs[v][min(runs[v])].get("ngram_init")) for v in sorted(runs)]
    info = [(v, i) for v, i in info if i]
    if info:
        lines += [
            "",
            "## Init diagnostics (held-out calibration windows, lowest seed)",
            "",
        ]
        for v, i in info:
            lines.append(
                f"* **{v}**: " + ", ".join(f"{k}={val}" for k, val in i.items())
            )
    text = "\n".join(lines) + "\n"
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "summary.md").write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("screens", nargs="+")
    p.add_argument("--base", default="A-base")
    p.add_argument(
        "--name",
        default=None,
        help="output dir for the summary (default: first screen)",
    )
    a = p.parse_args()
    print(report(a.screens, a.base, a.name or a.screens[0]))


if __name__ == "__main__":
    main()
