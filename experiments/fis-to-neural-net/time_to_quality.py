"""Time-to-quality: how long each initialization takes to get *good enough*.

The question this experiment exists to answer is not "which arm ends lowest"
-- given enough epochs a randomly initialized network gets there too, and on
these datasets it often ends slightly lower. The question is how much wall-clock
time each initialization needs to reach a quality bar, with the FIS fit and the
conversion charged against the arms that used them.

So the reporting is a *time-to-target* table, at several targets, rather than a
league table of final scores. Targets are set relative to the best test RMSE any
arm achieves on that seed, so they are the same bar for every arm and are not
chosen to flatter one:

    1.50x, 1.25x, 1.10x, 1.05x the best RMSE seen on that seed

plus the FIS's own accuracy, which is the bar the conversion has to clear for
"the network starts where the FIS left off" to mean anything.

Reads `results.json` (already written by `run_experiment.py`) and needs no
re-run: every arm's per-epoch test curve, per-epoch cost, and setup cost are
recorded there.

    python experiments/fis-to-neural-net/time_to_quality.py

Writes `time_to_quality.md` next to this file.
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

TARGET_MULTIPLES = (1.50, 1.25, 1.10, 1.05)
REFERENCE_ARM = "he-all"  # the arm that got no FIS output of any kind


def seconds_to(rec, target):
    """Wall-clock seconds for `rec` to first reach `target` test RMSE, or None.

    Setup cost is included: for the hot arms that is the TRIBBLE fit plus the
    conversion. An arm that is already below the target at epoch 0 still pays
    its setup, which is the entire point of charging it.
    """
    curve = np.asarray(rec["curve_test_rmse"], dtype=float)
    hit = np.flatnonzero(curve <= target)
    if not hit.size:
        return None, None
    epoch = int(rec["curve_epochs"][hit[0]])
    return epoch, rec["setup_seconds"] + epoch * rec["seconds_per_epoch"]


def fmt(values, hits, n):
    if not hits:
        return "never"
    txt = f"{np.mean(values):.2f}"
    if len(hits) < n:
        txt += f" ({len(hits)}/{n})"
    return txt


def main() -> int:
    path = os.path.join(HERE, "results.json")
    if not os.path.exists(path):
        print(f"no {path}; run run_experiment.py first")
        return 1
    with open(path) as fh:
        blob = json.load(fh)
    results = blob["results"]
    arms = list(results[0]["arms"])

    lines = [
        "# Time to quality",
        "",
        "Seconds of wall clock for each initialization to *first* reach a target "
        "test RMSE, averaged over seeds. The TRIBBLE fit and the conversion are "
        "charged to the `hot` arms. Targets are multiples of the best RMSE any arm "
        "reached on that seed, so every arm faces the same bar.",
        "",
        f"`speedup` is {REFERENCE_ARM}'s time divided by that arm's time at the "
        "same target: >1 means faster to the same quality. Cells read `never` "
        "when the arm did not reach the target within the epoch budget; a "
        "parenthesised `(k/n)` means only k of n seeds got there and the mean "
        "covers those.",
        "",
    ]

    by_ds: dict[str, list] = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)

    for ds, rows in by_ds.items():
        lines += [f"## {ds}", ""]

        # Per-seed reference points.
        best = {
            r["seed"]: min(min(r["arms"][a]["curve_test_rmse"]) for a in arms)
            for r in rows
        }
        fis_rmse = {r["seed"]: r["fis"]["test_rmse"] for r in rows}

        targets = [
            (f"{m:.2f}x best", {s: best[s] * m for s in best}) for m in TARGET_MULTIPLES
        ]
        targets.insert(0, ("FIS parity", fis_rmse))

        header = "| arm | " + " | ".join(t[0] for t in targets) + " |"
        lines += [
            "Wall-clock seconds to target (mean over seeds):",
            "",
            header,
            "|" + "---|" * (len(targets) + 1),
        ]
        secs_by_arm: dict[str, list] = {}
        for arm in arms:
            cells = []
            for label, per_seed in targets:
                vals, hits = [], []
                for r in rows:
                    _ep, sec = seconds_to(r["arms"][arm], per_seed[r["seed"]])
                    if sec is not None:
                        vals.append(sec)
                        hits.append(r["seed"])
                cells.append(fmt(vals, hits, len(rows)))
                secs_by_arm.setdefault(arm, []).append(np.mean(vals) if hits else None)
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

        ref = secs_by_arm.get(REFERENCE_ARM, [])
        lines += [
            "",
            f"Speedup over `{REFERENCE_ARM}` at the same target:",
            "",
            header.replace("| arm |", "| arm |"),
            "|" + "---|" * (len(targets) + 1),
        ]
        for arm in arms:
            if arm == REFERENCE_ARM:
                continue
            cells = []
            for i in range(len(targets)):
                mine = secs_by_arm[arm][i]
                theirs = ref[i] if i < len(ref) else None
                if mine is None:
                    cells.append("never")
                elif theirs is None:
                    cells.append("only arm to arrive")
                elif mine <= 0:
                    cells.append("inf")
                else:
                    cells.append(f"{theirs / mine:.1f}x")
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

        lines += [
            "",
            "Epochs to the same targets (mean over seeds):",
            "",
            header,
            "|" + "---|" * (len(targets) + 1),
        ]
        for arm in arms:
            cells = []
            for _label, per_seed in targets:
                vals, hits = [], []
                for r in rows:
                    ep, _sec = seconds_to(r["arms"][arm], per_seed[r["seed"]])
                    if ep is not None:
                        vals.append(ep)
                        hits.append(r["seed"])
                cells.append(fmt(vals, hits, len(rows)) if hits else "never")
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

        setup = {
            a: np.mean([r["arms"][a]["setup_seconds"] for r in rows]) for a in arms
        }
        per_ep = {
            a: np.mean([r["arms"][a]["seconds_per_epoch"] for r in rows]) for a in arms
        }
        lines += [
            "",
            "Fixed costs: "
            + ", ".join(
                f"`{a}` setup {setup[a]:.2f}s + {1000 * per_ep[a]:.0f}ms/epoch"
                for a in arms
            ),
            "",
            f"Best RMSE reached by any arm: {np.mean(list(best.values())):.3f}; "
            f"FIS: {np.mean(list(fis_rmse.values())):.3f}.",
            "",
        ]

    out = os.path.join(HERE, "time_to_quality.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {os.path.relpath(out, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
