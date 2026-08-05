"""Figures for the fitting stage: which optimiser won, and what it did to the rules.

Panel A compares the hand-written rule bases against the fitted ones on the **validation**
split, in the frontier-relative units the main result uses: q is tour length over what the
swept baseline reaches at the same cost, so q = 1 is the frontier and lower is better. It
shows both that fitting closes most of the gap and that it does not cross the frontier.

Panel B draws the membership functions the winning run arrived at against the ones it
started from. This is the part of a fuzzy system that is supposed to stay legible, so it
is worth being able to see whether fitting kept it that way or smeared every term into
every other.

Run:  python figures_tuning.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import fis  # noqa: E402

import paths  # noqa: E402

MARKER = {"ga": "o", "pso": "s", "aco": "^"}
COLOUR = {"gaussian": "tab:blue", "triangular": "tab:red"}
TERM_NAME = ("LOW", "MED", "HIGH")
EFFORT_INPUTS = (
    "edge excess", "past failures", "turn sharpness", "progress", "edge rank", "peakedness",
)


def figure(log, tuned, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # --- panel A: what fitting did, on the split it was selected on and on test
    ax = axes[0]
    labels, hand, fitted = [], [], []
    for rec in log:
        labels.append(f"{rec['optimizer'].upper()}\n{rec['mf_kind']}\n{rec['evaluations']} evals")
        hand.append(rec["hand_valid_ratio"])
        fitted.append(rec["valid_ratio"])
    xs = np.arange(len(labels))
    w = 0.38
    ax.bar(xs - w / 2, hand, w, color="tab:gray", label="hand-written")
    ax.bar(xs + w / 2, fitted, w, color="tab:green", label="GA-fitted")
    ax.axhline(1.0, color="tab:red", lw=1.4, ls="--",
               label="the baseline frontier (q = 1)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("validation q  (tour length / frontier at equal cost)")
    ax.set_ylim(0.98, max(max(hand), 1.25) * 1.02)
    ax.set_title(
        "Fitting closes most of the validation gap\n(lower is better; q = 1 is the frontier)",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    # --- panel B: the membership functions, before and after
    ax = axes[1]
    xs = np.linspace(0.0, 1.0, fis.MF_RES + 1)
    hand = fis.EFFORT_TAB
    fitted = tuned["effort_tab"] if tuned is not None else None
    names = EFFORT_INPUTS[: hand.shape[0]]
    offsets = np.arange(len(names))[::-1]
    for i, (name, off) in enumerate(zip(names, offsets)):
        for t in range(fis.N_TERMS):
            ax.plot(
                xs,
                off + 0.9 * hand[i, t],
                color="tab:gray",
                lw=1.0,
                ls="--",
                alpha=0.8,
                label="hand-written" if (i == 0 and t == 0) else None,
            )
            if fitted is not None:
                ax.plot(
                    xs,
                    off + 0.9 * fitted[i, t],
                    color=f"C{t}",
                    lw=1.9,
                    label=TERM_NAME[t] if i == 0 else None,
                )
        ax.text(0.01, off + 0.72, name, fontsize=8, color="black")
    ax.set_yticks(offsets)
    ax.set_yticklabels([f"in {i}" for i in range(len(names))])
    ax.set_xlabel("normalised input value")
    ax.set_title(
        "EFFORT membership functions, fitted against hand-written", fontsize=10
    )
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small", choices=("small", "large"))
    ap.add_argument("--log", default=None, help="default: results/tune_<scale>.json")
    ap.add_argument("--tuned", default=None, help="default: results/tuned_<scale>.npz")
    ap.add_argument("--out", default=str(paths.FIGURES / "fis_tsp_tuning.png"))
    args = ap.parse_args()
    if args.log is None:
        args.log = str(paths.tune_log(args.scale))
    if args.tuned is None:
        args.tuned = str(paths.tuned(args.scale))
    log = json.loads(Path(args.log).read_text())
    tuned = np.load(args.tuned) if Path(args.tuned).exists() else None
    paths.ensure()
    print(f"wrote {figure(log, tuned, args.out)}")


if __name__ == "__main__":
    main()
