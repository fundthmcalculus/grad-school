"""Figures for the fitting stage: which optimiser won, and what it did to the rules.

Panel A places every (optimiser, membership-function form) pair on the same
cost-versus-quality plane the main result uses, measured on the **validation** split —
the one the search never optimised against. The hand-written rule base and the baseline
LK are marked, so the panel answers two questions at once: did fitting help, and did any
optimiser beat the others by more than the spread between repeats.

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

HERE = Path(__file__).resolve().parent

MARKER = {"ga": "o", "pso": "s", "aco": "^"}
COLOUR = {"gaussian": "tab:blue", "triangular": "tab:red"}
TERM_NAME = ("LOW", "MED", "HIGH")
EFFORT_INPUTS = ("edge excess", "past failures", "turn sharpness", "progress")


def figure(log, tuned, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # --- panel A: optimiser comparison on the validation split
    ax = axes[0]
    base_gap = log[0]["valid_base_gap"]
    ax.axhline(
        base_gap,
        color="tab:gray",
        ls="--",
        lw=1.5,
        label=f"baseline LK ({base_gap:.2f}%, cost 1.0x)",
    )
    ax.axvline(1.0, color="tab:gray", ls=":", lw=1.0)
    seen_hand = False
    for rec in log:
        ax.scatter(
            rec["valid_cost_ratio"],
            rec["valid_gap"],
            marker=MARKER.get(rec["optimizer"], "o"),
            color=COLOUR.get(rec["mf_kind"], "black"),
            s=95,
            zorder=4,
            label=f"{rec['optimizer'].upper()} / {rec['mf_kind']}",
        )
        if not seen_hand:
            ax.scatter(
                rec["hand_valid_cost_ratio"],
                rec["hand_valid_gap"],
                marker="X",
                color="black",
                s=110,
                zorder=5,
                label="hand-written rules",
            )
            seen_hand = True
    ax.set_xlabel("predicted cost, relative to the baseline LK")
    ax.set_ylabel("mean % over optimum, validation split")
    ax.set_title(
        "Fitted rule bases on unseen instances\n(down and left is better)", fontsize=10
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    # --- panel B: the membership functions, before and after
    ax = axes[1]
    xs = np.linspace(0.0, 1.0, fis.MF_RES + 1)
    hand = fis.EFFORT_TAB
    fitted = tuned["effort_tab"] if tuned is not None else None
    offsets = np.arange(len(EFFORT_INPUTS))[::-1]
    for i, (name, off) in enumerate(zip(EFFORT_INPUTS, offsets)):
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
    ax.set_yticklabels([f"input {i}" for i in range(len(EFFORT_INPUTS))])
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
    ap.add_argument("--log", default=str(HERE / "tune_opt_log.json"))
    ap.add_argument("--tuned", default=str(HERE / "tuned_opt.npz"))
    ap.add_argument("--out", default=str(HERE / "figures" / "fis_tsp_tuning.png"))
    args = ap.parse_args()
    log = json.loads(Path(args.log).read_text())
    tuned = np.load(args.tuned) if Path(args.tuned).exists() else None
    Path(args.out).parent.mkdir(exist_ok=True)
    print(f"wrote {figure(log, tuned, args.out)}")


if __name__ == "__main__":
    main()
