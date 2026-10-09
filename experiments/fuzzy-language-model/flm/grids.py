"""Sweep definitions. Importing this module registers them with ``flm.sweep``.

tune     -- equal-budget hyperparameter search, identical for every arm:
            lr in {1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1} x shortconv in {0, 4}, d=32, L=2, seed 0, 10M chars.
scaling  -- the main result: every arm at widths 16..96 (≈5K–150K params; 128 dropped for CPU time), L=2, seeds 0-2, 30M chars,
            each arm at its own best `tune` setting (chosen by tune's val BPC alone).
"""

from __future__ import annotations

import json
from pathlib import Path

from flm.sweep import ARMS, GRIDS, OUT, register, scaling_grid

TUNE_LRS = [
    "1e-3",
    "3e-3",
    "1e-2",
    "3e-2",
    "6e-2",
    "1e-1",
]  # grid grew twice; each extension applied to every arm (RESULTS.md amendments)
TUNE_CONV = ["0", "4"]

tune = []
for arm, argv in ARMS.items():
    for lr in TUNE_LRS:
        for sc in TUNE_CONV:
            tune.append(
                (
                    f"{arm}_lr{lr}_sc{sc}",
                    argv
                    + [
                        "--d",
                        "32",
                        "--layers",
                        "2",
                        "--seed",
                        "0",
                        "--chars",
                        "10e6",
                        "--lr",
                        lr,
                        "--shortconv",
                        sc,
                    ],
                )
            )
register("tune", tune)


def best_tune(arm: str) -> list[str]:
    """argv for arm's best tune cell; raises if the tune sweep is incomplete for that arm."""
    best, cells = None, 0
    for lr in TUNE_LRS:
        for sc in TUNE_CONV:
            f = OUT / "tune" / f"{arm}_lr{lr}_sc{sc}.json"
            if not f.exists():
                continue
            cells += 1
            b = json.loads(f.read_text())["val_bpc"]
            if b == b and (best is None or b < best[0]):
                best = (b, lr, sc)
    if cells < len(TUNE_LRS) * len(TUNE_CONV) or best is None:
        raise RuntimeError(f"tune sweep incomplete for {arm} ({cells} cells)")
    return ["--lr", best[1], "--shortconv", best[2]]


def _scaling():
    try:
        arms = {a: v + best_tune(a) for a, v in ARMS.items()}
    except RuntimeError:
        return []
    return scaling_grid([16, 24, 32, 48, 64, 96], [2], [0, 1, 2], "30e6", arms=arms)


register("scaling", _scaling())


def _ablate():
    """One-variable controls, each paired with an existing `scaling` cell (same d, L, seed, chars):

    gauss-mlp    = softmax-mlp with the mixer swapped for Gaussian-TSK   (isolates the mixer, H2)
    softmax-tsk  = softmax-mlp with the MLP swapped for the TSK FFN      (isolates the FFN, H3)
    <arm>+datainit = a TSK arm with data-driven rule centers            (isolates rule init, H4)
    The swaps reuse softmax-mlp's tuned lr/shortconv so that exactly one thing changes.
    """
    try:
        sm = best_tune("softmax-mlp")
        tsk_arms = {
            a: ARMS[a] + best_tune(a) for a in ("flm", "frlm-acc", "frlm-delta")
        }
    except RuntimeError:
        return []
    arms = {
        "gauss-mlp": ["--mixer", "gauss", "--ffn", "mlp"] + sm,
        "softmax-tsk": ["--mixer", "softmax", "--ffn", "tsk"] + sm,
        "softmax-tsk+datainit": [
            "--mixer",
            "softmax",
            "--ffn",
            "tsk",
            "--rule-init",
            "data",
        ]
        + sm,
    }
    arms.update(
        {f"{a}+datainit": v + ["--rule-init", "data"] for a, v in tsk_arms.items()}
    )
    return scaling_grid([32, 64], [2], [0, 1, 2], "30e6", arms=arms)


register("ablate", _ablate())


def _headline():
    """Seeds 3-9 at d=32 (~20K params, the MacroStories size) for every scaling arm, so the
    headline width carries the ten-seed protocol when merged with `scaling` seeds 0-2.
    """
    try:
        arms = {a: v + best_tune(a) for a, v in ARMS.items()}
    except RuntimeError:
        return []
    return scaling_grid([32], [2], list(range(3, 10)), "30e6", arms=arms)


register("headline", _headline())


def _fuzzyfix():
    """Design iteration on the fuzzy layers' rule saturation (Cui, Wu & Xu 2021), NOT part of
    the registered head-to-head. Each FLM/FRLM arm at its tuned lr/shortconv, d=32, seed 0,
    10M chars (the tune budget, so the exp_norm=sum/random-init cell is the existing tune cell):
    exp_norm in {sum, sqrt, mean} x rule-init in {random, data}, minus that existing cell.
    """
    try:
        arms = {a: ARMS[a] + best_tune(a) for a in ("flm", "frlm-acc", "frlm-delta")}
    except RuntimeError:
        return []
    jobs = []
    for arm, argv in arms.items():
        for en in ("sum", "sqrt", "mean"):
            for ri in ("random", "data"):
                if en == "sum" and ri == "random":
                    continue
                jobs.append(
                    (
                        f"{arm}_en{en}_ri{ri}",
                        argv
                        + [
                            "--d",
                            "32",
                            "--layers",
                            "2",
                            "--seed",
                            "0",
                            "--chars",
                            "10e6",
                            "--exp-norm",
                            en,
                            "--rule-init",
                            ri,
                        ],
                    )
                )
    return jobs


register("fuzzyfix", _fuzzyfix())


# ----------------------------------------------------------------------------- FRLM v2 (HTSK)
# Added 2026-10-09 after `fuzzyfix` (see RESULTS.md). Kept OUT of ARMS on purpose, so that
# the already-queued `scaling` / `ablate` / `headline` grids are unchanged and cannot go
# silently empty while these arms are untuned. They get the identical tune grid.
ARMS_V2 = {
    "frlm-acc-htsk": [
        "--mixer",
        "fuzzy",
        "--ffn",
        "tsk",
        "--decay",
        "data",
        "--exp-norm",
        "mean",
    ],
    "frlm-delta-htsk": [
        "--mixer",
        "fuzzydelta",
        "--ffn",
        "tsk",
        "--decay",
        "data",
        "--exp-norm",
        "mean",
    ],
}

register(
    "tune",  # same sweep dir as the original tune, so best_tune() reads these cells too
    GRIDS["tune"]
    + [
        (
            f"{arm}_lr{lr}_sc{sc}",
            argv
            + [
                "--d",
                "32",
                "--layers",
                "2",
                "--seed",
                "0",
                "--chars",
                "10e6",
                "--lr",
                lr,
                "--shortconv",
                sc,
            ],
        )
        for arm, argv in ARMS_V2.items()
        for lr in TUNE_LRS
        for sc in TUNE_CONV
    ],
)


def _v2(widths, seeds):
    try:
        arms = {a: v + best_tune(a) for a, v in ARMS_V2.items()}
    except RuntimeError:
        return []
    return scaling_grid(widths, [2], seeds, "30e6", arms=arms)


register("scaling2", _v2([16, 24, 32, 48, 64, 96], [0, 1, 2]))
register("headline2", _v2([32], list(range(3, 10))))
