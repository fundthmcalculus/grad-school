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


def _h9():
    """Direct one-variable test of H9 (where does the HTSK gain live?), registered in
    RESULTS.md before running: 2x2 of exponent normalization on the recurrent mixer x on
    the TSK FFN, at each arm's tuned lr/shortconv, d=32, 10M chars. frlm-acc at seeds 0-2
    (also a first seed-noise estimate for the fuzzyfix gain); frlm-delta at seed 0."""
    try:
        arms = {a: ARMS[a] + best_tune(a) for a in ("frlm-acc", "frlm-delta")}
    except RuntimeError:
        return []
    jobs = []
    for arm, argv in arms.items():
        for seed in (0, 1, 2) if arm == "frlm-acc" else (0,):
            for mx in ("sum", "mean"):
                for ff in ("sum", "mean"):
                    jobs.append(
                        (
                            f"{arm}_mx{mx}_ff{ff}_s{seed}",
                            argv
                            + ["--d", "32", "--layers", "2", "--seed", str(seed)]
                            + [
                                "--chars",
                                "10e6",
                                "--exp-norm",
                                mx,
                                "--ffn-exp-norm",
                                ff,
                            ],
                        )
                    )
    return jobs


register("h9", _h9())


# ----------------------------------------------------------------------------- GPU offload
# Added 2026-10-09 at the user's request to speed up the run. The ten-seed headline cell
# runs as a complete, single-platform grid (seeds 0-9 at d=32) on the GPU host:
#     .venv/bin/python -m flm.sweep headline10 --device cuda --workers 8
#     .venv/bin/python -m flm.sweep headline10v2 --device cuda --workers 8
# The laptop's planned seeds-3..9 `headline`/`headline2` grids (meant to merge with CPU
# seeds 0-2) are disabled with SKIP markers, so no table cell mixes platforms. GPU seeds
# 0-2 duplicate the CPU `scaling` cells: a free CPU-vs-GPU reproducibility check.
def _headline10():
    try:
        arms = {a: v + best_tune(a) for a, v in ARMS.items()}
    except RuntimeError:
        return []
    return scaling_grid([32], [2], list(range(10)), "30e6", arms=arms)


register("headline10", _headline10())
register("headline10v2", _v2([32], list(range(10))))


def _lrwidth():
    """Per-width LR check (registered in RESULTS.md before running): every arm at d=96,
    seed 0, 30M chars, with its d=32-tuned lr divided by 3 and by 10 (shortconv unchanged).
    The undivided cell is the existing `scaling` seed-0 run."""
    try:
        arms = {a: v + best_tune(a) for a, v in {**ARMS, **ARMS_V2}.items()}
    except RuntimeError:
        return []
    jobs = []
    for arm, argv in arms.items():
        i = argv.index("--lr")
        base = float(argv[i + 1])
        for div in (3, 10):
            a2 = list(argv)
            a2[i + 1] = f"{base / div:.4g}"
            jobs.append(
                (
                    f"{arm}_lrdiv{div}_d96_L2_s0",
                    a2
                    + ["--d", "96", "--layers", "2", "--seed", "0", "--chars", "30e6"],
                )
            )
    return jobs


register("lrwidth", _lrwidth())


# ----------------------------------------------------------------------------- the ~2 BPC FRLM
# Added 2026-10-10 (user: "focus on the 2 bpc fuzzy recurrent model"). See RESULTS.md.


def _minsize():
    """Where does the best FRLM actually cross 2.0 BPC? Widths below the scaling grid,
    with DeltaNet (the best neural arm at small size) as the reference. Tuned settings as in
    `scaling`; seeds 0-2."""
    try:
        arms = {
            "frlm-delta-htsk": ARMS_V2["frlm-delta-htsk"]
            + best_tune("frlm-delta-htsk"),
            "delta-mlp": ARMS["delta-mlp"] + best_tune("delta-mlp"),
        }
    except RuntimeError:
        return []
    return scaling_grid([8, 10, 12, 14], [2], [0, 1, 2], "30e6", arms=arms)


LEAN_VARIANTS = {
    # each changes ONE thing relative to frlm-delta-htsk at d=16 (later flags override)
    "ffnonly": ["--exp-norm", "sum", "--ffn-exp-norm", "mean"],
    "wdim": ["--width-share", "dim"],
    "wrule": ["--width-share", "rule"],
    "share2": ["--layers", "2", "--n-unique", "1"],
    "share3": ["--layers", "3", "--n-unique", "1"],
    "share4": ["--layers", "4", "--n-unique", "1"],
}


def _lean():
    """Design-iteration screen at d=16 (the ~2 BPC regime), seeds 0-1, at frlm-delta-htsk's
    tuned lr/shortconv. The reference is the existing scaling2 frlm-delta-htsk_d16 cells.
    """
    try:
        base = ARMS_V2["frlm-delta-htsk"] + best_tune("frlm-delta-htsk")
    except RuntimeError:
        return []
    common = ["--d", "16", "--layers", "2", "--chars", "30e6"]
    return [
        (f"frlm-delta-htsk-{v}_d16_L2_s{s}", base + common + ["--seed", str(s)] + extra)
        for v, extra in LEAN_VARIANTS.items()
        for s in (0, 1)
    ]


register("minsize", _minsize())
register("lean", _lean())
