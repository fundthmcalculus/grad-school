"""Goal track: make the fuzzy LM as capable as possible under 100K total parameters.

Screens run on CUDA at d=64, L=2, seed 0, 10M chars, shortconv 4. They are design
iteration (one seed): a variant earns a place only after a multi-seed confirmation.
"""

from __future__ import annotations

from flm.sweep import GRIDS, OUT, register

GPU = ["--device", "cuda"]


def cell(name, argv, d=64, lr="3e-2", chars="10e6", seed=0, L=2, sc="4"):
    return (
        name,
        argv
        + ["--d", str(d), "--layers", str(L), "--seed", str(seed), "--chars", chars]
        + ["--lr", lr, "--shortconv", sc]
        + GPU,
    )


# which sublayer carries the deficit? every mixer x FFN pairing at d=64
_MIX = {
    "softmax": ["--mixer", "softmax"],
    "gauss": ["--mixer", "gauss"],
    "linear": ["--mixer", "linear"],
    "fuzzydelta": ["--mixer", "fuzzydelta", "--decay", "data"],
    "fuzzy": ["--mixer", "fuzzy", "--decay", "data"],
}
_FFN = {"mlp": ["--ffn", "mlp"], "tsk": ["--ffn", "tsk"]}
register(
    "v3-attrib",
    [
        cell(f"{m}-{f}_lr{lr}", _MIX[m] + _FFN[f], lr=lr)
        for m in _MIX
        for f in _FFN
        for lr in ("1e-2", "3e-2")
    ],
)


# FFN redesign: fuzzy mixer (FRLM, accumulating) at d=64 with TSK-FFN variants.
# name -> (ffn argv). ffn-mult sets rules R = mult*d; tsk-dim is the antecedent dim.
_FUZ = ["--mixer", "fuzzy", "--decay", "data", "--ffn", "tsk"]
FFN_VARIANTS = {
    "z0": [],  # original: zero-order, per-dim widths, softmax, R=2d, da=d
    "z0-rw": ["--ffn-width", "rule"],
    "z0-rw-m4-da16": ["--ffn-width", "rule", "--ffn-mult", "4", "--tsk-dim", "16"],
    "z0-rw-htsk": ["--ffn-width", "rule", "--ffn-exp-norm", "mean"],
    "z0-raw": ["--ffn-width", "rule", "--ffn-norm", "none"],
    "o1": ["--ffn-order", "1", "--ffn-width", "rule", "--ffn-mult", "2"],
    "o1-htsk": ["--ffn-order", "1", "--ffn-width", "rule", "--ffn-exp-norm", "mean"],
    "o1-raw": ["--ffn-order", "1", "--ffn-width", "rule", "--ffn-norm", "none"],
    "o1-da16": ["--ffn-order", "1", "--ffn-width", "rule", "--tsk-dim", "16"],
    "o1-da16-htsk": [
        "--ffn-order",
        "1",
        "--ffn-width",
        "rule",
        "--tsk-dim",
        "16",
        "--ffn-exp-norm",
        "mean",
    ],
    "o1-da16-raw": [
        "--ffn-order",
        "1",
        "--ffn-width",
        "rule",
        "--tsk-dim",
        "16",
        "--ffn-norm",
        "none",
    ],
    "o1-da16-m1": [
        "--ffn-order",
        "1",
        "--ffn-width",
        "rule",
        "--tsk-dim",
        "16",
        "--ffn-mult",
        "1",
    ],
}
register(
    "v3-ffn",
    [
        cell(f"{k}_lr{lr}", _FUZ + v, lr=lr)
        for k, v in FFN_VARIANTS.items()
        for lr in ("1e-2", "3e-2")
    ],
)


# round 2: refine around o1-da16-raw (first-order, raw RBF firing, tied rule widths)
_O1R = ["--ffn-order", "1", "--ffn-norm", "none", "--ffn-width", "rule"]
R2 = {
    "o1r-da8": _O1R + ["--tsk-dim", "8"],
    "o1r-da16": _O1R + ["--tsk-dim", "16"],
    "o1r-da32": _O1R + ["--tsk-dim", "32"],
    "o1r-da16-m3": _O1R + ["--tsk-dim", "16", "--ffn-mult", "3"],
    "o1r-da16-m1.5": _O1R + ["--tsk-dim", "16", "--ffn-mult", "1.5"],
    "o1r-da16-htsk": _O1R + ["--tsk-dim", "16", "--ffn-exp-norm", "mean"],
    "o1r-da16-wdim": ["--ffn-order", "1", "--ffn-norm", "none", "--tsk-dim", "16"],
}
jobs = [cell(f"{k}_lr3e-2", _FUZ + v) for k, v in R2.items()]
jobs += [cell(f"{k}_lr6e-2", _FUZ + R2[k], lr="6e-2") for k in ("o1r-da16", "o1r-da8")]
jobs += [cell("o1r-da16_lr3e-2_s1", _FUZ + R2["o1r-da16"], seed=1)]
register("v3-ffn2", jobs)


# round 3: 3 seeds on the round-2 leaders, plus the fuzzy-mixer + MLP reference
_W = ["--tsk-dim", "16"]
R3 = {
    "o1r-da16": _O1R + _W,
    "o1r-da16-htsk": _O1R + _W + ["--ffn-exp-norm", "mean"],
    "o1r-da16-htsk-wdim": [
        "--ffn-order",
        "1",
        "--ffn-norm",
        "none",
        "--ffn-exp-norm",
        "mean",
    ]
    + _W,
    "o1r-da8-htsk": _O1R + ["--tsk-dim", "8", "--ffn-exp-norm", "mean"],
    "o1r-da16-htsk-m2.5": _O1R + _W + ["--ffn-exp-norm", "mean", "--ffn-mult", "2.5"],
}
jobs = [cell(f"{k}_s{s}", _FUZ + v, seed=s) for k, v in R3.items() for s in (0, 1, 2)]
jobs += [
    cell(
        f"fuzzy-mlp_s{s}",
        ["--mixer", "fuzzy", "--decay", "data", "--ffn", "mlp"],
        seed=s,
    )
    for s in (0, 1, 2)
]
register("v3-ffn3", jobs)


# round 4: with the new FFN fixed, vary the mixer / rules / heads / depth (2 seeds).
FFN_STAR = [
    "--ffn",
    "tsk",
    "--ffn-order",
    "1",
    "--ffn-norm",
    "none",
    "--ffn-exp-norm",
    "mean",
    "--tsk-dim",
    "16",
]
_ACC = ["--mixer", "fuzzy", "--decay", "data"]
_DEL = ["--mixer", "fuzzydelta", "--decay", "data"]
R4 = {  # name -> (argv, cell kwargs)
    "acc": (_ACC, {}),
    "acc-mean": (_ACC + ["--exp-norm", "mean", "--ffn-exp-norm", "mean"], {}),
    "acc-r8": (_ACC + ["--rules", "8"], {}),
    "acc-r32": (_ACC + ["--rules", "32"], {}),
    "acc-h4": (_ACC + ["--heads", "4"], {}),
    "acc-h1": (_ACC + ["--heads", "1"], {}),
    "acc-fixeddecay": (["--mixer", "fuzzy", "--decay", "fixed"], {}),
    "delta": (_DEL, {}),
    "delta-mean": (_DEL + ["--exp-norm", "mean", "--ffn-exp-norm", "mean"], {}),
    "gauss": (["--mixer", "gauss"], {}),
    "acc-L3d48": (_ACC, {"d": 48, "L": 3}),
    "acc-L1d80": (_ACC, {"d": 80, "L": 1}),
    "acc-L4d40": (_ACC, {"d": 40, "L": 4}),
}
jobs = []
for k, (v, kw) in R4.items():
    for s in (0, 1):
        jobs.append(cell(f"{k}_s{s}", v + FFN_STAR, seed=s, **kw))
register("v3-mix", jobs)


# round 5: more heads, combined with HTSK mixer / delta form / depth
_M = ["--exp-norm", "mean", "--ffn-exp-norm", "mean"]
R5 = {
    "acc-h8": (_ACC + ["--heads", "8"], {}),
    "acc-mean-h4": (_ACC + _M + ["--heads", "4"], {}),
    "acc-mean-h8": (_ACC + _M + ["--heads", "8"], {}),
    "acc-h4-r32": (_ACC + ["--heads", "4", "--rules", "32"], {}),
    "delta-mean-h4": (_DEL + _M + ["--heads", "4"], {}),
    "delta-mean-h8": (_DEL + _M + ["--heads", "8"], {}),
    "acc-h4-L3d48": (_ACC + ["--heads", "4"], {"d": 48, "L": 3}),
    "acc-h4-d72": (_ACC + ["--heads", "4"], {"d": 72}),
}
jobs = []
for k, (v, kw) in R5.items():
    for s in (0, 1):
        jobs.append(cell(f"{k}_s{s}", v + FFN_STAR, seed=s, **kw))
register("v3-heads", jobs)


# round 6: heads to 16, spend the remaining budget (rules, FFN width), delta-mean base
_DM = _DEL + _M
R6 = {
    "dm-h8-r32": (_DM + ["--heads", "8", "--rules", "32"], {}),
    "dm-h8-r8": (_DM + ["--heads", "8", "--rules", "8"], {}),
    "dm-h8-m2.5": (_DM + ["--heads", "8", "--ffn-mult", "2.5"], {}),
    "dm-h8-da24": (_DM + ["--heads", "8", "--tsk-dim", "24"], {}),
    "dm-h8-sc8": (_DM + ["--heads", "8"], {"sc": "8"}),
    "dm-h8-d68": (_DM + ["--heads", "8"], {"d": 68}),
}
jobs = []
for k, (v, kw) in R6.items():
    sc = kw.pop("sc", None)
    for s in (0, 1):
        c = cell(f"{k}_s{s}", v + FFN_STAR, seed=s, **kw)
        if sc:  # later --shortconv overrides the earlier one in argparse
            c = (c[0], c[1] + ["--shortconv", sc])
        jobs.append(c)
register("v3-spend", jobs)


# round 7: fill the budget to just under 100K (verified by flm param count), delta-mean h8 base
_B = _DM + ["--heads", "8"]
R7 = {  # name -> (argv, kwargs); sc appended last so it overrides cell()'s default of 4
    "A-sc8-m2.4": (_B + ["--ffn-mult", "2.4"], {"sc": "8"}),
    "B-sc16-m2.25": (_B + ["--ffn-mult", "2.25"], {"sc": "16"}),
    "C-sc8-r24-m2.25": (_B + ["--rules", "24", "--ffn-mult", "2.25"], {"sc": "8"}),
    "D-d72-m1.5-sc8": (_B + ["--ffn-mult", "1.5"], {"d": 72, "sc": "8"}),
    "A-lr2e-2": (_B + ["--ffn-mult", "2.4"], {"lr": "2e-2", "sc": "8"}),
    "A-lr4e-2": (_B + ["--ffn-mult", "2.4"], {"lr": "4e-2", "sc": "8"}),
}
jobs = []
for k, (v, kw) in R7.items():
    for s in (0, 1):
        c = cell(f"{k}_s{s}", v + FFN_STAR, seed=s, **kw)
        jobs.append(c)
register("v3-fill", jobs)


# final: the chosen config "D" (d=72, 8 heads, 16 rules/head, delta-form HTSK mixer, first-order
# raw-firing HTSK TSK FFN with 16-dim antecedents, 1.5x rules, shortconv 8) = 97,520 params.
D = _B + ["--ffn-mult", "1.5"] + FFN_STAR[:]  # FFN_STAR carries --ffn tsk etc.
# neural reference at the same long budget: delta-mlp d=64 with its CPU-tune pick (lr 6e-2, sc 4)
REF = ["--mixer", "delta", "--decay", "data", "--ffn", "mlp"]
jobs = [cell(f"D_30M_s{s}", D, d=72, sc="8", chars="30e6", seed=s) for s in (0, 1, 2)]
jobs += [cell("D_100M_s0", D, d=72, sc="8", chars="100e6")]
jobs += [cell("delta-mlp-d64_100M_s0", REF, d=64, sc="4", lr="6e-2", chars="100e6")]
register("v3-final", jobs)


# size-matched neural reference for the final: delta-mlp at ~91-96K params, 30M chars, 3 seeds,
# heads in {2, 4, 8} (the fuzzy arm was tuned over heads; the neural arm gets the same freedom),
# lr 3e-2 and sc 4 (its tune pick at d=32 was lr 6e-2; 3e-2 is the next grid point and the
# fuzzy arm's lr). Not re-tuned further: disclosed in RESULTS.md.
jobs = []
for d, h in ((74, 2), (72, 4), (72, 8)):
    for s in (0, 1, 2):
        jobs.append(
            cell(
                f"delta-mlp-d{d}-h{h}_30M_s{s}",
                REF + ["--heads", str(h)],
                d=d,
                sc="4",
                chars="30e6",
                seed=s,
            )
        )
register("v3-ref", jobs)


# LR study (registered in RESULTS.md 2026-10-10 "goal track: learning rate"):
# the chosen fuzzy config D and the size-matched neural reference over lr, 30M chars, seeds 0-1.
# lr 3e-2 already exists for both (D_30M_s*, delta-mlp-d72-h8_30M_s*).
LRS = ["1e-2", "2e-2", "4e-2", "6e-2"]
jobs = []
for lr in LRS:
    for s in (0, 1):
        jobs.append(
            cell(f"D_lr{lr}_30M_s{s}", D, d=72, sc="8", chars="30e6", seed=s, lr=lr)
        )
        jobs.append(
            cell(
                f"delta-mlp-d72-h8_lr{lr}_30M_s{s}",
                REF + ["--heads", "8"],
                d=72,
                sc="4",
                chars="30e6",
                seed=s,
                lr=lr,
            )
        )
register("v3-lr", jobs)


# ---------------------------------------------------------------- bigger models + post-training
# Registered in RESULTS.md ("bigger fuzzy models and post-training"). Bases are saved (.pt) so the
# post-training grid can resume from them.
jobs = []
for d in (72, 80, 88, 96):
    for s in (0, 1):
        name, argv = cell(f"D-d{d}_30M_s{s}", D, d=d, sc="8", chars="30e6", seed=s)
        jobs.append((name, argv + ["--save-model"]))
name, argv = cell(
    "teacher-delta-d128_60M_s0",
    REF + ["--heads", "8"],
    d=128,
    sc="4",
    chars="60e6",
    seed=0,
)
jobs.append((name, argv + ["--save-model"]))
register("v3-big", jobs[-1:] + jobs[:-1])  # teacher first: it gates the KD arms

_POST = ["--lr", "3e-3", "--warmup", "50", "--chars", "10e6", "--eval-at-start"]
_TEACH = str(OUT / "v3-big" / "teacher-delta-d128_60M_s0.pt")
jobs = []
for d, seeds in ((72, (0, 1)), (96, (0,))):
    for s in seeds:
        base = str(OUT / "v3-big" / f"D-d{d}_30M_s{s}.pt")
        common = ["--init-from", base, "--seed", str(100 + s)]
        arms = {
            "cont": [],
            "ema": ["--ema-decay", "0.999"],
            "kd": ["--teacher", _TEACH, "--kd-weight", "0.5"],
            "kd-ema": [
                "--teacher",
                _TEACH,
                "--kd-weight",
                "0.5",
                "--ema-decay",
                "0.999",
            ],
        }
        for k, extra in arms.items():
            if d == 96 and k.startswith("kd"):
                continue  # the teacher comparison is run at the 97.5K size only
            c = cell(f"D-d{d}_{k}_s{s}", D, d=d, sc="8", chars="10e6", seed=s)
            jobs.append((c[0], c[1] + common + _POST + extra))
register("v3-post", jobs)

# same jobs as v3-big, large widths first, writing into v3-big: a second worker group that
# meets the first in the middle (a job whose record exists is skipped)
GRIDS_BIG = [j for j in GRIDS["v3-big"] if not j[0].startswith("teacher")]
register("v3-big-rev", GRIDS_BIG[::-1])


# v5-bpe-probe: subword vs character D at similar parameter counts, 10M chars, seed 0 (RESULTS.md H17)
jobs = []
for d, tok in ((72, "bpe512"), (80, "bpe512"), (96, "")):
    name = f"D-d{d}-{tok or 'char'}_10M_s0"
    c = cell(name, D, d=d, sc="8", chars="10e6", seed=0)
    jobs.append((name, c[1] + (["--tok", tok] if tok else [])))
register("v5-bpe-probe", jobs)


# v5-bpe30 (H18): subword D d=80 at 30M chars, 2 seeds, saved for sampling
jobs = []
for s in (0, 1):
    name = f"D-d80-bpe512_30M_s{s}"
    c = cell(name, D, d=80, sc="8", chars="30e6", seed=s)
    jobs.append((name, c[1] + ["--tok", "bpe512", "--save-model"]))
register("v5-bpe30", jobs)


# v5-sub100k (H19/H20): subword D under 100K parameters, 30M chars, 2 seeds, saved
jobs = []
for tok, d in (("bpe384", 64), ("bpe512", 56)):
    for s in (0, 1):
        name = f"D-d{d}-{tok}_30M_s{s}"
        c = cell(name, D, d=d, sc="8", chars="30e6", seed=s)
        jobs.append((name, c[1] + ["--tok", tok, "--save-model"]))
register("v5-sub100k", jobs)


# v5-spend (H21): subword D variants under 100K, 10M chars, 2 seeds, warm-up 100
_SP = {
    "a-512-d56": ("bpe512", 56, []),
    "b-512-d56-m2.0": ("bpe512", 56, ["--ffn-mult", "2.0"]),
    "c-512-d56-da24": ("bpe512", 56, ["--tsk-dim", "24"]),
    "d-768-d48": ("bpe768", 48, []),
    "e-1024-d48": ("bpe1024", 48, []),
}
jobs = []
for k, (tok, d, extra) in _SP.items():
    for s in (0, 1):
        c = cell(f"{k}_10M_s{s}", D, d=d, sc="8", chars="10e6", seed=s)
        jobs.append((c[0], c[1] + extra + ["--tok", tok, "--warmup", "100"]))
register("v5-spend", jobs)


# v5-final100k (H22/H23): BPE1024 D d=48 (93.3K), the best model under 100K
jobs = []
for s in (0, 1):
    n_ = f"D-d48-bpe1024_30M_s{s}"
    c = cell(n_, D, d=48, sc="8", chars="30e6", seed=s)
    jobs.append((n_, c[1] + ["--tok", "bpe1024", "--warmup", "100", "--save-model"]))
c = cell("D-d48-bpe1024_100M_s0", D, d=48, sc="8", chars="100e6", seed=0)
jobs.append((c[0], c[1] + ["--tok", "bpe1024", "--warmup", "100", "--save-model"]))
register("v5-final100k", jobs)


# v5-long (H24/H25): BPE1024 D d=48 at 300M chars, two learning rates, seed 0, saved
jobs = []
for lr in ("3e-2", "2e-2"):
    n_ = f"D-d48-bpe1024_300M_lr{lr}_s0"
    c = cell(n_, D, d=48, sc="8", chars="300e6", seed=0, lr=lr)
    jobs.append((n_, c[1] + ["--tok", "bpe1024", "--warmup", "100", "--save-model"]))
register("v5-long", jobs)


# v6-d-lr-edge: D's best lr was the top of the v6-d-lr grid (6e-2), so extend it (same cells otherwise)
_ed = []
for lr in ("8e-2", "1e-1"):
    for s in (0, 1):
        c = cell(f"D-lr{lr}_10M_s{s}", D, d=48, sc="8", chars="10e6", seed=s, lr=lr)
        _ed.append((c[0], c[1] + ["--tok", "bpe1024", "--warmup", "100"]))
register("v6-d-lr-edge", _ed)


# v7-vocab (H27): larger vocabularies under 100K at 10M chars, 2 seeds
_V7 = {
    "b1536-d40-h8": ("bpe1536", 40, "8"),
    "b2048-d32-h8": ("bpe2048", 32, "8"),
    "b1536-d36-h4": ("bpe1536", 36, "4"),
}
jobs = []
for k, (tok, d, h) in _V7.items():
    for s in (0, 1):
        c = cell(f"{k}_10M_s{s}", D, d=d, sc="8", chars="10e6", seed=s)
        jobs.append((c[0], c[1] + ["--heads", h, "--tok", tok, "--warmup", "100"]))
register("v7-vocab", jobs)


# v7-confirm (H28): the two larger-vocabulary winners at 30M chars, 2 seeds, saved
jobs = []
for k, (tok, d, h) in (
    ("b1536-d40-h8", _V7["b1536-d40-h8"]),
    ("b2048-d32-h8", _V7["b2048-d32-h8"]),
):
    for s in (0, 1):
        c = cell(f"{k}_30M_s{s}", D, d=d, sc="8", chars="30e6", seed=s)
        jobs.append(
            (
                c[0],
                c[1] + ["--heads", h, "--tok", tok, "--warmup", "100", "--save-model"],
            )
        )
register("v7-confirm", jobs)


# v8a-ctx (G29): context length 128 / 512 tokens, BPE1024 D d=48, 30M chars, 2 seeds, saved
jobs = []
for ctx in (128, 512):
    for s in (0, 1):
        c = cell(
            f"D-d48-bpe1024-ctx{ctx}_30M_s{s}", D, d=48, sc="8", chars="30e6", seed=s
        )
        jobs.append(
            (
                c[0],
                c[1]
                + [
                    "--tok",
                    "bpe1024",
                    "--warmup",
                    "100",
                    "--ctx",
                    str(ctx),
                    "--save-model",
                ],
            )
        )
register("v8a-ctx", jobs)


# v8a-ctx-b16 (G29b; sweep dir stays v8-ctx-b16): ctx 512 at batch 16 so tokens per step and step count match the 256 baseline
jobs = []
for s in (0, 1):
    c = cell(f"D-d48-bpe1024-ctx512b16_30M_s{s}", D, d=48, sc="8", chars="30e6", seed=s)
    jobs.append(
        (
            c[0],
            c[1]
            + [
                "--tok",
                "bpe1024",
                "--warmup",
                "100",
                "--ctx",
                "512",
                "--batch",
                "16",
                "--save-model",
            ],
        )
    )
register("v8-ctx-b16", jobs)


# v12-long512 (G30/G31): ctx 512 / 32 rules per head at 300M chars, saved
_c = cell(
    "D-d48-bpe1024-ctx512-r32_300M_s0",
    D,
    d=48,
    sc="8",
    chars="300e6",
    seed=0,
    lr="6e-2",
)
register(
    "v12-long512",
    [
        (
            _c[0],
            _c[1]
            + [
                "--tok",
                "bpe1024",
                "--warmup",
                "100",
                "--ctx",
                "512",
                "--batch",
                "16",
                "--rules",
                "32",
                "--save-model",
            ],
        )
    ],
)
