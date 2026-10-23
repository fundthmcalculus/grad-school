"""Post-training quantisation of an FP32-trained all-fuzzy checkpoint, on the CPU (`PLAN_PRECISION.md`, fork task 2026-10-23).

    CUDA_VISIBLE_DEVICES= python -m flm.ptq_probe [checkpoint] [--chars 300000] [--threads 3] [--out report/ptq_probe.json]

Weights are rounded to a uniform symmetric grid and put back as floats ("fake quantisation": the loss of precision is exact, the storage saving is accounted for separately), then the model is scored in bits per character on the same validation text the training runs use.  No retraining, no calibration data, no clipping search: the simplest PTQ.
Groups of parameters: emb_head (crisp head token table), emb_tail (fuzzy tail memberships), tier_up (tail up-projection), proj (low-rank A/B projections of the mixer), ffn_ante (feed-forward rule centres and log-widths), ffn_cons (feed-forward consequents, slopes), mix_other (the mixer's centres, widths, gains, gate and beta rules), norm (every normaliser).
Per-row = one scale per row of a matrix (per slice along the first axis); per-tensor = one scale for the tensor; vectors are always per-tensor.  Size counts the weights at their bit width plus 16 bits per scale.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import flm.coherence_fuzzy as CF
import flm.train as T
from flm.bpe import BPE
from flm.models import ModelConfig, TinyLM

GROUPS = [
    "emb_head",
    "emb_tail",
    "tier_up",
    "proj",
    "ffn_ante",
    "ffn_cons",
    "mix_other",
    "norm",
]


def group_of(name: str) -> str:
    if name.startswith("emb_h"):
        return "emb_head"
    if name.startswith("emb_t"):
        return "emb_tail"
    if name.startswith("tier_up"):
        return "tier_up"
    if ".n1." in name or ".n2." in name or name.startswith("norm."):
        return "norm"
    if name.endswith("A.weight") or name.endswith("B.weight"):
        return "proj"
    if "ffn.centers" in name or "ffn.log_width" in name:
        return "ffn_ante"
    if "ffn." in name:
        return "ffn_cons"
    return "mix_other"


def fake_quant(w: torch.Tensor, bits: int, per_row: bool):
    """Round to a symmetric uniform grid of 2**bits - 1 levels; returns (dequantised tensor, number of scales)."""
    qmax = 2 ** (bits - 1) - 1
    if per_row and w.dim() >= 2:
        flat = w.reshape(w.shape[0], -1)
        scale = flat.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
        q = torch.round(flat / scale).clamp(-qmax, qmax)
        return (q * scale).reshape(w.shape), flat.shape[0]
    scale = w.abs().max().clamp_min(1e-12) / qmax
    return torch.round(w / scale).clamp(-qmax, qmax) * scale, 1


def quantise(state, bits_by_group: dict, per_row: bool):
    """New state dict with the listed groups quantised; returns it with the bytes the weights would take."""
    out, nbytes = {}, 0.0
    for k, v in state.items():
        g = group_of(k)
        b = bits_by_group.get(g)
        if b is None or b >= 32:
            out[k] = v
            nbytes += v.numel() * 4
        elif b == 16:  # fp16 round trip
            out[k] = v.half().float()
            nbytes += v.numel() * 2
        elif b == -16:  # bf16 round trip
            out[k] = v.bfloat16().float()
            nbytes += v.numel() * 2
        else:
            q, n_scales = fake_quant(v, b, per_row)
            out[k] = q
            nbytes += v.numel() * b / 8 + n_scales * 2
    return out, nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "ckpt", nargs="?", default="outputs/v70-shape300/F7_v6144_h640_d32_floor0_s0.pt"
    )
    ap.add_argument("--chars", type=int, default=300_000)
    ap.add_argument("--full-chars", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default="report/ptq_probe.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    CF.configure(a.ckpt, ck)
    model = TinyLM(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state"])
    model.eval()
    bp = BPE(int(ck["tok"].removeprefix("bpe")))
    T.TOK = bp
    val = bp.load("valid")
    base = {k: v.clone() for k, v in ck["state"].items()}
    cpu = torch.device("cpu")
    rows = []

    def score(name, bits_by_group, per_row=True, chars=None, autocast=False):
        state, nbytes = quantise(base, bits_by_group, per_row)
        model.load_state_dict(state)
        t0 = time.perf_counter()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            bpc = T.evaluate(model, val, 512, chars or a.chars, device=cpu)
        rows.append(
            dict(name=name, bits=bits_by_group, per_row=per_row, chars=chars or a.chars,
                 bpc=bpc, bytes=int(nbytes), seconds=round(time.perf_counter() - t0, 1))
        )  # fmt: skip
        print(
            f"{name:34s} {bpc:.5f}  {int(nbytes):7d} B  ({rows[-1]['seconds']}s)",
            flush=True,
        )
        return bpc

    # 0. sanity: the CPU fp32 evaluation reproduces the training run's number (0.95577 at 1M characters)
    ref_full = score("fp32 (full 1M chars)", {}, chars=a.full_chars)
    ref = score("fp32", {})
    allg = {g: None for g in GROUPS}
    # 1. casts
    score("fp16 weights", {g: 16 for g in GROUPS})
    score("bf16 weights", {g: -16 for g in GROUPS})
    try:
        score("bf16 weights + bf16 autocast", {g: -16 for g in GROUPS}, autocast=True)
    except (
        RuntimeError
    ) as e:  # flm/fuzzy_tokens.py _core writes bf16 into an fp32 buffer: autocast does not run on the fuzzy-tail path
        rows.append(dict(name="bf16 weights + bf16 autocast", error=str(e)[:200]))
        print("bf16 autocast FAILED:", str(e)[:120], flush=True)
    # 2. everything quantised
    for bits in (8, 6, 5, 4, 3, 2):
        for per_row in (True, False):
            score(
                f"all int{bits} {'row' if per_row else 'tensor'}",
                {g: bits for g in GROUPS},
                per_row,
            )
    # 3. sensitivity: one group at a time
    for bits in (4, 3, 2):
        for g in GROUPS:
            score(f"only {g} int{bits} row", {g: bits})
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(
        json.dumps(dict(ckpt=a.ckpt, ref=ref, ref_full=ref_full, rows=rows), indent=1)
    )
    print("reference", ref, "full", ref_full)


if __name__ == "__main__":
    main()
