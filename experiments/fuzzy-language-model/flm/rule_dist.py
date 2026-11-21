"""Deconstruct a trained FLM's output distribution into its TSK rules (direct path, exact).

    python -m flm.rule_dist outputs/<grid>/<run>.pt [--windows 128] [--len 256]

The last operation of the model is ``logits = RMSNorm(x_L) E^T`` with ``x_L = x_0 + sum_l (mix_l + ffn_l)``. RMSNorm is a
per-position scalar times a fixed per-dimension weight, so the logits are an EXACT sum of per-component terms:

    logits_t = s_t * [ (x_0 * nw) E^T  +  sum_l (mix_l * nw) E^T  +  sum_l sum_r g_{l,r}(t) * (conseq_{l,r} * nw) E^T ]

where g_{l,r}(t) = firing_r * slope_r(x) is a scalar gate and ``pi_{l,r} = (conseq_{l,r} * nw) E^T`` is a FIXED vocabulary
profile: every FFN rule adds the same vector over the vocabulary, scaled by its gate. That is what makes the output
distribution decomposable rule by rule (and a rulebook a possible distillation target). Layer-0 rules also act through
layer 1, which this decomposition does not see; the sparse-rule test below is a true forward pass, so it does.

Experiments printed:
  A  manipulation check: the decomposition reproduces the model's logits;
  B  direct-path importance: BPC when whole component classes are removed from the logits;
  C  sparse rulebook: BPC when only the top-k rules per position (per layer) are allowed to fire (true forward pass);
  D  profile redundancy: effective rank of the rule profiles and near-duplicate rules;
  E  faithfulness of a rule's stated consequent: when a rule dominates a position, how often is the true next token among
     the top tokens of its signed profile (vs the same-size set drawn at random by token frequency).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flm.models import TSKFFN
from flm.rules import load_model, make_codec

LN2 = math.log(2)


@torch.no_grad()
def decompose(model, idx):
    """Return dict with final logits, scale s (B,T), and the direct-path terms (see module docstring)."""
    E = model.emb.weight
    nw = model.norm.weight
    x = model.embed(idx)
    parts = {"x0": x}
    gates, mixes = [], []
    for blk in model.blocks:
        m = blk.mix(blk.mixer_in(x))
        x = x + m
        mixes.append(m)
        n = blk.n2(x)
        ffn = blk.ffn
        assert isinstance(ffn, TSKFFN) and ffn.order == 1
        g = ffn.firing(n) * ffn.slope(n)
        gates.append(g)
        x = x + g @ ffn.conseq
    rms = (
        x.pow(2)
        .mean(-1, keepdim=True)
        .add(model.norm.eps or torch.finfo(x.dtype).eps)
        .sqrt()
    )
    s = 1.0 / rms
    pis = [(blk.ffn.conseq * nw) @ E.T for blk in model.blocks]  # (R, V) per layer
    out = {
        "logits": model.logits(x),
        "s": s,
        "emb": (parts["x0"] * nw) @ E.T * s,
        "mix": [(m * nw) @ E.T * s for m in mixes],
        "gates": gates,
        "pis": pis,
        "ffn": [(g * s) @ p for g, p in zip(gates, pis)],
    }
    return out


def bpc_of(logits, tgt, lens):
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum"
    ).item()
    return nll / int(lens[tgt.numpy()].astype(np.int64).sum()) / LN2


@torch.no_grad()
def topk_bpc(model, idx, lens, k, batch=32):
    """True forward pass in which only the top-k rules per position (by |gate| * ||conseq * nw||) fire in every layer."""
    nw = model.norm.weight
    undo = []
    for blk in model.blocks:
        ffn = blk.ffn
        scale = (ffn.conseq * nw).norm(dim=-1)  # (R,)
        orig = ffn.forward

        def fwd(x, _f=ffn, _sc=scale, _k=k):
            g = _f.firing(x) * _f.slope(x)
            if _k < g.shape[-1]:
                top = (g.abs() * _sc).topk(_k, dim=-1).indices
                mask = torch.zeros_like(g).scatter_(-1, top, 1.0)
                g = g * mask
            return g @ _f.conseq

        ffn.forward = fwd
        undo.append((ffn, orig))
    tot_nll, chars = 0.0, 0
    for i in range(0, len(idx), batch):
        xb = idx[i : i + batch]
        lg = model(xb[:, :-1])
        tot_nll += F.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), xb[:, 1:].reshape(-1), reduction="sum"
        ).item()
        chars += int(lens[xb[:, 1:].numpy()].astype(np.int64).sum())
    for ffn, orig in undo:
        ffn.forward = orig
    return tot_nll / chars / LN2


@torch.no_grad()
def lowrank_bpc(model, idx, lens, bases, r, batch=32):
    """True forward pass in which every layer's gate vector is projected onto the top-r right singular vectors of the
    gate matrix (measured on the same windows with the unmodified model): a student with r independent rules.
    """
    undo = []
    for blk, V in zip(model.blocks, bases):
        ffn = blk.ffn
        Vr = V[:, :r]
        orig = ffn.forward

        def fwd(x, _f=ffn, _V=Vr):
            g = _f.firing(x) * _f.slope(x)
            return ((g @ _V) @ _V.T) @ _f.conseq

        ffn.forward = fwd
        undo.append((ffn, orig))
    tot, chars = 0.0, 0
    for i in range(0, len(idx), batch):
        xb = idx[i : i + batch]
        lg = model(xb[:, :-1])
        tot += F.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), xb[:, 1:].reshape(-1), reduction="sum"
        ).item()
        chars += int(lens[xb[:, 1:].numpy()].astype(np.int64).sum())
    for ffn, orig in undo:
        ffn.forward = orig
    return tot / chars / LN2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--windows", type=int, default=128)
    p.add_argument("--len", type=int, default=256)
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--out", default="")
    a = p.parse_args()
    torch.set_num_threads(4)
    model, tok = load_model(a.ckpt)
    codec = make_codec(tok)
    lens = (
        np.asarray(codec.bpe.lens)
        if tok
        else np.ones(model.cfg.vocab_size, dtype=np.uint8)
    )
    idx = codec.windows(a.windows, a.len + 1) if a.len else None
    tgt = idx[:, 1:]
    res = {"ckpt": a.ckpt, "windows": a.windows, "len": a.len}

    D = decompose(model, idx[:, :-1])
    # A
    diff = (D["emb"] + sum(D["mix"]) + sum(D["ffn"]) - D["logits"]).abs().max().item()
    res["A_max_abs_logit_error"] = diff
    full = bpc_of(D["logits"], tgt, lens)
    res["full_bpc"] = full
    print(
        f"A  reconstruction max |error| = {diff:.2e}   full-model BPC on these windows = {full:.4f}"
    )
    # B
    comp = {
        "emb": D["emb"],
        "mix0": D["mix"][0],
        "mix1": D["mix"][1],
        "ffn0": D["ffn"][0],
        "ffn1": D["ffn"][1],
    }
    allsum = sum(comp.values())
    B = {}
    for name in comp:
        B[f"drop {name}"] = bpc_of(allsum - comp[name], tgt, lens)
    B["rules only (ffn0+ffn1)"] = bpc_of(comp["ffn0"] + comp["ffn1"], tgt, lens)
    B["ffn1 only"] = bpc_of(comp["ffn1"], tgt, lens)
    B["emb + rules"] = bpc_of(comp["emb"] + comp["ffn0"] + comp["ffn1"], tgt, lens)
    B["mixers only"] = bpc_of(comp["mix0"] + comp["mix1"], tgt, lens)
    res["B"] = B
    print(
        "B  direct-path importance (BPC; lower = better; the downstream effect of layer-0 terms is NOT included):"
    )
    for k, v in B.items():
        print(f"   {k:26s} {v:8.4f}")
    # C
    ks = [1, 2, 4, 8, 16, 32, model.blocks[0].ffn.R]
    C = {k: topk_bpc(model, idx, lens, k) for k in ks}
    res["C_topk_bpc"] = {str(k): v for k, v in C.items()}
    print(
        "C  sparse rulebook, true forward, top-k rules per position per layer (R = %d):"
        % model.blocks[0].ffn.R
    )
    for k, v in C.items():
        print(f"   k={k:3d}  BPC {v:8.4f}   (+{v - full:.4f} vs all rules)")
    # F
    bases = []
    for g in D["gates"]:
        G = g.reshape(-1, g.shape[-1])
        bases.append(torch.linalg.svd(G, full_matrices=False).Vh.T)
    sv = [torch.linalg.svdvals(g.reshape(-1, g.shape[-1])) for g in D["gates"]]
    res["F_gate_energy"] = [
        {r: float(s_[:r].pow(2).sum() / s_.pow(2).sum()) for r in (4, 8, 16, 24, 36)}
        for s_ in sv
    ]
    Rn = model.blocks[0].ffn.R
    Fr = {r: lowrank_bpc(model, idx, lens, bases, r) for r in (4, 8, 16, 24, 36, Rn)}
    res["F_lowrank_bpc"] = {str(k): v for k, v in Fr.items()}
    print(
        "F  low-rank student: gate vectors projected to rank r in every layer (true forward):"
    )
    for r, v in Fr.items():
        e = [f"{x[r] * 100:.1f}%" if r in x else "-" for x in res["F_gate_energy"]]
        print(
            f"   rank {r:3d}  BPC {v:8.4f}  (+{v - full:.4f})   gate energy kept (L0, L1): {e}"
        )
    # D
    res["D"] = []
    print("D  profile redundancy (centered rule profiles over the vocabulary):")
    for li, pi in enumerate(D["pis"]):
        P = pi - pi.mean(-1, keepdim=True)
        sv = torch.linalg.svdvals(P)
        pr = (sv.pow(2).sum() ** 2 / sv.pow(4).sum()).item()
        Pn = F.normalize(P, dim=-1)
        cs = Pn @ Pn.T
        cs.fill_diagonal_(-1)
        nn_cos = cs.max(-1).values
        d = {
            "layer": li,
            "rules": P.shape[0],
            "participation_rank": pr,
            "near_dup_gt_0.9": int((nn_cos > 0.9).sum()),
            "median_nn_cos": nn_cos.median().item(),
        }
        res["D"].append(d)
        print(f"   layer {li}: {d}")
    # E
    print(
        f"E  faithfulness of the stated consequent (target in the top-{a.topn} of a rule's signed profile, last layer):"
    )
    last = len(D["pis"]) - 1
    g, pi = D["gates"][last], D["pis"][last]
    score = g.abs() * (model.blocks[last].ffn.conseq * model.norm.weight).norm(dim=-1)
    dom = score.argmax(-1)  # (B,T) dominant rule
    sign = torch.sign(g.gather(-1, dom[..., None]))[..., 0]
    topi = pi.topk(a.topn, dim=-1).indices  # raise side
    topn_ = (-pi).topk(a.topn, dim=-1).indices  # lower side
    hit_stated = torch.zeros(dom.shape)
    for r in range(pi.shape[0]):
        m = dom == r
        if not m.any():
            continue
        raise_set = topi[r]
        lower_set = topn_[r]
        t = tgt[m]
        pos = sign[m] >= 0
        h = torch.where(pos, torch.isin(t, raise_set), torch.isin(t, lower_set))
        hit_stated[m] = h.float()
    # model's own top-n as an upper reference, and the base rate of drawing n tokens by frequency
    mt = D["logits"].topk(a.topn, -1).indices
    hit_model = (mt == tgt[..., None]).any(-1).float()
    freq = torch.bincount(tgt.reshape(-1), minlength=pi.shape[1]).float()
    freq /= freq.sum()
    base = float(freq.topk(a.topn).values.sum())
    res["E"] = {
        "stated_top_hit": hit_stated.mean().item(),
        "model_top_hit": hit_model.mean().item(),
        "frequency_top_hit": base,
    }
    print(
        f"   rule's stated consequent hits the true token: {hit_stated.mean():.3f}   model's own top-{a.topn}: {hit_model.mean():.3f}   the {a.topn} most frequent tokens: {base:.3f}"
    )
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
