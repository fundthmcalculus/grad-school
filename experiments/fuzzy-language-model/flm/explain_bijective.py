"""Explainability tests for the invertible meaning-layer models (`PLAN_BIJECTIVE.md`).

    BIJ=ci FUZZ_FFN_R=156 FUZZ_FFN_R0=109 GRAD_SCHOOL_DATA=<root> python -m flm.explain_bijective CKPT.pt OUT.json [--windows 256]

Run once per checkpoint (set ``BIJ=tied`` and 186/130 for the tied head).  Four questions, all on validation windows of 256 characters:
  lens      what does the state say after each block?  The state is read through the model's own output stage (for ``ci``: the exact inverse G^-1 then F^-1 and the nearest codeword; for ``tied``: the tied head), giving next-character accuracy, bits per character, and the share of positions where the decoded state is still the current character.
  rules     per block, how many rules does the feed-forward system really use (effective number exp(H) of the normalised firing), how concentrated is the winning rule, and how much does knowing the winning rule tell about the next character (mutual information in bits, against what the previous character alone tells)?
  ablation  is the rulebook causal?  Bits per character with the 8 most-used rules of a block silenced, against 8 random rules of the same block (10 draws).
  book      for the 8 most-used rules of each block: the characters that most often come next when the rule wins, with their share, as plain words.
Plus greedy continuations of a few prompts (no sampling) with a repetition measure.
"""

import json
import math
import os
import sys

import numpy as np
import torch

import flm.run_bijective as RB
from flm.data import VOCAB_SIZE, decode, encode, load
from flm.fuzzy_parts import FuzzyFFNRaw
from flm.models import ModelConfig, TinyLM

BIJ = os.environ.get("BIJ", "ci")
CTX = 256
PROMPTS = [
    "Once upon a time, there was a little girl named",
    "The meeting is scheduled for",
    "In this paper we show that",
    "def main():\n    ",
    "The president of the United States",
]


def build(path, device):
    TinyLM.__init__ = RB._new_init
    TinyLM.embed = RB._embed
    TinyLM.logits = RB._logits
    TinyLM.n_params = RB._n_params
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyLM(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state"])
    return model.to(device).eval()


def windows(n, device, seed=0):
    v = load("valid")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(v) - CTX - 1, size=n)
    x = np.stack([v[s : s + CTX + 1] for s in starts]).astype(np.int64)
    t = torch.as_tensor(x, device=device)
    return t[:, :-1], t[:, 1:]


def bits(logits, y):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)
    ).item() / math.log(2)


def mi(a, b, na, nb):
    """Mutual information in bits between two integer arrays."""
    j = np.zeros((na, nb))
    np.add.at(j, (a, b), 1)
    j /= j.sum()
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return float((j[nz] * np.log2(j[nz] / (pa @ pb)[nz])).sum())


def chars(ids):
    return decode(ids).replace("\n", "\\n")


@torch.no_grad()
def lens(model, x, y):
    out = []
    h = model.embed(x)
    states = [("embedding", h)]
    for i, blk in enumerate(model.blocks):
        h = blk(h)
        states.append((f"block {i}", h))
    for name, h in states:
        lg = model.logits(h)
        pred = lg.argmax(-1)
        out.append(
            dict(
                stage=name,
                next_acc=(pred == y).float().mean().item(),
                bpc=bits(lg, y),
                reads_current_char=(pred == x).float().mean().item(),
            )
        )
    return out


def firing(ffn, x):
    """Raw Gaussian firing strength of every rule (product t-norm, exponent 1/d), before the first-order slope."""
    u = ffn.antecedent_input(x)
    a = torch.exp(-2.0 * ffn.log_width)
    ca = ffn.centers * a
    quad = (u * u) @ a.T - 2.0 * (u @ ca.T) + (ffn.centers * ca).sum(-1)
    return torch.exp(-0.5 * ffn.scale * quad)


def block_inputs(model, x):
    """The normalised input of every block's feed-forward system."""
    h = model.embed(x)
    ins = []
    for blk in model.blocks:
        h = h + blk.mix(blk.mixer_in(h))
        ins.append(blk.n2(h))
        h = h + blk.ffn(ins[-1])
    return ins


@torch.no_grad()
def rules(model, x, y):
    ins = block_inputs(model, x)
    xs, ys = x.cpu().numpy().ravel(), y.cpu().numpy().ravel()
    prev_mi = mi(xs, ys, VOCAB_SIZE, VOCAB_SIZE)
    res, books = [], []
    for i, (blk, u) in enumerate(zip(model.blocks, ins)):
        w = firing(blk.ffn, u)
        p = w.reshape(-1, w.shape[-1]).float()
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-30)
        eff = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum(-1))
        win = p.argmax(-1).cpu().numpy()
        R = p.shape[-1]
        use = np.bincount(win, minlength=R)
        order = np.argsort(-use)
        res.append(
            dict(
                block=i,
                rules=R,
                effective_rules=eff.mean().item(),
                top_rule_share=float(use[order[0]] / use.sum()),
                rules_ever_winning=int((use > 0).sum()),
                mi_rule_next=mi(win, ys, R, VOCAB_SIZE),
                mi_prev_next=prev_mi,
                mi_rule_prev=mi(win, xs, R, VOCAB_SIZE),
                top8_share=float(use[order[:8]].sum() / use.sum()),
            )
        )
        rows = []
        for r in order[:8]:
            m = win == r
            nxt = np.bincount(ys[m], minlength=VOCAB_SIZE)
            prv = np.bincount(xs[m], minlength=VOCAB_SIZE)
            top = np.argsort(-nxt)[:5]
            ptop = np.argsort(-prv)[:3]
            rows.append(
                dict(
                    rule=int(r),
                    wins=float(use[r] / use.sum()),
                    next=[[chars([c]), float(nxt[c] / nxt.sum())] for c in top],
                    after=[[chars([c]), float(prv[c] / prv.sum())] for c in ptop],
                )
            )
        books.append(rows)
    return res, books


@torch.no_grad()
def ablation(model, x, y, k=8, draws=10, seed=0):
    ins = block_inputs(model, x)
    base = bits(model(x), y)
    out = []
    orig = FuzzyFFNRaw.forward
    for i, blk in enumerate(model.blocks):
        w = firing(blk.ffn, ins[i]).reshape(-1, blk.ffn.R)
        use = np.bincount(w.argmax(-1).cpu().numpy(), minlength=blk.ffn.R)
        top = np.argsort(-use)[:k]
        rng = np.random.default_rng(seed + i)

        def run(idx):
            def fwd(self, h):
                if self is not blk.ffn:
                    return orig(self, h)
                g = firing(self, h)
                g[..., idx] = 0.0
                if self.order == 1:
                    g = g * self.slope(h)
                return g @ self.conseq

            FuzzyFFNRaw.forward = fwd
            try:
                return bits(model(x), y)
            finally:
                FuzzyFFNRaw.forward = orig

        rand = [
            run(rng.choice(blk.ffn.R, size=k, replace=False)) - base
            for _ in range(draws)
        ]
        out.append(
            dict(
                block=i,
                base_bpc=base,
                top_cost=run(top) - base,
                random_cost_mean=float(np.mean(rand)),
                random_cost_max=float(np.max(rand)),
            )
        )
    return out


@torch.no_grad()
def greedy(model, device, n=300):
    outs = []
    for p in PROMPTS:
        ids = torch.as_tensor(encode(p)[None], device=device).long()
        for _ in range(n):
            nxt = model(ids[:, -CTX:])[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
        t = decode(ids[0].cpu().numpy())
        g = t[len(p) :]
        grams = [g[j : j + 8] for j in range(len(g) - 7)]
        outs.append(
            dict(
                prompt=p,
                text=g,
                distinct_8grams=len(set(grams)) / max(1, len(grams)),
            )
        )
    return outs


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path, out = sys.argv[1], sys.argv[2]
    n = (
        int(sys.argv[sys.argv.index("--windows") + 1])
        if "--windows" in sys.argv
        else 256
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(path, dev)
    torch.manual_seed(0)
    x, y = windows(n, dev)
    r, books = rules(model, x, y)
    res = dict(
        arm=BIJ,
        ckpt=path,
        windows=n,
        lens=lens(model, x, y),
        rules=r,
        book=books,
        ablation=ablation(model, x, y),
        greedy=greedy(model, dev),
    )
    json.dump(res, open(out, "w"), indent=1)
    for l in res["lens"]:
        print(l)
    for l in r:
        print(l)
    for l in res["ablation"]:
        print(l)
    for g in res["greedy"]:
        print(
            repr(g["prompt"]),
            "->",
            repr(g["text"][:160]),
            round(g["distinct_8grams"], 2),
        )


if __name__ == "__main__":
    main()
