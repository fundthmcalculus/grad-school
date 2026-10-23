"""A DUST-like forward-only gradient estimator for the 100K FLM, on the CPU (exploration; owner's question: can a fuzzy LM be trained this way, and is it faster?).

DUST (qlabs-eng/dust, https://qlabs.sh/research/dust) estimates a layer's output error from forward passes alone: Gaussian noise is added to the layer's output at every token, each token's noise is scored by the change in loss (its own and, with a decay gamma, that of later tokens), and the scored noise is averaged over a population of draws; the weight gradient is then the usual outer product of that output error with the layer's input.
Adaptation to the FLM (our own, not the authors' code, which also treats attention internals, the head slab and the residual scalars specially): the model is cut into units, each with the parameters it owns -- the embedding; for every block the mixer sub-layer (norm, short convolution, fuzzy delta-rule mixer) and the feed-forward sub-layer (norm, TSK rule base); the head (final norm and the tied embedding).  The output of a unit (the residual stream after it; the logits for the head) is jittered, only the later stages are recomputed, and the unit's own parameter gradient is the local vector-Jacobian product of its output error through the unit alone (cached input, no chain across units: that is the only place autograd is used, and it never crosses a unit boundary).
For a linear layer this is exactly DUST's outer product; for the fuzzy rule bases and the delta-rule mixer it is the natural generalisation (local backprop inside a unit).

    python -m flm.dust_cpu cosine          # estimate quality against the true gradient, by population size
    python -m flm.dust_cpu train --method dust|backprop-sgd|backprop-adam --steps 200 ...
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import flm.train as T
from flm.bpe import BPE
from flm.run_tribble import argv


def build(seed: int = 0, vocab: int = 1024):
    a = T.get_args(argv("random", seed) + ["--device", "cpu"])
    a.vocab_size = vocab
    torch.manual_seed(seed)
    return T.TinyLM(T.make_config(a)), a


class Units:
    """The model as a chain of stages with the parameters each owns."""

    def __init__(self, m):
        self.m = m
        self.stages, self.params, self.names = [], [], []
        self.stages.append(lambda idx: m.embed(idx))
        self.params.append([m.emb.weight] + ([m.up.weight] if m.up is not None else []))
        self.names.append("emb")
        for b, blk in enumerate(m.blocks):
            self.stages.append(lambda x, blk=blk: x + blk.mix(blk.mixer_in(x)))
            self.params.append(
                list(blk.n1.parameters())
                + (list(blk.conv.parameters()) if blk.conv is not None else [])
                + list(blk.mix.parameters())
            )
            self.names.append(f"mix{b}")
            if blk.ffn is not None:
                self.stages.append(lambda x, blk=blk: x + blk.ffn(blk.n2(x)))
                self.params.append(
                    list(blk.n2.parameters()) + list(blk.ffn.parameters())
                )
                self.names.append(f"ffn{b}")
        self.params.append(list(m.norm.parameters()) + [m.emb.weight])
        self.names.append("head")
        self.n = len(self.stages)  # the stages before the head

    @torch.no_grad()
    def clean(self, idx):
        xs = [idx]
        for f in self.stages:
            xs.append(f(xs[-1]))
        return xs  # xs[k+1] is the output of stage k; xs[-1] the input of the head

    def head(self, h):
        return self.m.logits(h)


def tok_loss(logits, targets):
    """Per-token cross-entropy (natural log), shape of ``targets``."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).view(targets.shape)


def reward(c, gamma):
    """r_t = sum_{s >= t} gamma^(s-t) c_s along the last dimension."""
    if gamma == 0:
        return c
    r = c.clone()
    for t in range(c.shape[-1] - 2, -1, -1):
        r[..., t] = c[..., t] + gamma * r[..., t + 1]
    return r


def dust_grads(U: Units, idx, tgt, K, sig, gamma, k_head=None, gen=None):
    """Fills ``p.grad`` of every parameter with the DUST-like estimate (gradient of the MEAN token loss).  Returns the clean mean loss.
    ``sig``: dict unit-name -> noise scale relative to the rms of the jittered tensor; ``gamma``: dict unit-name -> credit decay.
    """
    m = U.m
    for p in m.parameters():
        p.grad = None
    B, T_ = tgt.shape
    ntok = B * T_
    xs = U.clean(idx)
    logits0 = U.head(xs[-1])
    loss0 = tok_loss(logits0, tgt)
    k_head = k_head or K

    def accumulate(k, g_out):
        """Local vector-Jacobian product of unit k's output error through the unit alone."""
        with torch.enable_grad():
            x_in = xs[k] if k == 0 else xs[k].detach()
            y = U.stages[k](x_in) if k < U.n else U.head(x_in)
            ps = U.params[k]
            gs = torch.autograd.grad(y, ps, g_out, allow_unused=True)
        for p, g in zip(ps, gs):
            if g is not None:
                p.grad = g.clone() if p.grad is None else p.grad + g

    with torch.no_grad():
        # the head: jitter the logits directly (cheap; the head only affects its own token)
        s = sig["head"] * logits0.std()
        a = torch.randn((k_head,) + logits0.shape, generator=gen)
        l_h = tok_loss(
            logits0.unsqueeze(0) + s * a, tgt.unsqueeze(0).expand(k_head, -1, -1)
        )
        c = l_h.mean(0, keepdim=True) - l_h
        g_logits = -(c.unsqueeze(-1) * a).sum(0) / (k_head * s) / ntok
    accumulate(U.n, g_logits)
    for k in range(U.n):
        name = U.names[k]
        x_k = xs[k + 1]
        s = sig[name] * x_k.std()
        with torch.no_grad():
            a = torch.randn((K,) + x_k.shape, generator=gen)
            x = (x_k.unsqueeze(0) + s * a).reshape((K * B,) + x_k.shape[1:])
            for j in range(k + 1, U.n):
                x = U.stages[j](x)
            l_k = tok_loss(U.head(x), tgt.repeat(K, 1)).view(K, B, T_)
            c = l_k.mean(0, keepdim=True) - l_k
            r = reward(c, gamma[name])
            g_out = -(r.unsqueeze(-1) * a).sum(0) / (K * s) / ntok
        accumulate(k, g_out)
    return float(loss0.mean())


def true_grads(m, idx, tgt):
    for p in m.parameters():
        p.grad = None
    loss = tok_loss(m(idx), tgt).mean()
    loss.backward()
    return float(loss)


def flat(ps):
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in ps
        ]
    )


def cosine(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def get_data(vocab=1024, train_tokens=2_000_000, val_tokens=100_000):
    bp = BPE(vocab)
    tr = np.asarray(bp.load("train")[:train_tokens]).astype(np.int64)
    va = np.asarray(bp.load("valid")[:val_tokens]).astype(np.int64)
    return bp, torch.from_numpy(tr), torch.from_numpy(va)


def batch(data, B, T_, rng):
    st = rng.integers(0, len(data) - T_ - 2, B)
    x = torch.stack([data[i : i + T_ + 1] for i in st])
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def val_bits(m, bp, va, T_=128, n=200):
    """Mean cross-entropy in bits per token and per character (the BPE token lengths) on n fixed windows."""
    m.eval()
    tot, chars = 0.0, 0
    lens = torch.from_numpy(np.asarray(bp.lens).astype(np.int64))
    for i in range(n):
        x = va[i * (T_ + 1) : (i + 1) * (T_ + 1)]
        if len(x) < T_ + 1:
            break
        lo = tok_loss(m(x[None, :-1]), x[None, 1:])
        tot += float(lo.sum())
        chars += int(lens[x[1:]].sum())
    m.train()
    return tot / chars / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cosine", "train"])
    ap.add_argument(
        "--method",
        default="dust",
        choices=[
            "dust",
            "backprop-sgd",
            "backprop-adam",
            "backprop-lion",
            "backprop-muon",
        ],
    )
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument(
        "--gamma", type=float, default=0.0, help="credit decay of the mixer units"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument(
        "--warm",
        type=int,
        default=0,
        help="cosine mode: backprop steps first, so the test is at a trained point",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    m, args = build(a.seed)
    bp, tr, va = get_data()
    rng = np.random.default_rng(a.seed)
    U = Units(m)
    sig = {n: a.sigma for n in U.names}
    gam = {n: (a.gamma if n.startswith("mix") else 0.0) for n in U.names}
    ps_all = [p for p in m.parameters()]
    print("parameters", m.n_params(), "units", U.names, flush=True)
    if a.mode == "cosine":
        if a.warm:  # a briefly backprop-trained point (Adam)
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            for s in range(a.warm):
                x, y = batch(tr, 16, a.ctx, rng)
                opt.zero_grad()
                tok_loss(m(x), y).mean().backward()
                opt.step()
            print(
                "warm",
                a.warm,
                "steps; val bits/char",
                round(val_bits(m, bp, va), 4),
                flush=True,
            )
        x, y = batch(tr, a.batch, a.ctx, rng)
        true_grads(m, x, y)
        gt = {n: flat(U.params[k]) for k, n in enumerate(U.names)}
        gt_all = flat(ps_all)
        gen = torch.Generator().manual_seed(1)
        rows = []
        for K in [8, 32, 128, 512]:
            t0 = time.perf_counter()
            dust_grads(U, x, y, K, sig, gam, gen=gen)
            dt = time.perf_counter() - t0
            row = {
                "K": K,
                "seconds": round(dt, 2),
                "all": round(cosine(flat(ps_all), gt_all), 4),
            }
            for k, n in enumerate(U.names):
                if n.startswith(
                    ("mix", "ffn")
                ):  # their parameters are exclusive to the unit
                    row[n] = round(cosine(flat(U.params[k]), gt[n]), 4)
            rows.append(row)
            print(row, flush=True)
        return
    # ---- training
    params = [p for p in m.parameters()]
    if a.method == "backprop-adam":
        opt = torch.optim.Adam(params, lr=a.lr)
    elif a.method == "backprop-lion":
        from flm.cheap_opts import Lion

        opt = Lion(params, lr=a.lr)
    elif a.method == "backprop-muon":
        from flm.cheap_opts import Muon

        mats = {id(q) for q in params if q.dim() == 2 and q is not m.emb.weight}
        opt = Muon(params, mats, lr=a.lr, adam_lr=3e-3)
    else:
        opt = torch.optim.SGD(params, lr=a.lr, momentum=a.momentum)
    gen = torch.Generator().manual_seed(a.seed + 7)
    curve, t_train = [], 0.0
    curve.append((0, 0.0, round(val_bits(m, bp, va), 4)))
    for s in range(1, a.steps + 1):
        x, y = batch(tr, a.batch, a.ctx, rng)
        t0 = time.perf_counter()
        if a.method == "dust":
            loss = dust_grads(U, x, y, a.K, sig, gam, gen=gen)
        else:
            loss = true_grads(m, x, y)
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        t_train += time.perf_counter() - t0
        if not math.isfinite(loss):
            print("diverged at step", s)
            curve.append((s, round(t_train, 1), float("nan")))
            break
        if s % a.eval_every == 0 or s == a.steps:
            v = val_bits(m, bp, va)
            curve.append((s, round(t_train, 1), round(v, 4)))
            print(
                f"step {s} train_loss {loss:.3f} val_bpc {v:.4f} train_seconds {t_train:.1f}",
                flush=True,
            )
    rec = dict(
        method=a.method,
        args=vars(a),
        params=m.n_params(),
        tokens_per_step=a.batch * a.ctx,
        curve=curve,
        seconds_per_step=t_train / max(1, a.steps),
    )
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
