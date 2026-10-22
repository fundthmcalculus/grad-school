"""Sequence-level unlikelihood post-training of an all-fuzzy checkpoint (`PLAN_REPEAT.md` part 2).

``python -m flm.post_ul <checkpoint> <out.pt> --alpha 1 [--steps 400] [--lr 3e-3]``

Every step: 8 prefixes of 48 tokens from the training stream are continued by 96 tokens with the model's own deployed sampler (temperature 0.5, top-k 10, no gradient); a forward pass with gradient over prefix + continuation flags every continuation token that completes
a trigram already present earlier in the sequence (the prefix included) and adds ``alpha * mean(-log(1 - p(token | context)))`` over the flagged tokens; the ordinary next-token cross-entropy on a fresh 16 x 512 batch is added so the language model is retained.
``--alpha 0`` is the control: the same optimiser, schedule and data, no sampling and no extra term.  AdamW with the pretraining groups (weight decay on matrices except rule centres), betas (0.9, 0.98), gradient clip 1.0, linear warm-up of 20 steps then cosine decay.
Writes the checkpoint (with ``emb_mode: tail`` so that `flm.coherence_fuzzy.configure` rebuilds the fuzzy tail) and ``<out>.json`` (validation bits per character before and after, mean unlikelihood and flagged share over the run).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import flm.coherence_fuzzy as CF
import flm.train as T
from flm.bpe import BPE
from flm.models import ModelConfig, TinyLM

PREFIX, GEN, N_SEQ = 48, 96, 8


def sample(model, ids, gen):
    """Continue ``ids`` (B x L) by GEN tokens at temperature 0.5, top-k 10, no gradient."""
    for _ in range(GEN):
        logits = model(ids[:, -model.cfg.max_len :])[:, -1].float() / 0.5
        v, _ = torch.topk(logits, 10)
        logits[logits < v[:, [-1]]] = -float("inf")
        ids = torch.cat(
            [ids, torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)], 1
        )
    return ids


def repeat_mask(seq):
    """Positions (B x L bool) whose token completes a trigram that occurs earlier in the same sequence; only continuation positions count."""
    m = torch.zeros(seq.shape, dtype=torch.bool)
    for b in range(seq.shape[0]):
        t = seq[b].tolist()
        seen = set()
        for i in range(2, len(t)):
            g = (t[i - 2], t[i - 1], t[i])
            if g in seen and i >= PREFIX:
                m[b, i] = True
            seen.add(g)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    CF.configure(a.ckpt, ck)
    model = TinyLM(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state"])
    model = model.cuda()
    bp = BPE(int(ck["tok"].removeprefix("bpe")))
    T.TOK = bp
    train, val = bp.load("train"), np.asarray(bp.load("valid"))
    bpc0 = T.evaluate(model, val, 512, 1_000_000, device=torch.device("cuda"))
    decay = [
        p for n, p in model.named_parameters() if p.dim() >= 2 and "centers" not in n
    ]
    other = [
        p
        for n, p in model.named_parameters()
        if not (p.dim() >= 2 and "centers" not in n)
    ]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=a.lr,
        betas=(0.9, 0.98),
    )
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    hi = len(train) - 520
    log = dict(ul=[], flagged=[], ce=[])
    t0 = time.perf_counter()
    for s in range(a.steps):
        lr = a.lr * (
            (s + 1) / 20
            if s < 20
            else 0.5 * (1 + math.cos(math.pi * (s - 20) / max(1, a.steps - 20)))
        )
        for g in opt.param_groups:
            g["lr"] = lr
        ul = torch.zeros((), device="cuda")
        flagged = 0.0
        if a.alpha > 0:
            st = rng.integers(0, hi, N_SEQ)
            pref = torch.from_numpy(
                np.stack([train[i : i + PREFIX] for i in st]).astype(np.int64)
            ).cuda()
            model.eval()
            with torch.no_grad():
                seq = sample(model, pref, gen)
            model.train()
            mask = repeat_mask(seq.cpu())[
                :, 1:
            ].cuda()  # target positions of the shifted sequence
            logits = model(seq[:, :-1]).float()
            p_tok = torch.softmax(logits, -1).gather(-1, seq[:, 1:, None])[..., 0]
            ulv = -torch.log1p(-p_tok.clamp(max=1 - 1e-6))
            n = mask.sum()
            flagged = float(n) / mask.numel() * (seq.shape[1] / (seq.shape[1] - PREFIX))
            if n > 0:
                ul = (ulv * mask).sum() / n
        st = rng.integers(0, hi, 16)
        xb = torch.from_numpy(
            np.stack([train[i : i + 513] for i in st]).astype(np.int64)
        ).cuda()
        model.train()
        lg = model(xb[:, :-1])
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), xb[:, 1:].reshape(-1))
        loss = ce + a.alpha * ul
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log["ul"].append(float(ul))
        log["flagged"].append(flagged)
        log["ce"].append(float(ce))
        if s % 50 == 0 or s == a.steps - 1:
            print(
                f"step {s}/{a.steps} ce {float(ce):.4f} ul {float(ul):.4f} flagged {flagged:.3f} lr {lr:.5f} t={time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    bpc1 = T.evaluate(model, val, 512, 1_000_000, device=torch.device("cuda"))
    torch.save(
        dict(
            config=ck["config"],
            state=model.state_dict(),
            tok=ck["tok"],
            emb_mode="tail",
        ),
        a.out,
    )
    Path(a.out).with_suffix(".json").write_text(
        json.dumps(
            dict(
                ckpt=a.ckpt,
                alpha=a.alpha,
                steps=a.steps,
                lr=a.lr,
                bpc_before=bpc0,
                bpc_after=bpc1,
                ul_mean=float(np.mean(log["ul"])),
                flagged_first50=float(np.mean(log["flagged"][:50])),
                flagged_last50=float(np.mean(log["flagged"][-50:])),
                seconds=time.perf_counter() - t0,
            ),
            indent=1,
        )
    )
    print(f"validation bits per character {bpc0:.5f} -> {bpc1:.5f}; saved {a.out}")
