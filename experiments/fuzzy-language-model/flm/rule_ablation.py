"""Is the rulebook causal? Silence the rules the book cites and see what it costs (PLAN_COHERENCE.md).

    .venv/bin/python -m flm.rule_ablation CKPT.pt [--k 8] [--draws 20] [--windows 200] [--out ab.json]

For each TSK-FFN layer, bits per character on validation windows with (a) nothing removed, (b) the k most
used rules silenced (the rules ``flm.rules`` lists first: most positions won), (c) k random rules of the same
layer silenced, ``--draws`` times. A rulebook is causal if (b) costs clearly more than the (c) spread. Silencing
sets a rule's firing to zero, so its first-order output (and bias) disappears from the sum; with a normalised
FFN the other rules are not renormalised, which is stated rather than hidden. Also reports the
coverage the book has: the share of positions where a listed rule is the winning rule.

Registered before running (CPU session, 2026-10-11): on the 100M BPE1024 D checkpoint, silencing the 8 most
used rules of a layer costs more than the 95th percentile of 20 random 8-rule draws, in both layers.
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
def bpc(model, windows: torch.Tensor, lens: np.ndarray, batch: int = 32) -> float:
    nll, chars = 0.0, 0
    for i in range(0, len(windows), batch):
        xb = windows[i : i + batch]
        logits = model(xb[:, :-1])
        nll += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1), reduction="sum"
        ).item()
        chars += int(lens[xb[:, 1:].numpy()].astype(np.int64).sum())
    return nll / chars / LN2


class Silence:
    """Context manager: zero the firing of chosen rules of one TSK FFN."""

    def __init__(self, ffn: TSKFFN, rules):
        self.ffn, self.rules = ffn, list(rules)

    def __enter__(self):
        self.orig = self.ffn.firing
        idx = torch.tensor(self.rules, dtype=torch.long)

        def firing(x, _o=self.orig, _i=idx):
            w = _o(x).clone()
            w[..., _i] = 0.0
            return w

        self.ffn.firing = firing
        return self

    def __exit__(self, *exc):
        self.ffn.firing = self.orig


@torch.no_grad()
def usage(model, ffn: TSKFFN, windows: torch.Tensor):
    """Winning-rule share per rule over all positions (what the rulebook sorts by)."""
    wins, seen = None, 0
    orig = ffn.firing

    def firing(x, _o=orig):
        nonlocal wins, seen
        w = _o(x)
        c = torch.bincount(w.argmax(-1).reshape(-1), minlength=w.shape[-1]).double()
        wins = c if wins is None else wins + c
        seen += w.shape[0] * w.shape[1]
        return w

    ffn.firing = firing
    try:
        for i in range(0, len(windows), 32):
            model(windows[i : i + 32, :-1])
    finally:
        ffn.firing = orig
    return (wins / seen).numpy()


def ablate(model, windows, lens, k=8, draws=20, seed=0):
    rng = np.random.default_rng(seed)
    base = bpc(model, windows, lens)
    out = {"base_bpc": round(base, 4), "layers": []}
    for li, blk in enumerate(model.blocks):
        if not isinstance(blk.ffn, TSKFFN):
            continue
        share = usage(model, blk.ffn, windows)
        R = len(share)
        top = [int(r) for r in np.argsort(-share)[:k]]
        with Silence(blk.ffn, top):
            cited = bpc(model, windows, lens)
        rand = []
        for _ in range(draws):
            pick = rng.choice(R, size=k, replace=False).tolist()
            with Silence(blk.ffn, pick):
                rand.append(bpc(model, windows, lens))
        rand_a = np.array(rand)
        out["layers"].append(
            {
                "layer": li,
                "rules": R,
                "k": k,
                "cited_rules": top,
                "coverage_of_cited": round(float(share[top].sum()), 4),
                "cited_cost": round(cited - base, 4),
                "random_cost_mean": round(float(rand_a.mean() - base), 4),
                "random_cost_p95": round(float(np.percentile(rand_a, 95) - base), 4),
                "random_cost_max": round(float(rand_a.max() - base), 4),
                "causal": bool(cited > np.percentile(rand_a, 95)),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, tok = load_model(a.ckpt)
    codec = make_codec(tok)
    if codec.kind != "bpe":
        raise SystemExit("rule_ablation needs a subword checkpoint")
    T = min(model.cfg.max_len, 256)
    windows = codec.windows(a.windows, T + 1)
    rep = {
        "ckpt": str(a.ckpt),
        "window_tokens": T,
        **ablate(model, windows, np.asarray(codec.bpe.lens), a.k, a.draws),
    }
    text = json.dumps(rep, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
