"""B1 gate (PLAN_FROZEN_HEAD.md section 4.5): a consequent-first (TRIBBLE-order) construction of the blocks under a frozen head.

    python -m flm.tribble_head.construct <donor.pt> [--out JSON] [--eval-chars N] [--build-windows N]

The head (codebook C, F, G, log_tau) comes from a trained donor and is not changed.  Blocks are re-initialised and then
constructed with no gradient steps:

* targets: the residual stream after the whole stack must reach the embedding of the NEXT character, T_t = G(F(C_{y_t})),
  because the output logits are -0.5 ||F^-1 G^-1 h - C_v||^2 / tau^2 (``flm.run_bijective._logits``).  Each block is given an equal
  share of the displacement from h0 = embed(x_t) to T_t (the residual split (l+1)/L).
* antecedents (order-1 TSK FFN, first half of the state): rule centres are k-means inside each next-character class
  (class share of the rules proportional to its frequency; at least one rule per class present); widths are the global
  per-dimension std of the antecedent input (the data init of ``flm.models.init_rules_from_data``).
* consequents: closed-form ridge least squares of the block's residual target on the rule firings.  The order-1 slope
  is fixed to 1 (weight 0, bias 1) during construction, so the consequents are order-0 for this step.
* mixers (the fuzzy delta mixers) stay at their random initialisation.  Their outputs are part of the residual the FFN fits.

Reported: validation bits per character of (a) the same head with random blocks, (b) the constructed model with no training,
(c) unigram and bigram character models fitted on the training slice and scored on the same validation characters.
The gate: (b) beats (a) by at least 1 bit and beats the unigram by at least 0.3 bits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("BIJ", "ci")
os.environ.setdefault("FUZZ_FFN_R", "68")
os.environ.setdefault("FUZZ_FFN_R0", "48")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import flm.run_bijective as RB  # noqa: E402
import flm.train as T  # noqa: E402
from flm.data import VOCAB_SIZE, decode, load  # noqa: E402
from flm.models import ModelConfig, TinyLM, set_widths_from_std  # noqa: E402

HEAD_KEYS = ("codebook", "F", "G", "log_tau")
CTX = 256


def char_classes() -> np.ndarray:
    """Class of each vocabulary id, used to group the next-character targets: 0 lower, 1 upper, 2 digit, 3 space/newline, 4 mark, 5 other (EOT/UNK)."""
    cls = np.zeros(VOCAB_SIZE, dtype=np.int64)
    for i in range(VOCAB_SIZE):
        s = decode(np.array([i]))
        if len(s) != 1:
            cls[i] = 5
        elif s.islower():
            cls[i] = 0
        elif s.isupper():
            cls[i] = 1
        elif s.isdigit():
            cls[i] = 2
        elif s in " \n":
            cls[i] = 3
        elif s.isalnum():
            cls[i] = 5
        else:
            cls[i] = 4
    return cls


def build(donor_path: str, seed: int) -> tuple[TinyLM, dict]:
    """A fresh model with the donor's head loaded and every block freshly initialised with ``seed``."""
    TinyLM.__init__ = RB._new_init
    TinyLM.embed = RB._embed
    TinyLM.logits = RB._logits
    TinyLM.n_params = RB._n_params
    ck = torch.load(donor_path, map_location="cpu", weights_only=False)
    torch.manual_seed(seed)
    model = TinyLM(ModelConfig(**ck["config"]))
    head = {k: v for k, v in ck["state"].items() if k.split(".")[0] in HEAD_KEYS}
    missing, unexpected = model.load_state_dict(head, strict=False)
    assert not unexpected, unexpected
    assert set(k.split(".")[0] for k in head) == set(HEAD_KEYS), head.keys()
    model.eval()
    return model, ck


def firing_raw(ffn, u: torch.Tensor) -> torch.Tensor:
    """Raw firing strengths of a FuzzyFFNRaw on its antecedent input u (N, taps*d), as in its forward pass (no normalisation)."""
    a = torch.exp(-2.0 * ffn.log_width)
    ca = ffn.centers * a
    quad = (u * u) @ a.T - 2.0 * (u @ ca.T) + (ffn.centers * ca).sum(-1)
    return torch.exp(-0.5 * ffn.scale * quad)


def kmeans(u: torch.Tensor, k: int, iters: int, gen: torch.Generator) -> torch.Tensor:
    if u.shape[0] <= k:
        idx = torch.randint(0, u.shape[0], (k,), generator=gen)
        return u[idx].clone()
    c = u[torch.randperm(u.shape[0], generator=gen)[:k]].clone()
    for _ in range(iters):
        d = torch.cdist(u, c)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = u[m].mean(0)
    return c


def allocate(R: int, counts: np.ndarray) -> np.ndarray:
    """Rules per class, proportional to class frequency, at least one for every class present, summing to R."""
    present = counts > 0
    share = np.where(present, counts / counts.sum(), 0.0)
    n = np.where(present, np.maximum(1, np.floor(share * R)), 0).astype(int)
    while n.sum() < R:
        n[np.argmax(np.where(present, share * R - n, -1e9))] += 1
    while n.sum() > R:
        j = np.argmax(np.where(n > 1, n - share * R, -1e9))
        n[j] -= 1
    return n


@torch.no_grad()
def construct(
    model: TinyLM,
    x_ids: torch.Tensor,
    y_ids: torch.Tensor,
    cls: np.ndarray,
    seed: int,
    ridge: float = 1e-2,
) -> dict:
    """x_ids, y_ids: (N, CTX) current and next characters.  Builds the blocks in order and returns per-block diagnostics."""
    gen = torch.Generator().manual_seed(seed)
    L = len(model.blocks)
    h0 = model.embed(x_ids)  # (N, CTX, d)
    tgt = model.embed(y_ids)
    x = h0
    labels = torch.as_tensor(cls[y_ids.numpy()]).reshape(-1)
    diag = {}
    for li, blk in enumerate(model.blocks):
        ffn = (
            blk.ffn
        )  # FuzzyFFNRaw: raw-input Gaussian antecedents (layer 0 reads two tokens: taps=2)
        mid = x + blk.mix(blk.mixer_in(x))
        share = (li + 1) / L
        target = (h0 + share * (tgt - h0)).reshape(-1, h0.shape[-1])
        Y = target - mid.reshape(-1, mid.shape[-1])
        n2 = blk.n2(mid)
        u = ffn.antecedent_input(n2).reshape(-1, ffn.centers.shape[1])
        # antecedents: k-means inside each next-character class
        counts = np.bincount(labels.numpy(), minlength=6).astype(float)
        alloc = allocate(ffn.R, counts)
        centers, row = [], 0
        for c in range(6):
            if alloc[c] == 0:
                continue
            uc = u[labels == c]
            centers.append(kmeans(uc, int(alloc[c]), 10, gen))
        ffn.centers.copy_(torch.cat(centers, 0)[: ffn.R])
        set_widths_from_std(ffn.log_width, u.std(0))
        if ffn.order == 1:  # construction uses order-0 consequents: slope fixed to 1
            ffn.slope.weight.zero_()
            ffn.slope.bias.fill_(1.0)
        # consequents: closed-form ridge on the firing
        phi = firing_raw(ffn, u).double()
        lam = ridge * phi.pow(2).sum(0).mean().item() + 1e-12
        A = phi.T @ phi + lam * torch.eye(ffn.R, dtype=torch.double)
        W = torch.linalg.solve(A, phi.T @ Y.double())
        ffn.conseq.copy_(W.float())
        pred = phi.float() @ ffn.conseq
        resid = Y - pred
        diag[f"block{li}"] = {
            "rules_per_class": alloc.tolist(),
            "ffn_fit_r2": float(1 - resid.pow(2).sum() / Y.pow(2).sum()),
            "firing_active_frac": float((phi > 1e-3 * phi.max()).float().mean()),
        }
        x = mid + ffn(n2)
    return diag


@torch.no_grad()
def windows(
    arr: np.ndarray, n_win: int, offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    seg = np.asarray(arr[offset : offset + n_win * (CTX + 1) + 1], dtype=np.int64)
    x = np.stack([seg[i * (CTX + 1) : i * (CTX + 1) + CTX] for i in range(n_win)])
    y = np.stack(
        [seg[i * (CTX + 1) + 1 : i * (CTX + 1) + CTX + 1] for i in range(n_win)]
    )
    return torch.from_numpy(x), torch.from_numpy(y)


def bits_ngram(train: np.ndarray, val: np.ndarray, order: int) -> float:
    """Add-one-smoothed unigram (order 1) or bigram (order 2) character model fitted on ``train``, bits per character on ``val``."""
    V = VOCAB_SIZE
    tr = np.asarray(train, dtype=np.int64)
    va = np.asarray(val, dtype=np.int64)
    if order == 1:
        cnt = np.bincount(tr, minlength=V).astype(float) + 1.0
        p = cnt / cnt.sum()
        return float(-np.log2(p[va]).mean())
    cnt = np.ones((V, V))
    np.add.at(cnt, (tr[:-1], tr[1:]), 1.0)
    p = cnt / cnt.sum(1, keepdims=True)
    return float(-np.log2(p[va[:-1], va[1:]]).mean())


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("donor")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-chars", type=int, default=200_000)
    ap.add_argument("--build-windows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--zero-mix",
        action="store_true",
        help="zero each fuzzy mixer's output projection before construction (blocks = FFN only)",
    )
    a = ap.parse_args(argv)
    cls = char_classes()
    train = load("train")
    val = load("valid")
    n_eval = a.eval_chars // (CTX + 1)
    xb, yb = windows(train, a.build_windows, offset=0)

    # (a) random blocks, the donor head
    rand, ck = build(a.donor, a.seed)
    bpc_random = T.evaluate(rand, val, CTX, a.eval_chars, batch=64)

    # (b) constructed blocks
    cons, _ = build(a.donor, a.seed)
    if a.zero_mix:
        with torch.no_grad():
            for blk in cons.blocks:
                for p_ in blk.mix.out.parameters():
                    p_.zero_()
    diag = construct(cons, xb, yb, cls, a.seed)
    bpc_cons = T.evaluate(cons, val, CTX, a.eval_chars, batch=64)

    # (c) n-gram references on the same validation characters
    n_chars = n_eval * (CTX + 1)
    v = np.asarray(val[:n_chars], dtype=np.int64)
    uni = bits_ngram(np.asarray(train[:20_000_000], dtype=np.int64), v, 1)
    big = bits_ngram(np.asarray(train[:20_000_000], dtype=np.int64), v, 2)

    rec = {
        "donor": a.donor,
        "donor_val_bpc": ck.get("val_bpc"),
        "seed": a.seed,
        "zero_mix": a.zero_mix,
        "build_windows": a.build_windows,
        "eval_chars": n_chars,
        "bpc_random_blocks": bpc_random,
        "bpc_constructed_no_training": bpc_cons,
        "bpc_unigram": uni,
        "bpc_bigram": big,
        "gate_beats_random_by_1_bit": bpc_random - bpc_cons >= 1.0,
        "gate_beats_unigram_by_0_3": uni - bpc_cons >= 0.3,
        "construction": diag,
    }
    print(json.dumps(rec, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    main(sys.argv[1:])
