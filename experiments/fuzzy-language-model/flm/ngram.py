"""Character n-gram statistics: the crisp TSK rule base that the fuzzy rules start from.

A count-based n-gram language model is a zero-order TSK system with crisp antecedents:
"IF the last k characters are g THEN the next-character distribution is p(.|g)". The
fuzzy rules of the FFN are initialised from these (``flm.models.init_rules_from_ngrams``).

    .venv/bin/python -m flm.ngram            # reference BPC of order-0..3 count models

``context order k`` = number of preceding characters conditioned on: k=0 unigram, k=1
bigram (c_t -> c_{t+1}), k=2 trigram (c_{t-1} c_t -> c_{t+1}), k=3 four-gram.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from flm.data import VOCAB_SIZE, load
from flm.models import TSKFFN, TinyLM, set_widths_from_std

V = VOCAB_SIZE
LN2 = math.log(2.0)


def count_tables(
    n_chars: int = 100_000_000,
    max_k: int = 3,
    chunk: int = 10_000_000,
    train: np.ndarray | None = None,
) -> list[np.ndarray]:
    """counts[k] has shape (V,)*(k+1): counts[k][c_{t-k+1..t}, c_{t+1}] over the first
    ``n_chars`` training characters (chunked so peak memory stays small)."""
    train = load("train") if train is None else train
    tabs = [np.zeros((V,) * (k + 1), dtype=np.int64) for k in range(max_k + 1)]
    for start in range(0, n_chars, chunk):
        a = np.asarray(train[start : start + chunk + max_k], dtype=np.int64)
        n = len(a) - max_k
        if n <= 0:
            break
        for k in range(max_k + 1):
            # window of k+1 chars ending at offset max_k, so every order sees the same targets
            idx = np.zeros(n, dtype=np.int64)
            for j in range(k + 1):
                idx = idx * V + a[max_k - k + j : max_k - k + j + n]
            tabs[k] += np.bincount(idx, minlength=V ** (k + 1)).reshape(tabs[k].shape)
    return tabs


def conditionals(tabs: list[np.ndarray], beta: float = 2.0) -> list[np.ndarray]:
    """Smoothed p_k(y | context) of shape (V,)*k + (V,), with Dirichlet back-off:
    p_k(y|g) = (n(g,y) + beta * p_{k-1}(y|g')) / (n(g) + beta), g' = g without its oldest char.
    """
    probs = []
    for k, t in enumerate(tabs):
        t = t.astype(np.float64)
        if k == 0:
            p = (t + 1.0) / (t.sum() + V)  # add-one over the vocabulary
        else:
            n = t.sum(-1, keepdims=True)
            lower = probs[
                k - 1
            ]  # (V,)*(k-1) + (V,): broadcast over the oldest context char
            lower = np.broadcast_to(lower[None], t.shape)
            p = (t + beta * lower) / (n + beta)
        probs.append(p)
    return probs


def window_bpc(probs: list[np.ndarray], k: int, windows: np.ndarray) -> float:
    """Mean bits/char of the order-k model on (n_win, ctx+1) windows, scored exactly as the
    neural eval does: targets are positions 1..ctx, and a position with fewer than k
    preceding characters in its window falls back to the lower order available there."""
    n_win, L = windows.shape
    total, count = 0.0, 0
    for i in range(1, L):
        kk = min(k, i)
        ctx = windows[:, i - kk : i]
        if kk == 0:
            p = np.broadcast_to(probs[0], (n_win, V))
        else:
            p = probs[kk][tuple(ctx[:, j] for j in range(kk))]  # (n_win, V)
        total += -np.log(p[np.arange(n_win), windows[:, i]]).sum()
        count += n_win
    return total / count / LN2


def validation_windows(eval_chars: int = 1_000_000, ctx: int = 256) -> np.ndarray:
    val = load("valid")
    n_win = eval_chars // (ctx + 1)
    return np.asarray(val[: n_win * (ctx + 1)], dtype=np.int64).reshape(n_win, ctx + 1)


KAPPAS = (0.1, 0.2, 0.4, 0.8)  # rule width as a fraction of the antecedent's global std
ALPHAS = (1.0, 4.0, 16.0, 64.0)  # scale of the decoded consequents
GAINS = (
    1.0,
    4.0,
    16.0,
    32.0,
    64.0,
)  # final-norm gain: the output temperature of the tied readout
ELSE_WIDTH = 3.0  # the background rule is this many times wider than a context rule


def _bpc(model: TinyLM, windows: torch.Tensor) -> float:
    logits = model(windows[:, :-1])
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), windows[:, 1:].reshape(-1)
    )
    return loss.item() / LN2


@torch.no_grad()
def init_rules_from_ngrams(
    model: TinyLM,
    stats_windows: torch.Tensor,
    calib_windows: torch.Tensor,
    k: int,
    tabs: list[np.ndarray],
    probs: list[np.ndarray],
    mode: str = "full",
) -> dict:
    """Make layer 0's TSK feed-forward a fuzzy version of the order-k n-gram model.

    The n_ctx = min(R - 1, V**k) most frequent length-k contexts g each get a rule: its
    antecedent centre is the mean of the rule's own antecedent input u over the sample
    positions whose context is g (placed empirically through the real forward pass, so it is
    right whatever the layer sees), and its consequent a_r is decoded through the pseudo-
    inverse of the tied embedding so that the logit lens of a_r reads log p(next | g). Rule
    R - 1 is a wide background rule falling back to the unigram. Any rules in between (when
    R - 1 > V**k) keep their random centres and get a zero consequent, so they are inert.
    Rule width (kappa), consequent scale (alpha) and the final-norm gain are calibrated on
    the init BPC of ``calib_windows``.

    The tied embedding is first aligned to the n-gram statistics: its columns become the
    top-d principal directions of the frequency-weighted conditional log-probs, so the d-dim
    readout can represent n-gram distributions at all (a random d=16 embedding cannot: its
    rank-16 readout of the bigram log-probs scores 7.5 BPC, the aligned one 3.57).

    A first-order feed-forward (ffn_order = 1, output = firing * (slope . x) * a_r) is made
    zero-order-equivalent at init (slope weight 0, bias 1) so the decoded consequents mean
    what they say; the first-order terms then grow from zero under training.

    ``mode``: "emb" = aligned embedding only; "placement" = that plus rule placement and
    widths, with consequents and output gain left at their random init; "full" = everything.
    Returns diagnostics: alpha, kappa, gain, init BPC, coverage, rule accuracy.
    """
    blk = model.blocks[0]
    ffn = blk.ffn
    assert (
        isinstance(ffn, TSKFFN) and k in (1, 2) and mode in ("emb", "placement", "full")
    )
    R, da = ffn.R, ffn.da
    old_conseq = ffn.conseq.detach().clone()

    # aligned embedding: top-d eigenvectors of G = sum_g n_g L_g^T L_g / sum_g n_g
    Lall = np.log(probs[k].reshape(-1, V))
    Lall = Lall - Lall.mean(-1, keepdims=True)
    nall = tabs[k].reshape(-1, V).sum(-1).astype(np.float64)
    gram = (Lall * nall[:, None]).T @ Lall / nall.sum()
    _, Q = np.linalg.eigh(gram)
    d = model.emb.weight.shape[1]
    model.emb.weight.copy_(torch.from_numpy(Q[:, ::-1][:, :d].copy()).float())
    if mode == "emb":
        return {"k": k, "mode": mode, "init_bpc": round(_bpc(model, calib_windows), 4)}
    if getattr(ffn, "order", 0) == 1:
        ffn.slope.weight.zero_()
        ffn.slope.bias.fill_(1.0)

    idx = stats_windows[:, :-1]
    x = model.emb(idx)
    x = x + blk.mix(blk.mixer_in(x))
    u = ffn.ante(blk.n2(x))  # (B, T, da)
    ctx = (
        idx
        if k == 1
        else torch.cat(
            [torch.full_like(idx[:, :1], -1), idx[:, :-1] * V + idx[:, 1:]], 1
        )
    )

    n_ctx = min(R - 1, V**k)
    ctx_counts = torch.from_numpy(tabs[k].reshape(-1, V).sum(-1))
    top = torch.topk(ctx_counts, n_ctx).indices
    rule_of_ctx = torch.full((V**k,), -1, dtype=torch.long)
    rule_of_ctx[top] = torch.arange(n_ctx)
    pos_rule = torch.where(
        ctx >= 0, rule_of_ctx[ctx.clamp(min=0)], torch.full_like(ctx, -1)
    )
    u_flat, r_flat = u.reshape(-1, da), pos_rule.reshape(-1)
    valid = r_flat >= 0
    sums = torch.zeros(R, da).index_add_(0, r_flat[valid], u_flat[valid])
    n = torch.bincount(r_flat[valid], minlength=R).float()
    centers = sums / n.clamp(min=1)[:, None]
    for r in (
        torch.nonzero(n[:n_ctx] < 3).flatten().tolist()
    ):  # a context the sample barely saw
        centers[r] = u_flat[torch.randint(len(u_flat), (1,))][0]
    new_centers = (
        ffn.centers.detach().clone()
    )  # rules between n_ctx and R-1 keep their init
    new_centers[:n_ctx] = centers[:n_ctx]
    new_centers[R - 1] = u_flat.mean(0)  # the background rule sits at the overall mean
    ffn.centers.copy_(new_centers)
    std = u_flat.std(0)

    logp = np.log(probs[k].reshape(-1, V)[top.numpy()])  # (n_ctx, V)
    uni = np.log(probs[0])[None]
    target = np.concatenate(
        [logp, uni], 0
    )  # last row: unigram, for the background rule
    target = torch.from_numpy(target - target.mean(-1, keepdims=True)).to(
        model.emb.weight.dtype
    )
    coef = (
        target @ torch.linalg.pinv(model.emb.weight).T
    )  # (n_ctx + 1, d): E a_r ~ log p_r
    base = torch.zeros(R, d, dtype=coef.dtype)
    base[:n_ctx] = coef[:n_ctx]
    base[R - 1] = coef[n_ctx]
    old_gain = model.norm.weight.detach().clone()

    def set_widths(kappa):
        set_widths_from_std(ffn.log_width, std)
        ffn.log_width.add_(math.log(kappa))
        if (
            ffn.log_width.shape[0] == R
        ):  # per-rule widths: make the background rule wide
            ffn.log_width[R - 1] += math.log(ELSE_WIDTH)

    best = (float("inf"), None, None, None)
    for kappa in KAPPAS:
        set_widths(kappa)
        for alpha in ALPHAS:
            ffn.conseq.copy_(alpha * base)
            for gain in GAINS:
                model.norm.weight.fill_(gain)
                bpc = _bpc(model, calib_windows)
                if bpc < best[0]:
                    best = (bpc, kappa, alpha, gain)
    calibrated_bpc, kappa, alpha, gain = best
    set_widths(kappa)
    ffn.conseq.copy_(alpha * base)
    model.norm.weight.fill_(gain)
    if (
        mode == "placement"
    ):  # placement and widths only: also undo the readout calibration
        ffn.conseq.copy_(old_conseq)
        model.norm.weight.copy_(old_gain)
    init_bpc = _bpc(model, calib_windows)

    # manipulation check: does the winning rule at a covered position have that position's context?
    fire = ffn.firing(blk.n2(x)).reshape(-1, R)
    rule_acc = (fire.argmax(-1)[valid] == r_flat[valid]).float().mean().item()
    return {
        "k": k,
        "kappa": kappa,
        "alpha": alpha,
        "gain": gain,
        "init_bpc": round(init_bpc, 4),
        "calibrated_init_bpc": round(calibrated_bpc, 4),
        "coverage": round(valid.float().mean().item(), 4),
        "rule_acc": round(rule_acc, 4),
        "rules_winning": int((torch.bincount(fire.argmax(-1), minlength=R) > 0).sum()),
        "n_context_rules": n_ctx,
        "mode": mode,
    }


def ngram_init_for_model(
    model: TinyLM, k: int, seed: int, mode: str = "full", n_chars: int = 50_000_000
) -> dict:
    """Counts + init for a model, with its own RNG so the training data order is untouched."""
    tabs = count_tables(n_chars=n_chars, max_k=k)
    probs = conditionals(tabs)
    train = load("train")
    rng = np.random.default_rng(seed + 10_000)
    ctx = model.cfg.max_len

    def sample(n):
        starts = rng.integers(0, len(train) - ctx - 2, n)
        return torch.from_numpy(
            np.stack([train[i : i + ctx + 1] for i in starts]).astype(np.int64)
        )

    return init_rules_from_ngrams(model, sample(128), sample(64), k, tabs, probs, mode)


def main() -> None:
    import time

    t = time.time()
    tabs = count_tables()
    probs = conditionals(tabs)
    w = validation_windows()
    print(f"counted {tabs[0].sum():,} chars in {time.time() - t:.0f}s")
    for k, name in enumerate(["unigram", "bigram", "trigram", "four-gram"]):
        n_ctx = int((tabs[k].sum(-1) > 0).sum()) if k else 1
        print(
            f"order {k} ({name}): {window_bpc(probs, k, w):.4f} bits/char   [{n_ctx:,} observed contexts]"
        )


if __name__ == "__main__":
    main()
