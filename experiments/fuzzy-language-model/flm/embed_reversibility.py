"""How reversible is the shared (tied) embedding?  ``python -m flm.embed_reversibility ckpt ...`` (`PLAN_SHAPE.md` E7).

The full table E (V x d, in token-id order) is rebuilt from the state dict: head rows are ``emb_h``; tail rows are ``tier_up(mu(emb_t))`` (fuzzy tail: ``mu`` softmax, or tanh when the file name says so) or ``tier_up(emb_t)`` (crisp tail); a fuzzy head (``head_proto``) is ``mu(emb_h) @ head_proto.T``.
Reversibility of the embed -> de-embed round trip, with the model's own tied scoring (score of token s for a state h is <h, E[s]>) and h = E[t]: the share of tokens t whose top-1 decode is t, the same weighted by token frequency, the top-10 share and the mean rank of t (the cosine variant is vacuous: a row is always its own best cosine match).
Also the row-norm ratio (a norm-biased decode favours large rows) and the singular-value spectrum (effective rank) of E.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_FREQ = {3072: "outputs/freq_bpe3072.npz", 4096: "outputs/freq_bpe4096.npz"}


def table(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st, V = ck["state"], ck["config"]["vocab_size"]
    freq = np.load(REPO_FREQ[V])["freq"]
    order = np.argsort(-freq, kind="stable")
    NH = st["emb_h.weight"].shape[0]
    head, tail = order[:NH], order[NH:]
    stem = Path(path).name
    tanh = "tanh" in stem
    mu = (lambda x: torch.tanh(x)) if tanh else (lambda x: torch.softmax(x, -1))
    fuzzy_tail = any(n in stem for n in ("e2e", "lr6_wu", "fz_lr"))
    eh = st["emb_h.weight"]
    if "head_proto" in st:
        eh = mu(eh) @ st["head_proto"].T
    et = st["emb_t.weight"]
    et = mu(et) if fuzzy_tail else et
    et = et @ st["tier_up.weight"].T
    E = torch.zeros(V, eh.shape[1])
    E[head], E[tail] = eh, et
    return E.double().numpy(), freq, head, tail


def report(path):
    E, freq, head, tail = table(path)
    V = len(E)
    w = freq / freq.sum()
    G = E @ E.T
    rank_dot = (G > G.diagonal()[:, None]).sum(1)  # tokens scoring higher than t itself
    n = np.linalg.norm(E, axis=1)
    for name, ids in (("all", np.arange(V)), ("head", head), ("tail", tail)):
        r = rank_dot[ids]
        wt = w[ids] / w[ids].sum()
        print(
            f"  {name:5s} dot: top-1 {np.mean(r == 0):.3f} (freq-weighted {np.sum(wt * (r == 0)):.3f}), top-10 {np.mean(r < 10):.3f}, mean rank {np.mean(r):.1f}"
        )
    s = np.linalg.svd(E, compute_uv=False)
    pr = (s**2).sum() ** 2 / (s**4).sum()
    print(
        f"  mean row norm head {n[head].mean():.2f}, tail {n[tail].mean():.2f} (ratio {n[head].mean() / n[tail].mean():.2f}); effective rank of E {pr:.1f} of {E.shape[1]}"
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p)
        report(p)
