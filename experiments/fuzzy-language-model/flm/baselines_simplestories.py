"""Trivial baselines on the SimpleStories BPE3072 stream (`PLAN_NEXTROUND.md` C0): unigram, interpolated bigram and trigram token models fit on the first N training tokens, scored in bits per CHARACTER on the first 1M characters of the validation stream
(the estimator of `flm.corpus_baselines`, with a sparse trigram table because 3072^3 does not fit in memory).  Run with GRAD_SCHOOL_DATA=<repo>/data/simplestories.

    python -m flm.baselines_simplestories [N]
"""

import sys

import numpy as np

from flm.bpe import BPE

V = 3072
N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 170_000_000
bp = BPE(V)
tr = np.asarray(bp.load("train")[:N]).astype(np.int64)
va_all = np.asarray(bp.load("valid"))
lens = np.asarray(bp.lens).astype(np.int64)
cum = np.cumsum(lens[va_all])
va = va_all[: int(np.searchsorted(cum, 1_000_000)) + 1].astype(np.int64)
chars = lens[va[1:]].sum()

uni = np.bincount(tr, minlength=V) + 0.5
uni = uni / uni.sum()
big = (
    np.bincount(tr[:-1] * V + tr[1:], minlength=V * V).reshape(V, V).astype(np.float64)
)


def lookup(keys_train, keys_query):
    u, c = np.unique(keys_train, return_counts=True)
    i = np.clip(np.searchsorted(u, keys_query), 0, len(u) - 1)
    return np.where(u[i] == keys_query, c[i], 0).astype(np.float64)


tri_keys = (tr[:-2] * V + tr[1:-1]) * V + tr[2:]
ctx_keys = tr[:-2] * V + tr[1:-1]


def bpc(p):
    return float(-np.log2(p).sum() / chars)


p1 = uni[va[1:]]
rb = big.sum(1, keepdims=True)
lam = 0.9
p2 = (
    lam * big[va[:-1], va[1:]] / np.maximum(rb[va[:-1], 0], 1) + (1 - lam) * uni[va[1:]]
)
c3 = lookup(tri_keys, (va[:-2] * V + va[1:-1]) * V + va[2:])
r3 = lookup(ctx_keys, va[:-2] * V + va[1:-1])
lam3 = 0.7
p3 = lam3 * c3 / np.maximum(r3, 1) + (1 - lam3) * p2[1:]
print(
    f"unigram {bpc(p1):.4f}  bigram {bpc(p2):.4f}  trigram {bpc(p3) * chars / lens[va[2:]].sum():.4f} (BPC; {N:,} train tokens, {chars:,} val chars)"
)
