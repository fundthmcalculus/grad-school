"""Trivial baselines on the Cosmopedia BPE1024 stream (`PLAN_CORPUS.md` C0): unigram and interpolated bigram/trigram token models fit on the
first N training tokens, scored in bits per CHARACTER on the first 1M characters of the validation stream.  Run with
GRAD_SCHOOL_DATA=<repo>/data/cosmopedia."""

import sys

import numpy as np

from flm.bpe import BPE

N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 300_000_000
bp = BPE(1024)
V = 1024
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
tri = (
    np.bincount((tr[:-2] * V + tr[1:-1]) * V + tr[2:], minlength=V**3)
    .reshape(V, V, V)
    .astype(np.float32)
)


def bpc(p):
    return float(-np.log2(p).sum() / chars)


p1 = uni[va[1:]]
rb = big.sum(1, keepdims=True)
lam = 0.9
p2 = (
    lam * big[va[:-1], va[1:]] / np.maximum(rb[va[:-1], 0], 1) + (1 - lam) * uni[va[1:]]
)
r3 = tri.sum(2)
c3 = tri[va[:-2], va[1:-1], va[2:]]
lam3 = 0.7
p3 = lam3 * c3 / np.maximum(r3[va[:-2], va[1:-1]], 1) + (1 - lam3) * p2[1:]
print(
    f"unigram {bpc(p1):.4f}  bigram {bpc(p2):.4f}  trigram {bpc(p3) * chars / lens[va[2:]].sum():.4f} (BPC; {N:,} train tokens, {chars:,} val chars)"
)
