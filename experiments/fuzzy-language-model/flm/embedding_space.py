"""Could one fixed character dictionary serve every model?  Compare the character vectors that trained models learn.

    BIJ=ci FUZZ_FFN_R=156 FUZZ_FFN_R0=109 python -m flm.embedding_space dump LABEL CKPT.pt     # writes outputs/embspace/LABEL.npz (codebook C, and for ci the meaning-space state G(F(C)))
    python -m flm.embedding_space ppmi LABEL VALID.bin D                                          # a data-derived dictionary (PPMI of characters within 2 positions, SVD to D dimensions)
    python -m flm.embedding_space compare A B [C ...]                                             # every pair: how alike the two spaces are, up to rotation and scale

Spaces compared on the 98 symbols: (1) distance-matrix correlation (Spearman between the 4,753 pairwise distances; blind to rotation and scale), (2) orthogonal Procrustes (centre, scale to unit norm, best rotation; reported as the share of variance left unexplained), (3) linear CKA, (4) the average overlap of each symbol's 5 nearest neighbours.  Also, per space, how well the character classes (lower case, upper case, digit, space and newline, punctuation) separate: mean between-class over mean within-class distance.  A random matrix scores about 0 on (1) and (4).
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path("outputs/embspace")
VOCAB = 98


def symbol_class(i):
    if i in (0, 1):
        return "special"
    if i == 2:
        return "space/newline"
    c = chr(i - 3 + 32)
    if c == " ":
        return "space/newline"
    if c.islower():
        return "lower"
    if c.isupper():
        return "upper"
    if c.isdigit():
        return "digit"
    return "punct"


CLASSES = np.array([symbol_class(i) for i in range(VOCAB)])


def pdist(a):
    d = a[:, None] - a[None]
    return np.sqrt((d * d).sum(-1))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def rsa(a, b):
    iu = np.triu_indices(len(a), 1)
    return spearman(pdist(a)[iu], pdist(b)[iu])


def procrustes(a, b):
    """Share of variance of b left unexplained by the best rotation of a (both centred and scaled; dimensions padded to match)."""
    a, b = a - a.mean(0), b - b.mean(0)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    k = max(a.shape[1], b.shape[1])
    a = np.pad(a, ((0, 0), (0, k - a.shape[1])))
    b = np.pad(b, ((0, 0), (0, k - b.shape[1])))
    u, s, vt = np.linalg.svd(a.T @ b)
    return float(1.0 - s.sum())  # unit-norm matrices: |a R - b|^2 / 2


def cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return float(
        np.linalg.norm(a.T @ b) ** 2
        / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b))
    )


def knn(a, k=5):
    d = pdist(a)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, 1)[:, :k]


def knn_overlap(a, b, k=5):
    na, nb = knn(a, k), knn(b, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(na, nb)]))


def class_separation(a):
    d = pdist(a)
    same = CLASSES[:, None] == CLASSES[None]
    off = ~np.eye(len(a), dtype=bool)
    keep = CLASSES != "special"
    m = keep[:, None] & keep[None] & off
    return float(d[m & ~same].mean() / d[m & same].mean())


def dump(label, ckpt):
    import torch

    import flm.explain_bijective as E

    OUT.mkdir(parents=True, exist_ok=True)
    m = E.build(ckpt, "cpu")
    with torch.no_grad():
        if E.BIJ == "tied":
            arrs = dict(C=m.emb.weight.numpy())
        else:
            c = m.codebook.detach()
            arrs = dict(C=c.numpy())
            if E.BIJ == "ci":
                arrs["Z"] = m.G(m.F(c)[0]).numpy()
            elif E.BIJ in ("c", "z"):
                arrs["Z"] = m.F(c)[0].numpy()
    np.savez(OUT / f"{label}.npz", **arrs)
    print(label, {k: v.shape for k, v in arrs.items()})


def ppmi(label, valid, dim, window=2, n=50_000_000):
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.fromfile(valid, dtype=np.uint8, count=n).astype(np.int64)
    co = np.zeros((VOCAB, VOCAB))
    for lag in range(1, window + 1):
        np.add.at(co, (x[:-lag], x[lag:]), 1)
    co = co + co.T
    p = co / co.sum()
    pm = p.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p / (pm @ pm.T))
    pmi = np.where(np.isfinite(pmi), np.maximum(pmi, 0.0), 0.0)
    u, s, _ = np.linalg.svd(pmi)
    e = (u * np.sqrt(s))[:, : int(dim)]
    np.savez(OUT / f"{label}.npz", C=e)
    print(label, e.shape)


def compare(labels):
    sp = {l: np.load(OUT / f"{l}.npz") for l in labels}
    keys = [(l, k) for l in labels for k in sp[l].files]
    print("class separation (between / within):")
    for l, k in keys:
        print(f"  {l}.{k}: {class_separation(sp[l][k]):.2f}")
    rng = np.random.default_rng(0)
    r1, r2 = rng.normal(size=(VOCAB, 56)), rng.normal(size=(VOCAB, 56))
    print(
        f"random vs random: rsa {rsa(r1, r2):+.3f}  procrustes-unexplained {procrustes(r1, r2):.3f}  cka {cka(r1, r2):.3f}  knn {knn_overlap(r1, r2):.3f}"
    )
    print("pairs (C = codebook, Z = meaning-space state):")
    for (la, ka), (lb, kb) in combinations(keys, 2):
        if la == lb:
            continue
        a, b = sp[la][ka], sp[lb][kb]
        print(
            f"  {la}.{ka} vs {lb}.{kb}: rsa {rsa(a, b):+.3f}  procrustes-unexplained {procrustes(a, b):.3f}  cka {cka(a, b):.3f}  knn5 {knn_overlap(a, b):.3f}"
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1]
    if mode == "dump":
        dump(sys.argv[2], sys.argv[3])
    elif mode == "ppmi":
        ppmi(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        compare(sys.argv[2:])
