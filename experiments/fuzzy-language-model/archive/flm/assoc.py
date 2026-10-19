"""Token statistics for clustered / aligned embeddings: unigram frequency and bigram association (PPMI-SVD)."""

from __future__ import annotations

import numpy as np


def bigram_counts(tokens: np.ndarray, V: int) -> np.ndarray:
    t = np.asarray(tokens, dtype=np.int64)
    return (
        np.bincount(t[:-1] * V + t[1:], minlength=V * V)
        .reshape(V, V)
        .astype(np.float64)
    )


def ppmi_svd(C: np.ndarray, k: int, alpha: float = 0.75):
    """Return (L, R): k-dim vectors of tokens as predecessors (rows of C) and as successors (columns of C).

    PPMI with context smoothing (alpha), then a truncated SVD; L = U sqrt(S), R = V sqrt(S).
    """
    tot = C.sum()
    pw = C.sum(1, keepdims=True) / tot
    pc = C.sum(0) ** alpha
    pc = pc[None, :] / pc.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / tot) / (pw * pc))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0), neginf=0.0, posinf=0.0)
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    s = np.sqrt(S[:k])
    return U[:, :k] * s, Vt[:k].T * s


def equal_mass_groups(freq: np.ndarray, R: int) -> np.ndarray:
    """Group ids by frequency rank so every group holds about the same probability mass (rare tokens are lumped)."""
    order = np.argsort(-freq, kind="stable")
    cum = np.cumsum(freq[order]) / max(freq.sum(), 1)
    g = np.minimum((cum * R - 1e-9).astype(int), R - 1)
    out = np.zeros(len(freq), dtype=np.int64)
    out[order] = g
    return out
