"""``python -m flm.run_rep {unlikelihood,weight} <train args>``: flm.train with a repetition-aware training loss (PLAN_NEXT.md item 4).

unlikelihood : CE + lam * mean_t sum_{c in last K input tokens, c != target} -log(1 - p_t(c))   (Welleck et al.)
weight       : CE at positions whose target completes a trigram already seen earlier in the window is weighted 1 + lam
Only the training loss changes; validation (flm.train.evaluate, no grad) is the plain CE.  lam = REP_LAMBDA (default 0.5), K = REP_K (32).
"""

import os
import sys

import torch
import torch.nn.functional as F0

import flm.train as T
from flm.models import TinyLM

MODE = sys.argv[1]
LAM = float(os.environ.get("REP_LAMBDA", "0.5"))
K = int(os.environ.get("REP_K", "32"))
_last = {}
_fwd = TinyLM.forward


def _forward(self, idx, *a, **k):
    _last["x"] = idx
    return _fwd(self, idx, *a, **k)


def repeat_mask(x, y):
    """(B, T) bool: the trigram (x[t-1], x[t], y[t]) already occurred at some s < t."""
    p0, p1, p2 = x[:, :-1], x[:, 1:], y[:, 1:]
    eq = (
        (p0[:, :, None] == p0[:, None, :])
        & (p1[:, :, None] == p1[:, None, :])
        & (p2[:, :, None] == p2[:, None, :])
    )
    rep = torch.tril(eq, -1).any(-1)
    return torch.cat([torch.zeros_like(rep[:, :1]), rep], 1)


def _ce(logits, target, *a, **k):
    ce = F0.cross_entropy(logits, target, *a, **k)
    x = _last.get("x")
    if not logits.requires_grad or x is None or k:
        return ce
    B, Tn = x.shape
    V = logits.shape[-1]
    lg = logits.reshape(B, Tn, V)
    y = target.reshape(B, Tn)
    if MODE == "weight":
        per = F0.cross_entropy(logits, target, reduction="none").reshape(B, Tn)
        w = 1.0 + LAM * repeat_mask(x, y).float()
        return (per * w).mean()
    # unlikelihood over the last K input tokens (including the current one)
    xp = F0.pad(x, (K - 1, 0), value=-1)
    win = xp.unfold(1, K, 1)  # (B, T, K)
    m = torch.zeros(B, Tn, V + 1, device=x.device)
    m.scatter_(2, (win + 1).clamp(min=0) * (win >= 0), 1.0)  # slot 0 collects padding
    m[..., 0] = 0
    m = m[..., 1:]
    m.scatter_(2, y[..., None], 0.0)
    p = torch.softmax(lg.float(), -1)
    ul = -(torch.log1p(-p.clamp(max=1 - 1e-6)) * m).sum(-1).mean()
    return ce + LAM * ul


class _F:
    def __getattr__(self, n):
        return getattr(F0, n)

    cross_entropy = staticmethod(_ce)


if __name__ == "__main__":
    assert MODE in ("unlikelihood", "weight")
    TinyLM.forward = _forward
    T.F = _F()
    T.main(sys.argv[2:])
