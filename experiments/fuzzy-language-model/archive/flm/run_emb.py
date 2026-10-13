"""``python -m flm.run_emb <variant> <train args>``: flm.train with the tied token embedding configured differently (`v28-emb`).

ppmi    : embedding initialised to the bigram PPMI-SVD successor vectors (rms 0.1), everything else random
ppmip   : same with the predecessor vectors (the embedding is also the input representation)
std05   : the random embedding scaled x0.5 (init std 0.05; baseline 0.1)
std20   : the random embedding scaled x2   (init std 0.2)
nowd    : the embedding is exempt from weight decay (baseline: decayed like every matrix)
"""

import sys

import torch

import flm.train as T
from flm.models import TinyLM
from flm.run_conseq import _stats

V = sys.argv[1]
_init = TinyLM.__init__


def _new_init(self, cfg, *a, **k):
    _init(self, cfg, *a, **k)
    w = self.emb.weight.data
    if V in ("ppmi", "ppmip"):
        pred, succ, _ = _stats(cfg.d_model, cfg.vocab_size)
        m = torch.from_numpy(succ if V == "ppmi" else pred).float()
        w.copy_(m * (0.1 / m.pow(2).mean().sqrt()))
    elif V == "std05":
        w.mul_(0.5)
    elif V == "std20":
        w.mul_(2.0)


_AdamW = torch.optim.AdamW


def _adamw(groups, **k):
    if V == "nowd":
        dec, oth = groups[0], groups[1]
        keep = [p for p in dec["params"] if p is not _model[0].emb.weight]
        oth = dict(oth, params=oth["params"] + [_model[0].emb.weight])
        groups = [dict(dec, params=keep), oth]
    return _AdamW(groups, **k)


_model = []
if __name__ == "__main__":
    assert V in ("ppmi", "ppmip", "std05", "std20", "nowd")

    def _init_track(self, cfg, *a, **k):
        _new_init(self, cfg, *a, **k)
        _model.append(self)

    TinyLM.__init__ = _init_track
    torch.optim.AdamW = _adamw
    T.main(sys.argv[2:])
