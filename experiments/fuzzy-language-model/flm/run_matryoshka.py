"""``python -m flm.run_matryoshka <train args>``: flm.train with a Matryoshka (nested-prefix) tied-output loss (`v34-mrl`).

Training loss = sum_k w_k CE(h[:, :k] @ E[:, :k].T) over the prefixes in MRL_KS (weights MRL_W, normalised to sum 1; the last
prefix is the full width).  Evaluation and inference use the full vector, so the model has exactly the reference parameters.
PLAN_MATRYOSHKA.md M1/M2.  train.py / models.py are untouched: the final state is stashed in TinyLM.logits and flm.train's
`F` is shimmed so the training-mode (mean) cross-entropy is replaced by the nested sum; evaluation (sum) is unchanged.
"""

import os

import torch.nn.functional as F

import flm.train as T
from flm.models import TinyLM

KS = [int(x) for x in os.environ.get("MRL_KS", "8,16,32,48").split(",")]
W = [float(x) for x in os.environ.get("MRL_W", "1,1,1,1").split(",")]
assert len(KS) == len(W)
_logits = TinyLM.logits
_state = {}


def _new_logits(self, h):
    h = self.norm(h)
    if self.training:
        _state["h"], _state["emb"] = h, self.emb.weight
    return h @ self.emb.weight.T


class _F:
    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def cross_entropy(logits, target, *a, **k):
        if k.get("reduction", "mean") != "mean" or "h" not in _state:
            return F.cross_entropy(logits, target, *a, **k)
        h, E = _state.pop("h"), _state["emb"]
        h = h.reshape(-1, h.shape[-1])
        tot = 0.0
        for kk, w in zip(KS, W):
            tot = tot + w * F.cross_entropy(h[:, :kk] @ E[:, :kk].T, target)
        return tot / sum(W)


if __name__ == "__main__":
    TinyLM.logits = _new_logits
    T.F = _F()
    T.main()
