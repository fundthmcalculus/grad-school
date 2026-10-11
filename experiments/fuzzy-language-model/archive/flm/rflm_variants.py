"""RFLM (fuzzy-delta mixer) variants, registered into ``flm.models.MIXERS`` on import.

fuzzydelta_static : a *fixed recurrence*. The write strength beta is one learned constant per head instead of a function
                    of the input (pair with ``--decay fixed`` for a static forgetting factor too): the recurrence
                    S_t = alpha (S_{t-1} + beta (v_t - S_{t-1} k_t) k_t^T) is then the same linear system at every step;
                    only the key phi(k_t) and the value v_t carry the input.
fuzzydelta_cpread : *consequent propagation* on the read side. Rule r's consequent also speaks for related rules:
                    o_t = sum_r phi'_r(q_t) theta_r with phi' = phi + phi A, A (R x R) shared by the heads, zero-initialised
                    (so the model starts exactly equal to D).
fuzzydelta_cpboth : the same propagation on both sides (the write key too), so the implied kernel stays symmetric.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from flm.models import MIXERS, FuzzyDeltaMixer, ModelConfig


class _StaticBeta(nn.Module):
    """Stands in for ``nn.Linear(d, H)`` in the delta mixer: the same (B, T, H) logits for every input."""

    def __init__(self, H: int):
        super().__init__()
        self.logit = nn.Parameter(
            torch.zeros(H)
        )  # sigmoid(0) = 0.5, the data-dependent gate's init mean

    def forward(self, x):
        return self.logit.expand(x.shape[0], x.shape[1], -1)


class StaticFuzzyDelta(FuzzyDeltaMixer):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.beta = _StaticBeta(self.H)


class _Propagate(FuzzyDeltaMixer):
    both = False

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.prop = nn.Parameter(torch.zeros(self.R, self.R))

    def features(self, x):
        fq, fk, v = super().features(x)
        fq = fq + fq @ self.prop
        if self.both:
            fk = fk + fk @ self.prop
        return fq, fk, v


class PropagateRead(_Propagate):
    both = False


class PropagateBoth(_Propagate):
    both = True


MIXERS["fuzzydelta_static"] = StaticFuzzyDelta
MIXERS["fuzzydelta_cpread"] = PropagateRead
MIXERS["fuzzydelta_cpboth"] = PropagateBoth
