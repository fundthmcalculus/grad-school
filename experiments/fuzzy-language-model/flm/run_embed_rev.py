"""All-fuzzy training with the reversibility terms of `PLAN_SHAPE.md` E7 added to the loss (``train.py`` and ``models.py`` untouched; the logged training loss is the plain cross-entropy, the extra terms only add gradient).

``FUZZ_ORTHO=lam``  adds lam * ||W^T W - I||^2 for the prototype up-projections (the tail's ``tier_up``, and ``head_up`` with ``FUZZ_TOK_HEADRANK``): orthonormal prototypes make the fuzzy-set membership vector exactly recoverable from the embedding (defuzzification inverts fuzzification).
``FUZZ_CYCLE=w``    adds w * CE(embed(t) . E^T, t) over up to 512 distinct target tokens of the batch (E = the whole embedding table): the shared table is trained to decode its own embedding.
The head rank itself is ``FUZZ_TOK_HEADRANK=K`` (flm/fuzzy_tokens.py); an arm with only that variable runs through ``flm.run_allfuzzy``.
``python -m flm.run_embed_rev <train args>`` (environment as for ``flm.run_allfuzzy``).
"""

import os
import sys

import torch
import torch.nn.functional as F

import flm.fuzzy_tokens as FT
import flm.run_allfuzzy as RA
import flm.run_tiered as R
import flm.train as T
from flm.models import TinyLM

ORTHO = float(os.environ.get("FUZZ_ORTHO", "0"))
SCALED = (
    os.environ.get("FUZZ_ORTHO_SCALED") == "1"
)  # orthonormal up to one learned global scale c: ||W^T W - c I||^2
CYCLE = float(os.environ.get("FUZZ_CYCLE", "0"))
holder = {}
real_init = RA._init
orig_ce = F.cross_entropy


def init(self, *a, **k):
    real_init(self, *a, **k)
    if SCALED:
        self.ortho_c = torch.nn.Parameter(torch.ones(()))
    holder["m"] = self


def penalty(m, target):
    pen = torch.zeros((), device=target.device)
    if ORTHO:
        for name in ("tier_up", "head_up"):
            mod = getattr(m, name, None)
            if mod is not None:
                g = mod.weight.T @ mod.weight
                c = m.ortho_c if SCALED else 1.0
                eye = c * torch.eye(g.shape[0], device=g.device)
                pen = pen + ORTHO * ((g - eye) ** 2).sum()
    if CYCLE:
        t = torch.unique(target.reshape(-1))
        if len(t) > 512:
            t = t[torch.randperm(len(t), device=t.device)[:512]]
        table = m.embed(torch.arange(m.cfg.vocab_size, device=t.device))
        pen = pen + CYCLE * orig_ce(m.embed(t) @ table.T, t)
    return pen


def ce(logits, target, *a, **k):
    loss = orig_ce(logits, target, *a, **k)
    m = holder.get("m")
    if (
        m is not None
        and m.training
        and torch.is_grad_enabled()
        and not a
        and k.get("reduction", "mean") == "mean"
    ):
        aux = penalty(m, target)
        if os.environ.get("FUZZ_DEBUG") and holder.get("n", 0) < 3:
            holder["n"] = holder.get("n", 0) + 1
            print(
                "aux term", float(aux), "requires_grad", aux.requires_grad, flush=True
            )
        loss = loss + (
            aux - aux.detach()
        )  # the value stays the plain loss; the gradient includes the term
    return loss


if __name__ == "__main__":
    F.cross_entropy = ce
    TinyLM.__init__ = init
    TinyLM.embed = FT._embed if RA._emb else R._embed
    TinyLM.logits = FT._logits if RA._emb else R._logits
    TinyLM.n_params = R._n_params
    T.main(sys.argv[1:])
