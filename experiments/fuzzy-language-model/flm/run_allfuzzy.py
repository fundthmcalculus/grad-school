"""flm.train with fuzzy replacements for non-fuzzy parts (`python -m flm.run_allfuzzy <train args>`; env FUZZ_NORM = fuzzy|scale, FUZZ_GATE = 1, FUZZ_FFN = raw|raw0 with FUZZ_FFN_R rules, FUZZ_PROJ = tsk0|tskl, FUZZ_FFN_TAPS / FUZZ_FFN_R0 for the layer-0 raw local rules, FUZZ_INIT=data, FUZZ_PROJ_RANKS=qkv,out (default 32,20), FUZZ_EMB=tail|all: fuzzy token tables of flm/fuzzy_tokens.py).

Composes with the tiered embedding (TIER_NH / TIER_E / TIER_FREQ as for flm.run_tiered).  train.py and models.py are untouched.
"""

import os
import sys

import flm.fuzzy_tokens as FT
import flm.run_tiered as R
import flm.train as T
from flm.fuzzy_parts import install
from flm.models import TinyLM

_norm = os.environ.get("FUZZ_NORM", "")
_gate = os.environ.get("FUZZ_GATE", "") == "1"
_rules = int(os.environ.get("FUZZ_GATE_R", "6"))
_ffn = os.environ.get("FUZZ_FFN", "")
_ffn_r = int(os.environ.get("FUZZ_FFN_R", "52"))
_proj = os.environ.get("FUZZ_PROJ", "")
_taps = int(os.environ.get("FUZZ_FFN_TAPS", "1"))
_r0 = int(os.environ.get("FUZZ_FFN_R0", "0"))
_di = os.environ.get("FUZZ_INIT", "") == "data"
_ranks = tuple(int(x) for x in os.environ.get("FUZZ_PROJ_RANKS", "32,20").split(","))
_emb = os.environ.get("FUZZ_EMB", "")


def _init(self, cfg, *a, **k):
    R._new_init(self, cfg, *a, **k)
    if _emb:
        FT.fuzzify_init(self, _emb)
    install(self, _norm, _gate, _rules, _ffn, _ffn_r, _proj, _taps, _r0, _di, _ranks)


if __name__ == "__main__":
    TinyLM.__init__ = _init
    TinyLM.embed = FT._embed if _emb else R._embed
    TinyLM.logits = FT._logits if _emb else R._logits
    TinyLM.n_params = R._n_params
    T.main(sys.argv[1:])
