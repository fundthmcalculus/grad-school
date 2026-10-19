"""flm.train with fuzzy replacements for non-fuzzy parts (`python -m flm.run_allfuzzy <train args>`; env FUZZ_NORM = fuzzy|scale, FUZZ_GATE = 1).

Composes with the tiered embedding (TIER_NH / TIER_E / TIER_FREQ as for flm.run_tiered).  train.py and models.py are untouched.
"""

import os
import sys

import flm.run_tiered as R
import flm.train as T
from flm.fuzzy_parts import install
from flm.models import TinyLM

_norm = os.environ.get("FUZZ_NORM", "")
_gate = os.environ.get("FUZZ_GATE", "") == "1"
_rules = int(os.environ.get("FUZZ_GATE_R", "6"))


def _init(self, cfg, *a, **k):
    R._new_init(self, cfg, *a, **k)
    install(self, _norm, _gate, _rules)


if __name__ == "__main__":
    TinyLM.__init__ = _init
    TinyLM.embed = R._embed
    TinyLM.logits = R._logits
    TinyLM.n_params = R._n_params
    T.main(sys.argv[1:])
