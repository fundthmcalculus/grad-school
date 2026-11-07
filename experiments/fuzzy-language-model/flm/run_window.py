"""``python -m flm.run_window <train args>``: the bijective all-fuzzy model (``flm.run_bijective``, ``BIJ=ci``) with the
sliding-window mixers of ``flm.windowed`` registered (``--mixer bsolve`` / ``--mixer wdelta``, window from env ``WIN``).

Frozen dictionary: set ``FIXED_C=outputs/embspace/oh48.npz`` (built by ``python -m flm.canonical_dictionary_oh``) to
freeze the codebook to the named dictionary through ``flm.run_fixed_embedding``; leave it unset for the learned codebook.

    BIJ=ci WIN=32 python -m flm.run_window --mixer wdelta --decay data ... --out X.json
    BIJ=ci WIN=32 FIXED_C=outputs/embspace/oh48.npz python -m flm.run_window --mixer wdelta ... --out Y.json
"""

import os
import sys

import flm.windowed  # noqa: F401  (registers wgauss / bsolve / wdelta in models.MIXERS)
import flm.run_bijective as RB
import flm.run_fixed_embedding as FE
from flm.models import TinyLM

if __name__ == "__main__":
    RB.T.evaluate = RB._evaluate_logged
    if os.environ.get("FIXED_C") or os.environ.get("FIXED_FG"):
        if os.environ.get("FIXED_PULL"):
            from torch.optim.optimizer import register_optimizer_step_post_hook

            register_optimizer_step_post_hook(FE._pull_back)
        TinyLM.__init__ = FE._fixed_init
    else:
        TinyLM.__init__ = RB._new_init
    TinyLM.embed = RB._embed
    TinyLM.logits = RB._logits
    TinyLM.n_params = RB._n_params
    TinyLM.recovery = RB.recovery
    RB.T.main(sys.argv[1:])
