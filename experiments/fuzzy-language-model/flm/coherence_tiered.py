"""``python -m flm.coherence_tiered out ckpt...``: flm.coherence_set with the frequency-tiered embedding patches applied
(TIER_NH / TIER_E must match the checkpoint; checkpoints without the tiered keys are loaded by the stock class).
"""

import sys

import flm.coherence_set as C
import flm.run_tiered as R
from flm.models import TinyLM

_stock_init = TinyLM.__init__
_stock = dict(embed=TinyLM.embed, logits=TinyLM.logits, n_params=TinyLM.n_params)
_orig_load = TinyLM.load_state_dict


def _load(self, sd, *a, **k):
    tiered = any(key.startswith("emb_h") for key in sd)
    if tiered and not hasattr(self, "emb_h"):
        raise RuntimeError("tiered checkpoint into a stock model")
    return _orig_load(self, sd, *a, **k)


if __name__ == "__main__":
    # tiered and stock models are built per checkpoint: decide by checkpoint name
    import torch

    real_load = torch.load

    def load(path, *a, **k):
        ck = real_load(path, *a, **k)
        tiered = any(key.startswith("emb_h") for key in ck["state"])
        TinyLM.__init__ = R._new_init if tiered else _stock_init
        TinyLM.embed = R._embed if tiered else _stock["embed"]
        TinyLM.logits = R._logits if tiered else _stock["logits"]
        return ck

    torch.load = load
    C.torch.load = load
    C.main_args = None
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    C.main()
