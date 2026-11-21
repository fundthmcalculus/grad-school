"""``COH_SEEDS=70,71 python -m flm.coherence_vocab out ckpt...``: flm.coherence_set for checkpoints with different tiered-embedding shapes and vocabularies.

flm.coherence_tiered fixes TIER_NH / TIER_E / TIER_FREQ for the whole run; here each checkpoint sets them from its own weights (head = rows of
``emb_h``, tail width = columns of ``emb_t``, token-frequency file by vocabulary: the BPE1024 default, outputs/freq_bpe<V>.npz otherwise),
so BPE1024 and BPE3072 tiered models can share one blind set.  Stock checkpoints are loaded by the stock class.
"""

import os
import sys

import torch

import flm.coherence_set as C
import flm.run_tiered as R
from flm.models import TinyLM

_stock = dict(
    init=TinyLM.__init__,
    embed=TinyLM.embed,
    logits=TinyLM.logits,
    n_params=TinyLM.n_params,
)
if __name__ == "__main__":
    real_load = torch.load

    def load(path, *a, **k):
        ck = real_load(path, *a, **k)
        st = ck["state"]
        tiered = any(key.startswith("emb_h") for key in st)
        if tiered:
            R.NH, R.E = st["emb_h.weight"].shape[0], st["emb_t.weight"].shape[1]
            v = ck["config"]["vocab_size"]
            os.environ["TIER_FREQ"] = (
                "outputs/v22-tribble/ppmi_k48_v1024.npz"
                if v == 1024
                else f"outputs/freq_bpe{v}.npz"
            )
        TinyLM.__init__ = R._new_init if tiered else _stock["init"]
        TinyLM.embed = R._embed if tiered else _stock["embed"]
        TinyLM.logits = R._logits if tiered else _stock["logits"]
        return ck

    torch.load = load
    C.torch.load = load
    C.main()
