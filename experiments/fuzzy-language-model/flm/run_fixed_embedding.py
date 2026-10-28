"""``flm.run_bijective`` with the character dictionary frozen to a given matrix (a shared, fixed vector space).

    FIXED_C=outputs/embspace/LABEL.npz[:KEY] BIJ=ci FUZZ_FFN_R=156 FUZZ_FFN_R0=109 python -m flm.run_fixed_embedding <the usual train arguments>

``FIXED_C`` names an .npz made by ``flm.embedding_space`` (default key ``C``); its width must equal the model width.  The matrix is centred and rescaled to the entry spread the codebook is initialised with (0.5) and frozen; everything else (F, the gain, the blocks, tau) trains as usual.  The frozen dictionary still counts in the printed parameter total (98 x d, 5,488 at d 56); when one dictionary is shared by every model it is not a per-model cost.
"""

import os
import sys

import numpy as np
import torch

import flm.run_bijective as RB
from flm.models import TinyLM

_init = RB._new_init


def _fixed_init(self, cfg, *a, **k):
    _init(self, cfg, *a, **k)
    path, _, key = os.environ["FIXED_C"].partition(":")
    c = np.load(path)[key or "C"].astype(np.float32)
    assert c.shape == tuple(self.codebook.shape), (c.shape, tuple(self.codebook.shape))
    c = (c - c.mean(0)) / c.std()
    self.codebook.data.copy_(torch.as_tensor(c * 0.5))
    self.codebook.requires_grad_(False)


if __name__ == "__main__":
    RB.T.evaluate = RB._evaluate_logged
    TinyLM.__init__ = _fixed_init
    TinyLM.embed = RB._embed
    TinyLM.logits = RB._logits
    TinyLM.n_params = RB._n_params
    TinyLM.recovery = RB.recovery
    RB.T.main(sys.argv[1:])
