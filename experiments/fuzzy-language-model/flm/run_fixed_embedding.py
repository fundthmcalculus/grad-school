"""``flm.run_bijective`` with the character dictionary frozen to a given matrix (a shared, fixed vector space).

    FIXED_C=outputs/embspace/LABEL.npz[:KEY] BIJ=ci FUZZ_FFN_R=156 FUZZ_FFN_R0=109 python -m flm.run_fixed_embedding <the usual train arguments>

``FIXED_FG=<checkpoint.pt>`` instead freezes the codebook, F and the gain from that trained ci model (same width).  ``FIXED_C`` names an .npz made by ``flm.embedding_space`` (default key ``C``); its width must equal the model width.  The matrix is centred and rescaled to the entry spread the codebook is initialised with (0.5) and frozen; everything else (F, the gain, the blocks, tau) trains as usual.  The frozen dictionary still counts in the printed parameter total (98 x d, 5,488 at d 56); when one dictionary is shared by every model it is not a per-model cost.
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
    fg = os.environ.get("FIXED_FG")
    if (
        fg
    ):  # codebook, F and the gain all taken from a trained model and frozen: a universal invertible meaning layer
        sd = torch.load(fg, map_location="cpu", weights_only=False)["state"]
        keep = {
            k: v for k, v in sd.items() if k.split(".")[0] in ("codebook", "F", "G")
        }
        self.load_state_dict(keep, strict=False)
        for n_, p_ in self.named_parameters():
            if n_.split(".")[0] in ("codebook", "F", "G"):
                p_.requires_grad_(False)
        return
    path, _, key = os.environ["FIXED_C"].partition(":")
    c = np.load(path)[key or "C"].astype(np.float32)
    assert c.shape == tuple(self.codebook.shape), (c.shape, tuple(self.codebook.shape))
    c = (c - c.mean(0)) / c.std()
    self.codebook.data.copy_(torch.as_tensor(c * 0.5))
    if not os.environ.get(
        "FIXED_TRAIN"
    ):  # FIXED_TRAIN=1: the matrix only initialises the codebook, which then trains
        self.codebook.requires_grad_(False)
    pull = float(os.environ.get("FIXED_PULL", "0"))
    if pull and os.environ.get("FIXED_TRAIN"):
        _PULL.update(cb=self.codebook, c0=self.codebook.data.clone(), k=pull)


_PULL = {}


def _pull_back(optimizer, args, kwargs):
    """FIXED_PULL=k (with FIXED_TRAIN=1): after every optimiser step move the codebook the fraction k of the way back to its constructed value, a drift penalty that keeps the dictionary readable while letting it adapt."""
    if _PULL:
        cb = _PULL["cb"]
        with torch.no_grad():
            c0 = _PULL["c0"].to(cb.device)
            cb.add_(_PULL["k"] * (c0 - cb))


if __name__ == "__main__":
    RB.T.evaluate = RB._evaluate_logged
    if os.environ.get("FIXED_PULL"):
        from torch.optim.optimizer import register_optimizer_step_post_hook

        register_optimizer_step_post_hook(_pull_back)
    TinyLM.__init__ = _fixed_init
    TinyLM.embed = RB._embed
    TinyLM.logits = RB._logits
    TinyLM.n_params = RB._n_params
    TinyLM.recovery = RB.recovery
    RB.T.main(sys.argv[1:])
