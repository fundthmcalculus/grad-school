"""Parity check (a script, not a pytest test): allfuzzy_lm.py (plain numpy) against the trained torch model, logits at every position of a prompt.
PYTHONPATH=. python allfuzzy/check_parity.py outputs/v52-e2e/e2e_tail_300M_s0.pt [allfuzzy/e2e_tail_300M_s0.npz]      (needs torch and the repo; the script itself does not)
A fuzzy-head checkpoint (``head_proto`` in the state) is rebuilt with signed memberships when its file name says 'tanh' (as flm/coherence_fuzzy.py does).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flm.coherence_fuzzy as CF  # noqa: E402
import flm.fuzzy_tokens as FT  # noqa: E402
import flm.run_tiered as R  # noqa: E402
from allfuzzy_lm import AllFuzzyLM  # noqa: E402
from flm.models import ModelConfig, TinyLM  # noqa: E402

if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    st = ck["state"]
    R.NH, R.E = st["emb_h.weight"].shape[0], st["emb_t.weight"].shape[1]
    os.environ["TIER_FREQ"] = f"outputs/freq_bpe{ck['config']['vocab_size']}.npz"
    stem = Path(sys.argv[1]).name
    FT.FORM = "tanh" if "tanh" in stem or "_tp" in stem else "softmax"
    CF._state.update(
        tiered=True, state=st, emb_mode="all" if "head_proto" in st else "tail"
    )
    TinyLM.__init__, TinyLM.embed, TinyLM.logits = CF._init, FT._embed, FT._logits
    model = TinyLM(ModelConfig(**ck["config"]))
    model.load_state_dict(st)
    model.eval()
    lm = AllFuzzyLM(*(sys.argv[2:3] and [sys.argv[2]]))
    text = "Once upon a time, there was a little girl named Lily. She loved to play in the park with her dog."
    ids = lm.tok.encode(text).ids
    with torch.no_grad():
        ref = model(torch.tensor([ids]))[0].numpy()
    lm.reset()
    mine = np.stack([lm.step(i)[0] for i in ids])
    err = np.abs(ref - mine).max()
    print(
        f"{len(ids)} positions, max |logit difference| numpy vs torch: {err:.2e}; argmax agrees at {(ref.argmax(1) == mine.argmax(1)).mean():.0%} of positions"
    )
    assert err < 1e-3, err
