"""Does a rebuilt checkpoint reproduce its recorded validation loss?  ``python -m flm.verify_rebuild ckpt ...`` (run before generating coherence texts from a checkpoint; `flm.coherence_fuzzy.configure` does the rebuild).

Scores each checkpoint in bits per character on the first 1M characters of the validation stream of its own tokenizer (``GRAD_SCHOOL_DATA`` and, for a corpus other than TinyStories, ``COH_TIER_FREQ`` as for `flm.coherence_fuzzy`) and compares with the ``val_bpc`` in the checkpoint's result file; exits non-zero if any differs by more than 2e-4.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import flm.coherence_fuzzy as CF
import flm.train as T
from flm.bpe import BPE
from flm.models import ModelConfig, TinyLM

if __name__ == "__main__":
    bad = 0
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        CF.configure(path, ck)
        bp = BPE(int(ck["tok"].removeprefix("bpe")))
        T.TOK = bp
        m = TinyLM(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["state"])
        got = T.evaluate(m, np.asarray(bp.load("valid")), 512, 1_000_000)
        want = json.loads(Path(path).with_suffix(".json").read_text())["val_bpc"]
        ok = abs(got - want) < 2e-4
        bad += not ok
        print(
            f"{'ok ' if ok else 'BAD'} {path}: rebuilt {got:.5f}, recorded {want:.5f}",
            flush=True,
        )
    sys.exit(1 if bad else 0)
