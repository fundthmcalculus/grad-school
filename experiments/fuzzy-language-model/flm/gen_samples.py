"""Generate review samples from saved checkpoints (CPU): python -m flm.gen_samples out.json ckpt...

Each checkpoint x prompt x (temperature, top_k) x 2 seeds -> 400 characters. The generator
resamples with the full 256-character window, exactly as `TinyLM.generate` does in training.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from flm.data import decode as char_decode
from flm.data import encode as char_encode
from flm.models import ModelConfig, TinyLM

PROMPTS = [
    "Once upon a time",
    "Lily and Ben went to the park.",
    "The little dog was sad because",
    "One day, a girl named Mia found",
]
SETTINGS = [(0.7, 20), (1.0, None)]


def main():
    out, ckpts = sys.argv[1], sys.argv[2:]
    torch.set_num_threads(4)
    res = []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu")
        m = TinyLM(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["state"])
        m.eval()
        if ck.get("tok"):  # subword checkpoint
            from flm.bpe import BPE

            bp = BPE(int(ck["tok"].removeprefix("bpe")))
            encode, decode = bp.encode, bp.decode
        else:
            encode, decode = char_encode, char_decode
        meta = json.loads(Path(c).with_suffix(".json").read_text())
        item = {
            "name": Path(c).stem,
            "params": meta["params_total"],
            "val_bpc": meta["val_bpc"],
            "mixer": meta["config"]["mixer"],
            "samples": [],
        }
        for p in PROMPTS:
            for temp, k in SETTINGS:
                for seed in (0, 1):
                    g = torch.Generator().manual_seed(seed)
                    idx = torch.from_numpy(encode(p).astype(np.int64))[None]
                    y = m.generate(idx, 400, temperature=temp, top_k=k, generator=g)
                    item["samples"].append(
                        {
                            "prompt": p,
                            "temp": temp,
                            "top_k": k,
                            "seed": seed,
                            "text": decode(y[0].tolist()),
                        }
                    )
        res.append(item)
        print("done", item["name"], flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
