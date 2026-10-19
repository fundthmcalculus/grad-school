"""Build a blinded set of generated stories for model-graded coherence.

    python -m flm.coherence_set report/coherence ckpt...

Every checkpoint gets the same 8 (prompt, seed) pairs at T 0.5 / top-k 10, generated until 600 characters
exist and then cut to exactly 600 characters, so models of different tokenizers are compared on equal text
length. Writes <out>_blind.json (id + text, shuffled, labels removed) and <out>_key.json (id -> model).
"""

import json
import os
import random
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
CHARS = 600
SEEDS = tuple(
    int(x) for x in os.environ.get("COH_SEEDS", "0,1").split(",")
)  # fresh seeds => fresh texts


def gen(model, enc, dec, prompt, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.from_numpy(enc(prompt).astype(np.int64))[None]
    text = dec(ids[0].tolist())
    while len(text) < CHARS + 40:
        ids = model.generate(ids, 60, temperature=0.5, top_k=10, generator=g)
        text = dec(ids[0].tolist())
    return text[:CHARS]


def main():
    out, ckpts = sys.argv[1], sys.argv[2:]
    torch.set_num_threads(4)
    items = []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu")
        m = TinyLM(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["state"])
        m.eval()
        if ck.get("tok"):
            from flm.bpe import BPE

            bp = BPE(int(ck["tok"].removeprefix("bpe")))
            enc, dec = bp.encode, bp.decode
        else:
            enc, dec = char_encode, char_decode
        meta = json.loads(Path(c).with_suffix(".json").read_text())
        for p in PROMPTS:
            for seed in SEEDS:
                items.append(
                    {
                        "model": Path(c).stem,
                        "params": meta["params_total"],
                        "prompt": p,
                        "seed": seed,
                        "text": gen(m, enc, dec, p, seed),
                    }
                )
        print("done", Path(c).stem, flush=True)
    random.Random(7).shuffle(items)
    for i, it in enumerate(items):
        it["id"] = f"S{i + 1:03d}"
    Path(out + "_blind.json").write_text(
        json.dumps([{"id": it["id"], "text": it["text"]} for it in items], indent=1)
    )
    Path(out + "_key.json").write_text(
        json.dumps(
            {
                it["id"]: {k: it[k] for k in ("model", "params", "prompt", "seed")}
                for it in items
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
