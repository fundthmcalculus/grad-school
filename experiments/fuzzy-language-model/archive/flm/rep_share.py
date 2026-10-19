"""Repeated-trigram share of generated text (PLAN_NEXT.md item 4): ``python -m flm.rep_share out.json ckpt...``
200 new tokens per text, 4 prompts x 25 seeds = 100 texts per checkpoint, T 0.5 / top-k 10.  Share = fraction of
token trigrams of the continuation that already occurred earlier in the same text (prompt included).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from flm.bpe import BPE
from flm.coherence_decode import PROMPTS
from flm.models import ModelConfig, TinyLM


def share(toks, start):
    seen, rep, n = set(), 0, 0
    for i in range(len(toks) - 2):
        tri = tuple(toks[i : i + 3])
        if i + 2 >= start:
            n += 1
            rep += tri in seen
        seen.add(tri)
    return rep / max(n, 1)


@torch.no_grad()
def main():
    out, cks = sys.argv[1], sys.argv[2:]
    torch.set_num_threads(4)
    res = {}
    for c in cks:
        ck = torch.load(c, map_location="cpu")
        m = TinyLM(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["state"])
        m.eval()
        bp = BPE(int(ck["tok"].removeprefix("bpe")))
        sh = []
        for p in PROMPTS[:4]:
            ids = torch.from_numpy(np.asarray(bp.encode(p), dtype=np.int64))[None]
            for s in range(25):
                g = torch.Generator().manual_seed(1000 + s)
                y = m.generate(ids, 200, temperature=0.5, top_k=10, generator=g)[
                    0
                ].tolist()
                sh.append(share(y, ids.shape[1]))
        res[c] = {"mean": float(np.mean(sh)), "sd": float(np.std(sh)), "n": len(sh)}
        print(c, res[c], flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
