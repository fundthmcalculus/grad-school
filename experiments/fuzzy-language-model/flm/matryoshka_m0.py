"""M0 of PLAN_MATRYOSHKA: post-hoc truncation curve of a plainly trained model (no training).

    python -m flm.matryoshka_m0 outputs/v18-3b/D-d48-bpe1024-ctx512-r32_3000M_s0.pt

For each k, val BPC when (out) the tied output uses only the first k dimensions of the final state and embedding,
and (both) additionally the input embedding is zeroed beyond k.  The null an MRL objective has to beat.
"""

import json
import sys

import torch

from flm import train
from flm.bpe import BPE
from flm.models import ModelConfig, TinyLM

KS = (8, 16, 24, 32, 40, 48)


def main(path):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu")
    m = TinyLM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["state"])
    m.to(dev).eval()
    train.TOK = BPE(int(ck["tok"].removeprefix("bpe")))
    val = train.TOK.load("valid")
    ctx = m.cfg.max_len if ck["config"].get("max_len") else 512
    d = m.cfg.d_model
    orig_logits, orig_embed = m.logits, m.embed
    res = {}
    for k in KS:
        if k > d:
            continue

        def logits_k(h, k=k):
            h = m.norm(h)
            return h[..., :k] @ m.emb.weight[:, :k].T

        def embed_k(idx, k=k):
            e = orig_embed(idx)
            return torch.cat([e[..., :k], torch.zeros_like(e[..., k:])], -1)

        m.logits = logits_k
        m.embed = orig_embed
        out = train.evaluate(m, val, ctx, 1_000_000, device=dev)
        m.embed = embed_k
        both = train.evaluate(m, val, ctx, 1_000_000, device=dev)
        res[k] = {"out_only": round(out, 4), "out_and_in": round(both, 4)}
        print(k, res[k], flush=True)
    m.logits, m.embed = orig_logits, orig_embed
    res["full"] = round(train.evaluate(m, val, ctx, 1_000_000, device=dev), 4)
    print("full", res["full"])
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1])
