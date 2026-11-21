"""Score a Pythia checkpoint in bits per character on the validation text of our corpora (`PLAN_PYTHIA.md` stage P0).

``python -m flm.score_pythia [--model EleutherAI/pythia-14m] [--rev <sha>] [--chars 1000000] [--ctx 512] [--device cpu]``
Text: the first ``--chars`` characters of each validation set (TinyStories valid file, SimpleStories test shard, the first documents of the last Cosmopedia shard), documents joined by the model's end-of-text token, the same non-overlapping 513-token windows as `flm.train.evaluate` (512 targets per window, the first token of a window is context only).
Bits per character = sum of target-token negative log-likelihood in bits / characters those tokens span (from the tokenizer's offsets, so it is exact for the raw text, not for our 98-symbol mapped stream; the two differ only by the 0.06% of non-ASCII characters we map to one symbol).
Writes ``report/pythia_p0.json``.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path(__file__).resolve().parents[3] / "data"
EOT = "<|endoftext|>"


def folded_docs(folded_dir, chars):
    """Validation documents of a streamed Pile sample (``flm.prep_pile``: folded text, one file per shard, documents ended by the unit separator)."""
    out = []
    for f in sorted(Path(folded_dir).glob("s*.valid.txt")):
        out += [d for d in f.read_text(encoding="ascii").split("") if d]
    keep, n = [], 0
    for d in out:
        if n >= chars:
            break
        keep.append(d)
        n += len(d) + len(EOT)
    return keep


def docs(name, chars):
    if name == "tinystories":
        raw = (DATA / "tinystories/TinyStoriesV2-GPT4-valid.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        out = [d.strip() for d in raw.split(EOT)]
    elif name == "simplestories":
        out = pd.read_parquet(
            DATA / "simplestories/raw/data/test-00000-of-00001.parquet"
        )["story"].tolist()
    else:
        out = pd.read_parquet(
            DATA / "cosmopedia/raw/cosmopedia-v2/train-00103-of-00104.parquet"
        )["text"].tolist()
    keep, n = [], 0
    for d in out:
        if n >= chars:
            break
        keep.append(d)
        n += len(d) + len(EOT)
    return keep


@torch.no_grad()
def score(model, tok, texts, ctx, device):
    text = EOT.join(texts)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, off = np.array(enc["input_ids"]), np.array(enc["offset_mapping"])
    span = (off[:, 1] - off[:, 0]).astype(np.int64)
    n_win = len(ids) // (ctx + 1)
    bits, nchar = 0.0, 0
    for w in range(n_win):
        sl = slice(w * (ctx + 1), (w + 1) * (ctx + 1))
        x = torch.from_numpy(ids[sl]).long()[None].to(device)
        lp = torch.log_softmax(model(x[:, :-1]).logits.float(), -1)
        nll = -lp.gather(-1, x[:, 1:, None])[..., 0]
        bits += float(nll.sum()) / math.log(2)
        nchar += int(span[sl][1:].sum())
    return bits / nchar, n_win * ctx, nchar


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-14m")
    ap.add_argument("--rev", default="cf967c0a9a04383db6f7b1108d86b2962634b4ac")
    ap.add_argument("--chars", type=int, default=1_000_000)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="report/pythia_p0.json")
    ap.add_argument(
        "--folded-dir",
        default="",
        help="score only the folded validation text of a streamed Pile sample (e.g. data/pile6144/text), same characters our model is scored on",
    )
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.rev).eval()
    model.to(a.device)
    n_par = sum(p.numel() for p in model.parameters())
    res = dict(model=a.model, revision=a.rev, params=n_par, ctx=a.ctx, corpora={})
    names = (
        ("pile_folded",)
        if a.folded_dir
        else ("tinystories", "simplestories", "cosmopedia")
    )
    for name in names:
        texts = (
            folded_docs(a.folded_dir, a.chars) if a.folded_dir else docs(name, a.chars)
        )
        b, nt, nc = score(model, tok, texts, a.ctx, a.device)
        res["corpora"][name] = dict(bpc=b, target_tokens=nt, target_chars=nc)
        print(name, f"{b:.4f} bpc over {nt} tokens / {nc} chars", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
