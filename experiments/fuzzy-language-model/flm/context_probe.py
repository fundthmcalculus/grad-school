"""Does the model use long context? Three probes on a saved subword checkpoint (CPU or GPU).

    .venv/bin/python -m flm.context_probe CKPT.pt [--len 2048] [--windows 64] [--out report.json]

1. **Bits by position.** Sequential validation windows of ``--len`` tokens (longer than the training
   context is allowed for mixers without RoPE, i.e. the delta-form fuzzy mixer D) scored per token;
   bits per character reported for position buckets 0-255, 256-511, 512-1023, 1024+. If extra context
   helps, later buckets are cheaper *for the same text*; a model trained at ctx 256 that cannot use
   what lies beyond it shows a jump at 256.
2. **Bits by distance from the story start** (tokens since the last <|eot|>). Stories are ~250-350
   tokens at BPE1024, so context beyond the current story is another story: it cannot help, and a
   model that "uses" it is only reading noise. Only tokens whose story start lies inside the window
   are counted.
3. **Name recall.** A story opens with "a little girl named N", then ``k`` neutral filler sentences,
   then 'her mom said, "Come here, ' and the score is log p(N) - log p(M) for a distractor name M,
   in bits. The same prompts with the name statement replaced by a neutral phrase give each pair's
   prior margin; the reported recall gain is named minus control per pair (the prior cancels).
   ``prior_abs_margin`` is the size of that prior, which the gain should clearly exceed (the
   signed mean of the prior is 0 by symmetry over ordered pairs, so it is not a check).

Registered before any run (CPU session, 2026-10-11): on the 100M BPE1024 D checkpoint (ctx 256),
the recall gain at k = 0 exceeds 2 bits and falls toward 0 by k = 24 filler sentences (~600 tokens);
bits by position drop between bucket 0 and 1 (more context) and do not improve past 256 for this
model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flm.rules import load_model, make_codec

LN2 = math.log(2)
POS_EDGES = (0, 256, 512, 1024, 10**9)
STORY_EDGES = (0, 64, 128, 256, 512, 10**9)
NAMES = ("Lily", "Tom", "Mia", "Ben", "Sue", "Max", "Anna", "Sam")
FILLER = (
    " She loved to play with her toys.",
    " One day, she went to the park with her mom.",
    " They saw a big tree and a little bird.",
    " The sun was warm and the grass was soft.",
    " The bird sang a happy song.",
    " They sat down and ate a snack.",
)
OPEN_NAMED = "Once upon a time, there was a little girl named {n}."
OPEN_NEUTRAL = "Once upon a time, there was a little girl."
QUERY = ' Then her mom said, "Come here,'


def bucket(x: int, edges) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2


@torch.no_grad()
def position_bits(model, codec, length: int, windows: int, start: int = 900_000):
    """Per-bucket bits per character by window position and by distance from the story start."""
    val = codec.bpe.load("valid")
    lens = np.asarray(codec.bpe.lens).astype(np.int64)
    eot = codec.bpe.eot
    nll_pos = np.zeros(len(POS_EDGES) - 1)
    chr_pos = np.zeros(len(POS_EDGES) - 1)
    nll_st = np.zeros(len(STORY_EDGES) - 1)
    chr_st = np.zeros(len(STORY_EDGES) - 1)
    n_ok = 0
    for w in range(windows):
        a = start + w * length
        ids = np.asarray(val[a : a + length + 1], dtype=np.int64)
        if len(ids) < length + 1:
            break
        x = torch.from_numpy(ids)[None]
        logits = model(x[:, :-1])[0]
        nll = F.cross_entropy(logits, x[0, 1:], reduction="none").numpy() / LN2
        tl = lens[ids[1:]]
        since = -1
        for t in range(length):
            if ids[t] == eot:
                since = 0
            elif since >= 0:
                since += 1
            b = bucket(t, POS_EDGES)
            nll_pos[b] += nll[t]
            chr_pos[b] += tl[t]
            if since >= 0:
                s = bucket(since, STORY_EDGES)
                nll_st[s] += nll[t]
                chr_st[s] += tl[t]
        n_ok += 1
    div = lambda a, b: [
        None if c == 0 else round(float(n / c), 4) for n, c in zip(a, b)
    ]  # noqa: E731
    return {
        "windows": n_ok,
        "length": length,
        "by_position": {
            f"{POS_EDGES[i]}-{POS_EDGES[i+1] if POS_EDGES[i+1] < 10**9 else 'end'}": v
            for i, v in enumerate(div(nll_pos, chr_pos))
        },
        "by_story_distance": {
            f"{STORY_EDGES[i]}-{STORY_EDGES[i+1] if STORY_EDGES[i+1] < 10**9 else 'end'}": v
            for i, v in enumerate(div(nll_st, chr_st))
        },
        "chars_by_position": [int(c) for c in chr_pos],
    }


def continuation_logprob(model, codec, prefix: str, cont: str) -> float | None:
    """log2 p(cont | prefix) summed over the continuation tokens, or None if the tokenizer merges
    across the prefix/continuation boundary (then the split is not a clean token boundary).
    """
    p = codec.bpe.encode(prefix)
    f = codec.bpe.encode(prefix + cont)
    if len(f) <= len(p) or not np.array_equal(f[: len(p)], p):
        return None
    x = torch.from_numpy(f)[None]
    with torch.no_grad():
        lp = F.log_softmax(model(x[:, :-1])[0], dim=-1)
    tgt = x[0, 1:]
    got = lp[torch.arange(len(tgt)), tgt][len(p) - 1 :]
    return float(got.sum() / LN2)


def filler(k: int) -> str:
    return "".join(FILLER[i % len(FILLER)] for i in range(k))


def name_recall(model, codec, ks=(0, 1, 3, 6, 12, 24)):
    rows = []
    for k in ks:
        named, ctrl, skipped = [], [], 0
        for n in NAMES:
            for m in NAMES:
                if m == n:
                    continue
                vals = []
                for opening in (OPEN_NAMED.format(n=n), OPEN_NEUTRAL):
                    pre = opening + filler(k) + QUERY
                    a = continuation_logprob(model, codec, pre, " " + n)
                    b = continuation_logprob(model, codec, pre, " " + m)
                    vals.append(None if a is None or b is None else a - b)
                if None in vals:
                    skipped += 1
                    continue
                named.append(vals[0])
                ctrl.append(vals[1])
        if not named:
            rows.append({"k": k, "skipped": skipped})
            continue
        ntok = len(codec.bpe.encode(OPEN_NAMED.format(n="Lily") + filler(k) + QUERY))
        rows.append(
            {
                "k": k,
                "prompt_tokens": ntok,
                "margin_named": round(float(np.mean(named)), 3),
                "prior_abs_margin": round(float(np.mean(np.abs(ctrl))), 3),
                "recall_gain_bits": round(float(np.mean(named) - np.mean(ctrl)), 3),
                "pairs": len(named),
                "skipped": skipped,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, tok = load_model(a.ckpt)
    codec = make_codec(tok)
    if codec.kind != "bpe":
        raise SystemExit("context_probe needs a subword checkpoint")
    rep = {
        "ckpt": str(a.ckpt),
        "train_ctx": model.cfg.max_len,
        "params": model.n_params(),
        "positions": position_bits(model, codec, a.len, a.windows),
        "name_recall": name_recall(model, codec),
    }
    text = json.dumps(rep, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
