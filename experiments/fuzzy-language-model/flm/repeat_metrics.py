"""Repetition and plausibility metrics for generated stories (`PLAN_REPEAT.md`).

``python -m flm.repeat_metrics score <dir>``: for every ``<dir>/texts_<setting>.json`` (a list of {ckpt, prompt, text}) print and write ``<dir>/summary.json``: the word-level repetition metrics (`word_metrics`) and the plausibility of each text, the mean negative log-likelihood per token under an
independent, better model: the hybrid H3 at warm-up 1000 (``outputs/v72-hybshape300/H3_v6144_h640_d40_wu1000_s0.pt``; the same BPE6144 tokenizer as the all-fuzzy F7 whose texts are scored), so that a decoding setting that buys fewer repeats with implausible text shows up as a higher value.
The judge is loaded in this process only (the model classes are patched globally by ``flm.coherence_fuzzy.configure``, so a fuzzy model and the hybrid cannot live in one process): generate first (``flm.decode_sweep``), score second.
"""

import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

JUDGE = "outputs/v72-hybshape300/H3_v6144_h640_d40_wu1000_s0.pt"


def words(t: str):
    return re.findall(r"[A-Za-z']+", t.lower())


def rep3(w):
    seen, rep, n = set(), 0, 0
    for i in range(len(w) - 2):
        g = tuple(w[i : i + 3])
        n += 1
        rep += g in seen
        seen.add(g)
    return rep / max(n, 1)


def near(w, k=6):
    c = n = 0
    for i, x in enumerate(w):
        if len(x) > 3:
            n += 1
            c += x in w[max(0, i - k) : i]
    return c / max(n, 1)


def distinct(w, n):
    g = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return len(set(g)) / max(len(g), 1)


def word_metrics(texts):
    ws = [words(t) for t in texts]
    return dict(
        rep3=st.mean(rep3(w) for w in ws),
        near=st.mean(near(w) for w in ws),
        distinct1=st.mean(distinct(w, 1) for w in ws),
        distinct2=st.mean(distinct(w, 2) for w in ws),
        n_texts=len(texts),
    )


def judge_nll(texts):
    """Mean negative log-likelihood (nats per token) of each text under the judge model."""
    import torch

    import flm.coherence_fuzzy as CF
    from flm.bpe import BPE
    from flm.models import ModelConfig, TinyLM

    ck = torch.load(JUDGE, map_location="cpu", weights_only=False)
    CF.configure(JUDGE, ck)
    m = TinyLM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["state"])
    m = m.cuda().eval()
    bp = BPE(int(ck["tok"].removeprefix("bpe")))
    out = []
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor(bp.encode(t).astype(np.int64))[None].cuda()
            ids = ids[:, : m.cfg.max_len]
            lg = m(ids[:, :-1]).float()
            out.append(torch.nn.functional.cross_entropy(lg[0], ids[0, 1:]).item())
    return out


if __name__ == "__main__":
    d = Path(sys.argv[2])
    summary = {}
    for f in sorted(d.glob("texts_*.json")):
        rows = json.loads(f.read_text())
        texts = [r["text"] for r in rows]
        s = word_metrics(texts)
        s["judge_nll"] = st.mean(judge_nll(texts))
        name = f.stem.removeprefix("texts_")
        summary[name] = s
        print(
            f"{name:12s} repeated trigrams {s['rep3']:.3f}  near-repeat {s['near']:.3f}  distinct-1 {s['distinct1']:.3f} distinct-2 {s['distinct2']:.3f}  judge NLL {s['judge_nll']:.3f}",
            flush=True,
        )
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
