"""Decode-time settings for the all-fuzzy model's stories (`PLAN_REPEAT.md` part 1): ``python -m flm.decode_sweep [setting ...]``.

Generates 4 story prompts x 12 samples per checkpoint (F7 at 300M, seeds 0 and 1: ``outputs/v70-shape300``), 600 characters each, under each setting and writes ``outputs/v73-repeat/texts_<setting>.json``; score them with ``python -m flm.repeat_metrics score outputs/v73-repeat``.
Settings: S0 the baseline of every rated set (temperature 0.5, top-k 10); S1-S3 sampling only; S4-S5 a token that would complete a trigram already in the text is blocked; S6 a frequency penalty over the last 64 tokens; S7 a repetition penalty over the last 32 tokens; S8-S9 DRY (a penalty on the token that would extend a sequence already repeated).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import flm.coherence_fuzzy as CF
from flm.bpe import BPE
from flm.models import ModelConfig, TinyLM

CKPTS = [
    "outputs/v70-shape300/F7_v6144_h640_d32_s0.pt",
    "outputs/v70-shape300/F7_v6144_h640_d32_s1.pt",
]
PROMPTS = [
    "Once upon a time",
    "Lily and Ben went to the park.",
    "The little dog was sad because",
    "One day, a girl named Mia found",
]
SETTINGS = {
    "S0": dict(T=0.5, k=10),
    "S1": dict(T=0.7, k=10),
    "S2": dict(T=0.8, top_p=0.9),
    "S3": dict(T=0.7, min_p=0.05),
    "S4": dict(T=0.5, k=10, nrep=3),
    "S5": dict(T=0.7, k=10, nrep=3),
    "S6": dict(T=0.5, k=10, freq=0.5, freq_w=64),
    "S7": dict(T=0.5, k=10, rep=1.3, rep_w=32),
    "S8": dict(T=0.5, k=10, dry=(0.8, 1.75, 2)),
    "S9": dict(T=0.7, k=10, dry=(0.8, 1.75, 2)),
}
N_SAMPLES, NEW_TOKENS, CHARS = 12, 170, 600


def adjust(logits, ids, s):
    """Penalties and blocks on the raw logits of each sequence (ids: B x L, the prompt and everything sampled so far)."""
    for b in range(logits.shape[0]):
        t = ids[b].tolist()
        if s.get("nrep"):
            n = s["nrep"]
            tail = tuple(t[-(n - 1) :])
            banned = {
                t[i + n - 1]
                for i in range(len(t) - n + 1)
                if tuple(t[i : i + n - 1]) == tail
            }
            if banned:
                logits[b, list(banned)] = -float("inf")
        if s.get("freq"):
            c = np.bincount(t[-s["freq_w"] :], minlength=logits.shape[1])
            logits[b] -= s["freq"] * torch.from_numpy(c).to(logits.device).float()
        if s.get("rep"):
            u = torch.tensor(sorted(set(t[-s["rep_w"] :])), device=logits.device)
            v = logits[b, u]
            logits[b, u] = torch.where(v > 0, v / s["rep"], v * s["rep"])
        if s.get("dry"):
            mult, base, allowed = s["dry"]
            last, pen = t[-1], {}
            for j in range(len(t) - 1):
                if t[j] == last:
                    m = 1
                    while m < 20 and j - m >= 0 and t[j - m] == t[len(t) - 1 - m]:
                        m += 1
                    if m >= allowed:
                        p = mult * base ** (m - allowed)
                        pen[t[j + 1]] = max(pen.get(t[j + 1], 0.0), p)
            for tok, p in pen.items():
                logits[b, tok] -= p
    return logits


def sample_batch(model, ids, s, gen):
    for _ in range(NEW_TOKENS):
        ctx = ids[:, -model.cfg.max_len :]
        logits = model(ctx)[:, -1].float()
        logits = adjust(logits, ids, s) / s["T"]
        if s.get("k"):
            v, _ = torch.topk(logits, s["k"])
            logits[logits < v[:, [-1]]] = -float("inf")
        p = torch.softmax(logits, -1)
        if s.get("top_p"):
            sp, si = torch.sort(p, descending=True)
            keep = torch.cumsum(sp, -1) - sp < s["top_p"]
            p = p * torch.zeros_like(p, dtype=torch.bool).scatter(1, si, keep)
        if s.get("min_p"):
            p = p * (p >= s["min_p"] * p.max(-1, keepdim=True).values)
        p = p / p.sum(-1, keepdim=True)
        ids = torch.cat([ids, torch.multinomial(p, 1, generator=gen)], 1)
    return ids


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = ""
    if (
        "--ckpts" in args
    ):  # other checkpoints (e.g. the post-trained ones): --ckpts a.pt b.pt --tag ul3 S0 S8 -> texts_ul3_S0.json ...
        i = args.index("--ckpts")
        j = next(
            (k for k in range(i + 1, len(args)) if args[k].startswith("--")), len(args)
        )
        CKPTS = args[i + 1 : j]
        args = args[:i] + args[j:]
    if "--tag" in args:
        i = args.index("--tag")
        tag, args = args[i + 1] + "_", args[:i] + args[i + 2 :]
    names = args or list(SETTINGS)
    out = Path("outputs/v73-repeat")
    out.mkdir(parents=True, exist_ok=True)
    models = []
    for path in CKPTS:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        CF.configure(path, ck)
        m = TinyLM(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["state"])
        bp = BPE(int(ck["tok"].removeprefix("bpe")))
        models.append((Path(path).stem, m.cuda().eval(), bp))
    for name in names:
        rows = []
        for ci, (cname, m, bp) in enumerate(models):
            for pi, prompt in enumerate(PROMPTS):
                gen = torch.Generator(device="cuda").manual_seed(1000 + 10 * ci + pi)
                p0 = torch.tensor(bp.encode(prompt).astype(np.int64))[None]
                ids = p0.repeat(N_SAMPLES, 1).cuda()
                with torch.no_grad():
                    full = sample_batch(m, ids, SETTINGS[name], gen)
                for r in range(N_SAMPLES):
                    text = bp.decode(full[r].tolist())[:CHARS]
                    rows.append(dict(ckpt=cname, prompt=prompt, text=text))
        (out / f"texts_{tag}{name}.json").write_text(json.dumps(rows))
        print(name, len(rows), "texts", flush=True)
