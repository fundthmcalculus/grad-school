"""Read a trained FLM/FRLM as a rulebook.

    .venv/bin/python -m flm.rules outputs/<sweep>/<run>.pt [--windows 400] [--top 8]

For every TSK component in the model:

* **Sequence-mixer rules** (``fuzzy`` / ``fuzzydelta`` mixers, per head): the antecedent
  of rule r is a Gaussian region of key/query space. It is described by the text
  contexts whose *key* membership in r is highest on validation data (the write side:
  "what gets stored in rule r"). The consequent is in-context state, so it has no
  fixed reading; we report how often and how crisply the rule fires instead.
* **Channel-mixer rules** (``tsk`` FFN): antecedent as above (the contexts that fire the
  rule most); the constant consequent a_r is read through the logit lens, i.e. the
  final RMSNorm and the tied unembedding, as "THEN raise these next characters".
  The lens ignores later layers, so it is a direct-path reading, not the whole effect.

Statistics per rule: mean firing (usage), share of positions where it is the winning
rule, and mean max-membership over all positions (partition crispness, 1/R = uniform,
1 = crisp).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from flm.data import decode, load
from flm.models import FuzzyDeltaMixer, FuzzyRecurrentMixer, ModelConfig, TinyLM, TSKFFN


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TinyLM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["state"])
    return m.eval()


def capture(model, windows):
    """Run windows through the model, recording key-side memberships and FFN firing."""
    rec = {}
    handles = []
    for li, blk in enumerate(getattr(model, "blocks", [])):
        mix = blk.mix
        if isinstance(mix, (FuzzyRecurrentMixer, FuzzyDeltaMixer)):
            orig = mix.features

            def feats(x, _orig=orig, _li=li):
                fq, fk, v = _orig(x)
                rec.setdefault(("mix", _li), []).append(fk.detach())  # (B,H,T,R)
                return fq, fk, v

            mix.features = feats
            handles.append((mix, "features", orig))
        if isinstance(blk.ffn, TSKFFN):
            ffn = blk.ffn
            orig_f = ffn.firing

            def firing(x, _orig=orig_f, _li=li):
                w = _orig(x)
                rec.setdefault(("ffn", _li), []).append(w.detach())  # (B,T,R)
                return w

            ffn.firing = firing
            handles.append((ffn, "firing", orig_f))
    with torch.no_grad():
        for i in range(0, len(windows), 32):
            model(windows[i : i + 32])
    for obj, name, orig in handles:
        setattr(obj, name, orig)
    return {k: torch.cat(v, 0) for k, v in rec.items()}


def show(ctx_ids):
    s = decode(ctx_ids).replace("\n<|eot|>\n", "⏎⏎").replace("\n", "⏎")
    return s


def describe_rule(memb, windows, top, ctx_len=14):
    """memb: (N, T) membership of one rule over all positions."""
    flat = memb.reshape(-1)
    T = memb.shape[1]
    idx = torch.topk(flat, top * 20).indices
    examples, suffixes = [], Counter()
    for j in idx.tolist():
        n, t = divmod(j, T)
        lo = max(0, t - ctx_len + 1)
        ctx = windows[n, lo : t + 1].tolist()
        suffixes[show(ctx[-2:])] += 1
        if len(examples) < top:
            examples.append(f"`{show(ctx)}`")
    return examples, suffixes.most_common(3)


def logit_lens(model, vec, k=6):
    with torch.no_grad():
        logits = model.norm(vec[None]) @ model.emb.weight.T
        p = torch.softmax(logits[0], -1)
    top = torch.topk(p, k)
    return ", ".join(
        f"`{show([i])}` {v:.2f}"
        for v, i in zip(top.values.tolist(), top.indices.tolist())
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument(
        "--max-rules",
        type=int,
        default=0,
        help="limit rules shown per component (0 = all)",
    )
    a = ap.parse_args()
    model = load_model(a.checkpoint)
    T = model.cfg.max_len
    val = load("valid")
    # second half of the validation slice: disjoint from the 1M chars used for val BPC
    start = 2_000_000
    w = torch.from_numpy(
        np.asarray(val[start : start + a.windows * T], dtype=np.int64)
    ).view(a.windows, T)
    rec = capture(model, w)
    lines = [
        f"# Rulebook: `{Path(a.checkpoint).name}`",
        "",
        f"config: `{model.cfg.to_dict()}`",
        "",
    ]
    for (kind, li), m in sorted(rec.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if kind == "mix":
            N, H, Tt, R = m.shape
            for h in range(H):
                mh = m[:, h]  # (N,T,R)
                crisp = mh.max(-1).values.mean().item()
                win = torch.bincount(mh.argmax(-1).reshape(-1), minlength=R).float() / (
                    N * Tt
                )
                lines += [
                    f"## Layer {li} sequence mixer, head {h}: {R} rules (crispness {crisp:.2f}; uniform = {1 / R:.2f})",
                    "",
                ]
                order = torch.argsort(-win).tolist()
                for r in order[: a.max_rules or R]:
                    ex, suf = describe_rule(mh[..., r], w, a.top)
                    lines.append(
                        f"- **rule {r}** — wins {win[r]:.1%} of positions, mean membership {mh[..., r].mean():.3f}"
                    )
                    lines.append(
                        f"  - stores contexts ending in: {', '.join(f'`{s}` ({c})' for s, c in suf)}"
                    )
                    lines.append(f"  - strongest: {' · '.join(ex)}")
                lines.append("")
        else:
            N, Tt, R = m.shape
            ffn = model.blocks[li].ffn
            crisp = m.max(-1).values.mean().item()
            win = torch.bincount(m.argmax(-1).reshape(-1), minlength=R).float() / (
                N * Tt
            )
            alive = int((win > 0.001).sum())
            lines += [
                f"## Layer {li} TSK channel mixer: {R} rules, {alive} win >0.1% of positions (crispness {crisp:.2f}; uniform = {1 / R:.3f})",
                "",
            ]
            order = torch.argsort(-win).tolist()
            for r in order[: a.max_rules or R]:
                ex, suf = describe_rule(m[..., r], w, a.top)
                lines.append(
                    f"- **rule {r}** — wins {win[r]:.1%}; IF context ends like {', '.join(f'`{s}`' for s, _ in suf)} THEN raise {logit_lens(model, ffn.conseq[r])}"
                )
                lines.append(f"  - strongest: {' · '.join(ex)}")
            lines.append("")
    out = Path(a.checkpoint).with_suffix(".rules.md")
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
