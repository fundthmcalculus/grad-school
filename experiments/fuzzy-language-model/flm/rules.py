"""Read a trained FLM/FRLM as a rulebook.

    .venv/bin/python -m flm.rules outputs/<grid>/<run>.pt [--windows 400] [--top 6] [--max-rules 20]

Works on character models and on subword (``--tok bpeN``) checkpoints, on the zero-order softmax-normalised
TSK FFN and on the goal-track first-order raw-firing one. For every TSK component:

* **Sequence-mixer rules** (``fuzzy`` / ``fuzzydelta``, per head): a Gaussian region of key/query space,
  described by the contexts whose *key* membership is highest on held-out text (the write side, "what gets
  stored in rule r"). The consequent is in-context state, so it has no fixed reading; usage and crispness
  are reported instead.
* **Channel-mixer (FFN) rules**: the antecedent is described by the contexts that fire the rule most. The
  consequent is read through the logit lens (the final RMSNorm and the tied unembedding), as "THEN raise
  these next tokens".
  * *zero-order:* output = sum_r w_r a_r, so the lens of a_r is the whole consequent.
  * *first-order:* output = sum_r w_r (v_r . x + b_r) a_r. The rule pushes the next-token logits along
    +a_r when its gain g_r = v_r . x + b_r is positive and along -a_r when it is negative, so both lenses are
    shown with the mean gain over the positions where the rule wins. With raw firing (``ffn_norm=none``) the
    firings are not a distribution: "firing" is reported, not "membership".

The lens ignores later layers, so it is a direct-path reading, not the whole effect.

Statistics per rule: share of positions where it wins, mean firing (usage), and per component the mean
max-membership over all positions (crispness; 1/R = uniform, 1 = crisp, for normalised firing only).
Tokens are shown as pieces: ``·`` is a space, ``↵`` a newline, ``⏎⏎`` the story separator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from flm.data import decode as char_decode
from flm.data import load as char_load
from flm.models import FuzzyDeltaMixer, FuzzyRecurrentMixer, ModelConfig, TinyLM, TSKFFN


class CharCodec:
    kind = "char"

    def piece(self, ids) -> str:
        return _vis(char_decode(ids))

    def windows(self, n: int, T: int, start: int = 2_000_000) -> torch.Tensor:
        val = char_load("valid")
        return torch.from_numpy(
            np.asarray(val[start : start + n * T], dtype=np.int64)
        ).view(n, T)


class BPECodec:
    kind = "bpe"

    def __init__(self, v: int):
        from flm.bpe import EOT_CH, BPE

        self.bpe, self.eot_ch = BPE(v), EOT_CH

    def piece(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            tok = self.bpe.tok.id_to_token(i)
            out.append("⏎⏎" if tok == self.eot_ch else self.bpe.tok.decode([i]))
        return _vis("".join(out))

    def windows(self, n: int, T: int, start: int = 900_000) -> torch.Tensor:
        val = self.bpe.load("valid")
        return torch.from_numpy(
            np.asarray(val[start : start + n * T], dtype=np.int64)
        ).view(n, T)


def _vis(s: str) -> str:
    return s.replace("\n<|eot|>\n", "⏎⏎").replace("\n", "↵").replace(" ", "·")


def make_codec(tok: str):
    return BPECodec(int(tok.removeprefix("bpe"))) if tok else CharCodec()


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TinyLM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["state"])
    return m.eval(), ck.get("tok", "") or ""


@torch.no_grad()
def capture(model, windows):
    """Run windows through the model, recording key-side memberships (mixers) and, for each TSK FFN, its
    firing and its input (needed for the first-order gains)."""
    rec, undo = {}, []
    for li, blk in enumerate(model.blocks):
        mix = blk.mix
        if isinstance(mix, (FuzzyRecurrentMixer, FuzzyDeltaMixer)):
            orig = mix.features

            def feats(x, _orig=orig, _li=li):
                fq, fk, v = _orig(x)
                rec.setdefault(("mix", _li), []).append(fk.detach())  # (B,H,T,R)
                return fq, fk, v

            mix.features = feats
            undo.append((mix, "features", orig))
        if isinstance(blk.ffn, TSKFFN):
            ffn = blk.ffn

            def pre(_m, args, _li=li):
                rec.setdefault(("ffn_x", _li), []).append(args[0].detach())  # (B,T,d)

            h = ffn.register_forward_pre_hook(pre)
            orig_f = ffn.firing

            def firing(x, _orig=orig_f, _li=li):
                w = _orig(x)
                rec.setdefault(("ffn", _li), []).append(w.detach())  # (B,T,R)
                return w

            ffn.firing = firing
            undo.append((ffn, "firing", orig_f))
            undo.append((h, None, None))
    for i in range(0, len(windows), 32):
        model(windows[i : i + 32])
    for obj, name, orig in undo:
        obj.remove() if name is None else setattr(obj, name, orig)
    return {k: torch.cat(v, 0) for k, v in rec.items()}


def describe_rule(memb, windows, codec, top, ctx_len=14):
    """memb: (N, T) activation of one rule over all positions -> strongest contexts and common endings."""
    flat = memb.reshape(-1)
    T = memb.shape[1]
    idx = torch.topk(flat, min(len(flat), top * 20)).indices
    examples, endings = [], Counter()
    for j in idx.tolist():
        n, t = divmod(j, T)
        ctx = windows[n, max(0, t - ctx_len + 1) : t + 1].tolist()
        endings[codec.piece(ctx[-1:])] += 1
        if len(examples) < top:
            examples.append(f"`{codec.piece(ctx)}`")
    return examples, endings.most_common(3)


@torch.no_grad()
def lens(model, vec, codec, k=6):
    """Top-k next tokens the vector pushes towards (final norm, tied unembedding)."""
    p = torch.softmax(model.norm(vec[None]) @ model.emb.weight.T, -1)[0]
    top = torch.topk(p, k)
    return ", ".join(
        f"`{codec.piece([i])}` {v:.2f}"
        for v, i in zip(top.values.tolist(), top.indices.tolist())
    )


def rulebook(model, tok, windows, codec, top=6, max_rules=20) -> str:
    rec = capture(model, windows)
    lines = [
        f"config: `{model.cfg.to_dict()}`",
        f"tokenizer: `{tok or 'character'}`",
        "",
    ]
    for (kind, li), m in sorted(rec.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if kind == "mix":
            N, H, T, R = m.shape
            for h in range(H):
                mh = m[:, h]
                crisp = mh.max(-1).values.mean().item()
                win = torch.bincount(mh.argmax(-1).reshape(-1), minlength=R).float() / (
                    N * T
                )
                lines += [
                    f"## Layer {li} sequence mixer, head {h}: {R} rules (crispness {crisp:.2f}; uniform = {1 / R:.2f})",
                    "",
                ]
                for r in torch.argsort(-win).tolist()[: max_rules or R]:
                    ex, end = describe_rule(mh[..., r], windows, codec, top)
                    lines.append(
                        f"- **rule {r}**: wins {win[r]:.1%} of positions, mean membership {mh[..., r].mean():.3f}"
                    )
                    lines.append(
                        f"  - stores contexts ending in: {', '.join(f'`{s}` ({c})' for s, c in end)}"
                    )
                    lines.append(f"  - strongest: {' · '.join(ex)}")
                lines.append("")
        elif kind == "ffn":
            N, T, R = m.shape
            ffn = model.blocks[li].ffn
            first = getattr(ffn, "order", 0) == 1
            raw = getattr(ffn, "norm", "softmax") == "none"
            win_idx = m.argmax(-1)
            win = torch.bincount(win_idx.reshape(-1), minlength=R).float() / (N * T)
            alive = int((win > 0.001).sum())
            head = f"## Layer {li} TSK feed-forward ({'first' if first else 'zero'}-order, {'raw' if raw else 'normalised'} firing): {R} rules, {alive} win > 0.1% of positions"
            if not raw:
                head += f" (crispness {m.max(-1).values.mean().item():.2f}; uniform = {1 / R:.3f})"
            lines += [head, ""]
            gain = None
            if first:
                x_in = rec[("ffn_x", li)]
                gain = ffn.slope(x_in)  # (N,T,R): rule r's gain at every position
            for r in torch.argsort(-win).tolist()[: max_rules or R]:
                ex, end = describe_rule(m[..., r], windows, codec, top)
                head_r = f"- **rule {r}**: wins {win[r]:.1%}, mean firing {m[..., r].mean():.3f}"
                lines.append(head_r)
                lines.append(
                    f"  - fires on contexts ending in {', '.join(f'`{s}`' for s, _ in end)}; strongest: {' · '.join(ex)}"
                )
                o = ffn.conseq[r].detach()
                if first:
                    wins_here = win_idx == r
                    g = (
                        gain[..., r][wins_here].mean().item()
                        if wins_here.any()
                        else float("nan")
                    )
                    sd = (
                        gain[..., r][wins_here].std().item()
                        if wins_here.sum() > 1
                        else float("nan")
                    )
                    lines.append(
                        f"  - gain where it wins: {g:+.2f} ± {sd:.2f}; THEN raise {lens(model, o, codec)}; when the gain is negative, raise {lens(model, -o, codec)}"
                    )
                else:
                    lines.append(f"  - THEN raise {lens(model, o, codec)}")
            lines.append("")
    return "\n".join(lines) + "\n"


def ffn_rule_rows(model, windows, codec, top=3, max_rules=15) -> list[dict]:
    """Structured form of the FFN part of the rulebook: one dict per rule (most used first, per layer).

    Keys: layer, rule, wins (share of positions where it wins), firing (mean), gain / gain_sd (first-order
    only, else None), fires_on (its most common context endings), examples (strongest contexts), raises
    and raises_neg (logit-lens readings; raises_neg is None for zero-order rules)."""
    rec = capture(model, windows)
    rows = []
    for (kind, li), m in sorted(rec.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if kind != "ffn":
            continue
        N, T, R = m.shape
        ffn = model.blocks[li].ffn
        first = getattr(ffn, "order", 0) == 1
        win_idx = m.argmax(-1)
        win = torch.bincount(win_idx.reshape(-1), minlength=R).float() / (N * T)
        gain = ffn.slope(rec[("ffn_x", li)]) if first else None
        for r in torch.argsort(-win).tolist()[:max_rules]:
            ex, end = describe_rule(m[..., r], windows, codec, top)
            o = ffn.conseq[r].detach()
            g = gd = None
            if first:
                wh = win_idx == r
                g = gain[..., r][wh].mean().item() if wh.any() else None
                gd = gain[..., r][wh].std().item() if wh.sum() > 1 else None
            rows.append(
                {
                    "layer": li,
                    "rule": r,
                    "wins": win[r].item(),
                    "firing": m[..., r].mean().item(),
                    "gain": g,
                    "gain_sd": gd,
                    "fires_on": [s for s, _ in end],
                    "examples": [e.strip("`") for e in ex],
                    "raises": lens(model, o, codec),
                    "raises_neg": lens(model, -o, codec) if first else None,
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument(
        "--max-rules", type=int, default=20, help="rules shown per component (0 = all)"
    )
    a = ap.parse_args()
    model, tok = load_model(a.checkpoint)
    codec = make_codec(tok)
    windows = codec.windows(a.windows, model.cfg.max_len)
    out = Path(a.checkpoint).with_suffix(".rules.md")
    out.write_text(
        f"# Rulebook: `{Path(a.checkpoint).name}`\n\n"
        + rulebook(model, tok, windows, codec, a.top, a.max_rules)
    )
    print(out)


if __name__ == "__main__":
    main()
