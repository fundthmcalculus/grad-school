"""Inference memory per arm: floats of state carried between generated tokens.

    .venv/bin/python -m flm.state --d 32 --layers 2 --heads 2 --rules 16

Quadratic mixers carry a KV cache that grows with context t (2 * d floats per token per
layer). Recurrent mixers carry a fixed state:
  linear/gla : per head  dh x dh (S) + dh (z)
  delta      : per head  dh x dh
  fuzzy      : per head  R x dh (S) + R (z)      -- R rule consequents + their evidence
  fuzzydelta : per head  R x dh                   -- R rule consequents
  gru        : d per layer
plus (shortconv - 1) * d per layer for the causal conv buffer, if enabled.
The numbers are read from instantiated modules where possible, not just from formulas.
"""

from __future__ import annotations

import argparse

from flm.models import ModelConfig, TinyLM


def state_floats(
    mixer: str, d: int, L: int, H: int, R: int, t: int, shortconv: int = 0
) -> int:
    dh = d // H
    conv = (shortconv - 1) * d if shortconv > 1 else 0
    per_layer = {
        "softmax": 2 * d * t,
        "gauss": 2 * d * t,
        "linear": H * (dh * dh + dh),
        "delta": H * dh * dh,
        "fuzzy": H * (R * dh + R),
        "fuzzydelta": H * R * dh,
        "gru": d,
    }[mixer]
    return L * (per_layer + (0 if mixer == "gru" else conv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--rules", type=int, default=16)
    a = ap.parse_args()
    print(
        f"| mixer | params (d={a.d}, L={a.layers}) | state @ t=256 | @ t=4096 | @ t=65536 |"
    )
    print("|---|---|---|---|---|")
    for mixer, ffn in [
        ("softmax", "mlp"),
        ("gauss", "tsk"),
        ("linear", "mlp"),
        ("delta", "mlp"),
        ("fuzzy", "tsk"),
        ("fuzzydelta", "tsk"),
        ("gru", "none"),
    ]:
        m = TinyLM(
            ModelConfig(
                vocab_size=98,
                d_model=a.d,
                n_layers=a.layers,
                n_heads=a.heads,
                mixer=mixer,
                ffn=ffn,
                n_rules=a.rules,
            )
        )
        cells = [
            f"{state_floats(mixer, a.d, a.layers, a.heads, a.rules, t):,}"
            for t in (256, 4096, 65536)
        ]
        print(f"| {mixer}+{ffn} | {m.n_params():,} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
