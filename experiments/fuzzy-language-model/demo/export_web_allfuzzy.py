"""Export the all-fuzzy model (allfuzzy/e2e_tail_300M_s0.npz) for the in-browser demo: weights, rule views, a parity check; run from the experiment directory.

    python demo/export_web_allfuzzy.py
    python demo/export_web_allfuzzy.py --npz allfuzzy/e2e_all_tanh_300M_s1.npz --id flm-allfuzzy-embed --checkpoint outputs/v59-e2e300/e2e_all_tanh_300M_s1.pt --label "..."

Writes demo/web/models/flm-allfuzzy.{json,bin,rules.json,check.json}, demo/web/tokenizer-3072.json, and appends the model to demo/web/models/index.json.
The rule views are computed with the numpy model of allfuzzy/allfuzzy_lm.py on 16 validation windows: how often each feed-forward rule is the strongest
(|strength| x ||consequent x output weight||) at a position, and the word pieces it raises.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "allfuzzy"))
from allfuzzy_lm import AllFuzzyLM  # noqa: E402

WEB = ROOT / "demo" / "web"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "allfuzzy" / "e2e_tail_300M_s0.npz"))
    ap.add_argument("--id", default="flm-allfuzzy")
    ap.add_argument("--checkpoint", default="outputs/v52-e2e/e2e_tail_300M_s0.pt")
    ap.add_argument(
        "--label",
        default="99K ALL-FUZZY: every block component a fuzzy system, tail token table fuzzy (300M characters)",
    )
    ap.add_argument(
        "--vocab", type=int, default=3072, help="the vocabulary of the checkpoint"
    )
    ap.add_argument(
        "--web",
        default=str(WEB),
        help="the demo folder to write into (default demo/web)",
    )
    ap.add_argument(
        "--valid",
        default="",
        help="validation token file for the rule views (default data/tinystories/valid_bpe<V>.bin)",
    )
    ap.add_argument(
        "--tokenizer-json",
        default="",
        help="tokenizer file of the checkpoint (default allfuzzy/bpe<V>.json)",
    )
    ap.add_argument(
        "--tok-name",
        default="",
        help="tokenizer name in the manifest and the file tokenizer-<name>.json (default bpe<V>)",
    )
    args = ap.parse_args()
    WEB = Path(args.web)
    TOKJSON = (
        Path(args.tokenizer_json)
        if args.tokenizer_json
        else ROOT / "allfuzzy" / f"bpe{args.vocab}.json"
    )
    TOKNAME = args.tok_name or f"bpe{args.vocab}"
    MID = args.id
    V = args.vocab
    m = AllFuzzyLM(args.npz, tokenizer=TOKJSON)
    d, V = m.d, m.V
    # tensors (float32, flat names as in the checkpoint) + the head / tail id lists
    names = [
        k for k in m.W if not k.startswith("cfg_") and k not in ("head_ids", "tail_ids")
    ] + ["head_ids", "tail_ids"]
    table, chunks, off = [], [], 0
    for k in names:
        a = np.asarray(m.W[k], dtype="<f4")
        table.append({"name": k, "shape": list(a.shape), "offset": off})
        chunks.append(a.tobytes())
        off += a.nbytes
    (WEB / "models" / f"{MID}.bin").write_bytes(b"".join(chunks))
    n_params = int(
        m.W["cfg_params_total"]
        if "cfg_params_total" in m.W
        else sum(
            np.asarray(m.W[k]).size for k in names if k not in ("head_ids", "tail_ids")
        )
    )
    manifest = {
        "id": MID,
        "label": args.label,
        "params": n_params,
        "tok": TOKNAME,
        "dtype": "float32",
        "config": {
            "allfuzzy": True,
            "vocab_size": V,
            "d_model": d,
            "n_layers": m.L,
            "n_heads": m.H,
            "n_rules": m.R_mem,
            "max_len": m.max_len,
            "head": int(len(m.head_ids)),
            "tanh": bool("cfg_tanh" in m.W and int(m.W["cfg_tanh"])),
        },
        "tensors": table,
        "checkpoint": args.checkpoint,
    }
    (WEB / "models" / f"{MID}.json").write_text(json.dumps(manifest))
    # rule usage on validation windows
    val = np.fromfile(
        args.valid or ROOT.parent.parent / "data" / "tinystories" / f"valid_bpe{V}.bin",
        dtype=np.uint16,
    )
    wins = [np.zeros(len(m.W[f"blocks.{l}.ffn.centers"])) for l in range(m.L)]
    mean_strength = [np.zeros(len(w)) for w in wins]
    scale = [
        np.linalg.norm(m.W[f"blocks.{l}.ffn.conseq"] * m.W["norm.weight"], axis=1)
        for l in range(m.L)
    ]
    n = 0
    for w in range(16):
        ids = val[900_000 + w * 200 : 900_000 + w * 200 + 128].astype(int)
        m.reset()
        for t in ids:
            _, info = m.step(int(t))
            for l in range(m.L):
                s = info["ffn_strength"][l]
                wins[l][np.argmax(np.abs(s) * scale[l])] += 1
                mean_strength[l] += s
            n += 1
    layers = []
    for l in range(m.L):
        pi = m.profile(l)
        order = np.argsort(-wins[l])[:24]
        rules = []
        for r in order:
            sign = 1.0 if mean_strength[l][r] >= 0 else -1.0
            rules.append(
                {
                    "rule": int(r),
                    "wins_pct": round(100 * wins[l][r] / n, 2),
                    "raises": [m.piece(i) for i in np.argsort(-pi[r] * sign)[:6]],
                }
            )
        layers.append({"layer": l, "rules": rules})
    cfgs = [
        {
            "layer": l,
            "conv": None,
            "mixer": "FuzzyDeltaMixer",
            "heads": m.H,
            "mixer_rules_per_head": m.R_mem,
            "head_dim": m.dh,
            "ffn": "FuzzyFFNRaw",
            "ffn_rules": int(len(m.W[f"blocks.{l}.ffn.centers"])),
            "ffn_antecedent_dim": int(m.W[f"blocks.{l}.ffn.centers"].shape[1]),
            "ffn_order": 1,
            "ffn_norm": "none",
        }
        for l in range(m.L)
    ]
    structure = {
        "d_model": d,
        "n_layers": m.L,
        "layers": cfgs,
        "params": n_params,
        "allfuzzy": True,
    }
    (WEB / "models" / f"{MID}.rules.json").write_text(
        json.dumps(
            {"structure": structure, "rules": {"layers": layers}, "detail": {}},
            separators=(",", ":"),
        )
    )
    # parity reference for the browser test
    ids = list(map(int, np.random.default_rng(1).integers(0, V, 48)))
    m.reset()
    last = None
    for i in ids:
        last, _ = m.step(i)
    (WEB / "models" / f"{MID}.check.json").write_text(
        json.dumps({"ids": ids, "last_logits": [float(x) for x in last]})
    )
    shutil.copyfile(TOKJSON, WEB / f"tokenizer-{TOKNAME.removeprefix('bpe')}.json")
    if (
        V != 3072 and not args.tokenizer_json
    ):  # reference encodings for the browser tokenizer test (3072 has its own, made earlier)
        sys.path.insert(0, str(ROOT))
        from flm.bpe import BPE

        bp = BPE(V)
        texts = [
            "Once upon a time",
            "Lily and Ben went to the park.",
            "The little dog was sad because",
            'One day, a girl named Mia found a shiny red ball!\n\nShe said, "Wow, it\'s so big."',
            "  leading spaces,  double  spaces and a trailing newline\n",
            "It's they're we've I'm you'll he'd",
        ]
        out = [{"text": t, "ids": [int(i) for i in bp.encode(t)]} for t in texts]
        (WEB / f"tokenizer-{V}.check.json").write_text(json.dumps(out))
    idx = [
        e
        for e in json.loads((WEB / "models" / "index.json").read_text())
        if e["id"] != MID
    ]
    idx.append(
        {
            "id": MID,
            "label": manifest["label"],
            "params": n_params,
            "ctx": m.max_len,
            "d": d,
            "layers": m.L,
            "tok": TOKNAME,
        }
    )
    (WEB / "models" / "index.json").write_text(json.dumps(idx, indent=1))
    print(
        f"{MID}: {n_params:,} params, bin {off / 1e6:.2f} MB, {n} validation positions for the rule usage"
    )
