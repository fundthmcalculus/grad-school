"""Export a trained ``flm.run_bijective`` checkpoint for the local browser demo (``demo/bijective/``).

    python demo/export_bijective.py --ckpt outputs/bij-demo/c_lr0.04_s0.pt --arm c --ffn-r 32 --ffn-r0 21 --id bij-10k-c --label "..." [--out demo/bijective/models]

Writes ``<out>/<id>.{json,bin,check.json}``: every block tensor of the checkpoint under its own name (the blocks are the all-fuzzy blocks ``flm.js`` already runs), the invertible layer's
parameters (``F.steps.*``), the codebook and tau, and the effective embedding table (z_v = F(C_v), or C_v for the identity arm) in the slot the engine reads as a one-tier embedding
(``emb_h.weight`` with every symbol a head token and an empty tail), so ``AllFuzzyModel`` runs the blocks unchanged and ``demo/bijective/bij.js`` replaces only the output scores.
``check.json`` holds the torch model's logits after a fixed prompt for ``demo/bijective/test_bij.mjs``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--arm", required=True, choices=["tied", "ident", "z", "c", "ci"])
ap.add_argument("--ffn-r", type=int, required=True)
ap.add_argument("--ffn-r0", type=int, required=True)
ap.add_argument("--id", required=True)
ap.add_argument("--label", default="")
ap.add_argument(
    "--val",
    type=float,
    default=0.0,
    help="validation bits per character, shown on the page",
)
ap.add_argument("--out", default="demo/bijective/models")
a = ap.parse_args()
os.environ.update(BIJ=a.arm, FUZZ_FFN_R=str(a.ffn_r), FUZZ_FFN_R0=str(a.ffn_r0))
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import flm.run_bijective as RB  # noqa: E402
from flm.data import encode  # noqa: E402
from flm.models import ModelConfig, TinyLM  # noqa: E402

TinyLM.__init__, TinyLM.embed, TinyLM.logits = RB._new_init, RB._embed, RB._logits
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ck["config"]
model = TinyLM(ModelConfig(**cfg))
model.load_state_dict(ck["state"])
model.eval()
V, d = cfg["vocab_size"], cfg["d_model"]
with torch.no_grad():
    if a.arm == "tied":
        table = model.emb.weight
    elif a.arm == "ident":
        table = model.codebook
    elif a.arm == "ci":
        table = model.G(model.F(model.codebook)[0])
    else:
        table = model.F(model.codebook)[0]
tensors = {
    k: v.detach().float().numpy() for k, v in ck["state"].items() if k != "emb.weight"
}
tensors["emb_h.weight"] = table.detach().float().numpy()
if (
    a.arm == "ci"
):  # the invertible gain: the browser needs its inverse, which reads the log-rms grid; the engine also wants a (here unused) final norm
    from flm.fuzzy_parts import FuzzyNorm  # noqa: E402

    with torch.no_grad():
        ug, vg, du = model.G._grid()
    tensors["G.vg"] = vg.detach().float().numpy()
    for k, v in FuzzyNorm(d, mode="fuzzy").state_dict().items():
        tensors["norm." + k] = v.detach().float().numpy()
tensors["head_ids"] = np.arange(V, dtype=np.float32)
tensors["tail_ids"] = np.zeros(0, dtype=np.float32)
tensors["emb_t.weight"] = np.zeros((0, 1), dtype=np.float32)
tensors["tier_up.weight"] = np.zeros((d, 1), dtype=np.float32)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
chunks, off, meta = [], 0, []
for k, v in tensors.items():
    b = np.ascontiguousarray(v, dtype="<f4").tobytes()
    meta.append({"name": k, "shape": list(v.shape), "offset": off})
    chunks.append(b)
    off += len(b)
(out / f"{a.id}.bin").write_bytes(b"".join(chunks))
steps = (
    sorted({int(k.split(".")[2]) for k in tensors if k.startswith("F.steps.")})
    if a.arm in ("z", "c", "ci")
    else []
)
bij = {
    "arm": a.arm,
    "tau": float(torch.exp(model.log_tau)) if a.arm != "tied" else None,
    "couplings": len(steps),
    "s_max": float(RB.SMAX),
    "g_lo": float(model.G.lo) if a.arm == "ci" else None,
    "g_hi": float(model.G.hi) if a.arm == "ci" else None,
    "g_grid": int(model.G.G) if a.arm == "ci" else None,
    "val_bpc": a.val,
}
man = {
    "id": a.id,
    "label": a.label or a.id,
    "params": int(model.n_params()),
    "dtype": "float32",
    "config": {
        "vocab_size": V,
        "d_model": d,
        "n_heads": cfg["n_heads"],
        "n_rules": cfg["n_rules"],
        "n_layers": cfg["n_layers"],
        "max_len": cfg["max_len"],
        "tanh": 0,
    },
    "bij": bij,
    "tensors": meta,
}
(out / f"{a.id}.json").write_text(json.dumps(man))
prompt = (
    np.concatenate(
        [[0], encode("Once upon a time, there was a little girl named Lily.")]
    )
    .astype(int)
    .tolist()
)
with torch.no_grad():
    logits = model(torch.tensor([prompt]))[0, -1].numpy().tolist()
(out / f"{a.id}.check.json").write_text(
    json.dumps({"ids": prompt, "last_logits": logits})
)
print(f"{a.id}: {man['params']:,} parameters, bin {off / 1e3:.0f} KB, arm {a.arm}")
