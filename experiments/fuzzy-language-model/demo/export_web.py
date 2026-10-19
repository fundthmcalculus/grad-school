"""Export trained FLM checkpoints for the in-browser demo (``demo/web``).

    .venv/bin/python demo/export_web.py            # all models listed in MODELS

Per model this writes, under ``demo/web/models/``:

* ``<id>.bin``  float32 (or float16, ``fp16``) tensors, concatenated; ``<id>.json`` manifest: label, config, tensor table.
* ``<id>.rules.json``  the precomputed rule views of the Python demo (structure card, most used rules, every FFN rule's
  membership functions and data histograms over 48 validation windows) so the page needs no data and no server.

The browser engine (``web/flm.js``) implements the dense tied-embedding model. A frequency-tiered checkpoint
(``flm.run_tiered``) is *densified* first: its input embedding of token i and its tied output row are both
``model.embed(i)`` (head rows are ``emb_h``, tail rows are ``tier_up @ emb_t``), so a plain ``(V, d)`` matrix reproduces
it exactly; the export verifies that against the original model's logits before writing anything.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import rule_views as S  # noqa: E402  (structure / rule_list / rule_detail)
from flm.models import ModelConfig, TinyLM  # noqa: E402
from flm.rules import make_codec  # noqa: E402

WEB = HERE / "web"
# id, checkpoint, label, tiered (head, tail dims), dtype
MODELS = [
    (
        "flm96k",
        "outputs/v18-3b/D-d48-bpe1024-ctx512-r32_3000M_s0.pt",
        "96K, 3B characters (reference recipe)",
        None,
        "float32",
    ),
    (
        "flm98k-tiered",
        "outputs/v35-tiered1b/t768e16m225_1000M_s1.pt",
        "98K tiered embedding, 1B characters (best sub-100K likelihood)",
        (768, 16),
        "float32",
    ),
    (
        "flm1m-over-cap",
        "outputs/v29-big1b/d192-L4_1000M_s0.pt",
        "1.32M, 1B characters (over the 100K cap, for comparison)",
        None,
        "float16",
    ),
]
GRID = {"n_windows": 48, "bins": 24}


def load_dense(path: Path, tiered):
    """Return (dense stock TinyLM, config dict, token-level n_params of the ORIGINAL checkpoint)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"])
    if tiered is None:
        orig = TinyLM(cfg)
        orig.load_state_dict(ck["state"])
    else:
        import flm.run_tiered as R

        R.NH, R.E = tiered
        saved = (
            TinyLM.__init__,
            TinyLM.embed,
            TinyLM.logits,
            TinyLM.n_params,
        )
        TinyLM.__init__, TinyLM.embed, TinyLM.logits, TinyLM.n_params = (
            R._new_init,
            R._embed,
            R._logits,
            R._n_params,
        )
        import os

        os.environ.setdefault("TIER_FREQ", "outputs/v22-tribble/ppmi_k48_v1024.npz")
        orig = TinyLM(cfg)
        orig.load_state_dict(ck["state"])
    orig.eval()
    n_params = orig.n_params()
    with torch.no_grad():
        E = orig.embed(
            torch.arange(cfg.vocab_size)
        ).clone()  # (V, d) dense tied embedding
        probe = torch.randint(0, cfg.vocab_size, (2, 64))
        ref = orig(probe)
    if tiered is not None:
        TinyLM.__init__, TinyLM.embed, TinyLM.logits, TinyLM.n_params = saved
    dense_cfg = ModelConfig(**{**ck["config"], "emb_dim": 0})
    dense = TinyLM(dense_cfg)
    sd = {
        k: v
        for k, v in ck["state"].items()
        if not k.startswith(("emb", "tier_up", "up."))
    }
    sd["emb.weight"] = E
    dense.load_state_dict(sd)
    dense.eval()
    with torch.no_grad():
        err = (dense(probe) - ref).abs().max().item()
    assert err < 1e-4, f"densified model differs from the original: max |dlogit| {err}"
    print(f"  densified: max |dlogit| vs the original model = {err:.2e}")
    return dense, ck["config"], n_params, ck.get("tok", "")


def tensors_of(model: TinyLM) -> dict[str, np.ndarray]:
    out = {"emb": model.emb.weight, "norm": model.norm.weight}
    for li, b in enumerate(model.blocks):
        p = f"l{li}."
        out[p + "n1"] = b.n1.weight
        out[p + "conv"] = b.conv.conv.weight[:, 0, :]  # (d, width)
        m, f = b.mix, b.ffn
        out[p + "m.centers"] = m.centers
        out[p + "m.log_width"] = m.log_width
        out[p + "m.qkv"] = m.qkv.weight
        out[p + "m.out"] = m.out.weight
        out[p + "m.gate_w"] = m.gate.weight
        out[p + "m.gate_b"] = m.gate.bias
        out[p + "m.beta_w"] = m.beta.weight
        out[p + "m.beta_b"] = m.beta.bias
        out[p + "n2"] = b.n2.weight
        out[p + "f.centers"] = f.centers
        out[p + "f.log_width"] = f.log_width
        out[p + "f.conseq"] = f.conseq
        out[p + "f.ante"] = f.ante.weight
        out[p + "f.slope_w"] = f.slope.weight
        out[p + "f.slope_b"] = f.slope.bias
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out.items()}


def pct(h):  # histogram (max 1) -> integers 0..100
    return [int(round(100 * x)) for x in h]


def rule_views(mid: str, dense: TinyLM, tok: str) -> dict:
    """The Python demo's precomputed views for one model, via its own functions (the model is injected into its cache)."""
    S.CACHE[mid] = (dense, make_codec(tok))
    S.STATS.pop(mid, None)
    st = S.structure(mid)
    rl = S.rule_list(mid)
    detail = {}
    listed = {(L["layer"], r["rule"]) for L in rl["layers"] for r in L["rules"]}
    for li, blk in enumerate(dense.blocks):
        # small models: every rule; large ones: only the listed (most used) rules, the page falls back to the
        # membership functions read from the weights for any other rule
        for r in range(blk.ffn.R):
            if blk.ffn.R * len(dense.blocks) > 300 and (li, r) not in listed:
                continue
            d = S.rule_detail(mid, li, r)
            for dim in d["dims"]:
                dim["all"], dim["fires"] = pct(dim["all"]), pct(dim["fires"])
            detail[f"{li}.{r}"] = d
        print(f"  rule detail: layer {li} done", flush=True)
    return {"structure": st, "rules": rl, "detail": detail}


def main():
    (WEB / "models").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT.parents[1] / "data" / "tinystories" / "bpe1024.json",
        WEB / "tokenizer.json",
    )
    index = []
    for mid, ck, label, tiered, dtype in MODELS:
        print(mid, flush=True)
        dense, cfg, n_params, tok = load_dense(ROOT / ck, tiered)
        ts = tensors_of(dense)
        table, off, chunks = [], 0, []
        for name, a in ts.items():
            b = a.astype("<f4" if dtype == "float32" else "<f2").tobytes()
            table.append({"name": name, "shape": list(a.shape), "offset": off})
            chunks.append(b)
            off += len(b)
        (WEB / "models" / f"{mid}.bin").write_bytes(b"".join(chunks))
        manifest = {
            "id": mid,
            "label": label,
            "params": n_params,
            "tok": tok,
            "dtype": dtype,
            "config": cfg,
            "tensors": table,
            "checkpoint": ck,
        }
        (WEB / "models" / f"{mid}.json").write_text(json.dumps(manifest))
        views = rule_views(mid, dense, tok)
        (WEB / "models" / f"{mid}.rules.json").write_text(
            json.dumps(views, separators=(",", ":"))
        )
        # reference logits for the browser self-test (float32 dense model, fixed token sequence)
        ids = torch.randint(
            0, 1024, (1, 48), generator=torch.Generator().manual_seed(1)
        )
        with torch.no_grad():
            logits = dense(ids)[0, -1]
        (WEB / "models" / f"{mid}.check.json").write_text(
            json.dumps(
                {"ids": ids[0].tolist(), "last_logits": [float(x) for x in logits]}
            )
        )
        index.append(
            {
                "id": mid,
                "label": label,
                "params": n_params,
                "ctx": cfg["max_len"],
                "d": cfg["d_model"],
                "layers": cfg["n_layers"],
                "tok": tok,
            }
        )
        print(
            f"  {mid}: {n_params:,} params, bin {off / 1e6:.2f} MB, "
            f"rules {(WEB / 'models' / (mid + '.rules.json')).stat().st_size / 1e6:.2f} MB",
            flush=True,
        )
    (WEB / "models" / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
