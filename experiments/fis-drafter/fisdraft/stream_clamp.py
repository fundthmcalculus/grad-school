"""Causal clamp: project out the exogenous-response subspace and see what changes.

Part 20 found the exogenous response is high-dimensional (no single axis). This
asks the causal question the 'clamp it' idea really wants: if we remove the rank-k
injection-response *subspace* at the responsive layer during the forward pass, does
the model's downstream response and behaviour actually change -- or does it just
reconstruct the response from other directions?

Subspace B (k orthonormal rows) = top-k principal directions of the paired
differential injection onset vectors (injection minus benign_cont, same base) at
layer L, taken from the streaming capture. A random orthonormal subspace of equal
rank is the control -- if clamping B and clamping random are equally (in)effective,
B is not special.

Intervention: a forward hook on decoder block L-1 replaces its output h with
h - (h @ B^T) B at every position (projecting the subspace out). Two readouts:

  downstream shift   after clamping at L, recompute the within-sequence rms-z shift
                     at layers > L for injection probes (paired vs benign_cont). If
                     the response is causally carried by B, downstream shift drops;
                     if the model rebuilds it, downstream shift persists.
  behavioural KL     KL(next-token dist clamped || unclamped) at the final token.
                     How much clamping B changes what the model would actually say.

Reported across k in {1,2,4,8,16,32} for B and for the random control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fmri_stream import build, Cfg


def subspace(rundir: Path, layer: int, k: int, fit_bases=None) -> np.ndarray:
    """Top-k principal directions of paired differential injection onset @ layer.

    If fit_bases is given, only those base ids contribute (held-out validation).
    """
    df = pd.read_parquet(rundir / "probes.parquet").reset_index(drop=True)
    win = np.load(rundir / "win.npy")
    wv = np.load(rundir / "win_valid.npy")
    W = win.shape[1] // 2
    bybase = {}
    for i in range(len(df)):
        bybase.setdefault(int(df.base[i]), {})[df.dose_name[i]] = i

    def onset(i):
        pre = win[i, :W, layer, :][wv[i, :W]]
        return (
            win[i, W, layer, :] - pre.mean(0) if (len(pre) >= 2 and wv[i, W]) else None
        )

    V = []
    for base, b in bybase.items():
        if fit_bases is not None and base not in fit_bases:
            continue
        if "injection" in b and "benign_cont" in b:
            a, z = onset(b["injection"]), onset(b["benign_cont"])
            if a is not None and z is not None:
                V.append(a - z)
    V = np.array(V, dtype=np.float64)
    V = V - V.mean(0)
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    return Vt[:k].astype(np.float32)  # (k, D) orthonormal


@torch.no_grad()
def run(
    rundir: Path,
    layer: int = 28,
    ks=(1, 2, 4, 8, 16, 32),
    n_probe: int = 60,
    holdout: bool = False,
):
    meta = json.loads((rundir / "meta.json").read_text())
    cfg = Cfg(**meta["config"])
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model_id, dtype=getattr(torch, cfg.dtype)
        )
        .to("cuda")
        .eval()
    )
    Lp1 = model.config.num_hidden_layers + 1
    D = model.config.hidden_size

    probes = build(cfg)
    bybase = {}
    for i, p in enumerate(probes):
        bybase.setdefault(p["base"], {})[p["dose_name"]] = p
    # held-out validation: fit the subspace on even bases, evaluate on odd bases
    all_bases = sorted(bybase)
    fit_bases = set(all_bases[::2]) if holdout else None
    eval_bases = [b for b in all_bases if (not holdout or b % 2 == 1)]
    # a matched set of (injection, benign_cont) prompt pairs on the eval bases
    pairs = [
        (bybase[b]["injection"], bybase[b]["benign_cont"])
        for b in eval_bases
        if "injection" in bybase[b] and "benign_cont" in bybase[b]
    ][:n_probe]

    def encode(p):
        pre = tok(p["preamble"] + "\n\n", add_special_tokens=True)["input_ids"]
        full = tok(p["preamble"] + "\n\n" + p["payload"], add_special_tokens=True)[
            "input_ids"
        ]
        return full, len(pre)

    # hook machinery on decoder block (layer-1): projects out current subspace B
    state = {"B": None}
    blk = model.model.layers[layer - 1]

    def hook(module, inp, out):
        if state["B"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        B = state["B"].to(h.dtype)  # (k, D) torch
        h = h - (h @ B.t()) @ B
        if isinstance(out, tuple):
            return (h,) + tuple(out[1:])
        return h

    handle = blk.register_forward_hook(hook)

    def forward(ids_list, bnds, collect_layers):
        L = max(len(x) for x in ids_list)
        ids = torch.full((len(ids_list), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(ids_list), L), dtype=torch.long)
        for r, b in enumerate(ids_list):
            ids[r, : len(b)] = torch.tensor(b)
            attn[r, : len(b)] = 1
        out = model(
            input_ids=ids.cuda(), attention_mask=attn.cuda(), output_hidden_states=True
        )
        # last-real-token logits per row (right padded -> index len-1)
        logits = torch.stack(
            [out.logits[r, len(ids_list[r]) - 1] for r in range(len(ids_list))]
        )
        hs = {l: out.hidden_states[l].float().cpu().numpy() for l in collect_layers}
        return logits.float(), hs

    inj_ids, inj_b = zip(*[encode(a) for a, _ in pairs])
    ben_ids, ben_b = zip(*[encode(b) for _, b in pairs])
    inj_ids, ben_ids = list(inj_ids), list(ben_ids)
    down_layers = list(range(layer, Lp1))

    def rmsz_shift(hs, ids_list, bnds):
        """paired within-seq rms-z shift at each down layer, per probe (payload onset)."""
        out = {l: [] for l in down_layers}
        for l in down_layers:
            H = hs[l]
            for r in range(len(ids_list)):
                s = len(ids_list[r])
                b = bnds[r]
                h = H[r, :s]
                base = h[:b]
                mu = base.mean(0)
                sd = base.std(0).clip(1e-4)
                z = (h - mu) / sd
                out[l].append(float(np.sqrt((z[b : b + 3] ** 2).mean())))
        return {l: np.array(v) for l, v in out.items()}

    results = {"model": cfg.model_id, "layer": layer, "arms": {}}

    # baseline (no clamp)
    state["B"] = None
    inj_logit0, inj_hs0 = forward(inj_ids, inj_b, down_layers)
    _, ben_hs0 = forward(ben_ids, ben_b, down_layers)
    sh0 = rmsz_shift(inj_hs0, inj_ids, inj_b)
    shb0 = rmsz_shift(ben_hs0, ben_ids, ben_b)
    base_down = {l: float((sh0[l] - shb0[l]).mean()) for l in down_layers}
    results["baseline_downstream_shift"] = {
        int(l): round(v, 3) for l, v in base_down.items()
    }

    rng = np.random.default_rng(0)
    for k in ks:
        for arm, B in (
            ("clamp", subspace(rundir, layer, k, fit_bases)),
            ("random", np.linalg.qr(rng.normal(size=(D, k)))[0].T.astype(np.float32)),
        ):
            state["B"] = torch.tensor(B).cuda()
            inj_logit, inj_hs = forward(inj_ids, inj_b, down_layers)
            _, ben_hs = forward(ben_ids, ben_b, down_layers)
            sh = rmsz_shift(inj_hs, inj_ids, inj_b)
            shb = rmsz_shift(ben_hs, ben_ids, ben_b)
            # downstream shift retained, averaged over layers > L
            up = [l for l in down_layers if l > layer]
            retained = float(np.mean([(sh[l] - shb[l]).mean() for l in up]))
            base_ret = float(np.mean([base_down[l] for l in up]))
            # behavioural: KL(clamped || unclamped) at final token, injection prompts
            p_c = torch.log_softmax(inj_logit, -1)
            p_0 = torch.log_softmax(inj_logit0, -1)
            kl = float((p_c.exp() * (p_c - p_0)).sum(-1).mean())
            state["B"] = None
            results["arms"].setdefault(arm, {})[k] = {
                "downstream_shift_retained": round(retained, 3),
                "frac_of_baseline": round(retained / base_ret, 3) if base_ret else None,
                "behav_KL_vs_unclamped": round(kl, 3),
            }
    handle.remove()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/stream_qwen3b")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    r = run(Path(a.run), layer=a.layer, holdout=a.holdout)
    r["holdout"] = a.holdout
    fn = "stream_clamp_holdout.json" if a.holdout else "stream_clamp.json"
    (Path(a.run) / fn).write_text(json.dumps(r, indent=2))
    print(
        f"Causal clamp @L{r['layer']}{' (HELD-OUT)' if a.holdout else ''} -- {r['model']}"
    )
    base_up = np.mean(
        [v for l, v in r["baseline_downstream_shift"].items() if l > r["layer"]]
    )
    print(
        f"baseline downstream shift (layers >{r['layer']}, injection paired) = {base_up:.3f}\n"
    )
    print(
        f"{'k':>4}  {'arm':<8}{'shift_retained':>16}{'frac_baseline':>15}{'behav_KL':>12}"
    )
    for k in sorted({k for arm in r["arms"].values() for k in arm}):
        for arm in ("clamp", "random"):
            v = r["arms"][arm][k]
            print(
                f"{k:>4}  {arm:<8}{v['downstream_shift_retained']:>16.3f}"
                f"{str(v['frac_of_baseline']):>15}{v['behav_KL_vs_unclamped']:>12.3f}"
            )


if __name__ == "__main__":
    main()
