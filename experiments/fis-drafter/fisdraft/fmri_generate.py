"""Generation-time 'fMRI' + output-logit-shape capture.

The prompt monitor (fmri_capture.py) reads only the *prompt* forward pass. This
module extends the atlas into generation: it greedy-decodes K tokens and, at every
generated step, records two things that the prompt pass cannot see:

  1. streaming hidden states -- the per-layer activation at each freshly generated
     token position. Aggregated to gen_mean / gen_last with the SAME
     (n_probes, n_layers+1, hidden) shape as the prompt atlas, so the same detector
     and the same layer_features pipeline apply unchanged and the two are directly
     comparable.

  2. output logit shape -- for each step, the shape of the next-token distribution
     the model actually emits: entropy, top-1 mass, top-5 mass, top1-top2 logit
     margin, and effective support (perplexity). This is the original 'the logit
     SHAPE will tell us something' theory, made measurable. These are reduced to
     scalars on the GPU per step (never materialising the 150k-vocab logits for all
     probes) and aggregated to a per-probe shape-feature vector.

Step indexing: generate's hidden_states[0] is the prompt forward pass, whose final
position is exactly the prompt readout (== act_last). We therefore take the readout
of step 0 and the hidden states of the freshly generated tokens (steps >=1) as the
streaming trajectory. scores[t] is the logit vector that produced generated token t.

Greedy decoding is used so the trajectory is deterministic; sampling is a separate
axis (it perturbs which tokens are drawn, hence the downstream activations and
scores) and is exposed via --do-sample for the sampling-sensitivity check.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts_anomaly import (
    build_anomaly_battery,
    dose_response_battery,
    injection_battery,
    embedded_injection_battery,
)

# per-step logit-shape scalar names, in column order
SHAPE_STEP = ["entropy", "max_prob", "top5_mass", "margin", "ppl"]


@dataclass
class Cfg:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    batch_size: int = 16
    seed: int = 0
    chat_template: bool = True
    dtype: str = "bfloat16"
    system_prompt: str = ""
    max_new_tokens: int = 24
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    mode: str = "injection"
    out: str = "runs/gen_qwen3b"


@torch.no_grad()
def _step_shape(logits: torch.Tensor) -> torch.Tensor:
    """(B, V) logits -> (B, 5) shape scalars, all on-device."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(-1)  # nats
    top5 = torch.topk(p, 5, dim=-1).values
    max_prob = top5[:, 0]
    top5_mass = top5.sum(-1)
    top2_logit = torch.topk(logits.float(), 2, dim=-1).values
    margin = top2_logit[:, 0] - top2_logit[:, 1]
    ppl = torch.exp(entropy)
    return torch.stack([entropy, max_prob, top5_mass, margin, ppl], dim=-1)


@torch.no_grad()
def run(cfg: Cfg) -> Path:
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model_id, dtype=getattr(torch, cfg.dtype)
        )
        .to("cuda")
        .eval()
    )

    if cfg.mode == "dose":
        probes = dose_response_battery(seed=cfg.seed)
    elif cfg.mode == "injection":
        probes = injection_battery(seed=cfg.seed, source="deepset")
    elif cfg.mode in ("jailbreak", "safeguard", "spml"):
        probes = injection_battery(seed=cfg.seed, source=cfg.mode)
    elif cfg.mode == "embedded":
        probes = embedded_injection_battery(seed=cfg.seed)
    else:
        probes = build_anomaly_battery(seed=cfg.seed, tokenizer=tok)

    def encode(p):
        if cfg.chat_template and tok.chat_template:
            msgs = (
                [{"role": "system", "content": cfg.system_prompt}]
                if cfg.system_prompt
                else []
            )
            msgs.append({"role": "user", "content": p.text})
            t = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            return tok(t, add_special_tokens=False)["input_ids"]
        return tok(p.text, add_special_tokens=True)["input_ids"]

    enc = [encode(p) for p in probes]
    order = np.argsort([len(e) for e in enc])

    n_layers = model.config.num_hidden_layers
    D = model.config.hidden_size
    K = cfg.max_new_tokens
    n = len(probes)
    gen_mean = np.zeros((n, n_layers + 1, D), dtype=np.float32)
    gen_last = np.zeros((n, n_layers + 1, D), dtype=np.float32)
    readout = np.zeros((n, n_layers + 1, D), dtype=np.float32)  # step-0 == act_last
    shape_steps = np.full((n, K, len(SHAPE_STEP)), np.nan, dtype=np.float32)
    gen_len = np.zeros(n, dtype=np.int64)
    tok_len = np.zeros(n, dtype=np.int64)

    gen_kw = dict(
        max_new_tokens=K,
        do_sample=cfg.do_sample,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=True,
        pad_token_id=tok.pad_token_id,
    )
    if cfg.do_sample:
        gen_kw.update(temperature=cfg.temperature, top_p=cfg.top_p)

    for b0 in range(0, n, cfg.batch_size):
        bidx = order[b0 : b0 + cfg.batch_size]
        batch = [enc[i] for i in bidx]
        L = max(len(b) for b in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), L), dtype=torch.long)
        for r, b in enumerate(batch):
            ids[r, L - len(b) :] = torch.tensor(b)
            attn[r, L - len(b) :] = 1
        ids, attn = ids.cuda(), attn.cuda()

        out = model.generate(input_ids=ids, attention_mask=attn, **gen_kw)
        steps = len(out.scores)  # actual generated steps
        B = len(batch)

        # which generated positions are real (not past EOS)? track per-row eos.
        seq = out.sequences[:, L:]  # (B, steps)
        alive = torch.ones(B, dtype=torch.bool, device=ids.device)
        stepmask = torch.zeros((B, steps), dtype=torch.bool, device=ids.device)
        for t in range(steps):
            stepmask[:, t] = alive
            alive = alive & (seq[:, t] != tok.eos_token_id)
        gl = stepmask.sum(1).clamp_min(1)  # >=1 to avoid /0

        # streaming hidden states, aggregated on the fly, per layer
        # step 0: prompt forward, readout = last real prompt position (== act_last)
        for l in range(n_layers + 1):
            hs0 = out.hidden_states[0][l][:, -1, :].float()  # (B, D)
            readout[bidx, l] = hs0.cpu().numpy()
        # accumulate gen_mean / gen_last over generated steps (t>=1 emit one token)
        acc = torch.zeros((B, n_layers + 1, D), device=ids.device)
        last = torch.zeros((B, n_layers + 1, D), device=ids.device)
        for t in range(1, steps):
            m = stepmask[:, t].float().view(B, 1)  # (B,1) per-row alive
            for l in range(n_layers + 1):
                h = out.hidden_states[t][l][:, -1, :].float()  # (B, D)
                acc[:, l] += h * m
                # update last where this row is still alive
                last[:, l] = torch.where(m.bool(), h, last[:, l])
        denom = (gl - 1).clamp_min(1).float().view(B, 1, 1)  # steps 1.. counted
        gm = acc / denom
        gen_mean[bidx] = gm.cpu().numpy()
        gen_last[bidx] = last.cpu().numpy()

        # logit shape per step
        for t in range(steps):
            sh = _step_shape(out.scores[t])  # (B, 5)
            valid = stepmask[:, t].cpu().numpy()
            arr = sh.cpu().numpy()
            arr[~valid] = np.nan
            shape_steps[bidx, t] = arr

        for r, i in enumerate(bidx):
            gen_len[i] = int(gl[r].item())
            tok_len[i] = len(batch[r])
        print(f"[gen] {b0 + len(bidx)}/{n}", flush=True)
        del out, acc, last
        torch.cuda.empty_cache()

    np.save(outdir / "gen_mean.npy", gen_mean)
    np.save(outdir / "gen_last.npy", gen_last)
    np.save(outdir / "readout.npy", readout)
    np.save(outdir / "shape_steps.npy", shape_steps)
    np.save(outdir / "gen_len.npy", gen_len)
    np.save(outdir / "tok_len.npy", tok_len)

    import pandas as pd

    df = pd.DataFrame(
        {
            "idx": range(n),
            "text": [p.text for p in probes],
            "label": [p.label for p in probes],
            "tok_len": tok_len,
            "gen_len": gen_len,
        }
    )
    df.to_parquet(outdir / "probes.parquet", index=False)
    meta = {
        "config": asdict(cfg),
        "n_probes": n,
        "n_layers": n_layers,
        "hidden": D,
        "max_new_tokens": K,
        "shape_step_cols": SHAPE_STEP,
        "labels": df.label.value_counts().to_dict(),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["labels"], indent=2))
    return outdir


def main():
    ap = argparse.ArgumentParser()
    for f, d in asdict(Cfg()).items():
        if isinstance(d, bool):
            ap.add_argument(f"--{f.replace('_', '-')}", action="store_true")
        else:
            ap.add_argument(f"--{f.replace('_', '-')}", type=type(d), default=d)
    a = ap.parse_args()
    run(Cfg(**vars(a)))


if __name__ == "__main__":
    main()
