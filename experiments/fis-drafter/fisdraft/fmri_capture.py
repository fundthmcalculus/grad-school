"""'fMRI' of an SLM: record every layer's activation while it reads a prompt.

For each probe, one forward pass over the prompt (no generation), capturing at
every layer l = 0..L:

  act_mean[l]   masked mean of the layer's hidden states over the prompt's real
                tokens -- the sustained 'regional activity' as the model reads
  act_last[l]   the layer's hidden state at the final prompt token -- the
                'readout' position where the whole prompt has been integrated

Both are kept because they are different measurements: the mean is aggregate
activity over the whole prompt, the last token is the integrated summary the
model would condition its first generated token on. The anomaly experiment uses
the mean as primary and reports the last-token variant as a check.

The output is a tensor (n_probes, n_layers+1, hidden) per variant, which is the
activation atlas the detector learns 'normal' from.
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
)


@dataclass
class Cfg:
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    batch_size: int = 32
    seed: int = 0
    chat_template: bool = True
    dtype: str = "float32"
    system_prompt: str = ""
    mode: str = "battery"  # or "dose"
    out: str = "runs/fmri"


@torch.no_grad()
def run(cfg: Cfg) -> Path:
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=getattr(torch, cfg.dtype)
    )
    model.to("cuda").eval()

    if cfg.mode == "dose":
        probes = dose_response_battery(seed=cfg.seed)
    elif cfg.mode == "injection":
        probes = injection_battery(seed=cfg.seed, source="deepset")
    elif cfg.mode in ("jailbreak", "safeguard", "spml"):
        probes = injection_battery(seed=cfg.seed, source=cfg.mode)
    else:
        probes = build_anomaly_battery(seed=cfg.seed, tokenizer=tok)

    # encode
    def encode(p):
        if cfg.chat_template and tok.chat_template:
            msgs = (
                [{"role": "system", "content": cfg.system_prompt}]
                if cfg.system_prompt
                else []
            )
            msgs.append({"role": "user", "content": p.text})
            t = tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            return tok(t, add_special_tokens=False)["input_ids"]
        return tok(p.text, add_special_tokens=True)["input_ids"]

    enc = [encode(p) for p in probes]
    order = np.argsort([len(e) for e in enc])

    n_layers = model.config.num_hidden_layers
    D = model.config.hidden_size
    act_mean = np.zeros((len(probes), n_layers + 1, D), dtype=np.float32)
    act_last = np.zeros((len(probes), n_layers + 1, D), dtype=np.float32)
    tok_len = np.zeros(len(probes), dtype=np.int64)

    for b0 in range(0, len(order), cfg.batch_size):
        bidx = order[b0 : b0 + cfg.batch_size]
        batch = [enc[i] for i in bidx]
        L = max(len(b) for b in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), L), dtype=torch.long)
        for r, b in enumerate(batch):
            ids[r, L - len(b) :] = torch.tensor(b)
            attn[r, L - len(b) :] = 1
        ids, attn = ids.cuda(), attn.cuda()

        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        m = attn.unsqueeze(-1).float()  # (B, L, 1)
        denom = m.sum(1).clamp_min(1.0)
        for l, hs in enumerate(out.hidden_states):
            hs = hs.float()
            mean_l = (hs * m).sum(1) / denom  # (B, D)
            last_l = hs[:, -1, :]  # left-padded, so -1 is the last real token
            act_mean[bidx, l] = mean_l.cpu().numpy()
            act_last[bidx, l] = last_l.cpu().numpy()
        for r, i in enumerate(bidx):
            tok_len[i] = len(batch[r])
        print(f"[fmri] {b0 + len(bidx)}/{len(order)}", flush=True)

    np.save(outdir / "act_mean.npy", act_mean)
    np.save(outdir / "act_last.npy", act_last)
    np.save(outdir / "tok_len.npy", tok_len)

    import pandas as pd

    df = pd.DataFrame(
        {
            "idx": range(len(probes)),
            "text": [p.text for p in probes],
            "label": [p.label for p in probes],
            "tok_len": tok_len,
        }
    )
    df.to_parquet(outdir / "probes.parquet", index=False)
    meta = {
        "config": asdict(cfg),
        "n_probes": len(probes),
        "n_layers": n_layers,
        "hidden": D,
        "labels": df.label.value_counts().to_dict(),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["labels"], indent=2))
    return outdir


def main():
    ap = argparse.ArgumentParser()
    for f, d in asdict(Cfg()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(d), default=d)
    a = ap.parse_args()
    run(Cfg(**vars(a)))


if __name__ == "__main__":
    main()
