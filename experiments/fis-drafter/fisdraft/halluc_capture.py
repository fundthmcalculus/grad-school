"""Generate a labelled hallucination set, with the confounds designed out.

Design. Each dolly `closed_qa` item supplies a context paragraph, a question,
and a reference answer extractable from that context. Every question is asked
**twice**:

  with_context   context + question -- the model can read the answer off
  no_context     the same question, alone -- it must recall or fabricate

The manipulation is controlled: identical question wording, identical answer
format, only the evidence differs. That is what makes it usable, given that
`experiments/fuzzy-lm-anomaly.md` found a detector reading *prompt style*
reaching ~0.9 AUROC on a task it had no business predicting.

Two rules follow from that study and are enforced here rather than checked
afterwards:

* **The label is measured correctness, never the condition.** Labelling by
  condition would let any detector win by reading prompt length. Each
  generation is graded against the reference answer, and `with_context`
  generations that are wrong are positives just like `no_context` ones.
* **`n_tokens` is recorded as a first-class feature**, because answer length
  alone reached AUROC 0.843 in that study. Any detector that cannot beat it is
  not a detector.

Generation is greedy so the record is deterministic and re-gradeable.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .capture import distribution_stats


def normalise(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, ref: str) -> float:
    """SQuAD-style token F1. Robust to the model being verbose around the fact."""
    p, r = normalise(pred).split(), normalise(ref).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


def recall_of_ref(pred: str, ref: str) -> float:
    """Fraction of reference tokens present. Lenient to the model padding out
    an answer, which a short-answer F1 punishes for the wrong reason."""
    p, r = Counter(normalise(pred).split()), normalise(ref).split()
    if not r:
        return 0.0
    return sum(min(p[t], 1) for t in set(r)) / len(set(r))


@dataclass
class Cfg:
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    n_items: int = 600
    max_new_tokens: int = 40
    batch_size: int = 32
    seed: int = 0
    max_ctx_chars: int = 1200
    out: str = "runs/halluc"


@torch.no_grad()
def run(cfg: Cfg) -> Path:
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.filter(
        lambda r: r["category"] == "closed_qa"
        and 0 < len(r["context"]) <= cfg.max_ctx_chars
        and 0 < len(r["response"]) <= 200
    )
    ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.n_items, len(ds))))

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32)
    model.to("cuda").eval()

    items = []
    for i, r in enumerate(ds):
        for cond in ("with_context", "no_context"):
            q = r["instruction"]
            user = f"{r['context']}\n\n{q}" if cond == "with_context" else q
            text = tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            items.append(
                {
                    "item_id": i,
                    "cond": cond,
                    "question": q,
                    "reference": r["response"],
                    "ids": tok(text, add_special_tokens=False)["input_ids"],
                }
            )

    order = np.argsort([len(x["ids"]) for x in items])
    recs, hid_mean, hid_last, step_rows = [], [], [], []

    for b0 in range(0, len(order), cfg.batch_size):
        bidx = order[b0 : b0 + cfg.batch_size]
        batch = [items[i] for i in bidx]
        L = max(len(x["ids"]) for x in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), L), dtype=torch.long)
        for r_, x in enumerate(batch):
            ids[r_, L - len(x["ids"]) :] = torch.tensor(x["ids"])
            attn[r_, L - len(x["ids"]) :] = 1
        ids, attn = ids.cuda(), attn.cuda()

        past, cur = None, ids
        alive = torch.ones(len(batch), dtype=torch.bool, device="cuda")
        gen = [[] for _ in batch]
        per = [[] for _ in batch]
        hs_acc = [[] for _ in batch]

        for t in range(cfg.max_new_tokens):
            o = model(
                input_ids=cur,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = o.past_key_values
            logits = o.logits[:, -1, :]
            h = o.hidden_states[-1][:, -1, :].float()
            lnorm = torch.stack(
                [x[:, -1, :].float().norm(dim=-1) for x in o.hidden_states], dim=1
            )
            st = distribution_stats(logits)
            nxt = logits.argmax(-1)

            s_cpu = {k: v.cpu().numpy() for k, v in st.items()}
            h_cpu, ln_cpu = h.cpu().numpy(), lnorm.cpu().numpy()
            nx = nxt.cpu().numpy()
            for r_ in range(len(batch)):
                if not bool(alive[r_]):
                    continue
                gen[r_].append(int(nx[r_]))
                per[r_].append({k: float(v[r_]) for k, v in s_cpu.items()})
                hs_acc[r_].append((h_cpu[r_], ln_cpu[r_]))

            alive = alive & (nxt != tok.eos_token_id)
            if not bool(alive.any()):
                break
            cur = nxt.unsqueeze(-1)
            attn = torch.cat(
                [attn, torch.ones((len(batch), 1), dtype=torch.long, device="cuda")], 1
            )

        for r_, x in enumerate(batch):
            if not per[r_]:
                continue
            txt = tok.decode(gen[r_], skip_special_tokens=True).strip()
            P = pd.DataFrame(per[r_])
            rec = {
                "item_id": x["item_id"],
                "cond": x["cond"],
                "question": x["question"],
                "reference": x["reference"],
                "answer": txt,
                "n_tokens": len(gen[r_]),
                "prompt_len": len(x["ids"]),
                "f1": token_f1(txt, x["reference"]),
                "ref_recall": recall_of_ref(txt, x["reference"]),
            }
            # Per-generation aggregates of every per-step statistic. `max` is
            # included for all of them because ent_max beat ent_mean by +0.116
            # in the prior study and the reason was never established -- so the
            # aggregator is treated as a searchable choice, not a default.
            for c in P.columns:
                v = P[c].to_numpy()
                rec[f"{c}__mean"] = float(v.mean())
                rec[f"{c}__max"] = float(v.max())
                rec[f"{c}__min"] = float(v.min())
                rec[f"{c}__last"] = float(v[-1])
                rec[f"{c}__std"] = float(v.std()) if len(v) > 1 else 0.0
            recs.append(rec)
            hs = np.stack([a for a, _ in hs_acc[r_]])
            lns = np.stack([b for _, b in hs_acc[r_]])
            hid_mean.append(hs.mean(0))
            hid_last.append(hs[-1])
            step_rows.append(lns.mean(0))

        print(f"[halluc] {b0 + len(bidx)}/{len(order)}", flush=True)

    df = pd.DataFrame(recs)
    df.to_parquet(outdir / "gens.parquet", index=False)
    np.save(outdir / "hid_mean.npy", np.stack(hid_mean))
    np.save(outdir / "hid_last.npy", np.stack(hid_last))
    np.save(outdir / "layer_norm_mean.npy", np.stack(step_rows))
    (outdir / "meta.json").write_text(
        json.dumps({"config": asdict(cfg), "n": len(df)}, indent=2)
    )

    print(df.groupby("cond")[["f1", "ref_recall", "n_tokens"]].mean().round(3))
    return outdir


def main():
    ap = argparse.ArgumentParser()
    for f, d in asdict(Cfg()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(d), default=d)
    a = ap.parse_args()
    run(Cfg(**vars(a)))


if __name__ == "__main__":
    main()
