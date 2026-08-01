"""Stage 2 -- run the frozen SLM and record what it did internally.

The model is never modified: weights are loaded in eval/inference mode and we
only read activations and logits back out.

For every prompt we record

* the decoded answer plus its ground-truth label,
* per-generated-token distribution statistics (entropy, max-prob, margin,
  chosen log-prob) aggregated over the answer,
* the mean-pooled residual-stream activation at **every** layer, pooled over
  generated tokens only.

Pooling over generated tokens (not the prompt) matters: the prompt is shared
across a whole template family, so prompt activations would let the detector
cheat by recognising the template rather than the failure.

Writes ``capture_meta.parquet`` (scalar features + labels) and
``capture_hidden.npy`` (N, n_layers+1, hidden) float16.
"""

import argparse
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path(__file__).parent / "data"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Phrases that mark the model declining or flagging the premise. Matching any of
# these makes the response an abstention rather than a fabrication.
_ABSTAIN = re.compile(
    # 1. explicit ignorance
    r"\b(i (?:do not|don't) know|i'm not (?:sure|aware|certain)|i am not (?:sure|aware|certain)|"
    r"not aware of any|"
    # 2. denial of the premise -- the desired response to a false-premise probe
    r"no such|does not exist|doesn't exist|did not exist|never existed|not a real|"
    r"is fictional|appears to be fictional|is not a recognized|no record of|"
    r"there is no|there was no|no known|not a valid|"
    # 3. capability / assistant-persona refusals
    r"as an ai|as a text-based ai|i'm just an ai|i am just an ai|"
    r"i (?:do not|don't) have (?:the ability|access|any information|information|data)|"
    r"i (?:cannot|can't|can not) (?:find|provide|access|answer|verify)|"
    r"i'm (?:unable|not able)|i am (?:unable|not able)|unable to (?:find|provide|verify|access)|"
    r"my training data|my knowledge (?:cutoff|base)|outside of my train)\b",
    re.IGNORECASE,
)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    """SQuAD-style normalisation for exact match."""
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def label_row(row: dict, answer: str) -> str:
    """Ground-truth label for one generation.

    Returns one of ``correct`` / ``abstain`` / ``hallucination``.
    """
    if _ABSTAIN.search(answer):
        return "abstain"
    if row["family"] == "falsepremise":
        # Subject does not exist, and the model did not push back.
        return "hallucination" if answer.strip() else "abstain"
    # v3 long-form probes are labelled by GROUNDEDNESS, not correctness: a
    # paragraph cannot be graded reliably, but a real subject anchors the answer
    # to something and an invented subject cannot. See build_prompts_v3.py.
    if row["family"] == "longform_real":
        return "grounded" if answer.strip() else "abstain"
    if row["family"] == "longform_fake":
        return "hallucination" if answer.strip() else "abstain"
    norm = normalize(answer)
    if not norm:
        return "abstain"
    if any(normalize(a) and normalize(a) in norm for a in row["answers"]):
        return "correct"
    return "hallucination"


@torch.inference_mode()
def run_batch(model, tok, questions, max_new_tokens):
    """Generate for a batch and pull out per-example features.

    Returns (texts, stats list, pooled hidden (B, L+1, H) float16).
    """
    chats = [
        tok.apply_chat_template([{"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    enc = tok(chats, return_tensors="pt", padding=True).to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tok.pad_token_id,
    )

    seqs = out.sequences[:, prompt_len:]              # (B, T)
    B, T = seqs.shape
    eos = tok.eos_token_id

    # Valid mask: tokens up to and including the first EOS.
    is_eos = seqs == eos
    first_eos = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((B,), T - 1,
                                                                               device=seqs.device))
    idx = torch.arange(T, device=seqs.device).unsqueeze(0)
    valid = idx <= first_eos.unsqueeze(1)             # (B, T) bool
    lengths = valid.sum(1).clamp(min=1)

    # --- distribution statistics ------------------------------------------
    # out.scores is a T-list of (B, V); stack to (B, T, V) in fp32 for stability.
    scores = torch.stack(out.scores, dim=1).float()
    logprobs = torch.log_softmax(scores, dim=-1)
    probs = logprobs.exp()
    ent = -(probs * logprobs).sum(-1)                              # (B, T)
    top2 = probs.topk(2, dim=-1).values
    maxp, margin = top2[..., 0], top2[..., 0] - top2[..., 1]
    chosen = logprobs.gather(-1, seqs.unsqueeze(-1)).squeeze(-1)   # (B, T)

    m = valid.float()

    def agg(x):
        """mean / min / max / std over valid positions only."""
        s = (x * m).sum(1) / lengths
        big = x.masked_fill(~valid, float("inf"))
        small = x.masked_fill(~valid, float("-inf"))
        var = (((x - s.unsqueeze(1)) ** 2) * m).sum(1) / lengths
        return s, big.min(1).values, small.max(1).values, var.clamp(min=0).sqrt()

    stats = []
    ent_m, ent_lo, ent_hi, ent_sd = agg(ent)
    mp_m, mp_lo, mp_hi, mp_sd = agg(maxp)
    mg_m, mg_lo, mg_hi, mg_sd = agg(margin)
    lp_m, lp_lo, lp_hi, lp_sd = agg(chosen)
    ppl = (-lp_m).exp()

    for b in range(B):
        stats.append({
            "n_tokens": int(lengths[b]),
            "ent_mean": float(ent_m[b]), "ent_min": float(ent_lo[b]),
            "ent_max": float(ent_hi[b]), "ent_std": float(ent_sd[b]),
            "ent_first": float(ent[b, 0]),
            "maxp_mean": float(mp_m[b]), "maxp_min": float(mp_lo[b]),
            "maxp_max": float(mp_hi[b]), "maxp_std": float(mp_sd[b]),
            "maxp_first": float(maxp[b, 0]),
            "margin_mean": float(mg_m[b]), "margin_min": float(mg_lo[b]),
            "margin_std": float(mg_sd[b]), "margin_first": float(margin[b, 0]),
            "logp_mean": float(lp_m[b]), "logp_min": float(lp_lo[b]),
            "logp_std": float(lp_sd[b]),
            "perplexity": float(ppl[b]),
        })

    # --- hidden states, three pooling variants ----------------------------
    # out.hidden_states is a T-list; element t is a tuple over layers.
    # t=0 is the prompt forward pass, shape (B, prompt_len, H); its LAST
    # position is the state that decides the first answer token. t>0 are
    # single new tokens, shape (B, 1, H).
    #
    #   prompt -- state before the model commits to anything (enables a
    #             pre-emptive warning, before a token is emitted)
    #   first  -- representation of the first generated token
    #   mean   -- mean over generated tokens, masked to the real answer
    n_layers = len(out.hidden_states[0])
    H = model.config.hidden_size
    acc = torch.zeros(B, n_layers, H, device=model.device, dtype=torch.float32)
    h_prompt = torch.zeros(B, n_layers, H, device=model.device, dtype=torch.float32)
    h_first = torch.zeros(B, n_layers, H, device=model.device, dtype=torch.float32)

    for t in range(T):
        step = out.hidden_states[t]
        w = m[:, t].view(B, 1)                         # broadcasts against (B, H)
        for li, h in enumerate(step):
            vec = h[:, -1, :].float()                  # (B, H)
            acc[:, li, :] += vec * w
            if t == 0:
                h_prompt[:, li, :] = vec
            elif t == 1:
                h_first[:, li, :] = vec
    if T < 2:                                          # degenerate one-token answer
        h_first = h_prompt.clone()
    pooled = acc / lengths.view(B, 1, 1)

    hid = {
        "mean": pooled.half().cpu().numpy(),
        "prompt": h_prompt.half().cpu().numpy(),
        "first": h_first.half().cpu().numpy(),
    }
    texts = tok.batch_decode(seqs, skip_special_tokens=True)
    return texts, stats, hid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--prefix", default="capture",
                    help="output basename; v2 runs use 'capture_v2'")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16"],
                    help="Gemma 3 overflows in fp16 (emits only <pad>); it was "
                         "trained in bf16. Use bfloat16 for cross-model runs so "
                         "every model is compared under the same precision.")
    ap.add_argument("--model", default=MODEL_ID,
                    help="HF model id; the study's second-model check uses "
                         "Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (DATA / args.prompts).open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"                 # keeps last position a real token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="cuda")
    model.eval()
    for p in model.parameters():               # frozen: we only read it
        p.requires_grad_(False)

    metas = []
    hiddens = {"mean": [], "prompt": [], "first": []}
    for i in tqdm(range(0, len(rows), args.batch_size), desc="capture"):
        chunk = rows[i: i + args.batch_size]
        texts, stats, hid = run_batch(model, tok, [r["question"] for r in chunk],
                                      args.max_new_tokens)
        for row, text, st in zip(chunk, texts, stats):
            metas.append({**{k: row.get(k, "") for k in
                             ("id", "family", "subtype", "template", "question")},
                          "answer": text.strip(),
                          "label": label_row(row, text),
                          **st})
        for k, v in hid.items():
            hiddens[k].append(v)

    meta = pd.DataFrame(metas)
    meta.to_parquet(DATA / f"{args.prefix}_meta.parquet", index=False)
    for k, v in hiddens.items():
        np.save(DATA / f"{args.prefix}_hidden_{k}.npy", np.concatenate(v, axis=0))
    hidden = np.concatenate(hiddens["mean"], axis=0)

    print(f"\ncaptured {len(meta)} generations; hidden {hidden.shape} x3 poolings")
    print(pd.crosstab(meta["family"], meta["label"]))
    print(f"\npeak vram: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
