"""Instrument a small LM: record every next-token distribution it produces,
plus the intermediate states that produced it.

One row per generation step. Three tiers of recorded state, kept separate
because they have very different standing as *drafter inputs*:

  tier A  "free"       -- statistics of the distribution at step t-1, t-2, ...
                          In speculative decoding the target model's
                          verification pass hands you the true distribution at
                          the last accepted position at zero marginal cost, so
                          a drafter may legitimately condition on it.
  tier B  "cheap"      -- token identity and surface features, position,
                          n-gram context. No model forward pass.
  tier C  "expensive"  -- hidden states, per-layer norms. A drafter that needs
                          these has already paid for the forward pass it was
                          meant to avoid, so tier C is NOT deployable. It is
                          recorded to establish a *ceiling*: how much of the
                          predictable variance is reachable at all.

Keeping C out of the deployable feature set and reporting it anyway is the
point. It converts "the FIS scored R^2=0.4" from an unanchored number into
"the FIS scored 0.4 of a reachable 0.55".
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import build_battery, Prompt

TOPK = 128  # how many (index, logprob) pairs to keep per step
TAIL_LO, TAIL_HI = 10, 1000  # rank window for the power-law slope fit


# --------------------------------------------------------------------------
# Distribution statistics
# --------------------------------------------------------------------------


def distribution_stats(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Summarise a batch of next-token distributions.

    `logits` is (B, V) float32. Everything is computed from the *full*
    distribution before any truncation, so these are exact, not top-k
    approximations. Returns a dict of (B,) tensors.

    The statistics are chosen to be a candidate low-dimensional parameterisation
    of distribution *shape* -- the object the FIS is being asked to predict.
    """
    logprobs = F.log_softmax(logits.float(), dim=-1)
    probs = logprobs.exp()

    # Shannon entropy, and the variance of the surprisal under p ("varentropy").
    # Varentropy separates a flat distribution from a spike-plus-flat-tail
    # distribution that happens to share its entropy, so the pair carries
    # strictly more shape information than entropy alone.
    surprisal = -logprobs
    entropy = (probs * surprisal).sum(-1)
    varentropy = (probs * (surprisal - entropy.unsqueeze(-1)) ** 2).sum(-1)

    # Renyi-2 (collision) entropy: dominated by the head, so it moves when the
    # top few tokens rearrange even if Shannon entropy does not.
    renyi2 = -torch.log((probs**2).sum(-1).clamp_min(1e-30))

    srt, _ = torch.sort(logprobs, dim=-1, descending=True)
    sp = srt.exp()
    csum = sp.cumsum(-1)

    stats = {
        "entropy": entropy,
        "varentropy": varentropy,
        "renyi2": renyi2,
        "logsumexp": torch.logsumexp(logits.float(), dim=-1),
        "max_logit": logits.max(-1).values,
        "mean_logit": logits.float().mean(-1),
        "std_logit": logits.float().std(-1),
        "top1_logprob": srt[:, 0],
        "top2_logprob": srt[:, 1],
        # Margin in log space is the quantity the acceptance test is most
        # sensitive to: it sets how much probability mass a drafter loses by
        # getting the argmax wrong.
        "log_margin_12": srt[:, 0] - srt[:, 1],
        "top1_prob": sp[:, 0],
        "mass_top5": csum[:, 4],
        "mass_top10": csum[:, 9],
        "mass_top50": csum[:, 49],
        "mass_top128": csum[:, 127],
    }

    # Nucleus sizes: how many tokens carry 90% / 95% of the mass. This is the
    # "effective support size" and is what actually bounds a drafter's hit rate.
    for q in (0.90, 0.95, 0.99):
        stats[f"nucleus_{int(q * 100)}"] = (csum < q).sum(-1).float() + 1

    # Power-law slope of the sorted tail. Zipfian structure in next-token
    # distributions is folklore; fitting the exponent lets experiment 0 check
    # it rather than assume it.
    ranks = torch.arange(TAIL_LO, TAIL_HI, device=logits.device, dtype=torch.float32)
    lr = torch.log(ranks)
    lp = srt[:, TAIL_LO:TAIL_HI]
    lr_c = lr - lr.mean()
    lp_c = lp - lp.mean(-1, keepdim=True)
    stats["tail_slope"] = (lp_c * lr_c).sum(-1) / (lr_c**2).sum()
    # Residual of that fit: how *well* a power law describes the tail at all.
    pred = lp_c - stats["tail_slope"].unsqueeze(-1) * lr_c
    stats["tail_fit_resid"] = pred.pow(2).mean(-1).sqrt()

    return stats


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    n_dolly: int = 400
    max_new_tokens: int = 48
    batch_size: int = 32
    temperature: float = 1.0
    seed: int = 0
    dtype: str = "float32"
    device: str = "cuda"
    max_full_logprob_rows: int = 20000
    chat_template: bool = True
    out: str = "runs/capture"


def _encode(tok, prompts: list[Prompt], allow_chat: bool) -> list[list[int]]:
    """Encode each prompt, honouring its own `use_chat` flag.

    The template is decided per prompt, not per run: dolly rows are genuine
    user turns and want it, the synthetic completion probes are ruined by it.
    `allow_chat=False` forces raw completion everywhere (base-model runs).
    """
    ids = []
    for p in prompts:
        chat = allow_chat and p.use_chat and bool(tok.chat_template)
        if chat:
            text = tok.apply_chat_template(
                [{"role": "user", "content": p.text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = p.text
        ids.append(tok(text, add_special_tokens=not chat)["input_ids"])
    return ids


@torch.no_grad()
def run_capture(cfg: CaptureConfig) -> Path:
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype)
    model.to(cfg.device).eval()

    prompts = build_battery(cfg.n_dolly, seed=cfg.seed)
    encoded = _encode(tok, prompts, cfg.chat_template)
    order = np.argsort([len(e) for e in encoded])  # length-bucket to cut padding

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    vocab = model.config.vocab_size

    scalar_rows: list[dict] = []
    topk_idx: list[np.ndarray] = []
    topk_lp: list[np.ndarray] = []
    hidden_last: list[np.ndarray] = []
    layer_norms: list[np.ndarray] = []
    full_lp: list[np.ndarray] = []
    full_lp_rowid: list[int] = []

    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    row_id = 0

    for bstart in range(0, len(order), cfg.batch_size):
        bidx = order[bstart : bstart + cfg.batch_size]
        batch = [encoded[i] for i in bidx]
        maxlen = max(len(b) for b in batch)
        pad = tok.pad_token_id
        ids = torch.full((len(batch), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for r, b in enumerate(batch):
            ids[r, maxlen - len(b) :] = torch.tensor(b)
            attn[r, maxlen - len(b) :] = 1
        ids, attn = ids.to(cfg.device), attn.to(cfg.device)

        past = None
        cur = ids
        alive = torch.ones(len(batch), dtype=torch.bool, device=cfg.device)
        # Per-sequence history of the tier-A statistics, so a step can see
        # what the distribution looked like one and two steps earlier.
        hist: list[list[dict]] = [[] for _ in batch]

        for step in range(cfg.max_new_tokens):
            out = model(
                input_ids=cur,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            hs = out.hidden_states  # tuple: embeddings + one per layer
            h_last = hs[-1][:, -1, :].float()
            lnorms = torch.stack(
                [h[:, -1, :].float().norm(dim=-1) for h in hs], dim=1
            )  # (B, n_layers+1)

            stats = distribution_stats(logits)
            lp = F.log_softmax(logits.float(), dim=-1)
            tk = torch.topk(lp, TOPK, dim=-1)

            # sample the next token
            if cfg.temperature > 0:
                nxt = torch.multinomial(
                    F.softmax(logits.float() / cfg.temperature, dim=-1), 1
                ).squeeze(-1)
            else:
                nxt = logits.argmax(-1)

            s_cpu = {k: v.detach().cpu().numpy() for k, v in stats.items()}
            tk_i = tk.indices.detach().cpu().numpy().astype(np.int32)
            tk_v = tk.values.detach().cpu().numpy().astype(np.float32)
            h_cpu = h_last.detach().cpu().numpy().astype(np.float32)
            ln_cpu = lnorms.detach().cpu().numpy().astype(np.float32)
            nxt_cpu = nxt.detach().cpu().numpy()
            lp_cpu = None

            for r in range(len(batch)):
                if not bool(alive[r]):
                    continue
                gi = int(bidx[r])
                p = prompts[gi]
                rec = {
                    "row": row_id,
                    "prompt_id": gi,
                    "source": p.source,
                    "category": p.category,
                    "step": step,
                    "prompt_len": len(batch[r]),
                    "sampled_token": int(nxt_cpu[r]),
                    "input_token": int(cur[r, -1]),
                }
                for k, v in s_cpu.items():
                    rec[k] = float(v[r])

                # tier A lags
                h = hist[r]
                for lag in (1, 2, 3):
                    src = h[-lag] if len(h) >= lag else None
                    for k in ("entropy", "varentropy", "top1_prob", "log_margin_12"):
                        rec[f"prev{lag}_{k}"] = (
                            float(src[k]) if src is not None else np.nan
                        )
                if h:
                    ent_hist = np.array([x["entropy"] for x in h], dtype=np.float64)
                    rec["ent_ema_short"] = float(ent_hist[-3:].mean())
                    rec["ent_ema_long"] = float(ent_hist.mean())
                    rec["ent_cummax"] = float(ent_hist.max())
                else:
                    rec["ent_ema_short"] = np.nan
                    rec["ent_ema_long"] = np.nan
                    rec["ent_cummax"] = np.nan

                scalar_rows.append(rec)
                topk_idx.append(tk_i[r])
                topk_lp.append(tk_v[r])
                hidden_last.append(h_cpu[r])
                layer_norms.append(ln_cpu[r])

                # Reservoir-free subsample: keep full log-probs for a bounded
                # number of rows so the rank study (experiment 1) has exact
                # distributions to work with.
                if len(full_lp) < cfg.max_full_logprob_rows and rng.random() < 0.35:
                    if lp_cpu is None:
                        lp_cpu = lp.detach().cpu().numpy().astype(np.float16)
                    full_lp.append(lp_cpu[r])
                    full_lp_rowid.append(row_id)

                hist[r].append({k: float(v[r]) for k, v in s_cpu.items()})
                row_id += 1

            alive = alive & (nxt != tok.eos_token_id)
            if not bool(alive.any()):
                break
            cur = nxt.unsqueeze(-1)
            attn = torch.cat(
                [
                    attn,
                    torch.ones((len(batch), 1), dtype=torch.long, device=cfg.device),
                ],
                dim=1,
            )

        print(
            f"[capture] batch {bstart // cfg.batch_size + 1}"
            f"/{(len(order) + cfg.batch_size - 1) // cfg.batch_size} "
            f"rows={row_id} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    import pandas as pd

    df = pd.DataFrame(scalar_rows)
    df.to_parquet(outdir / "steps.parquet", index=False)
    np.save(outdir / "topk_idx.npy", np.stack(topk_idx))
    np.save(outdir / "topk_logprob.npy", np.stack(topk_lp))
    np.save(outdir / "hidden_last.npy", np.stack(hidden_last))
    np.save(outdir / "layer_norms.npy", np.stack(layer_norms))
    np.save(outdir / "full_logprob.npy", np.stack(full_lp))
    np.save(outdir / "full_logprob_rowid.npy", np.array(full_lp_rowid, dtype=np.int64))

    meta = {
        "config": asdict(cfg),
        "n_rows": int(row_id),
        "n_prompts": len(prompts),
        "vocab_size": int(vocab),
        "hidden_size": int(hidden_size),
        "n_layers": int(n_layers),
        "topk": TOPK,
        "n_full_logprob": len(full_lp),
        "elapsed_s": time.time() - t0,
        "torch": torch.__version__,
        "transformers_model": cfg.model_id,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return outdir


def main() -> None:
    ap = argparse.ArgumentParser()
    for f, default in asdict(CaptureConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(default), default=default)
    a = ap.parse_args()
    run_capture(CaptureConfig(**{k: v for k, v in vars(a).items()}))


if __name__ == "__main__":
    main()
