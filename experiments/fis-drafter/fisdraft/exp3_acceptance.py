"""Experiment 3 -- does a cheap ranker plus a predicted shape actually accept?

Tests variant V2 end to end on the acceptance metric, without running a single
generation step. The reason that is possible, and exact rather than an
approximation: speculative decoding's expected acceptance rate is

    alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q)

and `runs/*/full_logprob.npy` already holds the exact `p` for 20,000 captured
steps. Given a candidate set and a shape rule, `q` is determined, so `alpha`
follows in closed form. Every number here is an exact expectation over the
captured distributions, not a sampled estimate -- there is no Monte-Carlo error
to argue about.

`q` is materialised as a genuine normalised categorical over the whole
vocabulary (DESIGN.md 2.1):

    q(x) = s_j                                   x is the j-th candidate
    q(x) = (1 - sum_j s_j) * u(x) / sum_tail u   otherwise

so the acceptance test stays the exact one and speculative decoding remains
lossless. Nothing here needs a relaxed acceptance criterion.

The factorial design is (ranker x shape). The oracle arms matter as much as the
real ones: an oracle *ranker* with a real shape isolates the shape model's
contribution, and a real ranker with an oracle *shape* gives the ceiling that
any shape model could reach on that candidate set. Without both, a bad result
cannot be attributed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .prompts import build_battery


# --------------------------------------------------------------------------
# Context reconstruction
# --------------------------------------------------------------------------


def rebuild_contexts(rundir: Path, df: pd.DataFrame):
    """Prompt encodings and per-prompt generated token sequences.

    Prompt token ids are not stored in the capture -- they are recovered by
    re-encoding, which is deterministic given the seed and the battery
    definition. The generated half comes from `sampled_token` ordered by step.

    Returns `(enc, gen)` rather than pre-joined contexts so a step can be given
    exactly the prefix it saw (`enc[pid] + gen[pid][:step]`); handing every step
    the whole generation would let the prompt-lookup ranker match against tokens
    that had not been produced yet.
    """
    from transformers import AutoTokenizer
    from .capture import _encode

    meta = json.loads((rundir / "meta.json").read_text())
    cfg = meta["config"]
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    prompts = build_battery(cfg["n_dolly"], seed=cfg["seed"])
    enc = _encode(tok, prompts, cfg["chat_template"])

    gen = {
        pid: [int(t) for t in g.sort_values("step").sampled_token.to_numpy()]
        for pid, g in df.groupby("prompt_id")
    }
    return enc, gen


# --------------------------------------------------------------------------
# Rankers -- each returns candidate token ids for one step, cheapest first
# --------------------------------------------------------------------------


class UnigramRanker:
    """Context-free. The floor: any ranker that loses to this is worthless."""

    name = "unigram"

    def fit(self, seqs):
        c = Counter()
        for s in seqs:
            c.update(s)
        self.order = [t for t, _ in c.most_common()]

    def candidates(self, ctx, k):
        return self.order[:k]


class BigramRanker:
    """P(next | current token). One table lookup, no model."""

    name = "bigram"

    def fit(self, seqs):
        d = defaultdict(Counter)
        c = Counter()
        for s in seqs:
            c.update(s)
            for a, b in zip(s, s[1:]):
                d[a][b] += 1
        self.tab = {a: [t for t, _ in v.most_common(64)] for a, v in d.items()}
        self.backoff = [t for t, _ in c.most_common(64)]

    def candidates(self, ctx, k):
        out = list(self.tab.get(ctx[-1], [])[:k])
        for t in self.backoff:
            if len(out) >= k:
                break
            if t not in out:
                out.append(t)
        return out[:k]


class PromptLookupRanker:
    """Prompt-lookup decoding: find the most recent repeat of the current
    n-gram in the context and propose what followed it.

    This is the drafter that reportedly gets 2-4x on input-grounded tasks. It
    is also the one with a known failure mode -- it needs lexical overlap
    between context and continuation -- so it is exactly the arm that should
    degrade on open-ended dolly prompts, and that prediction is testable here.
    """

    name = "prompt_lookup"

    def __init__(self, max_n=3, min_n=1):
        self.max_n, self.min_n = max_n, min_n

    def fit(self, seqs):
        self.bg = BigramRanker()
        self.bg.fit(seqs)

    def candidates(self, ctx, k):
        out: list[int] = []
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(ctx) <= n:
                continue
            pat = ctx[-n:]
            for i in range(len(ctx) - n - 1, -1, -1):
                if ctx[i : i + n] == pat:
                    t = ctx[i + n]
                    if t not in out:
                        out.append(t)
                    if len(out) >= k:
                        return out[:k]
        for t in self.bg.candidates(ctx, k):
            if len(out) >= k:
                break
            if t not in out:
                out.append(t)
        return out[:k]


# --------------------------------------------------------------------------
# Shapes -- given K candidates, how much mass does each get?
# --------------------------------------------------------------------------


def shape_zipf(k: int, top1: float, expo: float) -> np.ndarray:
    """Zipf-decayed head with a prescribed top-1 mass.

    Experiment 0 measured the sorted tail exponent at -1.651 +/- 0.296 and
    nearly invariant across regimes, so a Zipf head with a *fitted* top-1 and a
    *fixed* exponent is the natural two-parameter family -- and it is the one
    an FIS can actually supply, since top-1 is one of the shape targets it
    predicts.
    """
    r = np.arange(1, k + 1, dtype=np.float64) ** expo
    r = r / r.sum()
    return np.clip(top1, 1e-4, 0.999) * r / r[0] if k > 1 else np.array([top1])


def build_q(
    p_row: torch.Tensor, cand: list[int], head: np.ndarray, tail_u: torch.Tensor
) -> torch.Tensor:
    """Materialise the full normalised draft distribution over the vocabulary."""
    q = tail_u.clone()
    idx = torch.tensor(cand, device=q.device, dtype=torch.long)
    q[idx] = 0.0
    s = float(head.sum())
    tail_mass = max(0.0, 1.0 - s)
    tot = float(q.sum())
    q = q * (tail_mass / tot) if tot > 0 else q
    q[idx] = torch.tensor(head, device=q.device, dtype=q.dtype)
    return q


# --------------------------------------------------------------------------


def run(rundir: Path, k: int = 8, device: str = "cuda", limit: int = 20000) -> dict:
    df = pd.read_parquet(rundir / "steps.parquet")
    rowids = np.load(rundir / "full_logprob_rowid.npy")
    lp = np.load(rundir / "full_logprob.npy", mmap_mode="r")
    n = min(limit, len(rowids))

    enc, gen = rebuild_contexts(rundir, df)
    by_row = df.set_index("row")

    # Rankers are fitted on a disjoint set of prompts from the ones evaluated,
    # so the n-gram tables cannot have memorised the exact continuations they
    # are being scored on.
    pids = sorted(df.prompt_id.unique())
    rng = np.random.default_rng(0)
    rng.shuffle(pids)
    fit_p = set(pids[: int(0.6 * len(pids))])
    eval_p = set(pids[int(0.6 * len(pids)) :])
    fit_seqs = [list(enc[p]) + gen.get(p, []) for p in fit_p]

    rankers = [UnigramRanker(), BigramRanker(), PromptLookupRanker()]
    for r in rankers:
        r.fit(fit_seqs)

    vocab = lp.shape[1]
    tail_u = torch.full((vocab,), 1.0 / vocab, device=device, dtype=torch.float32)

    shapes = {
        "oracle": None,
        "zipf_top1_0.30": (0.30, -1.651),
        "zipf_top1_0.56": (0.56, -1.651),  # the measured marginal mean top-1
        "zipf_top1_0.80": (0.80, -1.651),
    }

    acc: dict[str, list[float]] = defaultdict(list)
    hit: dict[str, list[float]] = defaultdict(list)
    # Per-category, because the known failure mode of n-gram/retrieval drafters
    # is low lexical overlap between context and continuation. Dolly's
    # context-bearing categories (closed_qa, information_extraction,
    # summarization) and the synthetic `structured` probes are where a lookup
    # ranker should do well; brainstorming and creative_writing are where it
    # should collapse. Pooling them hides exactly the effect that decides
    # whether V2 has a niche.
    acc_cat: dict[tuple, list[float]] = defaultdict(list)
    hit_cat: dict[tuple, list[float]] = defaultdict(list)
    n_eval = 0

    pid_of = by_row.prompt_id.to_dict()
    step_of = by_row.step.to_dict()
    cat_of = by_row.category.to_dict()

    for i in range(n):
        row = int(rowids[i])
        if row not in pid_of:
            continue
        pid, step = int(pid_of[row]), int(step_of[row])
        # Score only prompts the rankers were not fitted on.
        if pid not in eval_p:
            continue
        n_eval += 1
        ctx = list(enc[pid]) + gen.get(pid, [])[:step]
        p = torch.softmax(
            torch.from_numpy(np.asarray(lp[i], dtype=np.float32)).to(device), dim=-1
        )
        true_top = int(p.argmax())

        cat = str(cat_of[row])
        for r in rankers:
            cand = r.candidates(ctx, k)
            if not cand:
                continue
            hit[r.name].append(float(true_top in cand))
            hit_cat[(r.name, cat)].append(float(true_top in cand))
            for sname, sp in shapes.items():
                if sp is None:
                    head = p[torch.tensor(cand, device=device)].cpu().numpy().astype(
                        np.float64
                    )
                else:
                    head = shape_zipf(len(cand), sp[0], sp[1])
                q = build_q(p, cand, head, tail_u)
                a = float(torch.minimum(p, q).sum())
                acc[f"{r.name}|{sname}"].append(a)
                acc_cat[(r.name, sname, cat)].append(a)

        # Oracle ranker: the true top-k identities. Upper bound for this k.
        ocand = torch.topk(p, k).indices.cpu().tolist()
        hit["oracle_ranker"].append(1.0)
        for sname, sp in shapes.items():
            if sp is None:
                head = p[torch.tensor(ocand, device=device)].cpu().numpy().astype(
                    np.float64
                )
            else:
                head = shape_zipf(len(ocand), sp[0], sp[1])
            q = build_q(p, ocand, head, tail_u)
            acc[f"oracle_ranker|{sname}"].append(float(torch.minimum(p, q).sum()))

    def tokens_per_pass(a, gamma):
        return (1 - a ** (gamma + 1)) / (1 - a) if a < 1 else gamma + 1

    out = {
        "k": k,
        "n_steps_scanned": n,
        "n_steps_scored": n_eval,
        "hit_rate_true_argmax_in_candidates": {
            kk: float(np.mean(v)) for kk, v in hit.items()
        },
        "acceptance": {
            kk: {
                "mean_alpha": float(np.mean(v)),
                "median_alpha": float(np.median(v)),
                "tokens_per_pass_gamma4": tokens_per_pass(float(np.mean(v)), 4),
            }
            for kk, v in sorted(acc.items())
        },
        "hit_rate_by_category": {
            f"{r}|{c}": {"hit": float(np.mean(v)), "n": len(v)}
            for (r, c), v in sorted(hit_cat.items())
        },
        "acceptance_by_category": {
            f"{r}|{s}|{c}": {"mean_alpha": float(np.mean(v)), "n": len(v)}
            for (r, s, c), v in sorted(acc_cat.items())
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=4000)
    a = ap.parse_args()
    rundir = Path(a.run)
    res = run(rundir, k=a.k, limit=a.limit)
    (rundir / f"exp3_acceptance_k{a.k}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
