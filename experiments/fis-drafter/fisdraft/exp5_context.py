"""Experiment 5 -- is `h_t` predictable from the context token embeddings?

Experiment 4 closed the persistence route: `h_{t-1}` carries R^2 = 0 about
`h_t` once the constant mean is removed. This asks the structurally different
question -- not "where was the state one step ago" but "what is in the context
now" -- using only lookups into the same tied embedding matrix `E` that the
LM head already is. No transformer forward pass, no new parameters beyond a
linear map.

Everything is scored in **alpha**, not R^2. R^2 on a 576-dimensional target is
not interpretable against the thing we care about, and experiment 4 gave us the
conversion: relative L2 error 0.10 -> alpha 0.89, 0.20 -> 0.79, 0.40 -> 0.60.
The two reference points every arm has to beat are the training mean
(alpha 0.081) and persistence (alpha 0.061).

Arms, cheapest first. All are O(context) table lookups plus one small matmul:

  mean_h          the unconditional mean -- the floor
  prev_h          reuse h_{t-1} -- the "free" signal that already failed
  cond_mean_1     E[h | last token], a 49152 x 576 lookup table
  cond_mean_2     E[h | last two tokens], sparser, backs off to cond_mean_1
  ridge_e1        linear map from the last token's embedding
  ridge_bag       linear map from [e_t, mean of last 4, last 16, decayed mean]
  ridge_bag_prev  the same, plus h_{t-1}

Ridge is used rather than anything larger because this is a *screen*: if a
linear map over context embeddings lands near the precision budget, the route
is open and deserves a better model. If it lands near the floor, the route is
closed and no model class rescues it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DECAY = 0.9


def load_head(model_id: str, device: str) -> torch.Tensor:
    """The output head `W`, which defines `logits = h @ W.T`."""
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    E = m.get_output_embeddings().weight.detach().to(device)
    del m
    return E


def load_input_embeddings(model_id: str, device: str) -> tuple[torch.Tensor, bool]:
    """The *input* embedding table, used to featurise context tokens.

    For SmolLM2 this is the same matrix as the output head (tied), so the
    distinction is invisible. For an untied model such as pythia it is a
    different matrix, and featurising the context with the output head would be
    the wrong lookup -- the question "what tokens are in the context" is asked
    of the input table. Returns the table and whether it is tied.
    """
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    tied = (
        m.get_input_embeddings().weight.data_ptr()
        == m.get_output_embeddings().weight.data_ptr()
    )
    Ein = m.get_input_embeddings().weight.detach().to(device)
    del m
    return Ein, tied


def alpha_of(H_hat: torch.Tensor, P: torch.Tensor, E: torch.Tensor, chunk=2048):
    """Mean acceptance and argmax agreement, chunked over the vocabulary matmul."""
    a, agree, n = 0.0, 0.0, 0
    for i in range(0, len(H_hat), chunk):
        q = torch.softmax(H_hat[i : i + chunk] @ E.T, dim=-1)
        p = P[i : i + chunk]
        a += float(torch.minimum(p, q).sum(-1).sum())
        agree += float((q.argmax(-1) == p.argmax(-1)).sum())
        n += len(q)
    return a / n, agree / n


def build_context_features(rundir: Path, df: pd.DataFrame, Ein: torch.Tensor, device):
    """Bag-of-embedding features over the tokens the step actually saw."""
    from .exp3_acceptance import rebuild_contexts

    enc, gen = rebuild_contexts(rundir, df)

    rows = df.row.to_numpy()
    pid = df.prompt_id.to_numpy()
    step = df.step.to_numpy()

    n = len(df)
    feat = torch.zeros((n, 4 * Ein.shape[1]), device=device)
    last_tok = np.zeros(n, dtype=np.int64)
    prev_tok = np.zeros(n, dtype=np.int64)

    cache: dict[int, list[int]] = {}
    for i in range(n):
        p, s = int(pid[i]), int(step[i])
        if p not in cache:
            cache[p] = list(enc[p]) + gen.get(p, [])
        full = cache[p]
        ctx = full[: len(enc[p]) + s]
        if not ctx:
            ctx = [0]
        last_tok[i] = ctx[-1]
        prev_tok[i] = ctx[-2] if len(ctx) > 1 else ctx[-1]

        idx = torch.tensor(ctx[-64:], device=device, dtype=torch.long)
        emb = Ein[idx]
        w = torch.pow(
            torch.tensor(DECAY, device=device),
            torch.arange(len(idx) - 1, -1, -1, device=device, dtype=torch.float32),
        )
        feat[i] = torch.cat(
            [
                emb[-1],
                emb[-4:].mean(0),
                emb[-16:].mean(0),
                (emb * w.unsqueeze(-1)).sum(0) / w.sum(),
            ]
        )
    return feat, last_tok, prev_tok


def ridge_fit(X, Y, lam=1.0):
    """Closed-form ridge with an intercept, on GPU."""
    ones = torch.ones((len(X), 1), device=X.device)
    Xa = torch.cat([X, ones], dim=1)
    A = Xa.T @ Xa
    A += lam * torch.eye(A.shape[0], device=A.device)
    B = Xa.T @ Y
    return torch.linalg.solve(A, B)


def ridge_apply(W, X):
    ones = torch.ones((len(X), 1), device=X.device)
    return torch.cat([X, ones], dim=1) @ W


def run(rundir: Path, device="cuda", n_eval=6000, seed=0, lam=10.0) -> dict:
    meta = json.loads((rundir / "meta.json").read_text())
    E = load_head(meta["config"]["model_id"], device)
    Ein, tied = load_input_embeddings(meta["config"]["model_id"], device)
    H = torch.from_numpy(np.load(rundir / "hidden_last.npy")).to(device)
    df = pd.read_parquet(rundir / "steps.parquet").reset_index(drop=True)

    feat, last_tok, prev_tok = build_context_features(rundir, df, Ein, device)

    rng = np.random.default_rng(seed)
    pids = df.prompt_id.unique()
    rng.shuffle(pids)
    tr_p = set(pids[: int(0.7 * len(pids))])
    is_tr = df.prompt_id.isin(tr_p).to_numpy()

    # previous-step row for each step (None at step 0)
    row_by = {
        (int(p), int(s)): int(r)
        for r, p, s in zip(df.row, df.prompt_id, df.step)
    }
    row_index = {int(r): i for i, r in enumerate(df.row.to_numpy())}
    prev_i = np.full(len(df), -1, dtype=np.int64)
    for i, (p, s) in enumerate(zip(df.prompt_id.to_numpy(), df.step.to_numpy())):
        r = row_by.get((int(p), int(s) - 1))
        if r is not None:
            prev_i[i] = row_index[r]
    has_prev = prev_i >= 0

    tr_i = np.where(is_tr & has_prev)[0]
    te_i = np.where((~is_tr) & has_prev)[0]
    te_i = rng.choice(te_i, size=min(n_eval, len(te_i)), replace=False)

    Hte = H[te_i]
    P = torch.softmax(Hte @ E.T, dim=-1) if len(te_i) <= 4096 else None
    if P is None:
        P = torch.cat(
            [
                torch.softmax(H[te_i[i : i + 2048]] @ E.T, dim=-1)
                for i in range(0, len(te_i), 2048)
            ]
        )

    Htr = H[tr_i]
    mu = Htr.mean(0, keepdim=True)
    res: dict = {"n_train": len(tr_i), "n_test": len(te_i),
                 "tied_embeddings": bool(tied), "arms": {}}

    def record(name, H_hat):
        a, agree = alpha_of(H_hat, P, E)
        rel = float(
            ((H_hat - Hte).norm(dim=-1) / Hte.norm(dim=-1)).median()
        )
        relc = float(
            (((H_hat - mu) - (Hte - mu)).norm(dim=-1) / (Hte - mu).norm(dim=-1)).median()
        )
        res["arms"][name] = {
            "alpha": a,
            "argmax_agree": agree,
            "median_rel_l2": rel,
            "median_rel_l2_centred": relc,
        }
        print(
            f"  {name:16s} alpha {a:.4f}  argmax {agree:.4f}  "
            f"relL2 {rel:.3f}  (centred {relc:.3f})",
            flush=True,
        )

    record("mean_h", mu.expand(len(te_i), -1))
    record("prev_h", H[prev_i[te_i]])

    # conditional-mean lookup tables -- pure lookups, no matmul
    V, D = E.shape[0], H.shape[1]
    for order, toks in (("cond_mean_1", last_tok), ("cond_mean_2", None)):
        if order == "cond_mean_1":
            key_tr = torch.from_numpy(toks[tr_i]).to(device)
            key_te = torch.from_numpy(toks[te_i]).to(device)
            n_keys = V
        else:
            # bigram key, hashed into a bounded table with backoff to unigram
            kt = (torch.from_numpy(prev_tok).to(device).long() * 1000003
                  + torch.from_numpy(last_tok).to(device).long()) % (1 << 22)
            key_tr, key_te, n_keys = kt[tr_i], kt[te_i], 1 << 22
        tab = torch.zeros((n_keys, D), device=device)
        cnt = torch.zeros(n_keys, device=device)
        tab.index_add_(0, key_tr, Htr)
        cnt.index_add_(0, key_tr, torch.ones(len(tr_i), device=device))
        seen = cnt > 0
        tab[seen] /= cnt[seen].unsqueeze(-1)
        pred = tab[key_te]
        miss = ~seen[key_te]
        if order == "cond_mean_2":
            uni = torch.zeros((V, D), device=device)
            ucnt = torch.zeros(V, device=device)
            ktr = torch.from_numpy(last_tok[tr_i]).to(device)
            uni.index_add_(0, ktr, Htr)
            ucnt.index_add_(0, ktr, torch.ones(len(tr_i), device=device))
            us = ucnt > 0
            uni[us] /= ucnt[us].unsqueeze(-1)
            pred[miss] = uni[torch.from_numpy(last_tok[te_i]).to(device)][miss]
            miss = miss & ~us[torch.from_numpy(last_tok[te_i]).to(device)]
        pred[miss] = mu
        res.setdefault("coverage", {})[order] = float((~miss).float().mean())
        record(order, pred)
        del tab, cnt

    # linear maps over context embeddings
    Ftr, Fte = feat[tr_i], feat[te_i]
    d = Ein.shape[1]
    W = ridge_fit(Ftr[:, :d], Htr, lam)
    record("ridge_e1", ridge_apply(W, Fte[:, :d]))

    W = ridge_fit(Ftr, Htr, lam)
    record("ridge_bag", ridge_apply(W, Fte))

    Ftr2 = torch.cat([Ftr, H[prev_i[tr_i]]], dim=1)
    Fte2 = torch.cat([Fte, H[prev_i[te_i]]], dim=1)
    W = ridge_fit(Ftr2, Htr, lam)
    record("ridge_bag_prev", ridge_apply(W, Fte2))

    res["reference_precision_budget"] = {
        "rel_l2_0.10": 0.894, "rel_l2_0.20": 0.788, "rel_l2_0.40": 0.598,
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--n-eval", type=int, default=6000)
    a = ap.parse_args()
    rundir = Path(a.run)
    r = run(rundir, n_eval=a.n_eval)
    (rundir / "exp5_context.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
