"""Experiment 4 -- work in embedding space, where the problem actually lives.

SmolLM2 ties its embeddings and its LM head carries no bias, so

    logits = h @ E.T      h in R^576,  E in R^{49152 x 576}

The 49,152-dimensional logit vector is an exact linear image of a
576-dimensional one. Every earlier ceiling in this study was measured in
log-probability space, which was the wrong coordinate system: experiment 1a
needed rank 512 to reach alpha = 0.86 and rank 256 for 0.57, which in hindsight
was rediscovering `hidden_size` the hard way, through a softmax, in 49,152
coordinates.

Three things this settles that log-prob space could not:

1. **Identity is free.** Which tokens get the mass is decided by which
   embedding rows have the largest inner product with `h`. A model that
   predicts `h` gets the candidate identities from a MIPS/top-k against `E` --
   it never has to name a token. The shape-vs-identity split that killed V3
   does not apply here, because both come out of the same 576 numbers.

2. **The rank question is posed correctly.** A rank-k projection of `h`
   preserves the geometry that determines the argmax; a rank-k projection of
   the log-probs does not.

3. **There is a precision budget.** `alpha` as a function of relative error in
   `h` says how accurately any predictor has to be to be worth building --
   before building one.

Also: `logits = h @ E.T` means the 180 MB `hidden_last.npy` regenerates every
distribution exactly, and the 2 GB `full_logprob.npy` is redundant. The
identity is verified against the stored log-probs before anything is built on
it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 576]
NOISE = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]


def load_head(model_id: str, device: str) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    head = m.get_output_embeddings()
    assert head.bias is None, "this analysis assumes an unbiased LM head"
    E = head.weight.detach().to(device)
    del m
    return E


def alpha_of(logits_hat: torch.Tensor, p_true: torch.Tensor) -> torch.Tensor:
    """Acceptance rate = sum_x min(p, q) = 1 - TV, row-wise."""
    q = torch.softmax(logits_hat, dim=-1)
    return torch.minimum(p_true, q).sum(-1)


def run(rundir: Path, device="cuda", n_eval=8000, seed=0) -> dict:
    meta = json.loads((rundir / "meta.json").read_text())
    E = load_head(meta["config"]["model_id"], device)
    H = np.load(rundir / "hidden_last.npy")
    df = pd.read_parquet(rundir / "steps.parquet")

    out: dict = {"hidden_size": int(E.shape[1]), "vocab": int(E.shape[0])}

    # ---- 0. verify logits = h @ E.T against the stored log-probs ----------
    rowids = np.load(rundir / "full_logprob_rowid.npy")
    lp = np.load(rundir / "full_logprob.npy", mmap_mode="r")
    chk = np.arange(min(512, len(rowids)))
    h_chk = torch.from_numpy(H[rowids[chk]]).to(device)
    got = torch.log_softmax(h_chk @ E.T, dim=-1)
    want = torch.from_numpy(np.asarray(lp[chk], dtype=np.float32)).to(device)
    err = (got - want).abs()
    out["identity_check"] = {
        "n": int(len(chk)),
        "max_abs_logprob_err": float(err.max()),
        "mean_abs_logprob_err": float(err.mean()),
        "note": "stored log-probs are float16, so ~1e-3 is the storage floor",
    }
    del lp, want, got, h_chk

    # ---- evaluation split, by prompt --------------------------------------
    rng = np.random.default_rng(seed)
    pids = df.prompt_id.unique()
    rng.shuffle(pids)
    tr_p = set(pids[: int(0.6 * len(pids))])
    tr_rows = df[df.prompt_id.isin(tr_p)].row.to_numpy()
    te_rows = df[~df.prompt_id.isin(tr_p)].row.to_numpy()
    te_rows = rng.choice(te_rows, size=min(n_eval, len(te_rows)), replace=False)

    Htr = torch.from_numpy(H[tr_rows]).to(device)
    Hte = torch.from_numpy(H[te_rows]).to(device)
    P = torch.softmax(Hte @ E.T, dim=-1)

    mu = Htr.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(Htr - mu, q=min(576, Htr.shape[0] - 1), niter=4)
    tot = float(((Htr - mu) ** 2).sum())

    # ---- 1. rank-k in embedding space -------------------------------------
    curve = []
    for k in RANKS:
        if k > V.shape[1]:
            continue
        Vk = V[:, :k]
        Hk = ((Hte - mu) @ Vk) @ Vk.T + mu
        a = alpha_of(Hk @ E.T, P)
        curve.append(
            {
                "rank": k,
                "mean_alpha": float(a.mean()),
                "median_alpha": float(a.median()),
                "explained_var_frac": float((S[:k] ** 2).sum()) / tot,
                "argmax_agree": float(
                    ((Hk @ E.T).argmax(-1) == P.argmax(-1)).float().mean()
                ),
            }
        )
        print(
            f"  rank {k:4d}  alpha {a.mean():.4f}  "
            f"argmax {curve[-1]['argmax_agree']:.4f}  "
            f"expvar {curve[-1]['explained_var_frac']:.4f}",
            flush=True,
        )
    out["rank_curve_embedding_space"] = curve

    # ---- 2. precision budget: alpha vs relative error in h ----------------
    # Isotropic Gaussian perturbation scaled to a target relative L2 error.
    # This is the number that says how good an h-predictor has to be.
    hn = Hte.norm(dim=-1, keepdim=True)
    noise_curve = []
    g = torch.Generator(device=device).manual_seed(seed)
    for eps in NOISE:
        z = torch.randn(Hte.shape, device=device, generator=g)
        z = z / z.norm(dim=-1, keepdim=True) * hn * eps
        a = alpha_of((Hte + z) @ E.T, P)
        noise_curve.append(
            {
                "rel_l2_error": eps,
                "mean_alpha": float(a.mean()),
                "argmax_agree": float(
                    (((Hte + z) @ E.T).argmax(-1) == P.argmax(-1)).float().mean()
                ),
            }
        )
        print(
            f"  eps {eps:5.3f}  alpha {a.mean():.4f}  "
            f"argmax {noise_curve[-1]['argmax_agree']:.4f}",
            flush=True,
        )
    out["precision_budget"] = noise_curve

    # ---- 3. the trivial baselines a predictor must beat -------------------
    # Persistence: reuse the previous step's hidden state. In speculative
    # decoding the verification pass returns it at zero cost, so this is what
    # "do nothing" actually scores -- and any learned predictor has to beat it.
    prev = {int(r): None for r in te_rows}
    d_idx = df.set_index("row")
    pid_te = d_idx.loc[te_rows, "prompt_id"].to_numpy()
    step_te = d_idx.loc[te_rows, "step"].to_numpy()
    row_by = {
        (int(p), int(s)): int(r)
        for r, p, s in zip(
            df.row.to_numpy(), df.prompt_id.to_numpy(), df.step.to_numpy()
        )
    }
    ok, prev_rows = [], []
    for i, (p_, s_) in enumerate(zip(pid_te, step_te)):
        r = row_by.get((int(p_), int(s_) - 1))
        if r is not None:
            ok.append(i)
            prev_rows.append(r)
    ok = np.array(ok)
    Hprev = torch.from_numpy(H[np.array(prev_rows)]).to(device)
    a_prev = alpha_of(Hprev @ E.T, P[ok])
    a_mean = alpha_of(mu.expand(len(ok), -1) @ E.T, P[ok])
    out["baselines"] = {
        "n": int(len(ok)),
        "previous_step_h": {
            "mean_alpha": float(a_prev.mean()),
            "argmax_agree": float(
                ((Hprev @ E.T).argmax(-1) == P[ok].argmax(-1)).float().mean()
            ),
        },
        "train_mean_h": {"mean_alpha": float(a_mean.mean())},
    }
    print(f"  prev-step h  alpha {a_prev.mean():.4f}")
    print(f"  mean h       alpha {a_mean.mean():.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--n-eval", type=int, default=8000)
    a = ap.parse_args()
    rundir = Path(a.run)
    res = run(rundir, n_eval=a.n_eval)
    (rundir / "exp4_embedding.json").write_text(json.dumps(res, indent=2))
    print("\nidentity check:", json.dumps(res["identity_check"]))


if __name__ == "__main__":
    main()
