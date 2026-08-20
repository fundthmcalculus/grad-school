"""Experiment 1a -- the representational ceiling on any k-output drafter.

Question: if a model emits k numbers per step, and those k numbers are
*perfect*, what acceptance rate does the resulting draft distribution get?

Method: learn a rank-k basis for the log-probability space on a training half,
project the held-out half onto it (giving each test row its oracle
coefficients), re-normalise, and measure total variation against the true
distribution. Since acceptance in speculative decoding is exactly

    alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q)

the resulting curve converts directly into an acceptance ceiling.

Three properties make this worth running before any FIS exists:

* It is an upper bound on *every* k-output model, fuzzy or otherwise. The
  coefficients are obtained by projection -- i.e. handed to the model for free.
  No predictor can do better than being told the answer.
* The basis is fit on a disjoint half, so the bound is honest rather than
  in-sample.
* Rank 0 (the mean log-prob vector alone, no per-step information) is the
  do-nothing baseline. If rank 16 is not meaningfully above rank 0, then
  per-step shape prediction buys nothing and V3 is finished.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

RANKS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def total_variation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Row-wise TV distance between two batches of distributions."""
    return 0.5 * (p - q).abs().sum(-1)


def run(rundir: Path, device: str = "cuda", max_rank: int = 512, seed: int = 0) -> dict:
    lp = np.load(rundir / "full_logprob.npy")  # (N, V) float16 log-probs
    n, v = lp.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = n // 2
    tr_i, te_i = perm[:ntr], perm[ntr:]

    X = torch.from_numpy(lp).to(device=device, dtype=torch.float32)
    Xtr, Xte = X[tr_i], X[te_i]

    # True held-out distributions. Re-softmax rather than exp: the stored
    # log-probs are float16, so they do not sum to exactly 1.
    P = torch.softmax(Xte, dim=-1)

    mu = Xtr.mean(0, keepdim=True)
    Xtr_c = Xtr - mu
    Xte_c = Xte - mu

    q = min(max_rank, min(Xtr_c.shape) - 1)
    # Randomized PCA. niter>0 matters here: the log-prob spectrum decays
    # slowly enough that the default single pass misestimates later components.
    U, S, V = torch.pca_lowrank(Xtr_c, q=q, niter=4, center=False)

    total_energy = float((Xtr_c**2).sum())
    results = []
    for k in RANKS:
        if k > q:
            continue
        if k == 0:
            recon = mu.expand_as(Xte)
        else:
            Vk = V[:, :k]
            recon = (Xte_c @ Vk) @ Vk.T + mu
        Q = torch.softmax(recon, dim=-1)
        tv = total_variation(P, Q)
        explained = float((S[:k] ** 2).sum()) / total_energy if k > 0 else 0.0
        results.append(
            {
                "rank": k,
                "mean_tv": float(tv.mean()),
                "median_tv": float(tv.median()),
                "p90_tv": float(tv.quantile(0.9)),
                "acceptance_ceiling": float(1.0 - tv.mean()),
                "explained_var_frac": explained,
            }
        )
        print(
            f"rank {k:4d}  TV {tv.mean():.4f}  alpha_max {1 - tv.mean():.4f}  "
            f"expvar {explained:.4f}",
            flush=True,
        )

    # How many latent dimensions would be needed to hit acceptance rates that
    # published drafters actually report? This is the number that decides
    # whether an FIS is in the running at all.
    def rank_for(alpha: float):
        for r in results:
            if r["acceptance_ceiling"] >= alpha:
                return r["rank"]
        return None

    return {
        "n_total": int(n),
        "n_train": int(ntr),
        "n_test": int(n - ntr),
        "vocab": int(v),
        "curve": results,
        "rank_needed_for_alpha": {
            str(a): rank_for(a) for a in (0.5, 0.6, 0.7, 0.8, 0.9)
        },
        "singular_value_decay": [float(x) for x in S[: min(64, len(S))]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-rank", type=int, default=512)
    a = ap.parse_args()
    rundir = Path(a.run)
    res = run(rundir, device=a.device, max_rank=a.max_rank)
    (rundir / "exp1a_ceiling.json").write_text(json.dumps(res, indent=2))
    print("\nrank needed for alpha:", json.dumps(res["rank_needed_for_alpha"]))


if __name__ == "__main__":
    main()
