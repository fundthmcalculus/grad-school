"""Experiment 7 -- a rank-k bottleneck drafter, and whether an FIS can drive it.

Experiment 6 overturned section 9.2. The rank-k ceiling is not a property of
the problem, it was a property of PCA: at rank 16 an L2-optimal basis gives
alpha = 0.179 and an alpha-optimised one gives 0.394. So "an FIS emitting 8-16
numbers caps at 0.14-0.18" was measuring the wrong basis, and the architecture
the original hypothesis asked for -- a handful of numbers out of a rule base,
expanded to a full distribution -- deserves testing properly.

The architecture this implies:

    cheap features --[predictor]--> z in R^k --[decoder A]--> h_hat --> logits

with the decoder and the code space learned end-to-end against alpha, and the
predictor being the part an FIS could plausibly be.

Three predictors are compared **on the identical feature set, decoder and code
targets**, so the only thing varying is the function class:

  linear   ridge on the reduced features
  gbm      gradient boosting, one model per code dimension
  fis      `TribbleRegressor`, one per code dimension -- the actual proposal

The feature set is deliberately modest (PCA of the context bag, PCA of the
previous hidden state, plus the tier-A scalars). Handing an FIS 2,304 raw
embedding dimensions would destroy the readability that is its whole reason for
being here, and would walk into the TSK curse-of-dimensionality result
(Cui, Wu & Xu, arXiv:2102.04271).

Two reference points bound every arm: `oracle_code` (the true projection of
`h`, i.e. experiment 6's ceiling) and the end-to-end jointly-trained linear
predictor.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from .exp1b_predictive import TIER_A
from .exp5_context import (
    alpha_of,
    build_context_features,
    load_head,
    load_input_embeddings,
)

warnings.filterwarnings("ignore")


def run(rundir: Path, k=16, device="cuda", n_eval=6000, steps=1500, seed=0) -> dict:
    meta = json.loads((rundir / "meta.json").read_text())
    E = load_head(meta["config"]["model_id"], device)
    Ein, _ = load_input_embeddings(meta["config"]["model_id"], device)
    H = torch.from_numpy(np.load(rundir / "hidden_last.npy")).to(device)
    df = pd.read_parquet(rundir / "steps.parquet").reset_index(drop=True)
    D = H.shape[1]

    feat, _, _ = build_context_features(rundir, df, Ein, device)
    row_by = {(int(p), int(s)): int(r) for r, p, s in zip(df.row, df.prompt_id, df.step)}
    ridx = {int(r): i for i, r in enumerate(df.row.to_numpy())}
    prev = np.full(len(df), -1, dtype=np.int64)
    for i, (p, s) in enumerate(zip(df.prompt_id.to_numpy(), df.step.to_numpy())):
        r = row_by.get((int(p), int(s) - 1))
        if r is not None:
            prev[i] = ridx[r]
    ok = (prev >= 0) & np.isfinite(df[TIER_A].to_numpy()).all(1)

    rng = np.random.default_rng(seed)
    pids = df.prompt_id.unique()
    rng.shuffle(pids)
    tr_p = set(pids[: int(0.7 * len(pids))])
    is_tr = df.prompt_id.isin(tr_p).to_numpy()
    tr_i = rng.permutation(np.where(is_tr & ok)[0])[:20000]
    te_i = rng.permutation(np.where((~is_tr) & ok)[0])[:n_eval]

    Htr, Hte = H[tr_i], H[te_i]
    Pte = torch.cat(
        [torch.softmax(Hte[i : i + 2048] @ E.T, dim=-1) for i in range(0, len(Hte), 2048)]
    )
    mu = Htr.mean(0, keepdim=True)

    # ---- reduced, FIS-appropriate feature set -----------------------------
    ctx_np = feat.cpu().numpy()
    prev_np = H[prev].cpu().numpy()
    p_ctx = PCA(n_components=32, random_state=seed).fit(ctx_np[tr_i])
    p_prev = PCA(n_components=16, random_state=seed).fit(prev_np[tr_i])
    scal = df[TIER_A].to_numpy(dtype=np.float64)
    Xall = np.hstack([p_ctx.transform(ctx_np), p_prev.transform(prev_np), scal])
    names = (
        [f"ctx{i}" for i in range(32)] + [f"prev{i}" for i in range(16)] + list(TIER_A)
    )
    Xtr_np, Xte_np = Xall[tr_i], Xall[te_i]
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=device)
    Xte = torch.tensor(Xte_np, dtype=torch.float32, device=device)

    out: dict = {"k": k, "n_features": Xall.shape[1], "model": meta["config"]["model_id"]}

    # ---- end-to-end: learn code space + decoder + linear predictor --------
    _, _, V = torch.pca_lowrank(Htr - mu, q=min(128, len(Htr) - 1), niter=4)
    A = V[:, :k].T.clone().requires_grad_(True)          # decoder  k -> D
    Wp = torch.zeros((Xtr.shape[1] + 1, k), device=device)
    Wp[: Xtr.shape[1]] = (
        torch.linalg.lstsq(Xtr, (Htr - mu) @ V[:, :k]).solution
    )
    Wp = Wp.clone().requires_grad_(True)

    def pred_code(X, W):
        return torch.cat([X, torch.ones((len(X), 1), device=X.device)], 1) @ W

    opt = torch.optim.Adam([A, Wp], lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    for t in range(steps):
        idx = torch.randint(0, len(tr_i), (256,), generator=g).to(device)
        P = torch.softmax(Htr[idx] @ E.T, dim=-1)
        hh = pred_code(Xtr[idx], Wp) @ A + mu
        loss = -torch.minimum(P, torch.softmax(hh @ E.T, dim=-1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    A = A.detach()
    with torch.no_grad():
        a_lin, _ = alpha_of(pred_code(Xte, Wp.detach()) @ A + mu, Pte, E)
    out["e2e_linear"] = a_lin

    def best_codes(Htgt, n_steps=250):
        """Per-sample alpha-optimal code for this decoder.

        The L2-optimal projection `(h-mu) @ pinv(A)` is *not* the right target
        here: the decoder was trained against alpha, so the code that minimises
        reconstruction error is not the code that maximises acceptance. Using
        it produced an "oracle" that scored *below* the learned predictor,
        which is how the mistake surfaced. This optimises z directly.
        """
        z = ((Htgt - mu) @ torch.linalg.pinv(A)).clone().requires_grad_(True)
        o = torch.optim.Adam([z], lr=3e-2)
        for _ in range(n_steps):
            tot = 0.0
            for i in range(0, len(z), 4096):
                P = torch.softmax(Htgt[i : i + 4096] @ E.T, dim=-1)
                q = torch.softmax((z[i : i + 4096] @ A + mu) @ E.T, dim=-1)
                loss = -torch.minimum(P, q).sum(-1).mean()
                o.zero_grad()
                loss.backward()
                o.step()
                tot += float(loss)
        return z.detach()

    z_te = best_codes(Hte)
    with torch.no_grad():
        a_oracle, _ = alpha_of(z_te @ A + mu, Pte, E)
    out["oracle_code_same_decoder"] = a_oracle
    print(f"  oracle code (same decoder) : {a_oracle:.4f}")
    print(f"  end-to-end linear predictor: {a_lin:.4f}", flush=True)

    # ---- code targets every non-differentiable arm is fitted to -----------
    Ztr_tgt = best_codes(Htr).cpu().numpy()

    def decode_and_score(Zhat):
        with torch.no_grad():
            hh = torch.tensor(Zhat, dtype=torch.float32, device=device) @ A + mu
            return alpha_of(hh, Pte, E)[0]

    # ---- gbm ---------------------------------------------------------------
    from sklearn.ensemble import HistGradientBoostingRegressor

    Zg = np.column_stack([
        HistGradientBoostingRegressor(max_iter=200, random_state=seed)
        .fit(Xtr_np, Ztr_tgt[:, j])
        .predict(Xte_np)
        for j in range(k)
    ])
    out["gbm"] = decode_and_score(Zg)
    print(f"  gbm  (k={k} models)        : {out['gbm']:.4f}", flush=True)

    # ---- fis ---------------------------------------------------------------
    from tribblefis.gaussian_regressor import TribbleRegressor

    Xtr_df = pd.DataFrame(Xtr_np, columns=names)
    Xte_df = pd.DataFrame(Xte_np, columns=names)
    Zf = np.column_stack([
        TribbleRegressor(top_n=8, n_gaussians=3, n_output_buckets=7,
                         tsk_order="1st", norm_conorm="probability",
                         random_state=seed)
        .fit(Xtr_df, Ztr_tgt[:, j])
        .predict(Xte_df)
        for j in range(k)
    ])
    out["fis"] = decode_and_score(Zf)
    print(f"  fis  (k={k} TribbleRegr.)  : {out['fis']:.4f}", flush=True)

    # ---- linear on the same targets (control for the fitting, not the class)
    Wl = np.linalg.lstsq(
        np.hstack([Xtr_np, np.ones((len(Xtr_np), 1))]), Ztr_tgt, rcond=None
    )[0]
    Zl = np.hstack([Xte_np, np.ones((len(Xte_np), 1))]) @ Wl
    out["linear_same_targets"] = decode_and_score(Zl)
    print(f"  linear (same targets)      : {out['linear_same_targets']:.4f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("-k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    rundir = Path(a.run)
    r = run(rundir, k=a.k, steps=a.steps)
    (rundir / f"exp7_bottleneck_k{a.k}.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
