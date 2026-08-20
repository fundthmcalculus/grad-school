"""Experiment 6 -- does refining against the *acceptance* objective help?

Every arm so far was fitted by least squares: PCA minimises L2 reconstruction
of `h`, ridge minimises L2 prediction error of `h`. But the quantity that
decides a drafter is

    alpha = sum_x min(p(x), q(x)),   q = softmax(h_hat @ E.T)

and L2 is only a surrogate for it. The two can disagree sharply, because a
fixed L2 budget can be spent well or badly: error aligned with the embedding
directions that carry the top tokens costs a great deal of alpha, error in
directions no token occupies costs almost none. Section 1.1 already showed the
two metrics diverging in the other direction -- 74% explained variance buying
alpha = 0.18.

So this asks whether the ceilings measured in sections 9-11 are properties of
the *problem* or of the *objective used to fit*. This is the "closed-form fit,
then refine with an optimizer against the real loss" pattern `tribble-fis`
already uses for antecedents, applied to the thing we actually want.

Two tests, both initialised from the least-squares solution so the refinement
can only be credited with what it adds:

  T1  rank-k autoencoder, oracle coefficients. PCA basis vs an alpha-optimised
      basis. Tests whether the *representational* ceiling of section 9.2 is
      basis-dependent -- if a learned rank-16 basis reaches far past 0.18, then
      "16 numbers are not enough" was never the right conclusion.

  T2  the context predictor of section 10. Ridge weights vs alpha-refined
      weights. Tests the same question for the *predictive* ceiling.

alpha is subdifferentiable (min and abs are), so it can be optimised directly
rather than through a smooth proxy. Both are also run against KL(p||q) for
comparison, since a smooth surrogate closer to alpha than L2 may be the better
practical objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .exp5_context import (
    alpha_of,
    build_context_features,
    load_head,
    load_input_embeddings,
    ridge_apply,
    ridge_fit,
)


def alpha_loss(H_hat, P, E):
    """Negative mean acceptance. Directly the objective, not a proxy."""
    q = torch.softmax(H_hat @ E.T, dim=-1)
    return -torch.minimum(P, q).sum(-1).mean()


def kl_loss(H_hat, P, E):
    """KL(p || q): smooth, and unlike L2 it lives on the simplex."""
    logq = torch.log_softmax(H_hat @ E.T, dim=-1)
    return -(P * logq).sum(-1).mean()


LOSSES = {"alpha": alpha_loss, "kl": kl_loss}


def _minibatch_opt(
    params, forward, Htr_true, Etr, P_fn, loss_fn, steps=300, bs=256, lr=1e-3, seed=0
):
    """Adam on `params`; `forward(idx)` returns predicted h for those rows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    n = len(Htr_true)
    for t in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        P = P_fn(idx)
        loss = loss_fn(forward(idx), P, Etr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


def run(rundir: Path, device="cuda", n_eval=6000, seed=0, steps=400) -> dict:
    meta = json.loads((rundir / "meta.json").read_text())
    E = load_head(meta["config"]["model_id"], device)
    Ein, tied = load_input_embeddings(meta["config"]["model_id"], device)
    H = torch.from_numpy(np.load(rundir / "hidden_last.npy")).to(device)
    df = pd.read_parquet(rundir / "steps.parquet").reset_index(drop=True)
    D = H.shape[1]

    rng = np.random.default_rng(seed)
    pids = df.prompt_id.unique()
    rng.shuffle(pids)
    tr_p = set(pids[: int(0.7 * len(pids))])
    is_tr = df.prompt_id.isin(tr_p).to_numpy()
    tr_i = np.where(is_tr)[0]
    te_i = np.where(~is_tr)[0]
    te_i = rng.choice(te_i, size=min(n_eval, len(te_i)), replace=False)
    tr_i = rng.choice(tr_i, size=min(20000, len(tr_i)), replace=False)

    Htr, Hte = H[tr_i], H[te_i]
    Pte = torch.cat(
        [
            torch.softmax(Hte[i : i + 2048] @ E.T, dim=-1)
            for i in range(0, len(Hte), 2048)
        ]
    )
    mu = Htr.mean(0, keepdim=True)
    P_tr = lambda idx: torch.softmax(Htr[idx.to(device)] @ E.T, dim=-1)

    out: dict = {"model": meta["config"]["model_id"], "hidden": D, "T1": {}, "T2": {}}

    # ---------------- T1: is the rank-k ceiling basis-dependent? -----------
    _, S, V = torch.pca_lowrank(Htr - mu, q=min(256, len(Htr) - 1), niter=4)
    for k in (8, 16, 32):
        Vk = V[:, :k].contiguous()
        pca_hat = ((Hte - mu) @ Vk) @ Vk.T + mu
        a_pca, _ = alpha_of(pca_hat, Pte, E)

        row = {"pca": a_pca}
        for lname, lfn in LOSSES.items():
            B = Vk.clone().requires_grad_(True)  # encoder
            A = Vk.T.clone().requires_grad_(True)  # decoder
            fwd = lambda idx, B=B, A=A: ((Htr[idx.to(device)] - mu) @ B) @ A + mu
            _minibatch_opt(
                [B, A], fwd, Htr, E, P_tr, lfn, steps=steps, lr=3e-3, seed=seed
            )
            with torch.no_grad():
                hat = ((Hte - mu) @ B) @ A + mu
                row[lname], _ = alpha_of(hat, Pte, E)
        out["T1"][f"rank_{k}"] = row
        print(
            f"  T1 rank {k:3d}: pca {row['pca']:.4f} -> "
            f"alpha-opt {row['alpha']:.4f}  kl-opt {row['kl']:.4f}",
            flush=True,
        )

    # ---------------- T2: refine the context predictor ---------------------
    feat, _, _ = build_context_features(rundir, df, Ein, device)
    row_by = {
        (int(p), int(s)): int(r) for r, p, s in zip(df.row, df.prompt_id, df.step)
    }
    ridx = {int(r): i for i, r in enumerate(df.row.to_numpy())}
    prev = np.full(len(df), -1, dtype=np.int64)
    for i, (p, s) in enumerate(zip(df.prompt_id.to_numpy(), df.step.to_numpy())):
        r = row_by.get((int(p), int(s) - 1))
        if r is not None:
            prev[i] = ridx[r]
    ok = prev >= 0
    tr2 = np.array([i for i in tr_i if ok[i]])
    te2 = np.array([i for i in te_i if ok[i]])

    X = torch.cat([feat, H[prev]], dim=1)
    Xtr, Xte = X[tr2], X[te2]
    Ytr, Yte = H[tr2], H[te2]
    Pte2 = torch.cat(
        [
            torch.softmax(Yte[i : i + 2048] @ E.T, dim=-1)
            for i in range(0, len(Yte), 2048)
        ]
    )
    P_tr2 = lambda idx: torch.softmax(Ytr[idx.to(device)] @ E.T, dim=-1)

    W0 = ridge_fit(Xtr, Ytr, 10.0)
    a_ridge, _ = alpha_of(ridge_apply(W0, Xte), Pte2, E)
    out["T2"]["ridge"] = a_ridge
    print(f"  T2 ridge: {a_ridge:.4f}", flush=True)

    for lname, lfn in LOSSES.items():
        W = W0.clone().requires_grad_(True)
        fwd = lambda idx, W=W: ridge_apply(W, Xtr[idx.to(device)])
        _minibatch_opt([W], fwd, Ytr, E, P_tr2, lfn, steps=steps, lr=1e-3, seed=seed)
        with torch.no_grad():
            a, _ = alpha_of(ridge_apply(W, Xte), Pte2, E)
        out["T2"][lname] = a
        print(f"  T2 refined ({lname}): {a:.4f}", flush=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=6000)
    a = ap.parse_args()
    rundir = Path(a.run)
    r = run(rundir, n_eval=a.n_eval, steps=a.steps)
    (rundir / "exp6_refine.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
