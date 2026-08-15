"""Experiment 8 -- is 0.212 a function-class limit or a feature limit?

Experiment 7 left a 2.7x gap: an oracle rank-16 code reaches alpha = 0.574,
and the best predictor of that code from cheap features reaches 0.212. Two
different things could cause it.

  function-class limit   the features determine `h` well enough, and linear /
                         GBM / FIS are simply too weak to extract it
  feature limit          the information is not in the features at all, and no
                         predictor recovers it

A small nonlinear predictor separates them, and it is the arm that has been
missing: every predictor so far was linear in the features (ridge), additive in
them (GBM), or a rule base over eight of them (FIS).

Two feature sets, to avoid confounding the two questions:

  reduced   the 63 PCA/scalar features experiment 7 used -- isolates function
            class, since the features are held identical
  full      the raw 4x D bag-of-embeddings plus the previous hidden state --
            says whether experiment 7's PCA reduction was itself the bottleneck

Trained against KL(p||q) rather than alpha directly: experiment 6 measured KL
as the better surrogate (0.3048 vs 0.2767), because `min(p,q)` supplies no
gradient wherever q > p. Reported in alpha regardless.

Inference cost is measured per arm. A drafter that closes the gap but costs
more than the target model's forward pass has not closed anything.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from .exp1b_predictive import TIER_A
from .exp5_context import (
    alpha_of,
    build_context_features,
    load_head,
    load_input_embeddings,
)

warnings.filterwarnings("ignore")


def make_mlp(d_in, k, hidden):
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.GELU()]
        prev = h
    layers += [nn.Linear(prev, k)]
    return nn.Sequential(*layers)


def train(model, A, mu, X, Htgt, E, steps, lr, bs=256, seed=0, joint_decoder=True):
    params = list(model.parameters()) + ([A] if joint_decoder else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (bs,), generator=g).to(X.device)
        P = torch.softmax(Htgt[idx] @ E.T, dim=-1)
        logq = torch.log_softmax((model(X[idx]) @ A + mu) @ E.T, dim=-1)
        loss = -(P * logq).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


def run(rundir: Path, k=16, device="cuda", n_eval=6000, steps=4000, seed=0) -> dict:
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
    tr_i = rng.permutation(np.where(is_tr & ok)[0])[:30000]
    te_i = rng.permutation(np.where((~is_tr) & ok)[0])[:n_eval]

    Htr, Hte = H[tr_i], H[te_i]
    Pte = torch.cat(
        [torch.softmax(Hte[i : i + 2048] @ E.T, dim=-1) for i in range(0, len(Hte), 2048)]
    )
    mu = Htr.mean(0, keepdim=True)

    # feature banks
    ctx_np, prev_np = feat.cpu().numpy(), H[prev].cpu().numpy()
    p_ctx = PCA(n_components=32, random_state=seed).fit(ctx_np[tr_i])
    p_prev = PCA(n_components=16, random_state=seed).fit(prev_np[tr_i])
    red = np.hstack(
        [p_ctx.transform(ctx_np), p_prev.transform(prev_np),
         df[TIER_A].to_numpy(dtype=np.float64)]
    )
    banks = {
        "reduced": torch.tensor(red, dtype=torch.float32, device=device),
        "full": torch.cat([feat, H[prev]], dim=1),
    }
    # standardise each bank on train
    for name, B in banks.items():
        m, s = B[tr_i].mean(0, keepdim=True), B[tr_i].std(0, keepdim=True).clamp_min(1e-6)
        banks[name] = (B - m) / s

    _, _, V = torch.pca_lowrank(Htr - mu, q=min(128, len(Htr) - 1), niter=4)
    out: dict = {"k": k, "model": meta["config"]["model_id"], "arms": {}}

    for bank_name, B in banks.items():
        Xtr, Xte = B[tr_i], B[te_i]
        for hidden in ([256], [512, 512]):
            A = V[:, :k].T.clone().requires_grad_(True)
            net = make_mlp(Xtr.shape[1], k, hidden).to(device)
            train(net, A, mu, Xtr, Htr, E, steps, lr=1e-3, seed=seed)
            net.eval()
            with torch.no_grad():
                a, agree = alpha_of(net(Xte) @ A.detach() + mu, Pte, E)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(50):
                    net(Xte[:1]) @ A.detach() + mu
                torch.cuda.synchronize()
                us = 1e6 * (time.time() - t0) / 50
            tag = f"mlp{'x'.join(map(str, hidden))}_{bank_name}"
            out["arms"][tag] = {
                "alpha": a,
                "argmax_agree": agree,
                "n_features": int(Xtr.shape[1]),
                "params": sum(p.numel() for p in net.parameters()),
                "us_per_step": us,
            }
            print(
                f"  {tag:24s} alpha {a:.4f}  argmax {agree:.4f}  "
                f"{out['arms'][tag]['params'] / 1e3:.0f}k params  {us:.0f} us/step",
                flush=True,
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("-k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    rundir = Path(a.run)
    r = run(rundir, k=a.k, steps=a.steps)
    (rundir / f"exp8_nonlinear_k{a.k}.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
