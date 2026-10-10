"""Train one tiny LM on TinyStories characters on CPU and write a JSON run record.

    .venv/bin/python -m flm.train --mixer fuzzy --ffn tsk --d 32 --layers 2 --out outputs/runs/x.json

Every run sees the same character budget (``--chars``), the same validation slice
(the first ``--eval-chars`` of valid.bin, non-overlapping windows), and is seeded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flm.data import VOCAB_SIZE, decode, encode, load
from flm.models import ModelConfig, TinyLM, init_rules_from_data

LN2 = math.log(2.0)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mixer", default="softmax")
    p.add_argument("--ffn", default="mlp")
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--rules", type=int, default=16)
    p.add_argument("--ffn-mult", type=float, default=2.0)
    p.add_argument("--tsk-dim", type=int, default=0)
    p.add_argument("--shortconv", type=int, default=0)
    p.add_argument("--decay", default="fixed", choices=["none", "fixed", "data"])
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--chars", type=float, default=20e6, help="training budget in characters"
    )
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--device", default="cpu", help="cpu | cuda | cuda:N (recorded in the run JSON)"
    )
    p.add_argument("--eval-chars", type=int, default=1_000_000)
    p.add_argument(
        "--eval-every", type=int, default=0, help="steps between evals (0 = 4 evals)"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--save-model", action="store_true")
    p.add_argument(
        "--exp-norm",
        default="sum",
        choices=["sum", "mean", "sqrt"],
        help="TSK exponent over dims: sum | mean (HTSK) | sqrt",
    )
    p.add_argument(
        "--ffn-exp-norm",
        default="",
        choices=["", "sum", "mean", "sqrt"],
        help="override --exp-norm for the TSK FFN only",
    )
    p.add_argument("--width-share", default="full", choices=["full", "dim", "rule"])
    p.add_argument(
        "--n-unique", type=int, default=0, help="distinct blocks (0 = --layers)"
    )
    p.add_argument(
        "--rule-init",
        default="random",
        choices=["random", "data"],
        help="TSK rule centers: random or on data",
    )
    return p.parse_args(argv)


def make_config(a) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        d_model=a.d,
        n_layers=a.layers,
        n_heads=a.heads,
        mixer=a.mixer,
        ffn=a.ffn,
        ffn_mult=a.ffn_mult,
        tsk_dim=a.tsk_dim,
        n_rules=a.rules,
        decay=a.decay,
        shortconv=a.shortconv,
        max_len=a.ctx,
        exp_norm=a.exp_norm,
        ffn_exp_norm=a.ffn_exp_norm,
        width_share=a.width_share,
        n_unique=a.n_unique,
    )


@torch.no_grad()
def evaluate(
    model, val: np.ndarray, ctx: int, eval_chars: int, batch: int = 64
) -> float:
    """Mean bits per character over non-overlapping windows of the validation slice."""
    model.eval()
    n_win = eval_chars // (ctx + 1)
    dev = next(model.parameters()).device
    data = torch.from_numpy(np.asarray(val[: n_win * (ctx + 1)], dtype=np.int64)).view(
        n_win, ctx + 1
    )
    tot, cnt = 0.0, 0
    for i in range(0, n_win, batch):
        xb = data[i : i + batch].to(dev)
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1), reduction="sum"
        )
        tot += loss.item()
        cnt += xb[:, 1:].numel()
    model.train()
    return tot / cnt / LN2


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main(argv=None):
    a = get_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cfg = make_config(a)
    model = TinyLM(cfg)
    train, val = load("train"), load("valid")
    rule_init = {}
    if a.rule_init == "data":
        g = torch.Generator().manual_seed(a.seed)
        starts = rng.integers(0, len(train) - a.ctx - 1, 64)
        idx0 = torch.from_numpy(
            np.stack([train[i : i + a.ctx] for i in starts]).astype(np.int64)
        )
        rule_init = init_rules_from_data(model, idx0, generator=g)
    # init (incl. data-driven rule init) happens on CPU, so initial weights are identical
    # on every platform; batch sampling is numpy-seeded, so the data order is too
    device = torch.device(a.device)
    model.to(device)

    steps = int(math.ceil(a.chars / (a.batch * a.ctx)))
    eval_every = a.eval_every or max(1, steps // 4)
    decay_p = [
        p for n, p in model.named_parameters() if p.dim() >= 2 and "centers" not in n
    ]
    other_p = [
        p
        for n, p in model.named_parameters()
        if not (p.dim() >= 2 and "centers" not in n)
    ]
    opt = torch.optim.AdamW(
        [
            {"params": decay_p, "weight_decay": a.wd},
            {"params": other_p, "weight_decay": 0.0},
        ],
        lr=a.lr,
        betas=(0.9, 0.98),
    )

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        frac = (s - a.warmup) / max(1, steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    hi = len(train) - a.ctx - 2
    curve, evals = [], []
    t0 = time.perf_counter()
    train_time = 0.0
    ema = None
    for s in range(steps):
        ts = time.perf_counter()
        starts = rng.integers(0, hi, a.batch)
        xb = torch.from_numpy(
            np.stack([train[i : i + a.ctx + 1] for i in starts]).astype(np.int64)
        ).to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_time += time.perf_counter() - ts
        li = loss.item() / LN2
        if not math.isfinite(li):
            break
        ema = li if ema is None else 0.98 * ema + 0.02 * li
        if s % 50 == 0 or s == steps - 1:
            curve.append((s, round(ema, 4)))
        if (s + 1) % eval_every == 0 or s == steps - 1:
            evals.append((s + 1, round(evaluate(model, val, a.ctx, a.eval_chars), 5)))
            print(
                f"step {s + 1}/{steps} train {ema:.4f} val {evals[-1][1]:.4f} t={train_time:.0f}s",
                flush=True,
            )

    final = (
        evals[-1][1] if evals and math.isfinite(ema or float("nan")) else float("nan")
    )
    model.cpu()  # sample and checkpoint on CPU, whatever the training device
    g = torch.Generator().manual_seed(1234)
    prompt = torch.from_numpy(encode("Once upon a time").astype(np.int64))[None]
    sample = (
        decode(model.generate(prompt, 300, temperature=0.8, generator=g)[0].tolist())
        if math.isfinite(final)
        else ""
    )

    rec = {
        "args": vars(a),
        "config": cfg.to_dict(),
        "params_total": model.n_params(),
        "params_nonemb": model.n_params(exclude_embedding=True),
        "steps": steps,
        "chars_seen": steps * a.batch * a.ctx,
        "val_bpc": final,
        "evals": evals,
        "train_curve": curve,
        "train_seconds": round(train_time, 2),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "chars_per_sec": round(steps * a.batch * a.ctx / max(train_time, 1e-9)),
        "torch": torch.__version__,
        "threads": a.threads,
        "device": a.device,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else cpu_name()
        ),
        "host": socket.gethostname(),
        "sample": sample,
        "rule_init": rule_init,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    if a.save_model:
        torch.save(
            {"config": cfg.to_dict(), "state": model.state_dict()},
            out.with_suffix(".pt"),
        )
    print(
        f"{out.name}: params={rec['params_total']} val_bpc={final:.4f} {rec['chars_per_sec']} c/s {rec['wall_seconds']}s"
    )
    return rec


if __name__ == "__main__":
    main()
