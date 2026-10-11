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
    p.add_argument(
        "--fib-local",
        type=int,
        default=8,
        help="fibsoftmax/fibgauss: dense lags 0..N-1, then Fibonacci lags only",
    )
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
    p.add_argument(
        "--ante-taps", type=int, default=1, help="layer-0 TSK FFN antecedent taps"
    )
    p.add_argument(
        "--ngram-init",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="init layer-0 FFN rules from order-k n-gram stats (1 bigram, 2 trigram)",
    )
    p.add_argument(
        "--ngram-mode",
        default="full",
        choices=["emb", "placement", "full"],
        help="emb: aligned embedding only; placement: + rule placement; full: + decoded consequents",
    )
    p.add_argument(
        "--eval-at-start",
        action="store_true",
        help="also evaluate before the first step",
    )
    p.add_argument("--width-share", default="full", choices=["full", "dim", "rule"])
    p.add_argument(
        "--n-unique", type=int, default=0, help="distinct blocks (0 = --layers)"
    )
    p.add_argument(
        "--init-from", default="", help="post-training: start from this .pt checkpoint"
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="weight EMA (0 = off); evaluated at the end as val_bpc_ema",
    )
    p.add_argument(
        "--teacher",
        default="",
        help="distillation: .pt checkpoint of the teacher model",
    )
    p.add_argument(
        "--kd-weight", type=float, default=0.5, help="weight of the distillation term"
    )
    p.add_argument("--kd-temp", type=float, default=1.0)
    p.add_argument(
        "--tok", default="", help="subword stream, e.g. bpe512 (default: characters)"
    )
    p.add_argument("--ffn-order", type=int, default=0, choices=[0, 1])
    p.add_argument("--ffn-width", default="", choices=["", "dim", "rule", "shared"])
    p.add_argument("--ffn-norm", default="softmax", choices=["softmax", "none"])
    p.add_argument(
        "--rule-init",
        default="random",
        choices=["random", "data"],
        help="TSK rule centers: random or on data",
    )
    return p.parse_args(argv)


TOK = None  # a flm.bpe.BPE instance when training on subwords (set in main)


def make_config(a) -> ModelConfig:
    return ModelConfig(
        vocab_size=getattr(a, "vocab_size", 0) or VOCAB_SIZE,
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
        ffn_order=a.ffn_order,
        ffn_width=a.ffn_width,
        ffn_norm=a.ffn_norm,
        ante_taps=a.ante_taps,
        fib_local=a.fib_local,
    )


@torch.no_grad()
def evaluate(
    model,
    val: np.ndarray,
    ctx: int,
    eval_chars: int,
    batch: int = 64,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Mean bits per character over non-overlapping windows of the validation slice."""
    model.eval()
    lens = None
    if TOK is not None:  # subwords: eval_chars of text = however many tokens that takes
        lens = np.asarray(TOK.lens)
        cum = np.cumsum(lens[np.asarray(val[:eval_chars])].astype(np.int64))
        n_tok = int(np.searchsorted(cum, eval_chars)) + 1
        n_win = n_tok // (ctx + 1)
    else:
        n_win = eval_chars // (ctx + 1)
    dev = next(model.parameters()).device
    data = torch.from_numpy(np.asarray(val[: n_win * (ctx + 1)], dtype=np.int64)).view(
        n_win, ctx + 1
    )
    data = data.to(device)
    tot, cnt = 0.0, 0
    for i in range(0, n_win, batch):
        xb = data[i : i + batch].to(dev)
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1), reduction="sum"
        )
        tot += loss.item()
        if lens is None:
            cnt += xb[:, 1:].numel()
        else:  # bits per CHARACTER: divide by the characters the target tokens spell
            cnt += int(lens[xb[:, 1:].cpu().numpy()].astype(np.int64).sum())
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
    if a.ngram_init and a.tok:
        raise SystemExit(
            "--ngram-init is defined for the 98-symbol character stream only (its statistics, "
            "aligned embedding and decoded consequents assume that vocabulary); drop --tok or --ngram-init"
        )
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    global TOK
    cpt = 1.0
    if a.tok:
        from flm.bpe import BPE

        TOK = BPE(int(a.tok.removeprefix("bpe")))
        a.vocab_size = TOK.tok.get_vocab_size()
        train, val = TOK.load("train"), TOK.load("valid")
        cpt = TOK.chars_per_token()
    else:
        train, val = load("train"), load("valid")
    cfg = make_config(a)
    model = TinyLM(cfg)
    rule_init = {}
    if a.init_from:  # post-training: weights come from a pretrained checkpoint
        ck = torch.load(a.init_from, map_location="cpu")
        model.load_state_dict(ck["state"])  # strict: architecture must match
    elif a.rule_init == "data":
        g = torch.Generator().manual_seed(a.seed)
        starts = rng.integers(0, len(train) - a.ctx - 1, 64)
        idx0 = torch.from_numpy(
            np.stack([train[i : i + a.ctx] for i in starts]).astype(np.int64)
        )
        rule_init = init_rules_from_data(model, idx0, generator=g)
    ngram_info = {}
    if a.ngram_init and not a.init_from:
        from flm.ngram import ngram_init_for_model

        ngram_info = ngram_init_for_model(model, a.ngram_init, a.seed, a.ngram_mode)
        print("ngram init:", ngram_info, flush=True)
    # init (incl. data-driven rule init) happens on CPU, so initial weights are identical
    # on every platform; batch sampling is numpy-seeded, so the data order is too
    device = torch.device(a.device)
    model.to(device)
    on_gpu = device.type == "cuda"

    teacher = None
    if a.teacher:
        tck = torch.load(a.teacher, map_location="cpu")
        teacher = TinyLM(ModelConfig(**tck["config"]))
        teacher.load_state_dict(tck["state"])
        teacher.to(device).eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
    ema_model = None
    if a.ema_decay > 0:
        import copy

        ema_model = copy.deepcopy(model).eval()
        for q in ema_model.parameters():
            q.requires_grad_(False)

    steps = int(math.ceil(a.chars / (a.batch * a.ctx * cpt)))
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
    if a.eval_at_start:
        evals.append(
            (0, round(evaluate(model, val, a.ctx, a.eval_chars, device=device), 5))
        )
        print(f"step 0 val {evals[-1][1]:.4f}", flush=True)
    t0 = time.perf_counter()
    train_time = 0.0
    ema = None
    diverged_step = None
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
        if teacher is not None:
            with torch.no_grad():
                tl = teacher(xb[:, :-1]) / a.kd_temp
            kd = (
                F.kl_div(
                    F.log_softmax(logits / a.kd_temp, -1).reshape(-1, logits.shape[-1]),
                    F.softmax(tl, -1).reshape(-1, logits.shape[-1]),
                    reduction="batchmean",
                )
                * a.kd_temp**2
            )
            # the logged loss stays the hard-label CE; the optimiser sees the mixture
            loss_opt = (1 - a.kd_weight) * loss + a.kd_weight * kd
        else:
            loss_opt = loss
        opt.zero_grad(set_to_none=True)
        loss_opt.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema_model is not None:
            with torch.no_grad():
                d_ = min(a.ema_decay, (1 + s) / (10 + s))
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(d_).add_(pm.detach(), alpha=1 - d_)
        if on_gpu:  # CUDA is asynchronous: without this, train_time times launches
            torch.cuda.synchronize()
        train_time += time.perf_counter() - ts
        li = loss.item() / LN2
        if not math.isfinite(li):
            diverged_step = s + 1
            break
        ema = li if ema is None else 0.98 * ema + 0.02 * li
        if s % 50 == 0 or s == steps - 1:
            curve.append((s, round(ema, 4)))
        if (s + 1) % eval_every == 0 or s == steps - 1:
            evals.append(
                (
                    s + 1,
                    round(evaluate(model, val, a.ctx, a.eval_chars, device=device), 5),
                )
            )
            print(
                f"step {s + 1}/{steps} train {ema:.4f} val {evals[-1][1]:.4f} t={train_time:.0f}s",
                flush=True,
            )

    # A diverged run has no final value. Reporting the last finite eval would let a
    # truncated run compete in `best_tune` as if it had finished.
    val_bpc_ema = None
    if ema_model is not None and diverged_step is None:
        val_bpc_ema = round(
            evaluate(ema_model, val, a.ctx, a.eval_chars, device=device), 5
        )
        print(f"EMA({a.ema_decay}) val {val_bpc_ema:.4f}", flush=True)
    final = (
        evals[-1][1]
        if evals and diverged_step is None and math.isfinite(evals[-1][1])
        else float("nan")
    )
    model.cpu()  # sample and checkpoint on CPU, whatever the training device
    g = torch.Generator().manual_seed(1234)
    enc, dec = (TOK.encode, TOK.decode) if TOK is not None else (encode, decode)
    prompt = torch.from_numpy(enc("Once upon a time").astype(np.int64))[None]
    sample = ""
    if math.isfinite(final):
        try:  # a sampling failure must never discard a finished run
            sample = dec(
                model.generate(prompt, 300, temperature=0.8, generator=g)[0].tolist()
            )
        except Exception as e:  # noqa: BLE001
            sample = f"[sampling failed: {type(e).__name__}: {e}]"

    rec = {
        "args": vars(a),
        "config": cfg.to_dict(),
        "params_total": model.n_params(),
        "params_nonemb": model.n_params(exclude_embedding=True),
        "steps": steps,
        "chars_seen": int(steps * a.batch * a.ctx * cpt),
        "chars_per_token": cpt,
        "val_bpc": final,
        "val_bpc_ema": val_bpc_ema,
        "evals": evals,
        "train_curve": curve,
        "train_seconds": round(train_time, 2),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "chars_per_sec": round(steps * a.batch * a.ctx / max(train_time, 1e-9)),
        "torch": torch.__version__,
        "diverged_step": diverged_step,
        "threads": a.threads,
        "device": a.device,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else cpu_name()
        ),
        "host": socket.gethostname(),
        "sample": sample,
        "rule_init": rule_init,
        "ngram_init": ngram_info,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    if a.save_model:
        torch.save(
            {
                "config": cfg.to_dict(),
                "tok": a.tok,
                "state": {k: v.cpu() for k, v in model.state_dict().items()},
            },
            out.with_suffix(".pt"),
        )
    print(
        f"{out.name}: params={rec['params_total']} val_bpc={final:.4f} {rec['chars_per_sec']} c/s {rec['wall_seconds']}s"
    )
    return rec


if __name__ == "__main__":
    main()
