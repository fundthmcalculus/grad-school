"""Train one tiny LM on TinyStories characters on CPU and write a JSON run record.

    .venv/bin/python -m flm.train --mixer fuzzy --ffn tsk --d 32 --layers 2 --out outputs/runs/x.json

Every run sees the same character budget (``--chars``), the same validation slice
(the first ``--eval-chars`` of valid.bin, non-overlapping windows), and is seeded.
"""

from __future__ import annotations

import argparse
import json
import contextlib
import math
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import flm.ckpt as CK
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
        "--layer-mixers",
        default="",
        help="comma-separated mixer per layer (cycled), e.g. fuzzydelta,fibgauss",
    )
    p.add_argument(
        "--attn-heads", type=int, default=0, help="heads of attention mixers"
    )
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
    p.add_argument(
        "--lr-floor",
        type=float,
        default=0.0,
        help="final lr as a fraction of --lr (0: decay to zero; 0.1 reproduces the runs made before 2026-10-22)",
    )
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--clip", type=float, default=1.0, help="grad-norm clip (0 = off)")
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
    p.add_argument(
        "--emb-dim", type=int, default=0, help="factorized tied embedding width (0 = d)"
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
    p.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "tf32", "bf16", "fp16"],
        help="fp32 (default; exact), tf32 (Ampere+ matrix units), bf16 / fp16 autocast (fp16 uses a loss scaler); validation is also re-scored in fp32 (val_bpc_fp32)",
    )
    p.add_argument(
        "--ckpt-dir", default="", help="directory of resumable training checkpoints"
    )
    p.add_argument(
        "--ckpt-every", type=int, default=0, help="steps between checkpoints (0 = off)"
    )
    p.add_argument("--ckpt-keep", type=int, default=2, help="checkpoints kept")
    p.add_argument(
        "--resume",
        default="",
        help="auto (newest in --ckpt-dir), a checkpoint file or directory, or wandb:<entity>/<project>/<artifact>:latest",
    )
    p.add_argument(
        "--wandb", default="", help="Weights & Biases project to log to (off if empty)"
    )
    p.add_argument("--wandb-name", default="", help="Weights & Biases run name")
    p.add_argument(
        "--wandb-artifacts",
        action="store_true",
        help="also upload every checkpoint as a W&B artifact",
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
        emb_dim=a.emb_dim,
        ffn_width=a.ffn_width,
        ffn_norm=a.ffn_norm,
        ante_taps=a.ante_taps,
        fib_local=a.fib_local,
        layer_mixers=a.layer_mixers,
        attn_heads=a.attn_heads,
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
        betas=(0.9, a.beta2),
    )

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        frac = (s - a.warmup) / max(1, steps - a.warmup)
        return a.lr * (
            a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + math.cos(math.pi * frac))
        )

    hi = len(train) - a.ctx - 2
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.precision)
    if a.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    scaler = torch.amp.GradScaler("cuda") if a.precision == "fp16" and on_gpu else None

    def amp():
        return (
            torch.autocast(device.type, dtype=amp_dtype)
            if amp_dtype
            else contextlib.nullcontext()
        )

    curve, evals = [], []
    if a.eval_at_start:
        with amp():
            evals.append(
                (
                    0,
                    round(evaluate(model, val, a.ctx, a.eval_chars, device=device), 5),
                )
            )
        print(f"step 0 val {evals[-1][1]:.4f}", flush=True)
    t0 = time.perf_counter()
    train_time = 0.0
    ema = None
    diverged_step = None
    start = 0
    wb_id = None
    if a.resume:
        ck = CK.load(a.resume, a.ckpt_dir)
        if ck is not None:
            CK.check_compatible(ck["run_key"], a)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            rng.bit_generator.state = ck["np_rng"]
            torch.set_rng_state(ck["torch_rng"])
            if on_gpu and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ck["cuda_rng"], device)
            if ema_model is not None and ck.get("ema_model") is not None:
                ema_model.load_state_dict(ck["ema_model"])
            if scaler is not None and ck.get("scaler") is not None:
                scaler.load_state_dict(ck["scaler"])
            ema, curve, evals = ck["ema"], ck["curve"], ck["evals"]
            train_time, start, wb_id = ck["train_time"], ck["step"], ck.get("wandb_id")
            print(f"resumed from step {start}/{steps}", flush=True)
        else:
            print("--resume auto: no checkpoint yet, starting from step 0", flush=True)
    wb = CK.Logger(a.wandb, a.wandb_name or Path(a.out).stem, vars(a), wb_id)
    for s in range(start, steps):
        ts = time.perf_counter()
        starts = rng.integers(0, hi, a.batch)
        xb = torch.from_numpy(
            np.stack([train[i : i + a.ctx + 1] for i in starts]).astype(np.int64)
        ).to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        with amp():
            logits = model(xb[:, :-1])
            logits = logits.float()
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
        if scaler is not None:  # fp16: scale the loss, unscale before clipping
            scaler.scale(loss_opt).backward()
            scaler.unscale_(opt)
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss_opt.backward()
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
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
            wb.log(s + 1, train_bpc=ema, lr=lr_at(s))
        if (s + 1) % eval_every == 0 or s == steps - 1:
            with amp():
                evals.append(
                    (
                        s + 1,
                        round(
                            evaluate(model, val, a.ctx, a.eval_chars, device=device), 5
                        ),
                    )
                )
            wb.log(s + 1, val_bpc=evals[-1][1], train_seconds=train_time)
            print(
                f"step {s + 1}/{steps} train {ema:.4f} val {evals[-1][1]:.4f} t={train_time:.0f}s",
                flush=True,
            )
        if (
            a.ckpt_every
            and a.ckpt_dir
            and (s + 1) % a.ckpt_every == 0
            and s + 1 < steps
        ):
            path = CK.save(
                a.ckpt_dir,
                s + 1,
                dict(
                    step=s + 1,
                    run_key=CK.run_key(a),
                    model=model.state_dict(),
                    opt=opt.state_dict(),
                    np_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device) if on_gpu else None,
                    ema_model=ema_model.state_dict() if ema_model is not None else None,
                    scaler=scaler.state_dict() if scaler is not None else None,
                    ema=ema,
                    curve=curve,
                    evals=evals,
                    train_time=train_time,
                    wandb_id=wb.id,
                    torch=torch.__version__,
                ),
                keep=a.ckpt_keep,
            )
            if a.wandb_artifacts:
                wb.artifact(path, s + 1)

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
    val_bpc_fp32 = None
    if a.precision != "fp32" and math.isfinite(final):  # the metric of record is fp32
        val_bpc_fp32 = round(
            evaluate(model, val, a.ctx, a.eval_chars, device=device), 5
        )
        print(
            f"{a.precision} val {final:.5f}, fp32 re-score {val_bpc_fp32:.5f}",
            flush=True,
        )
    wb.finish(val_bpc=final, val_bpc_fp32=val_bpc_fp32, train_seconds=train_time)
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
        "val_bpc_fp32": val_bpc_fp32,
        "evals": evals,
        "train_curve": curve,
        "train_seconds": round(train_time, 2),
        "peak_mem_mb": (
            round(torch.cuda.max_memory_allocated(device) / 2**20) if on_gpu else None
        ),
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
