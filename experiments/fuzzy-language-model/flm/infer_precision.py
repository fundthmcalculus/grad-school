"""Inference precision of a trained all-fuzzy checkpoint: fp32 against fp16 against int8, quality and speed (`PLAN_INFER.md`).

    python -m flm.infer_precision <checkpoint> --name F7 [--data-root <dir with tinystories/ streams>] [--freq outputs/freq_bpe6144.npz] [--cpu-threads 3] [--out report/infer_precision_<name>.json]

Formats (each is the SAME trained fp32 weights, changed only for inference):
  fp32        the reference
  fp16        weights rounded to fp16, forward pass under fp16 autocast (the fuzzy layers keep their own fp32 operations)
  bf16        the same with bfloat16 (for completeness)
  int8        every weight matrix rounded to 8 bits with one scale per row (vectors per tensor), computed in fp32 ("fake" quantisation: exact for quality)
  int8-lazy   the weights held as int8 + scale and rebuilt to fp32 before every forward pass: what an int8 store costs in time when there is no int8 kernel for these layers
Quality (against fp32, on the validation text of the checkpoint's corpus): bits per character on the first --chars characters; over --windows windows of 512 pieces, the top-1 agreement, the mean KL(fp32 || format) in bits and the largest logit difference; and greedy 64-piece continuations of 20 prompts: the share identical to fp32's and the mean position of the first difference.
Speed (GPU if present, else CPU): forward-only throughput in pieces per second at 16 x 512, and the time per piece of a 64-piece greedy continuation with batch 1 (the model has no cache; every piece re-runs the context); and on the CPU the same decode with --cpu-threads threads.  Size is the stored weights.
"""

import argparse
import json
import os
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch

import flm.train as T
from flm.models import ModelConfig, TinyLM
from flm.ptq_probe import fake_quant

FORMATS = ["fp32", "fp16", "bf16", "int8", "int8-lazy"]


def build(ck, path):
    import flm.coherence_fuzzy as CF

    ck.setdefault(
        "emb_mode", "tail"
    )  # a checkpoint outside outputs/*-shape/ would be rebuilt with the stock tiered embedding
    CF.configure(path, ck)
    m = TinyLM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["state"])
    return m.eval()


def state_for(state, fmt):
    """The weights as the format stores them, returned as an fp32 state dict, with the stored size in bytes."""
    out, nbytes = {}, 0.0
    for k, v in state.items():
        if fmt in ("fp16", "bf16"):
            dt = torch.float16 if fmt == "fp16" else torch.bfloat16
            out[k] = v.to(dt).float()
            nbytes += v.numel() * 2
        elif fmt.startswith("int8"):
            q, n_scales = fake_quant(v, 8, per_row=True)
            out[k] = q
            nbytes += v.numel() + n_scales * 2
        else:
            out[k] = v
            nbytes += v.numel() * 4
    return out, int(nbytes)


class Lazy:
    """int8 + scale held per parameter and rebuilt to fp32 before each forward pass."""

    def __init__(self, model, state):
        self.model, self.q = model, {}
        for k, v in state.items():
            if v.dim() >= 2:
                flat = v.reshape(v.shape[0], -1)
                scale = flat.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
                self.q[k] = (
                    torch.round(flat / scale).clamp(-127, 127).to(torch.int8),
                    scale,
                    v.shape,
                )
        self.params = dict(model.named_parameters())

    def rebuild(self):
        with torch.no_grad():
            for k, (q, s, shape) in self.q.items():
                if k in self.params:
                    self.params[k].copy_((q.float() * s).reshape(shape))


def autocast_for(fmt, device):
    if fmt in ("fp16", "bf16") and device.type == "cuda":
        return torch.autocast(
            "cuda", dtype=torch.float16 if fmt == "fp16" else torch.bfloat16
        )
    return torch.autocast("cpu", enabled=False)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def throughput(model, fmt, device, ids, lazy=None, reps=3, batch=16, ctx=512):
    x = ids[: batch * ctx].view(batch, ctx).to(device)
    best = []
    for _ in range(reps + 1):
        sync(device)
        t0 = time.perf_counter()
        for _ in range(6):
            if lazy:
                lazy.rebuild()
            with autocast_for(fmt, device):
                model(x)
        sync(device)
        best.append(6 * batch * ctx / (time.perf_counter() - t0))
    return st.median(best[1:])  # the first repetition warms up


@torch.no_grad()
def decode(model, fmt, device, prompt, n_new=64, lazy=None):
    ids = prompt.clone().to(device)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_new):
        if lazy:
            lazy.rebuild()
        with autocast_for(fmt, device):
            lg = model(ids[:, -model.cfg.max_len :])[:, -1].float()
        ids = torch.cat([ids, lg.argmax(-1, keepdim=True)], 1)
    sync(device)
    return ids[0, prompt.shape[1] :].cpu(), (time.perf_counter() - t0) / n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--freq", default="")
    ap.add_argument("--chars", type=int, default=1_000_000)
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--cpu-threads", type=int, default=3)
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument(
        "--cpu-only",
        action="store_true",
        help="only the CPU measurements, added to the existing report",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.data_root:
        os.environ["GRAD_SCHOOL_DATA"] = a.data_root
    if a.freq:
        os.environ["COH_TIER_FREQ"] = a.freq
    from flm.bpe import BPE

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    bp = BPE(int(ck["tok"].removeprefix("bpe")))
    T.TOK = bp
    val = np.asarray(bp.load("valid"))
    ids_all = torch.from_numpy(val[: 400 * 512].astype(np.int64))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = {k: v.clone() for k, v in ck["state"].items()}
    out = Path(a.out or f"report/infer_precision_{a.name}.json")
    prompts = [ids_all[i * 512 : i * 512 + 32][None] for i in range(20)]
    if (
        a.cpu_only
    ):  # fill the CPU keys of an existing report (the GPU part is not repeated)
        res = json.loads(out.read_text())
    else:
        ref = build(ck, a.ckpt).to(dev)
        res = dict(
            name=a.name,
            ckpt=a.ckpt,
            params=sum(p.numel() for p in ref.parameters()),
            device=str(dev),
            formats={},
        )
        windows = ids_all[: a.windows * 512].view(a.windows, 512)
        with torch.no_grad():
            ref_logits = []
            for i in range(0, a.windows, 8):
                ref_logits.append(ref(windows[i : i + 8].to(dev)).float().cpu())
            ref_logits = torch.cat(ref_logits)
        ref_text = [decode(ref, "fp32", dev, p)[0] for p in prompts]
        for fmt in FORMATS:
            state, nbytes = state_for(base, fmt)
            m = build(ck, a.ckpt)
            m.load_state_dict(state)
            m = m.to(dev).eval()
            lazy = Lazy(m, base) if fmt == "int8-lazy" else None
            if lazy:  # the held weights are the int8 ones: start from a rebuild
                lazy.rebuild()
            r = dict(bytes=nbytes)
            with torch.no_grad():
                with autocast_for(fmt, dev):
                    r["bpc"] = round(
                        T.evaluate(m, val, 512, a.chars, batch=16, device=dev), 5
                    )
                agree, kl, mx = [], [], 0.0
                for i in range(0, a.windows, 8):
                    with autocast_for(fmt, dev):
                        lg = m(windows[i : i + 8].to(dev)).float().cpu()
                    rl = ref_logits[i : i + 8]
                    agree.append((lg.argmax(-1) == rl.argmax(-1)).float().mean().item())
                    lp, lq = torch.log_softmax(rl, -1), torch.log_softmax(lg, -1)
                    kl.append((lp.exp() * (lp - lq)).sum(-1).mean().item() / np.log(2))
                    mx = max(mx, (lg - rl).abs().max().item())
            r.update(
                top1_agreement=round(float(np.mean(agree)), 5),
                kl_bits=round(float(np.mean(kl)), 6),
                max_dlogit=round(mx, 4),
            )
            same, first = [], []
            for p, rt in zip(prompts, ref_text):
                t, _ = decode(m, fmt, dev, p, lazy=lazy)
                same.append(bool((t == rt).all()))
                d = (t != rt).nonzero()
                first.append(int(d[0]) if len(d) else len(rt))
            r.update(
                greedy_identical=round(float(np.mean(same)), 3),
                mean_first_difference=round(float(np.mean(first)), 1),
            )
            flat = ids_all.to(dev) if dev.type == "cuda" else ids_all
            r["pieces_per_s"] = round(throughput(m, fmt, dev, flat, lazy))
            _, per = decode(m, fmt, dev, prompts[0], lazy=lazy)
            r["ms_per_piece_decode"] = round(per * 1000, 2)
            res["formats"][fmt] = r
            print(a.name, fmt, r, flush=True)
            del m
    if not a.no_cpu:
        torch.set_num_threads(a.cpu_threads)
        cpu = torch.device("cpu")
        for fmt in ("fp32", "fp16", "int8", "int8-lazy"):
            state, _ = state_for(base, fmt)
            m = build(ck, a.ckpt)
            m.load_state_dict(state)
            m = m.eval()
            lazy = Lazy(m, base) if fmt == "int8-lazy" else None
            if lazy:
                lazy.rebuild()
            p = prompts[0][:, :32]
            decode(m, "fp32", cpu, p, 4, lazy)  # warm up
            _, per = decode(m, "fp32", cpu, p, 24, lazy)
            res["formats"][fmt]["cpu_ms_per_piece_decode"] = round(per * 1000, 2)
            cpu_ids = ids_all.view(-1)[: 4 * 512]
            res["formats"][fmt]["cpu_pieces_per_s"] = round(
                throughput(m, "fp32", cpu, cpu_ids, lazy, reps=2, batch=4)
            )
            with (
                torch.no_grad()
            ):  # quality on the CPU too: the same weights, the CPU's own arithmetic
                res["formats"][fmt]["cpu_bpc_300k"] = round(
                    T.evaluate(m, val, 512, 300_000, batch=4, device=cpu), 5
                )
            print(a.name, "cpu", fmt, res["formats"][fmt], flush=True)
        res["cpu_threads"] = a.cpu_threads
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
