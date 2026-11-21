"""Rule views of a trained FLM for the in-browser demo's exporter (``demo/export_web.py``): the structure card, the most used
rules, and every FFN rule's membership functions and data histograms over 48 validation windows.

Models are injected into ``CACHE`` by the exporter (``CACHE[model_id] = (model, codec)``); nothing here serves anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from flm.models import TSKFFN  # noqa: E402
from flm.rule_dist import decompose  # noqa: E402
from flm.rules import capture  # noqa: E402

CACHE: dict[str, tuple] = {}


def get(mid: str):
    return CACHE[mid]


STATS: dict[str, dict] = {}


def stats(mid: str) -> dict:
    """Per-model statistics over 48 validation windows, cached: FFN inputs, firings and the exact decomposition."""
    if mid not in STATS:
        model, codec = get(mid)
        idx = codec.windows(48, 257)
        with torch.no_grad():
            rec = capture(model, idx[:, :-1])
            D = decompose(model, idx[:, :-1])
        STATS[mid] = {"idx": idx, "rec": rec, "D": D}
    return STATS[mid]


def structure(mid: str) -> dict:
    model, codec = get(mid)
    cfg = model.cfg
    layers = []
    for li, blk in enumerate(model.blocks):
        mix = blk.mix
        ffn = blk.ffn
        layers.append(
            {
                "layer": li,
                "conv": getattr(blk.conv, "width", None),
                "mixer": type(mix).__name__,
                "heads": getattr(mix, "H", None),
                "mixer_rules_per_head": (
                    cfg.n_rules if hasattr(mix, "centers") else None
                ),
                "head_dim": getattr(mix, "dh", None),
                "ffn": type(ffn).__name__,
                "ffn_rules": getattr(ffn, "R", None),
                "ffn_antecedent_dim": getattr(ffn, "da", None),
                "ffn_order": getattr(ffn, "order", None),
                "ffn_norm": getattr(ffn, "norm", None),
            }
        )
    return {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "layers": layers,
        "params": model.n_params(),
    }


def rule_list(mid: str, top: int = 24) -> dict:
    model, codec = get(mid)
    S = stats(mid)
    D, idx = S["D"], S["idx"]
    out = []
    for li, blk in enumerate(model.blocks):
        ffn = blk.ffn
        if not isinstance(ffn, TSKFFN):
            continue
        g, pi = D["gates"][li], D["pis"][li]
        score = g.abs() * (ffn.conseq * model.norm.weight).norm(dim=-1)
        use = torch.bincount(score.argmax(-1).reshape(-1), minlength=ffn.R)
        rows = []
        for r in use.topk(min(top, ffn.R)).indices.tolist():
            sign = 1.0 if g[..., r].mean() >= 0 else -1.0
            rows.append(
                {
                    "rule": r,
                    "wins_pct": round(100 * use[r].item() / score[..., 0].numel(), 2),
                    "raises": [
                        piece(codec, t) for t in (pi[r] * sign).topk(6).indices.tolist()
                    ],
                }
            )
        out.append({"layer": li, "rules": rows})
    return {"layers": out}


def hist(
    values: torch.Tensor,
    lo: float,
    hi: float,
    bins: int = 24,
    weights: torch.Tensor | None = None,
) -> list[float]:
    """Histogram scaled to a maximum of 1; with ``weights`` each value counts by its weight."""
    pos = (
        ((values.clamp(lo, hi) - lo) / max(hi - lo, 1e-9) * bins)
        .long()
        .clamp(0, bins - 1)
    )
    w = torch.ones_like(values) if weights is None else weights
    h = torch.zeros(bins).index_add_(0, pos, w.to(torch.float32))
    return [round(x, 4) for x in (h / h.max().clamp_min(1e-9)).tolist()]


def rule_detail(mid: str, layer: int, rule: int) -> dict:
    """Membership functions of one FFN rule: per antecedent dimension a Gaussian (centre, effective width) and the
    distribution of the data along that dimension, overall and weighted by this rule's firing.
    """
    model, codec = get(mid)
    S = stats(mid)
    ffn = model.blocks[layer].ffn
    x = S["rec"][("ffn_x", layer)]
    with torch.no_grad():
        u = ffn.ante(x)  # (N,T,da)
        w = S["rec"][("ffn", layer)][..., rule]  # firing of this rule
        width = torch.exp(ffn.log_width).expand_as(ffn.centers)[rule]
        sig = (
            width / ffn.exp_scale**0.5
        )  # the product of these Gaussians over dimensions is the firing
        cen = ffn.centers[rule]
    dims = []
    for j in range(ffn.da):
        col = u[..., j].reshape(-1)
        lo, hi = float(col.quantile(0.01)), float(col.quantile(0.99))
        lo, hi = min(lo, cen[j].item() - 2 * sig[j].item()), max(
            hi, cen[j].item() + 2 * sig[j].item()
        )
        dims.append(
            {
                "dim": j,
                "center": round(cen[j].item(), 3),
                "sigma": round(sig[j].item(), 3),
                "data_sd": round(col.std().item(), 3),
                "selectivity": round(col.std().item() / max(sig[j].item(), 1e-6), 3),
                "lo": round(lo, 3),
                "hi": round(hi, 3),
                "all": hist(col, lo, hi),
                "fires": hist(col, lo, hi, weights=w.reshape(-1)),
            }
        )
    flat = w.reshape(-1)
    top = flat.topk(6).indices.tolist()
    T = w.shape[1]
    idx = S["idx"]
    ctxs = []
    for f in top:
        b, t = divmod(f, T)
        left = "".join(piece(codec, int(i)) for i in idx[b, max(0, t - 9) : t + 1])
        ctxs.append(
            {
                "firing": round(flat[f].item(), 3),
                "context": left,
                "next": piece(codec, int(idx[b, t + 1])),
            }
        )
    return {
        "layer": layer,
        "rule": rule,
        "firing_mean": round(flat.mean().item(), 4),
        "firing_p99": round(float(flat.quantile(0.99)), 3),
        "dims": sorted(dims, key=lambda d: -d["selectivity"]),
        "contexts": ctxs,
    }


def mixer_rule(mid: str, layer: int, head: int, rule: int) -> dict:
    """Centre and width per head dimension of one memory rule of the fuzzy recurrent mixer."""
    model, _ = get(mid)
    mix = model.blocks[layer].mix
    cen = mix.centers[head, rule]
    wd = torch.exp(mix.log_width[head, rule])
    return {
        "layer": layer,
        "head": head,
        "rule": rule,
        "center": [round(c, 3) for c in cen.tolist()],
        "width": [round(c, 3) for c in wd.tolist()],
    }


def piece(codec, i: int) -> str:
    return codec.bpe.decode([int(i)])
