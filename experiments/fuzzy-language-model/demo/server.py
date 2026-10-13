"""Local inference demo for the fuzzy language models.

    .venv/bin/python demo/server.py [--port 8765]          then open http://127.0.0.1:8765

Serves demo/index.html and a small JSON API on localhost only (CPU, torch). Checkpoints are discovered under
outputs/ (``*.pt`` whose config uses the BPE1024 tokenizer). For each generated token the server can also return
the top candidates and the rules that pushed it: the exact direct-path decomposition of ``flm.rule_dist`` gives every
FFN rule a signed contribution ``gate * profile[token]`` to that token's logit.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from flm.rule_dist import decompose  # noqa: E402
from flm.rules import capture, load_model, make_codec  # noqa: E402
from flm.models import TSKFFN  # noqa: E402

torch.set_num_threads(4)
LOCK = threading.Lock()
CACHE: dict[str, tuple] = {}
MODELS: list[dict] = []
PREFERRED = [
    "v18-3b/D-d48-bpe1024-ctx512-r32_3000M_s0.pt",
    "v17-1b/D-d48-bpe1024-ctx512-r32_1000M_s0.pt",
    "v12-long512/D-d48-bpe1024-ctx512-r32_300M_s0.pt",
    "v9-cpu/d192-L4_30M_s0.pt",
    "v9-cpu/d96-L2_30M_s0.pt",
    "v5-final100k/D-d48-bpe1024_100M_s0.pt",
]


def models() -> list[dict]:
    out = []
    for p in sorted((ROOT / "outputs").glob("*/*.pt")):
        rel = str(p.relative_to(ROOT / "outputs"))
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not str(ck.get("tok", "")).startswith("bpe"):
            continue
        cfg = ck["config"]
        n = sum(int(np.prod(v.shape)) for v in ck["state"].values())
        out.append(
            {
                "id": rel,
                "tok": ck["tok"],
                "params": n,
                "ctx": cfg.get("max_len"),
                "d": cfg.get("d_model"),
                "layers": cfg.get("n_layers"),
            }
        )
    rank = {k: i for i, k in enumerate(PREFERRED)}
    out.sort(key=lambda m: (rank.get(m["id"], 99), m["id"]))
    return out


def get(mid: str):
    if mid not in CACHE:
        model, tok = load_model(ROOT / "outputs" / mid)
        CACHE[mid] = (model, make_codec(tok))
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


def explain(
    model, codec, ids: torch.Tensor, tok: int, per_layer: int = 3
) -> list[dict]:
    """Top rules by |contribution| to ``tok``'s logit at the last position, per FFN layer."""
    D = decompose(model, ids)
    s = D["s"][0, -1]
    rows = []
    gates = []
    for li, (g, pi) in enumerate(zip(D["gates"], D["pis"])):
        gates.append([round(v, 2) for v in g[0, -1].tolist()])
        gate = g[0, -1]  # (R,)
        contrib = gate * s * pi[:, tok]
        for r in contrib.abs().topk(per_layer).indices.tolist():
            sign = 1.0 if gate[r] >= 0 else -1.0
            raises = (pi[r] * sign).topk(5).indices.tolist()
            rows.append(
                {
                    "layer": li,
                    "rule": r,
                    "gate": round(gate[r].item(), 3),
                    "contribution": round(contrib[r].item(), 3),
                    "raises": [piece(codec, t) for t in raises],
                }
            )
    return rows, gates


def generate(req: dict) -> dict:
    model, codec = get(req["model"])
    ctx = model.cfg.max_len
    prompt = req.get("prompt") or "Once upon a time"
    n_new = max(1, min(int(req.get("max_tokens", 80)), 400))
    temp = max(0.05, float(req.get("temperature", 0.5)))
    top_k = max(1, int(req.get("top_k", 10)))
    want_explain = bool(req.get("explain", True))
    g = torch.Generator().manual_seed(int(req.get("seed", 0)))
    ids = torch.from_numpy(codec.bpe.encode(prompt).astype(np.int64))[None]
    steps = []
    hist = ids[0].tolist()
    seen = {tuple(hist[i : i + 3]) for i in range(len(hist) - 2)}
    with torch.no_grad():
        for _ in range(n_new):
            window = ids[:, -ctx:]
            logits = model(window)[0, -1]
            p_all = torch.softmax(logits, -1)
            lg = logits / temp
            kth = lg.topk(min(top_k, lg.shape[-1])).values[-1]
            lg = lg.masked_fill(lg < kth, float("-inf"))
            probs = torch.softmax(lg, -1)
            tok = int(torch.multinomial(probs, 1, generator=g))
            top5 = p_all.topk(5)
            step = {
                "text": piece(codec, tok),
                "p": round(p_all[tok].item(), 4),
                "top": [
                    {"text": piece(codec, int(i)), "p": round(float(v), 4)}
                    for v, i in zip(top5.values, top5.indices)
                ],
            }
            if want_explain:
                try:
                    step["rules"], step["gates"] = explain(model, codec, window, tok)
                except AssertionError:
                    want_explain = False
                    step["rules"] = []
            tri = tuple(hist[-2:] + [tok])
            step["rep"] = len(tri) == 3 and tri in seen
            seen.add(tri)
            hist.append(tok)
            steps.append(step)
            ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
    return {"prompt": prompt, "steps": steps, "explained": want_explain}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(
                200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
        elif (
            self.path.startswith("/structure")
            or self.path.startswith("/rules")
            or self.path.startswith("/rule?")
            or self.path.startswith("/mixrule")
        ):
            from urllib.parse import parse_qs, urlparse

            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                with LOCK:
                    if u.path == "/structure":
                        out = structure(q["model"])
                    elif u.path == "/rules":
                        out = rule_list(q["model"])
                    elif u.path == "/rule":
                        out = rule_detail(q["model"], int(q["layer"]), int(q["rule"]))
                    else:
                        out = mixer_rule(
                            q["model"], int(q["layer"]), int(q["head"]), int(q["rule"])
                        )
                self._send(200, json.dumps(out).encode(), "application/json")
            except Exception as e:
                self._send(
                    500,
                    json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(),
                    "application/json",
                )
        elif self.path == "/models":
            self._send(200, json.dumps(MODELS).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
            with LOCK:
                out = generate(req)
            self._send(200, json.dumps(out).encode(), "application/json")
        except Exception as e:  # report to the page instead of dropping the connection
            self._send(
                500,
                json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(),
                "application/json",
            )

    def log_message(self, *a):
        pass


def main():
    global MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    MODELS = (
        models()
    )  # scanned once: reading every checkpoint header takes a few seconds
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(
        f"FLM demo on http://127.0.0.1:{a.port}  ({len(MODELS)} checkpoints found)",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
