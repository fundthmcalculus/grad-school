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
from flm.rules import load_model, make_codec  # noqa: E402

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


def piece(codec, i: int) -> str:
    return codec.bpe.decode([int(i)])


def explain(
    model, codec, ids: torch.Tensor, tok: int, per_layer: int = 3
) -> list[dict]:
    """Top rules by |contribution| to ``tok``'s logit at the last position, per FFN layer."""
    D = decompose(model, ids)
    s = D["s"][0, -1]
    rows = []
    for li, (g, pi) in enumerate(zip(D["gates"], D["pis"])):
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
    return rows


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
                    step["rules"] = explain(model, codec, window, tok)
                except AssertionError:
                    want_explain = False
                    step["rules"] = []
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
