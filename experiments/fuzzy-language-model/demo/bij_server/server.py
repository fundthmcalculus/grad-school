"""Local demo of the invertible-meaning-layer models (`PLAN_BIJECTIVE.md`): greedy text, which fuzzy rules fired and how they steered each character, and how the embedding space changes on de-embedding.

    python demo/bij_server/server.py [--port 8770] [--ckpt-dir outputs/bij-demo]        # then open http://127.0.0.1:8770/

CPU only, standard library HTTP, torch for the model.  Models are the 10K-parameter character-level checkpoints ``<arm>_lr<lr>_s0.pt`` that ``flm/run_bijective.py`` saves (arms ``c``, ``z``, ``ident``, ``tied``),
discovered in ``--ckpt-dir`` with their result JSON next to them.  Output is always the highest-likelihood character (no sampling).
Endpoints: GET /api/models; POST /api/generate {model, prompt, n}; POST /api/explain {model, text, pos}; POST /api/roundtrip {model, text}.
Not public: binds 127.0.0.1 only.
"""

import argparse
import json
import math
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # experiments/fuzzy-language-model
sys.path.insert(0, str(ROOT))

import flm.run_bijective as RB  # noqa: E402  (reads BIJ and the rule counts from the environment at import: set per model below)
from flm.data import VOCAB_SIZE, decode, encode  # noqa: E402
from flm.models import ModelConfig, TinyLM  # noqa: E402

RULES = {
    "c": (32, 21),
    "z": (32, 21),
    "ident": (42, 29),
    "tied": (42, 29),
}  # feed-forward rules per layer, as in run_bijective's size-matched arms
NAMES = {
    "c": "F, routed back through the inverse (c)",
    "z": "F, likelihood in meaning space (z)",
    "ident": "identity control (distance likelihood)",
    "tied": "tied dot-product head",
}
TinyLM.__init__ = RB._new_init
TinyLM.embed = RB._embed
TinyLM.logits = RB._logits
TinyLM.n_params = RB._n_params


def label(i: int) -> str:
    s = decode(np.array([i]))
    return {"\n": "↵", " ": "␣"}.get(s, s)


class Engine:
    def __init__(self, path: Path, arm: str):
        self.arm, self.path = arm, path
        RB.BIJ, RB.FFN_R, RB.FFN_R0 = arm, *RULES[arm]
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = TinyLM(ModelConfig(**ck["config"]))
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.ctx = ck["config"]["max_len"]
        info = (
            json.loads(path.with_suffix(".json").read_text())
            if path.with_suffix(".json").exists()
            else {}
        )
        self.val = info.get("val_bpc")
        self.params = info.get(
            "params_total", sum(p.numel() for p in self.model.parameters())
        )
        self.id = f"{arm}"
        self.stage_cache = None

    def use(self):
        RB.BIJ = self.arm

    # ------------------------------------------------------------------ forward with hooks
    @staticmethod
    def rule_s(mod, x):
        """Strength of every feed-forward rule at every position: membership firing times the first-order slope (``FuzzyFFNRaw.forward`` without the final matrix product)."""
        u = mod.antecedent_input(x)
        a = torch.exp(-2.0 * mod.log_width)
        ca = mod.centers * a
        quad = (u * u) @ a.T - 2.0 * (u @ ca.T) + (mod.centers * ca).sum(-1)
        w = torch.exp(-0.5 * mod.scale * quad)
        g = mod.slope(x)
        return w * g, w, g

    @torch.no_grad()
    def run(self, ids, pos=None, ablate=None):
        """Logits for ``ids`` (1, T); returns (logits, per-block (s, w, g, ffn_in), final normed hidden h).  ``ablate=(block, rule)``: remove that rule's output at ``pos`` before the rest of the model sees it."""
        self.use()
        caps, hs, handles = [], [], []

        def make(l):
            def hook(mod, inp, out):
                s, w, g = self.rule_s(mod, inp[0])
                caps.append((s[0], w[0], g[0], inp[0][0]))
                if ablate is not None and ablate[0] == l:
                    out = out.clone()
                    out[:, pos] -= s[0, pos, ablate[1], None] * mod.conseq[ablate[1]]
                    return out

            return hook

        for l, blk in enumerate(self.model.blocks):
            handles.append(blk.ffn.register_forward_hook(make(l)))
        handles.append(
            self.model.norm.register_forward_hook(lambda m, i, o: hs.append(o[0]))
        )
        try:
            logits = self.model(ids)
        finally:
            for h in handles:
                h.remove()
        return logits[0], caps, hs[0]

    # ------------------------------------------------------------------ API
    def ids_of(self, text):
        return torch.as_tensor(np.asarray(encode(text), dtype=np.int64))

    @torch.no_grad()
    def generate(self, prompt, n):
        ids = self.ids_of(prompt)
        if len(ids) == 0:
            ids = self.ids_of("\n")
        out, probs, ents = [], [], []
        for _ in range(n):
            window = ids[-self.ctx :][None]
            logits = self.run(window)[0][-1]
            p = torch.softmax(logits, -1)
            nxt = int(p.argmax())
            out.append(nxt), probs.append(float(p[nxt])), ents.append(
                float(-(p * torch.log2(p.clamp_min(1e-12))).sum())
            )
            ids = torch.cat([ids, torch.tensor([nxt])])
        return dict(
            prompt=prompt, text=decode(np.array(out)), probs=probs, entropy=ents
        )

    @torch.no_grad()
    def explain(self, text, pos, top_rules=6):
        """Explain the prediction made after ``text[:pos + 1]``: the top candidates, the rules that fired in every block with their steering effect, and the embedding-space picture."""
        ids = self.ids_of(text)[: pos + 1][-self.ctx :][None]
        T = ids.shape[1]
        logits, caps, h = self.run(ids)
        last = logits[-1]
        logp = torch.log_softmax(last, -1)
        chosen = int(logp.argmax())
        top = torch.topk(logp, 6)
        res = dict(
            context=decode(ids[0].numpy()),
            chosen=label(chosen),
            candidates=[
                dict(ch=label(int(i)), p=float(math.exp(v)))
                for v, i in zip(top.values, top.indices)
            ],
            blocks=[],
        )
        for l, (s, w, g, xin) in enumerate(caps):
            ffn = self.model.blocks[l].ffn
            s_last = s[-1]
            contrib = (
                s_last[:, None] * ffn.conseq
            )  # (R, d): what each rule adds to the residual stream here
            size = contrib.norm(dim=-1)
            order = torch.argsort(-size)[:top_rules]
            rows = []
            for r in order.tolist():
                lg = self.run(ids, pos=T - 1, ablate=(l, r))[0][-1]
                lp = torch.log_softmax(lg, -1)
                eff = float(logp[chosen] - lp[chosen])
                delta = (
                    logp.exp() - lp.exp()
                )  # probability change of every character caused by this rule being there (the rule present minus removed)
                boost = torch.topk(delta, 3)
                lower = torch.topk(-delta, 2)
                rows.append(
                    dict(
                        rule=r,
                        strength=float(s_last[r]),
                        firing=float(w[-1, r]),
                        slope=float(g[-1, r]),
                        size=float(size[r]),
                        effect=eff,
                        boosts=[
                            dict(ch=label(int(i)), d=float(v) * 100)
                            for v, i in zip(boost.values, boost.indices)
                        ],
                        lowers=[
                            dict(ch=label(int(i)), d=float(-v) * 100)
                            for v, i in zip(lower.values, lower.indices)
                        ],
                    )
                )
            res["blocks"].append(
                dict(
                    block=l,
                    rules=int(s.shape[1]),
                    active=int((w[-1] > 0.05).sum()),
                    total_size=float(size.sum()),
                    resid=float(xin[-1].norm()),
                    rows=rows,
                )
            )
        res["space"] = self.space(ids[0], h[-1], chosen)
        return res

    # ------------------------------------------------------------------ embedding space
    @torch.no_grad()
    def space(self, ids, h_last, chosen):
        m = self.model
        if self.arm == "tied":
            return dict(
                kind="none",
                note="The tied head has no separate embedding space to show: scores are dot products with the embedding table.",
            )
        self.use()
        C = m.codebook.detach()
        stages = [C]
        if self.arm in ("c", "z"):
            x = C
            for st in m.F.steps:
                x = st(x)[0]
                stages.append(x)
        Z = stages[-1]
        stack = torch.cat([C, Z], 0).numpy()
        mu = stack.mean(0)
        _, _, vt = np.linalg.svd(stack - mu, full_matrices=False)
        P = lambda t: ((t.numpy() - mu) @ vt[:2].T).round(4).tolist()  # noqa: E731
        out = dict(
            kind=self.arm,
            labels=[label(i) for i in range(C.shape[0])],
            C=P(C),
            Z=P(Z),
            tau=float(torch.exp(m.log_tau)),
        )
        if len(stages) > 2:
            out["mid"] = P(stages[1])
        out["context"] = [int(i) for i in ids[-10:]]
        h = h_last
        if self.arm == "c":
            path = [h]
            y = h
            for st in reversed(m.F.steps):
                y = st.inverse(y)
                path.append(y)
            u = path[-1]
            out["path"] = P(torch.stack(path))
            target = C
            d2 = ((u - C) ** 2).sum(-1)
            out["note"] = (
                "The output vector h lives in meaning space (next to the dark dots, F(C)); the inverse coupling steps carry it back to codebook space (the light dots, C), where the nearest codeword is the output."
            )
        else:
            out["path"] = P(h[None])
            target = Z if self.arm == "z" else C
            d2 = ((h - target) ** 2).sum(-1)
            out["note"] = (
                "The output vector h is compared with the embedded codewords F(C) directly: no inverse step."
                if self.arm == "z"
                else "Identity control: no invertible layer; h is compared with the codebook C directly."
            )
        near = torch.topk(-d2, 5)
        tau2 = float(torch.exp(2 * m.log_tau))
        out["nearest"] = [
            dict(ch=label(int(i)), d2=float(-v), logit=float(v / (2 * tau2)))
            for v, i in zip(near.values, near.indices)
        ]
        out["chosen"] = label(chosen)
        return out

    @torch.no_grad()
    def roundtrip(self, text):
        if self.arm not in ("c", "z"):
            return dict(ok=None, note="This model has no invertible layer.")
        self.use()
        raw = encode(text)
        ids = torch.as_tensor(np.asarray(raw, dtype=np.int64))
        res = {}
        for name, dt in (("float32", torch.float32), ("float64", torch.float64)):
            m = self.model.to(dt)
            C = m.codebook
            z = m.F(C[ids])[0]
            u = m.F.inverse(z)
            back = torch.cdist(u, C).argmin(-1)
            res[name] = dict(
                text=decode(back.numpy()),
                exact=bool(torch.equal(back, ids)),
                max_err=float((u - C[ids]).abs().max()) if len(ids) else 0.0,
                moved=float((z - C[ids]).norm(dim=-1).mean()) if len(ids) else 0.0,
            )
        self.model.to(torch.float32)
        res["symbols"] = len(ids)
        res["lossy_input"] = decode(np.asarray(raw)) != text
        res["via_symbols"] = decode(np.asarray(raw))
        return res


ENGINES: dict[str, Engine] = {}
LOCK = (
    threading.Lock()
)  # the models share module-level state (RB.BIJ): one request at a time


def discover(d: Path):
    for p in sorted(d.glob("*_lr*_s0.pt")):
        m = re.match(r"(tied|ident|z|c)_lr", p.name)
        if m and m[1] not in ENGINES:
            ENGINES[m[1]] = Engine(p, m[1])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(
                200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
        elif self.path == "/api/models":
            self.send(
                200,
                [
                    dict(id=k, name=NAMES[k], params=e.params, val=e.val, ctx=e.ctx)
                    for k, e in ENGINES.items()
                ],
            )
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            )
            eng = ENGINES[req["model"]]
            with LOCK:
                if self.path == "/api/generate":
                    out = eng.generate(
                        req.get("prompt", ""), max(1, min(int(req.get("n", 120)), 400))
                    )
                elif self.path == "/api/explain":
                    out = eng.explain(req["text"], int(req["pos"]))
                elif self.path == "/api/roundtrip":
                    out = eng.roundtrip(req.get("text", ""))
                else:
                    return self.send(404, {"error": "not found"})
            self.send(200, out)
        except Exception as e:  # report to the page, keep serving
            self.send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--ckpt-dir", default="outputs/bij-demo")
    a = ap.parse_args()
    torch.set_num_threads(2)
    discover(Path(a.ckpt_dir) if Path(a.ckpt_dir).is_absolute() else ROOT / a.ckpt_dir)
    print("models:", {k: e.path.name for k, e in ENGINES.items()}, flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"http://127.0.0.1:{a.port}/", flush=True)
    srv.serve_forever()
