"""Learning rates and weight decay per parameter group of the all-fuzzy model (`v64-groups`, PLAN_NEXTROUND.md track A2): the slow start of the fuzzy model (training-loss gap to the hybrid 1.34 nats at step 100, 0.35 at 1000)
may be the rule parameters (centres, widths, consequents) moving at the same rate as the weights.  30M characters, seeds 0-4, `e2e_tail`, lr 6e-2, warm-up 600, paired with `lr6_wu600` (1.1207).
Arms: `g_c03` / `g_c3` (centres and log-widths at 0.3x / 3x the learning rate), `g_s3` (consequents, offsets and gains at 3x), `g_wd0` (no weight decay on any rule parameter; the log-widths are decayed toward width 1 today).
``python -m flm.run_fuzzy_groups [arm ...]``; the training entry is the same module run with FUZZ_GROUPS_CHILD=1 (it patches the optimiser; train.py and models.py are untouched).  ``GROUPS_SMOKE=1`` shrinks the run to 0.2M characters.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

OUT = Path("outputs/v64-groups")
GRID = {  # name: (e2e arm, extra environment)
    "g_c03": ("e2e_tail", {"FUZZ_LRM": "centers|log_width=0.3"}),
    "g_c3": ("e2e_tail", {"FUZZ_LRM": "centers|log_width=3"}),
    "g_s3": ("e2e_tail", {"FUZZ_LRM": r"slope|conseq|offset|\.s$=3"}),
    "g_wd0": (
        "e2e_tail",
        {"FUZZ_WD0": r"centers|log_width|slope|conseq|offset|\.s$"},
    ),
}


def child():
    import torch

    import flm.fuzzy_tokens as FT
    import flm.run_allfuzzy as RA
    import flm.run_tiered as R
    import flm.train as T
    from flm.models import TinyLM

    lrm = [
        (re.compile(p), float(m))
        for p, m in (
            s.rsplit("=", 1) for s in os.environ.get("FUZZ_LRM", "").split(";") if s
        )
    ]
    wd0 = re.compile(os.environ["FUZZ_WD0"]) if os.environ.get("FUZZ_WD0") else None
    holder = {}
    real_init = RA._init

    def init(self, *a, **k):
        real_init(self, *a, **k)
        holder["model"] = self

    class GroupedAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            base = [g["lr"] for g in self.param_groups]
            for g in self.param_groups:
                g["lr"] = g["lr"] * g.get("lr_mult", 1.0)
            try:
                return super().step(closure)
            finally:
                for g, b in zip(self.param_groups, base):
                    g["lr"] = b

    def make(groups, lr, betas):
        names = {id(p): n for n, p in holder["model"].named_parameters()}
        out = []
        for g in groups:
            sub = {}
            for p in g["params"]:
                n = names[id(p)]
                mult = next((m for r, m in lrm if r.search(n)), 1.0)
                zero = wd0 is not None and wd0.search(n) is not None
                sub.setdefault((mult, zero), []).append((n, p))
            for (mult, zero), items in sub.items():
                out.append(
                    {
                        "params": [p for _, p in items],
                        "weight_decay": 0.0 if zero else g["weight_decay"],
                        "lr_mult": mult,
                    }
                )
                tag = f"lr x{mult:g}, wd {'0' if zero else g['weight_decay']}"
                print(
                    f"param group [{tag}]: {len(items)} tensors, {sum(p.numel() for _, p in items)} values, e.g. {items[0][0]}",
                    flush=True,
                )
        return GroupedAdamW(out, lr=lr, betas=betas)

    torch.optim.AdamW = lambda groups, lr, betas, **k: make(groups, lr, betas)
    TinyLM.__init__ = init
    TinyLM.embed = FT._embed if RA._emb else R._embed
    TinyLM.logits = FT._logits if RA._emb else R._logits
    TinyLM.n_params = R._n_params
    T.main(sys.argv[1:])


if __name__ == "__main__":
    if os.environ.get("FUZZ_GROUPS_CHILD") == "1":
        child()
        sys.exit(0)
    from flm.run_fuzzy_e2e import ARMS as E2E
    from flm.run_tribble import argv

    smoke = os.environ.get("GROUPS_SMOKE") == "1"
    out_dir = Path("outputs/smoke-groups") if smoke else OUT
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(GRID)
    for seed in range(1 if smoke else 5):
        for name in arms:
            out = out_dir / f"{name}_s{seed}.json"
            if out.exists():
                continue
            envx, over, mult = E2E[GRID[name][0]]
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = mult
            a[a.index("--tok") + 1] = "bpe3072"
            a[a.index("--warmup") + 1] = "600"
            if smoke:
                a[a.index("--chars") + 1] = "0.4e6"
            for k, v in over.items():
                if k in a:
                    a[a.index(k) + 1] = v
                else:
                    a += [k, v]
            env = {
                **os.environ,
                "TIER_NH": "384",
                "TIER_E": "12",
                "TIER_FREQ": "outputs/freq_bpe3072.npz",
                "PYTHONPATH": ".",
                "FUZZ_GROUPS_CHILD": "1",
                **envx,
                **GRID[name][1],
            }
            with open(out_dir / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_fuzzy_groups"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
