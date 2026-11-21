"""CUDA-graphed forward and backward for the all-fuzzy model (PLAN_NEXTROUND.md track A1): the fuzzy step issues about 2,700 small kernels (the hybrid 1,400), so it is launch-bound; capturing forward and backward as one graph removes the launch overhead.
The optimiser, clipping and evaluation stay eager; evaluation runs in eval mode, where the original forward is used.  ``train.py`` and ``models.py`` are untouched.

``python -m flm.run_fuzzy_graph <train args>`` trains with the graphed forward (env as for ``flm.run_allfuzzy``); ``GRAPH=0`` is the eager control.
``python -m flm.run_fuzzy_graph --check [hybrid|fuzzy]`` runs N steps eager and graphed on identical batches and prints per-step loss differences and seconds per step.
"""

import os
import sys
import time

if (
    "--check" in sys.argv
):  # the architecture environment must be set before flm.run_allfuzzy is imported
    _arm = (sys.argv[sys.argv.index("--check") + 1 :] or ["fuzzy"])[0]
    os.environ.update(
        TIER_NH="384", TIER_E="12", TIER_FREQ="outputs/freq_bpe3072.npz", PYTHONPATH="."
    )
    if _arm == "fuzzy":
        from flm.run_fuzzy_e2e import ARMS as _E2E

        os.environ.update(_E2E["e2e_tail"][0])

import torch

import flm.fuzzy_tokens as FT
import flm.run_allfuzzy as RA
import flm.run_tiered as R
from flm.models import TinyLM


def graphed_forward_patch():
    real_forward = TinyLM.forward

    def forward(self, idx, *a, **k):
        if (
            not self.training
            or not torch.is_grad_enabled()
            or os.environ.get("GRAPH", "1") == "0"
            or a
            or k
        ):
            return real_forward(self, idx, *a, **k)
        g = getattr(self, "_graphed", None)
        if g is None or g[0] != tuple(idx.shape):
            # warm-up on a side stream, then capture (make_graphed_callables does both)
            g = (
                tuple(idx.shape),
                torch.cuda.make_graphed_callables(
                    _Fn(self, real_forward), (idx.detach().clone(),)
                ),
            )
            self._graphed = g
        return g[1](idx)

    TinyLM.forward = forward


class _Fn(torch.nn.Module):
    """make_graphed_callables wants a module whose parameters it can see."""

    def __init__(self, model, real_forward):
        super().__init__()
        self.model, self.real_forward = model, real_forward

    def forward(self, idx):
        return self.real_forward(self.model, idx)


def patch_arch():
    TinyLM.__init__ = RA._init
    TinyLM.embed = FT._embed if RA._emb else R._embed
    TinyLM.logits = FT._logits if RA._emb else R._logits
    TinyLM.n_params = R._n_params


def check(arm, steps=60):
    import torch.nn.functional as F
    import flm.train as T
    from flm.run_tribble import argv

    a = argv("random", 0)
    a[a.index("--tok") + 1] = "bpe3072"
    if arm == "fuzzy":
        from flm.run_fuzzy_e2e import ARMS as E2E

        _, over, mult = E2E["e2e_tail"]
        a[a.index("--ffn-mult") + 1] = mult
        for k, v in over.items():
            a[a.index(k) + 1] = v
    args = T.get_args(a)
    args.vocab_size = 3072
    patch_arch()
    gen = torch.Generator().manual_seed(0)
    batches = [
        torch.randint(0, 3072, (args.batch, args.ctx + 1), generator=gen).cuda()
        for _ in range(steps)
    ]

    def run(graphed):
        torch.manual_seed(0)
        m = TinyLM(T.make_config(args)).cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        if graphed:
            os.environ["GRAPH"] = "1"
        else:
            os.environ["GRAPH"] = "0"
        losses, t = [], None
        for i, x in enumerate(batches):
            if i == 10:
                torch.cuda.synchronize()
                t = time.perf_counter()
            lg = m(x[:, :-1])
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        torch.cuda.synchronize()
        return losses, (time.perf_counter() - t) / (steps - 10)

    le, te = run(False)
    lg_, tg = run(True)
    diff = max(abs(x - y) for x, y in zip(le, lg_))
    print(
        f"{arm}: max |loss diff| over {steps} steps {diff:.2e}; eager {te*1000:.1f} ms/step, graphed {tg*1000:.1f} ms/step ({te/tg:.2f}x); loss {le[0]:.4f} -> {le[-1]:.4f} (eager) / {lg_[-1]:.4f} (graphed)"
    )


if __name__ == "__main__":
    if "--check" in sys.argv:
        arm = (sys.argv[sys.argv.index("--check") + 1 :] or ["fuzzy"])[0]
        check(arm)
    else:
        import flm.train as T

        patch_arch()
        graphed_forward_patch()
        T.main(sys.argv[1:])
