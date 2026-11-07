"""Wall clock and peak memory of the mixers at context T (CPU, one thread): ``python -m flm.bench_windowed``.

Forward + backward of one mixer layer on random input, batch 32, d 48, 2 heads, 6 rules, window WIN (default 32).
Peak memory is the process's max RSS growth for that call (MB). Wall clock is a median of repeats. These are
measurements of this implementation on this host, not of an optimised kernel: the bsolve and wdelta cores run a
Python loop over T (bsolve) or batched small solves (wdelta).
"""

import resource
import statistics
import time

import torch

import flm.windowed as W
from flm import models as M

torch.set_num_threads(1)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench(name, mod, x, reps):
    times = []
    before = rss_mb()
    for _ in range(reps):
        t0 = time.perf_counter()
        y = mod(x)
        y.square().mean().backward()
        times.append(time.perf_counter() - t0)
        mod.zero_grad(set_to_none=True)
    return statistics.median(times), rss_mb() - before


def main():
    torch.manual_seed(0)
    cfg = M.ModelConfig(vocab_size=98, d_model=48, n_heads=2, n_rules=6, decay="data", max_len=2048, mixer="fuzzydelta")
    w = W.window()
    print(f"window w = {w}, batch 32, d 48, 2 heads, 6 rules, 1 thread\n")
    print("| T | mixer | fwd+bwd ms (median) | peak RSS growth MB |")
    print("|---|---|---|---|")
    for T in (256, 512, 1024):
        x = torch.randn(32, T, 48)
        cells = [
            ("dense fuzzy delta", M.FuzzyDeltaMixer(cfg), T),
            ("bsolve (banded solve)", W.WindowFuzzyDeltaSolve(cfg), w),
            ("wdelta (restart window)", W.WindowFuzzyDeltaRestart(cfg), w),
        ]
        for name, mod, win in cells:
            if hasattr(mod, "win"):
                mod.win = win
            ms, mb = bench(name, mod, x, reps=3)
            print(f"| {T} | {name} | {ms * 1e3:.0f} | {mb:.0f} |")
        # the attention pair
        dense_att = M.GaussianTSKAttention(cfg)
        win_att = W.WindowGaussianAttention(cfg)
        win_att.win = w
        for name, mod in (("dense Gaussian attention", dense_att), (f"wgauss (window {w})", win_att)):
            ms, mb = bench(name, mod, x, reps=3)
            print(f"| {T} | {name} | {ms * 1e3:.0f} | {mb:.0f} |")


if __name__ == "__main__":
    main()
