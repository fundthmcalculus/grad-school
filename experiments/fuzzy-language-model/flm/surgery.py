"""Rule surgery on a trained all-fuzzy model ("brain surgery", `PLAN_SURGERY.md`): at a position where the model chose the wrong next piece, edit the consequent of the feed-forward rules that fired so that the piece the user names becomes the prediction, and measure what else changed.

``python -m flm.surgery collect``   run the model over validation windows and cache, for every position, the residual vector before the output gain controller and the strengths of the last block's feed-forward rules (CPU, 3 processes).
``python -m flm.surgery run``       the experiments on the cache (numpy, no model forward pass needed); writes ``report/surgery.json`` and ``report/surgery.md``.

Why the last block's feed-forward bank: the output scores are an exact function of the residual vector x before the final gain controller, x = x_mid + sum_r s_r c_r, where s_r is rule r's strength (firing x first-order consequent weight) and c_r its consequent vector.  Nothing after that block reads the bank's output (the memory state and the next token's antecedent read x_mid), so editing c_r changes the scores of a position only through s_r: an edit is *local* in the sense of the rule's own firing, and it cannot disturb later positions except through the rules' strengths there.  This is the property a dense model does not have.
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from allfuzzy.allfuzzy_lm import EPS, AllFuzzyLM, softmax

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data" / "tinystories"
WEIGHTS = (
    ROOT / "allfuzzy" / "F7re_300M_s0.npz"
)  # the demo's flm-allfuzzy-f7 (the shipped 100K model)
TOKENIZER = ROOT / "allfuzzy" / "bpe6144.json"
CACHE = ROOT / "outputs" / "surgery_cache.npz"
CTX = 512


class TracingLM(AllFuzzyLM):
    """AllFuzzyLM.step, additionally returning the residual before the final gain controller and the last block's rule strengths."""

    def step_trace(self, tok_id):
        W, H, dh = self.W, self.H, self.dh
        x = self.E[tok_id].copy()
        strength = None
        for l in range(self.L):
            b = f"blocks.{l}."
            h, _, _ = self.gain_controller(x, b + "n1.")
            qkv = self.gated_map(h, b + "mix.qkv.").reshape(3, H, dh)
            alpha_logit, _ = self.rule_gate(h, b + "mix.gate.")
            beta_logit, _ = self.rule_gate(h, b + "mix.beta.")
            y = np.zeros((H, dh))
            for hd in range(H):
                c, lw = W[b + "mix.centers"][hd], W[b + "mix.log_width"][hd]
                fq = self.memory_partition(qkv[0, hd], c, lw)
                fk = self.memory_partition(qkv[1, hd], c, lw)
                S = self.S[l][hd] * (1.0 / (1.0 + np.exp(-alpha_logit[hd])))
                S = S + (1.0 / (1.0 + np.exp(-beta_logit[hd]))) * np.outer(
                    qkv[2, hd] - S @ fk, fk
                )
                self.S[l][hd] = S
                y[hd] = S @ fq
            x = x + self.gated_map(y.reshape(-1), b + "mix.out.")
            n2, _, _ = self.gain_controller(x, b + "n2.")
            out, strength = self.feedforward(n2, self.prev_n2[l], l)
            self.prev_n2[l] = n2
            x = x + out
        return (
            x,
            strength,
        )  # x: the residual before norm.; strength: the last block's rule strengths


def final_logits(m: AllFuzzyLM, X: np.ndarray) -> np.ndarray:
    """Output scores for a batch of residuals X (n, d): the final gain controller (one system per row) and the tied table."""
    W = m.W
    z = 0.5 * np.log((X**2).mean(-1) + EPS)  # (n,)
    w = softmax(
        -0.5 * ((z[:, None] - W["norm.centers"]) / np.exp(W["norm.log_width"])) ** 2
    )
    gain = (
        w * (W["norm.offset"] + W["norm.slope"] * (z[:, None] - W["norm.centers"]))
    ).sum(-1)
    HN = X * np.exp(gain)[:, None] * W["norm.weight"]
    return HN @ m.E.T


def _work(args):
    lo, hi, tokens = args
    m = TracingLM(WEIGHTS, TOKENIZER)
    nR = m.W[f"blocks.{m.L - 1}.ffn.conseq"].shape[0]
    X = np.zeros((hi - lo, CTX, m.d), np.float32)
    S = np.zeros((hi - lo, CTX, nR), np.float32)
    for k, w in enumerate(range(lo, hi)):
        m.reset()
        seq = tokens[w * (CTX + 1) : (w + 1) * (CTX + 1)]
        for t in range(CTX):
            X[k, t], S[k, t] = m.step_trace(int(seq[t]))
    return X, S


def collect(procs=3):
    tokens = np.fromfile(DATA / "valid_bpe6144.bin", dtype=np.uint16)
    lens = np.load(DATA / "lens_bpe6144.npy")
    cum = np.cumsum(lens[tokens[:1_000_000]].astype(np.int64))
    n_win = (int(np.searchsorted(cum, 1_000_000)) + 1) // (
        CTX + 1
    )  # the windows `flm.train.evaluate` scores
    cuts = np.linspace(0, n_win, procs + 1).astype(int)
    t0 = time.time()
    with Pool(procs) as p:
        parts = p.map(_work, [(cuts[i], cuts[i + 1], tokens) for i in range(procs)])
    X = np.concatenate([a for a, _ in parts])
    S = np.concatenate([b for _, b in parts])
    seq = tokens[: n_win * (CTX + 1)].reshape(n_win, CTX + 1).astype(np.int64)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, X=X, S=S, seq=seq, n_win=n_win)
    print(
        f"cached {n_win} windows x {CTX} positions in {time.time() - t0:.0f}s -> {CACHE}"
    )


if __name__ == "__main__":
    (
        {"collect": collect}[sys.argv[1]]()
        if len(sys.argv) > 1 and sys.argv[1] == "collect"
        else None
    )
