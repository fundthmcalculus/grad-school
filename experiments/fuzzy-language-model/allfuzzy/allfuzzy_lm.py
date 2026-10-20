"""The all-fuzzy language model, end to end, in plain numpy: text in, text out, every rule visible.

    from allfuzzy_lm import generate, write_rulebook
    print(generate("Once upon a time"))                 # the generated continuation, as a string
    write_rulebook("rulebook.txt")                       # every rule of every fuzzy system, in words and numbers

    python allfuzzy_lm.py --text "Once upon a time" --trace     # generation plus the rules that fired for each word piece
    python allfuzzy_lm.py --rules rulebook.txt                   # write the whole rule base

Needs numpy and the HF ``tokenizers`` package only (weights: e2e_tail_300M_s0.npz, tokenizer: bpe3072.json, both next to this file).

WHAT THIS MODEL IS.  A 99,019-parameter language model (BPE3072, two blocks, width 48) in which every block component is a fuzzy inference
system.  Each system below follows the same four steps, and the code marks them:
    FUZZIFY    turn the input into degrees of membership (Gaussian membership functions, one per rule and input dimension)
    FIRE       a rule's firing strength is the PRODUCT T-NORM of its memberships (the product of the Gaussians, written as exp of a sum)
    AGGREGATE  combine the rules (normalise the firings into a fuzzy partition, or keep raw firings)
    DEFUZZIFY  a weighted average (or sum) of the rules' consequents
The systems, per block: (1) two gain controllers instead of RMSNorm, (2) rule-gated low-rank maps instead of the q/k/v and output projections,
(3) two six-rule systems instead of the sigmoid gates, (4) a fuzzy associative memory (32 Gaussian rules per head, delta-rule consequents),
(5) a feed-forward bank of Gaussian rules on the raw input with first-order consequents (layer 0 reads the last two tokens: it replaces the
convolution).  Plus the output side: a fuzzy gain controller, a crisp table for the 384 most frequent word pieces and fuzzy sets over 12
prototypes for the rest.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
EPS = 1.1920929e-07  # float32 eps, as torch's RMS-style norms use


def softmax(v, axis=-1):
    e = np.exp(v - v.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def firing_logits(u, centers, log_width, scale):
    """FUZZIFY + FIRE.  Membership of input dimension j in rule r is a Gaussian exp(-0.5 ((u_j - c_rj) / s_rj)^2); the rule fires with the PRODUCT of its
    memberships, i.e. exp(-0.5 * sum_j ((u_j - c_rj) / s_rj)^2).  ``scale`` (1 / number of inputs) keeps the product from collapsing as dimensions grow.
    Returns the log of the firing strength of every rule."""
    a = np.exp(-2.0 * log_width)  # 1 / s^2
    return -0.5 * scale * (a * (u[None, :] - centers) ** 2).sum(axis=-1)


class AllFuzzyLM:
    def __init__(self, weights=HERE / "e2e_tail_300M_s0.npz", tokenizer=HERE / "bpe3072.json"):
        z = np.load(weights)
        self.W = {k: z[k] for k in z.files}
        self.tok = Tokenizer.from_file(str(tokenizer))
        self.d, self.L, self.H = int(z["cfg_d_model"]), int(z["cfg_n_layers"]), int(z["cfg_n_heads"])
        self.V, self.max_len = int(z["cfg_vocab_size"]), int(z["cfg_max_len"])
        self.dh = self.d // self.H
        self.R_mem = int(z["cfg_n_rules"])
        # token table: 'head' word pieces have a crisp row, 'tail' word pieces are fuzzy sets over 12 prototypes
        self.head_ids, self.tail_ids = z["head_ids"], z["tail_ids"]
        self.pos = np.zeros(self.V, dtype=int)
        self.is_head = np.zeros(self.V, dtype=bool)
        self.pos[self.head_ids], self.pos[self.tail_ids] = np.arange(len(self.head_ids)), np.arange(len(self.tail_ids))
        self.is_head[self.head_ids] = True
        self.tail_membership = softmax(self.W["emb_t.weight"])  # (tail pieces, 12): each piece's degrees of membership in the prototypes
        proto = self.W["tier_up.weight"]  # (48, 12): column k is prototype vector k
        self.E = np.zeros((self.V, self.d))  # the effective embedding of every piece (used for input and, tied, for output scores)
        self.E[self.head_ids] = self.W["emb_h.weight"]
        self.E[self.tail_ids] = self.tail_membership @ proto.T
        self.reset()

    # ------------------------------------------------------------------ the fuzzy systems
    def gain_controller(self, x, p):
        """Replaces RMSNorm.  One input variable, z = log rms(x), five Gaussian membership functions on it, first-order consequents
        b_r + k_r (z - c_r) giving a log-gain; defuzzified by the weighted average; output = x * exp(gain) * weight."""
        W = self.W
        z = 0.5 * np.log((x**2).mean() + EPS)
        w = softmax(-0.5 * ((z - W[p + "centers"]) / np.exp(W[p + "log_width"])) ** 2)  # FUZZIFY, FIRE, AGGREGATE
        gain = (w * (W[p + "offset"] + W[p + "slope"] * (z - W[p + "centers"]))).sum()  # DEFUZZIFY
        return x * np.exp(gain) * W[p + "weight"], w, gain

    def gated_map(self, x, p):
        """Replaces a linear projection.  u = B x (a small basis); eight Gaussian rules over u fire; each rule rescales the basis directions by its
        consequent s_r; the defuzzified rescaling is applied and mapped out by A:  y = A ((sum_r w_r s_r) * u)."""
        W = self.W
        u = W[p + "B.weight"] @ x
        w = softmax(firing_logits(u, W[p + "centers"], W[p + "log_width"], 1.0 / len(u)))  # FUZZIFY, FIRE, AGGREGATE
        return W[p + "A.weight"] @ ((w @ W[p + "s"]) * u)  # DEFUZZIFY

    def rule_gate(self, x, p):
        """Replaces a sigmoid gate.  Six Gaussian rules over the whole 48-number input; each rule's consequent is one logit per head; the
        weighted average of the consequents is the gate's logit (the caller squashes it to (0, 1))."""
        W = self.W
        w = softmax(firing_logits(x, W[p + "centers"], W[p + "log_width"], 1.0 / len(x)))
        return w @ W[p + "conseq"], w

    def memory_partition(self, u, centers, log_width):
        """The fuzzy associative memory's antecedents for one head: 32 Gaussian rules over a 6-number key/query; normalised firings = a fuzzy partition."""
        return softmax(firing_logits(u, centers, log_width, 1.0 / len(u)))

    def feedforward(self, n2, prev_n2, layer):
        """The feed-forward rule bank.  Antecedent: Gaussian memberships on the RAW normalised input (layer 0: on the current AND previous token's
        input side by side, which replaces the convolution), product t-norm, raw firing strengths.  First-order consequent: rule r pushes
        conseq_r with strength firing_r * (slope_r . n2 + b_r).  Output = sum of the pushes.  Returns the output and every rule's strength."""
        W, p = self.W, f"blocks.{layer}.ffn."
        u = np.concatenate([n2, prev_n2]) if W[p + "centers"].shape[1] == 2 * self.d else n2
        firing = np.exp(firing_logits(u, W[p + "centers"], W[p + "log_width"], 1.0 / len(u)))  # FUZZIFY, FIRE
        strength = firing * (W[p + "slope.weight"] @ n2 + W[p + "slope.bias"])  # first-order consequent weight
        return strength @ W[p + "conseq"], strength  # DEFUZZIFY (sum)

    # ------------------------------------------------------------------ one word piece in, next-piece scores out
    def reset(self):
        self.S = [np.zeros((self.H, self.dh, self.R_mem)) for _ in range(self.L)]  # the memory of each head: one column per rule
        self.prev_n2 = [np.zeros(self.d) for _ in range(self.L)]
        self.n_seen = 0

    def step(self, tok_id):
        W, d, H, dh = self.W, self.d, self.H, self.dh
        x = self.E[tok_id].copy()  # the piece's embedding (a crisp row, or the weighted average of the prototypes)
        info = {"ffn_strength": []}
        for l in range(self.L):
            b = f"blocks.{l}."
            h, _, _ = self.gain_controller(x, b + "n1.")
            qkv = self.gated_map(h, b + "mix.qkv.").reshape(3, H, dh)  # q, k, v for each head
            alpha_logit, _ = self.rule_gate(h, b + "mix.gate.")  # how much each head forgets
            beta_logit, _ = self.rule_gate(h, b + "mix.beta.")  # how hard each head writes
            y = np.zeros((H, dh))
            for hd in range(H):
                c, lw = W[b + "mix.centers"][hd], W[b + "mix.log_width"][hd]
                fq, fk = self.memory_partition(qkv[0, hd], c, lw), self.memory_partition(qkv[1, hd], c, lw)
                S = self.S[l][hd] * sigmoid(alpha_logit[hd])  # forget
                S = S + sigmoid(beta_logit[hd]) * np.outer(qkv[2, hd] - S @ fk, fk)  # write the prediction error into the rules fk selects
                self.S[l][hd] = S
                y[hd] = S @ fq  # read: defuzzify the rules' stored consequents with the query's firing
            x = x + self.gated_map(y.reshape(-1), b + "mix.out.")
            n2, _, _ = self.gain_controller(x, b + "n2.")
            out, strength = self.feedforward(n2, self.prev_n2[l], l)
            self.prev_n2[l] = n2
            info["ffn_strength"].append(strength)
            x = x + out
        hn, _, gain = self.gain_controller(x, "norm.")
        info["final_gain"] = float(np.exp(gain))
        logits = self.E @ hn  # tied output table: head pieces crisp, tail pieces = membership-weighted average of the prototype scores
        self.n_seen += 1
        return logits, info

    # ------------------------------------------------------------------ text in, text out
    def piece(self, i):
        return "\n<|eot|>\n" if i == 0 else self.tok.decode([int(i)])

    def generate(self, text, max_new_tokens=60, temperature=0.5, top_k=10, seed=0, include_prompt=False, trace=None):
        """Encode ``text``, run the fuzzy model one word piece at a time, sample the continuation and return it as a string.
        ``trace``: pass an empty list to receive, per generated piece, the feed-forward rules with the largest effect on its score."""
        ids = self.tok.encode(text).ids
        assert 0 < len(ids) < self.max_len, "empty prompt, or longer than the context"
        rng = np.random.default_rng(seed)
        self.reset()
        for i in ids:
            logits, info = self.step(i)
        out = []
        for _ in range(min(max_new_tokens, self.max_len - len(ids))):
            lg = logits / max(temperature, 1e-6)
            if top_k:
                lg = np.where(lg < np.sort(lg)[-top_k], -np.inf, lg)
            tok = int(rng.choice(self.V, p=softmax(lg)))
            if trace is not None:
                trace.append((self.piece(tok), self.explain(tok, info)))
            out.append(tok)
            logits, info = self.step(tok)
        gen = "".join(self.piece(i) for i in out)
        return (text + gen) if include_prompt else gen

    # ------------------------------------------------------------------ reading the rules
    def profile(self, layer):
        """What each feed-forward rule does to the output scores when it fires with a positive strength: pi_r = (conseq_r * final weight) . E^T."""
        return (self.W[f"blocks.{layer}.ffn.conseq"] * self.W["norm.weight"]) @ self.E.T  # (rules, pieces)

    def explain(self, tok, info, per_layer=3):
        """The feed-forward rules that pushed hardest for (or against) ``tok``: exact, because the output scores are linear in the rules' pushes."""
        rows = []
        for l in range(self.L):
            contrib = info["ffn_strength"][l] * info["final_gain"] * self.profile(l)[:, tok]
            for r in np.argsort(-np.abs(contrib))[:per_layer]:
                rows.append((l, int(r), float(info["ffn_strength"][l][r]), float(contrib[r])))
        return rows

    def rule_text(self, layer, r, top=8):
        """One feed-forward rule as IF-THEN text."""
        W, p = self.W, f"blocks.{layer}.ffn."
        c, lw = W[p + "centers"][r], W[p + "log_width"][r]
        D = len(c)
        sigma = np.exp(lw) * np.sqrt(D)  # the width a single dimension's membership function effectively has, given the 1/D exponent scale
        names = [f"x[{j}]" if j < self.d else f"prev[{j - self.d}]" for j in range(D)] if D > self.d else [f"x[{j}]" for j in range(D)]
        sel = np.argsort(sigma)[:4]  # the four narrowest (most selective) memberships
        ante = " AND ".join(f"{names[j]} is near {c[j]:+.2f} (width {sigma[j]:.2f})" for j in sel)
        pi = self.profile(layer)[r]
        up = ", ".join(repr(self.piece(i)) for i in np.argsort(-pi)[:top])
        down = ", ".join(repr(self.piece(i)) for i in np.argsort(pi)[:4])
        return (
            f"  rule {r:3d}: IF {ante} (and {D - 4} more memberships, product t-norm)\n"
            f"            THEN push the state by consequent vector c_r (norm {np.linalg.norm(W[p + 'conseq'][r]):.2f}) scaled by firing x slope "
            f"(slope norm {np.linalg.norm(W[p + 'slope.weight'][r]):.2f}, bias {W[p + 'slope.bias'][r]:+.2f}); with positive strength it raises {up}; lowers {down}"
        )


def write_rulebook(path="rulebook.txt", model=None):
    """Write every rule of every fuzzy system in the model."""
    m = model or AllFuzzyLM()
    W, out = m.W, []
    n_ffn = sum(len(W[f"blocks.{l}.ffn.centers"]) for l in range(m.L))
    out.append("THE ALL-FUZZY LANGUAGE MODEL: ITS RULE BASE")
    out.append(f"{m.L} blocks, width {m.d}; fuzzy systems per block: 2 gain controllers (5 rules each), 2 gated maps (8 rules each), 2 gates (6 rules each),")
    out.append(f"a memory ({m.H} heads x {m.R_mem} rules), a feed-forward bank; plus the output gain controller and the fuzzy token table.")
    out.append(f"Feed-forward rules in total: {n_ffn}; memory rules: {m.L * m.H * m.R_mem}.\n")
    out.append("== 1. TOKEN TABLE: the 12 prototypes that every tail word piece is a fuzzy set over (the 384 most frequent pieces have crisp rows)")
    mu = m.tail_membership
    for k in range(mu.shape[1]):
        top = np.argsort(-mu[:, k])[:10]
        out.append(f"  prototype {k:2d}: strongest members " + ", ".join(repr(m.piece(m.tail_ids[i])) for i in top))
    out.append("  (a tail piece's embedding = membership-weighted average of the prototypes; its output score = the same weighted average of the prototype scores)\n")
    out.append("== 2. GAIN-CONTROL RULES (replace RMSNorm): IF log-rms(x) is near c THEN log-gain = b + k (log-rms - c)")
    names = [f"blocks.{l}.{n}." for l in range(m.L) for n in ("n1", "n2")] + ["norm."]
    for p in names:
        c, lw, b, k = W[p + "centers"], np.exp(W[p + "log_width"]), W[p + "offset"], W[p + "slope"]
        out.append(f"  {p[:-1]}: " + "; ".join(f"[c {c[r]:+.2f}, width {lw[r]:.2f}, b {b[r]:+.2f}, k {k[r]:+.2f}]" for r in range(len(c))))
    for l in range(m.L):
        b = f"blocks.{l}."
        out.append(f"\n== 3. BLOCK {l}: MIXER")
        for nm, sysname in (("mix.qkv.", "q/k/v map"), ("mix.out.", "output map")):
            c, lw, s = W[b + nm + "centers"], np.exp(W[b + nm + "log_width"]), W[b + nm + "s"]
            out.append(f"  {sysname}: {len(c)} rules over {c.shape[1]} basis coordinates; IF the basis coordinates are near c THEN rescale the basis directions by s")
            for r in range(len(c)):
                out.append(f"    rule {r}: centre norm {np.linalg.norm(c[r]):.2f}, mean width {lw[r].mean():.2f}; rescaling s: mean {s[r].mean():.2f}, range [{s[r].min():+.2f}, {s[r].max():+.2f}]")
        for nm, what in (("mix.gate.", "forgetting"), ("mix.beta.", "write strength")):
            c, lw, cs = W[b + nm + "centers"], np.exp(W[b + nm + "log_width"]), W[b + nm + "conseq"]
            out.append(f"  {what} gate: {len(c)} rules over the 48-number input; THEN the head logits (before the sigmoid) are:")
            for r in range(len(c)):
                sel = np.argsort(lw[r])[:3]
                out.append(f"    rule {r}: IF " + " AND ".join(f"x[{j}] near {c[r][j]:+.2f}" for j in sel) + " (+45 more) THEN logits per head " + np.array2string(cs[r], precision=2, suppress_small=True))
        out.append(f"  memory: {m.H} heads x {m.R_mem} rules over each head's {m.dh}-number key/query; a rule's consequent is in-context state (a column of the head's memory),")
        out.append("          written by the delta rule at run time, so only the antecedents are fixed:")
        for hd in range(m.H):
            c, lw = W[b + "mix.centers"][hd], np.exp(W[b + "mix.log_width"][hd])
            for r in range(m.R_mem):
                out.append(f"    head {hd} rule {r:2d}: IF key near " + np.array2string(c[r], precision=2, suppress_small=True) + " widths " + np.array2string(lw[r] * np.sqrt(m.dh), precision=2, suppress_small=True))
        out.append(f"\n== 4. BLOCK {l}: FEED-FORWARD RULES ({len(W[b + 'ffn.centers'])} rules" + (", antecedent over the current AND previous token" if W[b + "ffn.centers"].shape[1] > m.d else "") + ")")
        for r in range(len(W[b + "ffn.centers"])):
            out.append(m.rule_text(l, r))
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(out)


_DEFAULT = None


def generate(text, **kw):
    """Text in, generated text out (the default all-fuzzy model).  Keyword arguments: max_new_tokens, temperature, top_k, seed, include_prompt, trace."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = AllFuzzyLM()
    return _DEFAULT.generate(text, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Once upon a time")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", action="store_true", help="print the feed-forward rules behind each generated word piece")
    ap.add_argument("--rules", metavar="FILE", help="write the whole rule base to FILE and exit")
    a = ap.parse_args()
    if a.rules:
        print(f"wrote {write_rulebook(a.rules)} lines to {a.rules}")
    else:
        tr = [] if a.trace else None
        print(a.text + generate(a.text, max_new_tokens=a.n, seed=a.seed, trace=tr))
        for piece, rows in tr or []:
            print(f"\n{piece!r}: " + "; ".join(f"layer {l} rule {r} (strength {s:+.2f}, effect {c:+.2f})" for l, r, s, c in rows))
