"""Consequent-first rule construction for the TSK FFN (the TRIBBLE order, applied to the language model).

TRIBBLE builds a fuzzy system by committing to the answers first (the output classes) and only then working
backward to the antecedents that select them.  ``init_rules_from_data`` (the existing ``--rule-init data``) is the
opposite, antecedent-first: it drops every rule's centre on a random input.  Here, layer by layer:

1. answers  : cluster the *targets* (the tied embedding of the next token) into R groups with a frequency-weighted
              spherical k-means over the vocabulary; every position belongs to the group of its own next token;
2. backward : rule r's antecedent is the Gaussian of the FFN inputs u of the positions in group r (centre = mean,
              width = std; HTSK/pooled widths when the width is shared);
3. answers' consequents: rule r's consequent is the group's centroid direction (``mode="centroid"``) or the ridge
              least-squares fit of the target embeddings to the rules' firing strengths (``mode="ls"``),
              times one scalar gain chosen per layer by cross-entropy on the construction batch.

Nothing here uses a gradient.  It only touches ``TSKFFN`` blocks, so it is a no-op for other FFNs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flm.models import TinyLM, TSKFFN, set_widths_from_std


def _spherical_kmeans(
    z: torch.Tensor, w: torch.Tensor, k: int, gen, iters: int = 15
) -> torch.Tensor:
    """Frequency-weighted spherical k-means over the rows of z (V, d); returns the group id of every row."""
    zn = F.normalize(z, dim=-1)
    V = zn.shape[0]
    k = min(k, int((w > 0).sum()))
    # weighted k-means++ seeding
    cent = zn[torch.multinomial(w / w.sum(), 1, generator=gen)]
    for _ in range(k - 1):
        d2 = (1 - zn @ cent.T).min(1).values.clamp(min=0) * w
        nxt = torch.multinomial(d2 / d2.sum(), 1, generator=gen)
        cent = torch.cat([cent, zn[nxt]], 0)
    for _ in range(iters):
        g = (zn @ cent.T).argmax(1)
        for r in range(k):
            m = g == r
            if m.any():
                cent[r] = F.normalize((zn[m] * w[m, None]).sum(0), dim=-1)
    return (zn @ cent.T).argmax(1)


@torch.no_grad()
def _ce(model: TinyLM, idx: torch.Tensor) -> float:
    lg = model(idx[:, :-1])
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), idx[:, 1:].reshape(-1)).item()


@torch.no_grad()
def init_conseq_first(
    model: TinyLM,
    idx: torch.Tensor,
    generator: torch.Generator | None = None,
    mode: str = "centroid",
    gains: tuple = (0.25, 1, 4, 16, 64, 256, 1024),
    ridge: float = 1e-2,
) -> dict:
    """Build every TSK FFN rule consequent-first from the windows ``idx`` (B, T+1). Returns diagnostics."""
    assert mode in ("centroid", "ls")
    V = model.cfg.vocab_size
    E = model.emb.weight if model.up is None else model.up(model.emb.weight)
    tgt = idx[:, 1:].reshape(-1)
    freq = torch.bincount(tgt, minlength=V).float()
    info = {}
    for li, blk in enumerate(model.blocks):
        x = model.embed(idx[:, :-1])
        for b in model.blocks[:li]:
            x = b(x)
        x = x + blk.mix(blk.mixer_in(x))
        ffn = blk.ffn
        if not isinstance(ffn, TSKFFN):
            continue
        n = blk.n2(x).reshape(-1, x.shape[-1])
        u = ffn.ante(blk.n2(x)).reshape(-1, ffn.da)
        R = ffn.R
        # 1. answers first: group the next-token embeddings
        grp_tok = _spherical_kmeans(E, freq + 1e-6 * (freq == 0), R, generator)
        grp = grp_tok[tgt]
        used = torch.unique(grp)
        # 2. backward to the antecedents of each answer group
        gstd = u.std(0)
        centers = ffn.centers.data.clone()
        stds = torch.empty(R, ffn.da)
        for r in range(R):
            m = grp == used[r % len(used)]
            if int(m.sum()) >= 4:
                centers[r] = u[m].mean(0)
                stds[r] = u[m].std(0).clamp(min=0.25 * gstd)
            else:
                centers[r] = u[torch.randint(0, len(u), (1,), generator=generator)][0]
                stds[r] = gstd
        ffn.centers.copy_(centers)
        if ffn.log_width.shape[0] == R:  # per-rule widths
            for r in range(R):
                set_widths_from_std(ffn.log_width[r : r + 1], stds[r])
        else:  # shared across rules: the pooled within-group std
            set_widths_from_std(ffn.log_width, stds.pow(2).mean(0).sqrt())
        # 3. consequents
        Et = E[tgt]
        if mode == "centroid":
            cons = torch.zeros(R, E.shape[1])
            for r in range(R):
                m = grp == used[r % len(used)]
                if m.any():
                    cons[r] = F.normalize(Et[m].mean(0), dim=-1)
        else:
            W = ffn.firing(
                n.view(-1, x.shape[-1])
            )  # (N, R); the first-order slope is set to 1
            A = W.T @ W + ridge * torch.eye(R) * W.pow(2).sum() / R
            cons = torch.linalg.solve(A, W.T @ F.normalize(Et, dim=-1))
        base_conseq = cons
        if ffn.order == 1:
            ffn.slope.bias.fill_(1.0)
            ffn.slope.weight.mul_(0.1)
        best = (float("inf"), 1.0)
        for g in gains:
            ffn.conseq.copy_(base_conseq * g)
            ce = _ce(model, idx)
            if ce < best[0]:
                best = (ce, g)
        ffn.conseq.copy_(base_conseq * best[1])
        info[f"L{li}.ffn"] = {"gain": best[1], "ce_nats": round(best[0], 4)}
    return info
