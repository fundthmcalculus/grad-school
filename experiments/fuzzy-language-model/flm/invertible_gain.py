"""An exactly invertible gain layer: the norm of the three-layer model made reversible by construction (``PLAN_BIJECTIVE.md``).

    python -m flm.invertible_gain [checkpoint.pt]      # layer checks, then the chain F -> gain -> identity blocks -> inverse gain -> F^-1 on text (CPU, seconds)

The model's ``FuzzyNorm`` is  y = x * exp(g(log rms x)) * w  with g an order-1 TSK system initialised to g(z) = -z, so the output rms is about 1 whatever the input's: it throws the scale away and cannot be inverted.
This layer keeps the same shape (a scalar gain from an order-1 TSK system of log rms, times a per-dimension weight) but makes the *log-rms map* strictly increasing, which is exactly what invertibility needs:

    u = log rms(x),   v(u) = lo + integral_lo^u s(t) dt,   s(t) = softplus(T(t)) + floor > 0,   T = order-1 TSK  (normalised Gaussian rules, consequents a_r + b_r (t - c_r))
    y = (x / rms x) * exp(v(u)) * w                      so  rms(y / w) = exp(v(u))

s is the slope of the log-gain map and is positive for every parameter setting, so v is strictly increasing for every parameter setting: the layer is a bijection of R^d minus the origin (the origin maps to itself) whatever is learned.
The integral is a trapezoid sum on a fixed grid and v is its piecewise-linear interpolant (linearly extrapolated with the end slopes outside the grid); the inverse interpolates the same segments the other way round, so
forward and inverse are exact inverses of each other up to float rounding, not an approximation of one another.  With s = 1 everywhere (the start) the layer is the identity up to the weights.

Inverse:  y' = y / w,  v = log rms(y'),  u = v^-1(v),  x = y' * exp(u - v).  The inverse is a function of its argument alone, so it is well defined for *any* vector, including one a block has changed: that is what lets the
output stage undo the input stage's gain after the blocks.
"""

import math
import sys

import torch
from torch import nn


class InvertibleGain(nn.Module):
    def __init__(
        self,
        d: int,
        R: int = 5,
        lo: float = -8.0,
        hi: float = 8.0,
        grid: int = 1024,
        floor: float = 1e-3,
    ):
        super().__init__()
        self.lo, self.hi, self.G, self.floor = lo, hi, grid, floor
        self.centers = nn.Parameter(torch.linspace(-3.0, 3.0, R))
        self.log_width = nn.Parameter(torch.full((R,), math.log(1.5)))
        self.a = nn.Parameter(
            torch.full((R,), math.log(math.e - 1.0))
        )  # softplus(a) = 1: log-gain slope 1, v(u) = u
        self.b = nn.Parameter(torch.zeros(R))
        self.log_w = nn.Parameter(torch.zeros(d))

    def slope(self, u):
        """s(u) > 0: the order-1 TSK system, through a softplus."""
        z = u.unsqueeze(-1) - self.centers
        w = torch.softmax(-0.5 * (z / torch.exp(self.log_width)) ** 2, dim=-1)
        return nn.functional.softplus((w * (self.a + self.b * z)).sum(-1)) + self.floor

    def _grid(self):
        ug = torch.linspace(
            self.lo, self.hi, self.G, dtype=self.a.dtype, device=self.a.device
        )
        s = self.slope(ug)
        du = (self.hi - self.lo) / (self.G - 1)
        vg = torch.cat(
            [
                ug.new_full((1,), self.lo),
                self.lo + torch.cumsum(0.5 * (s[1:] + s[:-1]) * du, 0),
            ]
        )
        return ug, vg, du

    def v_of_u(self, u, ug, vg, du):
        i = (torch.searchsorted(ug, u.contiguous()) - 1).clamp(0, self.G - 2)
        return vg[i] + (u - ug[i]) / du * (
            vg[i + 1] - vg[i]
        )  # not clamped in t: the end segments extrapolate

    def u_of_v(self, v, ug, vg, du):
        i = (torch.searchsorted(vg, v.contiguous()) - 1).clamp(0, self.G - 2)
        return ug[i] + (v - vg[i]) / (vg[i + 1] - vg[i]) * du

    @staticmethod
    def _log_rms(x):
        return 0.5 * torch.log(x.pow(2).mean(-1).clamp_min(1e-60))

    def forward(self, x):
        ug, vg, du = self._grid()
        u = self._log_rms(x)
        v = self.v_of_u(u, ug, vg, du)
        return x * torch.exp(v - u).unsqueeze(-1) * torch.exp(self.log_w)

    def inverse(self, y):
        ug, vg, du = self._grid()
        y = y / torch.exp(self.log_w)
        v = self._log_rms(y)
        u = self.u_of_v(v, ug, vg, du)
        return y * torch.exp(u - v).unsqueeze(-1)


def _perturb(g, seed=0, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (g.a, g.b, g.log_w, g.centers):
            p.add_(scale * 0.5 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
        g.log_width.add_(
            0.2 * torch.randn(g.log_width.shape, generator=gen, dtype=g.log_width.dtype)
        )
    return g


def layer_checks():
    torch.manual_seed(0)
    d = 16
    g = _perturb(InvertibleGain(d).double())
    ug, vg, du = g._grid()
    print(
        f"v is strictly increasing on the grid: min step {(vg[1:] - vg[:-1]).min().item():.2e} (> 0 for every parameter setting: s = softplus + {g.floor})"
    )
    print(
        f"slope range of the log-gain map after perturbation: s in [{g.slope(ug).min().item():.3f}, {g.slope(ug).max().item():.3f}]; v(u) - u spans {(vg - ug).min().item():+.2f} .. {(vg - ug).max().item():+.2f}"
    )
    for dt, name in ((torch.double, "float64"), (torch.float, "float32")):
        gg = g.to(dt)
        mags = 10.0 ** torch.linspace(
            -3, 3, 13, dtype=dt
        )  # rms from 1e-3 to 1e3: past both ends of the grid on purpose is covered below
        x = torch.randn(13, 64, d, dtype=dt) * mags[:, None, None]
        y = gg(x)
        back = gg.inverse(y)
        rel = ((back - x).norm(dim=-1) / x.norm(dim=-1)).max().item()
        print(
            f"{name}: gain round trip over rms 1e-3..1e3: max relative error {rel:.2e}"
        )
    g = g.double()
    far = (
        torch.randn(8, d, dtype=torch.double)
        * torch.tensor([1e-6, 1e-5, 1e-4, 1e4, 1e5, 1e6, 1e7, 1e8], dtype=torch.double)[
            :, None
        ]
    )
    print(
        f"far outside the grid (rms 1e-6..1e8): max relative error {((g.inverse(g(far)) - far).norm(dim=-1) / far.norm(dim=-1)).max().item():.2e}"
    )
    z = torch.randn(1000, d, dtype=torch.double) * 3.0
    print(
        f"inverse then forward on arbitrary vectors (the direction the output stage uses): max |error| {(g(g.inverse(z)) - z).abs().max().item():.2e}"
    )
    print(
        f"zero vector maps to {g(torch.zeros(1, d, dtype=torch.double)).abs().max().item():.1e} and back to {g.inverse(g(torch.zeros(1, d, dtype=torch.double))).abs().max().item():.1e}"
    )
    xs = torch.randn(4, d, dtype=torch.double)
    j = torch.autograd.functional.jacobian(lambda v: g(v), xs[0])
    print(
        f"Jacobian of the forward map is nonsingular: |det J| = {torch.linalg.det(j).abs().item():.3e}, condition number {torch.linalg.cond(j).item():.1f}"
    )


def chain_checks(path):
    import flm.verify_invertible as VI  # noqa: E402  (builds the trained c-arm model)
    from flm.data import decode, encode, load  # noqa: E402

    model = VI.build(path)
    d = model.cfg.d_model
    texts = [
        "Once upon a time, a little girl named Lily went to the park.",
        "The quick brown fox jumps over the lazy dog 0123456789 ?!;:",
        decode(load("valid")[:2000].astype("int64")),
    ]
    print(
        f"\nchain on the trained c-arm model (d {d}, {sum(p.numel() for p in model.parameters()):,} parameters): text -> C -> z = F(C) -> gain -> [blocks: identity] -> gain^-1 -> F^-1 -> nearest codeword -> text"
    )
    for dt, name in ((torch.double, "float64"), (torch.float, "float32")):
        m = model.to(dt)
        gain = _perturb(InvertibleGain(d).to(dt), seed=1)
        for t in texts:
            ids = torch.as_tensor(encode(t), dtype=torch.long)
            with torch.no_grad():
                z = m.F(m.codebook[ids])[0]  # embedding layer
                h = gain(z)  # the invertible gain, on the way in
                h = h  # the blocks: identity
                u = m.F.inverse(
                    gain.inverse(h)
                )  # gain undone, then the embedding layer backwards
                back = torch.cdist(u, m.codebook).argmin(-1)
                bad = int((back != ids).sum())
            same = decode(back.numpy()) == decode(ids.numpy())
            print(
                f"  {name}: {len(ids):5d} symbols, {bad} wrong, text identical: {same}, max |u - C| {(u - m.codebook[ids]).abs().max().item():.1e}; gain moved |z| by x{(h.norm(dim=-1) / z.norm(dim=-1)).mean().item():.2f} on average"
            )
    # robustness: noise added to h before the inverse (what a real block does) and how much survives
    m = model.to(torch.double)
    gain = _perturb(InvertibleGain(d).double(), seed=1)
    ids = torch.as_tensor(encode(texts[2]), dtype=torch.long)
    with torch.no_grad():
        h = gain(m.F(m.codebook[ids])[0])
        for rel in (0.0, 0.01, 0.03, 0.1, 0.3):
            hn = h + rel * h.norm(dim=-1, keepdim=True) / math.sqrt(
                d
            ) * torch.randn_like(h)
            back = torch.cdist(m.F.inverse(gain.inverse(hn)), m.codebook).argmin(-1)
            print(
                f"  noise {rel:4.2f} x rms on h: {(back == ids).float().mean().item() * 100:6.2f}% of symbols recovered"
            )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    layer_checks()
    chain_checks(
        sys.argv[1] if len(sys.argv) > 1 else "outputs/bij10k/verify/c_lr0.04_s0.pt"
    )
