"""An exactly invertible fuzzy system: order-1 TSK systems as the conditioners of coupling layers (``PLAN_BIJECTIVE.md``).

    python -m flm.bijective_tsk          # the invertibility checks (CPU, a few seconds)

The map F: R^N -> R^N is a stack of coupling layers.  Each splits its input into halves (a, b) and changes only one half, with scale and shift that are order-1 TSK systems of the *other* half:

    y_a = a,    y_b = b * exp(s(a)) + t(a)          inverse:   b = (y_b - t(y_a)) * exp(-s(y_a)),   a = y_a

so the inverse needs no solve and no approximation: it is the same two fuzzy systems evaluated on the half that was never touched.  The log-determinant is sum(s(a)), which makes F a change of variables with a
density, not only a bijection.  An order-1 TSK system is  T(a) = sum_r w_r(a) (A_r a + c_r)  with normalised Gaussian rule firings w_r(a): a fuzzy partition of locally linear maps.
The scale is bounded, s = s_max tanh(raw / s_max), so exp(s) stays in a known range and the round trip stays within float precision.
"""

import math

import torch
from torch import nn


class TSK1(nn.Module):
    """Order-1 TSK system R^d_in -> R^d_out: R Gaussian rules over the input (product t-norm, 1/d exponent), normalised firings w_r, consequents A_r x + c_r."""

    def __init__(self, d_in: int, d_out: int, R: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(R, d_in))
        self.log_width = nn.Parameter(torch.zeros(R, d_in))
        self.A = nn.Parameter(torch.randn(R, d_out, d_in) * (0.1 / math.sqrt(d_in)))
        self.c = nn.Parameter(torch.zeros(R, d_out))
        self.scale = 1.0 / d_in

    def firing(self, x):
        a = torch.exp(-2.0 * self.log_width)
        ca = self.centers * a
        quad = (x * x) @ a.T - 2.0 * (x @ ca.T) + (self.centers * ca).sum(-1)
        return torch.softmax(-0.5 * self.scale * quad, dim=-1)

    def forward(self, x):
        w = self.firing(x)  # (..., R)
        lin = (
            torch.einsum("rod,...d->...ro", self.A, x) + self.c
        )  # each rule's own linear map
        return (w.unsqueeze(-1) * lin).sum(-2)


class TSKCoupling(nn.Module):
    """One invertible step: half of the coordinates, rescaled and shifted by order-1 TSK systems of the other half."""

    def __init__(self, n: int, R: int, flip: bool, s_max: float = 2.0):
        super().__init__()
        self.na = n // 2 if not flip else n - n // 2  # size of the conditioning half
        self.nb = n - self.na
        self.flip, self.s_max = flip, s_max
        self.scale_sys = TSK1(self.na, self.nb, R)
        self.shift_sys = TSK1(self.na, self.nb, R)

    def _split(self, x):
        return (
            (x[..., self.nb :], x[..., : self.nb])
            if self.flip
            else (x[..., : self.na], x[..., self.na :])
        )

    def _join(self, a, b):
        return torch.cat([b, a], -1) if self.flip else torch.cat([a, b], -1)

    def _s(self, a):
        return self.s_max * torch.tanh(self.scale_sys(a) / self.s_max)

    def forward(self, x):
        a, b = self._split(x)
        s = self._s(a)
        return self._join(a, b * torch.exp(s) + self.shift_sys(a)), s.sum(-1)

    def inverse(self, y):
        a, yb = self._split(y)
        return self._join(a, (yb - self.shift_sys(a)) * torch.exp(-self._s(a)))


class FuzzyBijection(nn.Module):
    """F and F^-1 from the same parameters: the embedding layer run forwards is the de-embedding layer run backwards."""

    def __init__(self, n: int, R: int = 16, layers: int = 4, s_max: float = 2.0):
        super().__init__()
        self.steps = nn.ModuleList(
            TSKCoupling(n, R, flip=bool(i % 2), s_max=s_max) for i in range(layers)
        )

    def forward(self, x):
        logdet = 0.0
        for st in self.steps:
            x, ld = st(x)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, y):
        for st in reversed(self.steps):
            y = st.inverse(y)
        return y


def _checks():
    torch.manual_seed(0)
    n, R, V = 32, 16, 6144
    F = FuzzyBijection(n, R, layers=4).double()
    for (
        p
    ) in (
        F.parameters()
    ):  # a trained map is far from the initial one: perturb every parameter so the checks do not pass on a near-identity
        p.data += 0.5 * torch.randn_like(p)
    x = torch.randn(V, n, dtype=torch.double) * 2.0  # a codebook of V token vectors
    z, logdet = F(x)
    back = F.inverse(z)
    err = (back - x).abs().max().item()
    print(
        f"round trip F^-1(F(x)) on {V} codewords, N={n}: max |error| {err:.2e} (float64)"
    )
    zf = z.float()
    backf = F.float().inverse(zf)
    nearest = torch.cdist(backf, x.float()).argmin(-1)
    print(
        f"float32 round trip, nearest-codeword decode: {(nearest == torch.arange(V)).float().mean().item() * 100:.2f}% of tokens recovered, max |error| {(backf - x.float()).abs().max().item():.2e}"
    )
    F = F.double()
    # the log-determinant against the Jacobian of the forward map, on a few points
    for i in range(3):
        J = torch.autograd.functional.jacobian(lambda v: F(v)[0], x[i])
        sign, ld = torch.linalg.slogdet(J)
        print(
            f"point {i}: log|det J| autograd {ld.item():+.6f}  sum of scales {F(x[i])[1].item():+.6f}  (sign {sign.item():+.0f})"
        )
    # injectivity in the other direction: random points of the logic space map back and forth too
    zz = torch.randn(1000, n, dtype=torch.double) * 3.0
    print(
        f"F(F^-1(z)) on random logic-space points: max |error| {(F(F.inverse(zz))[0] - zz).abs().max().item():.2e}"
    )
    print(
        f"order-1 TSK parameters per coupling at N={n}, R={R}: {sum(p.numel() for p in F.steps[0].parameters()):,}"
    )


if __name__ == "__main__":
    _checks()
