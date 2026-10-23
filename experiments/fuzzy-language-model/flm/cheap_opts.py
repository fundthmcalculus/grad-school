"""Cheap first-order optimisers for the CPU exploration (`flm/dust_cpu.py`): Lion and a Muon-style optimiser.

Lion (Chen et al. 2023): update = sign(beta1 m + (1 - beta1) g), momentum m <- beta2 m + (1 - beta2) g; one state tensor, sign update.
Muon (Jordan et al. 2024, as used for small language models): for 2-D weight matrices, the momentum is orthogonalised by a few Newton-Schulz iterations before the step; everything else (embeddings, norms, rule centres and widths, biases, 3-D tensors) uses Adam.
"""

import torch


@torch.no_grad()
def newton_schulz(g, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if tall else x


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                m = st.setdefault("m", torch.zeros_like(p))
                upd = (b1 * m + (1 - b1) * p.grad).sign()
                p.mul_(1 - g["lr"] * g["weight_decay"]).add_(upd, alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


class Muon(torch.optim.Optimizer):
    """Muon on the 2-D matrices whose ids are in ``matrix_ids``; Adam for the rest."""

    def __init__(self, params, matrix_ids, lr=0.02, adam_lr=3e-3, momentum=0.95):
        super().__init__(params, dict(lr=lr, adam_lr=adam_lr, momentum=momentum))
        self.matrix_ids = matrix_ids
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                m = st.setdefault("m", torch.zeros_like(p))
                if id(p) in self.matrix_ids:
                    m.mul_(g["momentum"]).add_(p.grad)
                    upd = newton_schulz(p.grad + g["momentum"] * m)
                    upd *= max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                    p.add_(upd, alpha=-g["lr"])
                else:
                    v = st.setdefault("v", torch.zeros_like(p))
                    m.mul_(0.9).add_(p.grad, alpha=0.1)
                    v.mul_(0.98).addcmul_(p.grad, p.grad, value=0.02)
                    mh, vh = m / (1 - 0.9**self.t), v / (1 - 0.98**self.t)
                    p.addcdiv_(mh, vh.sqrt() + 1e-8, value=-g["adam_lr"])
