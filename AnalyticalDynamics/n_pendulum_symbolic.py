"""
Symbolic Lagrangian derivation of the planar n-link pendulum chain, via SymPy.

Generalizes the by-hand double-pendulum derivation in
DOUBLE_PENDULUM_REPORT.md to an arbitrary chain length n: builds T, V, and
L = T - V symbolically for n links, hands the Lagrangian to
sympy.physics.mechanics.LagrangesMethod to get the manipulator-equation form

    M(q) qddot = f(q, qdot)

and lambdifies M and f into a numeric right-hand-side usable by
scipy.integrate.odeint. Solving M qddot = f with a numeric linear solve at
each integration step (rather than symbolically inverting M) is what keeps
this tractable past n=2-3: symbolic inversion blows up combinatorially with
n, a numeric solve is always O(n^3).

State convention matches DoublePendulum: interleaved
[theta_1, omega_1, theta_2, omega_2, ..., theta_n, omega_n].
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp
from sympy import symbols, sin, cos, Rational, lambdify
from sympy.physics.mechanics import dynamicsymbols, LagrangesMethod


@dataclass
class NPendulumModel:
    n: int
    t: sp.Symbol
    g: sp.Symbol
    m: tuple
    l: tuple
    theta: list
    thetad: list
    T: sp.Expr
    V: sp.Expr
    L: sp.Expr
    M: sp.Matrix
    f: sp.Matrix


@lru_cache(maxsize=None)
def build_n_pendulum(n: int) -> NPendulumModel:
    """
    Symbolically derive the planar n-link pendulum's equations of motion.

    Cached by n since forming the Euler-Lagrange equations is the expensive
    step and the symbolic model doesn't depend on numeric parameter values.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    t = symbols('t')
    g = symbols('g', positive=True)
    m = symbols(f'm1:{n + 1}', positive=True)
    l = symbols(f'l1:{n + 1}', positive=True)
    theta = list(dynamicsymbols(f'theta1:{n + 1}'))
    thetad = [sp.diff(th, t) for th in theta]

    # Cumulative bob positions: bob i hangs off the chain of rods 1..i.
    x = [sum(l[k] * sin(theta[k]) for k in range(i + 1)) for i in range(n)]
    y = [-sum(l[k] * cos(theta[k]) for k in range(i + 1)) for i in range(n)]
    xd = [sp.diff(xi, t) for xi in x]
    yd = [sp.diff(yi, t) for yi in y]

    T = sp.expand_trig(sp.expand(sum(Rational(1, 2) * m[i] * (xd[i]**2 + yd[i]**2) for i in range(n))))
    T = sp.trigsimp(T)
    V = sum(m[i] * g * y[i] for i in range(n))
    L = T - V

    lm = LagrangesMethod(L, theta)
    lm.form_lagranges_equations()

    return NPendulumModel(n=n, t=t, g=g, m=m, l=l, theta=theta, thetad=thetad,
                           T=T, V=V, L=L, M=sp.simplify(lm.mass_matrix), f=sp.simplify(lm.forcing))


def make_state_space(n: int, m_vals, l_vals, g_val: float = 9.81):
    """
    Build a numeric equations-of-motion function for an n-link pendulum.

    Returns (rhs, model) where rhs(state, t) -> dstate/dt is compatible with
    scipy.integrate.odeint / OdeSystem.simulate, and model is the underlying
    NPendulumModel for inspection (M, f, L, ...).
    """
    model = build_n_pendulum(n)
    subs_params = (list(zip(model.m, m_vals)) + list(zip(model.l, l_vals))
                   + [(model.g, g_val)])
    M_num = model.M.subs(subs_params)
    f_num = model.f.subs(subs_params)

    all_syms = model.theta + model.thetad
    M_func = lambdify(all_syms, M_num, 'numpy')
    f_func = lambdify(all_syms, f_num, 'numpy')

    def rhs(state, _t):
        theta = state[0::2]
        omega = state[1::2]
        args = list(theta) + list(omega)
        Mm = np.asarray(M_func(*args), dtype=float)
        ff = np.asarray(f_func(*args), dtype=float).flatten()
        alpha = np.linalg.solve(Mm, ff)
        dstate = np.empty_like(state, dtype=float)
        dstate[0::2] = omega
        dstate[1::2] = alpha
        return dstate

    return rhs, model


def state_labels(n: int) -> list[str]:
    labels = []
    for i in range(1, n + 1):
        labels += [f'theta_{i}', f'omega_{i}']
    return labels


if __name__ == '__main__':
    print("Building and displaying the n=2 (double pendulum) model...")
    model2 = build_n_pendulum(2)
    print("Mass matrix M(q):")
    sp.pprint(model2.M)
    print("\nForcing f(q, qdot)  [M(q) qddot = f(q, qdot)]:")
    sp.pprint(model2.f)
