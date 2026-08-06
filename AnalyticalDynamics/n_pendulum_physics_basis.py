"""
Automates the "known denominator, physics-shaped numerator" decomposition
used by n2_physics_informed_v2_rational.py, generalized to arbitrary n via
the symbolic n-link model in n_pendulum_symbolic.py.

For M(q) qddot = f(q, qdot), Cramer's rule gives qddot_i = det(M_i) / det(M),
where M_i is M with column i replaced by f. det(M) depends only on theta
(given known m, l) -- the same shared, exactly-computable denominator for
every output, generalizing the double pendulum's "2m1+m2-m2*cos(2*Delta)"
bracket. Expanding det(M_i) (WITHOUT expanding compound trig arguments --
sin(theta_i - theta_j) stays atomic, it's just algebraically expanded)
gives a finite sum of additive terms; each becomes one physics-basis
feature (divided by det(M)) for output i's consequent equation. This
recovered the true double-pendulum equations exactly for n=2 (a 4-term and
3-term decomposition matching the by-hand derivation in
DOUBLE_PENDULUM_REPORT.md); for n=3 it produces 18 terms per output.
"""

from dataclasses import dataclass

import numpy as np
import sympy as sp

from n_pendulum_symbolic import build_n_pendulum


@dataclass
class PhysicsBasisModel:
    n: int
    detM_func: callable
    # per_output_terms[i] = list of (sympy_expr, lambdified_func) for output i
    per_output_terms: list
    per_output_names: list  # human-readable names, one list per output


def _term_name(term, idx):
    """Short, stable, order-independent-ish name for a basis term."""
    return f"t{idx}"


def derive_physics_basis(
    n: int, m_vals, l_vals, g_val: float = 9.81
) -> PhysicsBasisModel:
    model = build_n_pendulum(n)
    subs_params = (
        list(zip(model.m, m_vals)) + list(zip(model.l, l_vals)) + [(model.g, g_val)]
    )
    M_num = model.M.subs(subs_params)
    f_num = model.f.subs(subs_params)
    all_syms = model.theta + model.thetad

    detM_expr = sp.expand(M_num.det())
    detM_func = sp.lambdify(all_syms, detM_expr, "numpy")

    per_output_terms = []
    per_output_names = []
    for i in range(n):
        Mi = M_num.as_mutable()
        Mi[:, i] = f_num
        Ni = sp.expand(Mi.det())
        terms = sp.Add.make_args(Ni)
        term_entries = [(t, sp.lambdify(all_syms, t, "numpy")) for t in terms]
        per_output_terms.append(term_entries)
        per_output_names.append([_term_name(t, k) for k, t in enumerate(terms)])

    return PhysicsBasisModel(
        n=n,
        detM_func=detM_func,
        per_output_terms=per_output_terms,
        per_output_names=per_output_names,
    )


def compute_features(basis: PhysicsBasisModel, theta_arrs, omega_arrs):
    """theta_arrs / omega_arrs: length-n lists of (possibly array-valued) values.

    Returns (denom, [feature_matrix_per_output]), where feature_matrix_per_output[i]
    has shape (len(theta_arrs[0]), n_terms_i).
    """
    args = list(theta_arrs) + list(omega_arrs)
    denom = np.asarray(basis.detM_func(*args), dtype=float)
    feats = []
    for i in range(basis.n):
        cols = [
            np.asarray(tf(*args), dtype=float) / denom
            for (_expr, tf) in basis.per_output_terms[i]
        ]
        feats.append(np.column_stack(cols) if cols else np.zeros((len(denom), 0)))
    return denom, feats


def state_cols(n):
    return [f"theta_{i}" for i in range(1, n + 1)], [
        f"omega_{i}" for i in range(1, n + 1)
    ]
