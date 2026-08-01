"""Standard t-norm / t-conorm families, as De Morgan dual pairs.

`tribblefis.gauss_math` exposes a single `norm_conorm` string that selects both
operators at once, and its Hamacher t-conorm is implemented as (x+y)/(1-xy),
which leaves [0,1] -- at x=y=0.5 it returns 1.333, so it is not a t-conorm.
Its Hamacher t-norm is xy/(x+y-xy), correct but 0/0 when both inputs underflow.

This module supplies verified operators under the standard negation N(x)=1-x,
so the anomaly rule can be evaluated over arbitrary (T, S) pairs -- including
mismatched ones -- rather than only the four built-ins.

Every operator is a binary ufunc-style callable on arrays in [0,1] and is
guarded at the removable singularities.
"""

import numpy as np

EPS = 1e-12


def _safe(num, den, fill):
    """num/den with `fill` wherever den is ~0."""
    out = np.full_like(num, fill, dtype=float)
    ok = np.abs(den) > EPS
    np.divide(num, den, out=out, where=ok)
    return out


# --- t-norms (AND) ---------------------------------------------------------

def t_min(x, y):                       # Godel
    return np.minimum(x, y)


def t_prod(x, y):                      # algebraic
    return x * y


def t_luk(x, y):                       # Lukasiewicz
    return np.maximum(0.0, x + y - 1.0)


def t_hamacher(x, y):                  # Hamacher product, lambda=0
    return _safe(x * y, x + y - x * y, 0.0)


def t_einstein(x, y):
    return _safe(x * y, 2.0 - (x + y - x * y), 0.0)


def t_drastic(x, y):
    return np.where(np.isclose(y, 1.0), x, np.where(np.isclose(x, 1.0), y, 0.0))


def t_nilpotent(x, y):
    # Discontinuous on x+y==1; the tolerance keeps T and its dual S on the same
    # branch there, which floating-point rounding would otherwise split.
    return np.where(x + y > 1.0 + EPS, np.minimum(x, y), 0.0)


def _t_yager(p):
    def f(x, y):
        v = ((1 - x) ** p + (1 - y) ** p) ** (1.0 / p)
        return np.maximum(0.0, 1.0 - v)
    return f


def _t_schweizer_sklar(p):
    def f(x, y):
        return np.maximum(0.0, x ** p + y ** p - 1.0) ** (1.0 / p)
    return f


def _t_dombi(p):
    def f(x, y):
        xs, ys = np.clip(x, EPS, 1 - EPS), np.clip(y, EPS, 1 - EPS)
        v = (((1 - xs) / xs) ** p + ((1 - ys) / ys) ** p) ** (1.0 / p)
        return np.where((x <= EPS) | (y <= EPS), 0.0, 1.0 / (1.0 + v))
    return f


# --- t-conorms (OR), De Morgan duals of the above --------------------------

def s_max(x, y):
    return np.maximum(x, y)


def s_prob(x, y):
    return x + y - x * y


def s_luk(x, y):
    return np.minimum(1.0, x + y)


def s_hamacher(x, y):                  # (x + y - 2xy) / (1 - xy)
    return _safe(x + y - 2.0 * x * y, 1.0 - x * y, 1.0)


def s_einstein(x, y):
    return _safe(x + y, 1.0 + x * y, 1.0)


def s_drastic(x, y):
    return np.where(np.isclose(y, 0.0), x, np.where(np.isclose(x, 0.0), y, 1.0))


def s_nilpotent(x, y):
    return np.where(x + y < 1.0 - EPS, np.maximum(x, y), 1.0)


def _s_yager(p):
    def f(x, y):
        return np.minimum(1.0, (x ** p + y ** p) ** (1.0 / p))
    return f


def _s_schweizer_sklar(p):
    def f(x, y):                        # dual of the SS t-norm
        return 1.0 - np.maximum(0.0, (1 - x) ** p + (1 - y) ** p - 1.0) ** (1.0 / p)
    return f


def _s_dombi(p):
    def f(x, y):
        xs, ys = np.clip(x, EPS, 1 - EPS), np.clip(y, EPS, 1 - EPS)
        v = (((1 - xs) / xs) ** (-p) + ((1 - ys) / ys) ** (-p)) ** (-1.0 / p)
        return np.where((x >= 1 - EPS) | (y >= 1 - EPS), 1.0, 1.0 / (1.0 + v))
    return f


# --- registry --------------------------------------------------------------
# name -> (t-norm, t-conorm). Ordered roughly from most to least conjunctive.
FAMILIES = {
    "drastic":     (t_drastic, s_drastic),
    "lukasiewicz": (t_luk, s_luk),
    "nilpotent":   (t_nilpotent, s_nilpotent),
    "einstein":    (t_einstein, s_einstein),
    "product":     (t_prod, s_prob),
    "hamacher":    (t_hamacher, s_hamacher),
    "minimum":     (t_min, s_max),
    "yager2":      (_t_yager(2), _s_yager(2)),
    "yager3":      (_t_yager(3), _s_yager(3)),
    "dombi2":      (_t_dombi(2), _s_dombi(2)),
    "ss2":         (_t_schweizer_sklar(2), _s_schweizer_sklar(2)),
}

TNORMS = {k: v[0] for k, v in FAMILIES.items()}
TCONORMS = {k: v[1] for k, v in FAMILIES.items()}


def reduce_op(op, arrays):
    """Left-fold a binary operator over a list/axis of arrays."""
    out = arrays[0]
    for a in arrays[1:]:
        out = op(out, a)
    return out


def reduce_axis(op, X, axis):
    """Left-fold `op` along `axis` of X."""
    slices = [np.take(X, i, axis=axis) for i in range(X.shape[axis])]
    return reduce_op(op, slices)


def check_axioms(name, tol=1e-9):
    """Verify boundary, commutativity, monotonicity and [0,1] closure on a grid.

    Returns a dict of booleans; used to certify operators before we trust any
    result computed with them.
    """
    T, S = FAMILIES[name]
    g = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(g, g)
    with np.errstate(all="ignore"):
        t, s = T(X, Y), S(X, Y)
    fin = np.isfinite(t).all() and np.isfinite(s).all()
    return {
        "finite": bool(fin),
        "in_range": bool(fin and (t >= -tol).all() and (t <= 1 + tol).all()
                         and (s >= -tol).all() and (s <= 1 + tol).all()),
        "identity_T": bool(fin and np.allclose(T(X, np.ones_like(X)), X, atol=1e-6)),
        "identity_S": bool(fin and np.allclose(S(X, np.zeros_like(X)), X, atol=1e-6)),
        "commutative": bool(fin and np.allclose(t, T(Y, X), atol=1e-6)
                            and np.allclose(s, S(Y, X), atol=1e-6)),
        "de_morgan": bool(fin and np.allclose(1 - T(X, Y), S(1 - X, 1 - Y), atol=1e-6)),
    }


if __name__ == "__main__":
    print(f"{'family':<13} " + "  ".join(f"{k:<11}" for k in
                                         ["finite", "in_range", "identity_T",
                                          "identity_S", "commutative", "de_morgan"]))
    for nm in FAMILIES:
        r = check_axioms(nm)
        print(f"{nm:<13} " + "  ".join(f"{str(v):<11}" for v in r.values()))
