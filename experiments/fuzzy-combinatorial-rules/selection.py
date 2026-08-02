"""Combinatorial selectors: which membership functions belong to which rule.

Every selector answers the same question for one class c -- given the (n, d, k)
membership tensor and the one-vs-rest labelling, choose a boolean (d, k) mask --
and every one of them tunes its own hyper-parameters against the *training*
objective ``rule_objective`` only. Nothing here sees the test split.

The search space per class is ``(2^k - 1)^d`` and, for the model as a whole,
``(2^k - 1)^(d*C)``. For iris at k = 5 that is 31^4 = 923 521 per class and
about 7.9e17 for the model, so the point of the exercise is how close cheap
structure-driven choices get to the per-class optimum that ``exhaustive`` can
still compute at the small end.

Selectors
---------
``mass``       marginal MF mass per class, thresholded (the Wang-Mendel-style
               baseline: no interaction between variables at all).
``greedy``     steepest-ascent hill climb from all-don't-care, single MF flips.
``mst_mf``     MST over the d*k *membership functions*, edges weighted by how
               much two MFs co-fire on the class; single-linkage cut, keep the
               component carrying the most class-discriminative mass.
``mst_core``   MST over the class's *samples* in membership space; single-linkage
               cut discards straggler components, and the surviving core defines
               the MF mass. Outlier-robust support estimation.
``anneal``     simulated annealing from the best of the above.
``exhaustive`` every subset combination, when the space is small enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from rules import Evaluator, is_convex

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

ALPHA_GRID: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
QUANTILE_GRID: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
EXHAUSTIVE_BUDGET = 2_000_000  # subset combinations per class
_EPS = 1e-9


@dataclass
class Problem:
    """One class's selection problem."""

    m: Array  # (n, d, k) training memberships
    in_class: NDArray[np.bool_]  # (n,)
    lam: float = 1.0
    tnorm: str = "min"
    disjunction: str = "sum"
    seed: int = 0
    # Restrict every antecedent to a contiguous run of MFs. This is the
    # classical linguistic constraint, and it shrinks each variable's choice
    # from 2^k - 1 subsets to k(k+1)/2 intervals.
    convex: bool = False
    ev: Evaluator = field(init=False)

    def __post_init__(self) -> None:
        self.ev = Evaluator(self.m, self.in_class, self.lam, self.tnorm, self.disjunction)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.m.shape[0]), int(self.m.shape[1]), int(self.m.shape[2]))

    def score(self, s: BoolArray) -> float:
        return self.ev.score_subset(s)

    def repair(self, s: BoolArray) -> BoolArray:
        """Project onto the feasible set (a no-op unless ``convex``)."""
        return convex_hull_rows(s) if self.convex else s


def convex_hull_rows(s: BoolArray) -> BoolArray:
    """Fill the holes in every row, so each antecedent becomes one interval."""
    out = np.zeros_like(s)
    for i, row in enumerate(s):
        idx = np.flatnonzero(row)
        if idx.size:
            out[i, idx[0]:idx[-1] + 1] = True
        else:
            out[i, :] = True
    return out


# --------------------------------------------------------------------------
# shared pieces
# --------------------------------------------------------------------------
def _mass_to_subsets(mass: Array, alpha: float) -> BoolArray:
    """Per variable, take MFs in descending mass until they cover ``alpha`` of it.

    Always returns at least one MF per variable (an empty antecedent would zero
    the rule everywhere), and a full row is a don't-care.
    """
    d, k = mass.shape
    s = np.zeros((d, k), dtype=bool)
    for i in range(d):
        row = mass[i]
        total = row.sum()
        if total <= 0.0:
            s[i, :] = True  # variable carries no mass for this class: don't care
            continue
        order = np.argsort(-row)
        cum = np.cumsum(row[order]) / total
        take = int(np.searchsorted(cum, alpha - 1e-12) + 1)
        s[i, order[:take]] = True
    return s


def _best_over_alpha(prob: Problem, mass: Array) -> tuple[BoolArray, float]:
    best_s = np.ones(mass.shape, dtype=bool)
    best_j = -np.inf
    for alpha in ALPHA_GRID:
        s = prob.repair(_mass_to_subsets(mass, alpha))
        j = prob.score(s)
        if j > best_j:
            best_j, best_s = j, s
    return best_s, best_j


def class_mass(prob: Problem, rows: NDArray[np.int64] | None = None) -> Array:
    """(d, k) total membership mass of the class's samples (or a subset of them)."""
    idx = np.flatnonzero(prob.in_class) if rows is None else rows
    if idx.size == 0:
        return np.zeros(prob.m.shape[1:], dtype=np.float64)
    return prob.m[idx].sum(axis=0)


# --------------------------------------------------------------------------
# 1. marginal mass baseline
# --------------------------------------------------------------------------
def select_mass(prob: Problem) -> BoolArray:
    return _best_over_alpha(prob, class_mass(prob))[0]


# --------------------------------------------------------------------------
# 2. steepest-ascent hill climb over single MF flips
# --------------------------------------------------------------------------
def select_greedy(prob: Problem, s0: BoolArray | None = None) -> BoolArray:
    _, d, k = prob.shape
    s = np.ones((d, k), dtype=bool) if s0 is None else s0.copy()
    a = prob.ev.activation_matrix(s)
    cur = prob.ev.score(a)
    while True:
        best_gain, best_move, best_col = 0.0, None, None
        for i in range(d):
            for j in range(k):
                if s[i, j] and s[i].sum() == 1:
                    continue  # cannot empty a variable
                row = s[i].copy()
                row[j] = not row[j]
                if prob.convex and not is_convex(row):
                    continue
                col = prob.ev.column(row, i)
                trial = a.copy()
                trial[:, i] = col
                gain = prob.ev.score(trial) - cur
                if gain > best_gain + 1e-12:
                    best_gain, best_move, best_col = gain, (i, j), col
        if best_move is None:
            return s
        i, j = best_move
        s[i, j] = not s[i, j]
        a[:, i] = best_col  # type: ignore[assignment]
        cur += best_gain


# --------------------------------------------------------------------------
# 3. MST over the membership functions themselves
# --------------------------------------------------------------------------
def _cofiring_distance(mc: Array) -> Array:
    """1 - fuzzy Jaccard between every pair of MFs, over the class's samples.

    ``mc`` is (n_c, N) with N = d*k. Two MFs are close when they light up on the
    same class-c samples; MFs on the same variable are near-orthogonal under a
    Ruspini partition, so the tree naturally wires *across* variables.
    """
    n_c, n_nodes = mc.shape
    smin = np.empty((n_nodes, n_nodes), dtype=np.float64)
    smax = np.empty((n_nodes, n_nodes), dtype=np.float64)
    block = max(1, int(5e7 // max(n_nodes * n_nodes, 1)))
    smin[:] = 0.0
    smax[:] = 0.0
    for start in range(0, n_c, block):
        chunk = mc[start:start + block]
        pair_a = chunk[:, :, None]
        pair_b = chunk[:, None, :]
        smin += np.minimum(pair_a, pair_b).sum(axis=0)
        smax += np.maximum(pair_a, pair_b).sum(axis=0)
    jac = np.where(smax > 0.0, smin / np.maximum(smax, _EPS), 0.0)
    dist = 1.0 - jac
    np.fill_diagonal(dist, 0.0)
    return dist


def _mst_edges(dist: Array) -> tuple[NDArray[np.int64], NDArray[np.int64], Array]:
    """MST edge list of a dense symmetric distance matrix, sorted by weight."""
    # scipy's sparse graphs read 0 as "no edge", so nudge weights off zero.
    graph = dist + _EPS
    np.fill_diagonal(graph, 0.0)
    mst = minimum_spanning_tree(csr_matrix(graph)).tocoo()
    order = np.argsort(mst.data)
    return mst.row[order].astype(np.int64), mst.col[order].astype(np.int64), mst.data[order]


def select_mst_mf(prob: Problem) -> BoolArray:
    n, d, k = prob.shape
    n_nodes = d * k
    flat = prob.m.reshape(n, n_nodes)
    mc = flat[prob.in_class]
    if mc.shape[0] < 2:
        return select_mass(prob)

    # Node benefit: fuzzy precision above the class prior, times fuzzy support.
    prior = float(prob.in_class.mean())
    total = flat.sum(axis=0)
    precision = np.where(total > 0.0, mc.sum(axis=0) / np.maximum(total, _EPS), 0.0)
    support = mc.mean(axis=0)
    benefit = (precision - prior) * support

    rows, cols, w = _mst_edges(_cofiring_distance(mc))
    best_s = np.ones((d, k), dtype=bool)
    best_j = prob.score(best_s)
    # Single-linkage: sweep the cut level over the MST's own edge weights, which
    # walks the whole dendrogram from d*k singletons up to one component.
    for cut in np.unique(w):
        keep = w <= cut
        sub = csr_matrix(
            (np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n_nodes, n_nodes)
        )
        _, comp = connected_components(sub, directed=False)
        scores = np.bincount(comp, weights=benefit, minlength=comp.max() + 1)
        members = np.flatnonzero(comp == int(np.argmax(scores)))
        s = np.zeros((d, k), dtype=bool)
        s.reshape(-1)[members] = True
        empty = ~s.any(axis=1)
        s[empty, :] = True  # variable absent from the component -> don't care
        s = prob.repair(s)
        j = prob.score(s)
        if j > best_j:
            best_j, best_s = j, s
    return best_s


# --------------------------------------------------------------------------
# 4. MST over the class's samples -> outlier-robust core -> MF mass
# --------------------------------------------------------------------------
def select_mst_core(prob: Problem) -> BoolArray:
    n, d, k = prob.shape
    idx = np.flatnonzero(prob.in_class)
    if idx.size < 4:
        return select_mass(prob)
    phi = prob.m[idx].reshape(idx.size, d * k)
    dist = squareform(pdist(phi))
    rows, cols, w = _mst_edges(dist)
    n_c = idx.size

    best_s = np.ones((d, k), dtype=bool)
    best_j = prob.score(best_s)
    min_frac = 0.15
    for q in QUANTILE_GRID:
        cut = float(np.quantile(w, q)) if q < 1.0 else float(w.max())
        keep = w <= cut
        sub = csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n_c, n_c))
        _, comp = connected_components(sub, directed=False)
        sizes = np.bincount(comp)
        big = np.flatnonzero(sizes >= max(2, int(min_frac * n_c)))
        core_local = (
            np.flatnonzero(np.isin(comp, big))
            if big.size
            else np.flatnonzero(comp == int(np.argmax(sizes)))
        )
        mass = class_mass(prob, idx[core_local])
        s, j = _best_over_alpha(prob, mass)
        if j > best_j:
            best_j, best_s = j, s
    return best_s


# --------------------------------------------------------------------------
# 5. simulated annealing refinement
# --------------------------------------------------------------------------
def refine_anneal(prob: Problem, s0: BoolArray, iters: int = 4000) -> BoolArray:
    _, d, k = prob.shape
    rng = np.random.default_rng(prob.seed)
    s = s0.copy()
    a = prob.ev.activation_matrix(s)
    cur = prob.ev.score(a)
    best_s, best_j = s.copy(), cur
    t0, t1 = 0.05, 1e-4
    for step in range(iters):
        temp = t0 * (t1 / t0) ** (step / max(iters - 1, 1))
        i = int(rng.integers(d))
        j = int(rng.integers(k))
        if s[i, j] and s[i].sum() == 1:
            continue
        row = s[i].copy()
        row[j] = not row[j]
        if prob.convex and not is_convex(row):
            continue
        col = prob.ev.column(row, i)
        trial = a.copy()
        trial[:, i] = col
        cand = prob.ev.score(trial)
        if cand > cur or rng.random() < np.exp((cand - cur) / temp):
            s[i, j] = not s[i, j]
            a[:, i] = col
            cur = cand
            if cur > best_j:
                best_j, best_s = cur, s.copy()
    return best_s


# --------------------------------------------------------------------------
# 6. exhaustive enumeration
# --------------------------------------------------------------------------
def n_subsets(k: int, convex: bool = False) -> int:
    """Choices per variable: all nonempty subsets, or only the intervals."""
    return k * (k + 1) // 2 if convex else 2**k - 1


def exhaustive_feasible(
    d: int, k: int, budget: int = EXHAUSTIVE_BUDGET, convex: bool = False
) -> bool:
    try:
        return n_subsets(k, convex) ** d <= budget
    except OverflowError:  # pragma: no cover - only for absurd k
        return False


def select_exhaustive(
    prob: Problem, budget: int = EXHAUSTIVE_BUDGET, chunk: int = 8192
) -> BoolArray | None:
    """Global optimum of the one-vs-rest objective, or None if out of budget."""
    n, d, k = prob.shape
    if not exhaustive_feasible(d, k, budget, prob.convex):
        return None
    bits = _nonempty_masks(k, prob.convex)
    p = int(bits.shape[0])
    # (d, p, n): activation of every nonempty subset of every variable.
    per_var = np.stack([np.stack([_var_activation(prob, i, mask) for mask in bits])
                        for i in range(d)])
    total = p**d
    var_axis = np.arange(d)
    best_j, best_codes = -np.inf, np.zeros(d, dtype=np.int64)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        codes = np.stack(np.unravel_index(np.arange(start, stop), (p,) * d), axis=1)
        vals = combine_axis0(per_var[var_axis[None, :], codes].transpose(1, 0, 2), prob.tnorm)
        pos = vals[:, prob.in_class].mean(axis=1)
        neg = vals[:, ~prob.in_class].mean(axis=1)
        j = pos - prob.lam * neg
        hit = int(np.argmax(j))
        if j[hit] > best_j:
            best_j, best_codes = float(j[hit]), codes[hit]
    return bits[best_codes]


def _nonempty_masks(k: int, convex: bool = False) -> BoolArray:
    """Every nonempty subset of a variable's MFs (or only the intervals)."""
    codes = np.arange(1, 2**k, dtype=np.int64)[:, None]
    masks = ((codes >> np.arange(k)) & 1).astype(bool)
    if convex:
        masks = masks[[is_convex(row) for row in masks]]
    return masks


def _var_activation(prob: Problem, i: int, mask: BoolArray) -> Array:
    if prob.disjunction == "sum":
        return np.clip(prob.m[:, i, :] @ mask.astype(np.float64), 0.0, 1.0)
    return np.max(np.where(mask[None, :], prob.m[:, i, :], 0.0), axis=1)


def combine_axis0(a: Array, tnorm: str) -> Array:
    """t-norm over axis 0 of a (d, B, n) block -> (B, n)."""
    if tnorm == "min":
        return np.min(a, axis=0)
    if tnorm == "product":
        return np.prod(a, axis=0)
    raise ValueError(f"unknown t-norm {tnorm!r}")


# --------------------------------------------------------------------------
SELECTORS = {
    "mass": select_mass,
    "greedy": select_greedy,
    "mst_mf": select_mst_mf,
    "mst_core": select_mst_core,
}


def select(name: str, prob: Problem) -> BoolArray | None:
    """Run one named selector (``anneal`` and ``exhaustive`` are handled here)."""
    if name in SELECTORS:
        return SELECTORS[name](prob)
    if name == "anneal":
        candidates = [(prob.score(s), s) for s in (fn(prob) for fn in SELECTORS.values())]
        s0 = max(candidates, key=lambda t: t[0])[1]
        return refine_anneal(prob, s0)
    if name == "exhaustive":
        return select_exhaustive(prob)
    raise ValueError(f"unknown selector {name!r}")


def all_subset_scores(prob: Problem, budget: int = EXHAUSTIVE_BUDGET) -> Array | None:
    """Objective value of *every* subset combination -- for landscape statistics."""
    n, d, k = prob.shape
    if not exhaustive_feasible(d, k, budget, prob.convex):
        return None
    bits = _nonempty_masks(k, prob.convex)
    p = int(bits.shape[0])
    per_var = np.stack([np.stack([_var_activation(prob, i, mask) for mask in bits])
                        for i in range(d)])
    out = np.empty(p**d, dtype=np.float64)
    var_axis = np.arange(d)
    for start in range(0, p**d, 8192):
        stop = min(start + 8192, p**d)
        codes = np.stack(np.unravel_index(np.arange(start, stop), (p,) * d), axis=1)
        vals = combine_axis0(per_var[var_axis[None, :], codes].transpose(1, 0, 2), prob.tnorm)
        out[start:stop] = (
            vals[:, prob.in_class].mean(axis=1) - prob.lam * vals[:, ~prob.in_class].mean(axis=1)
        )
    return out


__all__ = [
    "Problem",
    "SELECTORS",
    "select",
    "select_exhaustive",
    "exhaustive_feasible",
    "all_subset_scores",
    "refine_anneal",
]
