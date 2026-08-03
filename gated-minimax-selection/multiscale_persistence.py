"""
Option D: Multi-Scale Persistence Selection (density-stratified block cover).
================================================================================

Motivation
----------
The production selector `selection.select_coverage_cover` is a *flat* set-cover:
it greedily takes persistence-significant blocks until every point is covered,
then stops. On well-separated clusters -- even at extreme density contrast (see
`run_multiscale.robustness_flat_is_scale_invariant`) -- this is already
scale-invariant, because the minimax transform makes bottleneck heights
scale-relative and the MAD-based gate is robust to a few huge-persistence blocks.

So flat set-cover is NOT broken on single-level multi-scale data. What it
*cannot* represent is genuinely **nested** structure: a diffuse super-cluster
that itself contains tighter sub-clusters, where BOTH levels are meaningful.
Greedy coverage terminates at whichever granularity covers the data first
(usually the coarse one) and never descends. Empirically, on a 2x3 nested
Gaussian hierarchy, flat coverage_cover scores ARI 1.00 against the coarse
2-cluster ground truth but only ~0.32 against the fine 6-cluster ground truth --
it is forced to commit to one level.

The scale problem is therefore not "pick the one right scale" but "there is no
single right scale." This module treats the number of scales as an OUTPUT.

Key idea
--------
In a single-linkage dendrogram the *birth height* of a block is (inversely) a
measure of the local density at which that cluster becomes a distinct connected
component: dense clusters are born low, diffuse clusters are born high. A genuine
cluster generation therefore occupies a contiguous band of birth heights, and
successive generations are separated by gaps in the (log) birth-height axis --
the same gaps that appear as horizontal strata in a persistence barcode.

Algorithm
---------
  1. Enumerate every dendrogram block (birth, death, persistence, members).
  2. Persistence-significance gate: keep blocks whose absolute persistence is a
     MAD outlier above the background (identical statistic to coverage_cover --
     this module *generalizes*, not replaces, the gated-minimax gate).
  3. Discover scale bands: cut the log-birth axis of the significant blocks at
     gaps larger than `band_gap_factor x` the median gap. Each contiguous run of
     births is one scale band (one "generation" of the hierarchy).
  4. Within each band, run persistence-gated greedy set-cover restricted to that
     band's blocks -> one flat partition per scale.
  5. Post-filter bands that cover too little of the data (spurious noise strata).

The result is a *scale hierarchy*: an ordered list of `BandSelection`s, fine to
coarse. `flatten_to_level` picks one granularity when a flat partition is needed;
`assign` defuzzifies any single band's blocks to hard labels by minimax
proximity (the same rule as `ivat_mf.hard_labels_proximity`).

When the data has only one scale (the ordinary battery datasets), band discovery
returns a single band and the output is identical to `coverage_cover` -- so this
is a strict generalization with no regression on the flat case.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import selection as S


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------

@dataclass
class BandSelection:
    """One scale band's flat selection of blocks."""
    band_id: int
    log_birth_lo: float          # band edges in log birth-height
    log_birth_hi: float
    birth_lo: float              # same edges in raw height units
    birth_hi: float
    blocks: List[dict]           # selected blocks (selection._all_blocks dicts)
    n_candidates: int            # significant blocks available in this band

    @property
    def k(self) -> int:
        return len(self.blocks)

    @property
    def covered(self) -> set:
        cov: set = set()
        for b in self.blocks:
            cov |= b['members']
        return cov

    def coverage_fraction(self, n: int) -> float:
        return len(self.covered) / n if n else 0.0


@dataclass
class MultiScaleSelection:
    """The full scale hierarchy: bands ordered fine (low birth) -> coarse."""
    bands: List[BandSelection] = field(default_factory=list)
    n: int = 0
    n_significant: int = 0
    band_edges_log: List[float] = field(default_factory=list)

    @property
    def n_scales(self) -> int:
        return len(self.bands)

    def granularities(self) -> List[int]:
        """Cluster count discovered at each scale, fine -> coarse."""
        return [b.k for b in self.bands]

    def flatten_to_level(self, level: int) -> BandSelection:
        """Return the band at `level` (0 = finest). Negative indexes from coarse."""
        return self.bands[level]

    def finest(self) -> Optional[BandSelection]:
        return self.bands[0] if self.bands else None

    def coarsest(self) -> Optional[BandSelection]:
        return self.bands[-1] if self.bands else None


# ---------------------------------------------------------------------------
# significance gate (shared statistic with coverage_cover)
# ---------------------------------------------------------------------------

def persistence_significance_threshold(persist: np.ndarray, gap_sigma: float) -> float:
    """MAD-scaled outlier threshold: median + gap_sigma * (1.4826 * MAD).

    This is exactly the eligibility gate used in `selection.select_coverage_cover`
    (a block is 'real' iff its absolute persistence is a statistical outlier above
    the bulk of the persistence diagram). Factored out so both the flat and the
    multi-scale selector share one definition of significance.
    """
    med = np.median(persist)
    mad = np.median(np.abs(persist - med)) + 1e-12
    sigma = 1.4826 * mad
    return med + gap_sigma * sigma


def significant_blocks(blocks: Sequence[dict], n: int,
                       gap_sigma: float = 2.0,
                       min_size: int = 3,
                       max_size_frac: float = 0.6) -> List[dict]:
    """Blocks that pass the size window and the persistence-outlier gate.

    The MAD threshold is estimated over the *entire* block population (all sizes),
    exactly as `selection.select_coverage_cover` does, so that on single-band data
    this selector reproduces the flat baseline's eligible set (verified in
    `run_multiscale.no_regression_flat`). Estimating it over only the size-filtered
    subset would shift the threshold and silently diverge from the baseline.
    """
    if not blocks:
        return []
    persist_all = np.array([b['persistence'] for b in blocks])
    thr = persistence_significance_threshold(persist_all, gap_sigma)
    ceiling = max_size_frac * n
    return [b for b in blocks
            if min_size <= b['size'] <= ceiling and b['persistence'] >= thr]


# ---------------------------------------------------------------------------
# scale-band discovery
# ---------------------------------------------------------------------------

def discover_band_edges(sig_blocks: Sequence[dict],
                        band_gap_factor: float = 3.0,
                        min_log_gap: float = 0.5) -> List[float]:
    """Find scale-band boundaries as large gaps in the log birth-height axis.

    A boundary is placed at the midpoint of any consecutive-birth gap that is
    larger than both `band_gap_factor * median_gap` and `min_log_gap`. Working in
    log-height makes the criterion scale-relative: a factor-of-e jump in scale is
    treated the same whether it occurs at height 0.1 or height 100.

    Returns interior boundaries in log-birth units (sorted ascending); the caller
    prepends -inf and appends +inf.
    """
    if len(sig_blocks) < 2:
        return []
    log_births = np.sort(np.log(np.array([b['birth'] for b in sig_blocks]) + 1e-12))
    gaps = np.diff(log_births)
    if len(gaps) == 0:
        return []
    med_gap = np.median(gaps)
    thr = max(band_gap_factor * med_gap, min_log_gap)
    edges = [(log_births[i] + log_births[i + 1]) / 2.0
             for i in range(len(gaps)) if gaps[i] > thr]
    return edges


# ---------------------------------------------------------------------------
# per-band set-cover (same greedy rule as coverage_cover, scoped to a band)
# ---------------------------------------------------------------------------

def _greedy_cover(band_blocks: Sequence[dict]) -> List[dict]:
    """Greedy set-cover of the band's own point universe by uncovered-point gain,
    ties broken by higher persistence. Overlap tolerated (recombined downstream by
    the fixed t-conorm, per the coverage_cover rationale)."""
    if not band_blocks:
        return []
    universe: set = set().union(*[b['members'] for b in band_blocks])
    covered: set = set()
    sel: List[dict] = []
    chosen_ids = set()
    while covered != universe:
        best, best_gain = None, 0
        for b in band_blocks:
            if id(b) in chosen_ids:
                continue
            gain = len(b['members'] - covered)
            if gain > best_gain or (gain == best_gain and best is not None
                                    and b['persistence'] > best['persistence']):
                best, best_gain = b, gain
        if best is None or best_gain == 0:
            break
        sel.append(best)
        chosen_ids.add(id(best))
        covered |= best['members']
    return sel


# ---------------------------------------------------------------------------
# top-level multi-scale selector
# ---------------------------------------------------------------------------

def select_multiscale(Dstar: np.ndarray,
                      gap_sigma: float = 2.0,
                      max_size_frac: float = 0.6,
                      min_size: int = 3,
                      band_gap_factor: float = 3.0,
                      min_log_gap: float = 0.5,
                      min_band_coverage: float = 0.15,
                      merge_antichain: bool = True,
                      nest_frac_thresh: float = 0.5) -> MultiScaleSelection:
    """Discover the scale hierarchy of `Dstar` (minimax / iVAT distance matrix).

    Args:
        Dstar: n x n minimax-transformed dissimilarity matrix.
        gap_sigma: persistence-outlier gate strength (shared with coverage_cover).
        max_size_frac, min_size: block size window.
        band_gap_factor, min_log_gap: scale-band discovery sensitivity.
        min_band_coverage: drop bands whose selection covers less than this
            fraction of points (removes spurious noise strata).

    Returns:
        MultiScaleSelection with bands ordered fine (low birth) -> coarse.
    """
    blocks, n = S._all_blocks(Dstar)
    sig = significant_blocks(blocks, n, gap_sigma=gap_sigma,
                             min_size=min_size, max_size_frac=max_size_frac)
    if not sig:
        return MultiScaleSelection(bands=[], n=n, n_significant=0, band_edges_log=[])

    edges = discover_band_edges(sig, band_gap_factor=band_gap_factor,
                                min_log_gap=min_log_gap)
    full_edges = [-np.inf] + edges + [np.inf]

    # Raw candidate bands: the significant blocks whose birth falls in each
    # birth-gap interval. (Selection/cover happens after the merge pass.)
    raw = []
    for lo, hi in zip(full_edges[:-1], full_edges[1:]):
        cands = [b for b in sig if lo <= np.log(b['birth'] + 1e-12) < hi]
        if cands:
            raw.append({'lo': lo, 'hi': hi, 'cands': cands})

    # Containment-aware merge (Phase 4): a birth-gap is only a genuine SCALE
    # boundary if the coarser band's blocks are ANCESTORS of (contain) the finer
    # band's blocks. If instead adjacent bands are an antichain (disjoint
    # siblings -- same level, merely different spreads), the split is spurious;
    # merge them. This fixes the log-separated over-segmentation from the scaling
    # study (each cluster was landing in its own 1-block band) without touching
    # genuine nested hierarchies (where every finer block IS contained coarser).
    if merge_antichain:
        changed = True
        while changed and len(raw) > 1:
            changed = False
            for i in range(len(raw) - 1):
                fine = _greedy_cover(raw[i]['cands'])
                coarse = _greedy_cover(raw[i + 1]['cands'])
                if not fine or not coarse:
                    continue
                nested = sum(1 for fb in fine
                             if any(fb['members'] <= cb['members'] for cb in coarse))
                if nested / len(fine) < nest_frac_thresh:
                    raw[i] = {'lo': raw[i]['lo'], 'hi': raw[i + 1]['hi'],
                              'cands': raw[i]['cands'] + raw[i + 1]['cands']}
                    del raw[i + 1]
                    changed = True
                    break

    bands: List[BandSelection] = []
    for band in raw:
        lo, hi = band['lo'], band['hi']
        sel = _greedy_cover(band['cands'])
        bs = BandSelection(
            band_id=len(bands),
            log_birth_lo=lo, log_birth_hi=hi,
            birth_lo=float(np.exp(lo)) if lo > -np.inf else 0.0,
            birth_hi=float(np.exp(hi)) if hi < np.inf else float('inf'),
            blocks=sel,
            n_candidates=len(band['cands']),
        )
        if bs.coverage_fraction(n) >= min_band_coverage and bs.k >= 1:
            bs.band_id = len(bands)
            bands.append(bs)

    # Deduplicate consecutive bands that produced the identical block set (can
    # happen when a discovered gap does not actually change the cover).
    deduped: List[BandSelection] = []
    for bs in bands:
        sig_key = frozenset(frozenset(b['members']) for b in bs.blocks)
        if deduped and frozenset(
                frozenset(b['members']) for b in deduped[-1].blocks) == sig_key:
            continue
        bs.band_id = len(deduped)
        deduped.append(bs)

    # Drop single-block bands: a band with one cluster is a no-information
    # partition (everything -> one label), typically the near-root scale. Keep
    # multi-cluster bands; if that leaves nothing, keep the finest single band so
    # the result is never empty when structure exists.
    informative = [bs for bs in deduped if bs.k >= 2]
    if not informative and deduped:
        informative = deduped[:1]
    for i, bs in enumerate(informative):
        bs.band_id = i

    return MultiScaleSelection(bands=informative, n=n, n_significant=len(sig),
                               band_edges_log=edges)


# ---------------------------------------------------------------------------
# defuzzification / evaluation helpers
# ---------------------------------------------------------------------------

def assign(blocks: Sequence[dict], Dstar: np.ndarray) -> np.ndarray:
    """Hard-assign every point to the block it is closest to in minimax distance.

    Mirrors `ivat_mf.hard_labels_proximity`: distance to a block is the minimum
    minimax distance to any of its members. Points get the label of the nearest
    block core. Returns an int label array of length n (all zeros if no blocks).
    """
    n = Dstar.shape[0]
    if not blocks:
        return np.zeros(n, dtype=int)
    # (k, n) matrix of minimax distance from each block to each point.
    dist = np.empty((len(blocks), n))
    for k, b in enumerate(blocks):
        mem = np.fromiter(b['members'], dtype=int)
        dist[k] = Dstar[mem, :].min(axis=0)
    return np.argmin(dist, axis=0)


def assign_band(band: BandSelection, Dstar: np.ndarray) -> np.ndarray:
    return assign(band.blocks, Dstar)


# ---------------------------------------------------------------------------
# Phase 1: direct fuzzy membership functions (persistence ramp, per block)
#
# A dendrogram block already carries a native graded membership -- the Mapping-2
# persistence ramp (ivat_mf.mapping2_persistence), built from the block's own
# birth/death heights with no medoid or Gaussian fit:
#
#     d_B(x) = min_{y in B} D*(x, y)                    minimax distance to block
#     mu_B(x) = clip( (death_B - d_B(x)) / (death_B - birth_B), 0, 1 )
#             = 1              for x in the core (d_B <= birth_B)
#
# So membership generation and selection share the same two numbers. This block
# turns a discovered scale band into a *fuzzy* partition (U, shape k_band x n)
# directly, rather than the hard argmin-distance label assign_band returns.
# ---------------------------------------------------------------------------

def block_membership(block: dict, Dstar: np.ndarray,
                     kernel: str = 'gaussian') -> np.ndarray:
    """Membership mu_B(x) in [0,1] for one block, length n, from the minimax
    distance d_B(x) = min_{y in B} D*(x, y).

    kernel:
      'ramp'     -- Phase 1 persistence ramp clip((death-d)/(death-birth),0,1),
                    core->1. CRISP by construction (no point has birth<d<death;
                    see notes/MF_PROGRESS_LOG.md Phase 1). Kept for the record.
      'gaussian' -- Phase 2 (default): mu = 2**(-(d/death)**2), i.e. a Gaussian in
                    minimax distance with HALF-MAX at the block's death (escape)
                    height -- the scale at which the block dissolves into its
                    parent. Members (d=0) read 1; the non-member skirt (d>=death)
                    is graded 0.5 -> 0. This is what makes the MF genuinely fuzzy;
                    argmax still reproduces the crisp labels.
    """
    mem = np.fromiter(block['members'], dtype=int)
    d = Dstar[mem, :].min(axis=0)
    h_b, h_d = block['birth'], block['death']
    if kernel == 'ramp':
        mu = np.clip((h_d - d) / (h_d - h_b + 1e-12), 0.0, 1.0)
        mu[d <= h_b + 1e-12] = 1.0
        return mu
    if kernel == 'gaussian':
        return np.exp(-np.log(2.0) * (d / (h_d + 1e-12)) ** 2)
    raise ValueError(f"unknown kernel {kernel!r}")


def band_memberships(band: BandSelection, Dstar: np.ndarray,
                     kernel: str = 'gaussian') -> np.ndarray:
    """Fuzzy partition (k_band x n) for one scale band: one MF per block."""
    n = Dstar.shape[0]
    if not band.blocks:
        return np.zeros((0, n))
    return np.vstack([block_membership(b, Dstar, kernel=kernel) for b in band.blocks])


def defuzzify_memberships(U: np.ndarray, band: BandSelection,
                          Dstar: np.ndarray) -> np.ndarray:
    """Hard labels from a fuzzy partition: argmax membership, ties (common at the
    saturated value 1.0, where several block cores overlap a point) broken by
    minimax proximity to the block -- the ivat_mf.hard_labels_proximity rule."""
    if U.shape[0] == 0:
        return np.zeros(U.shape[1], dtype=int)
    n = U.shape[1]
    dist = np.vstack([Dstar[np.fromiter(b['members'], dtype=int), :].min(axis=0)
                      for b in band.blocks])
    labels = np.empty(n, dtype=int)
    umax = U.max(axis=0)
    for i in range(n):
        top = np.where(U[:, i] >= umax[i] - 1e-9)[0]
        labels[i] = int(top[np.argmin(dist[top, i])]) if len(top) > 1 else int(top[0])
    return labels


def multiscale_memberships(Dstar: np.ndarray, kernel: str = 'gaussian', **kwargs):
    """Run multi-scale selection and emit a fuzzy partition per scale band.

    Returns (MultiScaleSelection, [U_band, ...]) where each U_band is a
    (k_band x n) matrix of memberships (default Gaussian kernel), fine -> coarse.
    """
    msel = select_multiscale(Dstar, **kwargs)
    return msel, [band_memberships(b, Dstar, kernel=kernel) for b in msel.bands]


# ---------------------------------------------------------------------------
# Phase 3: partition-of-unity per scale + the multi-scale fuzzy model
# ---------------------------------------------------------------------------

def normalize_partition(U: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Ruspini normalization: scale each point's memberships to sum to 1.
    Points with ~0 total membership (uncovered at this scale) are left all-zero
    -- a deliberate possibilistic choice: 'belongs to nothing here' is
    information, not something to smear into a uniform guess. Column-positive
    scaling never changes argmax, so hard labels are unaffected."""
    s = U.sum(axis=0, keepdims=True)
    return np.where(s > eps, U / np.where(s > eps, s, 1.0), 0.0)


@dataclass
class FuzzyHierarchy:
    """The multi-scale fuzzy model: one fuzzy partition per discovered scale,
    fine -> coarse. `U[i]` is a (k_i x n) membership matrix for band i."""
    bands: List[BandSelection]
    U: List[np.ndarray]
    normalized: bool
    kernel: str

    @property
    def n_scales(self) -> int:
        return len(self.bands)

    def granularities(self) -> List[int]:
        return [u.shape[0] for u in self.U]

    def level(self, i: int) -> np.ndarray:
        return self.U[i]

    def defuzzify(self, i: int, Dstar: np.ndarray) -> np.ndarray:
        return defuzzify_memberships(self.U[i], self.bands[i], Dstar)

    def partition_of_unity_error(self, i: int) -> Tuple[float, float]:
        """(max, mean) |sum_k mu_k(x) - 1| over COVERED points (those with any
        membership). Uncovered points are excluded -- they are legitimately 0."""
        s = self.U[i].sum(axis=0)
        covered = s > 1e-9
        if not covered.any():
            return 0.0, 0.0
        err = np.abs(s[covered] - 1.0)
        return float(err.max()), float(err.mean())

    def coverage(self, i: int) -> float:
        return float(np.mean(self.U[i].sum(axis=0) > 1e-9))


def build_fuzzy_hierarchy(Dstar: np.ndarray, kernel: str = 'gaussian',
                          normalize: bool = True, **kwargs) -> FuzzyHierarchy:
    """Build the multi-scale fuzzy model: discover scale bands, emit a kernel
    membership partition per band, and (optionally) Ruspini-normalize each to a
    partition of unity. Hard labels (argmax) are identical with or without
    normalization."""
    msel, Us = multiscale_memberships(Dstar, kernel=kernel, **kwargs)
    if normalize:
        Us = [normalize_partition(u) for u in Us]
    return FuzzyHierarchy(bands=msel.bands, U=Us, normalized=normalize, kernel=kernel)


# ---------------------------------------------------------------------------
# quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import ivat_mf as im
    import battery_hierarchical as BH
    from sklearn.metrics import adjusted_rand_score

    X, y_fine, y_coarse = BH.nested_gaussians()
    Dstar = im.minimax_transform(im.dissimilarity(X))
    msel = select_multiscale(Dstar)

    print(f"discovered {msel.n_scales} scales, granularities {msel.granularities()}")
    _, Us = multiscale_memberships(Dstar)
    for band, U in zip(msel.bands, Us):
        a = assign_band(band, Dstar)
        graded = float(np.mean((U > 1e-6) & (U < 1 - 1e-6)))
        print(f"  band {band.band_id}: k={band.k} "
              f"births[{band.birth_lo:.2f},{band.birth_hi:.2f}] "
              f"cov={band.coverage_fraction(msel.n):.2f} "
              f"ARI_fine={adjusted_rand_score(y_fine, a):.3f} "
              f"ARI_coarse={adjusted_rand_score(y_coarse, a):.3f} "
              f"graded_frac={graded:.3f}")
    print("(Phase 1 finding: graded_frac == 0 -- the birth/death ramp is crisp by "
          "construction; see notes/MF_PROGRESS_LOG.md)")
