"""
Coverage-driven block selection.

Key insight (from TRIBBLE's t-conorm recombination): over-segmentation is cheap
-- redundant/overlapping blocks over the same region can be combined by the
fixed t-conorm. What is EXPENSIVE is leaving a genuine population uncovered, or
spending a selection on a noise sliver.

So selection is reframed from "pick exactly c disjoint clusters" (which starves
clusters living at unlucky height scales) to a SET-COVER over the dendrogram:
choose a set of persistent blocks that covers as many points as possible at
each point's OWN natural scale, tolerating overlap.

We compare three selectors on varying_density (the case that broke before):

  topc_disjoint  : old approach - top-c by persistence, disjoint. (baseline)
  relpersist     : rank by death/birth ratio, disjoint, size-ceiling. (prev best)
  coverage_cover : greedy set-cover by persistence-eligible blocks until all
                   points covered; c is an OUTPUT, not an input.
"""

import numpy as np
import ivat_mf as im
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _all_blocks(Dstar):
    """Enumerate every internal node of the SL dendrogram as a candidate block,
    with birth (formation height), death (parent merge height), members."""
    n = Dstar.shape[0]
    Z = linkage(squareform(Dstar, checks=False), method='single')
    members = {i: {i} for i in range(n)}
    heights = {i: 0.0 for i in range(n)}
    parent_h = {}
    nodes = []
    for r, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b)
        nid = n + r
        members[nid] = members[a] | members[b]
        heights[nid] = h
        parent_h[a] = h
        parent_h[b] = h
        nodes.append(nid)
    root = n + len(Z) - 1
    parent_h[root] = Z[-1, 2] * 1.5 + 1e-9
    blocks = []
    for nid in nodes:
        if len(members[nid]) < 2:
            continue
        birth = heights[nid]
        death = parent_h.get(nid, birth)
        blocks.append({
            'node_id': nid, 'members': members[nid],
            'birth': birth, 'death': death,
            'persistence': death - birth,
            'rel_persistence': death / (birth + 1e-12),
            'size': len(members[nid]),
        })
    return blocks, n


def select_topc_disjoint(Dstar, c):
    blocks, n = _all_blocks(Dstar)
    blocks = [b for b in blocks if b['size'] >= 2]
    blocks.sort(key=lambda b: b['persistence'], reverse=True)
    sel = []
    for b in blocks:
        if any(b['members'] & s['members'] for s in sel):
            continue
        sel.append(b)
        if len(sel) == c:
            break
    return sel


def select_relpersist(Dstar, c):
    blocks, n = _all_blocks(Dstar)
    ceiling = 0.9 * n
    blocks = [b for b in blocks if 2 <= b['size'] <= ceiling]
    blocks.sort(key=lambda b: b['rel_persistence'], reverse=True)
    sel = []
    for b in blocks:
        if any(b['members'] & s['members'] for s in sel):
            continue
        sel.append(b)
        if len(sel) == c:
            break
    return sel


def select_coverage_cover(Dstar, gap_sigma=2.0, max_size_frac=0.6):
    """
    Greedy set-cover with a PERSISTENCE-GAP eligibility gate.

    Finding from the data: relative persistence (death/birth) cannot separate
    real structure from noise -- both had near-identical rel-persistence
    distributions. The signal that DOES separate them is absolute persistence
    standing out from the background: genuine clusters produce a few blocks
    whose persistence is a statistical outlier above the bulk, while noise has a
    diffuse persistence distribution with no standouts.

    Eligibility: a block is eligible iff its absolute persistence exceeds
    (median + gap_sigma * MAD-scaled-sigma) of the persistence distribution --
    i.e. it is an outlier in the persistence diagram. If NO block clears the
    gap, we select nothing (correctly declining to assert structure in noise).

    Among eligible blocks (size-capped so the near-root block is excluded),
    greedy set-cover by uncovered-point gain, overlap allowed.
    """
    blocks, n = _all_blocks(Dstar)
    ceiling = max_size_frac * n
    persist = np.array([b['persistence'] for b in blocks])
    med = np.median(persist)
    mad = np.median(np.abs(persist - med)) + 1e-12
    sigma = 1.4826 * mad  # MAD -> sigma for normal
    gap_threshold = med + gap_sigma * sigma

    elig = [b for b in blocks
            if b['size'] >= 3
            and b['size'] <= ceiling
            and b['persistence'] >= gap_threshold]

    if not elig:
        # No persistence outliers => no asserted structure.
        return []

    covered = set()
    sel = []
    all_pts = set(range(n))
    while covered != all_pts:
        best, best_gain = None, 0
        for b in elig:
            if b in sel:
                continue
            gain = len(b['members'] - covered)
            if gain > best_gain or (gain == best_gain and best is not None
                                    and b['persistence'] > best['persistence']):
                best, best_gain = b, gain
        if best is None or best_gain == 0:
            break
        sel.append(best)
        covered |= best['members']
    return sel


def coverage_of(sel, n):
    cov = set()
    for b in sel:
        cov |= b['members']
    return len(cov) / n


def purity_vs_truth(sel, y):
    """For each selected block, the dominant true label fraction among members
    (ignoring bridge/noise label -1). Reports mean purity -- how clean the
    selected blocks are as cluster proxies."""
    purities = []
    for b in sel:
        mem = np.array(sorted(b['members']), dtype=int)
        labs = y[mem]
        labs = labs[labs >= 0]
        if len(labs) == 0:
            continue
        counts = np.bincount(labs)
        purities.append(counts.max() / counts.sum())
    return np.mean(purities) if purities else np.nan
