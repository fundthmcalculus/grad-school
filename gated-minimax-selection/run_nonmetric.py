"""Non-Euclidean / non-metric extension of the gated-minimax investigation.

Standalone driver (run_all.py, the Chapter 5 driver, is deliberately not
touched): `python run_nonmetric.py` regenerates every number and figure here
deterministically from seed and writes

  - outputs/nonmetric_results.json  : all numeric results, one source of truth
    (written BEFORE the figures, so a figure crash cannot discard the numeric
    phase -- the failure mode AGENTS.md documents for run_all.py)
  - outputs/fig12_nonmetric_diagnostics.png
  - outputs/fig13_nonmetric_battery.png
  - outputs/fig14_violation_sweep.png
  - outputs/fig15_relational_multiscale.png

The five experiments:

E1 Diagnostics -- makes "non-Euclidean" quantitative. For each dataset:
   triangle-violation fraction and classical-MDS negative-eigenvalue ratio of
   raw D and of D* = minimax(D); an ultrametricity check of D*; and whether
   NERFCM's beta-spread ever activates on either. The theory being tested:
   D* of ANY symmetric dissimilarity is an ultrametric, finite ultrametrics
   are isometrically Euclidean-embeddable (Lemin), therefore D* should always
   be admissible (neg-eig ~ 0, beta never fires) -- the minimax transform is
   a canonical Euclideanizer, subsuming the beta-spread repair.

E2 Method battery -- NERFCM(D), NERFCM(D*), single-linkage at true k, and the
   three k-discovering selectors (persistence-gap cover, beta-plateau, and a
   relational bottleneck-bootstrap that resamples the matrix rather than
   coordinates) on five genuinely non-Euclidean dissimilarity families.

E3 Violation sweep -- corrupt a Euclidean 3-blob base by stretching (inflating
   D_ij: breaks embeddability, leaves small edges alone) or shortcutting
   (deflating D_ij: manufactures single-linkage bridges) at controlled rate
   and strength. Prediction: the two corruption directions hurt DIFFERENT
   methods -- D* shrugs off stretch and collapses under shortcut, relational
   averaging (NERFCM on raw D) degrades gracefully under both.

E4 Relational multi-scale -- Option D's select_multiscale on distance-matrix-
   only nested hierarchies, including the multi_scale_hierarchy dataset whose
   NERFCM ARI of 0.29 has stood in notes/RELATIONDATA.md as an open problem,
   plus a clean two-level relational hierarchy and non-metric variants of it.

E5 Real-data DTW -- the battery on DTW over N-CMAPSS DS01 flight altitude
   profiles (truth = flight class; DS02's dev units are all one class).
   Skipped gracefully when the gitignored .h5 is absent. The one dataset in
   the study where beta-spread activates naturally on raw D.
"""

from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

import ivat_mf as im
import selection as S
import selection_comparison as SC
import multiscale_persistence as MS
import nonmetric_data as ND
import relationdata as RD
from nerfcm import nerfcm

OUT = "./outputs"
SEEDS = [0, 1, 2, 3, 4]  # NERFCM restart seeds, matching run_all.py
DATASET_SEEDS = [0, 1, 2]  # replicate generators for the violation sweep

results: dict = {}


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------


def nerfcm_score(D, y, c):
    """(mean ARI, std ARI, max beta) over the NERFCM restart seeds."""
    aris, betas = [], []
    for s in SEEDS:
        U, beta, _ = nerfcm(D, c, seed=s)
        aris.append(adjusted_rand_score(y, np.argmax(U, axis=0)))
        betas.append(beta)
    return float(np.mean(aris)), float(np.std(aris)), float(np.max(betas))


def sl_ari_at_k(Dstar, y, k):
    """Single-linkage cut at the true k -- the no-selection baseline."""
    Z = linkage(squareform(Dstar, checks=False), method="single")
    lab = fcluster(Z, t=k, criterion="maxclust") - 1
    return float(adjusted_rand_score(y, lab))


def blocks_to_labels(Dstar, sel):
    """Assign every point to its nearest selected block (minimax distance),
    the same defuzzification run_all.py's cover_result uses."""
    n = Dstar.shape[0]
    Db = np.zeros((len(sel), n))
    for k, b in enumerate(sel):
        mem = np.array(sorted(b["members"]), dtype=int)
        Db[k] = Dstar[:, mem].min(axis=1)
    return np.argmin(Db, axis=0)


def score_selection(Dstar, y, sel):
    """(k_discovered, coverage, ARI-or-None) for a selected block set."""
    if not sel:
        return 0, 0.0, None
    lab = blocks_to_labels(Dstar, sel)
    return (
        len(sel),
        float(S.coverage_of(sel, Dstar.shape[0])),
        float(adjusted_rand_score(y, lab)),
    )


def select_bottleneck_bootstrap_relational(
    D,
    n_boots: int = 100,
    boot_frac: float = 0.8,
    max_size_frac: float = 0.6,
    seed: int = 42,
):
    """Relational mirror of SC.select_bottleneck_bootstrap.

    The coordinate version resamples points and recomputes distances; with
    only a matrix available, the fair analogue is to resample indices (with
    replacement, exactly like the original) and take the submatrix. Everything
    downstream -- persistence extraction, most-stable-gap voting, final
    threshold on the full matrix -- mirrors the original line for line.
    """
    n = D.shape[0]
    n_boot = int(n * boot_frac)
    rng = np.random.default_rng(seed)
    gap_counts: dict = {}
    for _ in range(n_boots):
        idx = rng.choice(n, size=n_boot, replace=True)
        Ds_boot = im.minimax_transform_fast(D[np.ix_(idx, idx)])
        blocks_boot, nb = SC._all_blocks(Ds_boot)
        ceiling = max_size_frac * nb
        persist = np.array(
            [b["persistence"] for b in blocks_boot if 3 <= b["size"] <= ceiling]
        )
        if len(persist) < 2:
            continue
        persist = np.sort(persist)[::-1]
        diffs = persist[:-1] / (persist[1:] + 1e-9)
        gap_idx = int(np.argmax(diffs))
        gap_counts[gap_idx] = gap_counts.get(gap_idx, 0) + 1

    if not gap_counts:
        return (
            0,
            [],
            {"method": "bottleneck_bootstrap_relational", "reason": "no_valid_boots"},
        )

    most_stable = max(gap_counts, key=gap_counts.get)
    Ds_full = im.minimax_transform_fast(D)
    blocks_full, n_full = SC._all_blocks(Ds_full)
    ceiling = max_size_frac * n_full
    persist_full = np.sort(
        np.array([b["persistence"] for b in blocks_full if 3 <= b["size"] <= ceiling])
    )[::-1]
    if len(persist_full) < 2:
        return (
            0,
            [],
            {"method": "bottleneck_bootstrap_relational", "reason": "too_few_blocks"},
        )
    if most_stable < len(persist_full) - 1:
        threshold = (persist_full[most_stable] + persist_full[most_stable + 1]) / 2
    else:
        threshold = persist_full[-1]

    elig = [
        b
        for b in blocks_full
        if 3 <= b["size"] <= ceiling and b["persistence"] >= threshold
    ]
    covered: set = set()
    sel: list = []
    all_pts = set(range(n_full))
    while covered != all_pts:
        best, best_gain = None, 0
        for b in elig:
            if b in sel:
                continue
            gain = len(b["members"] - covered)
            if gain > best_gain or (
                gain == best_gain
                and best is not None
                and b["persistence"] > best["persistence"]
            ):
                best, best_gain = b, gain
        if best is None or best_gain == 0:
            break
        sel.append(best)
        covered |= best["members"]
    meta = {
        "method": "bottleneck_bootstrap_relational",
        "gap_frequency": gap_counts[most_stable] / n_boots,
        "selected_threshold": float(threshold),
    }
    return len(sel), sel, meta


# ---------------------------------------------------------------------------
# E1: diagnostics
# ---------------------------------------------------------------------------


def run_diagnostics():
    """Quantify non-Euclideanness of D vs D* and test the ultrametric claim."""
    table = {}
    cases = [(name, fn, k) for name, (fn, k) in ND.BATTERY.items()]
    # The controlled-corruption reference points, at the sweep's center cell.
    D0, y0, _ = ND.euclidean_blobs()
    cases.append(("blobs_clean", lambda: (D0, y0), 3))
    cases.append(
        (
            "blobs_stretch",
            lambda: (ND.violate_pairs(D0, 0.2, 0.8, "stretch", seed=1), y0),
            3,
        )
    )
    cases.append(
        (
            "blobs_shortcut",
            lambda: (ND.violate_pairs(D0, 0.2, 0.8, "shortcut", seed=1), y0),
            3,
        )
    )
    # positive control: the one construction that DOES fire beta-spread, so a
    # zero in every other row is a measurement, not a blind spot.
    cases.append(("spiked_random(control)", ND.spiked_random, 2))
    for name, fn, k in cases:
        D, y = fn()
        Dstar = im.minimax_transform_fast(D)
        tv = ND.triangle_violation_stats(D)
        em_d = ND.euclidean_embeddability(D)
        em_s = ND.euclidean_embeddability(Dstar)
        # beta activation probed across restart seeds AND c in {2, 3, 4}:
        # activation depends on the NERFCM trajectory, not just admissibility,
        # so a single run understates it.
        beta_d = 0.0
        beta_s = 0.0
        for c in (2, 3, 4):
            _, _, b_d = nerfcm_score(D, y, c)
            _, _, b_s = nerfcm_score(Dstar, y, c)
            beta_d = max(beta_d, b_d)
            beta_s = max(beta_s, b_s)
        table[name] = {
            "n": int(D.shape[0]),
            "k_true": int(k),
            "ti_violation_pair_fraction": round(tv["pair_violation_fraction"], 4),
            "ti_max_violation_depth": round(tv["max_violation_depth"], 4),
            "neg_eig_ratio_D": round(em_d["neg_ratio"], 6),
            "neg_eig_ratio_Dstar": float(em_s["neg_ratio"]),
            "ultrametric_Dstar": bool(ND.is_ultrametric(Dstar)),
            "beta_max_D": round(beta_d, 6),
            "beta_max_Dstar": round(beta_s, 6),
        }
    results["diagnostics"] = table
    return table


# ---------------------------------------------------------------------------
# E2: method battery on the non-Euclidean datasets
# ---------------------------------------------------------------------------


def run_battery():
    table = {}
    for name, (fn, k_true) in ND.BATTERY.items():
        D, y = fn()
        Dstar = im.minimax_transform_fast(D)

        m_d, s_d, beta_d = nerfcm_score(D, y, k_true)
        m_ds, s_ds, beta_ds = nerfcm_score(Dstar, y, k_true)

        k_gap, sel_gap, _ = SC.select_persistence_gap(Dstar)
        k_bp, sel_bp, _ = SC.select_beta_plateau(Dstar)
        k_bb, sel_bb, meta_bb = select_bottleneck_bootstrap_relational(D)

        kg, covg, arig = score_selection(Dstar, y, sel_gap)
        kb, covb, arib = score_selection(Dstar, y, sel_bp)
        kbb, covbb, aribb = score_selection(Dstar, y, sel_bb)

        table[name] = {
            "n": int(D.shape[0]),
            "k_true": int(k_true),
            "NERFCM_D_ari": round(m_d, 3),
            "NERFCM_D_std": round(s_d, 3),
            "NERFCM_Dstar_ari": round(m_ds, 3),
            "NERFCM_Dstar_std": round(s_ds, 3),
            "beta_max_D": round(beta_d, 6),
            "beta_max_Dstar": round(beta_ds, 6),
            "SL_at_ktrue_ari": round(sl_ari_at_k(Dstar, y, k_true), 3),
            "persistence_gap": {"k": kg, "coverage": round(covg, 3), "ari": _r(arig)},
            "beta_plateau": {"k": kb, "coverage": round(covb, 3), "ari": _r(arib)},
            "bottleneck_bootstrap": {
                "k": kbb,
                "coverage": round(covbb, 3),
                "ari": _r(aribb),
                "gap_frequency": round(meta_bb.get("gap_frequency", 0.0), 3),
            },
        }
    results["battery"] = table
    return table


def _r(x, nd=3):
    return None if x is None else round(x, nd)


# ---------------------------------------------------------------------------
# E3: controlled violation sweep (stretch vs shortcut)
# ---------------------------------------------------------------------------

SWEEP_RATES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
SWEEP_STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SWEEP_FIXED_STRENGTH = 0.8
SWEEP_FIXED_RATE = 0.2


def _sweep_cell(mode, rate, strength):
    """Metrics for one (mode, rate, strength) cell, averaged over the dataset
    replicates. NERFCM numbers are additionally averaged over restart seeds."""
    accum = {
        "NERFCM_D_ari": [],
        "NERFCM_Dstar_ari": [],
        "cover_ari": [],
        "cover_k": [],
        "cover_coverage": [],
        "beta_max_D": [],
        "neg_eig_ratio_D": [],
        "ti_violation_pair_fraction": [],
    }
    for ds in DATASET_SEEDS:
        D0, y, _ = ND.euclidean_blobs(seed=206 + 100 * ds)
        D = ND.violate_pairs(D0, rate, strength, mode, seed=1 + ds)
        Dstar = im.minimax_transform_fast(D)
        m_d, _, beta_d = nerfcm_score(D, y, 3)
        m_ds, _, _ = nerfcm_score(Dstar, y, 3)
        sel = S.select_coverage_cover(Dstar)
        k_c, cov_c, ari_c = score_selection(Dstar, y, sel)
        em = ND.euclidean_embeddability(D)
        tv = ND.triangle_violation_stats(D)
        accum["NERFCM_D_ari"].append(m_d)
        accum["NERFCM_Dstar_ari"].append(m_ds)
        accum["cover_ari"].append(0.0 if ari_c is None else ari_c)
        accum["cover_k"].append(k_c)
        accum["cover_coverage"].append(cov_c)
        accum["beta_max_D"].append(beta_d)
        accum["neg_eig_ratio_D"].append(em["neg_ratio"])
        accum["ti_violation_pair_fraction"].append(tv["pair_violation_fraction"])
    cell = {k: round(float(np.mean(v)), 4) for k, v in accum.items()}
    cell["cover_ari_std"] = round(float(np.std(accum["cover_ari"])), 4)
    cell["NERFCM_D_ari_std"] = round(float(np.std(accum["NERFCM_D_ari"])), 4)
    cell["NERFCM_Dstar_ari_std"] = round(float(np.std(accum["NERFCM_Dstar_ari"])), 4)
    return cell


def run_violation_sweep():
    sweep = {"by_rate": {}, "by_strength": {}}
    for mode in ("stretch", "shortcut"):
        sweep["by_rate"][mode] = {
            str(r): _sweep_cell(mode, r, SWEEP_FIXED_STRENGTH) for r in SWEEP_RATES
        }
        sweep["by_strength"][mode] = {
            str(s): _sweep_cell(mode, SWEEP_FIXED_RATE, s) for s in SWEEP_STRENGTHS
        }
    sweep["params"] = {
        "rates": SWEEP_RATES,
        "strengths": SWEEP_STRENGTHS,
        "fixed_strength": SWEEP_FIXED_STRENGTH,
        "fixed_rate": SWEEP_FIXED_RATE,
        "dataset_seeds": DATASET_SEEDS,
        "base": "euclidean_blobs(n_per=20, sep=4.0, sigma=1.0)",
    }
    results["violation_sweep"] = sweep
    return sweep


# ---------------------------------------------------------------------------
# E4: relational multi-scale (Option D on distance-matrix-only hierarchies)
# ---------------------------------------------------------------------------


def _multiscale_case(D, levels):
    """Flat cover vs multi-scale bands, scored against every truth level.

    levels: dict name -> label array. Mirrors run_all.run_multiscale_numeric's
    reporting: the flat selector produces one partition scored against each
    level; multi-scale reports, per level, the best band's ARI.
    """
    Dstar = im.minimax_transform_fast(D)
    sel_flat = S.select_coverage_cover(Dstar)
    flat = {}
    for lname, y in levels.items():
        k_f, cov_f, ari_f = score_selection(Dstar, y, sel_flat)
        flat[lname] = {"k": k_f, "coverage": round(cov_f, 3), "ari": _r(ari_f)}

    msel = MS.select_multiscale(Dstar)
    band_labels = [MS.assign_band(band, Dstar) for band in msel.bands]
    band_ks = [len(band.blocks) for band in msel.bands]
    ms = {"n_bands": len(msel.bands), "band_granularities": band_ks, "per_level": {}}
    for lname, y in levels.items():
        best = None
        for lab in band_labels:
            ari = float(adjusted_rand_score(y, lab))
            if best is None or ari > best:
                best = ari
        ms["per_level"][lname] = _r(best)
    return {"flat": flat, "multiscale": ms}


def run_relational_multiscale():
    table = {}

    # (a) The formerly standing open problem: relationdata.multi_scale_hierarchy
    # was reported at NERFCM ARI 0.29 in notes/RELATIONDATA.md -- scored with
    # c=3 against its 6 fine labels. Two confounds were separated here:
    #   granularity mismatch -- score against both levels, not just fine;
    #   label noise -- the generator's leaf-expansion loop used to assign
    #     rng.integers(0, 4) labels regardless of where a leaf attached, so
    #     ~18% of DECLARED labels disagreed with the sub-cluster the distances
    #     encode. FIXED per issue #160: declared and structural labels now
    #     coincide, and declared_label_noise below reads 0. The structural
    #     scoring (connected components of {D < 2.0}; unambiguous, because the
    #     construction's scales are ~0.8 / ~4.6 / ~12.6) is kept both as the
    #     regression instrument and as the historical record of the
    #     decomposition.
    D, y_fine = RD.multi_scale_hierarchy()
    y_coarse = y_fine // 2
    from scipy.sparse.csgraph import connected_components

    _, y_struct = connected_components(D < 2.0, directed=False)
    # structural coarse level: components whose mutual distance is ~4.6
    n_comp = y_struct.max() + 1
    comp_med = np.zeros((n_comp, n_comp))
    for a in range(n_comp):
        for b in range(n_comp):
            if a != b:
                comp_med[a, b] = np.median(D[np.ix_(y_struct == a, y_struct == b)])
    _, comp_coarse = connected_components(
        (comp_med < 8.0) & (comp_med > 0), directed=False
    )
    y_struct_coarse = comp_coarse[y_struct]
    case = _multiscale_case(
        D,
        {
            "fine6": y_fine,
            "coarse3": y_coarse,
            "fine_structural": y_struct,
            "coarse_structural": y_struct_coarse,
        },
    )
    m3, _, _ = nerfcm_score(D, y_fine, 3)
    m3c, _, _ = nerfcm_score(D, y_coarse, 3)
    m6, _, _ = nerfcm_score(D, y_fine, 6)
    m6s, _, _ = nerfcm_score(D, y_struct, 6)
    case["nerfcm_reference"] = {
        "c3_vs_fine6": round(m3, 3),
        "c3_vs_coarse3": round(m3c, 3),
        "c6_vs_fine6": round(m6, 3),
        "c6_vs_fine_structural": round(m6s, 3),
    }
    case["declared_label_noise"] = {
        "n": int(len(y_fine)),
        "n_disagreeing": int(
            sum(
                (
                    y_fine[y_struct == c] != np.bincount(y_fine[y_struct == c]).argmax()
                ).sum()
                for c in range(n_comp)
            )
        ),
    }
    table["multi_scale_hierarchy(existing)"] = case

    # (b) Clean two-level relational hierarchy with explicit truth at both levels.
    D2, yf, yc = ND.relational_nested_hierarchy()
    table["relational_nested(clean)"] = _multiscale_case(
        D2, {"fine6": yf, "coarse3": yc}
    )

    # (c, d) The same hierarchy under the two corruption directions.
    D_st = ND.violate_pairs(D2, 0.2, 0.8, "stretch", seed=3)
    table["relational_nested(stretch r=.2 s=.8)"] = _multiscale_case(
        D_st, {"fine6": yf, "coarse3": yc}
    )
    D_sc = ND.violate_pairs(D2, 0.05, 0.8, "shortcut", seed=3)
    table["relational_nested(shortcut r=.05 s=.8)"] = _multiscale_case(
        D_sc, {"fine6": yf, "coarse3": yc}
    )

    results["relational_multiscale"] = table
    return table


# ---------------------------------------------------------------------------
# E5: real-data DTW -- N-CMAPSS DS01 flight altitude profiles
# ---------------------------------------------------------------------------

NCMAPSS_DS01 = "../NASA-CMAPSS/N-CMAPSS_DS01-005.h5"
E5_STRIDE = 300  # one altitude sample every ~5 min of flight (data is 1 Hz)
E5_PER_CLASS = 25


def load_flight_traces(path=NCMAPSS_DS01, stride=E5_STRIDE, per_class=E5_PER_CLASS):
    """Altitude traces per flight from N-CMAPSS DS01 dev, labeled by flight class.

    DS01 is used rather than the dissertation's usual DS02 because DS02's dev
    units are ALL flight class 3; DS01 has all three classes (195/194/164
    flights). Traces are subsampled at a fixed rate -- NOT length-normalized --
    so sequence length carries flight duration, which is what the class labels
    bin (~1-1.6h / 1.3-3.3h / 2.6-5.2h in this sample; the bins genuinely
    overlap, so no method should be expected to reach ARI 1.0). Altitude is
    scaled by 1e4 ft to O(1). Selection of `per_class` flights per class is
    deterministic (seeded permutation).
    """
    import h5py

    with h5py.File(path, "r") as f:
        A = f["A_dev"][:]
        alt = f["W_dev"][:, 0]
    key = (A[:, 0] * 10000 + A[:, 1]).astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_s, alt_s, fc_s = key[order], alt[order], A[order, 2]
    uk, starts = np.unique(key_s, return_index=True)
    ends = np.append(starts[1:], len(key_s))
    rng = np.random.default_rng(0)
    traces, labels, durations = [], [], []
    counts = {1: 0, 2: 0, 3: 0}
    for i in rng.permutation(len(uk)):
        fc = int(fc_s[starts[i]])
        if counts[fc] >= per_class:
            continue
        traces.append(alt_s[starts[i] : ends[i] : stride] / 1e4)
        labels.append(fc - 1)
        durations.append(int(ends[i] - starts[i]))
        counts[fc] += 1
    return traces, np.asarray(labels, dtype=int), np.asarray(durations, dtype=float)


def run_real_dtw():
    """The battery on a REAL non-metric dissimilarity: DTW over N-CMAPSS DS01
    flight altitude profiles, truth = flight class. Skipped (with a stub row in
    the JSON) when the gitignored .h5 is absent, so a fresh clone still runs."""
    import os

    if not os.path.exists(NCMAPSS_DS01):
        results["real_dtw_ncmapss"] = {
            "skipped": f"{NCMAPSS_DS01} not present (gitignored dataset)"
        }
        return None

    traces, y, durations = load_flight_traces()
    D = ND.pairwise(traces, ND.dtw_distance)
    Dstar = im.minimax_transform_fast(D)

    tv = ND.triangle_violation_stats(D)
    em_d = ND.euclidean_embeddability(D)
    em_s = ND.euclidean_embeddability(Dstar)

    m_d, s_d, beta_d = nerfcm_score(D, y, 3)
    m_ds, s_ds, _ = nerfcm_score(Dstar, y, 3)
    k_gap, sel_gap, _ = SC.select_persistence_gap(Dstar)
    k_bp, sel_bp, _ = SC.select_beta_plateau(Dstar)
    _, sel_bb, meta_bb = select_bottleneck_bootstrap_relational(D)
    kg, covg, arig = score_selection(Dstar, y, sel_gap)
    kb, covb, arib = score_selection(Dstar, y, sel_bp)
    kbb, covbb, aribb = score_selection(Dstar, y, sel_bb)

    # Reference ceiling: the class labels are duration BINS, so cluster on
    # duration alone. Methods can only be judged against this, not against 1.0.
    from sklearn.cluster import KMeans

    km = KMeans(3, n_init=10, random_state=0).fit(durations.reshape(-1, 1))
    dur_ari = float(adjusted_rand_score(y, km.labels_))

    table = {
        "dataset": "N-CMAPSS DS01-005 dev, altitude/1e4, stride 300, 25 flights/class",
        "n": int(D.shape[0]),
        "k_true": 3,
        "ti_violation_pair_fraction": round(tv["pair_violation_fraction"], 4),
        "neg_eig_ratio_D": round(em_d["neg_ratio"], 4),
        "neg_eig_ratio_Dstar": float(em_s["neg_ratio"]),
        "ultrametric_Dstar": bool(ND.is_ultrametric(Dstar)),
        "beta_max_D": round(beta_d, 6),
        "duration_only_kmeans_ari(reference ceiling)": round(dur_ari, 3),
        "NERFCM_D_ari": round(m_d, 3),
        "NERFCM_D_std": round(s_d, 3),
        "NERFCM_Dstar_ari": round(m_ds, 3),
        "NERFCM_Dstar_std": round(s_ds, 3),
        "SL_at_ktrue_ari": round(sl_ari_at_k(Dstar, y, 3), 3),
        "persistence_gap": {"k": kg, "coverage": round(covg, 3), "ari": _r(arig)},
        "beta_plateau": {"k": kb, "coverage": round(covb, 3), "ari": _r(arib)},
        "bottleneck_bootstrap": {
            "k": kbb,
            "coverage": round(covbb, 3),
            "ari": _r(aribb),
        },
    }
    results["real_dtw_ncmapss"] = table
    return table


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def save_figure(fig, filename):
    path = f"{OUT}/{filename}"
    fig.savefig(path, dpi=96, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_diagnostics(diag):
    names = list(diag.keys())
    x = np.arange(len(names))
    width = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    eps = 1e-18  # display floor so log-scale bars render for ~0 values
    nd = [max(diag[n]["neg_eig_ratio_D"], eps) for n in names]
    ns = [max(diag[n]["neg_eig_ratio_Dstar"], eps) for n in names]
    ax1.bar(x - width / 2, nd, width, label="raw D", color="steelblue")
    ax1.bar(x + width / 2, ns, width, label="D* (minimax)", color="coral")
    ax1.set_yscale("log")
    ax1.set_ylabel("negative-eigenvalue ratio (log)")
    ax1.set_title("Euclidean inadmissibility, before vs after minimax")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.axhline(1e-12, color="gray", lw=0.8, ls="--")
    ax1.text(0.02, 1.5e-12, "machine noise", fontsize=7, color="gray")
    ax1.legend()

    tv = [diag[n]["ti_violation_pair_fraction"] for n in names]
    ax2.bar(x, tv, 0.6, color="slategray")
    ax2.set_ylabel("triangle-violated pair fraction")
    ax2.set_title("Metricity of the raw dissimilarity")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)

    fig.suptitle(
        "D* is ultrametric, hence Euclidean-admissible, for every input family",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "fig12_nonmetric_diagnostics.png")


def fig_battery(table):
    names = list(table.keys())
    methods = [
        ("NERFCM(D)", lambda e: e["NERFCM_D_ari"]),
        ("NERFCM(D*)", lambda e: e["NERFCM_Dstar_ari"]),
        ("SL @ k_true", lambda e: e["SL_at_ktrue_ari"]),
        ("gap-cover", lambda e: e["persistence_gap"]["ari"]),
        ("beta-plateau", lambda e: e["beta_plateau"]["ari"]),
        ("bootstrap", lambda e: e["bottleneck_bootstrap"]["ari"]),
    ]
    x = np.arange(len(names))
    width = 0.13
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for i, (label, get) in enumerate(methods):
        vals = [get(table[n]) if get(table[n]) is not None else 0.0 for n in names]
        ax.bar(x + (i - 2.5) * width, vals, width, label=label)
    ax.set_ylabel("ARI vs planted clusters")
    ax.set_title(
        "Non-Euclidean battery: relational averaging vs minimax selection",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(-0.05, 1.12)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "fig13_nonmetric_battery.png")


def fig_violation_sweep(sweep):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    panels = [
        (
            "by_rate",
            "stretch",
            SWEEP_RATES,
            f"stretch: rate (strength={SWEEP_FIXED_STRENGTH})",
        ),
        (
            "by_rate",
            "shortcut",
            SWEEP_RATES,
            f"shortcut: rate (strength={SWEEP_FIXED_STRENGTH})",
        ),
        (
            "by_strength",
            "stretch",
            SWEEP_STRENGTHS,
            f"stretch: strength (rate={SWEEP_FIXED_RATE})",
        ),
        (
            "by_strength",
            "shortcut",
            SWEEP_STRENGTHS,
            f"shortcut: strength (rate={SWEEP_FIXED_RATE})",
        ),
    ]
    series = [
        ("NERFCM_D_ari", "NERFCM(D)", "steelblue"),
        ("NERFCM_Dstar_ari", "NERFCM(D*)", "coral"),
        ("cover_ari", "gap-cover(D*)", "seagreen"),
    ]
    for ax, (axis, mode, grid, title) in zip(axes.ravel(), panels):
        cells = sweep[axis][mode]
        for key, label, color in series:
            vals = [cells[str(g)][key] for g in grid]
            errs = [cells[str(g)][key + "_std"] for g in grid]
            ax.errorbar(
                grid,
                vals,
                yerr=errs,
                label=label,
                color=color,
                marker="o",
                ms=4,
                capsize=2,
            )
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)
        ax.set_ylabel("ARI")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel("violation strength")
    axes[1, 1].set_xlabel("violation strength")
    axes[0, 0].set_xlabel("violated pair fraction")
    axes[0, 1].set_xlabel("violated pair fraction")
    fig.suptitle(
        "Stretch vs shortcut corruption hurts different methods",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "fig14_violation_sweep.png")


def fig_relational_multiscale(table):
    # One x-position per (case, truth-set). The existing dataset appears twice:
    # against its declared labels (18% of which are noise -- see E4) and against
    # the structural labels the distances actually encode; the gap between the
    # two bars IS the label-noise finding.
    rows = []
    for case, e in table.items():
        if "fine_structural" in e["flat"]:
            rows.append((f"{case}\n(declared labels)", "fine6", "coarse3", e))
            rows.append(
                (
                    f"{case}\n(structural labels)",
                    "fine_structural",
                    "coarse_structural",
                    e,
                )
            )
        else:
            rows.append((case, "fine6", "coarse3", e))
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    for i, which in enumerate(("fine", "coarse")):
        flat_vals, ms_vals = [], []
        for _, fine_key, coarse_key, e in rows:
            key = fine_key if which == "fine" else coarse_key
            fv = e["flat"][key]["ari"]
            mv = e["multiscale"]["per_level"][key]
            flat_vals.append(0.0 if fv is None else fv)
            ms_vals.append(0.0 if mv is None else mv)
        ax.bar(
            x + (2 * i - 1.5) * width,
            flat_vals,
            width,
            label=f"flat / {which}",
            alpha=0.65,
        )
        ax.bar(
            x + (2 * i - 0.5) * width, ms_vals, width, label=f"multi-scale / {which}"
        )
    ax.set_ylabel("ARI vs truth level")
    ax.set_title(
        "Relational hierarchies: flat cover vs multi-scale bands (Option D)",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=10, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "fig15_relational_multiscale.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    import os

    os.makedirs(OUT, exist_ok=True)

    print("E1: diagnostics (non-Euclideanness of D vs D*)...")
    diag = run_diagnostics()
    print("E2: method battery on non-Euclidean datasets...")
    battery = run_battery()
    print("E3: violation sweep (stretch vs shortcut)...")
    sweep = run_violation_sweep()
    print("E4: relational multi-scale...")
    msc = run_relational_multiscale()
    print("E5: real-data DTW (N-CMAPSS DS01 flight profiles)...")
    real = run_real_dtw()

    results["seeds"] = {
        "nerfcm_restarts": list(SEEDS),
        "dataset_replicates(sweep)": list(DATASET_SEEDS),
        "generators": "fixed per generator; see nonmetric_data.py defaults",
    }

    # JSON first, figures second: a figure crash must not discard the numbers
    # (the failure mode AGENTS.md documents for run_all.py).
    with open(f"{OUT}/nonmetric_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Numeric results -> {OUT}/nonmetric_results.json")

    fig_diagnostics(diag)
    fig_battery(battery)
    fig_violation_sweep(sweep)
    fig_relational_multiscale(msc)
    print("Figures -> fig12..fig15 in", OUT)

    # console summary
    print("\nDIAGNOSTICS (is D* always admissible?):")
    for name, e in diag.items():
        print(
            f"  {name}: TIviol={e['ti_violation_pair_fraction']:.3f} "
            f"negeig D={e['neg_eig_ratio_D']:.4f} -> D*={e['neg_eig_ratio_Dstar']:.2e} "
            f"ultra(D*)={e['ultrametric_Dstar']} "
            f"beta(D)={e['beta_max_D']} beta(D*)={e['beta_max_Dstar']}"
        )
    print("\nBATTERY (ARI):")
    for name, e in battery.items():
        print(
            f"  {name}: NERFCM(D)={e['NERFCM_D_ari']} NERFCM(D*)={e['NERFCM_Dstar_ari']} "
            f"SL@k={e['SL_at_ktrue_ari']} | gap: k={e['persistence_gap']['k']} "
            f"ARI={e['persistence_gap']['ari']} | plateau: k={e['beta_plateau']['k']} "
            f"ARI={e['beta_plateau']['ari']} | boot: k={e['bottleneck_bootstrap']['k']} "
            f"ARI={e['bottleneck_bootstrap']['ari']}"
        )
    print("\nRELATIONAL MULTISCALE (ARI per level, flat vs best band):")
    for name, e in msc.items():
        for lname in e["flat"]:
            print(
                f"  {name} [{lname}]: flat={e['flat'][lname]['ari']} "
                f"(k={e['flat'][lname]['k']}) multiscale={e['multiscale']['per_level'][lname]} "
                f"(bands={e['multiscale']['band_granularities']})"
            )
    if real is not None:
        print("\nREAL-DATA DTW (N-CMAPSS DS01 flight classes):")
        print(
            f"  TIviol={real['ti_violation_pair_fraction']} "
            f"negeig D={real['neg_eig_ratio_D']} ultra(D*)={real['ultrametric_Dstar']} "
            f"beta(D)={real['beta_max_D']}"
        )
        print(
            f"  duration-only ceiling={real['duration_only_kmeans_ari(reference ceiling)']} | "
            f"NERFCM(D)={real['NERFCM_D_ari']} NERFCM(D*)={real['NERFCM_Dstar_ari']} "
            f"SL@k={real['SL_at_ktrue_ari']} | gap: k={real['persistence_gap']['k']} "
            f"ARI={real['persistence_gap']['ari']} | plateau: k={real['beta_plateau']['k']} "
            f"ARI={real['beta_plateau']['ari']} | boot: k={real['bottleneck_bootstrap']['k']} "
            f"ARI={real['bottleneck_bootstrap']['ari']}"
        )
    else:
        print("\nREAL-DATA DTW: skipped (N-CMAPSS DS01 .h5 not present)")


if __name__ == "__main__":
    main()
