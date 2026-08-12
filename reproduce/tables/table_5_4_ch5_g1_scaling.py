"""Table 5.4 -- Goal G1's scaling decision rule, minus the one-pass arm.

Chapter 5 SS5.4 and Chapter 7's Goal G1 (`07-goals-for-completion.md`, search
"### G1") both call for one measurement: three fixed-structure families from
`battery_hierarchical.SCALABLE` (`single_scale`, `many_scale`, `log_separated`)
at n = 100, 250, 500, 1000, 2000, 5000, comparing the ONE-PASS generator against
both the two-stage selector and the flat set-cover, on identical data, reporting
the recovered granularity vector, per-level adjusted Rand index (ARI) of the
defuzzified partition against each ground-truth level, and the partition-of-
unity error under Ruspini normalization.

WHAT THIS SCRIPT DOES NOT DO, STATED LOUDLY: it does not run the one-pass arm.
As of this run, `MEMBERSHIP_ROADMAP.md` Phase 5 ("one-pass generation") is
unimplemented -- confirmed by grep across `gated-minimax-selection/` for
`one_pass`/`onepass`/"one-pass" (no hits outside notes and this docstring) and
by `git log` on that directory (no commit mentions phase five or a one-pass
refactor; the latest membership-function work stops at Phase 4, "soft bands",
which a later commit found did not fix its target failure). `07-goals-for-
completion.md`'s own G1 status agrees: "Phase five, the one-pass refactor, is
plumbing and unattempted." So there is no one-pass generator to run, and this
script measures the two arms that DO exist: `selection.select_coverage_cover`
(flat set-cover) and `multiscale_persistence.select_multiscale` (two-stage
selector), at the full size grid the decision rule names. This directly answers
SS5.4's stated gap -- "no recorded run of the [8, 4, 2] recovery exists at any
size other than 96" -- for the two-stage arm, at ten seeds instead of one.

WHAT WAS ALREADY MEASURED, AND WHAT THIS ADDS. `gated-minimax-selection/notes/
SCALING_STUDY.md` and its backing `outputs/scaling_results.json` already ran
the two-stage selector across this exact grid and these exact three families --
but at a single fixed seed per family (the generator's own default), with no
flat-set-cover comparison at this grid, and with no partition-of-unity
measurement. This script adds the three things that were missing: (1) the flat
baseline at every (family, n) cell, (2) partition-of-unity error via
`multiscale_persistence.band_memberships` + `normalize_partition`, and (3) ten
seeds (`common.SEEDS`, overridable with `REPRO_SEEDS`) via the generator's own
`seed=` argument, so the granularity/ARI/PoU cells carry a real spread instead
of one draw. It reuses `SCALING_STUDY.md`'s fast transform
(`ivat_mf.minimax_transform_fast`) for the same reason: the O(n^3) reference is
infeasible past n~500.

GRANULARITY ACROSS SEEDS. The two-stage selector's number of discovered scales
can differ seed to seed (see `log_separated`'s known n-sensitivity in
`SCALING_STUDY.md` SS4). A vector of varying length has no mean, so the granularity
column reports the MODE vector (most common across the ten seeds) and how many
of the ten seeds produced it. Per-level ARI is always well-defined regardless of
how many scales were discovered: each ground-truth level is scored against
whichever discovered band matches it best (`max` over bands), exactly the rule
`SCALING_STUDY.md`'s own driver uses. Partition-of-unity error is the worst
(max) band's error at that seed, since a single-number summary has to pick one
side of the mean/max tradeoff and the decision rule cares about the worst case.

Run (from repo root, root .venv -- gated-minimax-selection has no submodule
pyproject, per `ch5-gated-minimax-all`'s manifest entry):
    python reproduce/tables/table_5_4_ch5_g1_scaling.py

Cost: transform is O(n^2); at ten seeds x 3 families x 6 sizes the n=5000 cells
dominate (~4.4s transform each) and the whole grid runs in a few minutes on the
2026-08 workstation (see the machine block below) -- well under the "ask before
long compute" bar, so this ran directly rather than being launched separately.
"""

from __future__ import annotations

import os
import statistics
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score

_TABLES = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_TABLES))
sys.path.insert(0, os.path.dirname(_TABLES))
import common as C  # noqa: E402

_GMS = os.path.join(_ROOT, "gated-minimax-selection")
sys.path.insert(0, _GMS)
import ivat_mf as im  # noqa: E402
import selection as S  # noqa: E402
import multiscale_persistence as MS  # noqa: E402
import battery_hierarchical as BH  # noqa: E402

SIZES = (100, 250, 500, 1000, 2000, 5000)
FAMILIES = ["single_scale", "many_scale", "log_separated"]


def _pou_error_for_blocks(blocks, Dstar):
    """(max, mean) Ruspini partition-of-unity error over covered points for one
    flat set of blocks -- the same statistic `FuzzyHierarchy.partition_of_unity_
    error` reports per band, applied to the flat selector's single partition."""
    if not blocks:
        return float("nan"), float("nan")
    U = np.vstack([MS.block_membership(b, Dstar) for b in blocks])
    Un = MS.normalize_partition(U)
    s = Un.sum(axis=0)
    covered = s > 1e-9
    if not covered.any():
        return float("nan"), float("nan")
    err = np.abs(s[covered] - 1.0)
    return float(err.max()), float(err.mean())


def _pou_error_for_bands(bands, Dstar):
    """Worst-band (max, mean) PoU error across a multi-scale hierarchy's bands."""
    if not bands:
        return float("nan"), float("nan")
    maxes, means = [], []
    for b in bands:
        mx, mn = _pou_error_for_blocks(b.blocks, Dstar)
        if not np.isnan(mx):
            maxes.append(mx)
            means.append(mn)
    if not maxes:
        return float("nan"), float("nan")
    return max(maxes), float(np.mean(means))


def _mode_vector(vectors):
    """Most common granularity vector (as a tuple) and how many seeds gave it."""
    counts = {}
    for v in vectors:
        t = tuple(v)
        counts[t] = counts.get(t, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])
    return list(best[0]), best[1], len(vectors)


def run_one(family, n, seed):
    gen, level_names, _expected = BH.SCALABLE[family]
    out = gen(n, seed=seed)
    X, levels = out[0], list(out[1:])
    Dstar = im.minimax_transform_fast(im.dissimilarity(X))

    # Flat set-cover.
    flat_sel = S.select_coverage_cover(Dstar)
    flat_labels = MS.assign(flat_sel, Dstar)
    flat_ari = [
        adjusted_rand_score(y, flat_labels) if flat_sel else float("nan")
        for y in levels
    ]
    flat_k = len(flat_sel)
    flat_pou_max, flat_pou_mean = _pou_error_for_blocks(flat_sel, Dstar)

    # Two-stage selector.
    msel = MS.select_multiscale(Dstar)
    band_assign = [MS.assign_band(b, Dstar) for b in msel.bands]
    ms_ari = [
        max((adjusted_rand_score(y, a) for a in band_assign), default=float("nan"))
        for y in levels
    ]
    ms_gran = msel.granularities()
    ms_pou_max, ms_pou_mean = _pou_error_for_bands(msel.bands, Dstar)

    return {
        "level_names": level_names,
        "flat_k": flat_k,
        "flat_ari": flat_ari,
        "flat_pou_max": flat_pou_max,
        "flat_pou_mean": flat_pou_mean,
        "ms_gran": ms_gran,
        "ms_ari": ms_ari,
        "ms_pou_max": ms_pou_max,
        "ms_pou_mean": ms_pou_mean,
    }


def fmt_levels(level_lists, fmt="{:.2f}"):
    """Transpose a per-seed list of per-level ARI lists into per-level 'mean±std'
    cells, joined into one bracketed string."""
    if not level_lists:
        return C.NA
    n_levels = len(level_lists[0])
    cells = []
    for lvl in range(n_levels):
        vals = [ll[lvl] for ll in level_lists if lvl < len(ll)]
        cells.append(C.cell(vals, fmt=fmt))
    return "[" + ", ".join(cells) + "]"


def main():
    seeds = C.SEEDS
    print(f"Table 5.4 -- G1 scaling decision rule (two-stage vs flat), seeds={seeds}")

    header = [
        "family",
        "n",
        "flat k (mean±std)",
        "flat ARI/level",
        "flat PoU err max (mean±std)",
        "two-stage granularity (mode; agreement)",
        "two-stage ARI/level",
        "two-stage PoU err max (mean±std)",
    ]
    rows = []
    raw_rows = []
    raw_header = [
        "family",
        "n",
        "seed",
        "flat_k",
        "flat_ari_per_level",
        "flat_pou_max",
        "flat_pou_mean",
        "ms_granularities",
        "ms_ari_per_level",
        "ms_pou_max",
        "ms_pou_mean",
    ]

    for family in FAMILIES:
        for n in SIZES:
            per_seed = []
            for seed in seeds:
                r = run_one(family, n, seed)
                per_seed.append(r)
                raw_rows.append(
                    [
                        family,
                        n,
                        seed,
                        r["flat_k"],
                        r["flat_ari"],
                        r["flat_pou_max"],
                        r["flat_pou_mean"],
                        r["ms_gran"],
                        r["ms_ari"],
                        r["ms_pou_max"],
                        r["ms_pou_mean"],
                    ]
                )
            flat_k_cell = C.cell([r["flat_k"] for r in per_seed], fmt="{:.1f}")
            flat_ari_cell = fmt_levels([r["flat_ari"] for r in per_seed])
            flat_pou_cell = C.cell(
                [r["flat_pou_max"] for r in per_seed], fmt="{:.2e}"
            )
            mode_vec, agree, total = _mode_vector([r["ms_gran"] for r in per_seed])
            ms_gran_cell = f"{mode_vec} ({agree}/{total})"
            ms_ari_cell = fmt_levels([r["ms_ari"] for r in per_seed])
            ms_pou_cell = C.cell([r["ms_pou_max"] for r in per_seed], fmt="{:.2e}")
            rows.append(
                [
                    family,
                    n,
                    flat_k_cell,
                    flat_ari_cell,
                    flat_pou_cell,
                    ms_gran_cell,
                    ms_ari_cell,
                    ms_pou_cell,
                ]
            )
            print(
                f"  {family:14s} n={n:5d}  flat_k={flat_k_cell:>10s}  "
                f"ms_gran={mode_vec} ({agree}/{total})  ms_ari={ms_ari_cell}"
            )

    C.write_csv(
        os.path.join(C.OUTPUT_DIR, "table_5_4_ch5_g1_scaling_raw.csv"),
        raw_header,
        raw_rows,
    )

    C.emit(
        "table_5_4_ch5_g1_scaling",
        "Table 5.4 -- Goal G1 decision rule: two-stage selector vs. flat "
        "set-cover, n = 100..5000 (one-pass arm not run -- see note)",
        header,
        rows,
        note=(
            "ONE-PASS ARM NOT MEASURED: `MEMBERSHIP_ROADMAP.md` Phase 5 (the "
            "one-pass generator G1 asks for) is unimplemented as of this run -- "
            "verified by grepping `gated-minimax-selection/` for one_pass/"
            "onepass/'one-pass' (no code hits) and by `git log` on that "
            "directory, whose newest membership-function commit is Phase 4 "
            "('soft bands'), later found not to fix its target failure. "
            "`07-goals-for-completion.md`'s G1 status agrees: phase five is "
            "'plumbing and unattempted.' This table therefore measures the two "
            "arms that exist -- `selection.select_coverage_cover` (flat) and "
            "`multiscale_persistence.select_multiscale` (two-stage) -- at the "
            "full size grid and ten seeds the decision rule names, extending "
            "`gated-minimax-selection/notes/SCALING_STUDY.md` (single seed, "
            "two-stage only, no partition-of-unity) with the flat comparison, "
            "partition-of-unity error, and seed spread it lacked. Two-stage "
            "'granularity' cells show the MODE vector across ten seeds and how "
            "many seeds produced it, because a discovered vector's length can "
            "vary seed to seed (see log_separated) and only a fixed-length "
            "quantity has a mean. Per-level ARI always has ten values regardless "
            "of how many scales were discovered: each ground-truth level is "
            "scored against its single best-matching band. Partition-of-unity "
            "error is the worst (max) band's error per seed, then meaned over "
            "seeds; per Ruspini normalization it is expected at machine "
            "precision by construction, and reporting it is a check on that "
            "construction rather than a discriminating measurement. Full "
            "per-seed values are in the sibling "
            "`table_5_4_ch5_g1_scaling_raw.csv`. transform: "
            "`ivat_mf.minimax_transform_fast` (the O(n^2) fast path; the O(n^3) "
            "reference is infeasible past n~500, per SCALING_STUDY.md SS1)."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
