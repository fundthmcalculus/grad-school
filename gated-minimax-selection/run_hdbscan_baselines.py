"""HDBSCAN* head-to-head baselines for the gated set-cover and band selectors.

Why this exists
---------------
`research/proposal-defense/PRIOR_ART_CH5.md` retired two of Chapter 5's claimed
contributions on prior-art grounds and left one requirement before anything is
claimed for the multi-scale selector:

    Required before claiming: HDBSCAN `leaf` extraction and a `dbscan_clustering(eps)`
    sweep on the nested [8,4,2] synthetic. If leaf recovers 8 and an eps sweep
    recovers 4 and 2, a reviewer will ask why an eps sweep is not the whole
    contribution.

This driver answers that, and does so on deliberately generous terms: HDBSCAN* is
run at several `min_cluster_size` values and both extraction methods, and the
best result per dataset is reported alongside the default. The eps sweep is run
densely enough to enumerate every partition the cut-distance family can produce.

The comparison is anchored by TKDD 2015 Corollary 3.5 (Campello, Moulavi, Zimek &
Sander): at ``mpts`` in {1, 2} mutual reachability equals the input dissimilarity
and HDBSCAN* *is* single-linkage on those distances. So at ``min_samples=1`` both
sides consume the identical hierarchy and the only thing under test is the
*extraction rule* -- our persistence-outlier gate versus excess-of-mass, leaf, or
a flat cut. `min_samples=5` is also run, to check the separate question of
whether a kNN core distance is usable at all on a non-metric dissimilarity.

Requires the `hdbscan` contrib package (for `leaf` extraction and
`dbscan_clustering`); scikit-learn's `HDBSCAN` covers eom/leaf but exposes no
cut-distance accessor. Run with an interpreter that has it installed:

    python run_hdbscan_baselines.py --out outputs/hdbscan_baselines.json

Writes the JSON of record plus a markdown table next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import battery as B  # noqa: E402
import battery_hierarchical as BH  # noqa: E402
import ivat_mf as im  # noqa: E402
import multiscale_persistence as MS  # noqa: E402
import nonmetric_data as ND  # noqa: E402
import relationdata as RD  # noqa: E402
import selection as S  # noqa: E402

MIN_CLUSTER_SIZES: Tuple[int, ...] = (3, 5, 10)
DEFAULT_MIN_CLUSTER_SIZE = 5
EPS_SWEEP_POINTS = 400


# ---------------------------------------------------------------------------
# datasets: every one arrives as (D, [truth levels fine -> coarse], label)
# ---------------------------------------------------------------------------


def _euclid(X: np.ndarray) -> np.ndarray:
    return cdist(X, X)


def _dataset_specs() -> List[Dict]:
    """Build the dataset battery.

    Each entry is ``{name, family, D, truths}`` where ``truths`` is a list of
    ``(level_name, y)`` ordered fine -> coarse. ``y`` may contain -1 for points
    with no ground-truth cluster (the bridge); those are dropped when scoring.
    """
    specs: List[Dict] = []

    def flat(name: str, family: str, gen: Callable[[], Tuple[np.ndarray, np.ndarray]]):
        X, y = gen()
        specs.append(
            {"name": name, "family": family, "D": _euclid(X), "truths": [("only", y)]}
        )

    # -- flat coordinate battery (Table 5.1) --------------------------------
    flat("two_gaussians", "flat", B.two_gaussians)
    flat("bridged_gaussians", "flat", B.bridged_gaussians)
    flat("concentric_rings", "flat", B.concentric_rings)
    flat("varying_density", "flat", B.varying_density)
    flat("uniform_noise", "flat", B.uniform_noise)

    # -- nested hierarchies (Table 5.2) ------------------------------------
    X, y_fine, y_coarse = BH.nested_gaussians()
    specs.append(
        {
            "name": "nested_gaussians",
            "family": "hierarchical",
            "D": _euclid(X),
            "truths": [("fine", y_fine), ("coarse", y_coarse)],
        }
    )
    X, y_f, y_m, y_c = BH.three_level_hierarchy()
    specs.append(
        {
            "name": "three_level_hierarchy",
            "family": "hierarchical",
            "D": _euclid(X),
            "truths": [("fine", y_f), ("medium", y_m), ("coarse", y_c)],
        }
    )
    X, y_fine, y_coarse = BH.density_hierarchy()
    specs.append(
        {
            "name": "density_hierarchy",
            "family": "hierarchical",
            "D": _euclid(X),
            "truths": [("fine", y_fine), ("coarse", y_coarse)],
        }
    )

    # -- genuinely non-metric families -------------------------------------
    for name, gen in (
        ("dtw_traces", ND.dtw_traces),
        ("edit_strings", ND.edit_strings),
        ("hamming_categorical", ND.hamming_categorical),
        ("graph_communities", ND.graph_communities),
        ("cosine_topics", ND.cosine_topics),
    ):
        D, y = gen()
        specs.append(
            {"name": name, "family": "non-metric", "D": D, "truths": [("only", y)]}
        )

    D, y_fine, y_coarse = ND.relational_nested_hierarchy()
    specs.append(
        {
            "name": "relational_nested_hierarchy",
            "family": "non-metric",
            "D": D,
            "truths": [("fine", y_fine), ("coarse", y_coarse)],
        }
    )

    # -- relational (shortest-path) matrices -------------------------------
    for name, gen in (
        ("three_clusters_tree", RD.three_clusters_tree),
        ("chain_then_ring", RD.chain_then_ring),
        ("multi_scale_hierarchy", RD.multi_scale_hierarchy),
    ):
        D, y = gen()
        specs.append(
            {"name": name, "family": "relational", "D": D, "truths": [("only", y)]}
        )

    return specs


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _n_clusters(labels: np.ndarray) -> int:
    """Number of real clusters, i.e. distinct labels other than the -1 noise flag."""
    return int(len({int(v) for v in labels} - {-1}))


def score(labels: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """ARI of `labels` against `y`, two ways, plus the predicted cluster count.

    ``ari`` keeps HDBSCAN's -1 noise flag as a label of its own (the usual
    convention, and the one that penalises abstention). ``ari_noise_excluded``
    scores only the points the method actually claimed. Ground-truth -1 (the
    bridge points, which belong to no true cluster) is dropped from both.
    """
    keep = y >= 0
    if not keep.any():
        return {"ari": float("nan"), "ari_noise_excluded": float("nan")}
    lab, truth = labels[keep], y[keep]
    claimed = lab >= 0
    return {
        "ari": float(adjusted_rand_score(truth, lab)),
        "ari_noise_excluded": (
            float(adjusted_rand_score(truth[claimed], lab[claimed]))
            if claimed.sum() > 1
            else float("nan")
        ),
    }


def _per_level(labels: np.ndarray, truths: Sequence[Tuple[str, np.ndarray]]) -> Dict:
    out = {}
    for level, y in truths:
        out[level] = score(labels, y)
    return out


# ---------------------------------------------------------------------------
# our selectors
# ---------------------------------------------------------------------------


def run_ours_flat(D: np.ndarray, truths) -> Dict:
    """The flat gated set-cover of §5.3.2, scored under `assign` (every point placed)."""
    Dstar = im.minimax_transform(D)
    sel = S.select_coverage_cover(Dstar)
    labels = MS.assign(sel, Dstar) if sel else np.full(len(D), -1)
    return {
        "k": len(sel),
        "coverage": S.coverage_of(sel, len(D)),
        "levels": _per_level(labels, truths),
    }


def run_ours_multiscale(D: np.ndarray, truths) -> Dict:
    """The band selector of §5.3.3: one partition per discovered density band."""
    Dstar = im.minimax_transform(D)
    ms = MS.select_multiscale(Dstar)
    bands = []
    for band in ms.bands:
        labels = MS.assign_band(band, Dstar)
        bands.append(
            {
                "band_id": band.band_id,
                "k": band.k,
                "coverage": band.coverage_fraction(len(D)),
                "levels": _per_level(labels, truths),
            }
        )
    best = {}
    for level, _ in truths:
        cands = [b["levels"][level]["ari"] for b in bands]
        cands = [c for c in cands if not np.isnan(c)]
        best[level] = max(cands) if cands else float("nan")
    return {
        "n_bands": len(bands),
        "granularities": [b["k"] for b in bands],
        "bands": bands,
        "best_ari_per_level": best,
    }


# ---------------------------------------------------------------------------
# HDBSCAN* baselines
# ---------------------------------------------------------------------------


def run_hdbscan_grid(D: np.ndarray, truths, min_samples_values=(1, 5)) -> Dict:
    """HDBSCAN* on the precomputed matrix over extraction x min_cluster_size x mpts.

    At ``min_samples=1`` this is single-linkage on `D` itself (TKDD Cor. 3.5), so
    the run isolates the extraction rule. ``min_samples=5`` additionally engages
    the kNN core distance, which is what tests whether a density estimate is
    usable on a non-metric input at all.
    """
    import hdbscan

    Dc = np.ascontiguousarray(D, dtype=np.float64)
    runs = []
    for mpts in min_samples_values:
        for method in ("eom", "leaf"):
            for mcs in MIN_CLUSTER_SIZES:
                if mcs >= len(D) // 2:
                    continue
                try:
                    clusterer = hdbscan.HDBSCAN(
                        metric="precomputed",
                        min_samples=mpts,
                        min_cluster_size=mcs,
                        cluster_selection_method=method,
                        allow_single_cluster=False,
                    )
                    labels = clusterer.fit_predict(Dc.copy())
                except Exception as exc:  # pragma: no cover - library edge cases
                    runs.append(
                        {
                            "min_samples": mpts,
                            "method": method,
                            "min_cluster_size": mcs,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                runs.append(
                    {
                        "min_samples": mpts,
                        "method": method,
                        "min_cluster_size": mcs,
                        "k": _n_clusters(labels),
                        "noise_fraction": float((labels < 0).mean()),
                        "levels": _per_level(labels, truths),
                    }
                )
    return {"runs": runs}


def run_eps_sweep(D: np.ndarray, truths, n_points: int = EPS_SWEEP_POINTS) -> Dict:
    """`dbscan_clustering(eps)` swept over the full range of cut distances.

    At ``min_samples=1`` a cut at height eps is exactly the connected components
    of the eps-threshold graph, i.e. a flat cut of the single-linkage dendrogram.
    Sweeping it enumerates every partition the DBSCAN* family can reach on this
    data, which is the strongest possible form of the "why isn't an eps sweep the
    whole contribution?" objection.

    Records the distinct partitions reached (not merely the distinct k), so the
    count reflects how many candidates a user would have to choose between.
    """
    import hdbscan

    Dc = np.ascontiguousarray(D, dtype=np.float64)
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_samples=1,
        min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
        allow_single_cluster=False,
    )
    clusterer.fit(Dc.copy())

    off = ~np.eye(len(D), dtype=bool)
    lo = float(D[off].min())
    hi = float(D[off].max())
    grid = np.unique(np.linspace(lo, hi, n_points))

    seen: Dict[Tuple[int, ...], Dict] = {}
    for eps in grid:
        try:
            labels = clusterer.dbscan_clustering(
                cut_distance=float(eps), min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE
            )
        except Exception:  # pragma: no cover
            continue
        # canonicalise: relabel by first appearance so equal partitions collapse
        remap: Dict[int, int] = {}
        canon = []
        for v in labels:
            v = int(v)
            if v < 0:
                canon.append(-1)
                continue
            if v not in remap:
                remap[v] = len(remap)
            canon.append(remap[v])
        key = tuple(canon)
        if key in seen:
            continue
        seen[key] = {
            "eps": float(eps),
            "k": _n_clusters(labels),
            "noise_fraction": float((labels < 0).mean()),
            "levels": _per_level(labels, truths),
        }

    partitions = sorted(seen.values(), key=lambda r: r["eps"])
    oracle = {}
    for level, _ in truths:
        cands = [p["levels"][level]["ari"] for p in partitions]
        cands = [c for c in cands if not np.isnan(c)]
        oracle[level] = max(cands) if cands else float("nan")
    return {
        "n_distinct_partitions": len(partitions),
        "k_values_reachable": sorted({p["k"] for p in partitions}),
        "oracle_best_ari_per_level": oracle,
        "partitions": partitions,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _best_run(runs: Sequence[Dict], level: str, method: str | None, mpts: int) -> Dict:
    cands = [
        r
        for r in runs
        if "error" not in r
        and r["min_samples"] == mpts
        and (method is None or r["method"] == method)
        and not np.isnan(r["levels"][level]["ari"])
    ]
    if not cands:
        return {}
    return max(cands, key=lambda r: r["levels"][level]["ari"])


def summarise(results: Dict) -> Dict:
    """Two aggregate analyses that decide what is actually claimable.

    `flat_fixed_setting` is the only fair accuracy comparison: our gate runs at
    one fixed `gap_sigma` on every dataset, so HDBSCAN* must be scored at one
    fixed `(min_samples, method, min_cluster_size)` too. Picking the best
    `min_cluster_size` per dataset -- which an earlier ad-hoc pass effectively did
    in reverse, by reporting only `min_cluster_size=3` -- measures a tuned
    baseline against an untuned one in whichever direction the reporter chose.

    `mcs_sensitivity` records how far HDBSCAN*'s score moves across
    `min_cluster_size` on each dataset. That spread, not the accuracy delta, is
    where the durable difference lies: there is no principled way to set
    `min_cluster_size` from an unlabelled dissimilarity matrix.
    """
    datasets = results["datasets"]
    flat_sets = [
        name
        for name, e in datasets.items()
        if len(e["levels"]) == 1
        and not np.isnan(e["ours_flat"]["levels"][e["levels"][0]]["ari"])
    ]

    ours_aris = [datasets[n]["ours_flat"]["levels"]["only"]["ari"] for n in flat_sets]
    ours_k_right = sum(
        1
        for n in flat_sets
        if datasets[n]["ours_flat"]["k"] == datasets[n]["truth_k"][0]
    )
    fixed = {
        "n_datasets": len(flat_sets),
        "datasets": flat_sets,
        "ours": {
            "setting": "gap_sigma=2.0 (module default, identical on every dataset)",
            "mean_ari": float(np.mean(ours_aris)),
            "k_correct": ours_k_right,
        },
        "hdbscan": [],
    }
    for mpts in (1, 5):
        for method in ("eom", "leaf"):
            for mcs in MIN_CLUSTER_SIZES:
                aris, kr = [], 0
                for name in flat_sets:
                    hit = [
                        r
                        for r in datasets[name]["hdbscan"]["runs"]
                        if r.get("min_samples") == mpts
                        and r.get("method") == method
                        and r.get("min_cluster_size") == mcs
                        and "error" not in r
                    ]
                    if not hit:
                        continue
                    aris.append(hit[0]["levels"]["only"]["ari"])
                    if hit[0]["k"] == datasets[name]["truth_k"][0]:
                        kr += 1
                if aris:
                    fixed["hdbscan"].append(
                        {
                            "min_samples": mpts,
                            "method": method,
                            "min_cluster_size": mcs,
                            "mean_ari": float(np.nanmean(aris)),
                            "k_correct": kr,
                            "n_scored": len(aris),
                        }
                    )
    fixed["best_hdbscan_setting"] = (
        max(fixed["hdbscan"], key=lambda r: r["mean_ari"]) if fixed["hdbscan"] else {}
    )

    sensitivity = {}
    for name, e in datasets.items():
        level = e["levels"][0]
        row = {}
        for method in ("eom", "leaf"):
            vals = [
                r["levels"][level]["ari"]
                for r in e["hdbscan"]["runs"]
                if r.get("min_samples") == 1
                and r.get("method") == method
                and "error" not in r
                and not np.isnan(r["levels"][level]["ari"])
            ]
            if vals:
                row[method] = {
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                    "spread": float(max(vals) - min(vals)),
                }
        if row:
            sensitivity[name] = row
    return {"flat_fixed_setting": fixed, "mcs_sensitivity": sensitivity}


def render_markdown(results: Dict) -> str:
    lines: List[str] = []
    lines.append("# HDBSCAN\\* baselines for the gated selectors")
    lines.append("")
    lines.append(
        "Generated by `gated-minimax-selection/run_hdbscan_baselines.py`. "
        "JSON of record: `outputs/hdbscan_baselines.json`."
    )
    lines.append("")
    lines.append(
        "At `min_samples=1` HDBSCAN\\* is single-linkage on the input dissimilarity "
        "(TKDD 2015 Cor. 3.5), so these rows compare **extraction rules on an "
        "identical hierarchy**. HDBSCAN columns are the best over "
        f"`min_cluster_size` in {list(MIN_CLUSTER_SIZES)} -- i.e. tuned in our "
        "favour's opposite direction."
    )
    lines.append("")

    lines.append("## Flat comparison (finest truth level)")
    lines.append("")
    lines.append(
        "| dataset | family | ours flat: ARI / k | HDBSCAN eom: ARI / k | "
        "HDBSCAN leaf: ARI / k | eps sweep oracle ARI | distinct partitions |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name, res in results["datasets"].items():
        level = res["levels"][0]
        ours = res["ours_flat"]
        runs = res["hdbscan"]["runs"]
        eom = _best_run(runs, level, "eom", 1)
        leaf = _best_run(runs, level, "leaf", 1)
        sweep = res["eps_sweep"]

        def cell(run: Dict) -> str:
            if not run:
                return "n/a"
            return f"{run['levels'][level]['ari']:.3f} / {run['k']}"

        ours_ari = ours["levels"][level]["ari"]
        ours_cell = (
            "n/a (no truth)" if np.isnan(ours_ari) else f"{ours_ari:.3f} / {ours['k']}"
        )
        oracle = sweep["oracle_best_ari_per_level"][level]
        lines.append(
            f"| {name} | {res['family']} | {ours_cell} | {cell(eom)} | {cell(leaf)} "
            f"| {'n/a' if np.isnan(oracle) else f'{oracle:.3f}'} "
            f"| {sweep['n_distinct_partitions']} |"
        )
    lines.append("")

    lines.append("## Nested datasets: can a baseline return the whole hierarchy?")
    lines.append("")
    lines.append(
        "| dataset | truth granularities | ours: bands / granularities | "
        "ours per-level ARI | HDBSCAN leaf per-level ARI | eps-sweep oracle per level |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, res in results["datasets"].items():
        if len(res["levels"]) < 2:
            continue
        ms = res["ours_multiscale"]
        runs = res["hdbscan"]["runs"]
        levels = res["levels"]
        ks = res["truth_k"]
        leaf_cells, ours_cells, sweep_cells = [], [], []
        for level in levels:
            leaf = _best_run(runs, level, "leaf", 1)
            leaf_cells.append(
                "n/a" if not leaf else f"{leaf['levels'][level]['ari']:.3f}"
            )
            v = ms["best_ari_per_level"][level]
            ours_cells.append("n/a" if np.isnan(v) else f"{v:.3f}")
            o = res["eps_sweep"]["oracle_best_ari_per_level"][level]
            sweep_cells.append("n/a" if np.isnan(o) else f"{o:.3f}")
        lines.append(
            f"| {name} | {ks} | {ms['n_bands']} / {ms['granularities']} "
            f"| {', '.join(ours_cells)} | {', '.join(leaf_cells)} "
            f"| {', '.join(sweep_cells)} |"
        )
    lines.append("")

    lines.append("## Does a kNN core distance survive a non-metric input?")
    lines.append("")
    lines.append(
        "`min_samples=5` engages the core-distance/mutual-reachability machinery. "
        "If it runs and scores well on the non-metric families, then "
        '"no density estimator is available here" is not a defensible framing.'
    )
    lines.append("")
    lines.append(
        "| dataset | family | best mpts=1 ARI | best mpts=5 ARI | mpts=5 errored |"
    )
    lines.append("|---|---|---:|---:|---|")
    for name, res in results["datasets"].items():
        level = res["levels"][0]
        runs = res["hdbscan"]["runs"]
        b1 = _best_run(runs, level, None, 1)
        b5 = _best_run(runs, level, None, 5)
        errs = sum(1 for r in runs if r["min_samples"] == 5 and "error" in r)
        tot5 = sum(1 for r in runs if r["min_samples"] == 5)
        c1 = "n/a" if not b1 else f"{b1['levels'][level]['ari']:.3f}"
        c5 = "n/a" if not b5 else f"{b5['levels'][level]['ari']:.3f}"
        lines.append(f"| {name} | {res['family']} | {c1} | {c5} | {errs}/{tot5} |")
    lines.append("")

    summ = results.get("summary")
    if not summ:
        return "\n".join(lines)

    fixed = summ["flat_fixed_setting"]
    lines.append("## The fair comparison: one fixed setting per method")
    lines.append("")
    lines.append(
        f"Mean ARI over the {fixed['n_datasets']} datasets with a flat ground truth. "
        "Our gate is run at its module default on every dataset; each HDBSCAN\\* row "
        "is likewise one setting held fixed across all of them. Choosing "
        "`min_cluster_size` per dataset is not available to either method without "
        "labels."
    )
    lines.append("")
    lines.append("| method | setting | mean ARI | k correct |")
    lines.append("|---|---|---:|---:|")
    lines.append(
        f"| **ours (gated set-cover)** | {fixed['ours']['setting']} "
        f"| **{fixed['ours']['mean_ari']:.3f}** "
        f"| {fixed['ours']['k_correct']}/{fixed['n_datasets']} |"
    )
    for r in sorted(fixed["hdbscan"], key=lambda r: -r["mean_ari"]):
        lines.append(
            f"| HDBSCAN\\* {r['method']} | mpts={r['min_samples']}, "
            f"min_cluster_size={r['min_cluster_size']} | {r['mean_ari']:.3f} "
            f"| {r['k_correct']}/{fixed['n_datasets']} |"
        )
    lines.append("")

    lines.append("## Where the durable difference is: `min_cluster_size` sensitivity")
    lines.append("")
    lines.append(
        "How far HDBSCAN\\*'s ARI moves across `min_cluster_size` in "
        f"{list(MIN_CLUSTER_SIZES)} at mpts=1, against our single-setting score. "
        "A large spread means the baseline's quality is set by a parameter with no "
        "unsupervised criterion behind it. On the four *nested* datasets the "
        "`ours (fixed)` column is the **flat** gate scored against the finest level, "
        "which it does not target -- the band selector is the like-for-like "
        "comparison there, and it is in the nested table above."
    )
    lines.append("")
    lines.append("| dataset | eom range | eom spread | leaf spread | ours (fixed) |")
    lines.append("|---|---|---:|---:|---:|")
    for name, row in summ["mcs_sensitivity"].items():
        res = results["datasets"][name]
        level = res["levels"][0]
        o = res["ours_flat"]["levels"][level]["ari"]
        eom = row.get("eom")
        leaf = row.get("leaf")
        eom_range = "n/a" if not eom else f"[{eom['min']:.3f}, {eom['max']:.3f}]"
        eom_spread = "n/a" if not eom else f"{eom['spread']:.3f}"
        leaf_spread = "n/a" if not leaf else f"{leaf['spread']:.3f}"
        ours_cell = "n/a" if np.isnan(o) else f"{o:.3f}"
        lines.append(
            f"| {name} | {eom_range} | {eom_spread} | {leaf_spread} | {ours_cell} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="outputs/hdbscan_baselines.json",
        help="JSON of record to write (markdown goes alongside it).",
    )
    ap.add_argument(
        "--only", default=None, help="Comma-separated dataset names to restrict to."
    )
    args = ap.parse_args(argv)

    specs = _dataset_specs()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s["name"] in wanted]

    results: Dict = {
        "config": {
            "min_cluster_sizes": list(MIN_CLUSTER_SIZES),
            "default_min_cluster_size": DEFAULT_MIN_CLUSTER_SIZE,
            "eps_sweep_points": EPS_SWEEP_POINTS,
            "hdbscan_impl": "hdbscan (contrib)",
        },
        "datasets": {},
    }

    for spec in specs:
        name, D, truths = spec["name"], spec["D"], spec["truths"]
        print(f"[{name}] n={len(D)} ...", flush=True)
        entry = {
            "family": spec["family"],
            "n": int(len(D)),
            "levels": [lvl for lvl, _ in truths],
            "truth_k": [int(len(set(y[y >= 0]))) for _, y in truths],
            "ours_flat": run_ours_flat(D, truths),
            "ours_multiscale": run_ours_multiscale(D, truths),
            "hdbscan": run_hdbscan_grid(D, truths),
            "eps_sweep": run_eps_sweep(D, truths),
        }
        results["datasets"][name] = entry
        print(
            f"    ours flat k={entry['ours_flat']['k']}, "
            f"bands={entry['ours_multiscale']['granularities']}, "
            f"eps partitions={entry['eps_sweep']['n_distinct_partitions']}",
            flush=True,
        )

    results["summary"] = summarise(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    md = out.with_suffix(".md")
    md.write_text(render_markdown(results))
    print(f"\nwrote {out}\nwrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
