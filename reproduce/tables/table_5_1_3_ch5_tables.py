"""Chapter 5 Tables 5.1, 5.2, 5.3 -- emitted from the gated-minimax results of record.

Chapter 5's pipeline is already deterministic: `gated-minimax-selection/run_all.py`
seeds everything and writes one `outputs/results.json` holding every number. What
was missing is this last step -- the three prose tables were transcribed from that
JSON by hand, so drift between the JSON and the chapter was undetectable in a way
it is not for the other chapters' tables.

This script does no computation. It reads the JSON of record and renders it, which
means a stale prose cell now shows up as a diff instead of going unnoticed. Run
`run_all.py` first if the JSON needs regenerating; it is the expensive part.

Each table below names the exact JSON field behind every column, because several
Chapter 5 columns come from different result blocks and the mapping is not obvious.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "gated-minimax-selection", "outputs", "results.json")

# Prose row label -> results.json dataset key. The chapter calls the two-Gaussian
# set "well_separated"; the driver calls it "two_gaussians".
BATTERY_ROWS = [
    ("concentric_rings", "concentric_rings"),
    ("bridged_gaussians", "bridged_gaussians"),
    ("well_separated", "two_gaussians"),
    ("varying_density", "varying_density"),
    ("uniform_noise", "uniform_noise"),
]

MULTISCALE_ROWS = ["nested_gaussians", "three_level_hierarchy", "density_hierarchy"]

SELECTORS = [
    ("persistence-gap gate (ours)", "persistence_gap"),
    ("beta-plateau [Bonis-Oudot]", "beta_plateau"),
    ("bottleneck-bootstrap [AuToMATo]", "bottleneck_bootstrap"),
]


def _f(value, fmt="{:.3f}"):
    """Render a JSON number, or the NA marker when the driver recorded null."""
    return C.NA if value is None else fmt.format(value)


def seed_note(results):
    """What to put in the footer's `seeds =` slot, taken from the driver's own record.

    This script computes nothing, so the harness's `common.SEEDS` describes nothing
    about these three tables -- and the footer used to print it anyway, stamping ten
    seeds onto a run made at five. `run_all.py` now records the seed sets it used, so
    the footer can state them.

    A JSON written before that change has no `seeds` key, and the honest thing to print
    then is that the run did not record it. Substituting `common.SEEDS` is what created
    the problem; substituting the values we happen to believe the driver uses would
    recreate it one step removed, because the next time a driver default moves the
    footer would go stale again silently.
    """
    seeds = results.get("seeds")
    if not seeds:
        return (
            "unrecorded -- this results.json predates the driver recording its "
            "own seed sets; re-run `run_all.py` to fill it in"
        )
    if isinstance(seeds, dict):
        return "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in seeds.items())
    return str(seeds)


def table_5_1(results):
    """The battery: baselines given k, versus the gate discovering it.

    Two ARIs are reported for the set-cover, and the difference between them is
    the whole bridged-Gaussians story. `main_table.cover_ari` scores only the
    points the cover actually claims; `persistence_methods.persistence_gap.ari`
    scores the full dataset, counting everything left uncovered as unassigned.
    On a cover with 53% coverage those are 0.982 and 0.001 respectively -- the
    same cover, judged on covered points versus on all points. Reporting only
    the first would flatter the method badly, so both are in the table.
    """
    main = results["main_table"]
    pm = results["persistence_methods"]
    header = [
        "Dataset",
        "single-linkage on D",
        "NERFCM on raw D",
        "NERFCM on D* (given k)",
        "ConiVAT (constrained)",
        "set-cover, covered pts",
        "set-cover, all pts (gated)",
        "k discovered",
        "coverage",
    ]
    rows = []
    for label, key in BATTERY_ROWS:
        r = main[key]
        gate = pm.get(key, {}).get("methods", {}).get("persistence_gap", {})
        rows.append(
            [
                label,
                _f(r.get("iVAT_SL_ari")),
                _f(r.get("NERFCM_D_ari")),
                _f(r.get("NERFCM_Dstar_ari")),
                _f(r.get("ConiVAT_ari")),
                _f(r.get("cover_ari")),
                _f(gate.get("ari")),
                r.get("cover_nblocks", C.NA),
                _f(r.get("cover_coverage")),
            ]
        )
    C.emit(
        "table_5_1_battery",
        "Table 5.1 -- The battery (adjusted Rand index)",
        header,
        rows,
        note=(
            "Baseline columns are `main_table.{iVAT_SL_ari, NERFCM_D_ari, "
            "NERFCM_Dstar_ari, ConiVAT_ari}`; both NERFCM columns and ConiVAT are "
            "GIVEN k, while the set-cover discovers it, which is the comparison the "
            "chapter rests on. The two set-cover columns are the SAME cover scored "
            "differently: `main_table.cover_ari` over covered points only, and "
            "`persistence_methods.<ds>.methods.persistence_gap.ari` over all points "
            "with uncovered ones unassigned. Quote the all-points column in prose; "
            "the covered-points column is diagnostic. On uniform_noise there is no "
            "ground-truth partition, so ARI is null by construction and the "
            "abstention has to be read off coverage (0.125) instead."
        ),
        seeds=seed_note(results),
    )


def table_5_2(results):
    """Multi-scale recovery: flat cover collapses a hierarchy, band-wise does not."""
    ms = results["multiscale_hierarchy"]
    header = [
        "Dataset",
        "levels",
        "flat cover (mean ARI)",
        "multi-scale (mean ARI)",
        "granularities recovered",
        "flat k",
    ]
    rows = []
    for key in MULTISCALE_ROWS:
        r = ms[key]
        rows.append(
            [
                key,
                "/".join(r.get("level_names", [])),
                _f(r.get("flat_mean_ari")),
                _f(r.get("ms_mean_ari")),
                str(r.get("ms_granularities", C.NA)),
                r.get("flat_k", C.NA),
            ]
        )
    C.emit(
        "table_5_2_multiscale",
        "Table 5.2 -- Multi-scale recovery (ARI, averaged over all ground-truth levels)",
        header,
        rows,
        note=(
            "From `multiscale_hierarchy.*` in results.json: flat_mean_ari, ms_mean_ari, "
            "ms_granularities. Averaging over ALL ground-truth levels is what makes the "
            "flat column look bad -- a flat cover lands one level exactly and misses the "
            "rest, so its per-level scores are recorded in flat_ari_per_level."
        ),
        seeds=seed_note(results),
    )


def table_5_3(results):
    """Selection bake-off: bridge-robustness against noise-conservatism."""
    pm = results["persistence_methods"]
    bridge, noise = pm["bridged_gaussians"], pm["uniform_noise"]
    header = [
        "Selection method",
        "bridge case (ARI)",
        "bridge k",
        "bridge coverage",
        "noise k (true: none)",
        "noise coverage",
    ]
    rows = []
    for label, key in SELECTORS:
        b, n = bridge["methods"][key], noise["methods"][key]
        rows.append(
            [
                label,
                _f(b.get("ari")),
                b.get("k_discovered", C.NA),
                _f(b.get("coverage")),
                n.get("k_discovered", C.NA),
                _f(n.get("coverage")),
            ]
        )
    C.emit(
        "table_5_3_selection",
        "Table 5.3 -- Selection-method comparison",
        header,
        rows,
        note=(
            "From `persistence_methods.{bridged_gaussians, uniform_noise}.methods.*`. "
            "ARI on uniform_noise is null by construction -- there is no ground-truth "
            "partition to score against -- so noise behaviour has to be read off k and "
            "coverage instead. Low coverage at any k IS the abstention: the gate blocks "
            "the blocks rather than reporting none. No universal winner; bridge-"
            "robustness and noise-conservatism trade off against one another."
        ),
        seeds=seed_note(results),
    )


def main():
    if not os.path.exists(RESULTS):
        print(
            f"  [skip] {RESULTS} not found -- run "
            f"`python run_all.py` in gated-minimax-selection/ first."
        )
        return 1
    with open(RESULTS, encoding="utf-8") as fh:
        results = json.load(fh)
    print("Chapter 5 tables from the gated-minimax results of record")
    print(f"  source: {RESULTS}")
    table_5_1(results)
    table_5_2(results)
    table_5_3(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
