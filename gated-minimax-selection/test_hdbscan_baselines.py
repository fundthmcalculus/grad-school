"""Tests for the HDBSCAN* baseline harness.

Everything here runs WITHOUT the `hdbscan` contrib package: the aggregation and
rendering layers are pure functions over a results dict, and the two selector
wrappers use only in-repo code. The `hdbscan`-dependent functions
(`run_hdbscan_grid`, `run_eps_sweep`) import the library lazily inside the
function body precisely so this file is collectable in CI, where it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

import battery as B
import run_hdbscan_baselines as H

# ---------------------------------------------------------------------------
# scoring primitives
# ---------------------------------------------------------------------------


def test_n_clusters_ignores_the_noise_flag():
    assert H._n_clusters(np.array([0, 0, 1, 1])) == 2
    assert H._n_clusters(np.array([-1, -1, 0, 0])) == 1
    assert H._n_clusters(np.array([-1, -1, -1])) == 0


def test_score_drops_ground_truth_negatives():
    """Bridge points (truth -1) belong to no true cluster and must not be scored."""
    y = np.array([0, 0, 1, 1, -1, -1])
    perfect = np.array([0, 0, 1, 1, 7, 7])  # arbitrary labels on the bridge
    assert H.score(perfect, y)["ari"] == pytest.approx(1.0)


def test_score_reports_both_noise_conventions():
    y = np.array([0, 0, 0, 1, 1, 1])
    # correct on everything it claims, abstains on one point
    labels = np.array([0, 0, -1, 1, 1, 1])
    out = H.score(labels, y)
    assert out["ari_noise_excluded"] == pytest.approx(1.0)
    # keeping -1 as its own label costs ARI, so the two must differ here
    assert out["ari"] < out["ari_noise_excluded"]


def test_score_is_nan_when_there_is_no_ground_truth():
    y = np.array([-1, -1, -1, -1])
    out = H.score(np.array([0, 0, 1, 1]), y)
    assert np.isnan(out["ari"])
    assert np.isnan(out["ari_noise_excluded"])


# ---------------------------------------------------------------------------
# our selectors, through the harness wrappers
# ---------------------------------------------------------------------------


def test_ours_flat_recovers_two_clean_gaussians():
    X, y = B.two_gaussians()
    res = H.run_ours_flat(H._euclid(X), [("only", y)])
    assert res["k"] == 2
    assert res["levels"]["only"]["ari"] == pytest.approx(1.0)


def test_ours_multiscale_recovers_the_nested_stack():
    """The [8, 4, 2] synthetic is the dataset the prior-art review turns on."""
    import battery_hierarchical as BH

    X, y_f, y_m, y_c = BH.three_level_hierarchy()
    truths = [("fine", y_f), ("medium", y_m), ("coarse", y_c)]
    res = H.run_ours_multiscale(H._euclid(X), truths)
    assert res["n_bands"] == 3
    assert res["granularities"] == [8, 4, 2]
    for level in ("fine", "medium", "coarse"):
        assert res["best_ari_per_level"][level] == pytest.approx(1.0)


def test_ours_multiscale_finds_no_bands_in_noise():
    X, y = B.uniform_noise()
    res = H.run_ours_multiscale(H._euclid(X), [("only", y)])
    assert res["n_bands"] == 0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _fake_results() -> dict:
    """A two-dataset results dict with hand-checkable aggregates."""

    def lvl(ari):
        return {"only": {"ari": ari, "ari_noise_excluded": ari}}

    def run(mpts, method, mcs, k, ari):
        return {
            "min_samples": mpts,
            "method": method,
            "min_cluster_size": mcs,
            "k": k,
            "noise_fraction": 0.0,
            "levels": lvl(ari),
        }

    def dataset(ours_ari, ours_k, truth_k, aris):
        runs = []
        for mcs, ari in zip(H.MIN_CLUSTER_SIZES, aris):
            runs.append(run(1, "eom", mcs, truth_k, ari))
            runs.append(run(1, "leaf", mcs, truth_k, ari / 2))
        return {
            "family": "flat",
            "n": 10,
            "levels": ["only"],
            "truth_k": [truth_k],
            "ours_flat": {"k": ours_k, "coverage": 1.0, "levels": lvl(ours_ari)},
            "ours_multiscale": {
                "n_bands": 1,
                "granularities": [ours_k],
                "bands": [],
                "best_ari_per_level": {"only": ours_ari},
            },
            "hdbscan": {"runs": runs},
            "eps_sweep": {
                "n_distinct_partitions": 4,
                "k_values_reachable": [1, 2],
                "oracle_best_ari_per_level": {"only": 1.0},
                "partitions": [],
            },
        }

    return {
        "config": {},
        "datasets": {
            "a": dataset(1.0, 2, 2, [0.2, 0.6, 1.0]),
            "b": dataset(0.5, 3, 2, [0.4, 0.4, 0.4]),
        },
    }


def test_summarise_means_are_over_a_single_fixed_setting():
    summ = H.summarise(_fake_results())
    fixed = summ["flat_fixed_setting"]
    assert fixed["n_datasets"] == 2
    # ours: (1.0 + 0.5) / 2
    assert fixed["ours"]["mean_ari"] == pytest.approx(0.75)
    # dataset a's k matches truth, b's does not
    assert fixed["ours"]["k_correct"] == 1
    by_mcs = {
        r["min_cluster_size"]: r
        for r in fixed["hdbscan"]
        if r["min_samples"] == 1 and r["method"] == "eom"
    }
    # each row averages ACROSS datasets at ONE mcs -- never the per-dataset best
    assert by_mcs[3]["mean_ari"] == pytest.approx(0.3)  # (0.2 + 0.4) / 2
    assert by_mcs[5]["mean_ari"] == pytest.approx(0.5)  # (0.6 + 0.4) / 2
    assert by_mcs[10]["mean_ari"] == pytest.approx(0.7)  # (1.0 + 0.4) / 2
    # the best fixed setting is mcs=10, and it is still below ours
    assert fixed["best_hdbscan_setting"]["min_cluster_size"] == 10
    assert fixed["best_hdbscan_setting"]["mean_ari"] < fixed["ours"]["mean_ari"]


def test_summarise_records_the_min_cluster_size_spread():
    """The spread, not the accuracy delta, is the claim -- so pin it."""
    summ = H.summarise(_fake_results())
    sens = summ["mcs_sensitivity"]
    assert sens["a"]["eom"]["spread"] == pytest.approx(0.8)  # 1.0 - 0.2
    assert sens["b"]["eom"]["spread"] == pytest.approx(0.0)
    assert sens["a"]["leaf"]["spread"] == pytest.approx(0.4)  # halved aris


def test_summarise_excludes_datasets_with_no_ground_truth():
    """uniform_noise has no partition, so it cannot enter an ARI mean."""
    results = _fake_results()
    noise = results["datasets"]["a"].copy()
    noise["ours_flat"] = {
        "k": 4,
        "coverage": 0.125,
        "levels": {"only": {"ari": float("nan"), "ari_noise_excluded": float("nan")}},
    }
    results["datasets"]["noise"] = noise
    summ = H.summarise(results)
    assert summ["flat_fixed_setting"]["n_datasets"] == 2
    assert "noise" not in summ["flat_fixed_setting"]["datasets"]


def test_render_markdown_runs_and_reports_both_analyses():
    results = _fake_results()
    results["summary"] = H.summarise(results)
    md = H.render_markdown(results)
    assert "The fair comparison" in md
    assert "min_cluster_size` sensitivity" in md
    # every dataset gets a row in the flat table
    for name in results["datasets"]:
        assert f"| {name} |" in md


def test_render_markdown_tolerates_a_missing_summary():
    md = H.render_markdown(_fake_results())
    assert "Flat comparison" in md
    assert "The fair comparison" not in md


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


def test_replicate_zero_uses_the_generators_own_defaults():
    """So --seeds 1 reproduces every number published before seeding existed."""
    assert H._seed_kwargs(0) == {}
    assert H._seed_kwargs(1) == {"seed": H.SEED_BASE + 1}
    assert H._seed_kwargs(9) == {"seed": H.SEED_BASE + 9}


def test_replicate_zero_matches_an_unseeded_call():
    X, y = B.two_gaussians()
    spec = next(s for s in H._dataset_specs(0) if s["name"] == "two_gaussians")
    np.testing.assert_array_equal(spec["D"], H._euclid(X))
    np.testing.assert_array_equal(spec["truths"][0][1], y)


def test_later_replicates_produce_different_data():
    a = next(s for s in H._dataset_specs(0) if s["name"] == "concentric_rings")
    b = next(s for s in H._dataset_specs(3) if s["name"] == "concentric_rings")
    assert not np.allclose(a["D"], b["D"])
    # ...but the same shape and the same ground-truth structure
    assert a["D"].shape == b["D"].shape
    assert set(a["truths"][0][1]) == set(b["truths"][0][1])


def test_every_dataset_accepts_a_seed_override():
    """A generator that silently ignored `seed=` would fake the whole spread."""
    base = {s["name"]: s["D"] for s in H._dataset_specs(0)}
    other = {s["name"]: s["D"] for s in H._dataset_specs(5)}
    assert set(base) == set(other)
    unchanged = [n for n in base if np.allclose(base[n], other[n])]
    assert not unchanged, f"seed had no effect on: {unchanged}"


def test_summarise_across_seeds_averages_whole_batteries():
    reps = [_fake_results()["datasets"] for _ in range(3)]
    # make replicate 1 and 2 differ so the spread is non-degenerate
    reps[1]["a"]["ours_flat"]["levels"]["only"]["ari"] = 0.5
    reps[2]["a"]["ours_flat"]["levels"]["only"]["ari"] = 0.0
    across = H.summarise_across_seeds(reps)
    assert across["n_replicates"] == 3
    # per-replicate battery means are (1.0+0.5)/2, (0.5+0.5)/2, (0.0+0.5)/2
    assert across["ours"]["mean"] == pytest.approx((0.75 + 0.5 + 0.25) / 3)
    assert across["ours"]["min"] == pytest.approx(0.25)
    assert across["ours"]["max"] == pytest.approx(0.75)
    assert across["ours"]["std"] > 0


def test_summarise_across_seeds_reports_band_stability():
    """Nested datasets only; the flat ones have a single truth level."""
    reps = []
    for grans in ([8, 4, 2], [8, 4, 2], [4, 2]):
        rep = _fake_results()["datasets"]
        rep["nested"] = {
            "family": "hierarchical",
            "n": 96,
            "levels": ["fine", "coarse"],
            "truth_k": [8, 4, 2],
            "ours_flat": {
                "k": 2,
                "coverage": 1.0,
                "levels": {
                    lv: {"ari": 1.0, "ari_noise_excluded": 1.0}
                    for lv in ("fine", "coarse")
                },
            },
            "ours_multiscale": {
                "n_bands": len(grans),
                "granularities": grans,
                "bands": [],
                "best_ari_per_level": {"fine": 1.0, "coarse": 1.0},
            },
            "hdbscan": {"runs": []},
            "eps_sweep": {
                "n_distinct_partitions": 7,
                "k_values_reachable": [2, 4, 8],
                "oracle_best_ari_per_level": {"fine": 1.0, "coarse": 1.0},
                "partitions": [],
            },
        }
        reps.append(rep)
    stab = H.summarise_across_seeds(reps)["band_recovery_stability"]
    assert set(stab) == {"nested"}
    row = stab["nested"]
    assert row["modal_granularities"] == [8, 4, 2]
    assert row["modal_agreement"] == pytest.approx(2 / 3)
    assert row["exact_truth_match"] == pytest.approx(2 / 3)
    assert row["distinct_vectors"] == 2
