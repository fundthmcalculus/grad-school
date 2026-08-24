"""Unit tests for run_all._print_summary and run_all._run_scaling_only.

These two functions were extracted out of run_all.main() in commit 0697de2 and
have no prior direct test coverage. Both are exercised here with small
synthetic/mocked inputs rather than the real (expensive / plotting) pipeline.
"""

import json

import pytest

import run_all


# ---------------------------------------------------------------------------
# _print_summary
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_results():
    """_print_summary reads the module-level `results` dict directly (for the
    ruspini / feature_space sections). Snapshot and restore it so this test
    doesn't leak state into other tests run in the same session."""
    saved = run_all.results
    run_all.results = {}
    yield run_all.results
    run_all.results = saved


def test_print_summary_prints_expected_values(capsys, saved_results):
    table = {
        "dataset_a": {
            "iVAT_SL_ari": 0.91,
            "NERFCM_D_ari": 0.55,
            "NERFCM_Dstar_ari": 0.87,
            "ConiVAT_ari": 0.60,
            "cover_ari": 0.75,
            "cover_nblocks": 3,
            "cover_coverage": 0.99,
            "mapping1_ari": 0.80,
            "mapping2_ari": 0.82,
            "kmeans_ari": 0.70,
            "mapping2_coverage": 0.95,
            "mapping2_convexity": 0.5,
        }
    }
    relational_table = {
        "rel_a": {
            "NERFCM_D_ari": 0.40,
            "NERFCM_Dstar_ari": 0.65,
            "k_true": 4,
            "n": 120,
        },
        "rel_b_missing": {
            "NERFCM_D_ari": None,
            "NERFCM_Dstar_ari": None,
            "k_true": 2,
            "n": 50,
        },
    }
    multiscale_table = {
        "ms_a": {
            "flat_k": 3,
            "flat_mean_ari": 0.5,
            "ms_n_scales": 2,
            "ms_granularities": [3, 6],
            "ms_mean_ari": 0.93,
            "flat_ari_per_level": [0.6, 0.4],
            "ms_best_ari_per_level": [0.95, 0.91],
        }
    }
    persistence_methods = {
        "pm_a": {
            "k_true": 3,
            "methods": {
                "gap": {"k_discovered": 3, "coverage": 0.98, "ari": 0.88},
                "beta_plateau": {"k_discovered": 3, "coverage": 0.97, "ari": 0.85},
            },
        }
    }

    run_all.results["ruspini"] = {
        "dataset_a": {"ari": 0.77, "partition_error_max": 0.02},
        "noise_ds": {"status": "noise_rejection"},
    }
    run_all.results["feature_space"] = {
        "dataset_a": {
            "ari_dissimilarity": 0.81,
            "ari_feature_space": 0.79,
            "surrogate_l2_mean": 0.03,
            "n_rules": 5,
        }
    }

    run_all._print_summary(table, relational_table, multiscale_table, persistence_methods)

    out = capsys.readouterr().out

    # header / OUT path
    assert "Done. Results and figures written to" in out
    assert run_all.OUT in out

    # main table
    assert "dataset_a: iVAT-SL=0.91 NERFCM(D)=0.55" in out
    assert "NERFCM(D*)=0.87 ConiVAT=0.6" in out
    assert "cover=0.75 (k=3, cov=0.99)" in out

    # relational table, including the None -> "n/a" branch and delta formatting
    assert "rel_a: NERFCM(D)=0.4 NERFCM(D*)=0.65 ΔAI=+0.250 (k=4, n=120)" in out
    assert "rel_b_missing: NERFCM(D)=n/a NERFCM(D*)=n/a ΔAI=n/a (k=2, n=50)" in out

    # multiscale table
    assert "ms_a: flat(k=3)=0.5" in out
    assert "multi-scale(scales=2, k=[3, 6])=0.93" in out
    assert "flat/level=[0.6, 0.4] ms/level=[0.95, 0.91]" in out

    # mappings + baselines (folded into main table)
    assert "dataset_a: M1=0.8 M2=0.82 kmeans=0.7 M2cov=0.95 M2convex=0.5" in out

    # persistence selection methods
    assert "pm_a (k_true=3): gap: k=3 cov=0.98 ARI=0.88 | beta_plateau: k=3 cov=0.97 ARI=0.85" in out

    # membership variants: ruspini + feature-space, including noise-rejection branch
    assert "dataset_a: RuspiniARI=0.77 POU_err(max)=0.02" in out
    assert "autoTunedARI=0.81 featARI=0.79 L2=0.03 rules=5" in out
    assert "noise_ds: (noise rejected)" in out


# ---------------------------------------------------------------------------
# _run_scaling_only
# ---------------------------------------------------------------------------


def test_run_scaling_only_writes_fake_results(monkeypatch, tmp_path):
    fake_table = {
        "single_scale": {
            "level_names": ["level0"],
            "expected": [3],
            "rows": [{"N": 10, "n": 10, "t_transform": 0.001, "t_select": 0.001}],
        }
    }

    calls = {"benchmark": 0, "fig": 0}

    def fake_run_scaling_benchmark(*args, **kwargs):
        calls["benchmark"] += 1
        return fake_table

    def fake_fig_scaling(scaling_table):
        calls["fig"] += 1
        # sanity: called with the table the fake benchmark returned
        assert scaling_table is fake_table

    monkeypatch.setattr(run_all, "run_scaling_benchmark", fake_run_scaling_benchmark)
    monkeypatch.setattr(run_all, "fig_scaling", fake_fig_scaling)
    # OUT is looked up at call time inside the function body (f"{OUT}/..."),
    # not captured at import/def time, so redirecting it here keeps this test
    # from touching the real gated-minimax-selection/outputs/ directory.
    monkeypatch.setattr(run_all, "OUT", str(tmp_path))

    run_all._run_scaling_only()

    assert calls["benchmark"] == 1
    assert calls["fig"] == 1

    out_file = tmp_path / "scaling_results.json"
    assert out_file.exists()
    with open(out_file) as f:
        written = json.load(f)
    assert written == {"scaling": fake_table}
