"""Unit tests for the pure table-formatting functions touched by the
itertuples refactor (commit 52830cb): report.md_table, write_artifact.table_html,
write_artifact.fidelity_html, write_benchmark.to_target_table.

These are pure formatting functions with no I/O, so all inputs here are small
synthetic DataFrames/dicts -- no real CMAPSS data needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import report
import write_artifact
import write_benchmark


# ---------------------------------------------------------------------------
# report.md_table
# ---------------------------------------------------------------------------


def test_md_table_normal_case():
    df = pd.DataFrame(
        {
            "arm": ["fis", "he"],
            "rmse": [11.23, 10.50],
            "n_parameters": [42, 100],
        }
    )
    out = report.md_table(
        df,
        ["arm", "rmse", "n_parameters"],
        ["model", "test RMSE", "params"],
        [None, "{:.2f}", "{:.0f}"],
    )
    lines = out.splitlines()
    assert lines[0] == "| model | test RMSE | params |"
    assert "| fis | 11.23 | 42 |" in out
    assert "| he | 10.50 | 100 |" in out


def test_md_table_missing_column_falls_back_to_nan_dashes():
    # `cols` names a column ("n_parameters") that is NOT in the DataFrame.
    # itertuples + getattr(r, c, np.nan) must degrade gracefully to "--",
    # matching iterrows' old row.get(c) behavior, instead of raising
    # AttributeError.
    df = pd.DataFrame({"arm": ["fis", "he"], "rmse": [11.23, 10.50]})
    out = report.md_table(
        df,
        ["arm", "rmse", "n_parameters"],
        ["model", "test RMSE", "params"],
        [None, "{:.2f}", "{:.0f}"],
    )
    assert "| fis | 11.23 | -- |" in out
    assert "| he | 10.50 | -- |" in out


def test_md_table_no_format_string_uses_str():
    df = pd.DataFrame({"arm": ["fis"], "label": ["seed"]})
    out = report.md_table(df, ["arm", "label"], ["model", "label"], [None, None])
    assert "| fis | seed |" in out


# ---------------------------------------------------------------------------
# write_artifact.table_html
# ---------------------------------------------------------------------------


def _arm_df():
    return pd.DataFrame(
        {
            "arm": ["fis", "he"],
            "n_parameters": [42.0, 100.0],
            "rmse": [11.23, 10.50],
            "mae": [8.0, 7.5],
            "rmse_endpoint": [5.0, 4.5],
            "total_s": [1.234, 2.345],
        }
    )


def test_table_html_normal_case():
    df = _arm_df()
    out = write_artifact.table_html(df, fis_rmse=11.23)
    assert "42" in out
    assert "11.23" in out
    assert "10.50" in out
    # he beats fis's rmse -> " win" class on that cell
    assert "n win" in out
    # fis row itself never gets "win"
    assert "class='fis'" in out


def test_table_html_missing_n_parameters_renders_dashes_not_raise():
    df = pd.DataFrame(
        {
            "arm": ["fis", "he"],
            "rmse": [11.23, 10.50],
            "mae": [8.0, 7.5],
            "rmse_endpoint": [5.0, 4.5],
            "total_s": [1.234, 2.345],
        }
    )
    out = write_artifact.table_html(df, fis_rmse=11.23)
    assert "<td class='n'>--</td>" in out


# ---------------------------------------------------------------------------
# write_artifact.fidelity_html
# ---------------------------------------------------------------------------


def test_fidelity_html_bad_class_threshold():
    fid = {
        "rows": [
            dict(
                top_n=5,
                n_features=5,
                n_hidden=10,
                fidelity_relative=1.5,  # > 1 -> "bad"
                additive_relative=0.9,
                frac_rows_outside_knots=0.1,
            ),
            dict(
                top_n=8,
                n_features=8,
                n_hidden=16,
                fidelity_relative=0.8,  # <= 1 -> not "bad"
                additive_relative=0.7,
                frac_rows_outside_knots=0.2,
            ),
        ]
    }
    out = write_artifact.fidelity_html(fid)
    rows = [line for line in out.splitlines() if line.startswith("<tr>")]
    assert len(rows) == 2
    bad_rows = [r for r in rows if " bad" in r]
    assert len(bad_rows) == 1
    assert "1.500" in bad_rows[0]
    good_row = [r for r in rows if "0.800" in r][0]
    assert " bad" not in good_row


def test_fidelity_html_filters_zero_top_n():
    fid = {
        "rows": [
            dict(
                top_n=0,
                n_features=3,
                n_hidden=5,
                fidelity_relative=2.0,
                additive_relative=0.5,
                frac_rows_outside_knots=0.0,
            ),
            dict(
                top_n=4,
                n_features=4,
                n_hidden=6,
                fidelity_relative=0.5,
                additive_relative=0.5,
                frac_rows_outside_knots=0.0,
            ),
        ]
    }
    out = write_artifact.fidelity_html(fid)
    rows = [line for line in out.splitlines() if line.startswith("<tr>")]
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# write_benchmark.to_target_table
# ---------------------------------------------------------------------------


def _to_target_entry(total_seconds):
    return dict(epoch=1.0, train_seconds=total_seconds - 0.1, total_seconds=total_seconds, rmse=9.0)


def test_to_target_table_median_vs_never_branches():
    # Target "rmse_10": 3 of 4 rows (majority, > half) reach it -> median reported.
    # Target "rmse_8": 1 of 4 rows (not > half) reach it -> "never".
    res = {
        "references": {"fis": {"test": {"rmse": 11.23}}},
        "arms": [
            dict(
                arm="he",
                to_target={
                    "fis_parity": _to_target_entry(1.0),
                    "rmse_12": _to_target_entry(1.0),
                    "rmse_10": _to_target_entry(2.0),
                    "rmse_9": None,
                    "rmse_8": _to_target_entry(5.0),
                },
            ),
            dict(
                arm="he",
                to_target={
                    "fis_parity": _to_target_entry(1.0),
                    "rmse_12": _to_target_entry(1.0),
                    "rmse_10": _to_target_entry(3.0),
                    "rmse_9": None,
                    "rmse_8": None,
                },
            ),
            dict(
                arm="he",
                to_target={
                    "fis_parity": _to_target_entry(1.0),
                    "rmse_12": _to_target_entry(1.0),
                    "rmse_10": _to_target_entry(4.0),
                    "rmse_9": None,
                    "rmse_8": None,
                },
            ),
            dict(
                arm="he",
                to_target={
                    "fis_parity": _to_target_entry(1.0),
                    "rmse_12": _to_target_entry(1.0),
                    "rmse_10": None,
                    "rmse_9": None,
                    "rmse_8": None,
                },
            ),
        ],
    }
    out = write_benchmark.to_target_table(res)
    lines = out.splitlines()
    # header includes the FIS parity value interpolated in
    assert "FIS parity (11.23)" in lines[0]
    data_row = lines[2]
    cells = [c.strip() for c in data_row.strip("|").split("|")]
    # cells: model, fis_parity, rmse_12, rmse_10, rmse_9, rmse_8
    assert cells[0] == report.ARM_LABEL.get("he", "he")
    assert cells[3] == f"{np.median([2.0, 3.0, 4.0]):.3f}"  # rmse_10: 3/4 -> median
    assert cells[4] == "never"  # rmse_9: 0/4
    assert cells[5] == "never"  # rmse_8: 1/4, not > half
