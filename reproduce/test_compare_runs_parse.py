#!/usr/bin/env python3
"""`parse_cell` must read the number a table cell is *about*.

The cells the harness emits carry prefixes and units -- "R2=0.644 ± 0.015",
"0.60 ± 0.04 s", "7.78 ± 0.50 MPa" -- so `parse_cell` searches rather than
full-matches. `re.search` returns the LEFTMOST match, which made "R2=0.939"
parse as the number 2: both sides of a comparison came back 2.0, every R2 cell
in every FIX_IMPACT report was reported "within noise" with a delta of exactly
zero, and regression drift was invisible for as long as the report existed.
"acc=0.729" was fine only by luck -- "acc" contains no digit.

These pin the prefixes and units actually present in the archived CSVs, so a
future tolerance-widening cannot re-open the same hole.

    uv run python reproduce/test_compare_runs_parse.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_runs import parse_cell  # noqa: E402


class TestParseCell(unittest.TestCase):
    def _eq(self, text, mean, std):
        got = parse_cell(text)
        self.assertIsNotNone(got, f"{text!r} parsed as None")
        self.assertAlmostEqual(got[0], mean, places=12, msg=f"mean of {text!r}")
        self.assertAlmostEqual(got[1], std, places=12, msg=f"std of {text!r}")

    def test_digit_bearing_prefix_is_not_the_value(self):
        # The regression this file exists for.
        self._eq("R2=0.939 ± 0.004", 0.939, 0.004)
        self._eq("R2=0.960 ± 0.003", 0.960, 0.003)
        self._eq("R2=-0.32", -0.32, 0.0)
        self._eq("R2=0.644 ± 0.015", 0.644, 0.015)

    def test_letter_only_prefix_still_works(self):
        self._eq("acc=0.729 ± 0.023", 0.729, 0.023)
        self._eq("acc=0.997 ± 0.001", 0.997, 0.001)

    def test_bare_numbers_and_signs(self):
        self._eq("0.859 ± 0.017", 0.859, 0.017)
        self._eq("+0.155", 0.155, 0.0)
        self._eq("-0.148", -0.148, 0.0)
        self._eq("1.2e-3 ± 4e-4", 0.0012, 0.0004)

    def test_trailing_units(self):
        self._eq("0.60 ± 0.04 s", 0.60, 0.04)
        self._eq("7.78 ± 0.50 MPa", 7.78, 0.50)
        self._eq("1269.20x", 1269.20, 0.0)
        self._eq("33.22 ± 0.12 s", 33.22, 0.12)

    def test_non_numeric_cells_are_none(self):
        for text in ("N/A", "float32", "float64", "", None, "identical", "yes"):
            self.assertIsNone(parse_cell(text), f"{text!r} should not parse")

    def test_std_is_not_taken_from_inside_an_identifier(self):
        # A std that came from a prefix would understate the noise band and turn
        # real changes into "within noise", which is the same failure mode.
        got = parse_cell("R2=0.939 ± 0.004")
        self.assertEqual(got[1], 0.004)


if __name__ == "__main__":
    unittest.main()
