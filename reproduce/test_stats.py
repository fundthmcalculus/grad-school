"""Unit tests for the reporting statistics (common.agg sample std; the paired
CI used by table_4_8_mf_dedup)."""

import os
import sys
import unittest
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


class TestAggSampleStd(unittest.TestCase):
    def test_agg_uses_sample_std(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, std = C.agg(vals)
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(std, statistics.stdev(vals))  # ddof=1
        self.assertNotAlmostEqual(std, statistics.pstdev(vals))  # not ddof=0

    def test_agg_single_and_empty(self):
        self.assertEqual(C.agg([2.5]), (2.5, 0.0))
        self.assertEqual(C.agg([]), (None, None))
        self.assertEqual(C.agg([None, 4.0]), (4.0, 0.0))  # None filtered


class TestPairedCI(unittest.TestCase):
    def test_ci_uses_t_quantile_sample_std(self):
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
        )
        import table_4_8_mf_dedup as T

        # A mean whose normal-1.96 CI would exclude zero but the wider t CI does not.
        # half_normal = 1.96*std/sqrt(n); half_t = 2.2622*std/sqrt(n) at n=10.
        n, std = 10, 0.10
        import math

        mean = (
            1.97 * std / math.sqrt(n)
        )  # just past the normal bound, inside the t bound
        self.assertFalse(T._ci_excludes_zero(mean, std, n))  # t-based: does NOT exclude
        # a clearly-nonzero mean still excludes zero
        self.assertTrue(T._ci_excludes_zero(10 * std, std, n))
        self.assertFalse(T._ci_excludes_zero(0.5, std, n=1))  # n<2 guard


if __name__ == "__main__":
    unittest.main()
