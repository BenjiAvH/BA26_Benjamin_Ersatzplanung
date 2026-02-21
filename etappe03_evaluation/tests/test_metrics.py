from __future__ import annotations

import unittest

from etappe03_evaluation.pareto import Point
from etappe03_evaluation.metrics import coverage, contribution_inclusive_unique


class TestMetrics(unittest.TestCase):
    def test_coverage(self) -> None:
        eps = 1e-9
        A = [Point(F=(1.0, 1.0, 1.0, 1.0), producers=frozenset({"V1"}))]
        B = [
            Point(F=(2.0, 2.0, 2.0, 2.0), producers=frozenset({"V2"})),
            Point(F=(0.0, 3.0, 3.0, 3.0), producers=frozenset({"V2"})),
        ]
        c = coverage(A, B, eps=eps)
        self.assertIsNotNone(c)
        self.assertAlmostEqual(float(c), 0.5)

    def test_coverage_empty_B(self) -> None:
        self.assertIsNone(coverage([], [], eps=1e-9))

    def test_contribution_inclusive_unique(self) -> None:
        P_star = [
            Point(F=(0.0, 0.0, 0.0, 0.0), producers=frozenset({"V1"})),
            Point(F=(1.0, 0.0, 0.0, 0.0), producers=frozenset({"V1", "V3"})),
            Point(F=(2.0, 0.0, 0.0, 0.0), producers=frozenset({"V3"})),
        ]
        methods = ["V1", "V2", "V3", "V4"]
        inc, uniq = contribution_inclusive_unique(P_star, methods=methods)
        self.assertAlmostEqual(inc["V1"], 2.0 / 3.0)
        self.assertAlmostEqual(inc["V3"], 2.0 / 3.0)
        self.assertAlmostEqual(inc["V2"], 0.0)
        self.assertAlmostEqual(uniq["V1"], 1.0 / 3.0)
        self.assertAlmostEqual(uniq["V3"], 1.0 / 3.0)
        self.assertAlmostEqual(uniq["V2"], 0.0)


if __name__ == "__main__":
    unittest.main()

