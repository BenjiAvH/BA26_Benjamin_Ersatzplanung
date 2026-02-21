from __future__ import annotations

import unittest

from etappe03_evaluation.pareto import Point, dedupe_points, dominates, non_dominated


class TestPareto(unittest.TestCase):
    def test_dominates_minimization(self) -> None:
        a = (1.0, 1.0, 1.0, 1.0)
        b = (2.0, 1.0, 1.0, 1.0)
        self.assertTrue(dominates(a, b, eps=1e-9))
        self.assertFalse(dominates(b, a, eps=1e-9))
        self.assertFalse(dominates(a, a, eps=1e-9))

    def test_non_dominated(self) -> None:
        pts = [
            Point(F=(1.0, 2.0, 0.0, 0.0), producers=frozenset({"V1"})),
            Point(F=(2.0, 1.0, 0.0, 0.0), producers=frozenset({"V2"})),
            Point(F=(2.0, 2.0, 0.0, 0.0), producers=frozenset({"V3"})),
        ]
        nd = non_dominated(pts, eps=1e-9)
        Fs = {p.F for p in nd}
        self.assertIn((1.0, 2.0, 0.0, 0.0), Fs)
        self.assertIn((2.0, 1.0, 0.0, 0.0), Fs)
        self.assertNotIn((2.0, 2.0, 0.0, 0.0), Fs)

    def test_dedupe_eps_and_producers_union(self) -> None:
        eps = 1e-3
        pts = [
            Point(F=(1.0, 2.0, 3.0, 4.0), producers=frozenset({"V1"})),
            Point(F=(1.0 + 5e-4, 2.0, 3.0, 4.0), producers=frozenset({"V3"})),
            Point(F=(10.0, 0.0, 0.0, 0.0), producers=frozenset({"V2"})),
        ]
        dd = dedupe_points(pts, eps=eps)
        self.assertEqual(len(dd), 2)
        merged = [p for p in dd if abs(p.F[0] - 1.0) <= eps][0]
        self.assertEqual(merged.producers, frozenset({"V1", "V3"}))


if __name__ == "__main__":
    unittest.main()

