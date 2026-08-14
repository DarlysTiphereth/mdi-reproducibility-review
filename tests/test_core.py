import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG
from data_generation import generate_experiment
from mdi import audited_supply, minmax_0_100, mobility_desert_index, routing_inefficiency, socioeconomic_pressure
from optimization import compare_optimization, objective


class CoreTests(unittest.TestCase):
    def test_formula_p_and_normalization(self):
        raw, scaled = socioeconomic_pressure(np.array([100, 200]), np.array([1, 2]), np.array([10, 20]))
        self.assertTrue(np.allclose(raw, [10, 5]))
        self.assertTrue(np.allclose(scaled, [100, 0]))

    def test_i_is_network_over_geodesic(self):
        self.assertTrue(np.allclose(routing_inefficiency(np.array([10, 20]), np.array([10, 10])), [1, 2]))

    def test_o_is_positive_and_not_above_frequency(self):
        o, h = audited_supply(6.0, np.array([10, 10, 10, 10]))
        self.assertTrue(0 < o <= 6.0)
        self.assertTrue(0 <= h <= 1)

    def test_mdi_and_minmax(self):
        raw, score = mobility_desert_index(np.array([10, 20]), np.array([1, 2]), np.array([2, 4]))
        self.assertTrue(np.allclose(raw, [5, 10]))
        self.assertTrue(np.allclose(score, [0, 100]))
        self.assertTrue(np.allclose(minmax_0_100(np.array([3, 3])), [0, 0]))

    def test_generation_reproducibility_and_budget(self):
        a = generate_experiment(CONFIG)
        b = generate_experiment(CONFIG)
        zones_a, zones_b = a[4], b[4]
        self.assertTrue(zones_a.equals(zones_b))
        runs, allocations, summary, post = compare_optimization(zones_a, CONFIG)
        sums = allocations.groupby(["method", "run"]).allocation.sum().to_numpy()
        self.assertTrue(np.allclose(sums, CONFIG.total_resource_units))
        stochastic = runs[runs.method.isin(["Elite-preserving stochastic search", "Random search"])]
        self.assertTrue((stochastic.evaluations == CONFIG.evaluation_budget).all())


if __name__ == "__main__":
    unittest.main()


class GreedyOptimalityTests(unittest.TestCase):
    """Two additional automated tests: the greedy marginal rule attains the exact
    integer optimum, and the marginal gains that make this true are decreasing.

    Supports the claim in Section 4.2 [Fox, 1966; Federgruen and Groenevelt,
    1986]. Runs on a small synthetic instance so the suite stays fast; the full
    19,448-allocation enumeration of the reported instance is in
    verify_greedy_optimality.py.
    """

    def _brute_force(self, p, i, o, budget):
        import itertools
        n = len(p)
        best = float("inf")
        for cut in itertools.combinations(range(budget + n - 1), n - 1):
            prev, alloc = -1, []
            for c in cut:
                alloc.append(c - prev - 1)
                prev = c
            alloc.append(budget + n - 1 - prev - 1)
            best = min(best, objective(np.asarray(alloc, dtype=float), p, i, o))
        return best

    def test_greedy_matches_integer_optimum(self):
        from optimization import deterministic_allocations
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = int(rng.integers(3, 7))
            p = rng.uniform(1.0, 100.0, n)
            i = rng.uniform(1.0, 3.0, n)
            o = rng.uniform(0.2, 8.0, n)
            budget = int(rng.integers(3, 11))
            cfg = CONFIG.__class__(**{**CONFIG.__dict__, "total_resource_units": budget})
            greedy = deterministic_allocations(p, i, o, cfg)["Greedy marginal allocation"]
            self.assertAlmostEqual(
                objective(greedy, p, i, o), self._brute_force(p, i, o, budget), places=10
            )

    def test_marginal_gains_are_decreasing(self):
        rng = np.random.default_rng(7)
        c = rng.uniform(0.1, 50.0, 6)
        o = rng.uniform(0.2, 10.0, 6)
        for idx in range(len(c)):
            gains = [c[idx] / ((o[idx] + k) * (o[idx] + k + 1)) for k in range(12)]
            for k in range(len(gains) - 1):
                self.assertLessEqual(gains[k + 1], gains[k] + 1e-15)
