import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG
from data_generation import generate_experiment
from mdi import audited_supply, minmax_0_100, mobility_desert_index, routing_inefficiency, socioeconomic_pressure
from optimization import compare_optimization


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
