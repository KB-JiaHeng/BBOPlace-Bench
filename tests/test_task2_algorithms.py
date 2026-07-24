from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from algorithm.moea.task2_core import (
    dominance_relation,
    dynamic_ideal_nadir,
    mask_to_hex,
    moead_neighbors,
    moead_reference_vectors,
    normalized_tchebycheff,
    phenotype_diagnostics,
    rank_and_crowding,
    sample_integer_population,
    swap_macro_pairs,
    uniform_complementary_children,
    update_historical_ideal,
)


class Task2CoreTest(unittest.TestCase):
    def test_sampling_matches_columnwise_integer_sampling(self):
        np.random.seed(7)
        actual = sample_integer_population(3, np.array([0, 2]), np.array([2, 4]))
        np.random.seed(7)
        expected = np.column_stack(
            [np.random.randint(0, 3, 3), np.random.randint(2, 5, 3)]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_uniform_children_are_complementary(self):
        np.random.seed(3)
        p1 = np.arange(12)
        p2 = np.arange(100, 112)
        c0, c1, mask = uniform_complementary_children(p1, p2)
        np.testing.assert_array_equal(np.where(mask, c0, c1), p2)
        np.testing.assert_array_equal(np.where(mask, c1, c0), p1)
        self.assertEqual(len(mask_to_hex(mask)), 4)

    def test_swap_exchanges_complete_macro_pairs(self):
        np.random.seed(5)
        x = np.array([10, 20, 30, 1, 2, 3])
        child, a, b = swap_macro_pairs(x, node_count=3)
        self.assertNotEqual(a, b)
        self.assertEqual(child[a], x[b])
        self.assertEqual(child[b], x[a])
        self.assertEqual(child[a + 3], x[b + 3])
        self.assertEqual(child[b + 3], x[a + 3])

    def test_nsga2_survival_keeps_extremes(self):
        F = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0], [5, 5]])
        selected, ranks, crowding = rank_and_crowding(F, n_survive=4)
        self.assertIn(0, selected)
        self.assertIn(4, selected)
        self.assertEqual(ranks[5], 1)
        self.assertTrue(np.isinf(crowding[0]))
        self.assertTrue(np.isinf(crowding[4]))

    def test_historical_ideal_never_regresses(self):
        ideal = update_historical_ideal(None, np.array([[4.0, 9.0], [6.0, 7.0]]))
        np.testing.assert_allclose(ideal, [4.0, 7.0])
        improved = update_historical_ideal(ideal, np.array([3.0, 8.0]))
        np.testing.assert_allclose(improved, [3.0, 7.0])
        regressing_observation = update_historical_ideal(improved, np.array([8.0, 10.0]))
        np.testing.assert_allclose(regressing_observation, [3.0, 7.0])

    def test_moead_vectors_neighbors_and_normalization(self):
        weights = moead_reference_vectors(5)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        np.testing.assert_allclose(weights[0], [0, 1])
        np.testing.assert_allclose(weights[-1], [1, 0])
        neighbors = moead_neighbors(weights, 3)
        np.testing.assert_array_equal(neighbors[:, 0], np.arange(5))
        F = np.array([[10, 100], [20, 50], [30, 0]], dtype=float)
        ideal, nadir = dynamic_ideal_nadir(F)
        values = normalized_tchebycheff(F, weights[[0, 2, 4]], ideal, nadir)
        np.testing.assert_allclose(values, [1.0, 0.25, 1.0])

    def test_dominance_and_phenotype_diagnostics(self):
        self.assertEqual(dominance_relation([1, 2], [2, 3]), "dominates")
        self.assertEqual(dominance_relation([2, 3], [1, 2]), "dominated")
        self.assertEqual(dominance_relation([1, 3], [2, 2]), "nondominated")
        d = phenotype_diagnostics(np.array([[0, 0], [2, 1]]), np.array([[0, 0], [1, 3]]))
        self.assertFalse(d["same_phenotype"])
        self.assertEqual(d["moved_macros"], 1)
        self.assertEqual(d["total_grid_displacement"], 3)


if __name__ == "__main__":
    unittest.main()
