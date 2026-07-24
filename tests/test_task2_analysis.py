from __future__ import annotations

import unittest

import numpy as np

from experiments.analyze_task2 import (
    deduplicate_by_phenotype,
    moead_state_diagnostics,
    trace_diagnostics,
)


class Task2AnalysisTest(unittest.TestCase):
    def test_phenotype_deduplication_precedes_front_analysis(self):
        F = np.asarray([[1.0, 3.0], [1.0, 3.0], [2.0, 2.0], [1e16, 1e16]])
        hashes = np.asarray(["a", "a", "b", "invalid"])
        unique_F, unique_hashes = deduplicate_by_phenotype(F, hashes)
        np.testing.assert_allclose(unique_F, [[1.0, 3.0], [2.0, 2.0]])
        np.testing.assert_array_equal(unique_hashes, ["a", "b"])

    def test_trace_diagnostics_exposes_redundancy_and_operator_effects(self):
        rows = [
            {
                "phase": "initialization",
                "decode_success": "1",
                "genotype_hash": "g0",
                "phenotype_hash": "p0",
            },
            {
                "phase": "evolution",
                "decode_success": "1",
                "genotype_hash": "g1",
                "phenotype_hash": "p0",
                "same_phenotype_parent1": "1",
                "same_phenotype_parent2": "0",
                "moved_macros_parent1": "0",
                "moved_macros_parent2": "4",
                "total_grid_displacement_parent1": "0",
                "total_grid_displacement_parent2": "12",
                "mean_grid_displacement_parent1": "0",
                "mean_grid_displacement_parent2": "3",
                "max_grid_displacement_parent1": "0",
                "max_grid_displacement_parent2": "7",
                "delta_hpwl_parent1": "0",
                "delta_hpwl_parent2": "-5",
                "delta_congestion_top10_parent1": "0",
                "delta_congestion_top10_parent2": "0.1",
                "survived": "1",
                "archive_entered": "0",
                "replacement_count": "2",
                "dominance_parent1": "equal",
                "dominance_parent2": "nondominated",
            },
        ]
        result = trace_diagnostics(rows)
        self.assertEqual(result["repeated_genotype_evaluations"], 0)
        self.assertAlmostEqual(result["cumulative_phenotype_redundancy"], 0.5)
        self.assertEqual(result["same_phenotype_parent1_rate"], 1.0)
        self.assertEqual(result["moved_macros"]["max"], 4.0)
        self.assertEqual(result["delta_hpwl"]["improvement_rate"], 0.5)
        self.assertEqual(result["replacement_count"]["max"], 2.0)

    def test_moead_state_requires_nonregressing_historical_ideal(self):
        rows = [
            {
                "ideal": "[10.0,20.0]",
                "unique_slot_phenotypes": "20",
                "max_single_phenotype_fraction": "0.05",
            },
            {
                "ideal": "[9.0,20.0]",
                "unique_slot_phenotypes": "8",
                "max_single_phenotype_fraction": "0.4",
            },
        ]
        result = moead_state_diagnostics(rows)
        self.assertTrue(result["historical_ideal_nonregressing"])
        self.assertEqual(result["minimum_unique_slot_phenotypes"], 8)
        self.assertEqual(result["maximum_single_phenotype_slot_fraction"], 0.4)


if __name__ == "__main__":
    unittest.main()
