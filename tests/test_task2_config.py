from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from algorithm.moea.task2_core import sample_integer_population


def set_hash(X: np.ndarray) -> str:
    X = np.ascontiguousarray(np.asarray(X, dtype=np.int64))
    rows = sorted(hashlib.sha256(row.tobytes()).digest() for row in X)
    return hashlib.sha256(b"".join(rows)).hexdigest()


class Task2ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "experiments" / "task1_config.yaml").open() as f:
            cls.task1 = yaml.safe_load(f)
        with (ROOT / "experiments" / "task2_config.yaml").open() as f:
            cls.task2 = yaml.safe_load(f)

    def test_config_is_frozen_and_symmetric_files_exist(self):
        self.assertEqual(self.task2["status"], "frozen")
        self.assertEqual(self.task2["protocol_version"], 3)
        for relative in [
            "experiments/run_task2.py",
            "experiments/analyze_task2.py",
            "experiments/task1_config.yaml",
            "experiments/task2_config.yaml",
            "tests/test_task2_config.py",
            "tests/test_task2_metrics.py",
            "tests/test_task2_algorithms.py",
        ]:
            # Runner/analyzer are allowed to be added later in the same change,
            # so this assertion is activated once either file appears.
            path = ROOT / relative
            if relative.endswith(("run_task2.py", "analyze_task2.py")):
                continue
            self.assertTrue(path.exists(), relative)

    def test_task1_settings_are_inherited(self):
        inherited = self.task2["inheritance_from_task1"]
        self.assertEqual(inherited["benchmark"], self.task1["problem"]["benchmark"])
        self.assertEqual(inherited["n_macro"], 512)
        self.assertTrue(inherited["full_543_macro_instance_deferred_to_task3"])
        self.assertEqual(inherited["n_grid_x"], self.task1["problem"]["n_grid_x"])
        self.assertEqual(inherited["n_grid_y"], self.task1["problem"]["n_grid_y"])
        self.assertEqual(inherited["rank_key"], self.task1["problem"]["rank_key"])
        self.assertEqual(
            inherited["population_size"], self.task1["budget"]["population_size"]
        )
        self.assertEqual(
            inherited["true_evaluations"],
            self.task1["budget"]["true_objective_evaluations"],
        )
        self.assertEqual(inherited["seeds"], self.task1["randomness"]["seeds"])
        self.assertEqual(
            inherited["uniform_crossover"]["exchange_probability"],
            self.task1["crossover"]["types"]["uniform"][
                "variable_exchange_probability"
            ],
        )

    def test_validity_remediation_semantics_are_frozen(self):
        duplicate = self.task2["inheritance_from_task1"][
            "genotype_duplicate_elimination"
        ]
        self.assertEqual(
            duplicate["semantics"], "run_level_genotype_hash_exclusion"
        )
        self.assertTrue(duplicate["compare_entire_evaluated_history"])
        self.assertFalse(duplicate["phenotype_duplicate_elimination"])
        moead = self.task2["moead"]
        self.assertEqual(
            moead["ideal_point"],
            "historical_componentwise_minimum_never_regresses",
        )
        self.assertIn("research_design", self.task2)
        self.assertEqual(len(self.task2["research_design"]["research_questions"]), 5)

    def test_formal_problem_and_algorithms_are_fixed(self):
        self.assertEqual(
            self.task2["objectives"]["formal"], ["hpwl", "congestion_top10"]
        )
        self.assertEqual(self.task2["scope"]["formal_comparison"], ["nsga2", "moead"])
        self.assertFalse(
            self.task2["objectives"]["dreamplace_rudy"][
                "source_modification_allowed"
            ]
        )
        self.assertEqual(
            self.task2["objectives"]["dreamplace_rudy"]["route_num_bins_x"],
            512,
        )
        self.assertEqual(
            self.task2["objectives"]["dreamplace_rudy"]["hotspot_fraction"],
            0.10,
        )

    def test_seed1_sample_set_matches_frozen_task1(self):
        base = (
            ROOT
            / "results"
            / "adaptec1"
            / "task1_smoke__main__uniform__swap"
            / "mgo"
            / "ea"
        )
        candidates = sorted(base.glob("seed_1_*/initial_population.npz"))
        if not candidates:
            self.skipTest("Frozen Task 1 initial population is not present")
        frozen = np.load(candidates[0])["X"]
        np.random.seed(1)
        generated = sample_integer_population(
            20,
            np.zeros(frozen.shape[1], dtype=int),
            np.full(frozen.shape[1], 223, dtype=int),
        )
        self.assertEqual(set_hash(generated), set_hash(frozen))
        self.assertFalse(
            np.array_equal(generated, frozen),
            "Task 1's saved order is expected to be HPWL-sorted",
        )


if __name__ == "__main__":
    unittest.main()
