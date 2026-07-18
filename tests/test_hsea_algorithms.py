from types import SimpleNamespace

import numpy as np

from src.hsea.moead import MOEADConfig, run_moead
from src.hsea.nsga2 import NSGA2Config, run_nsga2
from src.hsea.operators import Bounds
from src.hsea.problem import EvaluationBatch
from src.hsea.task1 import EAConfig, run_ea


class FakeEvaluator:
    def __init__(self):
        self.bounds = Bounds(n_macro=5, n_grid_x=12, n_grid_y=12)
        self.n_evaluations = 0

    def evaluate_single(self, population, count_cached=False):
        population = np.asarray(population)
        self.n_evaluations += len(population)
        hpwl = np.sum(population, axis=1).astype(float)
        return EvaluationBatch(hpwl[:, None], hpwl, np.zeros(len(population)), [{}] * len(population))

    def evaluate_multi(self, population, count_cached=False):
        population = np.asarray(population, dtype=float)
        self.n_evaluations += len(population)
        n = self.bounds.n_macro
        f = np.column_stack([
            np.sum(population, axis=1),
            np.sum((population[:, :n] - 6.0) ** 2, axis=1),
            np.sum((population[:, n:] - 3.0) ** 2, axis=1),
        ])
        return EvaluationBatch(f, f[:, 0], np.zeros(len(population)), [{}] * len(population))


def test_ea_respects_budget():
    evaluator = FakeEvaluator()
    result = run_ea(evaluator, EAConfig(population_size=8, max_evaluations=32), seed=1)
    assert result.evaluations == 32
    assert result.archive_f.shape == (1, 1)


def test_nsga2_respects_budget_and_has_archive():
    evaluator = FakeEvaluator()
    result = run_nsga2(evaluator, NSGA2Config(population_size=8, max_evaluations=32), seed=1)
    assert result.evaluations == 32
    assert result.archive_f.shape[1] == 3
    assert len(result.archive_f) >= 1


def test_moead_respects_budget_and_has_archive():
    evaluator = FakeEvaluator()
    result = run_moead(
        evaluator,
        MOEADConfig(population_size=8, max_evaluations=32, neighborhood_size=4),
        seed=1,
    )
    assert result.evaluations == 32
    assert result.archive_f.shape[1] == 3
    assert len(result.archive_f) >= 1
