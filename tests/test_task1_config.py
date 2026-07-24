from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
import importlib.util
import sys

import yaml

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OPERATORS = SRC / "operators"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(OPERATORS))

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival, comp_by_cv_and_fitness
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.variable import get
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize

from algorithm.ea.vanilla_ea import ExactEvaluationBudgetGA, InstrumentedMating
import crossover as crossover_module
import mutation as mutation_module
import sampling as sampling_module
from problem.pymoo_problem import MaskGuidedOptimizationPlacementProblem


class NoOpCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2, prob=1.0)

    def _do(self, problem, X, **kwargs):
        return X


class NoOpMutation(Mutation):
    def __init__(self):
        super().__init__(prob=1.0)

    def _do(self, problem, X, **kwargs):
        return X


class CountingPlacer:
    def __init__(self):
        self.n_evaluations = 0

    def evaluate(self, X):
        X = np.asarray(X)
        self.n_evaluations += len(X)
        return (
            np.sum(X, axis=1, dtype=float),
            np.zeros(len(X)),
            [{"x": tuple(row)} for row in X],
        )


class CountingProblem(Problem):
    def __init__(self, placer, n_ieq_constr=0):
        super().__init__(
            n_var=4,
            n_obj=1,
            n_ieq_constr=n_ieq_constr,
            xl=np.zeros(4, dtype=int),
            xu=np.full(4, 9, dtype=int),
            vtype=np.int64,
        )
        self.placer = placer

    def _evaluate(self, X, out, *args, **kwargs):
        fitness, overlap, macro_pos = self.placer.evaluate(X)
        out["F"] = np.asarray(fitness)
        if self.n_ieq_constr:
            out["G"] = np.zeros((len(X), self.n_ieq_constr))
        out["overlap_rate"] = overlap
        out["macro_pos"] = macro_pos


class Task1ConfigTest(TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            crossover_prob=1.0,
            uniform_prob_var=0.5,
            sbx_prob_var=0.5,
            sbx_prob_exch=1.0,
            sbx_prob_bin=0.5,
            sbx_eta=30.0,
            pm_prob=1.0,
            pm_prob_var=1.0,
            pm_eta=30.0,
        )

    def test_operator_probabilities_are_explicit(self):
        uniform = crossover_module.MaskGuidedOptimizationUniformCrossover(
            self.args
        )
        sbx = crossover_module.GuidGuideSBXCrossover(self.args)
        pm = mutation_module.MaskGuidedOptimizationPMMutation(self.args)

        self.assertEqual(float(get(uniform.prob)), 1.0)
        self.assertEqual(uniform.prob_var, 0.5)
        self.assertEqual(float(get(sbx.prob)), 1.0)
        self.assertEqual(float(get(sbx.prob_var)), 0.5)
        self.assertEqual(float(get(sbx.eta)), 30.0)
        self.assertEqual(float(get(pm.prob)), 1.0)
        self.assertEqual(float(get(pm.prob_var)), 1.0)
        self.assertEqual(float(get(pm.eta)), 30.0)

    def test_mgo_bounds_use_valid_grid_indices(self):
        placer = SimpleNamespace(placedb=SimpleNamespace(node_cnt=3))
        problem = MaskGuidedOptimizationPlacementProblem(224, 224, placer)
        np.testing.assert_array_equal(problem.xl, np.zeros(6))
        np.testing.assert_array_equal(problem.xu, np.full(6, 223))

    def test_sampling_marks_only_known_constraint_values(self):
        args = SimpleNamespace(
            n_sampling_repeat=1,
            record_func=lambda **kwargs: None,
        )
        placer = CountingPlacer()
        unconstrained = CountingProblem(placer, n_ieq_constr=0)
        population = sampling_module.GrideGuideRandomSampling(args, placer).do(
            unconstrained, 4
        )
        for individual in population:
            self.assertTrue({"F", "G", "H"}.issubset(individual.evaluated))

        constrained = CountingProblem(placer, n_ieq_constr=1)
        population = sampling_module.GrideGuideRandomSampling(args, placer).do(
            constrained, 4
        )
        for individual in population:
            self.assertIn("F", individual.evaluated)
            self.assertNotIn("G", individual.evaluated)
            self.assertIn("H", individual.evaluated)

    def test_remote_cpu_limits_are_explicit_and_bounded(self):
        with (ROOT / "config" / "default.yaml").open() as f:
            default = yaml.safe_load(f)
        with (ROOT / "experiments" / "task1_config.yaml").open() as f:
            protocol = yaml.safe_load(f)

        self.assertEqual(default["n_cpu_max"], 12)
        self.assertEqual(default["ray_object_store_memory_mb"], 512)
        self.assertEqual(protocol["protocol_version"], 2)
        self.assertEqual(protocol["compute"]["cpus_per_run"], 12)
        self.assertEqual(protocol["compute"]["concurrent_runs"], 8)
        self.assertEqual(protocol["compute"]["total_cpu_limit"], 96)
        self.assertEqual(protocol["compute"]["nice_level"], 10)
        self.assertEqual(
            protocol["compute"]["cpus_per_run"]
            * protocol["compute"]["concurrent_runs"],
            protocol["compute"]["total_cpu_limit"],
        )

        runner_path = ROOT / "experiments" / "run_task1.py"
        spec = importlib.util.spec_from_file_location("task1_runner", runner_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        config = module.Configuration("main", "uniform", "swap")
        command = module.command_for(
            sys.executable, config, 1, "formal", 10000, 12
        )
        self.assertIn("--n_cpu_max=12", command)
        self.assertIn("--ray_object_store_memory_mb=512", command)
        self.assertNotIn("--n_cpu_max=96", command)

    def test_exact_budget_handles_final_partial_batch(self):
        pop_size = 4
        max_evals = 13
        args = SimpleNamespace(
            n_sampling_repeat=1,
            record_func=lambda **kwargs: None,
        )
        placer = CountingPlacer()
        problem = CountingProblem(placer)
        sampling = sampling_module.GrideGuideRandomSampling(args, placer)
        selection = TournamentSelection(func_comp=comp_by_cv_and_fitness)
        crossover = NoOpCrossover()
        mutation = NoOpMutation()
        duplicate_elimination = NoDuplicateElimination()
        mating = InstrumentedMating(
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=duplicate_elimination,
        )
        algorithm = ExactEvaluationBudgetGA(
            pop_size=pop_size,
            n_offsprings=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=FitnessSurvival(),
            mating=mating,
            eliminate_duplicates=duplicate_elimination,
            max_evaluations=max_evals,
            initial_evaluations=pop_size,
            offspring_batch_size=pop_size,
        )

        minimize(
            problem,
            algorithm,
            termination=("n_eval", max_evals),
            seed=1,
            copy_algorithm=False,
            verbose=False,
        )

        self.assertEqual(placer.n_evaluations, max_evals)
        self.assertEqual(algorithm.evaluator.n_eval, max_evals)
        self.assertEqual(algorithm.last_batch_size, 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
