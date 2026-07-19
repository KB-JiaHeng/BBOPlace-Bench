import csv
import hashlib
import json
import logging
import os
import pickle
import time

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import (
    GA,
    FitnessSurvival,
    comp_by_cv_and_fitness,
)
from pymoo.core.duplicate import (
    DefaultDuplicateElimination,
    NoDuplicateElimination,
)
from pymoo.core.mating import Mating
from pymoo.core.population import Population
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize

from operators import REGISTRY as OPS_REGISTRY
from problem.pymoo_problem import (
    HyperparameterPlacementProblem,
    MaskGuidedOptimizationPlacementProblem,
    SequencePairPlacementProblem,
)
from utils.constant import INF

from ..basic_algo import BasicAlgo


class InstrumentedMating(Mating):
    """Mating with explicit duplicate-generation statistics.

    This follows pymoo's InfillCriterion.do implementation, but records how
    many candidate genotypes were rejected before objective evaluation.
    """

    def __init__(self, *args, n_max_iterations=100, **kwargs):
        super().__init__(*args, n_max_iterations=n_max_iterations, **kwargs)
        self.last_stats = self.empty_stats(0)

    @staticmethod
    def empty_stats(requested):
        return {
            "requested_offspring": int(requested),
            "raw_generated": 0,
            "accepted_unique_offspring": 0,
            "duplicate_rejections": 0,
            "truncated_offspring": 0,
            "mating_iterations": 0,
        }

    def do(self, problem, pop, n_offsprings, **kwargs):
        n_max_iterations = kwargs.get("n_max_iterations", self.n_max_iterations)
        off = Population.create()
        stats = self.empty_stats(n_offsprings)

        while len(off) < n_offsprings:
            n_remaining = n_offsprings - len(off)
            generated = self._do(problem, pop, n_remaining, **kwargs)
            stats["raw_generated"] += len(generated)

            generated = self.repair(problem, generated, **kwargs)
            before_elimination = len(generated)
            generated = self.eliminate_duplicates.do(generated, pop, off)
            stats["duplicate_rejections"] += before_elimination - len(generated)

            if len(off) + len(generated) > n_offsprings:
                keep = n_offsprings - len(off)
                stats["truncated_offspring"] += len(generated) - keep
                generated = generated[:keep]

            off = Population.merge(off, generated)
            stats["mating_iterations"] += 1

            if stats["mating_iterations"] >= n_max_iterations:
                break

        stats["accepted_unique_offspring"] = len(off)
        self.last_stats = stats
        return off


class ExactEvaluationBudgetGA(GA):
    """GA that counts initial sampling and never overshoots max_evaluations."""

    def __init__(
        self,
        *args,
        max_evaluations,
        initial_evaluations,
        offspring_batch_size,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_evaluations = int(max_evaluations)
        self.initial_evaluations = int(initial_evaluations)
        self.offspring_batch_size = int(offspring_batch_size)
        self.last_batch_size = 0
        self.last_mating_stats = InstrumentedMating.empty_stats(0)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)
        if self.evaluator.n_eval != 0:
            raise RuntimeError(
                "Initial population was unexpectedly reevaluated by pymoo: "
                f"n_eval={self.evaluator.n_eval}"
            )
        self.evaluator.n_eval = self.initial_evaluations
        self.last_batch_size = self.initial_evaluations
        self.last_mating_stats = InstrumentedMating.empty_stats(0)

    def _infill(self):
        remaining = self.max_evaluations - self.evaluator.n_eval
        if remaining <= 0:
            return None

        requested = min(self.offspring_batch_size, remaining)
        off = self.mating.do(
            self.problem,
            self.pop,
            requested,
            algorithm=self,
        )
        self.last_mating_stats = dict(self.mating.last_stats)
        self.last_batch_size = len(off)

        if len(off) == 0:
            raise RuntimeError(
                "Mating produced no unique offspring after "
                f"{self.mating.last_stats['mating_iterations']} attempts. "
                "The run is invalid; no random immigrants are injected."
            )

        if len(off) < requested:
            logging.warning(
                "Mating produced %d/%d unique offspring; continuing with the "
                "smaller batch and preserving the exact evaluation budget.",
                len(off),
                requested,
            )

        return off


class VanillaEA(BasicAlgo):
    def __init__(self, args, placer, logger):
        super().__init__(args=args, placer=placer, logger=logger)
        self.node_cnt = placer.placedb.node_cnt
        self.best_hpwl = INF
        self._initial_population_saved = False

        if args.placer == "mgo":
            self.problem = MaskGuidedOptimizationPlacementProblem(
                n_grid_x=args.n_grid_x,
                n_grid_y=args.n_grid_y,
                placer=placer,
            )
        elif args.placer == "sp":
            self.problem = SequencePairPlacementProblem(placer=placer)
        elif args.placer == "hpo":
            from placer.hpo_placer import params_space

            self.problem = HyperparameterPlacementProblem(
                params_space=params_space,
                placer=placer,
            )
        else:
            raise NotImplementedError

        self.args.__dict__.update(
            {"logger": logger, "record_func": self._record_results}
        )
        self.generation_metrics_file = os.path.join(
            args.result_path, "generation_metrics.csv"
        )
        with open(self.generation_metrics_file, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "evaluation_count",
                    "generation",
                    "phase",
                    "batch_evaluations",
                    "population_best_hpwl",
                    "population_mean_hpwl",
                    "population_std_hpwl",
                    "best_so_far_hpwl",
                    "elapsed_batch_seconds",
                    "average_seconds_per_evaluation",
                    "requested_offspring",
                    "accepted_unique_offspring",
                    "duplicate_rejections",
                    "raw_generated",
                    "truncated_offspring",
                    "mating_iterations",
                ]
            )

    def _validate_task1_config(self):
        required = {
            "placer": "mgo",
            "algorithm": "ea",
            "benchmark": "adaptec1",
            "n_population": 20,
            "n_sampling_repeat": 1,
            "n_offsprings": 20,
            "max_evals": 10000,
            "crossover_prob": 1.0,
            "uniform_prob_var": 0.5,
            "sbx_prob_var": 0.5,
            "sbx_prob_exch": 1.0,
            "sbx_prob_bin": 0.5,
            "pm_prob": 1.0,
            "pm_prob_var": 1.0,
            "eliminate_duplicates": True,
            "checkpoint_enabled": False,
            "n_grid_x": 224,
            "n_grid_y": 224,
            "rank_key": "area_sum",
            "sampling": "random",
            "record_evaluation_trace": True,
            "eval_gp_hpwl": False,
            "n_cpu_max": 12,
            "ray_object_store_memory_mb": 512,
        }
        smoke_test = bool(getattr(self.args, "task1_smoke_test", False))
        for key, expected in required.items():
            actual = getattr(self.args, key)
            if key == "max_evals" and smoke_test:
                if not (self.args.n_population <= actual < expected):
                    raise ValueError(
                        "Task 1 smoke-test max_evals must be in "
                        f"[{self.args.n_population}, {expected}): {actual}"
                    )
                continue
            if actual != expected:
                raise ValueError(
                    f"Task 1 protocol violation: {key}={actual!r}, "
                    f"expected {expected!r}"
                )
        family = getattr(self.args, "task1_family", "main")
        if family == "main":
            if self.args.sbx_eta != 30.0 or self.args.pm_eta != 30.0:
                raise ValueError("Main factorial runs require both eta values to be 30")
        elif family == "sbx_eta":
            if not (
                self.args.crossover == "sbx"
                and self.args.mutation == "swap"
                and self.args.sbx_eta in {5.0, 15.0}
                and self.args.pm_eta == 30.0
            ):
                raise ValueError("Invalid SBX eta sensitivity configuration")
        elif family == "pm_eta":
            if not (
                self.args.crossover == "uniform"
                and self.args.mutation == "pm"
                and self.args.pm_eta in {5.0, 15.0}
                and self.args.sbx_eta == 30.0
            ):
                raise ValueError("Invalid PM eta sensitivity configuration")
        else:
            raise ValueError(f"Unknown Task 1 experiment family: {family}")

        if self.args.crossover not in {"uniform", "sbx"}:
            raise ValueError(f"Unexpected Task 1 crossover: {self.args.crossover}")
        if self.args.mutation not in {
            "swap",
            "shift",
            "random_resetting",
            "shuffle",
            "pm",
        }:
            raise ValueError(f"Unexpected Task 1 mutation: {self.args.mutation}")
        if hasattr(self.args, "checkpoint"):
            raise ValueError("Task 1 runs must not load a checkpoint")

    def run(self):
        self._validate_task1_config()

        sampling = OPS_REGISTRY["sampling"][self.args.placer][
            self.args.sampling.lower()
        ](self.args, self.placer)
        crossover = OPS_REGISTRY["crossover"][self.args.placer][
            self.args.crossover.lower()
        ](self.args)
        mutation = OPS_REGISTRY["mutation"][self.args.placer][
            self.args.mutation.lower()
        ](self.args)

        selection = TournamentSelection(func_comp=comp_by_cv_and_fitness)
        survival = FitnessSurvival()
        duplicate_elimination = (
            DefaultDuplicateElimination()
            if self.args.eliminate_duplicates
            else NoDuplicateElimination()
        )
        mating = InstrumentedMating(
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=duplicate_elimination,
            n_max_iterations=self.args.duplicate_retry_limit,
        )

        initial_evaluations = (
            self.args.n_population * self.args.n_sampling_repeat
        )
        self._algo = ExactEvaluationBudgetGA(
            pop_size=self.args.n_population,
            n_offsprings=self.args.n_offsprings,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            mating=mating,
            eliminate_duplicates=duplicate_elimination,
            max_evaluations=self.args.max_evals,
            initial_evaluations=initial_evaluations,
            offspring_batch_size=self.args.n_offsprings,
            callback=self._save_callback,
        )

        self.t = time.time()
        minimize(
            problem=self.problem,
            algorithm=self._algo,
            termination=("n_eval", self.args.max_evals),
            seed=self.args.seed,
            copy_algorithm=False,
            verbose=True,
        )

        true_count = self.placer.true_n_eval
        pymoo_count = self._algo.evaluator.n_eval
        if true_count != self.args.max_evals or pymoo_count != self.args.max_evals:
            raise RuntimeError(
                "Evaluation-budget mismatch: "
                f"placer={true_count}, pymoo={pymoo_count}, "
                f"expected={self.args.max_evals}"
            )

        completion = {
            "status": "complete",
            "true_evaluation_count": int(true_count),
            "pymoo_evaluation_count": int(pymoo_count),
            "final_best_hpwl": float(self.best_hpwl),
            "generations": int(self._algo.n_iter - 1),
            "seed": int(self.args.seed),
            "crossover": self.args.crossover,
            "mutation": self.args.mutation,
            "sbx_eta": float(self.args.sbx_eta),
            "pm_eta": float(self.args.pm_eta),
            "task1_family": self.args.task1_family,
        }
        with open(
            os.path.join(self.args.result_path, "run_complete.json"),
            "w",
        ) as f:
            json.dump(completion, f, indent=2)

    def _save_initial_population(self, algo):
        X = np.ascontiguousarray(algo.pop.get("X"))
        F = np.ascontiguousarray(algo.pop.get("F"))
        np.savez_compressed(
            os.path.join(self.args.result_path, "initial_population.npz"),
            X=X,
            F=F,
        )
        metadata = {
            "n_individuals": int(len(X)),
            "x_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
            "f_sha256": hashlib.sha256(F.tobytes()).hexdigest(),
        }
        with open(
            os.path.join(self.args.result_path, "initial_population_hash.json"),
            "w",
        ) as f:
            json.dump(metadata, f, indent=2)
        self._initial_population_saved = True

    def _save_callback(self, algo):
        now = time.time()
        elapsed_batch = now - self.t
        self.t_total += elapsed_batch
        self.t = now

        self.n_eval = int(algo.evaluator.n_eval)
        if algo.n_iter == 1 and len(algo.pop) != self.args.n_population:
            raise RuntimeError(
                "Initial duplicate elimination changed the population size: "
                f"{len(algo.pop)} != {self.args.n_population}"
            )
        if self.n_eval != self.placer.true_n_eval:
            raise RuntimeError(
                "Evaluation counters diverged during the run: "
                f"placer={self.placer.true_n_eval}, pymoo={self.n_eval}"
            )

        hpwl = np.asarray(algo.pop.get("F")).reshape(-1)
        overlap_rate = np.asarray(algo.pop.get("overlap_rate")).reshape(-1)
        macro_pos_all = algo.pop.get("macro_pos")
        best_idx = int(np.argmin(hpwl))
        population_best = float(hpwl[best_idx])
        population_mean = float(np.mean(hpwl))
        population_std = float(np.std(hpwl))

        if population_best < self.best_hpwl:
            self.best_hpwl = population_best
            self.placer.save_placement(
                macro_pos=macro_pos_all[best_idx],
                n_eval=self.n_eval,
                hpwl=self.best_hpwl,
            )
            self.placer.plot(
                macro_pos=macro_pos_all[best_idx],
                n_eval=self.n_eval,
                hpwl=self.best_hpwl,
            )

        if not self._initial_population_saved:
            self._save_initial_population(algo)

        stats = algo.last_mating_stats
        phase = "initialization" if algo.n_iter == 1 else "evolution"
        average_seconds = self.t_total / max(1, self.n_eval)
        with open(self.generation_metrics_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.n_eval,
                    int(algo.n_iter),
                    phase,
                    int(algo.last_batch_size),
                    population_best,
                    population_mean,
                    population_std,
                    float(self.best_hpwl),
                    elapsed_batch,
                    average_seconds,
                    stats["requested_offspring"],
                    stats["accepted_unique_offspring"],
                    stats["duplicate_rejections"],
                    stats["raw_generated"],
                    stats["truncated_offspring"],
                    stats["mating_iterations"],
                ]
            )

        self.logger.add("HPWL/his_best", self.best_hpwl)
        self.logger.add("HPWL/pop_best", population_best)
        self.logger.add("HPWL/pop_avg", population_mean)
        self.logger.add("HPWL/pop_std", population_std)
        self.logger.add("EA/true_n_eval", self.n_eval)
        self.logger.add(
            "EA/duplicate_rejections", stats["duplicate_rejections"]
        )
        self.logger.step()

        self.placer.save_metrics(
            n_eval=self.n_eval,
            his_best_hpwl=self.best_hpwl,
            pop_best_hpwl=population_best,
            pop_avg_hpwl=population_mean,
            pop_std_hpwl=population_std,
            overlap_rate=float(overlap_rate[best_idx]),
            t_each_eval=elapsed_batch / max(1, algo.last_batch_size),
            avg_t_each_eval=average_seconds,
            avg_t_eval_solution=(
                self.placer.t_eval_solution_total / max(1, self.n_eval)
            ),
        )

        if self.t_total >= self.max_eval_time_second:
            raise TimeoutError(
                "Maximum run time reached: "
                f"{self.t_total:.2f}s >= {self.max_eval_time_second:.2f}s"
            )

        if self.args.checkpoint_enabled:
            self._save_checkpoint(
                population=algo.pop.get("X"),
                fitness=algo.pop.get("F"),
                n_gen=algo.n_gen,
            )

    def _save_checkpoint(self, population, fitness, n_gen):
        super()._save_checkpoint()
        with open(os.path.join(self.checkpoint_path, "ea.pkl"), "wb") as f:
            pickle.dump(
                {
                    "population": population,
                    "fitness": fitness,
                    "n_gen": n_gen,
                },
                file=f,
            )
