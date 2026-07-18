"""Task 1: a reproducible elitist single-objective evolutionary algorithm."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from src.hsea.operators import crossover_pair, mutate, random_population
from src.hsea.problem import PlacementEvaluator
from src.hsea.results import EvolutionResult

Array = np.ndarray


@dataclass(frozen=True)
class EAConfig:
    population_size: int = 32
    max_evaluations: int = 512
    crossover: str = "uniform"
    crossover_probability: float = 0.9
    mutation: str = "shift"
    mutation_probability: float = 0.8
    mutation_strength: int = 2
    max_step: int = 4
    tournament_size: int = 2
    elite_fraction: float = 0.10
    eta: float = 20.0


def _tournament(rng: np.random.Generator, fitness: Array, size: int) -> int:
    candidates = rng.integers(0, len(fitness), size=max(2, size))
    return int(candidates[np.argmin(fitness[candidates])])


def run_ea(evaluator: PlacementEvaluator, config: EAConfig, seed: int) -> EvolutionResult:
    if config.max_evaluations < config.population_size:
        raise ValueError("max_evaluations must be at least population_size")
    rng = np.random.default_rng(seed)
    start_eval = evaluator.n_evaluations
    start = time.perf_counter()
    population = random_population(rng, config.population_size, evaluator.bounds)
    batch = evaluator.evaluate_single(population, count_cached=True)
    fitness = batch.hpwl.copy()
    evaluations = evaluator.n_evaluations - start_eval
    history: list[dict[str, float]] = []

    def record(generation: int) -> None:
        history.append(
            {
                "generation": float(generation),
                "evaluations": float(evaluations),
                "best_hpwl": float(np.min(fitness)),
                "mean_hpwl": float(np.mean(fitness)),
                "std_hpwl": float(np.std(fitness)),
            }
        )

    record(0)
    generation = 0
    while evaluations < config.max_evaluations:
        generation += 1
        remaining = config.max_evaluations - evaluations
        offspring_count = min(config.population_size, remaining)
        offspring: list[Array] = []
        while len(offspring) < offspring_count:
            p1 = population[_tournament(rng, fitness, config.tournament_size)]
            p2 = population[_tournament(rng, fitness, config.tournament_size)]
            c1, c2 = crossover_pair(
                config.crossover,
                p1,
                p2,
                rng,
                evaluator.bounds,
                config.crossover_probability,
                eta=config.eta,
            )
            for child in (c1, c2):
                child = mutate(
                    config.mutation,
                    child,
                    rng,
                    evaluator.bounds,
                    config.mutation_probability,
                    strength=config.mutation_strength,
                    max_step=config.max_step,
                    eta=config.eta,
                )
                offspring.append(child)
                if len(offspring) >= offspring_count:
                    break
        offspring_x = np.asarray(offspring)
        offspring_f = evaluator.evaluate_single(offspring_x, count_cached=True).hpwl
        evaluations = evaluator.n_evaluations - start_eval
        combined_x = np.concatenate([population, offspring_x])
        combined_f = np.concatenate([fitness, offspring_f])
        elite_count = max(1, int(round(config.population_size * config.elite_fraction)))
        # Truncation selection is deterministic and preserves the requested elites.
        order = np.argsort(combined_f, kind="mergesort")
        selected = order[: config.population_size]
        population, fitness = combined_x[selected], combined_f[selected]
        record(generation)

    best = int(np.argmin(fitness))
    return EvolutionResult(
        algorithm="ea",
        seed=seed,
        config=asdict(config),
        population=population,
        objectives=fitness[:, None],
        archive_x=population[[best]],
        archive_f=fitness[[best], None],
        history=history,
        elapsed_seconds=time.perf_counter() - start,
        evaluations=evaluations,
    )
