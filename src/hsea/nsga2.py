"""Task 2 baseline implementation of NSGA-II."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from src.hsea.operators import crossover_pair, mutate, random_population
from src.hsea.pareto import environmental_selection, hypervolume, rank_and_crowding, update_archive
from src.hsea.problem import PlacementEvaluator
from src.hsea.results import EvolutionResult

Array = np.ndarray


@dataclass(frozen=True)
class NSGA2Config:
    population_size: int = 40
    max_evaluations: int = 800
    crossover: str = "uniform"
    crossover_probability: float = 0.9
    mutation: str = "shift"
    mutation_probability: float = 0.8
    mutation_strength: int = 2
    max_step: int = 4
    eta: float = 20.0
    archive_size: int = 200


def _binary_tournament(
    rng: np.random.Generator,
    rank: Array,
    crowding: Array,
) -> int:
    a, b = rng.integers(0, len(rank), size=2)
    if rank[a] < rank[b]:
        return int(a)
    if rank[b] < rank[a]:
        return int(b)
    if crowding[a] > crowding[b]:
        return int(a)
    if crowding[b] > crowding[a]:
        return int(b)
    return int(a if rng.random() < 0.5 else b)


def run_nsga2(evaluator: PlacementEvaluator, config: NSGA2Config, seed: int) -> EvolutionResult:
    if config.max_evaluations < config.population_size:
        raise ValueError("max_evaluations must be at least population_size")
    rng = np.random.default_rng(seed)
    start_eval = evaluator.n_evaluations
    start = time.perf_counter()
    population = random_population(rng, config.population_size, evaluator.bounds)
    objectives = evaluator.evaluate_multi(population, count_cached=True).objectives
    evaluations = evaluator.n_evaluations - start_eval
    archive_x, archive_f = update_archive(None, None, population, objectives, config.archive_size)
    history: list[dict[str, float]] = []

    def record(generation: int) -> None:
        lower = np.min(archive_f, axis=0)
        upper = np.max(archive_f, axis=0)
        history.append(
            {
                "generation": float(generation),
                "evaluations": float(evaluations),
                "front_size": float(len(archive_f)),
                "min_hpwl": float(np.min(archive_f[:, 0])),
                "min_rudy": float(np.min(archive_f[:, 1])),
                "min_congestion": float(np.min(archive_f[:, 2])),
                "local_hv": hypervolume(archive_f, lower, upper),
            }
        )

    generation = 0
    record(generation)
    while evaluations < config.max_evaluations:
        generation += 1
        rank, crowding, _ = rank_and_crowding(objectives)
        offspring_count = min(config.population_size, config.max_evaluations - evaluations)
        offspring: list[Array] = []
        while len(offspring) < offspring_count:
            p1 = population[_binary_tournament(rng, rank, crowding)]
            p2 = population[_binary_tournament(rng, rank, crowding)]
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
        offspring_f = evaluator.evaluate_multi(offspring_x, count_cached=True).objectives
        evaluations = evaluator.n_evaluations - start_eval
        population, objectives = environmental_selection(
            np.concatenate([population, offspring_x]),
            np.concatenate([objectives, offspring_f]),
            config.population_size,
        )
        archive_x, archive_f = update_archive(
            archive_x, archive_f, offspring_x, offspring_f, config.archive_size
        )
        record(generation)

    return EvolutionResult(
        algorithm="nsga2",
        seed=seed,
        config=asdict(config),
        population=population,
        objectives=objectives,
        archive_x=archive_x,
        archive_f=archive_f,
        history=history,
        elapsed_seconds=time.perf_counter() - start,
        evaluations=evaluations,
    )
