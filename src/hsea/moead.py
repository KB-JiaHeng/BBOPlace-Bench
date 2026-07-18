"""Task 2 baseline implementation of MOEA/D with Tchebycheff decomposition."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from src.hsea.operators import crossover_pair, mutate, random_population
from src.hsea.pareto import hypervolume, update_archive
from src.hsea.problem import PlacementEvaluator
from src.hsea.results import EvolutionResult

Array = np.ndarray


@dataclass(frozen=True)
class MOEADConfig:
    population_size: int = 40
    max_evaluations: int = 800
    neighborhood_size: int = 10
    neighborhood_mating_probability: float = 0.9
    maximum_replacements: int = 2
    crossover: str = "uniform"
    crossover_probability: float = 1.0
    mutation: str = "shift"
    mutation_probability: float = 0.8
    mutation_strength: int = 2
    max_step: int = 4
    eta: float = 20.0
    archive_size: int = 200


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def simplex_weights(n_points: int, n_objectives: int) -> Array:
    """Generate deterministic, approximately uniform simplex weights."""
    h = 1
    while math.comb(h + n_objectives - 1, n_objectives - 1) < n_points:
        h += 1
    candidates = np.asarray(list(_compositions(h, n_objectives)), dtype=float) / h
    if len(candidates) <= n_points:
        return candidates
    # Greedy farthest-point subset, seeded with all simplex vertices.
    selected: list[int] = []
    for j in range(n_objectives):
        selected.append(int(np.argmax(candidates[:, j])))
    selected = list(dict.fromkeys(selected))
    min_dist = np.min(
        np.linalg.norm(candidates[:, None, :] - candidates[selected][None, :, :], axis=2), axis=1
    )
    while len(selected) < n_points:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(candidates - candidates[idx], axis=1))
    weights = candidates[selected[:n_points]]
    return np.maximum(weights, 1e-6)


def _scalarized(f: Array, weight: Array, ideal: Array, nadir: Array) -> float:
    if not np.all(np.isfinite(f)):
        return float("inf")
    normalized = (f - ideal) / np.maximum(nadir - ideal, 1e-12)
    return float(np.max(weight * np.abs(normalized)))


def run_moead(evaluator: PlacementEvaluator, config: MOEADConfig, seed: int) -> EvolutionResult:
    if config.max_evaluations < config.population_size:
        raise ValueError("max_evaluations must be at least population_size")
    rng = np.random.default_rng(seed)
    start_eval = evaluator.n_evaluations
    start = time.perf_counter()
    population = random_population(rng, config.population_size, evaluator.bounds)
    objectives = evaluator.evaluate_multi(population, count_cached=True).objectives
    evaluations = evaluator.n_evaluations - start_eval
    n_obj = objectives.shape[1]
    weights = simplex_weights(config.population_size, n_obj)
    distance = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighborhood_size = min(max(2, config.neighborhood_size), config.population_size)
    neighborhoods = np.argsort(distance, axis=1)[:, :neighborhood_size]
    finite_initial = np.all(np.isfinite(objectives), axis=1)
    if not np.any(finite_initial):
        raise RuntimeError("MOEA/D initialization produced no legal finite placement")
    ideal = np.min(objectives[finite_initial], axis=0)
    nadir = np.max(objectives[finite_initial], axis=0)
    archive_x, archive_f = update_archive(None, None, population, objectives, config.archive_size)
    history: list[dict[str, float]] = []

    def record(generation: int) -> None:
        history.append(
            {
                "generation": float(generation),
                "evaluations": float(evaluations),
                "front_size": float(len(archive_f)),
                "min_hpwl": float(np.min(archive_f[:, 0])),
                "min_rudy": float(np.min(archive_f[:, 1])),
                "min_congestion": float(np.min(archive_f[:, 2])),
                "local_hv": hypervolume(archive_f, np.min(archive_f, axis=0), np.max(archive_f, axis=0)),
            }
        )

    generation = 0
    record(generation)
    while evaluations < config.max_evaluations:
        generation += 1
        for i in rng.permutation(config.population_size):
            if evaluations >= config.max_evaluations:
                break
            if rng.random() < config.neighborhood_mating_probability:
                pool = neighborhoods[i]
            else:
                pool = np.arange(config.population_size)
            parents = rng.choice(pool, size=2, replace=len(pool) < 2)
            child, _ = crossover_pair(
                config.crossover,
                population[parents[0]],
                population[parents[1]],
                rng,
                evaluator.bounds,
                config.crossover_probability,
                eta=config.eta,
            )
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
            child_f = evaluator.evaluate_multi(child[None, :], count_cached=True).objectives[0]
            evaluations = evaluator.n_evaluations - start_eval
            if not np.all(np.isfinite(child_f)):
                continue
            ideal = np.minimum(ideal, child_f)
            nadir = np.maximum(nadir, child_f)
            update_pool = neighborhoods[i].copy()
            rng.shuffle(update_pool)
            replacements = 0
            for j in update_pool:
                if _scalarized(child_f, weights[j], ideal, nadir) <= _scalarized(
                    objectives[j], weights[j], ideal, nadir
                ):
                    population[j] = child
                    objectives[j] = child_f
                    replacements += 1
                    if replacements >= config.maximum_replacements:
                        break
            archive_x, archive_f = update_archive(
                archive_x,
                archive_f,
                child[None, :],
                child_f[None, :],
                config.archive_size,
            )
        record(generation)

    return EvolutionResult(
        algorithm="moead",
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
