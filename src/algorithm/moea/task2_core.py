"""Pure Task 2 MOEA mechanics shared by the real runner and unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np

from task2.hashing import genotype_hash, phenotype_hash
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.core.population import Population
from pymoo.core.problem import Problem


_OBJECTIVE_PROBLEM = Problem(n_var=1, n_obj=2)


def sample_integer_population(
    n_samples: int,
    xl: np.ndarray,
    xu: np.ndarray,
) -> np.ndarray:
    """Match pymoo IntegerRandomSampling's column-wise RNG semantics."""
    xl = np.asarray(xl, dtype=int)
    xu = np.asarray(xu, dtype=int)
    if xl.shape != xu.shape:
        raise ValueError("Lower and upper bounds have different shapes")
    return np.column_stack(
        [np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(len(xl))]
    ).astype(np.int64, copy=False)


def uniform_complementary_children(
    parent1: np.ndarray,
    parent2: np.ndarray,
    probability: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent1 = np.asarray(parent1, dtype=np.int64)
    parent2 = np.asarray(parent2, dtype=np.int64)
    if parent1.shape != parent2.shape:
        raise ValueError("Parent shapes differ")
    mask = np.random.random(parent1.shape) < probability
    child0 = np.where(mask, parent2, parent1)
    child1 = np.where(mask, parent1, parent2)
    return child0.astype(np.int64, copy=False), child1.astype(np.int64, copy=False), mask


def swap_macro_pairs(x: np.ndarray, node_count: int) -> tuple[np.ndarray, int, int]:
    x = np.asarray(x, dtype=np.int64)
    if x.shape != (2 * node_count,):
        raise ValueError(f"Expected {(2 * node_count,)}, got {x.shape}")
    if node_count < 2:
        return x.copy(), 0, 0
    a, b = np.random.choice(node_count, size=2, replace=False)
    child = x.copy()
    child[a], child[b] = x[b], x[a]
    child[a + node_count], child[b + node_count] = (
        x[b + node_count],
        x[a + node_count],
    )
    return child, int(a), int(b)


def mask_to_hex(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")
    return packed.tobytes().hex()


def dominance_relation(a: np.ndarray, b: np.ndarray) -> str:
    """Return the relation of a to b for minimization."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_le = np.all(a <= b)
    b_le = np.all(b <= a)
    if a_le and np.any(a < b):
        return "dominates"
    if b_le and np.any(b < a):
        return "dominated"
    if np.array_equal(a, b):
        return "equal"
    return "nondominated"


def rank_and_crowding(
    objective_values: np.ndarray,
    n_survive: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objective_values = np.asarray(objective_values, dtype=float)
    if objective_values.ndim != 2 or objective_values.shape[1] != 2:
        raise ValueError("Expected an n-by-2 objective matrix")
    if n_survive is None:
        n_survive = len(objective_values)
    if not 0 < n_survive <= len(objective_values):
        raise ValueError("Invalid n_survive")
    population = Population.new(
        F=objective_values,
        original_index=np.arange(len(objective_values), dtype=int),
    )
    survival = RankAndCrowding(crowding_func="cd")
    # Compute diagnostic rank/crowding values for the complete candidate pool.
    # Standard crowding distance does not depend on n_survive; the second call
    # then performs the ordinary NSGA-II environmental selection.
    survival.do(_OBJECTIVE_PROBLEM, population, n_survive=len(population))
    ranks = np.asarray(population.get("rank"), dtype=float).copy()
    crowding = np.asarray(population.get("crowding"), dtype=float).copy()
    selected = survival.do(
        _OBJECTIVE_PROBLEM,
        population,
        n_survive=n_survive,
    )
    selected_indices = np.asarray(selected.get("original_index"), dtype=int)
    return selected_indices, ranks, crowding


def binary_tournament_rank_crowding(ranks: np.ndarray, crowding: np.ndarray) -> int:
    ranks = np.asarray(ranks)
    crowding = np.asarray(crowding)
    if len(ranks) != len(crowding) or len(ranks) < 2:
        raise ValueError("Tournament requires at least two ranked solutions")
    a, b = np.random.choice(len(ranks), size=2, replace=False)
    if ranks[a] < ranks[b]:
        return int(a)
    if ranks[b] < ranks[a]:
        return int(b)
    if crowding[a] > crowding[b]:
        return int(a)
    if crowding[b] > crowding[a]:
        return int(b)
    return int(a if np.random.random() < 0.5 else b)


def moead_reference_vectors(population_size: int) -> np.ndarray:
    if population_size < 2:
        raise ValueError("Bi-objective MOEA/D requires at least two vectors")
    first = np.linspace(0.0, 1.0, population_size)
    return np.column_stack([first, 1.0 - first])


def moead_neighbors(reference_vectors: np.ndarray, neighborhood_size: int) -> np.ndarray:
    reference_vectors = np.asarray(reference_vectors, dtype=float)
    if not 1 <= neighborhood_size <= len(reference_vectors):
        raise ValueError("Invalid MOEA/D neighborhood size")
    distance = np.linalg.norm(
        reference_vectors[:, None, :] - reference_vectors[None, :, :], axis=2
    )
    return np.argsort(distance, axis=1, kind="stable")[:, :neighborhood_size]


def dynamic_ideal_nadir(objective_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    objective_values = np.asarray(objective_values, dtype=float)
    finite = objective_values[np.all(np.isfinite(objective_values), axis=1)]
    if len(finite) == 0:
        raise ValueError("No finite objective vectors")
    return np.min(finite, axis=0), np.max(finite, axis=0)


def update_historical_ideal(
    current_ideal: np.ndarray | None,
    objective_values: np.ndarray,
) -> np.ndarray:
    """Update a MOEA/D ideal point without allowing historical regression."""
    values = np.atleast_2d(np.asarray(objective_values, dtype=float))
    finite = values[np.all(np.isfinite(values), axis=1)]
    if len(finite) == 0:
        if current_ideal is None:
            raise ValueError("Cannot initialize an ideal point without finite objectives")
        return np.asarray(current_ideal, dtype=float).copy()
    observed = np.min(finite, axis=0)
    if current_ideal is None:
        return observed.copy()
    current = np.asarray(current_ideal, dtype=float)
    if current.shape != observed.shape:
        raise ValueError("Historical ideal and objective dimensions differ")
    return np.minimum(current, observed)


def normalized_tchebycheff(
    objective_values: np.ndarray,
    weights: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    objective_values = np.atleast_2d(np.asarray(objective_values, dtype=float))
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    if len(objective_values) == 1 and len(weights) > 1:
        objective_values = np.repeat(objective_values, len(weights), axis=0)
    if len(weights) == 1 and len(objective_values) > 1:
        weights = np.repeat(weights, len(objective_values), axis=0)
    if objective_values.shape != weights.shape:
        raise ValueError("Objective and weight shapes are incompatible")
    scale = np.maximum(np.asarray(nadir) - np.asarray(ideal), epsilon)
    normalized = np.abs((objective_values - ideal) / scale)
    return np.max(normalized * weights, axis=1)


def phenotype_diagnostics(
    child_grid: np.ndarray | None,
    parent_grid: np.ndarray | None,
) -> dict[str, float | int | bool | None]:
    if child_grid is None or parent_grid is None:
        return {
            "same_phenotype": None,
            "moved_macros": None,
            "total_grid_displacement": None,
            "mean_grid_displacement": None,
            "max_grid_displacement": None,
        }
    child_grid = np.asarray(child_grid, dtype=int)
    parent_grid = np.asarray(parent_grid, dtype=int)
    if child_grid.shape != parent_grid.shape:
        raise ValueError("Phenotype-grid shapes differ")
    per_macro = np.sum(np.abs(child_grid - parent_grid), axis=1)
    return {
        "same_phenotype": bool(np.array_equal(child_grid, parent_grid)),
        "moved_macros": int(np.count_nonzero(per_macro)),
        "total_grid_displacement": int(np.sum(per_macro)),
        "mean_grid_displacement": float(np.mean(per_macro)),
        "max_grid_displacement": int(np.max(per_macro)),
    }
