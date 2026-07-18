"""Pareto utilities shared by NSGA-II, MOEA/D and result analysis."""

from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV

Array = np.ndarray


def dominates(a: Array, b: Array) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives: Array) -> list[Array]:
    objectives = np.asarray(objectives, dtype=float)
    n = len(objectives)
    dominates_set: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    first: list[int] = []
    for p in range(n):
        for q in range(p + 1, n):
            if dominates(objectives[p], objectives[q]):
                dominates_set[p].append(q)
                domination_count[q] += 1
            elif dominates(objectives[q], objectives[p]):
                dominates_set[q].append(p)
                domination_count[p] += 1
        if domination_count[p] == 0:
            first.append(p)
    fronts: list[Array] = []
    current = first
    while current:
        fronts.append(np.asarray(current, dtype=int))
        nxt: list[int] = []
        for p in current:
            for q in dominates_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        current = nxt
    return fronts


def non_dominated_indices(objectives: Array) -> Array:
    fronts = non_dominated_sort(objectives)
    return fronts[0] if fronts else np.empty(0, dtype=int)


def crowding_distance(objectives: Array) -> Array:
    objectives = np.asarray(objectives, dtype=float)
    n, m = objectives.shape
    if n == 0:
        return np.empty(0)
    if n <= 2:
        return np.full(n, np.inf)
    distance = np.zeros(n, dtype=float)
    for j in range(m):
        order = np.argsort(objectives[:, j], kind="mergesort")
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = objectives[order[-1], j] - objectives[order[0], j]
        if span <= 1e-15:
            continue
        distance[order[1:-1]] += (
            objectives[order[2:], j] - objectives[order[:-2], j]
        ) / span
    return distance


def rank_and_crowding(objectives: Array) -> tuple[Array, Array, list[Array]]:
    fronts = non_dominated_sort(objectives)
    rank = np.full(len(objectives), np.iinfo(np.int32).max, dtype=int)
    crowding = np.zeros(len(objectives), dtype=float)
    for r, front in enumerate(fronts):
        rank[front] = r
        crowding[front] = crowding_distance(objectives[front])
    return rank, crowding, fronts


def environmental_selection(population: Array, objectives: Array, size: int) -> tuple[Array, Array]:
    selected: list[int] = []
    for front in non_dominated_sort(objectives):
        if len(selected) + len(front) <= size:
            selected.extend(front.tolist())
            continue
        distances = crowding_distance(objectives[front])
        order = np.argsort(-distances, kind="mergesort")
        selected.extend(front[order[: size - len(selected)]].tolist())
        break
    idx = np.asarray(selected, dtype=int)
    return population[idx], objectives[idx]


def update_archive(
    archive_x: Array | None,
    archive_f: Array | None,
    new_x: Array,
    new_f: Array,
    max_size: int | None = None,
) -> tuple[Array, Array]:
    x = np.asarray(new_x) if archive_x is None else np.concatenate([archive_x, new_x])
    f = np.asarray(new_f, dtype=float) if archive_f is None else np.concatenate([archive_f, new_f])
    # Remove exact objective duplicates before Pareto filtering.
    _, unique_idx = np.unique(np.round(f, decimals=12), axis=0, return_index=True)
    x, f = x[unique_idx], f[unique_idx]
    nd = non_dominated_indices(f)
    x, f = x[nd], f[nd]
    if max_size is not None and len(x) > max_size:
        distance = crowding_distance(f)
        keep = np.argsort(-distance, kind="mergesort")[:max_size]
        x, f = x[keep], f[keep]
    return x, f


def normalize_objectives(values: Array, lower: Array, upper: Array) -> Array:
    return (np.asarray(values, dtype=float) - lower) / np.maximum(upper - lower, 1e-12)


def hypervolume(front: Array, lower: Array | None = None, upper: Array | None = None, ref: float = 1.1) -> float:
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return 0.0
    if lower is None:
        lower = np.min(front, axis=0)
    if upper is None:
        upper = np.max(front, axis=0)
    normalized = normalize_objectives(front, np.asarray(lower), np.asarray(upper))
    normalized = normalized[non_dominated_indices(normalized)]
    normalized = normalized[np.all(normalized <= ref, axis=1)]
    if len(normalized) == 0:
        return 0.0
    return float(HV(ref_point=np.full(front.shape[1], ref))(normalized))
