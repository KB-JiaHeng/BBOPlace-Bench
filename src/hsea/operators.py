"""Placement-aware evolutionary operators for the MGO representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Bounds:
    n_macro: int
    n_grid_x: int
    n_grid_y: int

    @property
    def n_var(self) -> int:
        return 2 * self.n_macro

    def clip(self, x: Array) -> Array:
        x = np.asarray(x)
        x[..., : self.n_macro] = np.clip(x[..., : self.n_macro], 0, self.n_grid_x - 1)
        x[..., self.n_macro :] = np.clip(x[..., self.n_macro :], 0, self.n_grid_y - 1)
        return x.astype(np.int64, copy=False)


def random_population(rng: np.random.Generator, size: int, bounds: Bounds) -> Array:
    x = np.empty((size, bounds.n_var), dtype=np.int64)
    x[:, : bounds.n_macro] = rng.integers(0, bounds.n_grid_x, size=(size, bounds.n_macro))
    x[:, bounds.n_macro :] = rng.integers(0, bounds.n_grid_y, size=(size, bounds.n_macro))
    return x


def clone_crossover(parent_a: Array, parent_b: Array, rng: np.random.Generator, bounds: Bounds) -> tuple[Array, Array]:
    return parent_a.copy(), parent_b.copy()


def pair_uniform_crossover(
    parent_a: Array,
    parent_b: Array,
    rng: np.random.Generator,
    bounds: Bounds,
    swap_probability: float = 0.5,
) -> tuple[Array, Array]:
    """Exchange complete macro coordinate pairs, never isolated x/y genes."""
    child_a, child_b = parent_a.copy(), parent_b.copy()
    mask = rng.random(bounds.n_macro) < swap_probability
    idx = np.flatnonzero(mask)
    for offset in (0, bounds.n_macro):
        genes = idx + offset
        child_a[genes], child_b[genes] = parent_b[genes], parent_a[genes]
    return child_a, child_b


def pair_two_point_crossover(
    parent_a: Array,
    parent_b: Array,
    rng: np.random.Generator,
    bounds: Bounds,
) -> tuple[Array, Array]:
    child_a, child_b = parent_a.copy(), parent_b.copy()
    if bounds.n_macro < 2:
        return child_a, child_b
    left, right = np.sort(rng.choice(bounds.n_macro + 1, size=2, replace=False))
    idx = np.arange(left, right)
    for offset in (0, bounds.n_macro):
        genes = idx + offset
        child_a[genes], child_b[genes] = parent_b[genes], parent_a[genes]
    return child_a, child_b


def pair_sbx_crossover(
    parent_a: Array,
    parent_b: Array,
    rng: np.random.Generator,
    bounds: Bounds,
    eta: float = 20.0,
) -> tuple[Array, Array]:
    """Integer-rounded simulated binary crossover applied to both coordinates."""
    a = parent_a.astype(float)
    b = parent_b.astype(float)
    u = rng.random(bounds.n_macro)
    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
    beta2 = np.concatenate([beta, beta])
    c1 = 0.5 * ((1.0 + beta2) * a + (1.0 - beta2) * b)
    c2 = 0.5 * ((1.0 - beta2) * a + (1.0 + beta2) * b)
    return bounds.clip(np.rint(c1)), bounds.clip(np.rint(c2))


def swap_mutation(x: Array, rng: np.random.Generator, bounds: Bounds, strength: int = 1) -> Array:
    y = x.copy()
    if bounds.n_macro < 2:
        return y
    for _ in range(max(1, strength)):
        i, j = rng.choice(bounds.n_macro, size=2, replace=False)
        y[[i, j]] = y[[j, i]]
        y[[i + bounds.n_macro, j + bounds.n_macro]] = y[[j + bounds.n_macro, i + bounds.n_macro]]
    return y


def shift_mutation(
    x: Array,
    rng: np.random.Generator,
    bounds: Bounds,
    strength: int = 1,
    max_step: int = 4,
) -> Array:
    y = x.copy()
    count = min(bounds.n_macro, max(1, strength))
    idx = rng.choice(bounds.n_macro, size=count, replace=False)
    dx = rng.integers(-max_step, max_step + 1, size=count)
    dy = rng.integers(-max_step, max_step + 1, size=count)
    zero = (dx == 0) & (dy == 0)
    dx[zero] = 1
    y[idx] += dx
    y[idx + bounds.n_macro] += dy
    return bounds.clip(y)


def reset_mutation(x: Array, rng: np.random.Generator, bounds: Bounds, strength: int = 1) -> Array:
    y = x.copy()
    count = min(bounds.n_macro, max(1, strength))
    idx = rng.choice(bounds.n_macro, size=count, replace=False)
    y[idx] = rng.integers(0, bounds.n_grid_x, size=count)
    y[idx + bounds.n_macro] = rng.integers(0, bounds.n_grid_y, size=count)
    return y


def shuffle_mutation(x: Array, rng: np.random.Generator, bounds: Bounds, strength: int = 4) -> Array:
    y = x.copy()
    count = min(bounds.n_macro, max(2, strength))
    idx = rng.choice(bounds.n_macro, size=count, replace=False)
    shuffled = rng.permutation(idx)
    y[idx] = x[shuffled]
    y[idx + bounds.n_macro] = x[shuffled + bounds.n_macro]
    return y


def polynomial_mutation(
    x: Array,
    rng: np.random.Generator,
    bounds: Bounds,
    strength: int = 1,
    eta: float = 20.0,
) -> Array:
    """Bounded integer polynomial mutation on selected macro coordinate pairs."""
    y = x.astype(float, copy=True)
    count = min(bounds.n_macro, max(1, strength))
    idx = rng.choice(bounds.n_macro, size=count, replace=False)
    for macro in idx:
        for gene, upper in ((macro, bounds.n_grid_x - 1), (macro + bounds.n_macro, bounds.n_grid_y - 1)):
            if upper <= 0:
                continue
            value = y[gene] / upper
            u = rng.random()
            if u < 0.5:
                delta = (2 * u + (1 - 2 * u) * (1 - value) ** (eta + 1)) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u) + 2 * (u - 0.5) * value ** (eta + 1)) ** (1 / (eta + 1))
            y[gene] += delta * upper
    return bounds.clip(np.rint(y))


CROSSOVERS: dict[str, Callable[..., tuple[Array, Array]]] = {
    "none": clone_crossover,
    "uniform": pair_uniform_crossover,
    "two_point": pair_two_point_crossover,
    "sbx": pair_sbx_crossover,
}

MUTATIONS: dict[str, Callable[..., Array]] = {
    "swap": swap_mutation,
    "shift": shift_mutation,
    "reset": reset_mutation,
    "shuffle": shuffle_mutation,
    "polynomial": polynomial_mutation,
}


def crossover_pair(
    name: str,
    parent_a: Array,
    parent_b: Array,
    rng: np.random.Generator,
    bounds: Bounds,
    probability: float,
    **kwargs,
) -> tuple[Array, Array]:
    if rng.random() >= probability:
        return parent_a.copy(), parent_b.copy()
    try:
        operator = CROSSOVERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown crossover {name!r}; choices={sorted(CROSSOVERS)}") from exc
    allowed = {
        "uniform": {"swap_probability"},
        "sbx": {"eta"},
        "none": set(),
        "two_point": set(),
    }[name]
    accepted = {k: v for k, v in kwargs.items() if k in allowed}
    return operator(parent_a, parent_b, rng, bounds, **accepted)


def mutate(
    name: str,
    x: Array,
    rng: np.random.Generator,
    bounds: Bounds,
    probability: float,
    **kwargs,
) -> Array:
    if rng.random() >= probability:
        return x.copy()
    try:
        operator = MUTATIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown mutation {name!r}; choices={sorted(MUTATIONS)}") from exc
    allowed = {
        "swap": {"strength"},
        "shift": {"strength", "max_step"},
        "reset": {"strength"},
        "shuffle": {"strength"},
        "polynomial": {"strength", "eta"},
    }[name]
    accepted = {k: v for k, v in kwargs.items() if k in allowed}
    return operator(x, rng, bounds, **accepted)
