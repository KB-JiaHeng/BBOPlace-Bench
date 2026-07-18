import numpy as np

from src.hsea.operators import Bounds, pair_two_point_crossover, pair_uniform_crossover, swap_mutation


def _pairs(x, n):
    return {(int(x[i]), int(x[i+n])) for i in range(n)}


def test_uniform_crossover_preserves_parent_coordinate_pairs():
    bounds = Bounds(n_macro=6, n_grid_x=100, n_grid_y=100)
    a = np.concatenate([np.arange(6), np.arange(10, 16)])
    b = np.concatenate([np.arange(20, 26), np.arange(30, 36)])
    rng = np.random.default_rng(7)
    c1, c2 = pair_uniform_crossover(a, b, rng, bounds)
    allowed = _pairs(a, 6) | _pairs(b, 6)
    assert _pairs(c1, 6) <= allowed
    assert _pairs(c2, 6) <= allowed


def test_two_point_crossover_preserves_parent_coordinate_pairs():
    bounds = Bounds(n_macro=6, n_grid_x=100, n_grid_y=100)
    a = np.concatenate([np.arange(6), np.arange(10, 16)])
    b = np.concatenate([np.arange(20, 26), np.arange(30, 36)])
    c1, c2 = pair_two_point_crossover(a, b, np.random.default_rng(2), bounds)
    allowed = _pairs(a, 6) | _pairs(b, 6)
    assert _pairs(c1, 6) <= allowed
    assert _pairs(c2, 6) <= allowed


def test_swap_mutation_swaps_complete_locations():
    bounds = Bounds(n_macro=5, n_grid_x=100, n_grid_y=100)
    x = np.concatenate([np.arange(5), np.arange(10, 15)])
    y = swap_mutation(x, np.random.default_rng(3), bounds)
    assert _pairs(y, 5) == _pairs(x, 5)
    assert not np.array_equal(x, y)
