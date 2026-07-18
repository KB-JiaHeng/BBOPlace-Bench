import numpy as np

from src.hsea.pareto import crowding_distance, environmental_selection, non_dominated_sort


def test_non_dominated_sort_known_fronts():
    f = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [3, 4], [5, 5]], dtype=float)
    fronts = non_dominated_sort(f)
    assert set(fronts[0]) == {0, 1, 2, 3}
    assert 4 in fronts[1]
    assert 5 in fronts[-1]


def test_environmental_selection_keeps_extremes():
    x = np.arange(10)[:, None]
    f = np.column_stack([np.arange(10), np.arange(9, -1, -1)]).astype(float)
    selected_x, selected_f = environmental_selection(x, f, 4)
    assert 0 in selected_x
    assert 9 in selected_x
    assert len(selected_x) == 4


def test_crowding_boundary_is_infinite():
    f = np.array([[0, 2], [1, 1], [2, 0]], dtype=float)
    d = crowding_distance(f)
    assert np.isinf(d[0]) and np.isinf(d[2])
