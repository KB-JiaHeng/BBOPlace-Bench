from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from experiments.audit_task2_metrics import (
    nondominated_indices,
    parse_saved_pl,
    positive_scalar_residual,
    relative_range,
)


class Task2MetricAuditHelperTest(unittest.TestCase):
    def test_saved_pl_is_converted_back_to_internal_canvas_coordinates(self):
        placedb = SimpleNamespace(
            macro_lst=["o0", "o1"],
            canvas_lx=100,
            canvas_ly=200,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "saved.pl"
            path.write_text(
                "UCLA pl 1.0\n"
                "o0 103 207 : N /FIXED\n"
                "o1 111 213 : N /FIXED\n"
            )
            positions = parse_saved_pl(path, placedb)
        self.assertEqual(positions, {"o0": (3.0, 7.0), "o1": (11.0, 13.0)})

    def test_nondominated_filter_operates_on_distinct_rows(self):
        F = np.asarray([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 4.0]])
        np.testing.assert_array_equal(nondominated_indices(F), [0, 1, 2])

    def test_relative_range_rejects_numerically_flat_signal(self):
        self.assertAlmostEqual(relative_range(np.asarray([2.0, 2.2])), 0.0952380952)
        self.assertLess(relative_range(np.asarray([2.0, 2.0 + 1e-12])), 1e-10)

    def test_positive_scalar_residual_distinguishes_near_copy(self):
        x = np.asarray([1.0, 2.0, 3.0])
        self.assertAlmostEqual(positive_scalar_residual(x, 4.0 * x), 0.0)
        self.assertGreater(positive_scalar_residual(x, np.asarray([4.0, 8.0, 13.0])), 1e-3)


if __name__ == "__main__":
    unittest.main()
