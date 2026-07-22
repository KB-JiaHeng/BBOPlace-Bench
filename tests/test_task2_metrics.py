from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from task2.metrics import RudyTopology, summarize_numpy_maps, top_fraction_mean


class Task2MetricPureTest(unittest.TestCase):
    def test_top_fraction_uses_ceiling_and_largest_values(self):
        values = np.arange(10, dtype=float)
        self.assertEqual(top_fraction_mean(values, 0.20), 8.5)
        self.assertEqual(top_fraction_mean(values, 0.01), 9.0)

    def test_demand_and_capacity_summaries_are_distinct(self):
        horizontal = np.array([[0.0, 2.0], [4.0, 0.0]])
        vertical = np.array([[0.0, 4.0], [0.0, 2.0]])
        summary = summarize_numpy_maps(
            horizontal,
            vertical,
            bin_area=2.0,
            horizontal_capacity=2.0,
            vertical_capacity=1.0,
            hotspot_fraction=0.5,
        )
        self.assertAlmostEqual(summary["rudy_top10"], 5.0)
        self.assertAlmostEqual(summary["congestion_top10"], 1.5)
        self.assertAlmostEqual(summary["total_overflow"], 1.0)
        self.assertAlmostEqual(summary["max_overflow"], 1.0)
        self.assertAlmostEqual(summary["overflow_fraction"], 0.25)

    def test_topology_uses_pin_centers_and_offsets(self):
        placedb = SimpleNamespace(
            macro_lst=["a", "b"],
            node_info={
                "a": {"id": 0, "size_x": 10, "size_y": 6},
                "b": {"id": 1, "size_x": 4, "size_y": 8},
            },
            net_info={
                "n": {
                    "id": 0,
                    "weight": 2.0,
                    "nodes": {
                        "a": {"x_offset": 1.0, "y_offset": -1.0},
                        "b": {"x_offset": -2.0, "y_offset": 3.0},
                    },
                }
            },
        )
        topology = RudyTopology.from_placedb(placedb)
        np.testing.assert_array_equal(topology.pin_macro_indices, [0, 1])
        np.testing.assert_allclose(topology.pin_offset_x, [6.0, 0.0])
        np.testing.assert_allclose(topology.pin_offset_y, [2.0, 7.0])
        np.testing.assert_array_equal(topology.netpin_start, [0, 2])
        np.testing.assert_array_equal(topology.flat_netpin, [0, 1])
        np.testing.assert_allclose(topology.net_weights, [2.0])


if __name__ == "__main__":
    unittest.main()
