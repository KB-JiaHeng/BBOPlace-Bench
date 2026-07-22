from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from task2.metrics import DreamplaceRudyMetric, RudyMetricConfig


def synthetic_placedb(pin_names: list[str]) -> SimpleNamespace:
    node_info = {
        name: {"id": i, "size_x": 0.0, "size_y": 0.0}
        for i, name in enumerate(pin_names)
    }
    return SimpleNamespace(
        macro_lst=pin_names,
        node_cnt=len(pin_names),
        node_info=node_info,
        net_info={
            "n": {
                "id": 0,
                "weight": 1.0,
                "nodes": {
                    name: {"x_offset": 0.0, "y_offset": 0.0}
                    for name in pin_names
                },
            }
        },
        canvas_width=4.0,
        canvas_height=4.0,
    )


class DreamplaceKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = os.environ.get("TASK2_RUDY_BACKEND", "cpu")
        cls.required = os.environ.get("TASK2_REQUIRE_DREAMPLACE") == "1"
        try:
            cls.metric2 = DreamplaceRudyMetric(
                synthetic_placedb(["a", "b"]),
                RudyMetricConfig(
                    backend=cls.backend,
                    num_bins_x=2,
                    num_bins_y=2,
                    unit_horizontal_capacity=1.0,
                    unit_vertical_capacity=1.0,
                    hotspot_fraction=0.25,
                    deterministic=True,
                ),
            )
        except RuntimeError as exc:
            if cls.required:
                raise
            raise unittest.SkipTest(str(exc))

    def test_two_pin_full_canvas_has_hand_calculated_uniform_map(self):
        result = self.metric2.evaluate(
            {"a": (0.0, 0.0), "b": (4.0, 4.0)},
            return_maps=True,
        )
        np.testing.assert_allclose(result.horizontal_demand_map, 1.0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(result.vertical_demand_map, 1.0, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(result.rudy_top10, 2.0, places=5)
        self.assertAlmostEqual(result.congestion_top10, 0.25, places=5)
        self.assertEqual(result.total_overflow, 0.0)

    def test_degree_four_applies_bundled_wdm_weight(self):
        metric4 = DreamplaceRudyMetric(
            synthetic_placedb(["a", "b", "c", "d"]),
            RudyMetricConfig(
                backend=self.backend,
                num_bins_x=2,
                num_bins_y=2,
                unit_horizontal_capacity=1.0,
                unit_vertical_capacity=1.0,
                hotspot_fraction=0.25,
                deterministic=True,
            ),
        )
        result = metric4.evaluate(
            {
                "a": (0.0, 0.0),
                "b": (4.0, 0.0),
                "c": (0.0, 4.0),
                "d": (4.0, 4.0),
            },
            return_maps=True,
        )
        np.testing.assert_allclose(
            result.horizontal_demand_map,
            1.0828,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.vertical_demand_map,
            1.0828,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_zero_width_bbox_is_finite(self):
        result = self.metric2.evaluate(
            {"a": (1.0, 0.0), "b": (1.0, 4.0)},
            return_maps=True,
        )
        self.assertTrue(np.isfinite(result.rudy_top10))
        self.assertTrue(np.isfinite(result.congestion_top10))
        self.assertTrue(np.all(np.isfinite(result.route_utilization_map)))


if __name__ == "__main__":
    unittest.main()
