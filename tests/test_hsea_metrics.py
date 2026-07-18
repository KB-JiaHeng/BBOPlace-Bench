from types import SimpleNamespace

import numpy as np

from src.hsea.metrics import RoutingMetricConfig, compute_hpwl, evaluate_routing_metrics


def fake_db():
    return SimpleNamespace(
        canvas_width=100.0,
        canvas_height=100.0,
        node_info={
            "a": {"size_x": 10.0, "size_y": 10.0},
            "b": {"size_x": 10.0, "size_y": 10.0},
        },
        net_info={
            "n1": {
                "nodes": {
                    "a": {"x_offset": 0.0, "y_offset": 0.0},
                    "b": {"x_offset": 0.0, "y_offset": 0.0},
                }
            }
        },
    )


def test_hpwl_is_pin_bbox_length():
    db = fake_db()
    pos = {"a": (0.0, 0.0), "b": (50.0, 20.0)}
    assert compute_hpwl(pos, db) == 70.0


def test_rudy_and_congestion_are_finite_positive():
    db = fake_db()
    pos = {"a": (0.0, 0.0), "b": (50.0, 20.0)}
    metrics = evaluate_routing_metrics(pos, db, RoutingMetricConfig(bins_x=10, bins_y=10))
    assert np.isfinite(metrics["rudy"]) and metrics["rudy"] > 0
    assert np.isfinite(metrics["congestion"]) and metrics["congestion"] >= metrics["rudy"]
