"""HPWL, RUDY and congestion-proxy metrics for macro placement.

RUDY follows the rectangular uniform wire-density model: a net's estimated
wire demand is spread uniformly over its pin bounding box.  The congestion
metric is a capacity-adjusted proxy, because ISPD2005 Bookshelf data does not
contain detailed layer/track capacities.  Macro blockage reduces each bin's
available capacity and the scalar score is the mean of the hottest bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RoutingMetricConfig:
    bins_x: int = 24
    bins_y: int = 24
    top_fraction: float = 0.10
    blockage_factor: float = 0.85
    minimum_capacity: float = 0.05


def _pin_coordinates(net: dict[str, Any], macro_pos: dict[str, tuple[float, float]], placedb: Any) -> tuple[Array, Array]:
    xs: list[float] = []
    ys: list[float] = []
    for macro, pin in net["nodes"].items():
        if macro not in macro_pos:
            continue
        x, y = macro_pos[macro]
        info = placedb.node_info[macro]
        xs.append(float(x) + 0.5 * info["size_x"] + float(pin.get("x_offset", 0.0)))
        ys.append(float(y) + 0.5 * info["size_y"] + float(pin.get("y_offset", 0.0)))
    return np.asarray(xs), np.asarray(ys)


def compute_hpwl(macro_pos: dict[str, tuple[float, float]], placedb: Any) -> float:
    if not macro_pos:
        return float("inf")
    total = 0.0
    for net in placedb.net_info.values():
        xs, ys = _pin_coordinates(net, macro_pos, placedb)
        if len(xs) < 2:
            continue
        weight = float(net.get("weight", 1.0))
        total += weight * ((xs.max() - xs.min()) + (ys.max() - ys.min()))
    return float(total)


def _bin_geometry(placedb: Any, config: RoutingMetricConfig) -> tuple[Array, Array, float, float]:
    x_edges = np.linspace(0.0, float(placedb.canvas_width), config.bins_x + 1)
    y_edges = np.linspace(0.0, float(placedb.canvas_height), config.bins_y + 1)
    return x_edges, y_edges, x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]


def _overlap_vector(edges: Array, low: float, high: float) -> Array:
    return np.maximum(0.0, np.minimum(edges[1:], high) - np.maximum(edges[:-1], low))


def compute_rudy_maps(
    macro_pos: dict[str, tuple[float, float]],
    placedb: Any,
    config: RoutingMetricConfig,
) -> tuple[Array, Array, Array]:
    x_edges, y_edges, bin_w, bin_h = _bin_geometry(placedb, config)
    horizontal = np.zeros((config.bins_x, config.bins_y), dtype=float)
    vertical = np.zeros_like(horizontal)
    min_w, min_h = 0.5 * bin_w, 0.5 * bin_h
    for net in placedb.net_info.values():
        xs, ys = _pin_coordinates(net, macro_pos, placedb)
        if len(xs) < 2:
            continue
        x0, x1 = float(np.min(xs)), float(np.max(xs))
        y0, y1 = float(np.min(ys)), float(np.max(ys))
        width = max(x1 - x0, min_w)
        height = max(y1 - y0, min_h)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        x0, x1 = max(0.0, cx - width / 2), min(float(placedb.canvas_width), cx + width / 2)
        y0, y1 = max(0.0, cy - height / 2), min(float(placedb.canvas_height), cy + height / 2)
        width, height = max(x1 - x0, 1e-12), max(y1 - y0, 1e-12)
        overlap = np.outer(_overlap_vector(x_edges, x0, x1), _overlap_vector(y_edges, y0, y1))
        area = width * height
        weight = float(net.get("weight", 1.0))
        # Horizontal and vertical expected wire length per unit bin area.
        horizontal += weight * (width / area) * overlap / (bin_w * bin_h)
        vertical += weight * (height / area) * overlap / (bin_w * bin_h)
    return horizontal, vertical, horizontal + vertical


def compute_blockage_map(
    macro_pos: dict[str, tuple[float, float]],
    placedb: Any,
    config: RoutingMetricConfig,
) -> Array:
    x_edges, y_edges, bin_w, bin_h = _bin_geometry(placedb, config)
    blocked = np.zeros((config.bins_x, config.bins_y), dtype=float)
    for macro, (x, y) in macro_pos.items():
        info = placedb.node_info[macro]
        overlap = np.outer(
            _overlap_vector(x_edges, float(x), float(x) + float(info["size_x"])),
            _overlap_vector(y_edges, float(y), float(y) + float(info["size_y"])),
        )
        blocked += overlap / (bin_w * bin_h)
    return np.clip(blocked, 0.0, 1.0)


def hot_bin_mean(values: Array, fraction: float) -> float:
    flat = np.asarray(values, dtype=float).ravel()
    if len(flat) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(flat) * fraction)))
    return float(np.mean(np.partition(flat, len(flat) - k)[-k:]))


def evaluate_routing_metrics(
    macro_pos: dict[str, tuple[float, float]],
    placedb: Any,
    config: RoutingMetricConfig,
    return_maps: bool = False,
) -> dict[str, Any]:
    horizontal, vertical, total = compute_rudy_maps(macro_pos, placedb, config)
    blockage = compute_blockage_map(macro_pos, placedb, config)
    capacity = np.maximum(config.minimum_capacity, 1.0 - config.blockage_factor * blockage)
    congestion = total / capacity
    result: dict[str, Any] = {
        "rudy": hot_bin_mean(total, config.top_fraction),
        "congestion": hot_bin_mean(congestion, config.top_fraction),
        "rudy_peak": float(np.max(total, initial=0.0)),
        "congestion_peak": float(np.max(congestion, initial=0.0)),
    }
    if return_maps:
        result.update(
            horizontal_rudy=horizontal,
            vertical_rudy=vertical,
            rudy_map=total,
            blockage_map=blockage,
            congestion_map=congestion,
        )
    return result
