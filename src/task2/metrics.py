"""DREAMPlace RUDY metrics for the Task 2 macro-placement objective.

The bundled DREAMPlace source is not modified.  This wrapper calls its compiled
``rudy_cpp`` or ``rudy_cuda`` extension directly so the raw horizontal and
vertical demand maps remain available before DREAMPlace's capacity
normalization is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from utils.constant import INF


@dataclass(frozen=True)
class RudyMetricConfig:
    backend: str = "cpu"
    dtype: str = "float32"
    deterministic: bool = True
    num_bins_x: int = 512
    num_bins_y: int = 512
    unit_horizontal_capacity: float = 1.5625
    unit_vertical_capacity: float = 1.45
    hotspot_fraction: float = 0.10
    cpu_threads: int | None = None

    def validate(self) -> None:
        if self.backend not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported RUDY backend: {self.backend}")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError(f"Unsupported RUDY dtype: {self.dtype}")
        if self.num_bins_x <= 0 or self.num_bins_y <= 0:
            raise ValueError("Routing-grid dimensions must be positive")
        if self.unit_horizontal_capacity <= 0 or self.unit_vertical_capacity <= 0:
            raise ValueError("Routing capacities must be positive")
        if not 0 < self.hotspot_fraction <= 1:
            raise ValueError("hotspot_fraction must be in (0, 1]")
        if self.cpu_threads is not None and self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive when specified")


@dataclass
class RudyMetricResult:
    valid: bool
    rudy_top10: float
    congestion_top10: float
    total_overflow: float
    max_overflow: float
    overflow_fraction: float
    raw_peak: float
    utilization_peak: float
    horizontal_demand_map: np.ndarray | None = None
    vertical_demand_map: np.ndarray | None = None
    horizontal_utilization_map: np.ndarray | None = None
    vertical_utilization_map: np.ndarray | None = None
    route_utilization_map: np.ndarray | None = None

    @classmethod
    def invalid(cls) -> "RudyMetricResult":
        return cls(
            valid=False,
            rudy_top10=float(INF),
            congestion_top10=float(INF),
            total_overflow=float(INF),
            max_overflow=float(INF),
            overflow_fraction=float(INF),
            raw_peak=float(INF),
            utilization_peak=float(INF),
        )


@dataclass(frozen=True)
class RudyTopology:
    macro_names: tuple[str, ...]
    pin_macro_indices: np.ndarray
    pin_offset_x: np.ndarray
    pin_offset_y: np.ndarray
    netpin_start: np.ndarray
    flat_netpin: np.ndarray
    net_weights: np.ndarray

    @classmethod
    def from_placedb(cls, placedb: Any) -> "RudyTopology":
        macro_names = tuple(placedb.macro_lst)
        macro_index = {name: index for index, name in enumerate(macro_names)}
        net_names = sorted(
            placedb.net_info,
            key=lambda name: (
                int(placedb.net_info[name].get("id", 0)),
                str(name),
            ),
        )

        pin_macro_indices: list[int] = []
        pin_offset_x: list[float] = []
        pin_offset_y: list[float] = []
        flat_netpin: list[int] = []
        netpin_start = [0]
        net_weights: list[float] = []
        pin_id = 0

        for net_name in net_names:
            net = placedb.net_info[net_name]
            node_names = sorted(
                net["nodes"],
                key=lambda name: (
                    int(placedb.node_info[name].get("id", macro_index[name])),
                    str(name),
                ),
            )
            if len(node_names) <= 1:
                raise ValueError(f"RUDY topology contains degree <= 1 net: {net_name}")
            for node_name in node_names:
                node = placedb.node_info[node_name]
                pin = net["nodes"][node_name]
                pin_macro_indices.append(macro_index[node_name])
                pin_offset_x.append(float(node["size_x"]) / 2.0 + float(pin["x_offset"]))
                pin_offset_y.append(float(node["size_y"]) / 2.0 + float(pin["y_offset"]))
                flat_netpin.append(pin_id)
                pin_id += 1
            netpin_start.append(len(flat_netpin))
            net_weights.append(float(net.get("weight", 1.0)))

        return cls(
            macro_names=macro_names,
            pin_macro_indices=np.asarray(pin_macro_indices, dtype=np.int64),
            pin_offset_x=np.asarray(pin_offset_x, dtype=np.float64),
            pin_offset_y=np.asarray(pin_offset_y, dtype=np.float64),
            netpin_start=np.asarray(netpin_start, dtype=np.int32),
            flat_netpin=np.asarray(flat_netpin, dtype=np.int32),
            net_weights=np.asarray(net_weights, dtype=np.float64),
        )


def top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot aggregate an empty map")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, math.ceil(fraction * values.size))
    start = values.size - k
    top = np.partition(values, start)[start:]
    return float(np.mean(top))


def summarize_numpy_maps(
    horizontal_demand: np.ndarray,
    vertical_demand: np.ndarray,
    *,
    bin_area: float,
    horizontal_capacity: float,
    vertical_capacity: float,
    hotspot_fraction: float,
) -> dict[str, float]:
    horizontal_demand = np.asarray(horizontal_demand, dtype=float)
    vertical_demand = np.asarray(vertical_demand, dtype=float)
    if horizontal_demand.shape != vertical_demand.shape:
        raise ValueError("Horizontal and vertical map shapes differ")
    raw = horizontal_demand + vertical_demand
    horizontal_utilization = horizontal_demand / (bin_area * horizontal_capacity)
    vertical_utilization = vertical_demand / (bin_area * vertical_capacity)
    utilization = np.maximum(np.abs(horizontal_utilization), np.abs(vertical_utilization))
    overflow = np.maximum(utilization - 1.0, 0.0)
    return {
        "rudy_top10": top_fraction_mean(raw, hotspot_fraction),
        "congestion_top10": top_fraction_mean(utilization, hotspot_fraction),
        "total_overflow": float(np.sum(overflow)),
        "max_overflow": float(np.max(overflow)),
        "overflow_fraction": float(np.mean(utilization > 1.0)),
        "raw_peak": float(np.max(raw)),
        "utilization_peak": float(np.max(utilization)),
    }


class DreamplaceRudyMetric:
    """Evaluate macro-level RUDY and utilization with DREAMPlace kernels."""

    def __init__(self, placedb: Any, config: RudyMetricConfig):
        config.validate()
        self.placedb = placedb
        self.config = config
        self.topology = RudyTopology.from_placedb(placedb)
        self.bin_size_x = float(placedb.canvas_width) / config.num_bins_x
        self.bin_size_y = float(placedb.canvas_height) / config.num_bins_y
        self.bin_area = self.bin_size_x * self.bin_size_y
        if self.bin_area <= 0:
            raise ValueError("Non-positive routing bin area")

        try:
            import torch
            import dreamplace.configure as dreamplace_configure
            import dreamplace.ops.rudy.rudy_cpp as rudy_cpp
        except Exception as exc:  # pragma: no cover - exercised on remote build
            raise RuntimeError(
                "The unmodified bundled DREAMPlace RUDY extension is not "
                "installed/importable. Build DREAMPlace before Task 2."
            ) from exc

        self.torch = torch
        self.rudy_cpp = rudy_cpp
        self.rudy_cuda = None
        compile_config = dreamplace_configure.compile_configurations
        if config.backend == "cuda":
            if compile_config.get("CUDA_FOUND") != "TRUE":
                raise RuntimeError("DREAMPlace was built without CUDA support")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RUDY requested but torch.cuda is unavailable")
            try:
                import dreamplace.ops.rudy.rudy_cuda as rudy_cuda
            except Exception as exc:  # pragma: no cover - remote build path
                raise RuntimeError("DREAMPlace CUDA RUDY extension is unavailable") from exc
            self.rudy_cuda = rudy_cuda

        if config.cpu_threads is not None:
            torch.set_num_threads(config.cpu_threads)
        self.device = torch.device("cuda" if config.backend == "cuda" else "cpu")
        self.torch_dtype = torch.float32 if config.dtype == "float32" else torch.float64
        numpy_dtype = np.float32 if config.dtype == "float32" else np.float64

        self.pin_macro_indices = self.topology.pin_macro_indices
        self.pin_offset_x = self.topology.pin_offset_x.astype(numpy_dtype, copy=False)
        self.pin_offset_y = self.topology.pin_offset_y.astype(numpy_dtype, copy=False)
        self.netpin_start = torch.as_tensor(
            self.topology.netpin_start, dtype=torch.int32, device=self.device
        ).contiguous()
        self.flat_netpin = torch.as_tensor(
            self.topology.flat_netpin, dtype=torch.int32, device=self.device
        ).contiguous()
        self.net_weights = torch.as_tensor(
            self.topology.net_weights.astype(numpy_dtype, copy=False),
            dtype=self.torch_dtype,
            device=self.device,
        ).contiguous()

    def macro_position_array(self, macro_pos: dict[str, tuple[float, float]]) -> np.ndarray:
        if not macro_pos or len(macro_pos) != len(self.topology.macro_names):
            raise ValueError("Incomplete or empty macro placement")
        array = np.asarray(
            [macro_pos[name] for name in self.topology.macro_names],
            dtype=np.float32 if self.config.dtype == "float32" else np.float64,
        )
        if array.shape != (len(self.topology.macro_names), 2):
            raise ValueError(f"Unexpected macro-position shape: {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("Macro placement contains non-finite coordinates")
        return array

    def _pin_tensor(self, macro_pos: dict[str, tuple[float, float]]):
        macro_xy = self.macro_position_array(macro_pos)
        pin_x = macro_xy[self.pin_macro_indices, 0] + self.pin_offset_x
        pin_y = macro_xy[self.pin_macro_indices, 1] + self.pin_offset_y
        pin_pos = np.concatenate([pin_x, pin_y])
        return self.torch.as_tensor(
            pin_pos, dtype=self.torch_dtype, device=self.device
        ).contiguous()

    def evaluate(
        self,
        macro_pos: dict[str, tuple[float, float]],
        *,
        return_maps: bool = False,
    ) -> RudyMetricResult:
        try:
            pin_pos = self._pin_tensor(macro_pos)
        except (KeyError, TypeError, ValueError):
            return RudyMetricResult.invalid()

        torch = self.torch
        horizontal = torch.zeros(
            (self.config.num_bins_x, self.config.num_bins_y),
            dtype=self.torch_dtype,
            device=self.device,
        )
        vertical = torch.zeros_like(horizontal)
        kernel = self.rudy_cuda.forward if self.config.backend == "cuda" else self.rudy_cpp.forward
        kernel(
            pin_pos,
            self.netpin_start,
            self.flat_netpin,
            self.net_weights,
            self.bin_size_x,
            self.bin_size_y,
            0.0,
            0.0,
            float(self.placedb.canvas_width),
            float(self.placedb.canvas_height),
            self.config.num_bins_x,
            self.config.num_bins_y,
            int(self.config.deterministic),
            horizontal,
            vertical,
        )

        raw = horizontal + vertical
        horizontal_utilization = horizontal / (
            self.bin_area * self.config.unit_horizontal_capacity
        )
        vertical_utilization = vertical / (
            self.bin_area * self.config.unit_vertical_capacity
        )
        route_utilization = torch.maximum(
            horizontal_utilization.abs(), vertical_utilization.abs()
        )
        overflow = torch.clamp(route_utilization - 1.0, min=0.0)
        k = max(
            1,
            math.ceil(
                self.config.hotspot_fraction
                * self.config.num_bins_x
                * self.config.num_bins_y
            ),
        )
        rudy_top10 = torch.topk(raw.reshape(-1), k, sorted=False).values.mean()
        congestion_top10 = torch.topk(
            route_utilization.reshape(-1), k, sorted=False
        ).values.mean()

        result = RudyMetricResult(
            valid=True,
            rudy_top10=float(rudy_top10.item()),
            congestion_top10=float(congestion_top10.item()),
            total_overflow=float(overflow.sum().item()),
            max_overflow=float(overflow.max().item()),
            overflow_fraction=float((route_utilization > 1.0).float().mean().item()),
            raw_peak=float(raw.max().item()),
            utilization_peak=float(route_utilization.max().item()),
        )
        if return_maps:
            result.horizontal_demand_map = horizontal.detach().cpu().numpy().copy()
            result.vertical_demand_map = vertical.detach().cpu().numpy().copy()
            result.horizontal_utilization_map = (
                horizontal_utilization.detach().cpu().numpy().copy()
            )
            result.vertical_utilization_map = (
                vertical_utilization.detach().cpu().numpy().copy()
            )
            result.route_utilization_map = (
                route_utilization.detach().cpu().numpy().copy()
            )
        return result
