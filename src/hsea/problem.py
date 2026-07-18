"""Bridge between the homework algorithms and BBOPlace-Bench's MGO placer."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from src.hsea.metrics import RoutingMetricConfig, evaluate_routing_metrics
from src.hsea.operators import Bounds
from src.placedb import PlaceDB
from src.placer.mgo_placer import MaskGuidedOptimizationPlacer

Array = np.ndarray


@dataclass
class EvaluationBatch:
    objectives: Array
    hpwl: Array
    overlap_rate: Array
    macro_positions: list[dict[str, tuple[float, float]]]


class PlacementEvaluator:
    """Evaluate MGO genotypes using a single shared placement decoder."""

    objective_names = ("hpwl", "rudy", "congestion")

    def __init__(self, args: SimpleNamespace, metric_config: RoutingMetricConfig | None = None):
        self.args = args
        if int(getattr(args, "n_cpu_max", 1)) > 1:
            import ray
            if not ray.is_initialized():
                ray.init(
                    num_cpus=int(args.n_cpu_max),
                    num_gpus=0,
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    log_to_driver=False,
                    logging_level="ERROR",
                )
        self.placedb = PlaceDB(args)
        self.placer = MaskGuidedOptimizationPlacer(args, self.placedb)
        self.bounds = Bounds(
            n_macro=self.placedb.node_cnt,
            n_grid_x=args.n_grid_x,
            n_grid_y=args.n_grid_y,
        )
        self.metric_config = metric_config or RoutingMetricConfig()
        self._single_cache: dict[bytes, tuple[float, float, dict[str, tuple[float, float]]]] = {}
        self._multi_cache: dict[bytes, tuple[Array, float, dict[str, tuple[float, float]]]] = {}
        self.n_evaluations = 0

    def _key(self, genome: Array) -> bytes:
        return np.asarray(genome, dtype=np.int32).tobytes()

    def decode(self, genome: Array) -> dict[str, tuple[float, float]]:
        genome = self.bounds.clip(np.asarray(genome).copy())
        return self.placer._genotype2phenotype(genome)

    def evaluate_single(self, population: Array, count_cached: bool = False) -> EvaluationBatch:
        population = self.bounds.clip(np.asarray(population).copy())
        hpwl = np.empty(len(population), dtype=float)
        overlap = np.empty(len(population), dtype=float)
        positions: list[dict[str, tuple[float, float]]] = [{} for _ in range(len(population))]
        missing_idx: list[int] = []
        missing_x: list[Array] = []
        for i, genome in enumerate(population):
            cached = self._single_cache.get(self._key(genome))
            if cached is None:
                missing_idx.append(i)
                missing_x.append(genome)
            else:
                hpwl[i], overlap[i], positions[i] = cached
                if count_cached:
                    self.n_evaluations += 1
        if missing_x:
            if len(missing_x) == 1:
                h, o, pos = self.placer._evaluate(missing_x[0])
                h_values, o_values, p_values = [h], [o], [pos]
            else:
                h_values, o_values, p_values = self.placer.evaluate(np.asarray(missing_x))
            for i, genome, h, o, pos in zip(missing_idx, missing_x, h_values, o_values, p_values):
                h = float(h)
                o = float(o)
                hpwl[i], overlap[i], positions[i] = h, o, pos
                self._single_cache[self._key(genome)] = (h, o, pos)
                self.n_evaluations += 1
        objectives = hpwl[:, None]
        return EvaluationBatch(objectives, hpwl, overlap, positions)

    def evaluate_multi(self, population: Array, count_cached: bool = False) -> EvaluationBatch:
        population = self.bounds.clip(np.asarray(population).copy())
        objectives = np.empty((len(population), 3), dtype=float)
        overlap = np.empty(len(population), dtype=float)
        positions: list[dict[str, tuple[float, float]]] = [{} for _ in range(len(population))]
        missing_idx: list[int] = []
        missing_x: list[Array] = []
        for i, genome in enumerate(population):
            cached = self._multi_cache.get(self._key(genome))
            if cached is None:
                missing_idx.append(i)
                missing_x.append(genome)
            else:
                objectives[i], overlap[i], positions[i] = cached
                if count_cached:
                    self.n_evaluations += 1
        if missing_x:
            single = self.evaluate_single(np.asarray(missing_x), count_cached=True)
            # evaluate_single increments the physical evaluation count exactly once.
            for local_i, (i, genome, h, o, pos) in enumerate(
                zip(missing_idx, missing_x, single.hpwl, single.overlap_rate, single.macro_positions)
            ):
                if not pos or not np.isfinite(h):
                    f = np.full(3, np.inf)
                else:
                    routing = evaluate_routing_metrics(pos, self.placedb, self.metric_config)
                    f = np.asarray([h, routing["rudy"], routing["congestion"]], dtype=float)
                objectives[i], overlap[i], positions[i] = f, float(o), pos
                self._multi_cache[self._key(genome)] = (f, float(o), pos)
        return EvaluationBatch(objectives, objectives[:, 0], overlap, positions)

    def clear_cache(self) -> None:
        self._single_cache.clear()
        self._multi_cache.clear()


def _find_benchmark_path(root: Path, base: str, benchmark: str) -> Path:
    expected = root / "benchmarks" / base / benchmark
    if (expected / f"{benchmark}.nodes").exists() or (expected / f"{benchmark}.def").exists():
        return expected
    benchmark_root = root / "benchmarks"
    if benchmark_root.exists():
        candidates = list(benchmark_root.rglob(f"{benchmark}.nodes")) + list(benchmark_root.rglob(f"{benchmark}.def"))
        if candidates:
            return candidates[0].parent
    raise FileNotFoundError(
        f"benchmark {benchmark!r} was not found under {benchmark_root}. "
        "Download the data described in README.md first."
    )


def build_evaluator(
    benchmark: str = "adaptec1",
    *,
    n_macro: int = 1_000_000,
    n_grid_x: int = 224,
    n_grid_y: int = 224,
    n_cpu: int = 1,
    result_path: str | Path | None = None,
    routing_bins: int = 24,
    top_fraction: float = 0.10,
) -> PlacementEvaluator:
    root = Path(__file__).resolve().parents[2]
    ispd = {f"adaptec{i}" for i in range(1, 5)} | {f"bigblue{i}" for i in range(1, 5)}
    iccad = {f"superblue{i}" for i in (1, 3, 4, 5, 7, 10, 16, 18)}
    if benchmark in ispd:
        base, benchmark_type = "ispd2005", "aux"
    elif benchmark in iccad:
        base, benchmark_type = "iccad2015", "def"
    else:
        raise ValueError(f"unsupported benchmark {benchmark!r}")
    benchmark_path = _find_benchmark_path(root, base, benchmark)
    if result_path is None:
        result_path = Path(tempfile.mkdtemp(prefix="hsea-evaluator-"))
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        benchmark=benchmark,
        placer="mgo",
        benchmark_base=base,
        benchmark_type=benchmark_type,
        benchmark_path=str(benchmark_path),
        n_macro=int(n_macro),
        n_grid_x=int(n_grid_x),
        n_grid_y=int(n_grid_y),
        n_cpu_max=int(n_cpu),
        rank_key="area_sum",
        result_path=str(result_path),
        eval_gp_hpwl=False,
        n_max_saving_placement=1,
        timeout_seconds=300,
    )
    metric_config = RoutingMetricConfig(
        bins_x=int(routing_bins),
        bins_y=int(routing_bins),
        top_fraction=float(top_fraction),
    )
    return PlacementEvaluator(args, metric_config)
