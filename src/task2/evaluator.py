"""Exact Task 2 objective evaluation and phenotype diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from algorithm.moea.task2_core import genotype_hash, phenotype_hash
from task2.metrics import DreamplaceRudyMetric, RudyMetricConfig, RudyMetricResult
from utils.constant import INF


@dataclass
class Task2Evaluation:
    evaluation_id: int
    x: np.ndarray
    genotype_hash: str
    decode_success: bool
    macro_pos: dict[str, tuple[float, float]] | None
    macro_grid: np.ndarray | None
    phenotype_hash: str | None
    f: np.ndarray
    hpwl: float
    rudy_top10: float
    congestion_top10: float
    total_overflow: float
    max_overflow: float
    overflow_fraction: float
    overlap_rate: float
    decode_seconds: float
    metric_seconds: float


class Task2ObjectiveEvaluator:
    """Decode MGO candidates and evaluate HPWL plus DREAMPlace congestion."""

    def __init__(self, args: Any, placer: Any):
        self.args = args
        self.placer = placer
        self.placedb = placer.placedb
        self.true_n_eval = 0
        self.metric = DreamplaceRudyMetric(
            self.placedb,
            RudyMetricConfig(
                backend=str(args.rudy_backend),
                dtype=str(args.rudy_dtype),
                deterministic=bool(args.rudy_deterministic),
                num_bins_x=int(args.rudy_num_bins_x),
                num_bins_y=int(args.rudy_num_bins_y),
                unit_horizontal_capacity=float(args.rudy_unit_horizontal_capacity),
                unit_vertical_capacity=float(args.rudy_unit_vertical_capacity),
                hotspot_fraction=float(args.rudy_hotspot_fraction),
                cpu_threads=int(args.rudy_cpu_threads)
                if getattr(args, "rudy_cpu_threads", None) is not None
                else None,
            ),
        )

    def _decode_many(
        self,
        X: np.ndarray,
        *,
        parallel: bool,
    ) -> tuple[list[tuple[float, float, dict]], float]:
        start = time.perf_counter()
        if parallel and len(X) > 1 and int(self.args.n_cpu_max) > 1:
            try:
                import ray
                from placer.basic_placer import evaluate_placer

                if ray.is_initialized():
                    futures = [evaluate_placer.remote(self.placer, x) for x in X]
                    results = list(ray.get(futures))
                else:
                    results = [self.placer._evaluate(x) for x in X]
            except Exception:
                # Never hide an objective failure after remote jobs were launched.
                # A local fallback is permitted only before a usable Ray runtime
                # exists; otherwise the exception must surface.
                try:
                    import ray

                    if ray.is_initialized():
                        raise
                except ModuleNotFoundError:
                    pass
                results = [self.placer._evaluate(x) for x in X]
        else:
            results = [self.placer._evaluate(x) for x in X]
        elapsed = time.perf_counter() - start
        return results, elapsed

    def _macro_grid(
        self,
        macro_pos: dict[str, tuple[float, float]],
    ) -> np.ndarray:
        grid = np.asarray(
            [
                (
                    int(round(macro_pos[name][0] / self.placer.grid_width)),
                    int(round(macro_pos[name][1] / self.placer.grid_height)),
                )
                for name in self.placedb.macro_lst
            ],
            dtype=np.int32,
        )
        if grid.shape != (self.placedb.node_cnt, 2):
            raise RuntimeError(f"Unexpected phenotype grid shape: {grid.shape}")
        return grid

    def evaluate_many(
        self,
        X: np.ndarray,
        *,
        parallel_decode: bool,
    ) -> list[Task2Evaluation]:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.int64))
        if X.ndim != 2 or X.shape[1] != 2 * self.placedb.node_cnt:
            raise ValueError(
                f"Expected candidates of shape (n,{2 * self.placedb.node_cnt}), "
                f"got {X.shape}"
            )
        decoded, decode_elapsed = self._decode_many(X, parallel=parallel_decode)
        decode_seconds = decode_elapsed / max(1, len(X))
        evaluations: list[Task2Evaluation] = []

        for x, (hpwl, overlap_rate, macro_pos) in zip(X, decoded):
            self.true_n_eval += 1
            hpwl = float(hpwl)
            overlap_rate = float(overlap_rate)
            decode_success = bool(
                macro_pos
                and len(macro_pos) == self.placedb.node_cnt
                and np.isfinite(hpwl)
                and hpwl < float(INF)
            )

            if decode_success:
                macro_grid = self._macro_grid(macro_pos)
                start = time.perf_counter()
                metric = self.metric.evaluate(macro_pos)
                metric_seconds = time.perf_counter() - start
                if not metric.valid:
                    raise RuntimeError(
                        "DREAMPlace metric rejected a complete legal MGO placement"
                    )
                f = np.asarray([hpwl, metric.congestion_top10], dtype=float)
                phenotype_digest = phenotype_hash(macro_grid)
                stored_macro_pos = dict(macro_pos)
            else:
                metric = RudyMetricResult.invalid()
                metric_seconds = 0.0
                f = np.asarray([float(INF), float(INF)], dtype=float)
                macro_grid = None
                phenotype_digest = None
                stored_macro_pos = None

            evaluations.append(
                Task2Evaluation(
                    evaluation_id=self.true_n_eval,
                    x=x.copy(),
                    genotype_hash=genotype_hash(x),
                    decode_success=decode_success,
                    macro_pos=stored_macro_pos,
                    macro_grid=macro_grid,
                    phenotype_hash=phenotype_digest,
                    f=f,
                    hpwl=hpwl if decode_success else float(INF),
                    rudy_top10=metric.rudy_top10,
                    congestion_top10=metric.congestion_top10,
                    total_overflow=metric.total_overflow,
                    max_overflow=metric.max_overflow,
                    overflow_fraction=metric.overflow_fraction,
                    overlap_rate=overlap_rate,
                    decode_seconds=decode_seconds,
                    metric_seconds=metric_seconds,
                )
            )
        return evaluations
