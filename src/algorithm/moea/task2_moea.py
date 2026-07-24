"""Controlled Task 2 implementations of standard NSGA-II and MOEA/D."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from task2.benchmark_fingerprint import (
    placedb_fingerprint,
    validate_expected_fingerprint,
)
from task2.evaluator import Task2Evaluation, Task2ObjectiveEvaluator
from utils.constant import INF
from utils.random_parser import set_seed

from ..basic_algo import BasicAlgo
from .task2_core import (
    binary_tournament_rank_crowding,
    dominance_relation,
    genotype_hash,
    mask_to_hex,
    moead_neighbors,
    moead_reference_vectors,
    normalized_tchebycheff,
    phenotype_diagnostics,
    rank_and_crowding,
    sample_integer_population,
    swap_macro_pairs,
    uniform_complementary_children,
    update_historical_ideal,
)


TRACE_COLUMNS = [
    "evaluation_id",
    "algorithm",
    "phase",
    "generation_or_sweep",
    "subproblem_id",
    "parent1_id",
    "parent2_id",
    "genotype_hash",
    "phenotype_hash",
    "decode_success",
    "hpwl",
    "rudy_top10",
    "congestion_top10",
    "total_overflow",
    "max_overflow",
    "overflow_fraction",
    "overlap_rate",
    "uniform_mask_hex",
    "selected_complementary_child",
    "swap_index_a",
    "swap_index_b",
    "swap_guide_l1_distance",
    "swap_macro_area_a",
    "swap_macro_area_b",
    "swap_macro_rank_a",
    "swap_macro_rank_b",
    "genotype_distance_parent1",
    "genotype_distance_parent2",
    "same_phenotype_parent1",
    "same_phenotype_parent2",
    "moved_macros_parent1",
    "moved_macros_parent2",
    "total_grid_displacement_parent1",
    "total_grid_displacement_parent2",
    "mean_grid_displacement_parent1",
    "mean_grid_displacement_parent2",
    "max_grid_displacement_parent1",
    "max_grid_displacement_parent2",
    "delta_hpwl_parent1",
    "delta_hpwl_parent2",
    "delta_rudy_top10_parent1",
    "delta_rudy_top10_parent2",
    "delta_congestion_top10_parent1",
    "delta_congestion_top10_parent2",
    "dominance_parent1",
    "dominance_parent2",
    "survived",
    "replacement_count",
    "replacement_slot_ids",
    "moead_ideal_before",
    "moead_ideal_after",
    "moead_nadir",
    "archive_entered",
    "decode_seconds",
    "metric_seconds",
]

GENERATION_COLUMNS = [
    "evaluation_count",
    "generation_or_sweep",
    "algorithm",
    "valid_population",
    "unique_genotypes",
    "unique_phenotypes",
    "max_single_phenotype_fraction",
    "evaluated_unique_genotypes_cumulative",
    "nondominated_population",
    "min_hpwl",
    "min_congestion_top10",
    "max_hpwl",
    "max_congestion_top10",
    "archive_size",
    "decode_failures_cumulative",
    "duplicate_rejections",
    "replacement_count",
    "elapsed_seconds",
    "valid_evaluations_per_second",
]


@dataclass
class Task2Individual:
    evaluation: Task2Evaluation
    rank: float = np.nan
    crowding: float = np.nan

    @property
    def id(self) -> int:
        return self.evaluation.evaluation_id

    @property
    def x(self) -> np.ndarray:
        return self.evaluation.x

    @property
    def f(self) -> np.ndarray:
        return self.evaluation.f

    @property
    def valid(self) -> bool:
        return self.evaluation.decode_success


@dataclass(frozen=True)
class VariationRecord:
    parent1: Task2Individual
    parent2: Task2Individual
    mask_hex: str
    selected_child: int
    swap_a: int
    swap_b: int


class Task2MOEA(BasicAlgo):
    """One runner with shared Task 1 mechanics and two standard MOEA baselines."""

    def __init__(self, args: Any, placer: Any, logger: Any):
        super().__init__(args=args, placer=placer, logger=logger)
        self._validate_config()
        self.method = str(args.task2_method)
        self.node_count = int(placer.placedb.node_cnt)
        self.macro_rank_by_name = {
            name: rank for rank, name in enumerate(placer.ranked_macro)
        }
        self.max_evals = int(args.max_evals)
        self.population_size = int(args.n_population)
        self.offspring_size = int(args.n_offsprings)
        self.start_time = time.perf_counter()
        self.archive: list[Task2Individual] = []
        self.trace_count = 0
        self.decode_failures = 0
        self.evaluated_genotype_hashes: set[str] = set()
        self.moead_historical_ideal: np.ndarray | None = None

        protocol_path = Path(args.ROOT_DIR) / "experiments" / "task2_protocol.yaml"
        with protocol_path.open() as f:
            self.protocol = yaml.safe_load(f)

        selection = self.protocol["inheritance_from_task1"]["macro_selection"]
        expected_fingerprint = {
            "node_count": int(selection["expected_node_count"]),
            "net_count": int(selection["expected_net_count"]),
            "macro_names_sha256": str(selection["expected_macro_names_sha256"]),
            "macro_geometry_sha256": str(
                selection["expected_macro_geometry_sha256"]
            ),
            "net_topology_sha256": str(selection["expected_net_topology_sha256"]),
            "macro_area_sum": int(selection["expected_macro_area_sum"]),
            "canvas": [int(value) for value in selection["expected_canvas"]],
        }
        self.benchmark_fingerprint = placedb_fingerprint(
            placer.placedb,
            benchmark_path=args.benchmark_path,
        )
        validate_expected_fingerprint(
            self.benchmark_fingerprint,
            expected_fingerprint,
        )
        with (Path(args.result_path) / "placedb_fingerprint.json").open("w") as f:
            json.dump(self.benchmark_fingerprint, f, indent=2, sort_keys=True)

        self.snapshot_evals = set(
            int(v) for v in self.protocol["logging"]["population_snapshots_at_evaluations"]
        )
        self.saved_snapshots: set[int] = set()

        self.trace_path = Path(args.result_path) / "evaluation_trace.csv"
        self.generation_path = Path(args.result_path) / "generation_metrics.csv"
        self.moead_state_path = Path(args.result_path) / "moead_state.csv"
        self.snapshot_dir = Path(args.result_path) / "population_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRACE_COLUMNS).writeheader()
        with self.generation_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=GENERATION_COLUMNS).writeheader()
        if self.method == "moead":
            with self.moead_state_path.open("w", newline="") as f:
                csv.DictWriter(
                    f,
                    fieldnames=[
                        "evaluation_count",
                        "sweep",
                        "ideal",
                        "nadir",
                        "slot_evaluation_ids",
                        "slot_phenotype_hashes",
                        "unique_slot_phenotypes",
                        "max_single_phenotype_fraction",
                    ],
                ).writeheader()

        self.objective = Task2ObjectiveEvaluator(args, placer)

    def _validate_config(self) -> None:
        required = {
            "placer": "mgo",
            "algorithm": "task2_moea",
            "benchmark": "adaptec1",
            "benchmark_variant": "adaptec1_upstream_native_top512_macro_subset",
            "n_macro": 512,
            "n_population": 20,
            "n_sampling_repeat": 1,
            "n_offsprings": 20,
            "eliminate_duplicates": True,
            "duplicate_retry_limit": 100,
            "checkpoint_enabled": False,
            "n_grid_x": 224,
            "n_grid_y": 224,
            "rank_key": "area_sum",
            "sampling": "random",
            "crossover": "uniform",
            "mutation": "swap",
            "crossover_prob": 1.0,
            "uniform_prob_var": 0.5,
            "eval_gp_hpwl": False,
            "rudy_dtype": "float32",
            "rudy_deterministic": True,
            "rudy_num_bins_x": 512,
            "rudy_num_bins_y": 512,
            "rudy_unit_horizontal_capacity": 1.5625,
            "rudy_unit_vertical_capacity": 1.45,
            "rudy_hotspot_fraction": 0.10,
            "moead_n_neighbors": 10,
        }
        for key, expected in required.items():
            actual = getattr(self.args, key)
            if actual != expected:
                raise ValueError(
                    f"Task 2 protocol violation: {key}={actual!r}, expected {expected!r}"
                )
        if self.args.task2_method not in {"nsga2", "moead"}:
            raise ValueError(f"Unknown Task 2 method: {self.args.task2_method}")
        if self.args.rudy_backend not in {"cpu", "cuda"}:
            raise ValueError(f"Unknown RUDY backend: {self.args.rudy_backend}")
        smoke = bool(getattr(self.args, "task2_smoke_test", False))
        if smoke:
            if not self.args.n_population <= self.args.max_evals < 10000:
                raise ValueError("Task 2 smoke budget must be in [20, 10000)")
        elif self.args.max_evals != 10000:
            raise ValueError("Formal Task 2 runs require exactly 10000 evaluations")
        if hasattr(self.args, "checkpoint"):
            raise ValueError("Task 2 protocol does not permit checkpoint loading")

    def _archive_add(self, individual: Task2Individual) -> bool:
        if not individual.valid:
            return False
        for incumbent in self.archive:
            relation = dominance_relation(incumbent.f, individual.f)
            if relation in {"dominates", "equal"}:
                return False
        self.archive = [
            incumbent
            for incumbent in self.archive
            if dominance_relation(individual.f, incumbent.f) != "dominates"
        ]
        self.archive.append(individual)
        return True

    @staticmethod
    def _csv_optional(value: Any) -> Any:
        return "" if value is None else value

    @staticmethod
    def _objective_delta(
        child: Task2Individual,
        parent: Task2Individual | None,
        attribute: str,
    ) -> Any:
        if parent is None or not child.valid or not parent.valid:
            return ""
        return float(getattr(child.evaluation, attribute) - getattr(parent.evaluation, attribute))

    @staticmethod
    def _json_vector(value: np.ndarray | None) -> str:
        if value is None:
            return ""
        return json.dumps(np.asarray(value, dtype=float).tolist(), separators=(",", ":"))

    @staticmethod
    def _relation(child: Task2Individual, parent: Task2Individual | None) -> str:
        if parent is None:
            return ""
        if not child.valid:
            return "invalid_child"
        if not parent.valid:
            return "dominates_invalid_parent"
        return dominance_relation(child.f, parent.f)

    def _trace(
        self,
        individual: Task2Individual,
        *,
        phase: str,
        generation_or_sweep: int,
        subproblem_id: int | None,
        variation: VariationRecord | None,
        survived: bool,
        replacement_count: int,
        archive_entered: bool,
        replacement_slots: list[int] | None = None,
        ideal_before: np.ndarray | None = None,
        ideal_after: np.ndarray | None = None,
        nadir: np.ndarray | None = None,
    ) -> None:
        parent1 = variation.parent1 if variation else None
        parent2 = variation.parent2 if variation else None
        d1 = phenotype_diagnostics(
            individual.evaluation.macro_grid,
            parent1.evaluation.macro_grid if parent1 else None,
        )
        d2 = phenotype_diagnostics(
            individual.evaluation.macro_grid,
            parent2.evaluation.macro_grid if parent2 else None,
        )
        swap_metadata: dict[str, Any] = {
            "distance": "",
            "area_a": "",
            "area_b": "",
            "rank_a": "",
            "rank_b": "",
        }
        if variation is not None:
            macro_a = self.placer.placedb.macro_lst[variation.swap_a]
            macro_b = self.placer.placedb.macro_lst[variation.swap_b]
            x = individual.x
            n = self.node_count
            swap_metadata = {
                "distance": int(
                    abs(int(x[variation.swap_a]) - int(x[variation.swap_b]))
                    + abs(int(x[variation.swap_a + n]) - int(x[variation.swap_b + n]))
                ),
                "area_a": int(self.placer.placedb.node_info[macro_a]["area"]),
                "area_b": int(self.placer.placedb.node_info[macro_b]["area"]),
                "rank_a": int(self.macro_rank_by_name[macro_a]),
                "rank_b": int(self.macro_rank_by_name[macro_b]),
            }
        row = {
            "evaluation_id": individual.id,
            "algorithm": self.method,
            "phase": phase,
            "generation_or_sweep": generation_or_sweep,
            "subproblem_id": "" if subproblem_id is None else subproblem_id,
            "parent1_id": "" if parent1 is None else parent1.id,
            "parent2_id": "" if parent2 is None else parent2.id,
            "genotype_hash": individual.evaluation.genotype_hash,
            "phenotype_hash": individual.evaluation.phenotype_hash or "",
            "decode_success": int(individual.valid),
            "hpwl": individual.evaluation.hpwl,
            "rudy_top10": individual.evaluation.rudy_top10,
            "congestion_top10": individual.evaluation.congestion_top10,
            "total_overflow": individual.evaluation.total_overflow,
            "max_overflow": individual.evaluation.max_overflow,
            "overflow_fraction": individual.evaluation.overflow_fraction,
            "overlap_rate": individual.evaluation.overlap_rate,
            "uniform_mask_hex": variation.mask_hex if variation else "",
            "selected_complementary_child": (
                variation.selected_child if variation else ""
            ),
            "swap_index_a": variation.swap_a if variation else "",
            "swap_index_b": variation.swap_b if variation else "",
            "swap_guide_l1_distance": swap_metadata["distance"],
            "swap_macro_area_a": swap_metadata["area_a"],
            "swap_macro_area_b": swap_metadata["area_b"],
            "swap_macro_rank_a": swap_metadata["rank_a"],
            "swap_macro_rank_b": swap_metadata["rank_b"],
            "genotype_distance_parent1": (
                "" if parent1 is None else int(np.count_nonzero(individual.x != parent1.x))
            ),
            "genotype_distance_parent2": (
                "" if parent2 is None else int(np.count_nonzero(individual.x != parent2.x))
            ),
            "same_phenotype_parent1": (
                "" if d1["same_phenotype"] is None else int(d1["same_phenotype"])
            ),
            "same_phenotype_parent2": (
                "" if d2["same_phenotype"] is None else int(d2["same_phenotype"])
            ),
            "moved_macros_parent1": self._csv_optional(d1["moved_macros"]),
            "moved_macros_parent2": self._csv_optional(d2["moved_macros"]),
            "total_grid_displacement_parent1": self._csv_optional(
                d1["total_grid_displacement"]
            ),
            "total_grid_displacement_parent2": self._csv_optional(
                d2["total_grid_displacement"]
            ),
            "mean_grid_displacement_parent1": self._csv_optional(
                d1["mean_grid_displacement"]
            ),
            "mean_grid_displacement_parent2": self._csv_optional(
                d2["mean_grid_displacement"]
            ),
            "max_grid_displacement_parent1": self._csv_optional(
                d1["max_grid_displacement"]
            ),
            "max_grid_displacement_parent2": self._csv_optional(
                d2["max_grid_displacement"]
            ),
            "delta_hpwl_parent1": self._objective_delta(individual, parent1, "hpwl"),
            "delta_hpwl_parent2": self._objective_delta(individual, parent2, "hpwl"),
            "delta_rudy_top10_parent1": self._objective_delta(
                individual, parent1, "rudy_top10"
            ),
            "delta_rudy_top10_parent2": self._objective_delta(
                individual, parent2, "rudy_top10"
            ),
            "delta_congestion_top10_parent1": self._objective_delta(
                individual, parent1, "congestion_top10"
            ),
            "delta_congestion_top10_parent2": self._objective_delta(
                individual, parent2, "congestion_top10"
            ),
            "dominance_parent1": self._relation(individual, parent1),
            "dominance_parent2": self._relation(individual, parent2),
            "survived": int(survived),
            "replacement_count": replacement_count,
            "replacement_slot_ids": json.dumps(
                replacement_slots or [], separators=(",", ":")
            ),
            "moead_ideal_before": self._json_vector(ideal_before),
            "moead_ideal_after": self._json_vector(ideal_after),
            "moead_nadir": self._json_vector(nadir),
            "archive_entered": int(archive_entered),
            "decode_seconds": individual.evaluation.decode_seconds,
            "metric_seconds": individual.evaluation.metric_seconds,
        }
        with self.trace_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=TRACE_COLUMNS).writerow(row)
        self.trace_count += 1

    def _save_population(self, path: Path, population: list[Task2Individual]) -> None:
        if not population:
            np.savez_compressed(
                path,
                X=np.empty((0, 2 * self.node_count), dtype=np.int64),
                F=np.empty((0, 2), dtype=float),
                evaluation_id=np.empty(0, dtype=np.int64),
                decode_success=np.empty(0, dtype=bool),
                phenotype_hash=np.empty(0, dtype=str),
                rudy_top10=np.empty(0, dtype=float),
                total_overflow=np.empty(0, dtype=float),
                max_overflow=np.empty(0, dtype=float),
                overflow_fraction=np.empty(0, dtype=float),
                macro_grid=np.empty((0, self.node_count, 2), dtype=np.int32),
            )
            return
        macro_grid = np.stack(
            [
                ind.evaluation.macro_grid
                if ind.evaluation.macro_grid is not None
                else np.full((self.node_count, 2), -1, dtype=np.int32)
                for ind in population
            ]
        )
        np.savez_compressed(
            path,
            X=np.stack([ind.x for ind in population]),
            F=np.stack([ind.f for ind in population]),
            evaluation_id=np.asarray([ind.id for ind in population], dtype=np.int64),
            decode_success=np.asarray([ind.valid for ind in population], dtype=bool),
            phenotype_hash=np.asarray(
                [ind.evaluation.phenotype_hash or "" for ind in population]
            ),
            rudy_top10=np.asarray(
                [ind.evaluation.rudy_top10 for ind in population], dtype=float
            ),
            total_overflow=np.asarray(
                [ind.evaluation.total_overflow for ind in population], dtype=float
            ),
            max_overflow=np.asarray(
                [ind.evaluation.max_overflow for ind in population], dtype=float
            ),
            overflow_fraction=np.asarray(
                [ind.evaluation.overflow_fraction for ind in population], dtype=float
            ),
            macro_grid=macro_grid,
        )

    def _save_initial(self, population: list[Task2Individual]) -> None:
        path = Path(self.args.result_path) / "initial_population.npz"
        self._save_population(path, population)
        X = np.ascontiguousarray(np.stack([ind.x for ind in population]))
        row_hashes = sorted(hashlib.sha256(row.tobytes()).digest() for row in X)
        metadata = {
            "n_individuals": len(population),
            "shape": list(X.shape),
            "dtype": str(X.dtype),
            "x_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
            "x_set_sha256": hashlib.sha256(b"".join(row_hashes)).hexdigest(),
            "ordering": "shared_generation_order_not_fitness_sorted",
        }
        with (Path(self.args.result_path) / "initial_population_hash.json").open("w") as f:
            json.dump(metadata, f, indent=2)

    def _maybe_snapshot(self, population: list[Task2Individual]) -> None:
        n_eval = self.objective.true_n_eval
        if n_eval in self.snapshot_evals and n_eval not in self.saved_snapshots:
            self._save_population(
                self.snapshot_dir / f"evaluation_{n_eval}.npz",
                population,
            )
            self.saved_snapshots.add(n_eval)

    def _population_metrics(
        self,
        population: list[Task2Individual],
        step: int,
        *,
        duplicate_rejections: int,
        replacement_count: int,
    ) -> None:
        valid = [ind for ind in population if ind.valid]
        if valid:
            F = np.stack([ind.f for ind in valid])
            _, ranks, _ = rank_and_crowding(F, n_survive=len(valid))
            nondominated = int(np.count_nonzero(ranks == 0))
            min_hpwl, min_congestion = np.min(F, axis=0)
            max_hpwl, max_congestion = np.max(F, axis=0)
        else:
            nondominated = 0
            min_hpwl = min_congestion = max_hpwl = max_congestion = float(INF)
        phenotype_hashes = [
            ind.evaluation.phenotype_hash
            for ind in valid
            if ind.evaluation.phenotype_hash is not None
        ]
        phenotype_counts = Counter(phenotype_hashes)
        unique_phenotypes = len(phenotype_counts)
        max_single_phenotype_fraction = (
            max(phenotype_counts.values()) / len(valid)
            if phenotype_counts and valid
            else 0.0
        )
        elapsed = time.perf_counter() - self.start_time
        row = {
            "evaluation_count": self.objective.true_n_eval,
            "generation_or_sweep": step,
            "algorithm": self.method,
            "valid_population": len(valid),
            "unique_genotypes": len({ind.evaluation.genotype_hash for ind in population}),
            "unique_phenotypes": unique_phenotypes,
            "max_single_phenotype_fraction": max_single_phenotype_fraction,
            "evaluated_unique_genotypes_cumulative": len(self.evaluated_genotype_hashes),
            "nondominated_population": nondominated,
            "min_hpwl": min_hpwl,
            "min_congestion_top10": min_congestion,
            "max_hpwl": max_hpwl,
            "max_congestion_top10": max_congestion,
            "archive_size": len(self.archive),
            "decode_failures_cumulative": self.decode_failures,
            "duplicate_rejections": duplicate_rejections,
            "replacement_count": replacement_count,
            "elapsed_seconds": elapsed,
            "valid_evaluations_per_second": (
                (self.objective.true_n_eval - self.decode_failures) / max(elapsed, 1e-12)
            ),
        }
        with self.generation_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=GENERATION_COLUMNS).writerow(row)

    def _initialize(self) -> list[Task2Individual]:
        # This reset deliberately mirrors pymoo minimize(seed=...) immediately
        # before Task 1 sampling, ensuring paired Task 1/Task 2 initial X.
        set_seed(int(self.args.seed))
        lower = np.zeros(2 * self.node_count, dtype=np.int64)
        upper = np.full(2 * self.node_count, int(self.args.n_grid_x) - 1, dtype=np.int64)
        X = sample_integer_population(self.population_size, lower, upper)
        initial_hashes = {genotype_hash(x) for x in X}
        if len(initial_hashes) != self.population_size:
            raise RuntimeError("Duplicate genotype occurred in the initial population")
        self.evaluated_genotype_hashes.update(initial_hashes)
        evaluated = self.objective.evaluate_many(X, parallel_decode=True)
        population = [Task2Individual(evaluation=e) for e in evaluated]
        self.decode_failures += sum(not ind.valid for ind in population)
        for individual in population:
            entered = self._archive_add(individual)
            self._trace(
                individual,
                phase="initialization",
                generation_or_sweep=0,
                subproblem_id=None,
                variation=None,
                survived=True,
                replacement_count=0,
                archive_entered=entered,
            )
        self._save_initial(population)
        self._maybe_snapshot(population)
        self._population_metrics(
            population,
            0,
            duplicate_rejections=0,
            replacement_count=0,
        )
        return population

    @staticmethod
    def _assign_rank_crowding(population: list[Task2Individual]) -> None:
        _, ranks, crowding = rank_and_crowding(
            np.stack([ind.f for ind in population]),
            n_survive=len(population),
        )
        for ind, rank, distance in zip(population, ranks, crowding):
            ind.rank = float(rank)
            ind.crowding = float(distance)

    def _run_nsga2(self, population: list[Task2Individual]) -> list[Task2Individual]:
        generation = 0
        while self.objective.true_n_eval < self.max_evals:
            generation += 1
            self._assign_rank_crowding(population)
            ranks = np.asarray([ind.rank for ind in population])
            crowding = np.asarray([ind.crowding for ind in population])
            target = min(
                self.offspring_size,
                self.max_evals - self.objective.true_n_eval,
            )
            seen = set(self.evaluated_genotype_hashes)
            candidate_X: list[np.ndarray] = []
            variation_records: list[VariationRecord] = []
            duplicate_rejections = 0
            mating_iterations = 0

            while (
                len(candidate_X) < target
                and mating_iterations < int(self.args.duplicate_retry_limit)
            ):
                mating_iterations += 1
                p1 = population[binary_tournament_rank_crowding(ranks, crowding)]
                p2 = population[binary_tournament_rank_crowding(ranks, crowding)]
                child0, child1, mask = uniform_complementary_children(
                    p1.x,
                    p2.x,
                    float(self.args.uniform_prob_var),
                )
                for selected_child, raw_child in enumerate((child0, child1)):
                    child, swap_a, swap_b = swap_macro_pairs(raw_child, self.node_count)
                    digest = genotype_hash(child)
                    if digest in seen:
                        duplicate_rejections += 1
                        continue
                    seen.add(digest)
                    self.evaluated_genotype_hashes.add(digest)
                    candidate_X.append(child)
                    variation_records.append(
                        VariationRecord(
                            parent1=p1,
                            parent2=p2,
                            mask_hex=mask_to_hex(mask),
                            selected_child=selected_child,
                            swap_a=swap_a,
                            swap_b=swap_b,
                        )
                    )
                    if len(candidate_X) == target:
                        break

            if not candidate_X:
                raise RuntimeError(
                    "NSGA-II generated no unique offspring within the retry limit"
                )
            if len(candidate_X) < target:
                logging.warning(
                    "NSGA-II generated %d/%d unique offspring; preserving exact "
                    "evaluation accounting with the smaller batch.",
                    len(candidate_X),
                    target,
                )

            evaluated = self.objective.evaluate_many(
                np.stack(candidate_X),
                parallel_decode=True,
            )
            offspring = [Task2Individual(evaluation=e) for e in evaluated]
            self.decode_failures += sum(not ind.valid for ind in offspring)
            archive_entered = [self._archive_add(ind) for ind in offspring]

            combined = population + offspring
            selected_indices, ranks_all, crowding_all = rank_and_crowding(
                np.stack([ind.f for ind in combined]),
                n_survive=self.population_size,
            )
            for ind, rank, distance in zip(combined, ranks_all, crowding_all):
                ind.rank = float(rank)
                ind.crowding = float(distance)
            population = [combined[int(index)] for index in selected_indices]
            surviving_ids = {ind.id for ind in population}

            for ind, variation, entered in zip(
                offspring,
                variation_records,
                archive_entered,
            ):
                self._trace(
                    ind,
                    phase="evolution",
                    generation_or_sweep=generation,
                    subproblem_id=None,
                    variation=variation,
                    survived=ind.id in surviving_ids,
                    replacement_count=int(ind.id in surviving_ids),
                    archive_entered=entered,
                )
            self._maybe_snapshot(population)
            self._population_metrics(
                population,
                generation,
                duplicate_rejections=duplicate_rejections,
                replacement_count=sum(ind.id in surviving_ids for ind in offspring),
            )
        return population

    def _record_moead_state(
        self,
        population: list[Task2Individual],
        sweep: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
    ) -> None:
        phenotype_hashes = [ind.evaluation.phenotype_hash or "" for ind in population]
        counts = Counter(value for value in phenotype_hashes if value)
        with self.moead_state_path.open("a", newline="") as f:
            csv.DictWriter(
                f,
                fieldnames=[
                    "evaluation_count",
                    "sweep",
                    "ideal",
                    "nadir",
                    "slot_evaluation_ids",
                    "slot_phenotype_hashes",
                    "unique_slot_phenotypes",
                    "max_single_phenotype_fraction",
                ],
            ).writerow(
                {
                    "evaluation_count": self.objective.true_n_eval,
                    "sweep": sweep,
                    "ideal": self._json_vector(ideal),
                    "nadir": self._json_vector(nadir),
                    "slot_evaluation_ids": json.dumps(
                        [ind.id for ind in population], separators=(",", ":")
                    ),
                    "slot_phenotype_hashes": json.dumps(
                        phenotype_hashes, separators=(",", ":")
                    ),
                    "unique_slot_phenotypes": len(counts),
                    "max_single_phenotype_fraction": (
                        max(counts.values()) / len(population) if counts else 0.0
                    ),
                }
            )

    def _run_moead(self, population: list[Task2Individual]) -> list[Task2Individual]:
        weights = moead_reference_vectors(self.population_size)
        neighbors = moead_neighbors(weights, int(self.args.moead_n_neighbors))
        np.savez_compressed(
            Path(self.args.result_path) / "moead_structure.npz",
            reference_vectors=weights,
            neighbors=neighbors,
        )
        valid_initial = [ind.f for ind in population if ind.valid]
        if not valid_initial:
            raise RuntimeError("MOEA/D cannot initialize without a valid objective vector")
        self.moead_historical_ideal = update_historical_ideal(
            None, np.stack(valid_initial)
        )
        initial_nadir = np.max(np.stack(valid_initial), axis=0)
        self._record_moead_state(
            population, 0, self.moead_historical_ideal, initial_nadir
        )
        sweep = 0
        while self.objective.true_n_eval < self.max_evals:
            sweep += 1
            evaluated_in_sweep = 0
            duplicate_rejections = 0
            replacements_in_sweep = 0
            for subproblem in np.random.permutation(self.population_size):
                if self.objective.true_n_eval >= self.max_evals:
                    break
                variation: VariationRecord | None = None
                child: np.ndarray | None = None
                current_hashes = set(self.evaluated_genotype_hashes)
                for _ in range(int(self.args.duplicate_retry_limit)):
                    parent_slots = np.random.choice(
                        neighbors[subproblem],
                        size=2,
                        replace=False,
                    )
                    p1 = population[int(parent_slots[0])]
                    p2 = population[int(parent_slots[1])]
                    child0, child1, mask = uniform_complementary_children(
                        p1.x,
                        p2.x,
                        float(self.args.uniform_prob_var),
                    )
                    selected_child = int(np.random.randint(0, 2))
                    raw_child = child0 if selected_child == 0 else child1
                    proposed, swap_a, swap_b = swap_macro_pairs(
                        raw_child,
                        self.node_count,
                    )
                    digest = genotype_hash(proposed)
                    if digest in current_hashes:
                        duplicate_rejections += 1
                        continue
                    current_hashes.add(digest)
                    self.evaluated_genotype_hashes.add(digest)
                    child = proposed
                    variation = VariationRecord(
                        parent1=p1,
                        parent2=p2,
                        mask_hex=mask_to_hex(mask),
                        selected_child=selected_child,
                        swap_a=swap_a,
                        swap_b=swap_b,
                    )
                    break
                if child is None or variation is None:
                    continue

                evaluation = self.objective.evaluate_many(
                    child[None, :],
                    parallel_decode=False,
                )[0]
                individual = Task2Individual(evaluation=evaluation)
                self.decode_failures += int(not individual.valid)
                entered = self._archive_add(individual)
                replacement_slots: list[int] = []
                valid_current = [ind.f for ind in population if ind.valid]
                if not valid_current:
                    raise RuntimeError("MOEA/D population lost all valid objective vectors")
                ideal_before = np.asarray(self.moead_historical_ideal, dtype=float).copy()
                ideal_after = ideal_before.copy()
                nadir = np.max(np.stack(valid_current), axis=0)

                if individual.valid:
                    ideal_after = update_historical_ideal(ideal_before, individual.f)
                    self.moead_historical_ideal = ideal_after.copy()
                    normalization_values = np.stack(valid_current + [individual.f])
                    nadir = np.max(normalization_values, axis=0)
                    slots = neighbors[subproblem]
                    old_values = normalized_tchebycheff(
                        np.stack([population[int(slot)].f for slot in slots]),
                        weights[slots],
                        ideal_after,
                        nadir,
                    )
                    new_values = normalized_tchebycheff(
                        individual.f,
                        weights[slots],
                        ideal_after,
                        nadir,
                    )
                    replacement_slots = [
                        int(slot)
                        for slot, new_value, old_value in zip(
                            slots,
                            new_values,
                            old_values,
                        )
                        if new_value < old_value
                    ]
                    for slot in replacement_slots:
                        population[slot] = individual

                replacement_count = len(replacement_slots)
                replacements_in_sweep += replacement_count
                evaluated_in_sweep += 1
                self._trace(
                    individual,
                    phase="evolution",
                    generation_or_sweep=sweep,
                    subproblem_id=int(subproblem),
                    variation=variation,
                    survived=replacement_count > 0,
                    replacement_count=replacement_count,
                    archive_entered=entered,
                    replacement_slots=replacement_slots,
                    ideal_before=ideal_before,
                    ideal_after=ideal_after,
                    nadir=nadir,
                )
                self._maybe_snapshot(population)

            if evaluated_in_sweep == 0:
                raise RuntimeError(
                    "MOEA/D generated no unique offspring during an entire sweep"
                )
            valid_current = [ind.f for ind in population if ind.valid]
            current_nadir = np.max(np.stack(valid_current), axis=0)
            self._record_moead_state(
                population,
                sweep,
                np.asarray(self.moead_historical_ideal, dtype=float),
                current_nadir,
            )
            self._population_metrics(
                population,
                sweep,
                duplicate_rejections=duplicate_rejections,
                replacement_count=replacements_in_sweep,
            )
        return population

    def run(self) -> None:
        population = self._initialize()
        if self.method == "nsga2":
            population = self._run_nsga2(population)
        else:
            population = self._run_moead(population)

        if self.objective.true_n_eval != self.max_evals:
            raise RuntimeError(
                f"Evaluation-budget mismatch: {self.objective.true_n_eval} != "
                f"{self.max_evals}"
            )
        if self.trace_count != self.max_evals:
            raise RuntimeError(
                f"Trace-count mismatch: {self.trace_count} != {self.max_evals}"
            )
        if len(population) != self.population_size:
            raise RuntimeError("Final population-size invariant was violated")

        self._save_population(
            Path(self.args.result_path) / "final_population.npz",
            population,
        )
        self._save_population(
            Path(self.args.result_path) / "offline_archive.npz",
            self.archive,
        )
        elapsed = time.perf_counter() - self.start_time
        completion = {
            "status": "complete",
            "task": 2,
            "method": self.method,
            "seed": int(self.args.seed),
            "definition_fingerprint": str(
                getattr(self.args, "task2_definition_fingerprint", "direct-test")
            ),
            "placedb_macro_names_sha256": self.benchmark_fingerprint[
                "macro_names_sha256"
            ],
            "placedb_net_topology_sha256": self.benchmark_fingerprint[
                "net_topology_sha256"
            ],
            "true_evaluation_count": self.objective.true_n_eval,
            "algorithm_evaluation_count": self.objective.true_n_eval,
            "trace_count": self.trace_count,
            "population_size": len(population),
            "archive_size": len(self.archive),
            "decode_failures": self.decode_failures,
            "evaluated_unique_genotypes": len(self.evaluated_genotype_hashes),
            "duplicate_semantics": "run_level_genotype_hash_exclusion",
            "moead_historical_ideal": (
                None
                if self.moead_historical_ideal is None
                else self.moead_historical_ideal.tolist()
            ),
            "rudy_backend": self.args.rudy_backend,
            "rudy_cpu_threads": getattr(self.args, "rudy_cpu_threads", None),
            "elapsed_seconds": elapsed,
            "valid_evaluations_per_second": (
                (self.max_evals - self.decode_failures) / max(elapsed, 1e-12)
            ),
        }
        with (Path(self.args.result_path) / "run_complete.json").open("w") as f:
            json.dump(completion, f, indent=2)
