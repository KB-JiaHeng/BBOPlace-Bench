#!/usr/bin/env python3
"""Audit Task 2 objective validity on stratified, phenotype-deduplicated layouts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm.moea.task2_core import sample_integer_population
from config.benchmark import benchmark_dict, benchmark_type_dict
from placedb import PlaceDB
from placer import REGISTRY as PLACER_REGISTRY
from task2.benchmark_fingerprint import placedb_fingerprint, validate_expected_fingerprint
from task2.hashing import genotype_hash
from task2.metrics import DreamplaceRudyMetric, RudyMetricConfig
from utils.compute_res import comp_overlap, comp_res
from utils.random_parser import set_seed


METRIC_FIELDS = [
    "rudy_top10",
    "congestion_top10",
    "total_overflow",
    "max_overflow",
    "overflow_fraction",
    "raw_peak",
    "utilization_peak",
]


def load_args() -> SimpleNamespace:
    config: dict[str, Any] = {}
    for path in [
        ROOT / "config" / "default.yaml",
        ROOT / "config" / "placer" / "mgo.yaml",
        ROOT / "config" / "algorithm" / "task2_moea.yaml",
    ]:
        with path.open() as f:
            config.update(yaml.safe_load(f) or {})
    with (ROOT / "experiments" / "task2_protocol.yaml").open() as f:
        protocol = yaml.safe_load(f)
    benchmark = "adaptec1"
    benchmark_base = next(base for base, names in benchmark_dict.items() if benchmark in names)
    config.update(
        {
            "ROOT_DIR": str(ROOT),
            "THIRDPARTY_DIR": str(ROOT / "thirdparty"),
            "SOURCE_DIR": str(SRC),
            "benchmark": benchmark,
            "benchmark_base": benchmark_base,
            "benchmark_type": benchmark_type_dict[benchmark_base],
            "benchmark_path": str(ROOT / "benchmarks" / benchmark_base / benchmark),
            "n_macro": int(protocol["inheritance_from_task1"]["n_macro"]),
            "seed": 1,
            "error_redirect": False,
            "result_path": str(ROOT / "experiments" / "task2_smoke" / "metric_audit_runtime"),
        }
    )
    return SimpleNamespace(**config)


def phenotype_digest(macro_pos: dict[str, tuple[float, float]], macro_names: list[str]) -> str:
    coordinates = np.ascontiguousarray(
        np.asarray([macro_pos[name] for name in macro_names], dtype=np.float64)
    )
    return hashlib.sha256(coordinates.tobytes()).hexdigest()


def parse_seed(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", str(path))
    return int(match.group(1)) if match else None


def parse_saved_pl(path: Path, placedb: Any) -> dict[str, tuple[float, float]]:
    macro_names = set(placedb.macro_lst)
    macro_pos: dict[str, tuple[float, float]] = {}
    with path.open() as f:
        for raw_line in f:
            fields = raw_line.split()
            if len(fields) < 3 or fields[0] not in macro_names:
                continue
            macro_pos[fields[0]] = (
                float(fields[1]) - float(placedb.canvas_lx),
                float(fields[2]) - float(placedb.canvas_ly),
            )
    if set(macro_pos) != macro_names:
        missing = sorted(macro_names - set(macro_pos))[:5]
        raise RuntimeError(f"Incomplete saved placement {path}; missing {missing}")
    return macro_pos


def collect_layout_bank(
    placer: Any,
    seeds: list[int],
    population_size: int,
    task1_results_root: Path,
) -> list[dict[str, Any]]:
    """Collect unique Task 1 initial layouts, saved best placements, and fallback initials."""
    placedb = placer.placedb
    node_count = int(placedb.node_cnt)
    records: list[dict[str, Any]] = []
    seen_genotypes: set[str] = set()
    seen_phenotypes: set[str] = set()

    def add_genotype(x: np.ndarray, source: str, source_path: str, seed: int | None, index: int) -> None:
        x = np.ascontiguousarray(np.asarray(x, dtype=np.int64))
        digest = genotype_hash(x)
        if digest in seen_genotypes:
            return
        seen_genotypes.add(digest)
        hpwl, overlap_rate, macro_pos = placer._evaluate(x)
        valid = bool(
            macro_pos
            and len(macro_pos) == node_count
            and np.isfinite(float(hpwl))
            and float(hpwl) < 1e16
        )
        phenotype = (
            phenotype_digest(dict(macro_pos), placedb.macro_lst) if valid else None
        )
        if phenotype:
            seen_phenotypes.add(phenotype)
        records.append(
            {
                "source": source,
                "source_path": source_path,
                "seed": seed,
                "index": index,
                "genotype_sha256": digest,
                "x": x.copy(),
                "decode_success": valid,
                "hpwl": float(hpwl) if valid else 1e16,
                "overlap_rate": float(overlap_rate),
                "macro_pos": dict(macro_pos) if valid else None,
                "phenotype_sha256": phenotype,
            }
        )

    initial_paths = sorted(task1_results_root.glob("task1_*/mgo/ea/seed_*/initial_population.npz"))
    for path in initial_paths:
        archive = np.load(path)
        if "X" not in archive:
            continue
        for index, x in enumerate(archive["X"]):
            add_genotype(x, "task1_initial", str(path), parse_seed(path), index)

    placement_paths = sorted(task1_results_root.glob("task1_*/mgo/ea/seed_*/placements/*.pl"))
    for index, path in enumerate(placement_paths):
        macro_pos = parse_saved_pl(path, placedb)
        phenotype = phenotype_digest(macro_pos, placedb.macro_lst)
        if phenotype in seen_phenotypes:
            continue
        seen_phenotypes.add(phenotype)
        hpwl = float(comp_res(macro_pos=macro_pos, placedb=placedb))
        overlap_rate = float(comp_overlap(macro_pos=macro_pos, placedb=placedb))
        records.append(
            {
                "source": "task1_saved_best",
                "source_path": str(path),
                "seed": parse_seed(path),
                "index": index,
                "genotype_sha256": "",
                "x": None,
                "decode_success": np.isfinite(hpwl) and hpwl < 1e16,
                "hpwl": hpwl,
                "overlap_rate": overlap_rate,
                "macro_pos": macro_pos,
                "phenotype_sha256": phenotype,
            }
        )

    lower = np.zeros(2 * node_count, dtype=np.int64)
    upper = np.full(2 * node_count, int(placer.args.n_grid_x) - 1, dtype=np.int64)
    for seed in seeds:
        set_seed(seed)
        X = sample_integer_population(population_size, lower, upper)
        for index, x in enumerate(X):
            add_genotype(x, "generated_initial_fallback", "", seed, index)

    return records


def metric_config(backend: str, cpu_threads: int, bins: int) -> RudyMetricConfig:
    return RudyMetricConfig(
        backend=backend,
        dtype="float32",
        deterministic=True,
        num_bins_x=bins,
        num_bins_y=bins,
        unit_horizontal_capacity=1.5625,
        unit_vertical_capacity=1.45,
        hotspot_fraction=0.10,
        cpu_threads=cpu_threads,
    )


def scoped_placedb(placedb: Any, net_names: set[str]) -> Any:
    scoped = copy.copy(placedb)
    scoped.net_info = {
        name: data for name, data in placedb.net_info.items() if name in net_names
    }
    return scoped


def nondominated_indices(F: np.ndarray) -> np.ndarray:
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)):
            keep[i] = False
    return np.flatnonzero(keep)


def finite_correlation(function: Any, x: np.ndarray, y: np.ndarray) -> float:
    value = float(function(x, y).statistic)
    return value if np.isfinite(value) else float("nan")


def relative_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    denominator = max(abs(float(np.median(values))), 1e-30)
    return float((np.max(values) - np.min(values)) / denominator)


def positive_scalar_residual(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2 or np.all(x[mask] == 0):
        return 0.0
    x = x[mask]
    y = y[mask]
    scale = max(float(np.dot(x, y) / np.dot(x, x)), 0.0)
    residual = np.linalg.norm(y - scale * x)
    return float(residual / max(np.linalg.norm(y), 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--backends", default="cpu")
    parser.add_argument("--repeat-layouts", type=int, default=5)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument(
        "--task1-results-root",
        type=Path,
        default=ROOT / "results" / "adaptec1",
    )
    args_cli = parser.parse_args()
    seeds = [int(v) for v in args_cli.seeds.split(",") if v]
    backends = [value.strip() for value in args_cli.backends.split(",") if value.strip()]
    if not backends or "cpu" not in backends or any(v not in {"cpu", "cuda"} for v in backends):
        raise ValueError("--backends must include cpu and may optionally include cuda")
    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    args = load_args()
    placedb = PlaceDB(args=args)
    with (ROOT / "experiments" / "task2_protocol.yaml").open() as f:
        protocol = yaml.safe_load(f)
    thresholds = protocol["metric_validity"]
    selection = protocol["inheritance_from_task1"]["macro_selection"]
    expected_fingerprint = {
        "node_count": int(selection["expected_node_count"]),
        "net_count": int(selection["expected_net_count"]),
        "macro_names_sha256": str(selection["expected_macro_names_sha256"]),
        "macro_geometry_sha256": str(selection["expected_macro_geometry_sha256"]),
        "net_topology_sha256": str(selection["expected_net_topology_sha256"]),
        "macro_area_sum": int(selection["expected_macro_area_sum"]),
        "canvas": [int(value) for value in selection["expected_canvas"]],
    }
    fingerprint = placedb_fingerprint(placedb, args.benchmark_path)
    validate_expected_fingerprint(fingerprint, expected_fingerprint)
    (output_dir / "placedb_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n"
    )

    placer = PLACER_REGISTRY["mgo"](args=args, placedb=placedb)
    records = collect_layout_bank(
        placer,
        seeds,
        args_cli.population_size,
        args_cli.task1_results_root.resolve(),
    )
    node_count = int(placedb.node_cnt)
    X = np.stack(
        [
            record["x"]
            if record["x"] is not None
            else np.full(2 * node_count, -1, dtype=np.int64)
            for record in records
        ]
    )
    np.savez_compressed(
        output_dir / "fixed_layout_bank.npz",
        X=X,
        has_genotype=np.asarray([record["x"] is not None for record in records]),
        source=np.asarray([record["source"] for record in records]),
        seed=np.asarray([record["seed"] if record["seed"] is not None else -1 for record in records]),
        index=np.asarray([record["index"] for record in records]),
        decode_success=np.asarray([record["decode_success"] for record in records]),
        hpwl=np.asarray([record["hpwl"] for record in records]),
        phenotype_sha256=np.asarray([record["phenotype_sha256"] or "" for record in records]),
    )

    metrics = {
        backend: DreamplaceRudyMetric(
            placedb, metric_config(backend, args_cli.cpu_threads, 512)
        )
        for backend in backends
    }
    metric_256 = DreamplaceRudyMetric(
        placedb, metric_config("cpu", args_cli.cpu_threads, 256)
    )
    high_degree_net = max(
        placedb.net_info,
        key=lambda name: len(placedb.net_info[name]["nodes"]),
    )
    all_nets = set(placedb.net_info)
    metric_without_high_degree = DreamplaceRudyMetric(
        scoped_placedb(placedb, all_nets - {high_degree_net}),
        metric_config("cpu", args_cli.cpu_threads, 512),
    )
    metric_only_high_degree = DreamplaceRudyMetric(
        scoped_placedb(placedb, {high_degree_net}),
        metric_config("cpu", args_cli.cpu_threads, 512),
    )

    valid_records = [record for record in records if record["decode_success"]]
    for record in valid_records:
        for backend, metric in metrics.items():
            result = metric.evaluate(record["macro_pos"])
            if not result.valid:
                raise RuntimeError(f"{backend} metric rejected valid layout")
            for field in METRIC_FIELDS:
                record[f"{backend}_{field}"] = float(getattr(result, field))
        record["cpu256_congestion_top10"] = float(
            metric_256.evaluate(record["macro_pos"]).congestion_top10
        )
        record["without_high_degree_congestion_top10"] = float(
            metric_without_high_degree.evaluate(record["macro_pos"]).congestion_top10
        )
        record["only_high_degree_congestion_top10"] = float(
            metric_only_high_degree.evaluate(record["macro_pos"]).congestion_top10
        )

    repeat_records: list[dict[str, Any]] = []
    representative_indices = np.linspace(
        0,
        max(0, len(valid_records) - 1),
        min(args_cli.repeat_layouts, len(valid_records)),
        dtype=int,
    )
    for representative_index in representative_indices:
        record = valid_records[int(representative_index)]
        for backend, metric in metrics.items():
            outputs = [
                metric.evaluate(record["macro_pos"], return_maps=True)
                for _ in range(args_cli.repeat_count)
            ]
            base = outputs[0]
            scalar_identical = all(
                all(getattr(base, field) == getattr(other, field) for field in METRIC_FIELDS)
                for other in outputs[1:]
            )
            map_fields = [
                "horizontal_demand_map",
                "vertical_demand_map",
                "horizontal_utilization_map",
                "vertical_utilization_map",
                "route_utilization_map",
            ]
            map_identical = all(
                np.array_equal(getattr(base, field), getattr(other, field))
                for other in outputs[1:]
                for field in map_fields
            )
            repeat_records.append(
                {
                    "source": record["source"],
                    "seed": record["seed"],
                    "index": record["index"],
                    "backend": backend,
                    "scalar_identical": scalar_identical,
                    "map_identical": map_identical,
                }
            )

    by_phenotype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid_records:
        by_phenotype[record["phenotype_sha256"]].append(record)
    unique_records = [rows[0] for rows in by_phenotype.values()]
    phenotype_objective_spread = max(
        (
            max(row["cpu_congestion_top10"] for row in rows)
            - min(row["cpu_congestion_top10"] for row in rows)
        )
        for rows in by_phenotype.values()
    ) if by_phenotype else float("inf")

    hpwl = np.asarray([record["hpwl"] for record in unique_records])
    congestion = np.asarray([record["cpu_congestion_top10"] for record in unique_records])
    rudy = np.asarray([record["cpu_rudy_top10"] for record in unique_records])
    congestion_256 = np.asarray([record["cpu256_congestion_top10"] for record in unique_records])
    without_high = np.asarray(
        [record["without_high_degree_congestion_top10"] for record in unique_records]
    )
    only_high = np.asarray(
        [record["only_high_degree_congestion_top10"] for record in unique_records]
    )
    F = np.column_stack([hpwl, congestion])
    nondominated = nondominated_indices(F)
    quality_cutoff = float(np.quantile(hpwl, float(thresholds["high_quality_hpwl_quantile"])))
    high_quality_mask = hpwl <= quality_cutoff
    source_counts = Counter(record["source"] for record in valid_records)

    spearman = finite_correlation(spearmanr, hpwl, congestion)
    kendall = finite_correlation(kendalltau, hpwl, congestion)
    grid_spearman = finite_correlation(spearmanr, congestion, congestion_256)
    leaveout_spearman = finite_correlation(spearmanr, congestion, without_high)
    scalar_residual = positive_scalar_residual(rudy, congestion)
    high_quality_relative_range = (
        relative_range(congestion[high_quality_mask])
        if np.count_nonzero(high_quality_mask) >= 2
        else 0.0
    )

    gates = {
        "minimum_valid_layouts": len(valid_records) >= int(thresholds["minimum_valid_layouts"]),
        "task1_initial_stratum_present": source_counts["task1_initial"] >= int(thresholds["minimum_task1_initial_layouts"]),
        "task1_saved_best_stratum_present": source_counts["task1_saved_best"] >= int(thresholds["minimum_task1_saved_best_layouts"]),
        "minimum_distinct_phenotypes": len(unique_records) >= int(thresholds["minimum_distinct_phenotypes"]),
        "phenotype_objective_consistency": phenotype_objective_spread <= float(thresholds["phenotype_objective_absolute_tolerance"]),
        "congestion_relative_range": relative_range(congestion) >= float(thresholds["minimum_congestion_relative_range"]),
        "high_quality_region_size": int(np.count_nonzero(high_quality_mask)) >= int(thresholds["minimum_high_quality_layouts"]),
        "high_quality_congestion_relative_range": high_quality_relative_range >= float(thresholds["minimum_high_quality_congestion_relative_range"]),
        "at_least_two_nondominated_distinct_phenotypes": len(nondominated) >= 2,
        "hpwl_congestion_not_nearly_monotonic": np.isfinite(spearman) and abs(spearman) < float(thresholds["maximum_absolute_hpwl_congestion_spearman"]),
        "rudy_congestion_not_near_scalar_multiple": scalar_residual >= float(thresholds["minimum_rudy_congestion_scalar_residual"]),
        "grid_resolution_ranking_stable": np.isfinite(grid_spearman) and grid_spearman >= float(thresholds["minimum_256_512_grid_spearman"]),
        "deterministic_repeat_identical_within_backend": all(
            row["scalar_identical"] and row["map_identical"] for row in repeat_records
        ),
    }

    cpu_gpu_summary: dict[str, Any] = {
        "status": "not_requested",
        "reason": "formal backend is CPU; CUDA equivalence is optional diagnostic evidence",
    }
    if "cuda" in metrics:
        cpu = np.asarray([[record[f"cpu_{field}"] for field in METRIC_FIELDS] for record in valid_records])
        cuda = np.asarray([[record[f"cuda_{field}"] for field in METRIC_FIELDS] for record in valid_records])
        absolute_error = np.abs(cpu - cuda)
        relative_error = absolute_error / np.maximum(np.maximum(np.abs(cpu), np.abs(cuda)), 1e-30)
        gates["cpu_gpu_allclose"] = bool(np.allclose(cpu, cuda, rtol=1e-5, atol=1e-6))
        cpu_gpu_summary = {
            "status": "completed",
            "max_absolute_error_all_scalars": float(np.max(absolute_error)),
            "max_relative_error_all_scalars": float(np.max(relative_error)),
        }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "adaptec1",
        "benchmark_variant": protocol["inheritance_from_task1"]["benchmark_variant"],
        "placedb_fingerprint": fingerprint,
        "layout_sources": dict(source_counts),
        "layouts_total": len(records),
        "layouts_valid": len(valid_records),
        "distinct_phenotypes": len(unique_records),
        "decode_failures": len(records) - len(valid_records),
        "decode_failure_rate": float((len(records) - len(valid_records)) / max(len(records), 1)),
        "hpwl_congestion_spearman": spearman,
        "hpwl_congestion_kendall": kendall,
        "congestion_relative_range": relative_range(congestion),
        "high_quality_hpwl_cutoff": quality_cutoff,
        "high_quality_layouts": int(np.count_nonzero(high_quality_mask)),
        "high_quality_congestion_relative_range": high_quality_relative_range,
        "nondominated_distinct_phenotype_count": int(len(nondominated)),
        "rudy_congestion_positive_scalar_fit_relative_residual": scalar_residual,
        "grid_resolution": {
            "bins": [256, 512],
            "ranking_spearman": grid_spearman,
        },
        "high_degree_net_audit": {
            "net": high_degree_net,
            "degree": len(placedb.net_info[high_degree_net]["nodes"]),
            "full_vs_leaveout_ranking_spearman": leaveout_spearman,
            "median_only_net_fraction_of_full_congestion": float(
                np.median(only_high / np.maximum(congestion, 1e-30))
            ),
            "maximum_only_net_fraction_of_full_congestion": float(
                np.max(only_high / np.maximum(congestion, 1e-30))
            ),
        },
        "phenotype_objective_max_absolute_spread": phenotype_objective_spread,
        "overflow": {
            "nonzero_layouts": int(sum(record["cpu_total_overflow"] > 0 for record in valid_records)),
            "maximum_total_overflow": float(max(record["cpu_total_overflow"] for record in valid_records)),
        },
        "cpu_gpu": cpu_gpu_summary,
        "repeatability": repeat_records,
        "gates": gates,
        "all_required_gates_pass": all(gates.values()),
    }

    csv_fields = [
        "source",
        "source_path",
        "seed",
        "index",
        "genotype_sha256",
        "phenotype_sha256",
        "decode_success",
        "hpwl",
        "overlap_rate",
        *[f"{backend}_{field}" for backend in backends for field in METRIC_FIELDS],
        "cpu256_congestion_top10",
        "without_high_degree_congestion_top10",
        "only_high_degree_congestion_top10",
        "nondominated_distinct_phenotype",
        "high_quality_region",
    ]
    nondominated_hashes = {
        unique_records[index]["phenotype_sha256"] for index in nondominated
    }
    with (output_dir / "fixed_layout_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in csv_fields}
            row["nondominated_distinct_phenotype"] = record.get("phenotype_sha256") in nondominated_hashes
            row["high_quality_region"] = bool(record["decode_success"] and record["hpwl"] <= quality_cutoff)
            writer.writerow(row)

    (output_dir / "metric_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_required_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
