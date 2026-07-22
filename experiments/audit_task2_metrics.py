#!/usr/bin/env python3
"""Audit the frozen Task 2 objectives on a fixed Task 1-compatible layout bank."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr
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
from task2.benchmark_fingerprint import (
    placedb_fingerprint,
    validate_expected_fingerprint,
)
from task2.metrics import DreamplaceRudyMetric, RudyMetricConfig
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
    config: dict = {}
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


def layout_bank(placer, seeds: list[int], population_size: int) -> tuple[list[dict], np.ndarray]:
    node_count = int(placer.placedb.node_cnt)
    lower = np.zeros(2 * node_count, dtype=np.int64)
    upper = np.full(2 * node_count, int(placer.args.n_grid_x) - 1, dtype=np.int64)
    records: list[dict] = []
    genotypes: list[np.ndarray] = []
    for seed in seeds:
        set_seed(seed)
        X = sample_integer_population(population_size, lower, upper)
        for index, x in enumerate(X):
            hpwl, overlap_rate, macro_pos = placer._evaluate(x)
            valid = bool(
                macro_pos
                and len(macro_pos) == node_count
                and np.isfinite(float(hpwl))
                and float(hpwl) < 1e16
            )
            records.append(
                {
                    "seed": seed,
                    "index": index,
                    "genotype_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                    "decode_success": valid,
                    "hpwl": float(hpwl) if valid else 1e16,
                    "overlap_rate": float(overlap_rate),
                    "macro_pos": dict(macro_pos) if valid else None,
                }
            )
            genotypes.append(x.copy())
    return records, np.stack(genotypes)


def metric_config(backend: str, cpu_threads: int) -> RudyMetricConfig:
    return RudyMetricConfig(
        backend=backend,
        dtype="float32",
        deterministic=True,
        num_bins_x=512,
        num_bins_y=512,
        unit_horizontal_capacity=1.5625,
        unit_vertical_capacity=1.45,
        hotspot_fraction=0.10,
        cpu_threads=cpu_threads,
    )


def nondominated_indices(F: np.ndarray) -> np.ndarray:
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
    return np.flatnonzero(keep)


def exact_scalar_multiple(x: np.ndarray, y: np.ndarray) -> bool:
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0)
    if np.count_nonzero(mask) < 2:
        return False
    ratios = y[mask] / x[mask]
    return bool(np.all(ratios == ratios[0]) and ratios[0] > 0)


def relative_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--backends", default="cpu")
    parser.add_argument("--repeat-layouts", type=int, default=5)
    parser.add_argument("--repeat-count", type=int, default=3)
    args_cli = parser.parse_args()
    seeds = [int(v) for v in args_cli.seeds.split(",") if v]
    backends = [value.strip() for value in args_cli.backends.split(",") if value.strip()]
    if (
        not backends
        or "cpu" not in backends
        or any(value not in {"cpu", "cuda"} for value in backends)
    ):
        raise ValueError("--backends must include cpu and may optionally include cuda")
    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    args = load_args()
    placedb = PlaceDB(args=args)
    with (ROOT / "experiments" / "task2_protocol.yaml").open() as f:
        protocol = yaml.safe_load(f)
    selection = protocol["inheritance_from_task1"]["macro_selection"]
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
    fingerprint = placedb_fingerprint(placedb, args.benchmark_path)
    validate_expected_fingerprint(fingerprint, expected_fingerprint)
    (output_dir / "placedb_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n"
    )
    placer = PLACER_REGISTRY["mgo"](args=args, placedb=placedb)
    records, genotypes = layout_bank(placer, seeds, args_cli.population_size)
    np.savez_compressed(
        output_dir / "fixed_layout_bank.npz",
        X=genotypes,
        seed=np.asarray([r["seed"] for r in records]),
        index=np.asarray([r["index"] for r in records]),
        decode_success=np.asarray([r["decode_success"] for r in records]),
        hpwl=np.asarray([r["hpwl"] for r in records]),
    )

    metrics = {
        backend: DreamplaceRudyMetric(
            placedb,
            metric_config(backend, args_cli.cpu_threads),
        )
        for backend in backends
    }

    valid_records = [record for record in records if record["decode_success"]]
    for record in valid_records:
        for backend, metric in metrics.items():
            result = metric.evaluate(record["macro_pos"])
            if not result.valid:
                raise RuntimeError(f"{backend} metric rejected valid layout")
            for field in METRIC_FIELDS:
                record[f"{backend}_{field}"] = float(getattr(result, field))

    repeat_records: list[dict] = []
    representative_indices = np.linspace(
        0, max(0, len(valid_records) - 1), min(args_cli.repeat_layouts, len(valid_records)), dtype=int
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
                    "seed": record["seed"],
                    "index": record["index"],
                    "backend": backend,
                    "scalar_identical": scalar_identical,
                    "map_identical": map_identical,
                }
            )

    cpu = np.asarray(
        [[r[f"cpu_{field}"] for field in METRIC_FIELDS] for r in valid_records]
    )
    hpwl = np.asarray([r["hpwl"] for r in valid_records])
    congestion = cpu[:, METRIC_FIELDS.index("congestion_top10")]
    rudy = cpu[:, METRIC_FIELDS.index("rudy_top10")]
    F = np.column_stack([hpwl, congestion])
    nd = nondominated_indices(F)
    spearman = float(spearmanr(hpwl, congestion).statistic)

    gates = {
        "sufficient_valid_fixed_layouts": len(valid_records) >= 2,
        "congestion_nonzero_variance": bool(np.var(congestion) > 0),
        "at_least_two_valid_nondominated_fixed_layouts": len(nd) >= 2,
        "hpwl_congestion_spearman_abs_less_than_0_999": bool(abs(spearman) < 0.999),
        "rudy_congestion_not_exact_positive_scalar_multiple": not exact_scalar_multiple(rudy, congestion),
        "deterministic_repeat_identical_within_backend": all(
            r["scalar_identical"] and r["map_identical"] for r in repeat_records
        ),
    }
    cpu_gpu_summary: dict = {
        "status": "not_requested",
        "reason": "formal backend is CPU; CPU/CUDA kernel equivalence is tested separately",
    }
    if "cuda" in metrics:
        cuda = np.asarray(
            [[r[f"cuda_{field}"] for field in METRIC_FIELDS] for r in valid_records]
        )
        abs_error = np.abs(cpu - cuda)
        rel_error = relative_error(cpu, cuda)
        gates["cpu_gpu_allclose_rtol_1e_5_atol_1e_6"] = bool(
            np.allclose(cpu, cuda, rtol=1e-5, atol=1e-6)
        )
        cpu_gpu_summary = {
            "status": "completed",
            "max_absolute_error_all_scalars": float(np.max(abs_error)),
            "max_relative_error_all_scalars": float(np.max(rel_error)),
            "per_field_max_absolute_error": {
                field: float(np.max(abs_error[:, i]))
                for i, field in enumerate(METRIC_FIELDS)
            },
            "per_field_max_relative_error": {
                field: float(np.max(rel_error[:, i]))
                for i, field in enumerate(METRIC_FIELDS)
            },
        }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "adaptec1",
        "benchmark_variant": protocol["inheritance_from_task1"]["benchmark_variant"],
        "backends": backends,
        "placedb_fingerprint": fingerprint,
        "seeds": seeds,
        "layouts_total": len(records),
        "layouts_valid": len(valid_records),
        "decode_failures": len(records) - len(valid_records),
        "decode_failure_rate": float((len(records) - len(valid_records)) / len(records)),
        "node_count": int(placedb.node_cnt),
        "net_count": len(placedb.net_info),
        "objective": ["hpwl", "congestion_top10"],
        "hpwl": {
            "min": float(np.min(hpwl)),
            "max": float(np.max(hpwl)),
            "mean": float(np.mean(hpwl)),
        },
        "rudy_top10": {
            "min": float(np.min(rudy)),
            "max": float(np.max(rudy)),
            "variance": float(np.var(rudy)),
        },
        "congestion_top10": {
            "min": float(np.min(congestion)),
            "max": float(np.max(congestion)),
            "variance": float(np.var(congestion)),
        },
        "overflow": {
            "nonzero_layouts": int(np.count_nonzero(cpu[:, METRIC_FIELDS.index("total_overflow")] > 0)),
            "max_total_overflow": float(np.max(cpu[:, METRIC_FIELDS.index("total_overflow")])),
            "max_overflow": float(np.max(cpu[:, METRIC_FIELDS.index("max_overflow")])),
        },
        "hpwl_congestion_spearman": spearman,
        "nondominated_count": int(len(nd)),
        "nondominated_layouts": [
            {"seed": valid_records[i]["seed"], "index": valid_records[i]["index"]}
            for i in nd
        ],
        "rudy_congestion_exact_positive_scalar_multiple": exact_scalar_multiple(rudy, congestion),
        "cpu_gpu": cpu_gpu_summary,
        "repeatability": repeat_records,
        "gates": gates,
        "all_required_gates_pass": all(gates.values()),
    }

    csv_fields = [
        "seed", "index", "genotype_sha256", "decode_success", "hpwl", "overlap_rate",
        *[f"{backend}_{field}" for backend in backends for field in METRIC_FIELDS],
        "nondominated",
    ]
    nd_keys = {(valid_records[i]["seed"], valid_records[i]["index"]) for i in nd}
    with (output_dir / "fixed_layout_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in csv_fields}
            row["nondominated"] = (record["seed"], record["index"]) in nd_keys
            writer.writerow(row)
    (output_dir / "metric_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["all_required_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
