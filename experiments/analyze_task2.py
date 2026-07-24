#!/usr/bin/env python3
"""Analyze Task 2 front quality and the Task 1 corner-question diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ROOT = Path(__file__).resolve().parents[1]
FORMAL_STATE = ROOT / "experiments" / "task2_runs" / "formal"
OUTPUT = ROOT / "experiments" / "task2_analysis"
DEFAULT_METRIC_AUDIT = ROOT / "experiments" / "task2_smoke" / "metric_audit"
CHECKPOINTS = [20, 1000, 5000, 10000]
REFERENCE_POINTS = [(1.1, 1.1), (1.05, 1.05), (1.2, 1.2)]


def valid_mask(F: np.ndarray) -> np.ndarray:
    values = np.asarray(F, dtype=float)
    return np.all(np.isfinite(values), axis=1) & np.all(values < 1.0e16, axis=1)


def valid_rows(F: np.ndarray) -> np.ndarray:
    values = np.asarray(F, dtype=float)
    return values[valid_mask(values)]


def deduplicate_by_phenotype(
    F: np.ndarray,
    phenotype_hashes: np.ndarray | list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(F, dtype=float)
    mask = valid_mask(values)
    values = values[mask]
    if phenotype_hashes is None:
        hashes = np.asarray([f"unknown-{index}" for index in range(len(values))])
    else:
        hashes = np.asarray(phenotype_hashes).astype(str)[mask]
    selected: list[int] = []
    first_by_hash: dict[str, int] = {}
    for index, digest in enumerate(hashes):
        key = digest if digest else f"unknown-{index}"
        if key in first_by_hash:
            incumbent = values[first_by_hash[key]]
            if not np.allclose(incumbent, values[index], rtol=0.0, atol=1.0e-9):
                raise RuntimeError(
                    f"One phenotype hash maps to inconsistent objectives: {key}"
                )
            continue
        first_by_hash[key] = index
        selected.append(index)
    indices = np.asarray(selected, dtype=int)
    return values[indices], hashes[indices]


def nondominated(F: np.ndarray) -> np.ndarray:
    values = valid_rows(F)
    if len(values) == 0:
        return values
    indices = NonDominatedSorting().do(values, only_non_dominated_front=True)
    return values[np.asarray(indices, dtype=int)]


def nondominated_phenotypes(
    F: np.ndarray,
    phenotype_hashes: np.ndarray | list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    values, hashes = deduplicate_by_phenotype(F, phenotype_hashes)
    if len(values) == 0:
        return values, hashes
    indices = np.asarray(
        NonDominatedSorting().do(values, only_non_dominated_front=True), dtype=int
    )
    return values[indices], hashes[indices]


def normalize(F: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    scale = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    if np.any(scale <= 0):
        raise RuntimeError(f"Degenerate fixed normalization range: {lower}, {upper}")
    return (np.asarray(F, dtype=float) - lower) / scale


def hypervolume(
    F: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ref: tuple[float, float],
) -> float:
    front = nondominated(F)
    if len(front) == 0:
        return 0.0
    return float(HV(ref_point=np.asarray(ref, dtype=float))(normalize(front, lower, upper)))


def load_population(path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path)
    F = np.asarray(archive["F"], dtype=float)
    hashes = (
        np.asarray(archive["phenotype_hash"]).astype(str)
        if "phenotype_hash" in archive.files
        else np.asarray(["" for _ in range(len(F))])
    )
    return F, hashes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def optional_floats(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        for column in columns:
            value = row.get(column, "")
            if value not in {"", None}:
                values.append(float(value))
    return np.asarray(values, dtype=float)


def optional_ints(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    return optional_floats(rows, columns).astype(int)


def boolean_rate(rows: list[dict[str, str]], columns: list[str]) -> float | None:
    values = optional_ints(rows, columns)
    return float(np.mean(values)) if len(values) else None


def distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def relative_range(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    denominator = max(abs(float(np.median(values))), 1.0e-30)
    return float((np.max(values) - np.min(values)) / denominator)


def finite_correlation(function: Any, x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    value = float(function(x, y).statistic)
    return value if np.isfinite(value) else None


def visited_region_diagnostics(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize objective informativeness over distinct visited phenotypes."""
    by_phenotype: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        if row.get("decode_success") != "1":
            continue
        values = np.asarray(
            [float(row["hpwl"]), float(row["congestion_top10"])], dtype=float
        )
        if not valid_mask(values[None, :])[0]:
            continue
        digest = row.get("phenotype_hash") or f"unknown-{index}"
        if digest in by_phenotype:
            if not np.allclose(by_phenotype[digest], values, rtol=0.0, atol=1.0e-9):
                raise RuntimeError(
                    f"Visited phenotype maps to inconsistent objectives: {digest}"
                )
            continue
        by_phenotype[digest] = values

    if not by_phenotype:
        raise RuntimeError("No valid visited phenotypes")
    F = np.stack(list(by_phenotype.values()))
    hpwl = F[:, 0]
    congestion = F[:, 1]
    cutoff = float(np.quantile(hpwl, 0.25))
    high_quality = congestion[hpwl <= cutoff]
    from scipy.stats import kendalltau, spearmanr

    return {
        "distinct_visited_phenotypes": len(F),
        "nondominated_distinct_phenotypes": len(nondominated(F)),
        "hpwl_congestion_spearman": finite_correlation(spearmanr, hpwl, congestion),
        "hpwl_congestion_kendall": finite_correlation(kendalltau, hpwl, congestion),
        "congestion_relative_range": relative_range(congestion),
        "high_quality_hpwl_cutoff": cutoff,
        "high_quality_distinct_phenotypes": len(high_quality),
        "high_quality_congestion_relative_range": relative_range(high_quality),
    }


def displacement_conditioned_diagnostics(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Condition parent-child outcomes on a small set of displacement bins."""
    bins = [("0", 0, 0), ("1-10", 1, 10), ("11-100", 11, 100), (">100", 101, None)]
    observations: list[dict[str, Any]] = []
    for row in rows:
        if row.get("phase") != "evolution":
            continue
        for parent in (1, 2):
            moved = row.get(f"moved_macros_parent{parent}", "")
            if moved in {"", None}:
                continue
            observations.append(
                {
                    "moved": int(float(moved)),
                    "survived": int(row["survived"]),
                    "dominance": row.get(f"dominance_parent{parent}", ""),
                    "delta_hpwl": float(row[f"delta_hpwl_parent{parent}"]),
                    "delta_congestion": float(row[f"delta_congestion_top10_parent{parent}"]),
                }
            )

    result: list[dict[str, Any]] = []
    for label, lower, upper in bins:
        selected = [
            item
            for item in observations
            if item["moved"] >= lower and (upper is None or item["moved"] <= upper)
        ]
        if not selected:
            continue
        delta_hpwl = np.asarray([item["delta_hpwl"] for item in selected])
        delta_congestion = np.asarray([item["delta_congestion"] for item in selected])
        result.append(
            {
                "displacement_bin": label,
                "parent_child_comparisons": len(selected),
                "survival_rate": float(np.mean([item["survived"] for item in selected])),
                "dominates_parent_rate": float(np.mean([item["dominance"] == "dominates" for item in selected])),
                "dominated_by_parent_rate": float(np.mean([item["dominance"] == "dominated" for item in selected])),
                "hpwl_improvement_rate": float(np.mean(delta_hpwl < 0)),
                "congestion_improvement_rate": float(np.mean(delta_congestion < 0)),
                "mean_delta_hpwl": float(np.mean(delta_hpwl)),
                "mean_delta_congestion_top10": float(np.mean(delta_congestion)),
            }
        )
    return result


def trace_diagnostics(rows: list[dict[str, str]]) -> dict[str, Any]:
    valid_rows_trace = [row for row in rows if row.get("decode_success") == "1"]
    evolution = [row for row in rows if row.get("phase") == "evolution"]
    genotype_hashes = [row["genotype_hash"] for row in rows if row.get("genotype_hash")]
    phenotype_hashes = [
        row["phenotype_hash"]
        for row in valid_rows_trace
        if row.get("phenotype_hash")
    ]
    phenotype_redundancy = (
        1.0 - len(set(phenotype_hashes)) / len(phenotype_hashes)
        if phenotype_hashes
        else None
    )
    moved = optional_ints(
        evolution, ["moved_macros_parent1", "moved_macros_parent2"]
    )
    total_displacement = optional_ints(
        evolution,
        [
            "total_grid_displacement_parent1",
            "total_grid_displacement_parent2",
        ],
    )
    mean_displacement = optional_floats(
        evolution,
        ["mean_grid_displacement_parent1", "mean_grid_displacement_parent2"],
    )
    max_displacement = optional_ints(
        evolution,
        ["max_grid_displacement_parent1", "max_grid_displacement_parent2"],
    )
    delta_hpwl = optional_floats(
        evolution, ["delta_hpwl_parent1", "delta_hpwl_parent2"]
    )
    delta_congestion = optional_floats(
        evolution,
        [
            "delta_congestion_top10_parent1",
            "delta_congestion_top10_parent2",
        ],
    )
    replacement = optional_ints(evolution, ["replacement_count"])
    dominance = Counter(
        value
        for row in evolution
        for value in [row.get("dominance_parent1", ""), row.get("dominance_parent2", "")]
        if value
    )
    return {
        "evaluation_count": len(rows),
        "decode_failure_count": len(rows) - len(valid_rows_trace),
        "repeated_genotype_evaluations": len(genotype_hashes) - len(set(genotype_hashes)),
        "valid_phenotype_count": len(phenotype_hashes),
        "unique_valid_phenotypes": len(set(phenotype_hashes)),
        "cumulative_phenotype_redundancy": phenotype_redundancy,
        "same_phenotype_parent1_rate": boolean_rate(
            evolution, ["same_phenotype_parent1"]
        ),
        "same_phenotype_parent2_rate": boolean_rate(
            evolution, ["same_phenotype_parent2"]
        ),
        "survival_rate": boolean_rate(evolution, ["survived"]),
        "archive_entry_rate": boolean_rate(evolution, ["archive_entered"]),
        "moved_macros": distribution_summary(moved),
        "total_grid_displacement": distribution_summary(total_displacement),
        "mean_grid_displacement": distribution_summary(mean_displacement),
        "max_grid_displacement": distribution_summary(max_displacement),
        "delta_hpwl": {
            **distribution_summary(delta_hpwl),
            "improvement_rate": float(np.mean(delta_hpwl < 0)) if len(delta_hpwl) else None,
        },
        "delta_congestion_top10": {
            **distribution_summary(delta_congestion),
            "improvement_rate": (
                float(np.mean(delta_congestion < 0)) if len(delta_congestion) else None
            ),
        },
        "replacement_count": distribution_summary(replacement),
        "dominance_relations": dict(dominance),
    }


def moead_slot_occupancy_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        hashes = list(json.loads(row["slot_phenotype_hashes"]))
        counts = Counter(hashes)
        probabilities = np.asarray(list(counts.values()), dtype=float) / len(hashes)
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        normalized_entropy = entropy / np.log(len(hashes)) if len(hashes) > 1 else 0.0
        adjacent_same = (
            float(np.mean([left == right for left, right in zip(hashes, hashes[1:])]))
            if len(hashes) > 1
            else 0.0
        )
        result.append(
            {
                "evaluation_count": int(row["evaluation_count"]),
                "sweep": int(row["sweep"]),
                "slot_count": len(hashes),
                "unique_slot_phenotypes": len(counts),
                "largest_phenotype_slot_fraction": float(max(counts.values()) / len(hashes)),
                "normalized_slot_entropy": normalized_entropy,
                "adjacent_same_phenotype_fraction": adjacent_same,
            }
        )
    return result


def moead_state_diagnostics(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("MOEA/D state file is empty")
    ideals = np.asarray([json.loads(row["ideal"]) for row in rows], dtype=float)
    nonregressing = bool(np.all(np.diff(ideals, axis=0) <= 1.0e-12))
    unique = np.asarray([int(row["unique_slot_phenotypes"]) for row in rows])
    concentration = np.asarray(
        [float(row["max_single_phenotype_fraction"]) for row in rows]
    )
    occupancy = moead_slot_occupancy_rows(rows)
    return {
        "state_rows": len(rows),
        "historical_ideal_nonregressing": nonregressing,
        "minimum_unique_slot_phenotypes": int(np.min(unique)),
        "final_unique_slot_phenotypes": int(unique[-1]),
        "maximum_single_phenotype_slot_fraction": float(np.max(concentration)),
        "final_single_phenotype_slot_fraction": float(concentration[-1]),
        "minimum_normalized_slot_entropy": min(row["normalized_slot_entropy"] for row in occupancy),
        "final_normalized_slot_entropy": occupancy[-1]["normalized_slot_entropy"],
        "maximum_adjacent_same_phenotype_fraction": max(row["adjacent_same_phenotype_fraction"] for row in occupancy),
        "final_adjacent_same_phenotype_fraction": occupancy[-1]["adjacent_same_phenotype_fraction"],
        "first_half_population_collapse_sweep": next(
            (row["sweep"] for row in occupancy if row["unique_slot_phenotypes"] <= row["slot_count"] / 2),
            None,
        ),
        "initial_ideal": ideals[0].tolist(),
        "final_ideal": ideals[-1].tolist(),
    }


def load_fixed_normalization(metric_summary_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    summary = json.loads(metric_summary_path.read_text())
    if summary.get("all_required_gates_pass") is not True:
        raise RuntimeError("Metric audit did not pass all required gates")
    calibration = summary["hypervolume_calibration"]
    lower = np.asarray(calibration["lower"], dtype=float)
    upper = np.asarray(calibration["upper"], dtype=float)
    return lower, upper, calibration


def load_mixed_task1_saved_best_reference(metric_csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(metric_csv_path)
    selected: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        if row.get("source") != "task1_saved_best":
            continue
        if str(row.get("decode_success", "")).lower() not in {"1", "true"}:
            continue
        digest = row.get("phenotype_sha256") or f"unknown-{index}"
        selected.setdefault(
            digest,
            np.asarray([float(row["hpwl"]), float(row["cpu_congestion_top10"])]),
        )
    if len(selected) < 5:
        raise RuntimeError(
            f"Expected at least five Task 1 saved-best reference phenotypes, found {len(selected)}"
        )
    return np.stack(list(selected.values())), np.asarray(list(selected.keys()))


def is_dominated_by(point: np.ndarray, candidates: np.ndarray) -> bool:
    return bool(
        np.any(np.all(candidates <= point, axis=1) & np.any(candidates < point, axis=1))
    )


def bootstrap_mean_ci(values: np.ndarray, samples: int = 20000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260724)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    means = np.mean(draws, axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def load_formal_paths() -> list[Path]:
    paths: list[Path] = []
    fingerprints: set[str] = set()
    benchmark_hashes: set[tuple[str, str]] = set()
    for manifest_path in sorted(FORMAL_STATE.glob("seed_*__*.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Incomplete formal manifest: {manifest_path}")
        path = Path(manifest["result_path"])
        completion = json.loads((path / "run_complete.json").read_text())
        if completion.get("status") != "complete" or completion.get("true_evaluation_count") != 10000:
            raise RuntimeError(f"Invalid formal result: {path}")
        fingerprints.add(str(completion.get("definition_fingerprint")))
        benchmark_hashes.add(
            (
                str(completion.get("placedb_macro_names_sha256")),
                str(completion.get("placedb_net_topology_sha256")),
            )
        )
        paths.append(path)
    if len(paths) != 10:
        raise RuntimeError(f"Expected 10 formal runs, found {len(paths)}")
    if len(fingerprints) != 1 or "None" in fingerprints:
        raise RuntimeError(f"Formal definition fingerprints differ: {fingerprints}")
    if len(benchmark_hashes) != 1:
        raise RuntimeError(f"Formal benchmark fingerprints differ: {benchmark_hashes}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--metric-audit-summary",
        type=Path,
        default=DEFAULT_METRIC_AUDIT / "metric_audit_summary.json",
    )
    parser.add_argument(
        "--metric-audit-csv",
        type=Path,
        default=DEFAULT_METRIC_AUDIT / "fixed_layout_metrics.csv",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    lower, upper, calibration = load_fixed_normalization(args.metric_audit_summary)
    task1_reference_F, task1_reference_hashes = load_mixed_task1_saved_best_reference(
        args.metric_audit_csv
    )
    paths = load_formal_paths()
    run_data: list[dict[str, Any]] = []
    common_final_fronts: list[np.ndarray] = []
    diagnostic_rows: list[dict[str, Any]] = []
    visited_region_rows: list[dict[str, Any]] = []
    displacement_conditioned_rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    moead_rows: list[dict[str, Any]] = []
    moead_occupancy_rows: list[dict[str, Any]] = []

    for path in paths:
        completion = json.loads((path / "run_complete.json").read_text())
        final_F, final_hashes = load_population(path / "final_population.npz")
        archive_F, archive_hashes = load_population(path / "offline_archive.npz")
        final_front, final_front_hashes = nondominated_phenotypes(final_F, final_hashes)
        archive_front, archive_front_hashes = nondominated_phenotypes(
            archive_F, archive_hashes
        )
        if len(final_front) == 0:
            raise RuntimeError(f"Formal final population has no valid front: {path}")
        common_final_fronts.append(final_front)
        trace = read_csv(path / "evaluation_trace.csv")
        diagnostics = trace_diagnostics(trace)
        if diagnostics["repeated_genotype_evaluations"] != 0:
            raise RuntimeError(f"Repeated genotype was evaluated in {path}")
        method = completion["method"]
        seed = int(completion["seed"])
        diagnostic_rows.append(
            {
                "method": method,
                "seed": seed,
                "path": str(path),
                **diagnostics,
            }
        )
        visited_region_rows.append(
            {"method": method, "seed": seed, **visited_region_diagnostics(trace)}
        )
        for conditioned in displacement_conditioned_diagnostics(trace):
            displacement_conditioned_rows.append(
                {"method": method, "seed": seed, **conditioned}
            )
        for row in read_csv(path / "generation_metrics.csv"):
            diversity_rows.append({"method": method, "seed": seed, **row})
        if method == "moead":
            state_rows = read_csv(path / "moead_state.csv")
            state = moead_state_diagnostics(state_rows)
            if not state["historical_ideal_nonregressing"]:
                raise RuntimeError(f"MOEA/D ideal regressed in {path}")
            moead_rows.append({"method": method, "seed": seed, "path": str(path), **state})
            for occupancy in moead_slot_occupancy_rows(state_rows):
                moead_occupancy_rows.append(
                    {"method": method, "seed": seed, **occupancy}
                )
        run_data.append(
            {
                "path": path,
                "method": method,
                "seed": seed,
                "completion": completion,
                "final_front": final_front,
                "final_front_hashes": final_front_hashes,
                "archive_front": archive_front,
                "archive_front_hashes": archive_front_hashes,
            }
        )

    common_front = nondominated(np.vstack(common_final_fronts))
    normalization = {
        **calibration,
        "formal_results_define_bounds": False,
        "reference_points": [list(point) for point in REFERENCE_POINTS],
        "formal_common_front_size": len(common_front),
    }
    (args.output / "hypervolume_normalization.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.output / "common_final_front.npz",
        F=common_front,
        normalized_F=normalize(common_front, lower, upper),
    )

    rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    for run in run_data:
        row: dict[str, Any] = {
            "method": run["method"],
            "seed": run["seed"],
            "path": str(run["path"]),
            "final_nondominated_distinct_phenotypes": len(run["final_front"]),
            "archive_nondominated_distinct_phenotypes": len(run["archive_front"]),
            "decode_failures": run["completion"]["decode_failures"],
            "elapsed_seconds": run["completion"]["elapsed_seconds"],
            "valid_evaluations_per_second": run["completion"]["valid_evaluations_per_second"],
        }
        for ref in REFERENCE_POINTS:
            label = f"hv_ref_{ref[0]:g}"
            row[f"final_{label}"] = hypervolume(
                run["final_front"], lower, upper, ref
            )
            row[f"archive_{label}"] = hypervolume(
                run["archive_front"], lower, upper, ref
            )
        rows.append(row)

        for checkpoint in CHECKPOINTS:
            snapshot_path = run["path"] / "population_snapshots" / f"evaluation_{checkpoint}.npz"
            if not snapshot_path.exists():
                raise RuntimeError(f"Missing formal snapshot: {snapshot_path}")
            snapshot_F, snapshot_hashes = load_population(snapshot_path)
            front, _ = nondominated_phenotypes(snapshot_F, snapshot_hashes)
            convergence_rows.append(
                {
                    "method": run["method"],
                    "seed": run["seed"],
                    "evaluation_count": checkpoint,
                    "distinct_nondominated_phenotypes": len(front),
                    "hv_ref_1.1": hypervolume(front, lower, upper, (1.1, 1.1)),
                }
            )

    write_rows(args.output / "final_hypervolume.csv", rows)
    write_rows(args.output / "hypervolume_convergence.csv", convergence_rows)
    write_rows(args.output / "run_diagnostics.csv", diagnostic_rows)
    write_rows(args.output / "visited_region_metric_validity.csv", visited_region_rows)
    write_rows(
        args.output / "operator_effect_by_displacement.csv",
        displacement_conditioned_rows,
    )
    write_rows(args.output / "phenotype_diversity_over_time.csv", diversity_rows)
    write_rows(args.output / "moead_collapse_diagnostics.csv", moead_rows)
    write_rows(args.output / "moead_slot_occupancy_over_time.csv", moead_occupancy_rows)

    paired: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in rows}):
        by_method = {row["method"]: row for row in rows if row["seed"] == seed}
        if set(by_method) != {"nsga2", "moead"}:
            raise RuntimeError(f"Unpaired seed {seed}: {set(by_method)}")
        paired.append(
            {
                "seed": seed,
                "nsga2_hv": by_method["nsga2"]["final_hv_ref_1.1"],
                "moead_hv": by_method["moead"]["final_hv_ref_1.1"],
                "nsga2_minus_moead": (
                    by_method["nsga2"]["final_hv_ref_1.1"]
                    - by_method["moead"]["final_hv_ref_1.1"]
                ),
            }
        )
    write_rows(args.output / "paired_hypervolume.csv", paired)

    task1_front, _ = nondominated_phenotypes(task1_reference_F, task1_reference_hashes)
    task2_front = common_front
    combined = np.vstack([task1_front, task2_front])
    combined_indices = np.asarray(
        NonDominatedSorting().do(combined, only_non_dominated_front=True), dtype=int
    )
    task1_contributions = int(np.count_nonzero(combined_indices < len(task1_front)))
    task2_contributions = int(np.count_nonzero(combined_indices >= len(task1_front)))
    reference_coverage = {
        "reference_scope": "Mixed Task 1 saved-best placements across available runs; not intermediate or full final populations and not an operator-specific reference",
        "task1_distinct_reference_phenotypes": len(task1_reference_F),
        "task1_nondominated_reference_phenotypes": len(task1_front),
        "task1_reference_hv_ref_1.1": hypervolume(
            task1_front, lower, upper, (1.1, 1.1)
        ),
        "task2_common_front_hv_ref_1.1": hypervolume(
            task2_front, lower, upper, (1.1, 1.1)
        ),
        "task1_front_points_dominated_by_task2": int(
            sum(is_dominated_by(point, task2_front) for point in task1_front)
        ),
        "task2_front_points_dominated_by_task1": int(
            sum(is_dominated_by(point, task1_front) for point in task2_front)
        ),
        "combined_front_task1_contributions": task1_contributions,
        "combined_front_task2_contributions": task2_contributions,
    }
    (args.output / "mixed_task1_saved_best_coverage.json").write_text(
        json.dumps(reference_coverage, indent=2, sort_keys=True) + "\n"
    )

    differences = np.asarray([row["nsga2_minus_moead"] for row in paired])
    standard_deviation = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    summary: dict[str, Any] = {
        "normalization": normalization,
        "paired_hv_differences": differences.tolist(),
        "paired_hv_difference_mean": float(np.mean(differences)),
        "paired_hv_difference_mean_bootstrap_95_ci": bootstrap_mean_ci(differences),
        "paired_hv_difference_std_sample": standard_deviation,
        "paired_standardized_effect": (
            float(np.mean(differences) / standard_deviation)
            if standard_deviation > 0
            else None
        ),
        "paired_hv_difference_median": float(np.median(differences)),
        "paired_hv_difference_min": float(np.min(differences)),
        "paired_hv_difference_max": float(np.max(differences)),
        "same_direction_pairs": int(
            max(np.count_nonzero(differences > 0), np.count_nonzero(differences < 0))
        ),
        "reference_sensitivity_ordering": {},
        "mixed_task1_saved_best_coverage": reference_coverage,
    }
    for ref in REFERENCE_POINTS:
        key = f"final_hv_ref_{ref[0]:g}"
        nsga = np.mean([row[key] for row in rows if row["method"] == "nsga2"])
        moead = np.mean([row[key] for row in rows if row["method"] == "moead"])
        summary["reference_sensitivity_ordering"][str(ref)] = {
            "nsga2_mean": float(nsga),
            "moead_mean": float(moead),
            "winner_by_mean": (
                "nsga2" if nsga > moead else "moead" if moead > nsga else "tie"
            ),
        }
    try:
        from scipy.stats import wilcoxon

        statistic, pvalue = wilcoxon(
            differences, alternative="two-sided", method="exact"
        )
        summary["paired_wilcoxon"] = {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "interpretation_guard": (
                "With five pairs, exact two-sided Wilcoxon cannot reach p<0.05 "
                "even when all nonzero differences have one sign; p>0.05 is not equivalence."
            ),
        }
    except Exception as exc:
        summary["paired_wilcoxon"] = {"unavailable": str(exc)}

    claim_evidence = {
        "rq1_front_quality": [
            "final_hypervolume.csv",
            "paired_hypervolume.csv",
            "hypervolume_convergence.csv",
            "summary.json",
        ],
        "rq2_diversity_and_collapse": [
            "run_diagnostics.csv",
            "phenotype_diversity_over_time.csv",
            "moead_collapse_diagnostics.csv",
            "moead_slot_occupancy_over_time.csv",
        ],
        "rq3_second_objective_validity": [
            str(args.metric_audit_summary),
            str(args.metric_audit_csv),
            "visited_region_metric_validity.csv",
        ],
        "rq4_combined_operator_effect": [
            "run_diagnostics.csv",
            "operator_effect_by_displacement.csv",
        ],
        "rq5_mixed_task1_saved_best_coverage": ["mixed_task1_saved_best_coverage.json"],
        "limitations": [
            "single reduced adaptec1 top-512 macro instance",
            "macro-only HPWL and utilization proxy",
            "Task 1 reference contains saved best placements, not complete final populations",
            "five paired seeds provide effect estimates but weak hypothesis-test resolution",
        ],
    }
    (args.output / "claim_evidence.json").write_text(
        json.dumps(claim_evidence, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
