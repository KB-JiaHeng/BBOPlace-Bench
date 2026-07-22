#!/usr/bin/env python3
"""Analyze formal Task 2 runs using one common hypervolume definition."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ROOT = Path(__file__).resolve().parents[1]
FORMAL_STATE = ROOT / "experiments" / "task2_runs" / "formal"
OUTPUT = ROOT / "experiments" / "task2_analysis"
CHECKPOINTS = [20, 1000, 5000, 10000]
REFERENCE_POINTS = [(1.1, 1.1), (1.05, 1.05), (1.2, 1.2)]


def valid_rows(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    return F[np.all(np.isfinite(F), axis=1) & np.all(F < 1.0e16, axis=1)]


def nondominated(F: np.ndarray) -> np.ndarray:
    F = valid_rows(F)
    if len(F) == 0:
        return F
    indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return F[np.asarray(indices, dtype=int)]


def normalize(F: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    scale = upper - lower
    if np.any(scale <= 0):
        raise RuntimeError(f"Degenerate common normalization range: {lower}, {upper}")
    return (np.asarray(F, dtype=float) - lower) / scale


def hypervolume(F: np.ndarray, lower: np.ndarray, upper: np.ndarray, ref: tuple[float, float]) -> float:
    front = nondominated(F)
    if len(front) == 0:
        return 0.0
    return float(HV(ref_point=np.asarray(ref, dtype=float))(normalize(front, lower, upper)))


def load_formal_paths() -> list[Path]:
    paths = []
    for manifest_path in sorted(FORMAL_STATE.glob("seed_*__*.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Incomplete formal manifest: {manifest_path}")
        path = Path(manifest["result_path"])
        completion = json.loads((path / "run_complete.json").read_text())
        if completion.get("status") != "complete" or completion.get("true_evaluation_count") != 10000:
            raise RuntimeError(f"Invalid formal result: {path}")
        paths.append(path)
    if len(paths) != 10:
        raise RuntimeError(f"Expected 10 formal runs, found {len(paths)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = load_formal_paths()
    run_data = []
    final_fronts = []
    for path in paths:
        completion = json.loads((path / "run_complete.json").read_text())
        final = np.load(path / "final_population.npz")
        archive = np.load(path / "offline_archive.npz")
        final_F = valid_rows(final["F"])
        archive_F = valid_rows(archive["F"])
        front = nondominated(final_F)
        if len(front) == 0:
            raise RuntimeError(f"Formal final population has no valid front: {path}")
        final_fronts.append(front)
        run_data.append(
            {
                "path": path,
                "method": completion["method"],
                "seed": int(completion["seed"]),
                "completion": completion,
                "final_F": final_F,
                "archive_F": archive_F,
            }
        )

    union = np.vstack(final_fronts)
    lower = np.min(union, axis=0)
    upper = np.max(union, axis=0)
    common_front = nondominated(union)
    normalization = {
        "source": "union_of_all_valid_final_nondominated_sets",
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "common_front_size": len(common_front),
        "reference_points": [list(point) for point in REFERENCE_POINTS],
    }
    (args.output / "hypervolume_normalization.json").write_text(
        json.dumps(normalization, indent=2)
    )
    np.savez_compressed(
        args.output / "common_final_front.npz",
        F=common_front,
        normalized_F=normalize(common_front, lower, upper),
    )

    rows = []
    convergence_rows = []
    for run in run_data:
        row = {
            "method": run["method"],
            "seed": run["seed"],
            "path": str(run["path"]),
            "final_nondominated_size": len(nondominated(run["final_F"])),
            "archive_nondominated_size": len(nondominated(run["archive_F"])),
            "decode_failures": run["completion"]["decode_failures"],
            "elapsed_seconds": run["completion"]["elapsed_seconds"],
            "valid_evaluations_per_second": run["completion"][
                "valid_evaluations_per_second"
            ],
        }
        for ref in REFERENCE_POINTS:
            label = f"hv_ref_{ref[0]:g}"
            row[f"final_{label}"] = hypervolume(
                run["final_F"], lower, upper, ref
            )
            row[f"archive_{label}"] = hypervolume(
                run["archive_F"], lower, upper, ref
            )
        rows.append(row)

        for checkpoint in CHECKPOINTS:
            snapshot_path = (
                run["path"]
                / "population_snapshots"
                / f"evaluation_{checkpoint}.npz"
            )
            if not snapshot_path.exists():
                raise RuntimeError(f"Missing formal snapshot: {snapshot_path}")
            F = np.load(snapshot_path)["F"]
            convergence_rows.append(
                {
                    "method": run["method"],
                    "seed": run["seed"],
                    "evaluation_count": checkpoint,
                    "hv_ref_1.1": hypervolume(F, lower, upper, (1.1, 1.1)),
                }
            )

    with (args.output / "final_hypervolume.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "hypervolume_convergence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(convergence_rows[0]))
        writer.writeheader()
        writer.writerows(convergence_rows)

    paired = []
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
    with (args.output / "paired_hypervolume.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    differences = np.asarray([row["nsga2_minus_moead"] for row in paired])
    summary = {
        "normalization": normalization,
        "paired_hv_difference_mean": float(np.mean(differences)),
        "paired_hv_difference_std_population": float(np.std(differences)),
        "paired_hv_difference_median": float(np.median(differences)),
        "paired_hv_difference_min": float(np.min(differences)),
        "paired_hv_difference_max": float(np.max(differences)),
        "reference_sensitivity_ordering": {},
    }
    for ref in REFERENCE_POINTS:
        key = f"final_hv_ref_{ref[0]:g}"
        nsga = np.mean([row[key] for row in rows if row["method"] == "nsga2"])
        moead = np.mean([row[key] for row in rows if row["method"] == "moead"])
        summary["reference_sensitivity_ordering"][str(ref)] = {
            "nsga2_mean": float(nsga),
            "moead_mean": float(moead),
            "winner_by_mean": "nsga2" if nsga > moead else "moead" if moead > nsga else "tie",
        }
    try:
        from scipy.stats import wilcoxon

        statistic, pvalue = wilcoxon(differences, alternative="two-sided", method="exact")
        summary["paired_wilcoxon"] = {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "interpretation_guard": (
                "With five pairs, this is auxiliary evidence only; p>0.05 is not "
                "evidence of equality."
            ),
        }
    except Exception as exc:
        summary["paired_wilcoxon"] = {"unavailable": str(exc)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
