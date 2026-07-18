"""Generate CSV, Markdown and figures for the Task 1/2 baseline experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.hsea.pareto import hypervolume
from src.hsea.results import load_result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(experiment_root: Path) -> dict:
    formal = experiment_root / "formal"
    report_dir = experiment_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    result_dirs = sorted(
        p.parent for p in formal.rglob("metadata.json") if (p.parent / "result.npz").exists()
    )
    loaded = [(directory, load_result(directory)) for directory in result_dirs]
    if not loaded:
        raise FileNotFoundError(f"no formal runs found under {formal}")

    multi_results = [r for _, r in loaded if r.archive_f.shape[1] == 3]
    all_multi = np.concatenate([r.archive_f for r in multi_results])
    lower = np.min(all_multi, axis=0)
    upper = np.max(all_multi, axis=0)

    run_rows: list[dict] = []
    for directory, result in loaded:
        row = {
            "algorithm": result.algorithm,
            "seed": result.seed,
            "evaluations": result.evaluations,
            "elapsed_seconds": result.elapsed_seconds,
            "front_size": len(result.archive_f),
            "min_hpwl": float(np.min(result.archive_f[:, 0])),
            "path": str(directory),
        }
        if result.archive_f.shape[1] == 3:
            row.update(
                hypervolume=hypervolume(result.archive_f, lower, upper),
                min_rudy=float(np.min(result.archive_f[:, 1])),
                min_congestion=float(np.min(result.archive_f[:, 2])),
            )
        run_rows.append(row)
    _write_csv(report_dir / "runs.csv", run_rows)

    aggregate: list[dict] = []
    for algorithm in sorted({row["algorithm"] for row in run_rows}):
        rows = [row for row in run_rows if row["algorithm"] == algorithm]
        entry = {
            "algorithm": algorithm,
            "runs": len(rows),
            "mean_min_hpwl": float(np.mean([r["min_hpwl"] for r in rows])),
            "std_min_hpwl": float(np.std([r["min_hpwl"] for r in rows])),
            "mean_elapsed_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])),
        }
        if algorithm != "ea":
            entry.update(
                mean_hypervolume=float(np.mean([r["hypervolume"] for r in rows])),
                std_hypervolume=float(np.std([r["hypervolume"] for r in rows])),
                mean_front_size=float(np.mean([r["front_size"] for r in rows])),
                mean_min_rudy=float(np.mean([r["min_rudy"] for r in rows])),
                mean_min_congestion=float(np.mean([r["min_congestion"] for r in rows])),
            )
        aggregate.append(entry)
    _write_csv(report_dir / "summary.csv", aggregate)

    # Task 1 convergence curves.
    plt.figure(figsize=(7, 4.5))
    for _, result in loaded:
        if result.algorithm != "ea":
            continue
        x = [row["evaluations"] for row in result.history]
        y = [row["best_hpwl"] for row in result.history]
        plt.plot(x, y, marker="o", label=f"seed {result.seed}")
    plt.xlabel("Function evaluations")
    plt.ylabel("Best-so-far HPWL")
    plt.title("Task 1 EA convergence on adaptec1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "task1_convergence.png", dpi=180)
    plt.close()

    # Pairwise projections of the 3-objective fronts.
    for objective_index, objective_name in ((1, "RUDY"), (2, "Congestion proxy")):
        plt.figure(figsize=(7, 4.5))
        for _, result in loaded:
            if result.algorithm == "ea":
                continue
            plt.scatter(
                result.archive_f[:, 0],
                result.archive_f[:, objective_index],
                s=22,
                alpha=0.75,
                label=f"{result.algorithm} seed {result.seed}",
            )
        plt.xlabel("HPWL")
        plt.ylabel(objective_name)
        plt.title(f"Task 2 Pareto projection: HPWL vs {objective_name}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        filename = "task2_pareto_hpwl_rudy.png" if objective_index == 1 else "task2_pareto_hpwl_congestion.png"
        plt.savefig(report_dir / filename, dpi=180)
        plt.close()

    best_task1 = json.loads((experiment_root / "search_task1" / "best_config.json").read_text())
    best_nsga2 = json.loads((experiment_root / "search_task2" / "best_nsga2_config.json").read_text())
    best_moead = json.loads((experiment_root / "search_task2" / "best_moead_config.json").read_text())
    normalization = {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "reference": [1.1, 1.1, 1.1],
    }
    (report_dir / "hv_normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")

    lines = [
        "# HSEA Homework 5 — Task 1 and Task 2 Baseline",
        "",
        "## Experimental protocol",
        "",
        "- Benchmark: `adaptec1` (543 movable macros in the full run).",
        "- Representation: MGO coordinate genotype with the BBOPlace WireMask legalizer.",
        "- Full-run grid: 224 × 224.",
        "- Parameter-search budget: 64 evaluations on a 128-macro proxy problem.",
        "- Formal budget: 72 function evaluations per run on the complete macro set.",
        "- Random seeds: 1, 2, 3.",
        "- Task 2 objectives: HPWL, top-bin RUDY, and blockage-adjusted congestion proxy.",
        "- Hypervolume: all Task 2 fronts normalized using the common minima/maxima in `hv_normalization.json`; reference point 1.1 in every dimension.",
        "",
        "## Selected configurations",
        "",
        "```json",
        json.dumps({"ea": best_task1, "nsga2": best_nsga2, "moead": best_moead}, indent=2),
        "```",
        "",
        "## Aggregate results",
        "",
        "| Algorithm | Runs | Mean min HPWL | Std HPWL | Mean HV | Std HV | Mean front size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {algorithm} | {runs} | {mean_min_hpwl:.6g} | {std_min_hpwl:.6g} | {hv} | {hv_std} | {front} |".format(
                **row,
                hv=f"{row['mean_hypervolume']:.6g}" if "mean_hypervolume" in row else "—",
                hv_std=f"{row['std_hypervolume']:.6g}" if "std_hypervolume" in row else "—",
                front=f"{row['mean_front_size']:.3g}" if "mean_front_size" in row else "—",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation for Task 3",
            "",
            "These results are deliberately baseline results. Task 3 should keep the same benchmark, decoder, objective definitions, evaluation budget, seeds, and HV normalization procedure, and change only the proposed improvement. Candidate improvements can then be compared against the saved EA, NSGA-II and MOEA/D fronts without changing the experimental protocol.",
            "",
            "The congestion objective is explicitly a proxy: Bookshelf ISPD2005 data does not expose detailed routing-layer capacities. It adjusts RUDY by macro blockage and reports the hottest-bin mean; it must not be described as post-routing overflow.",
        ]
    )
    (report_dir / "baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"runs": run_rows, "summary": aggregate, "normalization": normalization}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = generate_report(args.root)
    print(json.dumps({"runs": len(result["runs"]), "report": str(args.root / "report")}, indent=2))


if __name__ == "__main__":
    main()
