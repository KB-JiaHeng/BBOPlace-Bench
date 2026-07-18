"""Parameter search and baseline aggregation for Tasks 1 and 2."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from src.hsea.moead import MOEADConfig, run_moead
from src.hsea.nsga2 import NSGA2Config, run_nsga2
from src.hsea.pareto import hypervolume
from src.hsea.problem import build_evaluator
from src.hsea.results import EvolutionResult, load_result
from src.hsea.task1 import EAConfig, run_ea


def parse_seeds(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _make_evaluator(common: dict, result_path: Path):
    return build_evaluator(result_path=result_path, **common)


def task1_random_configs(trials: int, seed: int, evaluations: int) -> list[EAConfig]:
    rng = np.random.default_rng(seed)
    configs: list[EAConfig] = []
    choices = {
        "population_size": [16, 24, 32, 40],
        "crossover": ["uniform", "two_point", "sbx"],
        "crossover_probability": [0.7, 0.9, 1.0],
        "mutation": ["swap", "shift", "reset", "shuffle", "polynomial"],
        "mutation_probability": [0.4, 0.7, 0.9, 1.0],
        "mutation_strength": [1, 2, 4],
        "max_step": [2, 4, 8],
        "tournament_size": [2, 3],
    }
    # Include interpretable anchors before random combinations.
    anchors = [
        EAConfig(population_size=24, max_evaluations=evaluations, crossover="uniform", mutation="swap"),
        EAConfig(population_size=24, max_evaluations=evaluations, crossover="uniform", mutation="shift"),
        EAConfig(population_size=24, max_evaluations=evaluations, crossover="two_point", mutation="reset"),
        EAConfig(population_size=24, max_evaluations=evaluations, crossover="sbx", mutation="polynomial"),
    ]
    configs.extend(anchors[:trials])
    seen = {json.dumps(asdict(c), sort_keys=True) for c in configs}
    while len(configs) < trials:
        kwargs = {key: values[int(rng.integers(len(values)))] for key, values in choices.items()}
        cfg = EAConfig(max_evaluations=evaluations, **kwargs)
        token = json.dumps(asdict(cfg), sort_keys=True)
        if token not in seen:
            configs.append(cfg)
            seen.add(token)
    return configs


def run_task1_search(
    *,
    output: Path,
    common: dict,
    seeds: list[int],
    trials: int,
    evaluations: int,
    search_seed: int = 2026,
) -> tuple[list[dict], EAConfig]:
    output.mkdir(parents=True, exist_ok=True)
    configs = task1_random_configs(trials, search_seed, evaluations)
    run_rows: list[dict] = []
    for trial, config in enumerate(configs):
        for seed in seeds:
            run_dir = output / f"trial_{trial:03d}" / f"seed_{seed}"
            evaluator = _make_evaluator(common, run_dir / "evaluator")
            result = run_ea(evaluator, config, seed)
            result.save(run_dir)
            run_rows.append(
                {
                    "trial": trial,
                    "seed": seed,
                    "best_hpwl": float(result.archive_f[0, 0]),
                    "elapsed_seconds": result.elapsed_seconds,
                    **asdict(config),
                }
            )
    _write_rows(output / "runs.csv", run_rows)
    grouped: list[dict] = []
    for trial, config in enumerate(configs):
        values = [r["best_hpwl"] for r in run_rows if r["trial"] == trial]
        grouped.append(
            {
                "trial": trial,
                "mean_best_hpwl": float(np.mean(values)),
                "std_best_hpwl": float(np.std(values)),
                "min_best_hpwl": float(np.min(values)),
                **asdict(config),
            }
        )
    grouped.sort(key=lambda row: row["mean_best_hpwl"])
    _write_rows(output / "summary.csv", grouped)
    best = configs[int(grouped[0]["trial"])]
    (output / "best_config.json").write_text(json.dumps(asdict(best), indent=2), encoding="utf-8")
    return grouped, best


def task2_random_configs(
    algorithm: str,
    trials: int,
    seed: int,
    evaluations: int,
) -> list[NSGA2Config | MOEADConfig]:
    rng = np.random.default_rng(seed)
    configs: list[NSGA2Config | MOEADConfig] = []
    seen: set[str] = set()
    while len(configs) < trials:
        common = dict(
            population_size=int(rng.choice([24, 32, 40, 48])),
            max_evaluations=evaluations,
            crossover=str(rng.choice(["uniform", "two_point", "sbx"])),
            crossover_probability=float(rng.choice([0.8, 0.9, 1.0])),
            mutation=str(rng.choice(["shift", "swap", "reset", "polynomial"])),
            mutation_probability=float(rng.choice([0.5, 0.8, 1.0])),
            mutation_strength=int(rng.choice([1, 2, 4])),
            max_step=int(rng.choice([2, 4, 8])),
        )
        if algorithm == "nsga2":
            cfg = NSGA2Config(**common)
        elif algorithm == "moead":
            pop = common["population_size"]
            cfg = MOEADConfig(
                **common,
                neighborhood_size=int(rng.choice([5, 8, 10, min(16, pop)])),
                neighborhood_mating_probability=float(rng.choice([0.7, 0.9, 1.0])),
                maximum_replacements=int(rng.choice([1, 2, 4])),
            )
        else:
            raise ValueError(algorithm)
        token = json.dumps(asdict(cfg), sort_keys=True)
        if token not in seen:
            configs.append(cfg)
            seen.add(token)
    return configs


def run_task2_search(
    *,
    output: Path,
    common: dict,
    seeds: list[int],
    trials_per_algorithm: int,
    evaluations: int,
    search_seed: int = 2026,
) -> dict[str, dict]:
    output.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, int, int, EvolutionResult]] = []
    configs_by_algorithm: dict[str, list] = {}
    for offset, algorithm in enumerate(("nsga2", "moead")):
        configs = task2_random_configs(
            algorithm, trials_per_algorithm, search_seed + offset, evaluations
        )
        configs_by_algorithm[algorithm] = configs
        for trial, config in enumerate(configs):
            for seed in seeds:
                run_dir = output / algorithm / f"trial_{trial:03d}" / f"seed_{seed}"
                evaluator = _make_evaluator(common, run_dir / "evaluator")
                result = run_nsga2(evaluator, config, seed) if algorithm == "nsga2" else run_moead(evaluator, config, seed)
                result.save(run_dir)
                records.append((algorithm, trial, seed, result))
    all_fronts = np.concatenate([record[3].archive_f for record in records])
    lower, upper = np.min(all_fronts, axis=0), np.max(all_fronts, axis=0)
    run_rows: list[dict] = []
    for algorithm, trial, seed, result in records:
        run_rows.append(
            {
                "algorithm": algorithm,
                "trial": trial,
                "seed": seed,
                "hypervolume": hypervolume(result.archive_f, lower, upper),
                "front_size": len(result.archive_f),
                "min_hpwl": float(np.min(result.archive_f[:, 0])),
                "min_rudy": float(np.min(result.archive_f[:, 1])),
                "min_congestion": float(np.min(result.archive_f[:, 2])),
                "elapsed_seconds": result.elapsed_seconds,
                **result.config,
            }
        )
    _write_rows(output / "runs.csv", run_rows)
    best: dict[str, dict] = {}
    summary: list[dict] = []
    for algorithm, configs in configs_by_algorithm.items():
        for trial, config in enumerate(configs):
            subset = [r for r in run_rows if r["algorithm"] == algorithm and r["trial"] == trial]
            row = {
                "algorithm": algorithm,
                "trial": trial,
                "mean_hypervolume": float(np.mean([r["hypervolume"] for r in subset])),
                "std_hypervolume": float(np.std([r["hypervolume"] for r in subset])),
                "mean_front_size": float(np.mean([r["front_size"] for r in subset])),
                **asdict(config),
            }
            summary.append(row)
        candidates = [r for r in summary if r["algorithm"] == algorithm]
        chosen = max(candidates, key=lambda row: row["mean_hypervolume"])
        config = configs[int(chosen["trial"])]
        best[algorithm] = asdict(config)
        (output / f"best_{algorithm}_config.json").write_text(
            json.dumps(best[algorithm], indent=2), encoding="utf-8"
        )
    summary.sort(key=lambda row: (row["algorithm"], -row["mean_hypervolume"]))
    _write_rows(output / "summary.csv", summary)
    (output / "normalization.json").write_text(
        json.dumps({"lower": lower.tolist(), "upper": upper.tolist(), "reference": 1.1}, indent=2),
        encoding="utf-8",
    )
    return best


def analyze_result_tree(root: Path, output: Path | None = None) -> list[dict]:
    result_dirs = [p.parent for p in root.rglob("metadata.json") if (p.parent / "result.npz").exists()]
    results = [(directory, load_result(directory)) for directory in result_dirs]
    if not results:
        raise FileNotFoundError(f"no saved results found under {root}")
    multi = [result for _, result in results if result.archive_f.shape[1] > 1]
    if multi:
        all_fronts = np.concatenate([result.archive_f for result in multi])
        lower, upper = np.min(all_fronts, axis=0), np.max(all_fronts, axis=0)
    else:
        lower = upper = None
    rows: list[dict] = []
    for directory, result in results:
        row = {
            "path": str(directory),
            "algorithm": result.algorithm,
            "seed": result.seed,
            "evaluations": result.evaluations,
            "elapsed_seconds": result.elapsed_seconds,
            "front_size": len(result.archive_f),
            "min_hpwl": float(np.min(result.archive_f[:, 0])),
        }
        if result.archive_f.shape[1] > 1:
            row.update(
                hypervolume=hypervolume(result.archive_f, lower, upper),
                min_rudy=float(np.min(result.archive_f[:, 1])),
                min_congestion=float(np.min(result.archive_f[:, 2])),
            )
        rows.append(row)
    output = output or root / "analysis.csv"
    _write_rows(output, rows)
    return rows
