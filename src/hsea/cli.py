"""Command-line entry point for HSEA Homework 5 Tasks 1 and 2."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from src.hsea.experiments import (
    analyze_result_tree,
    parse_seeds,
    run_task1_search,
    run_task2_search,
)
from src.hsea.moead import MOEADConfig, run_moead
from src.hsea.nsga2 import NSGA2Config, run_nsga2
from src.hsea.problem import build_evaluator
from src.hsea.task1 import EAConfig, run_ea


def _common(args) -> dict:
    return dict(
        benchmark=args.benchmark,
        n_macro=args.n_macro,
        n_grid_x=args.grid,
        n_grid_y=args.grid,
        n_cpu=args.n_cpu,
        routing_bins=args.routing_bins,
        top_fraction=args.top_fraction,
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", default="adaptec1")
    parser.add_argument("--n-macro", type=int, default=1_000_000)
    parser.add_argument("--grid", type=int, default=224)
    parser.add_argument("--routing-bins", type=int, default=24)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--n-cpu", type=int, default=1)


def _load_config(cls, path: str | None, args):
    values = {}
    if path:
        values.update(json.loads(Path(path).read_text(encoding="utf-8")))
    for field in fields(cls):
        value = getattr(args, field.name, None)
        if value is not None:
            values[field.name] = value
    return cls(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one EA, NSGA-II or MOEA/D experiment")
    _add_common(run)
    run.add_argument("--algorithm", choices=["ea", "nsga2", "moead"], required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--config")
    run.add_argument("--population-size", type=int)
    run.add_argument("--max-evaluations", type=int)
    run.add_argument("--crossover")
    run.add_argument("--crossover-probability", type=float)
    run.add_argument("--mutation")
    run.add_argument("--mutation-probability", type=float)
    run.add_argument("--mutation-strength", type=int)
    run.add_argument("--max-step", type=int)
    run.add_argument("--neighborhood-size", type=int)
    run.add_argument("--neighborhood-mating-probability", type=float)
    run.add_argument("--maximum-replacements", type=int)

    task1 = sub.add_parser("search-task1", help="random parameter search for Task 1")
    _add_common(task1)
    task1.add_argument("--output", type=Path, required=True)
    task1.add_argument("--seeds", default="1,2")
    task1.add_argument("--trials", type=int, default=12)
    task1.add_argument("--max-evaluations", type=int, default=240)
    task1.add_argument("--search-seed", type=int, default=2026)

    task2 = sub.add_parser("search-task2", help="parameter search for NSGA-II and MOEA/D")
    _add_common(task2)
    task2.add_argument("--output", type=Path, required=True)
    task2.add_argument("--seeds", default="1")
    task2.add_argument("--trials", type=int, default=6)
    task2.add_argument("--max-evaluations", type=int, default=320)
    task2.add_argument("--search-seed", type=int, default=2026)

    analyze = sub.add_parser("analyze", help="aggregate saved result directories")
    analyze.add_argument("--root", type=Path, required=True)
    analyze.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "run":
        cls = {"ea": EAConfig, "nsga2": NSGA2Config, "moead": MOEADConfig}[args.algorithm]
        config = _load_config(cls, args.config, args)
        evaluator = build_evaluator(result_path=args.output / "evaluator", **_common(args))
        if args.algorithm == "ea":
            result = run_ea(evaluator, config, args.seed)
        elif args.algorithm == "nsga2":
            result = run_nsga2(evaluator, config, args.seed)
        else:
            result = run_moead(evaluator, config, args.seed)
        result.save(args.output)
        print(json.dumps({
            "algorithm": result.algorithm,
            "evaluations": result.evaluations,
            "elapsed_seconds": result.elapsed_seconds,
            "front_size": len(result.archive_f),
            "minimum_objectives": np.min(result.archive_f, axis=0).tolist(),
            "output": str(args.output),
        }, indent=2))
    elif args.command == "search-task1":
        summary, best = run_task1_search(
            output=args.output,
            common=_common(args),
            seeds=parse_seeds(args.seeds),
            trials=args.trials,
            evaluations=args.max_evaluations,
            search_seed=args.search_seed,
        )
        print(json.dumps({"best": best.__dict__, "summary": str(args.output / "summary.csv")}, indent=2))
    elif args.command == "search-task2":
        best = run_task2_search(
            output=args.output,
            common=_common(args),
            seeds=parse_seeds(args.seeds),
            trials_per_algorithm=args.trials,
            evaluations=args.max_evaluations,
            search_seed=args.search_seed,
        )
        print(json.dumps({"best": best, "summary": str(args.output / "summary.csv")}, indent=2))
    else:
        rows = analyze_result_tree(args.root, args.output)
        print(json.dumps({"runs": len(rows), "output": str(args.output or args.root / 'analysis.csv')}, indent=2))


if __name__ == "__main__":
    main()
