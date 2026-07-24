#!/usr/bin/env python3
"""Build the only valid Task 2 formal-launch decision from bound artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.analyze_task2 import moead_state_diagnostics
from experiments.run_task2 import (
    experiment_code_fingerprint,
    sha256_path,
    validate_completion,
)
from task2.reproducibility import environment_snapshot


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": sha256_path(resolved)}


def trace_has_unique_genotypes(path: Path) -> bool:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    hashes = [row["genotype_hash"] for row in rows]
    return len(hashes) == len(set(hashes))


def validate_test_report(
    path: Path,
    code_fingerprint: str,
    environment_fingerprint: str,
) -> tuple[dict[str, Any], list[Path]]:
    report = read_json(path)
    if report.get("status") != "passed":
        raise RuntimeError("test report is not passing")
    if report.get("dreamplace_kernel_required") is not True:
        raise RuntimeError("test report did not require the real DREAMPlace kernel")
    if int(report.get("skipped", -1)) != 0:
        raise RuntimeError("test report contains skipped tests")
    if report.get("code_fingerprint") != code_fingerprint:
        raise RuntimeError("test report code fingerprint mismatch")
    if report.get("environment_fingerprint") != environment_fingerprint:
        raise RuntimeError("test report environment fingerprint mismatch")
    log_path = Path(report["log_path"])
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    if not log_path.is_file() or sha256_path(log_path) != report.get("log_sha256"):
        raise RuntimeError("test log is missing or changed")
    return report, [path, log_path]


def validate_metric_audit(
    path: Path,
    code_fingerprint: str,
    environment_fingerprint: str,
) -> tuple[dict[str, Any], list[Path]]:
    summary = read_json(path)
    if summary.get("code_fingerprint") != code_fingerprint:
        raise RuntimeError("metric audit code fingerprint mismatch")
    if summary.get("environment_fingerprint") != environment_fingerprint:
        raise RuntimeError("metric audit environment fingerprint mismatch")
    if summary.get("all_required_gates_pass") is not True:
        failed = [key for key, value in summary.get("gates", {}).items() if not value]
        raise RuntimeError(f"metric audit gates failed: {failed}")
    if "hypervolume_calibration" not in summary:
        raise RuntimeError("metric audit has no fixed hypervolume calibration")
    csv_path = path.parent / "fixed_layout_metrics.csv"
    bank_path = path.parent / "fixed_layout_bank.npz"
    if not csv_path.is_file() or not bank_path.is_file():
        raise RuntimeError("metric audit data artifacts are missing")
    return summary, [path, csv_path, bank_path, path.parent / "placedb_fingerprint.json"]


def validate_smoke_summary(
    path: Path,
    code_fingerprint: str,
    environment_fingerprint: str,
    minimum_evaluations: int,
) -> tuple[dict[str, Any], list[Path]]:
    summary = read_json(path)
    if summary.get("mode") != "smoke":
        raise RuntimeError("algorithm smoke summary has the wrong mode")
    if int(summary.get("max_evals", 0)) < minimum_evaluations:
        raise RuntimeError("algorithm smoke budget is below the required minimum")
    if summary.get("code_fingerprint") != code_fingerprint:
        raise RuntimeError("algorithm smoke code fingerprint mismatch")
    if summary.get("environment_fingerprint") != environment_fingerprint:
        raise RuntimeError("algorithm smoke environment fingerprint mismatch")
    results = [Path(value) for value in summary.get("results", [])]
    coverage: set[tuple[str, int]] = set()
    bound = [path]
    for result_path in results:
        completion = read_json(result_path / "run_complete.json")
        method = str(completion["method"])
        seed = int(completion["seed"])
        coverage.add((method, seed))
        validate_completion(
            result_path,
            int(summary["max_evals"]),
            method,
            str(completion["definition_fingerprint"]),
            code_fingerprint,
            environment_fingerprint,
        )
        trace_path = result_path / "evaluation_trace.csv"
        if not trace_has_unique_genotypes(trace_path):
            raise RuntimeError(f"smoke evaluated a repeated genotype: {result_path}")
        if method == "moead":
            with (result_path / "moead_state.csv").open() as f:
                state = moead_state_diagnostics(list(csv.DictReader(f)))
            if not state["historical_ideal_nonregressing"]:
                raise RuntimeError(f"MOEA/D historical ideal regressed: {result_path}")
        names = [
            "run_complete.json",
            "initial_population_hash.json",
            "evaluation_trace.csv",
            "generation_metrics.csv",
            "placedb_fingerprint.json",
            "environment_fingerprint.json",
        ]
        if method == "moead":
            names.append("moead_state.csv")
        bound.extend(result_path / name for name in names)
    expected = {(method, seed) for method in ("nsga2", "moead") for seed in (1, 2)}
    if coverage != expected:
        raise RuntimeError(f"algorithm smoke coverage mismatch: {sorted(coverage)}")
    paired = summary.get("paired_initial_population", {})
    if set(map(int, paired.keys())) != {1, 2}:
        raise RuntimeError("algorithm smoke lacks paired initial populations for seeds 1 and 2")
    return summary, bound


def validate_performance_decision(
    path: Path,
    code_fingerprint: str,
    environment_fingerprint: str,
) -> tuple[dict[str, Any], list[Path]]:
    decision = read_json(path)
    if decision.get("status") != "selected":
        raise RuntimeError("performance decision did not select a stable scheduler")
    if decision.get("selection_rule") != "maximum_stable_valid_evaluations_per_second":
        raise RuntimeError("performance decision used an unexpected selection rule")
    if decision.get("code_fingerprint") != code_fingerprint:
        raise RuntimeError("performance decision code fingerprint mismatch")
    if decision.get("environment_fingerprint") != environment_fingerprint:
        raise RuntimeError("performance decision environment fingerprint mismatch")
    scheduler = decision.get("recommended_scheduler")
    if not isinstance(scheduler, dict) or scheduler.get("backend") != "cpu":
        raise RuntimeError("performance decision has no CPU scheduler recommendation")
    selected = decision.get("selected_candidate")
    if not isinstance(selected, dict) or selected.get("stable") is not True:
        raise RuntimeError("selected performance candidate is not stable")
    bound = [path]
    for candidate in decision.get("candidates", []):
        candidate_path = Path(candidate["path"])
        if candidate_path.is_file():
            bound.append(candidate_path)
        for result in candidate.get("results", []):
            completion_path = Path(result) / "run_complete.json"
            if completion_path.is_file():
                bound.append(completion_path)
    return decision, bound


def build_decision(
    *,
    test_report_path: Path,
    metric_summary_path: Path,
    smoke_summary_path: Path,
    performance_decision_path: Path,
    code_fingerprint: str,
    environment_fingerprint: str,
    minimum_smoke_evaluations: int = 1000,
) -> dict[str, Any]:
    gates: dict[str, bool] = {}
    failures: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    bound_paths: list[Path] = []

    validators = [
        (
            "tests",
            lambda: validate_test_report(
                test_report_path, code_fingerprint, environment_fingerprint
            ),
        ),
        (
            "metric_audit",
            lambda: validate_metric_audit(
                metric_summary_path, code_fingerprint, environment_fingerprint
            ),
        ),
        (
            "algorithm_smoke",
            lambda: validate_smoke_summary(
                smoke_summary_path,
                code_fingerprint,
                environment_fingerprint,
                minimum_smoke_evaluations,
            ),
        ),
        (
            "performance_selection",
            lambda: validate_performance_decision(
                performance_decision_path,
                code_fingerprint,
                environment_fingerprint,
            ),
        ),
    ]
    for name, validator in validators:
        try:
            payload, paths = validator()
            gates[name] = True
            evidence[name] = payload
            bound_paths.extend(paths)
        except Exception as exc:
            gates[name] = False
            failures[name] = str(exc)

    unique_paths: dict[Path, None] = {}
    for path in bound_paths:
        resolved = path.resolve()
        if resolved.is_file():
            unique_paths[resolved] = None
    allowed = all(gates.values())
    performance = evidence.get("performance_selection", {})
    metric = evidence.get("metric_audit", {})
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_launch_allowed": allowed,
        "code_fingerprint": code_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "minimum_smoke_evaluations": minimum_smoke_evaluations,
        "gates": gates,
        "failed_gates": failures,
        "recommended_scheduler": (
            performance.get("recommended_scheduler") if allowed else None
        ),
        "placedb_fingerprint": metric.get("placedb_fingerprint"),
        "hypervolume_calibration": metric.get("hypervolume_calibration"),
        "artifacts": [artifact(path) for path in sorted(unique_paths)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    smoke_root = ROOT / "experiments" / "task2_smoke"
    parser.add_argument("--test-report", type=Path, default=smoke_root / "test_report.json")
    parser.add_argument(
        "--metric-summary",
        type=Path,
        default=smoke_root / "metric_audit" / "metric_audit_summary.json",
    )
    parser.add_argument(
        "--smoke-summary",
        type=Path,
        default=ROOT / "experiments" / "task2_runs" / "smoke" / "summary.json",
    )
    parser.add_argument(
        "--performance-decision",
        type=Path,
        default=smoke_root / "performance_decision.json",
    )
    parser.add_argument("--minimum-smoke-evaluations", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=smoke_root / "smoke_decision.json")
    args = parser.parse_args()

    code = experiment_code_fingerprint()
    environment = environment_snapshot(ROOT)
    decision = build_decision(
        test_report_path=args.test_report.resolve(),
        metric_summary_path=args.metric_summary.resolve(),
        smoke_summary_path=args.smoke_summary.resolve(),
        performance_decision_path=args.performance_decision.resolve(),
        code_fingerprint=code,
        environment_fingerprint=environment["fingerprint"],
        minimum_smoke_evaluations=args.minimum_smoke_evaluations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    if not decision["formal_launch_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
