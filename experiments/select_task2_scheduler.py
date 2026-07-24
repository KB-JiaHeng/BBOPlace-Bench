#!/usr/bin/env python3
"""Select the fastest stable CPU scheduler from completed performance summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.run_task2 import experiment_code_fingerprint
from task2.reproducibility import environment_snapshot


def evaluate_candidate(
    path: Path,
    expected_code: str,
    expected_environment: str,
    expected_evaluations: int,
) -> dict:
    summary = json.loads(path.read_text())
    reasons: list[str] = []
    if summary.get("mode") != "performance":
        reasons.append("not_performance_mode")
    if summary.get("backend") != "cpu":
        reasons.append("formal_backend_must_be_cpu")
    if summary.get("code_fingerprint") != expected_code:
        reasons.append("code_fingerprint_mismatch")
    if summary.get("environment_fingerprint") != expected_environment:
        reasons.append("environment_fingerprint_mismatch")
    if int(summary.get("max_evals", -1)) != expected_evaluations:
        reasons.append("evaluation_budget_mismatch")
    results = [Path(value) for value in summary.get("results", [])]
    methods: set[str] = set()
    throughputs: list[float] = []
    for result_path in results:
        completion_path = result_path / "run_complete.json"
        if not completion_path.is_file():
            reasons.append(f"missing_completion:{result_path}")
            continue
        completion = json.loads(completion_path.read_text())
        methods.add(str(completion.get("method")))
        if completion.get("status") != "complete":
            reasons.append(f"incomplete:{result_path}")
        if completion.get("code_fingerprint") != expected_code:
            reasons.append(f"result_code_mismatch:{result_path}")
        if completion.get("environment_fingerprint") != expected_environment:
            reasons.append(f"result_environment_mismatch:{result_path}")
        value = completion.get("valid_evaluations_per_second")
        if value is None or float(value) <= 0:
            reasons.append(f"invalid_throughput:{result_path}")
        else:
            throughputs.append(float(value))
    if methods != {"nsga2", "moead"}:
        reasons.append(f"method_coverage:{sorted(methods)}")
    score = float(sum(throughputs)) if not reasons else 0.0
    return {
        "path": str(path),
        "stable": not reasons,
        "reasons": reasons,
        "aggregate_valid_evaluations_per_second": score,
        "backend": summary.get("backend"),
        "max_evals": summary.get("max_evals"),
        "cpus_per_run": summary.get("cpus_per_run"),
        "rudy_cpu_threads": summary.get("rudy_cpu_threads"),
        "concurrent_runs": summary.get("concurrent_runs"),
        "reserve_cpus": summary.get("reserve_cpus"),
        "gpu_indices": summary.get("gpu_indices", []),
        "results": [str(value) for value in results],
    }


def candidate_key(candidate: dict) -> tuple[int, int, int]:
    return (
        int(candidate["cpus_per_run"]),
        int(candidate["rudy_cpu_threads"]),
        int(candidate["concurrent_runs"]),
    )


def validate_candidate_coverage(
    candidates: list[dict],
    expected_matrix: list[dict],
) -> tuple[bool, list[str]]:
    expected = {candidate_key(candidate) for candidate in expected_matrix}
    actual = {candidate_key(candidate) for candidate in candidates}
    reasons: list[str] = []
    if actual != expected:
        reasons.append(f"candidate_matrix_mismatch:actual={sorted(actual)} expected={sorted(expected)}")
    if len(candidates) != len(actual):
        reasons.append("duplicate_candidate_configuration")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "task2_smoke" / "performance_decision.json",
    )
    args = parser.parse_args()
    with (ROOT / "experiments" / "task2_protocol.yaml").open() as f:
        protocol = yaml.safe_load(f)
    performance = protocol["smoke_gates"]["performance"]
    expected_evaluations = int(performance["evaluations_per_candidate"])
    expected_matrix = list(performance["candidate_matrix"])
    code = experiment_code_fingerprint()
    environment = environment_snapshot(ROOT)
    candidates = [
        evaluate_candidate(
            path.resolve(), code, environment["fingerprint"], expected_evaluations
        )
        for path in args.candidates
    ]
    coverage_complete, coverage_failures = validate_candidate_coverage(
        candidates, expected_matrix
    )
    stable = (
        [candidate for candidate in candidates if candidate["stable"]]
        if coverage_complete
        else []
    )
    selected = max(
        stable,
        key=lambda candidate: candidate["aggregate_valid_evaluations_per_second"],
        default=None,
    )
    recommendation = None
    if selected is not None:
        recommendation = {
            "backend": "cpu",
            "cpus_per_run": int(selected["cpus_per_run"]),
            "rudy_cpu_threads": int(selected["rudy_cpu_threads"]),
            "concurrent_runs": int(selected["concurrent_runs"]),
            "reserve_cpus": int(selected["reserve_cpus"]),
            "nice_level": 10,
            "gpu_indices": [],
        }
    decision = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "selected" if recommendation else "blocked",
        "selection_rule": "maximum_stable_valid_evaluations_per_second",
        "expected_evaluations_per_candidate": expected_evaluations,
        "expected_candidate_matrix": expected_matrix,
        "candidate_coverage_complete": coverage_complete,
        "candidate_coverage_failures": coverage_failures,
        "code_fingerprint": code,
        "environment_fingerprint": environment["fingerprint"],
        "candidates": candidates,
        "selected_candidate": selected,
        "recommended_scheduler": recommendation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    if recommendation is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
