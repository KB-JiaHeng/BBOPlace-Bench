from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.build_task2_smoke_decision import (
    build_decision,
    validate_test_report,
)
from experiments.run_task2 import load_formal_recommendation
from experiments.select_task2_scheduler import evaluate_candidate
from task2.reproducibility import sha256_file


class Task2SmokeDecisionTest(unittest.TestCase):
    def test_missing_evidence_builds_a_blocked_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = build_decision(
                test_report_path=root / "tests.json",
                metric_summary_path=root / "metrics.json",
                smoke_summary_path=root / "smoke.json",
                performance_decision_path=root / "performance.json",
                code_fingerprint="code",
                environment_fingerprint="environment",
            )
        self.assertFalse(decision["formal_launch_allowed"])
        self.assertEqual(set(decision["failed_gates"]), {
            "tests",
            "metric_audit",
            "algorithm_smoke",
            "performance_selection",
        })

    def test_test_report_is_bound_to_its_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "tests.log"
            log.write_text("passed")
            report = root / "tests.json"
            report.write_text(json.dumps({
                "status": "passed",
                "dreamplace_kernel_required": True,
                "skipped": 0,
                "code_fingerprint": "code",
                "environment_fingerprint": "environment",
                "log_path": str(log),
                "log_sha256": sha256_file(log),
            }))
            validate_test_report(report, "code", "environment")
            log.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                validate_test_report(report, "code", "environment")

    def test_formal_preflight_rejects_mutated_bound_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "evidence.json"
            artifact.write_text("original")
            decision_path = root / "smoke_decision.json"
            decision_path.write_text(json.dumps({
                "formal_launch_allowed": True,
                "code_fingerprint": "code",
                "environment_fingerprint": "environment",
                "recommended_scheduler": {
                    "backend": "cpu",
                    "cpus_per_run": 4,
                    "rudy_cpu_threads": 1,
                    "concurrent_runs": 2,
                },
                "artifacts": [{
                    "path": str(artifact),
                    "sha256": sha256_file(artifact),
                }],
            }))
            with patch("experiments.run_task2.SMOKE_DECISION", decision_path):
                recommendation = load_formal_recommendation("code", "environment")
                self.assertEqual(recommendation["backend"], "cpu")
                artifact.write_text("mutated")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    load_formal_recommendation("code", "environment")

    def test_scheduler_candidate_requires_both_algorithms_and_current_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for method, speed in [("nsga2", 5.0), ("moead", 3.0)]:
                result = root / method
                result.mkdir()
                (result / "run_complete.json").write_text(json.dumps({
                    "status": "complete",
                    "method": method,
                    "code_fingerprint": "code",
                    "environment_fingerprint": "environment",
                    "valid_evaluations_per_second": speed,
                }))
                results.append(str(result))
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "mode": "performance",
                "backend": "cpu",
                "code_fingerprint": "code",
                "environment_fingerprint": "environment",
                "cpus_per_run": 4,
                "rudy_cpu_threads": 1,
                "concurrent_runs": 2,
                "reserve_cpus": 32,
                "results": results,
            }))
            candidate = evaluate_candidate(summary, "code", "environment")
        self.assertTrue(candidate["stable"])
        self.assertEqual(candidate["aggregate_valid_evaluations_per_second"], 8.0)


if __name__ == "__main__":
    unittest.main()
