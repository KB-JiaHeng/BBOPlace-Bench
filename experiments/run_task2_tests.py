#!/usr/bin/env python3
"""Run the exact Task 2 test gate and emit a fingerprinted JSON report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.run_task2 import experiment_code_fingerprint
from task2.reproducibility import environment_snapshot, sha256_file


def count(pattern: str, text: str) -> int:
    matches = re.findall(pattern, text)
    return sum(int(value) for value in matches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "task2_smoke" / "test_report.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")

    env = os.environ.copy()
    env.update(
        {
            "TASK2_REQUIRE_DREAMPLACE": "1",
            "TASK2_RUDY_BACKEND": "cpu",
        }
    )
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout)
    passed = count(r"(\d+) passed", completed.stdout)
    skipped = count(r"(\d+) skipped", completed.stdout)
    failed = count(r"(\d+) failed", completed.stdout)
    errors = count(r"(\d+) error(?:s)?", completed.stdout)
    environment = environment_snapshot(ROOT)
    status = (
        "passed"
        if completed.returncode == 0 and passed > 0 and skipped == 0
        else "failed"
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "command": command,
        "returncode": completed.returncode,
        "dreamplace_kernel_required": True,
        "rudy_backend": "cpu",
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "code_fingerprint": experiment_code_fingerprint(),
        "environment_fingerprint": environment["fingerprint"],
        "environment": environment,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
