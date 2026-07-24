"""Reproducibility fingerprints for Task 2 code, environment, and binaries."""

from __future__ import annotations

import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {
            "available": True,
            "commit": output("rev-parse", "HEAD"),
            "branch": output("rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": output("status", "--porcelain", "--untracked-files=no"),
        }
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__}


def package_versions() -> dict[str, str | None]:
    names = ["numpy", "scipy", "pymoo", "ray", "torch", "PyYAML"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def module_binary(module_name: str) -> dict[str, Any]:
    try:
        spec = util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None
    if spec is None or spec.origin is None:
        return {"module": module_name, "available": False}
    path = Path(spec.origin).resolve()
    result: dict[str, Any] = {
        "module": module_name,
        "available": True,
        "path": str(path),
    }
    if path.is_file():
        result["size"] = path.stat().st_size
        result["sha256"] = sha256_file(path)
    return result


def dreamplace_snapshot(root: Path) -> dict[str, Any]:
    manifest_path = root / "experiments" / "task2_smoke" / "dreamplace_source_manifest.json"
    manifest: dict[str, Any] = {"available": manifest_path.exists()}
    if manifest_path.exists():
        content = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "path": str(manifest_path),
                "file_sha256": sha256_file(manifest_path),
                "source_aggregate_sha256": content.get("aggregate_sha256"),
                "source_file_count": content.get("file_count"),
            }
        )
    return {
        "source_manifest": manifest,
        "configure": module_binary("dreamplace.configure"),
        "rudy_cpu": module_binary("dreamplace.ops.rudy.rudy_cpp"),
        "rudy_cuda": module_binary("dreamplace.ops.rudy.rudy_cuda"),
    }


def environment_snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    payload = {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": package_versions(),
        "git": git_state(root),
        "dreamplace": dreamplace_snapshot(root),
        # Per-run bindings such as PYTHONHASHSEED, OMP_NUM_THREADS, and
        # CUDA_VISIBLE_DEVICES are recorded in runtime metadata but excluded
        # from this stable software/binary environment fingerprint.
        "stable_environment": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
        },
    }
    payload["fingerprint"] = canonical_sha256(payload)
    return payload
