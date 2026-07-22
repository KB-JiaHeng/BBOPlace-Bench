#!/usr/bin/env python3
"""Create or verify a stable manifest for the unmodified DREAMPlace source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXCLUDED_PARTS = {".git", "build", "install", "__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a"}


def source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        yield relative, path


def manifest(root: Path) -> dict:
    entries = []
    aggregate = hashlib.sha256()
    for relative, path in source_files(root):
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        relative_text = relative.as_posix()
        entries.append(
            {"path": relative_text, "size": size, "sha256": content_hash}
        )
        aggregate.update(relative_text.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(content_hash.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema_version": 1,
        "source_root_name": root.name,
        "file_count": len(entries),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    current = manifest(root)
    if args.verify:
        expected = json.loads(args.verify.read_text())
        if current != expected:
            expected_by_path = {entry["path"]: entry for entry in expected["files"]}
            current_by_path = {entry["path"]: entry for entry in current["files"]}
            changed = sorted(
                path
                for path in expected_by_path.keys() | current_by_path.keys()
                if expected_by_path.get(path) != current_by_path.get(path)
            )
            print(
                json.dumps(
                    {
                        "status": "mismatch",
                        "expected_aggregate": expected.get("aggregate_sha256"),
                        "current_aggregate": current.get("aggregate_sha256"),
                        "changed_paths": changed,
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "aggregate_sha256": current["aggregate_sha256"],
                    "file_count": current["file_count"],
                },
                indent=2,
            )
        )
        return
    text = json.dumps(current, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
