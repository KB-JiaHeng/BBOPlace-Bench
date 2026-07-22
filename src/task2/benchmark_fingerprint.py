"""Canonical fingerprints for the exact Task 2 benchmark subproblem."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_net_records(placedb: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    net_names = sorted(
        placedb.net_info,
        key=lambda name: (
            int(placedb.net_info[name].get("id", 0)),
            str(name),
        ),
    )
    for net_name in net_names:
        net = placedb.net_info[net_name]
        nodes = [
            [
                node_name,
                float(net["nodes"][node_name]["x_offset"]),
                float(net["nodes"][node_name]["y_offset"]),
            ]
            for node_name in sorted(net["nodes"])
        ]
        records.append(
            {
                "name": str(net_name),
                "weight": float(net.get("weight", 1.0)),
                "nodes": nodes,
            }
        )
    return records


def placedb_fingerprint(placedb: Any, benchmark_path: str | Path | None = None) -> dict[str, Any]:
    """Describe the actual node/net instance seen by MGO, HPWL, and RUDY."""
    macro_names = [str(name) for name in placedb.macro_lst]
    macro_blob = ("\n".join(macro_names) + "\n").encode("utf-8")
    macro_geometry_blob = json.dumps(
        [
            {
                "name": name,
                "size_x": int(placedb.node_info[name]["size_x"]),
                "size_y": int(placedb.node_info[name]["size_y"]),
                "area": int(placedb.node_info[name]["area"]),
            }
            for name in placedb.macro_lst
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    net_blob = json.dumps(
        _canonical_net_records(placedb),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    fingerprint: dict[str, Any] = {
        "schema_version": 2,
        "node_count": int(placedb.node_cnt),
        "net_count": int(placedb.net_cnt),
        "macro_names_sha256": _sha256_bytes(macro_blob),
        "macro_geometry_sha256": _sha256_bytes(macro_geometry_blob),
        "net_topology_sha256": _sha256_bytes(net_blob),
        "macro_area_sum": int(placedb.macro_area_sum),
        "canvas": [
            int(placedb.canvas_lx),
            int(placedb.canvas_ly),
            int(placedb.canvas_ux),
            int(placedb.canvas_uy),
        ],
    }

    if benchmark_path is not None:
        root = Path(benchmark_path)
        fingerprint["benchmark_file_sha256"] = {
            path.name: _sha256_bytes(path.read_bytes())
            for path in sorted(root.iterdir())
            if path.is_file()
        }
    return fingerprint


def validate_expected_fingerprint(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Raise on any expected field mismatch; extra actual metadata is allowed."""
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Task 2 benchmark fingerprint mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
