"""Serializable result containers and common output helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass
class EvolutionResult:
    algorithm: str
    seed: int
    config: dict[str, Any]
    population: Array
    objectives: Array
    archive_x: Array
    archive_f: Array
    history: list[dict[str, float]]
    elapsed_seconds: float
    evaluations: int

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "result.npz",
            population=self.population,
            objectives=self.objectives,
            archive_x=self.archive_x,
            archive_f=self.archive_f,
        )
        metadata = {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "config": self.config,
            "elapsed_seconds": self.elapsed_seconds,
            "evaluations": self.evaluations,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if self.history:
            keys = list(self.history[0].keys())
            with (directory / "history.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
        return directory


def load_result(directory: str | Path) -> EvolutionResult:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays = np.load(directory / "result.npz")
    history: list[dict[str, float]] = []
    history_file = directory / "history.csv"
    if history_file.exists():
        with history_file.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                history.append({k: float(v) for k, v in row.items()})
    return EvolutionResult(
        algorithm=metadata["algorithm"],
        seed=int(metadata["seed"]),
        config=metadata["config"],
        population=arrays["population"],
        objectives=arrays["objectives"],
        archive_x=arrays["archive_x"],
        archive_f=arrays["archive_f"],
        history=history,
        elapsed_seconds=float(metadata["elapsed_seconds"]),
        evaluations=int(metadata["evaluations"]),
    )
