"""Stable hashes for Task 2 genotypes and decoded macro placements."""

from __future__ import annotations

import hashlib

import numpy as np


def genotype_hash(x: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(x, dtype=np.int64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def phenotype_hash(grid_xy: np.ndarray | None) -> str | None:
    if grid_xy is None:
        return None
    values = np.ascontiguousarray(np.asarray(grid_xy, dtype=np.int32))
    return hashlib.sha256(values.tobytes()).hexdigest()
