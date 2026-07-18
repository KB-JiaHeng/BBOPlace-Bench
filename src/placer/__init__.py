"""Placement backends with optional DREAMPlace support."""

REGISTRY = {}

from .mgo_placer import MaskGuidedOptimizationPlacer
from .sp_placer import SPPlacer

REGISTRY["mgo"] = MaskGuidedOptimizationPlacer
REGISTRY["sp"] = SPPlacer

try:
    from .hpo_placer import HPOPlacer
except (ImportError, ModuleNotFoundError):
    HPOPlacer = None
else:
    REGISTRY["hpo"] = HPOPlacer
