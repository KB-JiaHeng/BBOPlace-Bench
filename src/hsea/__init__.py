"""HSEA homework implementation for BBOPlace-Bench.

The package contains reproducible single- and multi-objective evolutionary
baselines used for Homework 5 Tasks 1 and 2.
"""

from .problem import PlacementEvaluator, build_evaluator

__all__ = ["PlacementEvaluator", "build_evaluator"]
