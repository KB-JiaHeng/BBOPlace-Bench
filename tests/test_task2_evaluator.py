from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from task2.evaluator import Task2ObjectiveEvaluator


class Task2EvaluatorFailurePolicyTest(unittest.TestCase):
    def make_evaluator(self):
        evaluator = Task2ObjectiveEvaluator.__new__(Task2ObjectiveEvaluator)
        evaluator.args = SimpleNamespace(n_cpu_max=2)
        evaluator.placer = SimpleNamespace(
            _evaluate=Mock(return_value=(1.0, 0.0, {"m": (0.0, 0.0)}))
        )
        return evaluator

    def test_uninitialized_ray_uses_explicit_local_path(self):
        evaluator = self.make_evaluator()
        X = np.zeros((2, 2), dtype=np.int64)
        with patch("ray.is_initialized", return_value=False):
            results, _ = evaluator._decode_many(X, parallel=True)
        self.assertEqual(len(results), 2)
        self.assertEqual(evaluator.placer._evaluate.call_count, 2)

    def test_active_ray_failure_is_not_masked_by_local_fallback(self):
        evaluator = self.make_evaluator()
        X = np.zeros((2, 2), dtype=np.int64)
        with (
            patch("ray.is_initialized", return_value=True),
            patch("ray.get", side_effect=RuntimeError("worker failed")),
            patch("placer.basic_placer.evaluate_placer") as remote_evaluator,
        ):
            remote_evaluator.remote.side_effect = ["future-1", "future-2"]
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                evaluator._decode_many(X, parallel=True)
        evaluator.placer._evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
