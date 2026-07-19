import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from src.evaluator import Evaluator
from src.placer import REGISTRY
from src.placer.basic_placer import evaluate_placer
from src.placer.mgo_placer import MaskGuidedOptimizationPlacer


class _DummyEvaluatorPlacer:
    def __init__(self):
        self.received = None

    def evaluate(self, population):
        self.received = np.asarray(population)
        return [123.0], [0.0], [{}]


class _OneMacroPlaceDB:
    canvas_width = 2.0
    canvas_height = 2.0
    node_cnt = 1
    macro_lst = ["macro"]
    node_info = {
        "macro": {
            "size_x": 1.0,
            "size_y": 1.0,
            "area": 1.0,
        }
    }
    net_info = {}
    node_to_net_dict = {"macro": ()}


class FrameworkFixesTest(unittest.TestCase):
    def test_evaluator_accepts_one_dimensional_input(self):
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.placer = _DummyEvaluatorPlacer()

        hpwl = evaluator.evaluate(np.asarray([1.0, 2.0]))

        self.assertEqual(evaluator.placer.received.shape, (1, 2))
        np.testing.assert_array_equal(hpwl, np.asarray([123.0]))

    def test_cpu_placement_workers_do_not_require_a_gpu(self):
        self.assertEqual(evaluate_placer._default_options["num_gpus"], 0)

    def test_default_placer_is_registered(self):
        config = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["placer"], "mgo")
        self.assertIn(config["placer"], REGISTRY)

    def test_mgo_can_place_a_macro_at_the_upper_grid_boundary(self):
        with tempfile.TemporaryDirectory() as result_path:
            args = SimpleNamespace(
                result_path=result_path,
                n_max_saving_placement=1,
                eval_gp_hpwl=False,
                n_grid_x=2,
                n_grid_y=2,
                rank_key="area_sum",
            )
            placer = MaskGuidedOptimizationPlacer(args, _OneMacroPlaceDB())

            placement = placer._genotype2phenotype(np.asarray([1, 1]))

        self.assertEqual(placement["macro"], (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
