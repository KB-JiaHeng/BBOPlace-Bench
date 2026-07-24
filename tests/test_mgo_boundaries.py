from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from placer.mgo_placer import MaskGuidedOptimizationPlacer


def make_placer(result_path: Path, *, macro_size: int) -> MaskGuidedOptimizationPlacer:
    macro = "m0"
    placedb = SimpleNamespace(
        node_cnt=1,
        macro_lst=[macro],
        node_info={
            macro: {
                "id": 0,
                "size_x": macro_size,
                "size_y": macro_size,
                "area": macro_size * macro_size,
            }
        },
        net_info={},
        node_to_net_dict={macro: ()},
        canvas_width=4,
        canvas_height=4,
        macro_area_sum=macro_size * macro_size,
    )
    args = SimpleNamespace(
        n_grid_x=4,
        n_grid_y=4,
        rank_key="area_sum",
        result_path=str(result_path),
        n_max_saving_placement=1,
        eval_gp_hpwl=False,
    )
    return MaskGuidedOptimizationPlacer(args=args, placedb=placedb)


class MgoLegalBoundaryTest(unittest.TestCase):
    def test_one_grid_macro_can_use_last_grid_coordinate(self):
        with tempfile.TemporaryDirectory() as tmp:
            placer = make_placer(Path(tmp), macro_size=1)
            placement = placer._genotype2phenotype([3, 3])
            self.assertEqual(placement["m0"], (3.0, 3.0))

    def test_two_grid_macro_can_use_last_legal_lower_left_coordinate(self):
        with tempfile.TemporaryDirectory() as tmp:
            placer = make_placer(Path(tmp), macro_size=2)
            placement = placer._genotype2phenotype([3, 3])
            # The requested guide is outside the legal lower-left range. The
            # nearest legal point is exactly n_grid - scaled_size = 2.
            self.assertEqual(placement["m0"], (2.0, 2.0))

    def test_full_canvas_macro_has_exactly_one_legal_coordinate(self):
        with tempfile.TemporaryDirectory() as tmp:
            placer = make_placer(Path(tmp), macro_size=4)
            placement = placer._genotype2phenotype([3, 3])
            self.assertEqual(placement["m0"], (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
