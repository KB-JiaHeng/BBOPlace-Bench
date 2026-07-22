from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from placedb import PlaceDB
from task2.benchmark_fingerprint import (
    placedb_fingerprint,
    validate_expected_fingerprint,
)


def make_args(n_macro: int) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark="adaptec1",
        benchmark_base="ispd2005",
        benchmark_type="aux",
        benchmark_path=str(ROOT / "benchmarks" / "ispd2005" / "adaptec1"),
        n_macro=n_macro,
    )


class Task2BenchmarkFingerprintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "experiments" / "task2_protocol.yaml").open() as f:
            cls.protocol = yaml.safe_load(f)

    def test_exact_512_macro_subproblem_matches_frozen_fingerprint(self):
        placedb = PlaceDB(make_args(512))
        actual = placedb_fingerprint(placedb)
        selection = self.protocol["inheritance_from_task1"]["macro_selection"]
        expected = {
            "node_count": selection["expected_node_count"],
            "net_count": selection["expected_net_count"],
            "macro_names_sha256": selection["expected_macro_names_sha256"],
            "macro_geometry_sha256": selection["expected_macro_geometry_sha256"],
            "net_topology_sha256": selection["expected_net_topology_sha256"],
            "macro_area_sum": selection["expected_macro_area_sum"],
            "canvas": selection["expected_canvas"],
        }
        validate_expected_fingerprint(actual, expected)
        self.assertEqual(placedb.node_cnt * 2, 1024)

    def test_full_instance_diff_is_confined_to_known_high_degree_net(self):
        reduced = PlaceDB(make_args(512))
        full = PlaceDB(make_args(1_000_000))
        self.assertEqual((reduced.node_cnt, full.node_cnt), (512, 543))
        self.assertEqual((reduced.net_cnt, full.net_cnt), (693, 693))

        changed = []
        for net_name in full.net_info:
            reduced_nodes = set(reduced.net_info[net_name]["nodes"])
            full_nodes = set(full.net_info[net_name]["nodes"])
            if reduced_nodes != full_nodes:
                changed.append(
                    (net_name, len(reduced_nodes), len(full_nodes), full_nodes - reduced_nodes)
                )
        self.assertEqual(len(changed), 1)
        net_name, reduced_degree, full_degree, excluded = changed[0]
        self.assertEqual(net_name, "n7807")
        self.assertEqual((reduced_degree, full_degree, len(excluded)), (318, 349, 31))

    def test_cutoff_is_deterministic_but_area_tied(self):
        reduced = PlaceDB(make_args(512))
        full = PlaceDB(make_args(1_000_000))
        excluded = list(set(full.macro_lst) - set(reduced.macro_lst))
        self.assertEqual(len(excluded), 31)
        cutoff_area = reduced.node_info[reduced.macro_lst[-1]]["area"]
        self.assertTrue(all(full.node_info[name]["area"] == cutoff_area for name in excluded))
        self.assertEqual(cutoff_area, 31104)


if __name__ == "__main__":
    unittest.main()
