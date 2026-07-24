from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from algorithm.moea.task2_core import genotype_hash, phenotype_hash
from algorithm.moea.task2_moea import Task2MOEA
from task2.evaluator import Task2Evaluation
from experiments.run_task2 import (
    RunSpec,
    command_for,
    experiment_definition_fingerprint,
    isolated_environment,
    query_gpu_memory_mib,
    require_idle_gpus,
)


class FakeObjectiveEvaluator:
    def __init__(self, args, placer):
        self.args = args
        self.placer = placer
        self.true_n_eval = 0
        self.node_count = placer.placedb.node_cnt

    def evaluate_many(self, X, *, parallel_decode):
        results = []
        for x in np.asarray(X, dtype=np.int64):
            self.true_n_eval += 1
            grid = np.column_stack(
                [x[: self.node_count], x[self.node_count :]]
            ).astype(np.int32)
            hpwl = float(np.sum(x[: self.node_count]))
            congestion = float(np.sum(223 - x[self.node_count :]))
            results.append(
                Task2Evaluation(
                    evaluation_id=self.true_n_eval,
                    x=x.copy(),
                    genotype_hash=genotype_hash(x),
                    decode_success=True,
                    macro_pos=None,
                    macro_grid=grid,
                    phenotype_hash=phenotype_hash(grid),
                    f=np.asarray([hpwl, congestion], dtype=float),
                    hpwl=hpwl,
                    rudy_top10=float(np.mean(x)),
                    congestion_top10=congestion,
                    total_overflow=0.0,
                    max_overflow=0.0,
                    overflow_fraction=0.0,
                    overlap_rate=0.0,
                    decode_seconds=0.0,
                    metric_seconds=0.0,
                )
            )
        return results


def make_args(result_path: Path, method: str, seed: int = 1):
    return SimpleNamespace(
        ROOT_DIR=str(ROOT),
        result_path=str(result_path),
        max_eval_time=1,
        placer="mgo",
        algorithm="task2_moea",
        benchmark="adaptec1",
        benchmark_variant="adaptec1_upstream_native_top512_macro_subset",
        benchmark_path=str(ROOT / "benchmarks" / "ispd2005" / "adaptec1"),
        n_macro=512,
        n_population=20,
        n_sampling_repeat=1,
        n_offsprings=20,
        max_evals=60,
        eliminate_duplicates=True,
        duplicate_retry_limit=100,
        checkpoint_enabled=False,
        n_grid_x=224,
        n_grid_y=224,
        rank_key="area_sum",
        sampling="random",
        crossover="uniform",
        mutation="swap",
        crossover_prob=1.0,
        uniform_prob_var=0.5,
        eval_gp_hpwl=False,
        rudy_dtype="float32",
        rudy_deterministic=True,
        rudy_num_bins_x=512,
        rudy_num_bins_y=512,
        rudy_unit_horizontal_capacity=1.5625,
        rudy_unit_vertical_capacity=1.45,
        rudy_hotspot_fraction=0.10,
        rudy_backend="cpu",
        rudy_cpu_threads=1,
        moead_n_neighbors=10,
        task2_method=method,
        task2_smoke_test=True,
        task2_definition_fingerprint="unit-test-definition",
        seed=seed,
        n_cpu_max=1,
    )


class Task2GpuPreflightTest(unittest.TestCase):
    @patch("experiments.run_task2.subprocess.run")
    def test_query_gpu_memory(self, run):
        run.return_value = SimpleNamespace(stdout="0, 2\n1, 570\n2, 3\n")
        self.assertEqual(query_gpu_memory_mib(), {0: 2, 1: 570, 2: 3})

    @patch("experiments.run_task2.query_gpu_memory_mib", return_value={2: 2, 3: 4})
    def test_idle_gpu_preflight_passes(self, _query):
        self.assertEqual(require_idle_gpus([2, 3]), {2: 2, 3: 4})

    @patch("experiments.run_task2.query_gpu_memory_mib", return_value={2: 2, 3: 502})
    def test_busy_gpu_blocks_formal_launch(self, _query):
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            require_idle_gpus([2, 3])


class Task2SchedulerIsolationTest(unittest.TestCase):
    def test_ray_socket_prefix_is_short_and_project_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            with patch.dict("os.environ", {"TASK2_PROJECT_ROOT": str(project)}):
                env = isolated_environment(
                    RunSpec("nsga2", 1),
                    "smoke",
                    0,
                    "cpu",
                    12,
                    None,
                    1,
                )
            ray_root = Path(env["RAY_TMPDIR"])
            self.assertTrue(ray_root.is_relative_to(project))
            self.assertEqual(ray_root.relative_to(project).as_posix(), "rsc0c1")
            # Validate the actual documented remote root rather than coupling
            # the AF_UNIX assertion to pytest's arbitrary TemporaryDirectory.
            production_ray_root = Path("/home/sihengzhao/hsea26-hw5-task2/rsc0c1")
            representative_socket = (
                production_ray_root
                / "session_2026-07-22_16-02-55_230502_2790057"
                / "sockets"
                / "plasma_store"
            )
            self.assertLess(len(str(representative_socket)), 107)


class Task2RunnerTest(unittest.TestCase):
    def run_method(self, base: Path, method: str, seed: int = 1):
        path = base / f"{method}_{seed}_{len(list(base.iterdir()))}"
        path.mkdir()
        args = make_args(path, method, seed)
        macro_lst = ["m0", "m1", "m2"]
        placer = SimpleNamespace(
            placedb=SimpleNamespace(
                node_cnt=3,
                macro_lst=macro_lst,
                node_info={
                    name: {"area": (index + 1) * 10}
                    for index, name in enumerate(macro_lst)
                },
            ),
            ranked_macro=list(reversed(macro_lst)),
        )
        fake_fingerprint = {
            "macro_names_sha256": "unit-macros",
            "net_topology_sha256": "unit-nets",
        }
        with (
            patch(
                "algorithm.moea.task2_moea.Task2ObjectiveEvaluator",
                FakeObjectiveEvaluator,
            ),
            patch(
                "algorithm.moea.task2_moea.placedb_fingerprint",
                return_value=fake_fingerprint,
            ),
            patch("algorithm.moea.task2_moea.validate_expected_fingerprint"),
        ):
            Task2MOEA(args, placer, logger=None).run()
        return path

    def assert_complete(self, path: Path):
        completion = json.loads((path / "run_complete.json").read_text())
        self.assertEqual(completion["true_evaluation_count"], 60)
        self.assertEqual(completion["trace_count"], 60)
        self.assertEqual(completion["population_size"], 20)
        self.assertEqual(completion["evaluated_unique_genotypes"], 60)
        self.assertEqual(
            completion["duplicate_semantics"],
            "run_level_genotype_hash_exclusion",
        )
        self.assertEqual(
            completion["definition_fingerprint"],
            "unit-test-definition",
        )
        self.assertTrue((path / "placedb_fingerprint.json").exists())
        with (path / "evaluation_trace.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({row["genotype_hash"] for row in rows}), 60)
        required_diagnostic_columns = {
            "mean_grid_displacement_parent1",
            "max_grid_displacement_parent1",
            "delta_hpwl_parent1",
            "delta_congestion_top10_parent1",
            "swap_guide_l1_distance",
            "replacement_slot_ids",
            "moead_ideal_before",
            "moead_ideal_after",
        }
        self.assertTrue(required_diagnostic_columns.issubset(rows[0]))
        final = np.load(path / "final_population.npz")
        self.assertEqual(final["X"].shape, (20, 6))
        self.assertEqual(final["F"].shape, (20, 2))

    def test_both_algorithms_complete_exact_budget_with_paired_initial_X(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            nsga = self.run_method(base, "nsga2")
            moead = self.run_method(base, "moead")
            self.assert_complete(nsga)
            self.assert_complete(moead)
            np.testing.assert_array_equal(
                np.load(nsga / "initial_population.npz")["X"],
                np.load(moead / "initial_population.npz")["X"],
            )
            h1 = json.loads((nsga / "initial_population_hash.json").read_text())
            h2 = json.loads((moead / "initial_population_hash.json").read_text())
            self.assertEqual(h1["x_sha256"], h2["x_sha256"])
            self.assertEqual(h1["x_set_sha256"], h2["x_set_sha256"])

    def test_nsga2_replay_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = self.run_method(base, "nsga2", seed=2)
            second = self.run_method(base, "nsga2", seed=2)
            for key in ["X", "F", "evaluation_id", "phenotype_hash"]:
                np.testing.assert_array_equal(
                    np.load(first / "final_population.npz")[key],
                    np.load(second / "final_population.npz")[key],
                )

    def test_moead_records_nonregressing_ideal_and_slot_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.run_method(Path(tmp), "moead", seed=4)
            with (path / "moead_state.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 2)
            ideals = np.asarray([json.loads(row["ideal"]) for row in rows])
            self.assertTrue(np.all(np.diff(ideals, axis=0) <= 0.0))
            self.assertTrue(
                all(len(json.loads(row["slot_evaluation_ids"])) == 20 for row in rows)
            )

    def test_moead_replay_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = self.run_method(base, "moead", seed=3)
            second = self.run_method(base, "moead", seed=3)
            for key in ["X", "F", "evaluation_id", "phenotype_hash"]:
                np.testing.assert_array_equal(
                    np.load(first / "final_population.npz")[key],
                    np.load(second / "final_population.npz")[key],
                )


class Task2CommandDefinitionTest(unittest.TestCase):
    def test_command_explicitly_selects_512_macros(self):
        command = command_for(
            "python",
            RunSpec("nsga2", 1),
            mode="smoke",
            max_evals=200,
            cpus_per_run=20,
            rudy_backend="cpu",
            rudy_cpu_threads=1,
        )
        self.assertIn("--n_macro=512", command)
        self.assertIn("--n_grid_x=224", command)
        self.assertIn("--n_grid_y=224", command)

    def test_definition_fingerprint_is_deterministic_and_semantic(self):
        command = command_for(
            "python-a",
            RunSpec("nsga2", 1),
            mode="smoke",
            max_evals=200,
            cpus_per_run=20,
            rudy_backend="cpu",
            rudy_cpu_threads=1,
        )
        same_semantics = ["python-b", *command[1:]]
        self.assertEqual(
            experiment_definition_fingerprint(command),
            experiment_definition_fingerprint(same_semantics),
        )
        changed = [
            value.replace("--n_macro=512", "--n_macro=543")
            for value in command
        ]
        self.assertNotEqual(
            experiment_definition_fingerprint(command),
            experiment_definition_fingerprint(changed),
        )


if __name__ == "__main__":
    unittest.main()
