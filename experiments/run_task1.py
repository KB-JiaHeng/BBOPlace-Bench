#!/usr/bin/env python3
"""Run the frozen Task 1 experiment protocol in independent processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PROTOCOL_PATH = ROOT / "experiments" / "task1_protocol.yaml"
RUN_STATE_ROOT = ROOT / "experiments" / "task1_runs"
RESULTS_ROOT = ROOT / "results" / "adaptec1"


@dataclass(frozen=True)
class Configuration:
    family: str
    crossover: str
    mutation: str
    sbx_eta: float = 30.0
    pm_eta: float = 30.0

    @property
    def label(self) -> str:
        if self.family == "main":
            return f"main__{self.crossover}__{self.mutation}"
        if self.family == "sbx_eta":
            return f"sensitivity__sbx_eta_{self.sbx_eta:g}__swap"
        if self.family == "pm_eta":
            return f"sensitivity__uniform__pm_eta_{self.pm_eta:g}"
        raise ValueError(self.family)


def load_protocol() -> dict:
    with PROTOCOL_PATH.open() as f:
        protocol = yaml.safe_load(f)
    if protocol["status"] != "frozen" or protocol["protocol_version"] != 2:
        raise RuntimeError("Task 1 protocol is not frozen at version 2")
    return protocol


def main_configurations() -> list[Configuration]:
    mutations = ["swap", "shift", "random_resetting", "shuffle", "pm"]
    return [
        Configuration("main", crossover, mutation)
        for crossover in ["uniform", "sbx"]
        for mutation in mutations
    ]


def formal_configurations() -> list[Configuration]:
    return main_configurations() + [
        Configuration("sbx_eta", "sbx", "swap", sbx_eta=5.0),
        Configuration("sbx_eta", "sbx", "swap", sbx_eta=15.0),
        Configuration("pm_eta", "uniform", "pm", pm_eta=5.0),
        Configuration("pm_eta", "uniform", "pm", pm_eta=15.0),
    ]


def command_for(
    python: str,
    config: Configuration,
    seed: int,
    mode: str,
    max_evals: int,
    cpus_per_run: int,
) -> list[str]:
    name = f"task1_{mode}__{config.label}"
    values = {
        "name": name,
        "placer": "mgo",
        "algorithm": "ea",
        "benchmark": "adaptec1",
        "seed": seed,
        "max_evals": max_evals,
        "max_eval_time": 24,
        "run_mode": "single",
        "job_type": "train" if mode == "formal" else "debug",
        "use_wandb": False,
        "eval_gp_hpwl": False,
        "gpu": 0,
        "n_cpu_max": cpus_per_run,
        "ray_object_store_memory_mb": 512,
        "n_grid_x": 224,
        "n_grid_y": 224,
        "rank_key": "area_sum",
        "sampling": "random",
        "n_population": 20,
        "n_sampling_repeat": 1,
        "n_offsprings": 20,
        "eliminate_duplicates": True,
        "duplicate_retry_limit": 100,
        "checkpoint_enabled": False,
        "record_evaluation_trace": True,
        "task1_smoke_test": mode == "smoke",
        "task1_family": config.family,
        "crossover": config.crossover,
        "mutation": config.mutation,
        "crossover_prob": 1.0,
        "uniform_prob_var": 0.5,
        "sbx_prob_var": 0.5,
        "sbx_prob_exch": 1.0,
        "sbx_prob_bin": 0.5,
        "sbx_eta": config.sbx_eta,
        "pm_prob": 1.0,
        "pm_prob_var": 1.0,
        "pm_eta": config.pm_eta,
        "error_redirect": False,
        "n_max_saving_placement": 1,
    }

    def encode(value: object) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    return [
        python,
        "main.py",
        *[f"--{key}={encode(value)}" for key, value in values.items()],
    ]


def find_completed_result(name: str, seed: int, started_at: float) -> Path:
    base = RESULTS_ROOT / name / "mgo" / "ea"
    candidates = sorted(
        base.glob(f"seed_{seed}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        marker = path / "run_complete.json"
        if path.stat().st_mtime >= started_at - 2 and marker.exists():
            return path
    raise RuntimeError(f"No completed result directory found under {base}")


def validate_completion(result_path: Path, expected_evals: int) -> dict:
    with (result_path / "run_complete.json").open() as f:
        completion = json.load(f)
    if completion.get("status") != "complete":
        raise RuntimeError(f"Incomplete run: {result_path}")
    if completion.get("true_evaluation_count") != expected_evals:
        raise RuntimeError(
            f"Wrong true evaluation count in {result_path}: {completion}"
        )
    if completion.get("pymoo_evaluation_count") != expected_evals:
        raise RuntimeError(
            f"Wrong pymoo evaluation count in {result_path}: {completion}"
        )

    trace = result_path / "evaluation_trace.csv"
    with trace.open() as f:
        line_count = sum(1 for _ in f) - 1
    if line_count != expected_evals:
        raise RuntimeError(
            f"Trace has {line_count} evaluations, expected {expected_evals}: "
            f"{trace}"
        )
    return completion


def validate_initial_hash(mode_dir: Path, seed: int, result_path: Path) -> None:
    with (result_path / "initial_population_hash.json").open() as f:
        current = json.load(f)
    expected_path = mode_dir / f"seed_{seed}_initial_hash.json"
    if expected_path.exists():
        with expected_path.open() as f:
            expected = json.load(f)
        if current != expected:
            raise RuntimeError(
                "Initial population differs between configurations for seed "
                f"{seed}: expected={expected}, current={current}, "
                f"result={result_path}"
            )
    else:
        with expected_path.open("w") as f:
            json.dump(current, f, indent=2)


def existing_completed_result(manifest_path: Path, expected_evals: int) -> Path | None:
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open() as f:
            manifest = json.load(f)
        result_path = Path(manifest["result_path"])
        validate_completion(result_path, expected_evals)
        return result_path
    except Exception:
        return None


def run_one(
    python: str,
    config: Configuration,
    seed: int,
    mode: str,
    max_evals: int,
    mode_dir: Path,
    cpus_per_run: int,
) -> Path:
    key = f"seed_{seed}__{config.label}"
    manifest_path = mode_dir / f"{key}.json"
    completed = existing_completed_result(manifest_path, max_evals)
    if completed is not None:
        validate_initial_hash(mode_dir, seed, completed)
        print(f"[skip] {key} -> {completed}", flush=True)
        return completed

    command = command_for(
        python, config, seed, mode, max_evals, cpus_per_run
    )
    name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--name="))
    log_path = mode_dir / f"{key}.log"
    started_at = time.time()
    manifest = {
        "status": "running",
        "mode": mode,
        "seed": seed,
        "configuration": asdict(config),
        "command": command,
        "started_at_unix": started_at,
        "log_path": str(log_path),
        "temp_directory_id": hashlib.sha256(key.encode()).hexdigest()[:12],
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    temp_base = Path(
        os.environ.get("TASK1_TMP_ROOT", str(ROOT / ".task1_tmp"))
    ).resolve()
    short_id = hashlib.sha256(key.encode()).hexdigest()[:12]
    isolated_root = temp_base / short_id
    ray_tmp = isolated_root / "r"
    tmp_dir = isolated_root / "t"
    mpl_dir = isolated_root / "m"
    cache_dir = isolated_root / "c"
    wandb_dir = isolated_root / "w"
    for directory in [ray_tmp, tmp_dir, mpl_dir, cache_dir, wandb_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(seed),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "RAY_TMPDIR": str(ray_tmp),
            "TMPDIR": str(tmp_dir),
            "MPLCONFIGDIR": str(mpl_dir),
            "XDG_CACHE_HOME": str(cache_dir),
            "WANDB_DIR": str(wandb_dir),
        }
    )

    print(f"[start] {key}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            cwd=SRC,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    manifest["returncode"] = process.returncode
    manifest["finished_at_unix"] = time.time()
    if process.returncode != 0:
        manifest["status"] = "failed"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        tail = log_path.read_text(errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"Run failed: {key}\n" + "\n".join(tail)
        )

    result_path = find_completed_result(name, seed, started_at)
    completion = validate_completion(result_path, max_evals)
    validate_initial_hash(mode_dir, seed, result_path)

    manifest.update(
        {
            "status": "complete",
            "result_path": str(result_path),
            "completion": completion,
        }
    )
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[done] {key} best={completion['final_best_hpwl']:.6f}",
        flush=True,
    )
    return result_path


def scheduled_runs(
    configurations: Iterable[Configuration],
    seeds: Iterable[int],
    schedule_seed: int,
) -> Iterable[tuple[int, Configuration]]:
    configurations = list(configurations)
    for seed in seeds:
        block = configurations.copy()
        random.Random(schedule_seed + seed).shuffle(block)
        for config in block:
            yield seed, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "formal"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpus-per-run", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    mode_dir = RUN_STATE_ROOT / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        configurations = formal_configurations()
        seeds = [1]
        max_evals = 60
    else:
        configurations = formal_configurations()
        seeds = protocol["randomness"]["seeds"]
        max_evals = protocol["budget"]["true_objective_evaluations"]

    schedule_seed = protocol["randomness"]["execution_order"]["schedule_seed"]
    total = len(configurations) * len(seeds)
    print(
        f"Task 1 {args.mode}: {total} runs, max_evals={max_evals}, "
        f"{args.cpus_per_run} CPUs per run, {args.workers} concurrent runs",
        flush=True,
    )

    runs = list(scheduled_runs(configurations, seeds, schedule_seed))

    if args.cpus_per_run != protocol["compute"]["cpus_per_run"]:
        raise ValueError(
            f"cpus_per_run={args.cpus_per_run} violates protocol value "
            f"{protocol['compute']['cpus_per_run']}"
        )
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers > protocol["compute"]["concurrent_runs"]:
        raise ValueError(
            f"workers={args.workers} exceeds protocol cap "
            f"{protocol['compute']['concurrent_runs']}"
        )
    if args.workers * args.cpus_per_run > protocol["compute"]["total_cpu_limit"]:
        raise ValueError("Remote total CPU limit exceeded")
    if args.mode == "smoke" and args.workers != 1:
        raise ValueError("Smoke tests must run sequentially")

    def execute(run: tuple[int, Configuration]) -> Path:
        seed, config = run
        return run_one(
            python=args.python,
            config=config,
            seed=seed,
            mode=args.mode,
            max_evals=max_evals,
            mode_dir=mode_dir,
            cpus_per_run=args.cpus_per_run,
        )

    if args.workers == 1:
        for index, run in enumerate(runs, start=1):
            print(f"[{index}/{total}]", end=" ", flush=True)
            execute(run)
    else:
        # Create one trusted initial-population hash per seed before launching
        # other configurations of that seed concurrently. Anchors themselves
        # run concurrently across the five different seeds.
        anchors = [
            (seed, Configuration("main", "uniform", "swap"))
            for seed in seeds
        ]
        anchor_set = set(anchors)
        with ThreadPoolExecutor(max_workers=min(args.workers, len(anchors))) as pool:
            futures = {pool.submit(execute, run): run for run in anchors}
            for future in as_completed(futures):
                future.result()

        remaining = [run for run in runs if run not in anchor_set]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(execute, run): run for run in remaining}
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                print(
                    f"[parallel progress] {completed}/{len(remaining)}",
                    flush=True,
                )

    if args.mode == "formal":
        subprocess.run(
            [args.python, str(ROOT / "experiments" / "analyze_task1.py")],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
