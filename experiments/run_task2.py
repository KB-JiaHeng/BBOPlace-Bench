#!/usr/bin/env python3
"""Run Task 2 smoke, performance, or gated formal experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONFIG_PATH = ROOT / "experiments" / "task2_config.yaml"
RUN_STATE_ROOT = ROOT / "experiments" / "task2_runs"
RESULTS_ROOT = ROOT / "results" / "adaptec1"
SMOKE_DECISION = ROOT / "experiments" / "task2_smoke" / "smoke_decision.json"
DEFINITION_FILES = [
    CONFIG_PATH,
    ROOT / "config" / "default.yaml",
    ROOT / "config" / "placer" / "mgo.yaml",
    ROOT / "config" / "algorithm" / "task2_moea.yaml",
    ROOT / "src" / "utils" / "read_benchmark" / "read_aux.py",
    ROOT / "src" / "utils" / "compute_res.py",
    ROOT / "src" / "placer" / "mgo_placer.py",
    ROOT / "src" / "task2" / "benchmark_fingerprint.py",
    ROOT / "src" / "task2" / "metrics.py",
    ROOT / "src" / "task2" / "evaluator.py",
    ROOT / "src" / "algorithm" / "moea" / "task2_core.py",
    ROOT / "src" / "algorithm" / "moea" / "task2_moea.py",
]


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int

    @property
    def key(self) -> str:
        return f"seed_{self.seed}__{self.method}"


def load_protocol() -> dict:
    with CONFIG_PATH.open() as f:
        protocol = yaml.safe_load(f)
    if protocol.get("status") != "frozen" or protocol.get("protocol_version") != 3:
        raise RuntimeError("Task 2 protocol is not frozen at version 3")
    return protocol


def parse_csv(value: str, converter=str) -> list:
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def experiment_definition_fingerprint(command: list[str]) -> str:
    """Hash the semantic command and all files that define Task 2 behavior."""
    payload = {
        "command_without_python": command[1:],
        "definition_files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in DEFINITION_FILES
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def query_gpu_memory_mib() -> dict[int, int]:
    """Return current GPU memory use without importing CUDA into the scheduler."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    usage: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index_text, memory_text = [part.strip() for part in line.split(",", 1)]
        usage[int(index_text)] = int(memory_text)
    return usage


def require_idle_gpus(gpu_indices: list[int], *, maximum_idle_memory_mib: int = 64) -> dict[int, int]:
    """Block formal launch rather than sharing GPUs with unrelated workloads."""
    usage = query_gpu_memory_mib()
    missing = [index for index in gpu_indices if index not in usage]
    if missing:
        raise RuntimeError(f"Requested GPUs are absent from nvidia-smi: {missing}")
    busy = {
        index: usage[index]
        for index in gpu_indices
        if usage[index] > maximum_idle_memory_mib
    }
    if busy:
        raise RuntimeError(
            "Formal launch is blocked because recommended GPUs are not idle: "
            + json.dumps(busy, sort_keys=True)
        )
    return {index: usage[index] for index in gpu_indices}


def load_formal_recommendation() -> dict:
    if not SMOKE_DECISION.exists():
        raise RuntimeError(
            "Formal launch is blocked: task2_smoke/smoke_decision.json is absent"
        )
    decision = json.loads(SMOKE_DECISION.read_text())
    if decision.get("formal_launch_allowed") is not True:
        raise RuntimeError(
            "Formal launch is blocked by smoke decision: "
            + json.dumps(decision, indent=2)
        )
    recommendation = decision.get("recommended_scheduler")
    if not isinstance(recommendation, dict):
        raise RuntimeError("Smoke decision has no recommended_scheduler")
    return recommendation


def run_name(mode: str, method: str, backend: str, cpus: int, rudy_threads: int) -> str:
    if mode == "formal":
        return f"task2_formal__{method}"
    return f"task2_{mode}__{method}__{backend}__cpu{cpus}__rudy{rudy_threads}"


def command_for(
    python: str,
    spec: RunSpec,
    *,
    mode: str,
    max_evals: int,
    cpus_per_run: int,
    rudy_backend: str,
    rudy_cpu_threads: int,
) -> list[str]:
    name = run_name(
        mode,
        spec.method,
        rudy_backend,
        cpus_per_run,
        rudy_cpu_threads,
    )
    values = {
        "name": name,
        "placer": "mgo",
        "algorithm": "task2_moea",
        "task2_method": spec.method,
        "benchmark": "adaptec1",
        "benchmark_variant": "adaptec1_upstream_native_top512_macro_subset",
        "n_macro": 512,
        "seed": spec.seed,
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
        "record_evaluation_trace": False,
        "crossover": "uniform",
        "mutation": "swap",
        "crossover_prob": 1.0,
        "uniform_prob_var": 0.5,
        "rudy_backend": rudy_backend,
        "rudy_cpu_threads": rudy_cpu_threads,
        "rudy_dtype": "float32",
        "rudy_deterministic": True,
        "rudy_num_bins_x": 512,
        "rudy_num_bins_y": 512,
        "rudy_unit_horizontal_capacity": 1.5625,
        "rudy_unit_vertical_capacity": 1.45,
        "rudy_hotspot_fraction": 0.10,
        "moead_n_neighbors": 10,
        "task2_smoke_test": mode != "formal",
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
    base = RESULTS_ROOT / name / "mgo" / "task2_moea"
    candidates = sorted(
        base.glob(f"seed_{seed}_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path.stat().st_mtime >= started_at - 2 and (path / "run_complete.json").exists():
            return path
    raise RuntimeError(f"No completed Task 2 result found under {base}")


def validate_completion(
    path: Path,
    expected_evals: int,
    method: str,
    expected_definition_fingerprint: str | None = None,
) -> dict:
    completion = json.loads((path / "run_complete.json").read_text())
    required = {
        "status": "complete",
        "method": method,
        "true_evaluation_count": expected_evals,
        "algorithm_evaluation_count": expected_evals,
        "trace_count": expected_evals,
        "population_size": 20,
    }
    if expected_definition_fingerprint is not None:
        required["definition_fingerprint"] = expected_definition_fingerprint
    for key, expected in required.items():
        if completion.get(key) != expected:
            raise RuntimeError(
                f"Invalid completion field {key}: {completion.get(key)!r} != "
                f"{expected!r} in {path}"
            )
    with (path / "evaluation_trace.csv").open() as f:
        trace_count = sum(1 for _ in f) - 1
    if trace_count != expected_evals:
        raise RuntimeError(
            f"Trace has {trace_count} rows, expected {expected_evals}: {path}"
        )
    for filename in [
        "initial_population.npz",
        "initial_population_hash.json",
        "final_population.npz",
        "offline_archive.npz",
        "generation_metrics.csv",
        "placedb_fingerprint.json",
        "resolved_config.yaml",
        "runtime_metadata.json",
        "task2_config.yaml",
    ]:
        if not (path / filename).exists():
            raise RuntimeError(f"Missing {filename} in {path}")
    return completion


def existing_completed_result(
    manifest_path: Path,
    expected_evals: int,
    method: str,
    expected_definition_fingerprint: str,
) -> Path | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("definition_fingerprint") != expected_definition_fingerprint:
            return None
        path = Path(manifest["result_path"])
        validate_completion(
            path,
            expected_evals,
            method,
            expected_definition_fingerprint,
        )
        return path
    except Exception:
        return None


def allocate_core_sets(
    concurrent_runs: int,
    cpus_per_run: int,
    reserve_cpus: int,
) -> list[list[int]]:
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
    else:
        available = list(range(os.cpu_count() or 1))
    if reserve_cpus >= len(available):
        raise ValueError("reserve_cpus leaves no CPUs available")
    usable = available[: len(available) - reserve_cpus]
    required = concurrent_runs * cpus_per_run
    if required > len(usable):
        raise ValueError(
            f"Requested {required} CPUs but only {len(usable)} remain after reserve"
        )
    return [
        usable[index * cpus_per_run : (index + 1) * cpus_per_run]
        for index in range(concurrent_runs)
    ]


def isolated_environment(
    spec: RunSpec,
    mode: str,
    slot: int,
    backend: str,
    cpus_per_run: int,
    gpu: int | None,
    rudy_threads: int,
) -> dict[str, str]:
    project = Path(
        os.environ.get("TASK2_PROJECT_ROOT", str(ROOT / ".task2_runtime"))
    ).expanduser().resolve()
    run_id = f"{mode}__{spec.key}__slot_{slot}"
    isolated = project / "tmp" / "runs" / run_id
    directories = {
        "home": project / "home",
        # Ray appends a long session/sockets suffix and AF_UNIX has a 107-byte
        # path limit. Keep this Task-2-owned prefix deliberately short.
        "ray": project / f"r{mode[0]}{backend[0]}{slot}{cpus_per_run:x}{rudy_threads:x}",
        "tmp": isolated / "tmp",
        "mpl": project / "cache" / "matplotlib",
        "xdg": project / "cache" / "xdg",
        "torch": project / "cache" / "torch-extensions",
        "cuda": project / "cache" / "cuda",
        "pip": project / "pip-cache",
        "wandb": project / "cache" / "wandb",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(directories["home"]),
            "PYTHONHASHSEED": str(spec.seed),
            "OMP_NUM_THREADS": str(rudy_threads),
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
            "RAY_TMPDIR": str(directories["ray"]),
            "TMPDIR": str(directories["tmp"]),
            "MPLCONFIGDIR": str(directories["mpl"]),
            "XDG_CACHE_HOME": str(directories["xdg"]),
            "TORCH_EXTENSIONS_DIR": str(directories["torch"]),
            "CUDA_CACHE_PATH": str(directories["cuda"]),
            "PIP_CACHE_DIR": str(directories["pip"]),
            "WANDB_DIR": str(directories["wandb"]),
            "TASK2_PROJECT_ROOT": str(project),
        }
    )
    env["CUDA_VISIBLE_DEVICES"] = str(gpu) if backend == "cuda" and gpu is not None else ""
    return env


def run_one(
    spec: RunSpec,
    *,
    mode: str,
    python: str,
    max_evals: int,
    cpus_per_run: int,
    backend: str,
    rudy_threads: int,
    nice_level: int,
    slot: int,
    cores: list[int],
    gpu: int | None,
    mode_dir: Path,
) -> Path:
    key = spec.key
    manifest_label = (
        key
        if mode == "formal"
        else f"{key}__{backend}__cpu{cpus_per_run}__rudy{rudy_threads}"
    )
    manifest_path = mode_dir / f"{manifest_label}.json"
    command = command_for(
        python,
        spec,
        mode=mode,
        max_evals=max_evals,
        cpus_per_run=cpus_per_run,
        rudy_backend=backend,
        rudy_cpu_threads=rudy_threads,
    )
    definition_fingerprint = experiment_definition_fingerprint(command)
    command.append(f"--task2_definition_fingerprint={definition_fingerprint}")
    completed = existing_completed_result(
        manifest_path,
        max_evals,
        spec.method,
        definition_fingerprint,
    )
    if completed is not None:
        print(f"[skip] {key} -> {completed}", flush=True)
        return completed

    name = next(value.split("=", 1)[1] for value in command if value.startswith("--name="))
    launch = ["nice", "-n", str(nice_level)]
    if shutil.which("taskset") and cores:
        launch += ["taskset", "-c", ",".join(map(str, cores))]
    launch += command

    log_path = mode_dir / f"{manifest_label}.log"
    started = time.time()
    manifest = {
        "status": "running",
        "mode": mode,
        "spec": asdict(spec),
        "command": launch,
        "definition_fingerprint": definition_fingerprint,
        "started_at_unix": started,
        "slot": slot,
        "cores": cores,
        "gpu": gpu,
        "backend": backend,
        "rudy_cpu_threads": rudy_threads,
        "log_path": str(log_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"[start] {key} slot={slot} cores={cores[0]}-{cores[-1]} "
        f"gpu={gpu} backend={backend}",
        flush=True,
    )
    env = isolated_environment(
        spec, mode, slot, backend, cpus_per_run, gpu, rudy_threads
    )
    with log_path.open("w") as log:
        process = subprocess.run(
            launch,
            cwd=SRC,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if process.returncode != 0:
        manifest.update({"status": "failed", "returncode": process.returncode})
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise RuntimeError(f"Task 2 run failed: {key}; see {log_path}")

    result_path = find_completed_result(name, spec.seed, started)
    completion = validate_completion(
        result_path,
        max_evals,
        spec.method,
        definition_fingerprint,
    )
    manifest.update(
        {
            "status": "complete",
            "result_path": str(result_path),
            "completion": completion,
            "finished_at_unix": time.time(),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] {key} -> {result_path}", flush=True)
    return result_path


def validate_paired_initial_populations(results: Iterable[Path]) -> dict[int, dict]:
    by_seed: dict[int, list[tuple[Path, dict]]] = {}
    for path in results:
        completion = json.loads((path / "run_complete.json").read_text())
        seed = int(completion["seed"])
        metadata = json.loads((path / "initial_population_hash.json").read_text())
        by_seed.setdefault(seed, []).append((path, metadata))
    summary = {}
    for seed, entries in by_seed.items():
        ordered = {metadata["x_sha256"] for _, metadata in entries}
        unordered = {metadata["x_set_sha256"] for _, metadata in entries}
        if len(ordered) != 1 or len(unordered) != 1:
            raise RuntimeError(
                f"Initial populations differ for seed {seed}: "
                + json.dumps(
                    [
                        {"path": str(path), "metadata": metadata}
                        for path, metadata in entries
                    ],
                    indent=2,
                )
            )
        summary[seed] = entries[0][1]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "performance", "formal"], required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--methods", default="nsga2,moead")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--max-evals", type=int, default=200)
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpus-per-run", type=int, default=0)
    parser.add_argument("--rudy-cpu-threads", type=int, default=0)
    parser.add_argument("--concurrent-runs", type=int, default=0)
    parser.add_argument("--reserve-cpus", type=int, default=32)
    parser.add_argument("--gpu-indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nice-level", type=int, default=10)
    args = parser.parse_args()
    protocol = load_protocol()

    if args.mode == "formal":
        recommendation = load_formal_recommendation()
        backend = recommendation["backend"]
        cpus_per_run = int(recommendation["cpus_per_run"])
        rudy_threads = int(recommendation["rudy_cpu_threads"])
        concurrent_runs = int(recommendation["concurrent_runs"])
        reserve_cpus = int(recommendation.get("reserve_cpus", args.reserve_cpus))
        nice_level = int(recommendation.get("nice_level", args.nice_level))
        max_evals = int(protocol["budget_and_randomness"]["true_evaluations_per_run"])
        methods = list(protocol["scope"]["formal_comparison"])
        seeds = list(protocol["budget_and_randomness"]["seeds"])
        if backend == "cuda":
            gpu_indices = [int(value) for value in recommendation.get("gpu_indices", [])]
            if not gpu_indices:
                raise RuntimeError("CUDA formal recommendation has no gpu_indices")
            if concurrent_runs > len(gpu_indices):
                raise RuntimeError("CUDA concurrency exceeds recommended GPU list")
            formal_gpu_preflight = require_idle_gpus(gpu_indices[:concurrent_runs])
        else:
            gpu_indices = []
            formal_gpu_preflight = {}
    else:
        if args.backend == "auto" or args.cpus_per_run <= 0 or args.concurrent_runs <= 0:
            raise ValueError(
                "Smoke/performance modes require explicit backend, cpus-per-run, "
                "and concurrent-runs"
            )
        backend = args.backend
        cpus_per_run = args.cpus_per_run
        rudy_threads = args.rudy_cpu_threads or 1
        concurrent_runs = args.concurrent_runs
        reserve_cpus = args.reserve_cpus
        nice_level = args.nice_level
        max_evals = args.max_evals
        methods = parse_csv(args.methods)
        seeds = parse_csv(args.seeds, int)
        formal_gpu_preflight = {}
        if backend == "cuda":
            gpu_indices = parse_csv(args.gpu_indices, int)
            if concurrent_runs > len(gpu_indices):
                raise ValueError("CUDA concurrency exceeds available GPU indices")
        else:
            gpu_indices = []

    specs = [RunSpec(method, seed) for seed in seeds for method in methods]
    mode_dir = RUN_STATE_ROOT / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    core_sets = allocate_core_sets(
        concurrent_runs,
        cpus_per_run,
        reserve_cpus,
    )
    slots: queue.Queue[int] = queue.Queue()
    for slot in range(concurrent_runs):
        slots.put(slot)

    def scheduled(spec: RunSpec) -> Path:
        slot = slots.get()
        try:
            gpu = gpu_indices[slot] if backend == "cuda" else None
            return run_one(
                spec,
                mode=args.mode,
                python=args.python,
                max_evals=max_evals,
                cpus_per_run=cpus_per_run,
                backend=backend,
                rudy_threads=rudy_threads,
                nice_level=nice_level,
                slot=slot,
                cores=core_sets[slot],
                gpu=gpu,
                mode_dir=mode_dir,
            )
        finally:
            slots.put(slot)

    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=concurrent_runs) as executor:
        future_map = {executor.submit(scheduled, spec): spec for spec in specs}
        for future in as_completed(future_map):
            results.append(future.result())

    paired = validate_paired_initial_populations(results)
    summary = {
        "mode": args.mode,
        "backend": backend,
        "cpus_per_run": cpus_per_run,
        "rudy_cpu_threads": rudy_threads,
        "concurrent_runs": concurrent_runs,
        "reserve_cpus": reserve_cpus,
        "gpu_indices": gpu_indices,
        "formal_gpu_preflight": formal_gpu_preflight,
        "max_evals": max_evals,
        "results": [str(path) for path in sorted(results)],
        "paired_initial_population": paired,
        "completed_at_unix": time.time(),
    }
    (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
