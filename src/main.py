import os
import sys
import yaml
import psutil
import datetime
import json
import platform
import shutil
import socket
import subprocess
import traceback

sys.path.append(os.path.abspath(".."))

from config.benchmark import ROOT_DIR, BENCHMARK_DIR, benchmark_dict, benchmark_type_dict, benchmark_path_dict
THIRDPARTY_DIR = os.path.join(ROOT_DIR, "thirdparty")
SOURCE_DIR = os.path.join(ROOT_DIR, "src")

from types import SimpleNamespace
from logger import Logger
from utils.debug import *
from utils.random_parser import set_seed
from utils.res2sheet import res2sheet
from utils.res2sheet_incre import res2sheet_incre

sys.path.append(ROOT_DIR)
sys.path.append(THIRDPARTY_DIR)
sys.path.append(SOURCE_DIR)
sys.path.append(BENCHMARK_DIR)

os.environ["PYTHONPATH"] = ":".join(sys.path)

cpus = psutil.cpu_count(logical=True)

import logging
logging.root.name = 'BBO4Placement'
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)-7s] %(name)s - %(message)s',
                    stream=sys.stdout)


def process_benchmark_path(benchmark):
    is_benchmark_registered = False
    for benchmark_base in benchmark_dict:
        if benchmark in benchmark_dict[benchmark_base]:
            is_benchmark_registered = True
            break
    if not is_benchmark_registered:
        assert0(f"benchmark {benchmark} was not registered in config/benchmark.py")

    benchmark_path = os.path.join(BENCHMARK_DIR, benchmark_base, benchmark)
    benchmark_type = benchmark_type_dict[benchmark_base]
    
    return benchmark_path, benchmark_type, benchmark_base


def set_error_log(file):
    error_log = open(file, 'a')
    os.dup2(error_log.fileno(), 2)


def _serializable_config_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serializable_config_value(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _serializable_config_value(v) for k, v in value.items()
        }
    return str(value)


def save_run_metadata(args):
    resolved = {
        key: _serializable_config_value(value)
        for key, value in sorted(vars(args).items())
        if key not in {"logger", "record_func"}
    }
    with open(
        os.path.join(args.result_path, "resolved_config.yaml"),
        "w",
    ) as f:
        yaml.safe_dump(resolved, f, sort_keys=True)

    def git_output(*git_args):
        return subprocess.check_output(
            ["git", "-C", ROOT_DIR, *git_args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--porcelain")
    except Exception:
        git_commit = None
        git_status = "unavailable"

    import numpy
    import pymoo

    metadata = {
        "command": sys.argv,
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "pymoo_version": pymoo.__version__,
        "git_commit": git_commit,
        "git_status": git_status,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cpu_count_visible": cpus,
        "n_cpu_max": args.n_cpu_max,
        "gpu_index": args.gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "objective_evaluator": (
            "task2_cpu_hpwl_plus_unmodified_dreamplace_rudy"
            if args.algorithm == "task2_moea"
            else ("dreamplace_gp_hpwl" if args.eval_gp_hpwl else "cpu_comp_res")
        ),
    }
    with open(
        os.path.join(args.result_path, "runtime_metadata.json"),
        "w",
    ) as f:
        json.dump(metadata, f, indent=2)

    protocol_name = (
        "task2_protocol.yaml"
        if args.algorithm == "task2_moea"
        else "task1_protocol.yaml"
    )
    protocol_path = os.path.join(ROOT_DIR, "experiments", protocol_name)
    if os.path.exists(protocol_path):
        shutil.copy2(
            protocol_path,
            os.path.join(args.result_path, protocol_name),
        )


def process_args():
    # cmd config
    params = [arg.lstrip("--") for arg in sys.argv if arg.startswith("--")]

    cmd_config_dict = {}
    for arg in params:
        key, value = arg.split('=')
        try:
            cmd_config_dict[key] = eval(value)
        except:
            cmd_config_dict[key] = value


    # default config
    config_path = os.path.abspath("../config/default.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in cmd_config_dict.items():
        config_dict[key] = value

    config_dict["placer"] = config_dict["placer"]
    config_dict["algorithm"] = config_dict["algorithm"]

    # placer config
    with open(f"../config/placer/{config_dict['placer']}.yaml", 'r') as f:
        try:
            config_dict.update(yaml.load(f, Loader=yaml.FullLoader))
        except:
            pass
    
    # algo config
    with open(f"../config/algorithm/{config_dict['algorithm']}.yaml", 'r') as f:
        try:
            config_dict.update(yaml.load(f, Loader=yaml.FullLoader))
        except:
            pass

    for key, value in cmd_config_dict.items():
        config_dict[key] = value
    
    args = SimpleNamespace(**config_dict)
    args.benchmark_path, args.benchmark_type, args.benchmark_base = process_benchmark_path(config_dict["benchmark"])


    setattr(args, "ROOT_DIR", ROOT_DIR)
    setattr(args, "THIRDPARTY_DIR", THIRDPARTY_DIR)
    setattr(args, "SOURCE_DIR", SOURCE_DIR)

    return args

    
def single_run(args):
    
    # set seed
    set_seed(args.seed)

    # set unique token
    unique_token = "seed_{}_{}".format(args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    # set result path
    args.result_path = os.path.join(ROOT_DIR, 
                                    f"results/{args.benchmark}/{args.name}/{args.placer}/{args.algorithm}/{args.unique_token}")
    os.makedirs(args.result_path, exist_ok=True)
    save_run_metadata(args)

    # set error log
    error_log_file = os.path.join(args.result_path, "error.log")
    if args.error_redirect:
        set_error_log(file=error_log_file)
    args.error_log_file = error_log_file
    

    logger = Logger(args=args)
    placedb = PlaceDB(args=args)
    placer = PLACER_REGISTRY[args.placer](args=args, placedb=placedb)
    runner = ALGO_REGISTRY[args.algorithm](args=args, placer=placer, logger=logger)
    runner.run()
    logging.info("Exit single run")



if __name__ == "__main__":
    args = process_args()
    # The outer Task 2 scheduler binds each process to a physical GPU. Preserve
    # that binding; only use --gpu when no binding was supplied by the caller.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import ray
    from placedb import PlaceDB
    from placer import REGISTRY as PLACER_REGISTRY
    from algorithm import REGISTRY as ALGO_REGISTRY
    
    num_cpus = min(args.n_cpu_max, cpus)
        

    ray_temp_dir = os.environ.get(
        "RAY_TMPDIR", os.path.expanduser("~/tmp")
    )
    os.makedirs(ray_temp_dir, exist_ok=True)
    ray_num_gpus = 1 if args.eval_gp_hpwl else 0

    ray.init(
        num_cpus=num_cpus,
        num_gpus=ray_num_gpus,
        object_store_memory=int(args.ray_object_store_memory_mb * 1024 * 1024),
        include_dashboard=False,
        logging_level=logging.ERROR,
        _temp_dir=ray_temp_dir,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": f"{args.gpu}"}}
    )

    if args.run_mode == "single":
        single_run(args)
    elif args.run_mode == "result":
        sheet_path = os.path.join(ROOT_DIR, "sheets")
        os.makedirs(sheet_path, exist_ok=True)
        result_path = os.path.join(ROOT_DIR, f"results/{args.benchmark}/{args.name}/{args.placer}/{args.algorithm}")
        res2sheet(args=args, sheet_path=sheet_path, res_path=result_path)
    elif args.run_mode == "result_incre":
        origin_sheet_path = os.path.join(ROOT_DIR, "sheets", f"{args.benchmark_base}.csv")
        target_sheet_path = os.path.join(ROOT_DIR, "sheets", f"{args.benchmark_base}_incre.csv")
        result_path = os.path.join(ROOT_DIR, f"results/{args.benchmark}/{args.name}/{args.placer}/{args.algorithm}")
        res2sheet_incre(
            args=args, 
            origin_sheet_path=origin_sheet_path, 
            target_sheet_path=target_sheet_path, 
            res_path=result_path
        )
    else:
        raise NotImplementedError
    logging.info("Exit Main")
