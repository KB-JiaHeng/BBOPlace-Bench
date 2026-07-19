import os
import random

import numpy as np

try:
    import torch as th
except ModuleNotFoundError:  # CPU-only Task 1 does not require PyTorch.
    th = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if th is not None:
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


def get_state():
    state_dict = {
        "random": random.getstate(),
        "np_random": np.random.get_state(),
    }
    if th is not None:
        state_dict["th_random"] = th.get_rng_state()
        if th.cuda.is_available():
            state_dict["th_cuda_random"] = th.cuda.get_rng_state()
    return state_dict


def set_state(state_dict):
    random.setstate(state_dict["random"])
    np.random.set_state(state_dict["np_random"])
    if th is not None and "th_random" in state_dict:
        th.set_rng_state(state_dict["th_random"])
        if "th_cuda_random" in state_dict and th.cuda.is_available():
            th.cuda.set_rng_state(state_dict["th_cuda_random"])
