#!/usr/bin/env bash
# Source this file from the uploaded project root.
set -euo pipefail
TASK2_PROJECT_ROOT="${TASK2_PROJECT_ROOT:-$HOME/hsea26-hw5-task2}"
export TASK2_PROJECT_ROOT
export TASK2_REPO="$TASK2_PROJECT_ROOT/repo"
export TASK2_ENV="$TASK2_PROJECT_ROOT/env"
export TASK2_DREAMPLACE_INSTALL="$TASK2_PROJECT_ROOT/dreamplace-install"
export HOME="$TASK2_PROJECT_ROOT/home"
export CONDA_PKGS_DIRS="$TASK2_PROJECT_ROOT/conda-pkgs"
export PIP_CACHE_DIR="$TASK2_PROJECT_ROOT/pip-cache"
export XDG_CACHE_HOME="$TASK2_PROJECT_ROOT/cache/xdg"
export TORCH_EXTENSIONS_DIR="$TASK2_PROJECT_ROOT/cache/torch-extensions"
export CUDA_CACHE_PATH="$TASK2_PROJECT_ROOT/cache/cuda"
export MPLCONFIGDIR="$TASK2_PROJECT_ROOT/cache/matplotlib"
export WANDB_DIR="$TASK2_PROJECT_ROOT/cache/wandb"
export TMPDIR="$TASK2_PROJECT_ROOT/tmp/global"
export RAY_TMPDIR="$TASK2_PROJECT_ROOT/tmp/ray-global"
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1
export CUDACXX=/usr/local/cuda-12.1/bin/nvcc
export PATH="$TASK2_ENV/bin:/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="$TASK2_ENV/lib:/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TASK2_DREAMPLACE_INSTALL:$TASK2_REPO:$TASK2_REPO/src:${PYTHONPATH:-}"
mkdir -p \
  "$HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$TMPDIR" "$RAY_TMPDIR"
