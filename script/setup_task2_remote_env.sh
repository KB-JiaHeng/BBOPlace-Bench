#!/usr/bin/env bash
set -euo pipefail
TASK2_PROJECT_ROOT="${TASK2_PROJECT_ROOT:-$HOME/hsea26-hw5-task2}"
export TASK2_PROJECT_ROOT
source "$TASK2_PROJECT_ROOT/repo/script/task2_remote_env.sh"
LOG_DIR="$TASK2_PROJECT_ROOT/logs"
MANIFEST_DIR="$TASK2_PROJECT_ROOT/manifests"
BUILD_DIR="$TASK2_PROJECT_ROOT/build/dreamplace"
SOURCE_DIR="$TASK2_REPO/thirdparty/DREAMPlace_source"
BUILD_SOURCE_ROOT="$TASK2_PROJECT_ROOT/build-source"
BUILD_SOURCE_DIR="$BUILD_SOURCE_ROOT/DREAMPlace_source"
mkdir -p "$LOG_DIR" "$MANIFEST_DIR" "$BUILD_DIR" "$TASK2_DREAMPLACE_INSTALL"

exec > >(tee -a "$LOG_DIR/setup_environment.log") 2>&1

echo "[setup] project=$TASK2_PROJECT_ROOT"
echo "[setup] source=$SOURCE_DIR"
python3 "$TASK2_REPO/script/hash_dreamplace_source.py" \
  "$SOURCE_DIR" \
  --verify "$TASK2_REPO/experiments/task2_smoke/dreamplace_source_manifest.json"

# DREAMPlace's original top-level CMake generates an OpenTimer header inside its
# source directory. Build from a disposable byte-for-byte copy so the bundled
# source-of-record remains unchanged. The copy is removed on every exit.
rm -rf "$BUILD_SOURCE_ROOT"
mkdir -p "$BUILD_SOURCE_ROOT"
cp -a "$SOURCE_DIR" "$BUILD_SOURCE_DIR"
trap 'rm -rf "$BUILD_SOURCE_ROOT"' EXIT

if [[ ! -x "$TASK2_ENV/bin/python" ]]; then
  echo "[setup] creating a self-contained Task 2 environment from scratch"
  /home/sihengzhao/anaconda3/bin/conda create \
    --prefix "$TASK2_ENV" \
    python=3.10 \
    pip \
    "setuptools<81" \
    --yes
fi

# Build-only tools required by DREAMPlace's original top-level CMake project.
/home/sihengzhao/anaconda3/bin/conda install \
  --prefix "$TASK2_ENV" \
  bison \
  flex \
  --yes

# Install the same versions validated in the frozen Task 1 environment. The
# environment and all transient files remain inside TASK2_PROJECT_ROOT.
"$TASK2_ENV/bin/python" -m pip install --no-cache-dir \
  -r "$TASK2_REPO/experiments/remote_requirements.txt" \
  "pytest==8.3.5" \
  "shapely==2.0.6" \
  "cairocffi==1.6.1" \
  "gdown==6.1.0"
"$TASK2_ENV/bin/python" -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.1.0"
"$TASK2_ENV/bin/python" -m pip install --no-cache-dir "ninja>=1.10"

# Release Conda's Task-2-owned package cache before the out-of-source build.
rm -rf "$PIP_CACHE_DIR"/* "$CONDA_PKGS_DIRS"/*

"$TASK2_ENV/bin/python" - <<'PY' | tee "$MANIFEST_DIR/python_environment.txt"
import platform
import torch
import numpy
import pymoo
import ray
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("torch_cuda_available", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("pymoo", pymoo.__version__)
print("ray", ray.__version__)
print("torch_cxx11_abi", int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY

"$TASK2_ENV/bin/python" -m pip freeze > "$MANIFEST_DIR/pip_freeze.txt"
"$TASK2_ENV/bin/python" -m pip check | tee "$MANIFEST_DIR/pip_check.txt"
/usr/local/cuda-12.1/bin/nvcc --version > "$MANIFEST_DIR/nvcc_version.txt"
gcc --version > "$MANIFEST_DIR/gcc_version.txt"
cmake --version > "$MANIFEST_DIR/cmake_version.txt"
nvidia-smi -q > "$MANIFEST_DIR/nvidia_smi_q.txt"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
ABI="$($TASK2_ENV/bin/python - <<'PY'
import torch
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)"

echo "[build] configuring unmodified DREAMPlace, ABI=$ABI"
cmake -S "$BUILD_SOURCE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$TASK2_PROJECT_ROOT/dreamplace-cmake-install" \
  -DPython_EXECUTABLE="$TASK2_ENV/bin/python" \
  -DCMAKE_CXX_ABI="$ABI" \
  -DFLEX_INCLUDE_DIR="$TASK2_ENV/include" \
  -DFLEX_LIBRARY="$TASK2_ENV/lib/libfl.so" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 \
  -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-12.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=8.9 \
  -DCUDA_ARCH_BIN=8.9 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  2>&1 | tee "$LOG_DIR/dreamplace_configure.log"

echo "[build] building original rudy_cpp and rudy_cuda targets"
cmake --build "$BUILD_DIR" --target rudy_cpp rudy_cuda --parallel 16 \
  2>&1 | tee "$LOG_DIR/dreamplace_rudy_build.log"

rm -rf "$TASK2_DREAMPLACE_INSTALL/dreamplace"
mkdir -p "$TASK2_DREAMPLACE_INSTALL"
cp -a "$SOURCE_DIR/dreamplace" "$TASK2_DREAMPLACE_INSTALL/dreamplace"
rm -rf \
  "$TASK2_DREAMPLACE_INSTALL/dreamplace"/**/__pycache__ \
  2>/dev/null || true
cp "$BUILD_DIR/dreamplace/configure.py" \
  "$TASK2_DREAMPLACE_INSTALL/dreamplace/configure.py"
find "$BUILD_DIR/dreamplace/ops/rudy" -maxdepth 1 -type f \
  \( -name 'rudy_cpp*.so' -o -name 'rudy_cuda*.so' \) \
  -exec cp -v {} "$TASK2_DREAMPLACE_INSTALL/dreamplace/ops/rudy/" \;

CPU_SO_COUNT=$(find "$TASK2_DREAMPLACE_INSTALL/dreamplace/ops/rudy" -name 'rudy_cpp*.so' | wc -l)
CUDA_SO_COUNT=$(find "$TASK2_DREAMPLACE_INSTALL/dreamplace/ops/rudy" -name 'rudy_cuda*.so' | wc -l)
if [[ "$CPU_SO_COUNT" -ne 1 || "$CUDA_SO_COUNT" -ne 1 ]]; then
  echo "Expected exactly one CPU and one CUDA RUDY module, got $CPU_SO_COUNT/$CUDA_SO_COUNT" >&2
  exit 1
fi

python3 "$TASK2_REPO/script/hash_dreamplace_source.py" \
  "$SOURCE_DIR" \
  --verify "$TASK2_REPO/experiments/task2_smoke/dreamplace_source_manifest.json"

TASK2_REQUIRE_DREAMPLACE=1 TASK2_RUDY_BACKEND=cpu \
  "$TASK2_ENV/bin/python" -m pytest -q \
  "$TASK2_REPO/tests/test_task2_dreamplace_kernel.py" \
  2>&1 | tee "$LOG_DIR/kernel_cpu_test.log"
TASK2_REQUIRE_DREAMPLACE=1 TASK2_RUDY_BACKEND=cuda CUDA_VISIBLE_DEVICES=0 \
  "$TASK2_ENV/bin/python" -m pytest -q \
  "$TASK2_REPO/tests/test_task2_dreamplace_kernel.py" \
  2>&1 | tee "$LOG_DIR/kernel_cuda_test.log"

echo "[setup] complete"
