# USAGE: move this to the SwiftTransformer directory and run it

#!/usr/bin/env bash
set -euo pipefail

# ===== Config you may tweak ===================================================
ENV_NAME="distserve"                                  # use your existing env
REPO_DIR="/home/exouser/DistServe/SwiftTransformer"   # repo path
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"  # A100=8.0, H100=9.0
JOBS="${JOBS:-$(nproc)}"

# ===== Activate conda env =====================================================
echo ">>> Activating conda env: ${ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ===== Prefer system libstdc++ (avoid old conda one) ==========================
echo ">>> Forcing system libstdc++ (bypass older conda one)"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:${LD_PRELOAD-}"
echo ">>> Checking which libstdc++ cmake will use"
ldd "$(command -v cmake)" | grep -i libstdc++

# ===== Ensure tools in this env ===============================================
echo ">>> Ensuring build tools and compilers"
# Ninja inside the env (avoids system dependency)
conda install -y -c conda-forge ninja >/dev/null
# If you don’t have sudo this is fine (we rely on system gcc-11 already present)
sudo -n true >/dev/null 2>&1 && sudo apt-get update -y && sudo apt-get install -y gcc-11 g++-11 || true

# ===== Ensure CUDA toolkit & NCCL present in env ==============================
echo ">>> Checking CUDA toolkit presence"
if [ ! -d "$CONDA_PREFIX/targets/x86_64-linux" ]; then
  echo ">>> Installing CUDA 12.8 toolkit (headers + libs) into ${ENV_NAME}..."
  conda install -y -c conda-forge "cuda-toolkit=12.8"
fi

echo ">>> Ensuring NCCL is installed"
conda install -y -c conda-forge nccl >/dev/null

# ===== Ensure OpenMPI inside this env =========================================
echo ">>> Ensuring OpenMPI is installed in ${ENV_NAME}"
conda install -y -c conda-forge openmpi >/dev/null

# ===== Compilers & MPI wrappers ===============================================
echo ">>> Setting compiler and MPI env"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export OMPI_CC=/usr/bin/gcc-11
export OMPI_CXX=/usr/bin/g++-11

# ===== CUDA paths from conda-forge layout ====================================
export CUDA_HOME="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX"
export CUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX"
export CUDA_INCLUDE_DIRS="$CONDA_PREFIX/targets/x86_64-linux/include"
export CUDA_LIB_DIR="$CONDA_PREFIX/targets/x86_64-linux/lib"

# Prefer env’s libs at runtime/link time
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_LIB_DIR:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:${CMAKE_PREFIX_PATH:-}"

# ===== Configure & build ======================================================
echo ">>> Configuring CMake in $REPO_DIR"
cd "$REPO_DIR"
rm -rf build

PY_EXE="$(which python)"

# Use system cmake (with system libstdc++) and Ninja
/usr/bin/cmake -B build -G Ninja \
  -DBUILD_MODE=RELEASE \
  -DPYTHON_PATH="${PY_EXE}" \
  -DMPI_CXX_COMPILER="$CONDA_PREFIX/bin/mpicxx" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR" \
  -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT" \
  -DCMAKE_INCLUDE_PATH="$CUDA_INCLUDE_DIRS" \
  -DCMAKE_LIBRARY_PATH="$CUDA_LIB_DIR;$CONDA_PREFIX/lib"

echo ">>> Building SwiftTransformer (core targets only)"
cmake --build build -j"${JOBS}" --target \
  st_pybinding \
  run_gpt \
  benchmark_all_input_same \
  model_gpt model_opt model_llama2 model_gpt2 \
  layer kernel xformers_kernel xformers_autogen_impl util

echo ">>> Build complete."

# ===== Post-build summary =====================================================
PYLIB="$REPO_DIR/build/lib/libst_pybinding.so"
echo "---------------------------------------------------------------------"
echo "SwiftTransformer built successfully!"
echo "Built library:"
echo "  $PYLIB"
echo
echo "Use in Python (two common ways):"
cat <<'PYTHON_SNIPPET'

# Option 1: Torch custom op
import torch, os
torch.ops.load_library(os.path.abspath("PATH/TO/SwiftTransformer/build/lib/libst_pybinding.so"))

# Option 2: Pybind11 module
import sys, os
sys.path.append(os.path.abspath("PATH/TO/SwiftTransformer/build/lib"))
import st_pybinding
print(dir(st_pybinding))
PYTHON_SNIPPET
echo
echo "Make sure runtime libs are visible (already set by this script):"
echo '  export LD_LIBRARY_PATH="'$CUDA_LIB_DIR:$CONDA_PREFIX/lib'":$LD_LIBRARY_PATH"'
echo "---------------------------------------------------------------------"
