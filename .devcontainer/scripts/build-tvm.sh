#!/usr/bin/env bash
set -euo pipefail

TVM_VERSION="${TVM_VERSION:-v0.19.0}"
TVM_HOME="${TVM_HOME:-/opt/tvm}"
TVM_BUILD="${TVM_BUILD:-${TVM_HOME}/build}"
TVM_USE_CUDA="${TVM_USE_CUDA:-OFF}"
TVM_USE_VULKAN="${TVM_USE_VULKAN:-ON}"
LLVM_CONFIG="${LLVM_CONFIG:-/usr/bin/llvm-config-16}"

if [ ! -d "${TVM_HOME}/.git" ]; then
  git clone --recursive https://github.com/apache/tvm.git "${TVM_HOME}"
fi

cd "${TVM_HOME}"
git fetch --tags
git checkout "${TVM_VERSION}"
git submodule sync --recursive
git submodule update --init --recursive --jobs "$(nproc)"

mkdir -p "${TVM_BUILD}"
cp cmake/config.cmake "${TVM_BUILD}/config.cmake"
cat >> "${TVM_BUILD}/config.cmake" <<EOF
set(CMAKE_BUILD_TYPE Release)
set(USE_LLVM ${LLVM_CONFIG})
set(USE_RPC ON)
set(USE_GRAPH_EXECUTOR ON)
set(USE_AOT_EXECUTOR ON)
set(USE_PROFILER ON)
set(USE_VULKAN ${TVM_USE_VULKAN})
set(USE_CUDA ${TVM_USE_CUDA})
set(USE_CUDNN OFF)
set(USE_CUBLAS OFF)
set(USE_OPENCL OFF)
set(USE_METAL OFF)
set(BUILD_CPP_RUNTIME ON)
EOF

cmake -S "${TVM_HOME}" -B "${TVM_BUILD}" -G Ninja
cmake --build "${TVM_BUILD}" --parallel "$(nproc)"
python -m pip install -e "${TVM_HOME}/python"
