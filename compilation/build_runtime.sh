#!/usr/bin/env bash
# Build libtvm_runtime.so for a target architecture.
#
# The runtime library is the only TVM component that must run on the target
# device. It does not include the compiler, LLVM, or Relay. It is small (~2 MB)
# and loads pre-compiled model.so files at runtime.
#
# Usage:
#   bash compilation/build_runtime.sh raspi4_aarch64
#   bash compilation/build_runtime.sh raspi_armv7
#   bash compilation/build_runtime.sh x86_64
#
# The runtime package is written to:
#   examples/artifacts/runtime/<target>/
#
# Copy this directory alongside the compiled model artifacts when deploying to
# the target device. It contains:
#   libtvm_runtime.so  - runtime shared library for the target
#   include/           - headers needed to build examples/cpp on the target
#   python/            - TVM Python package used by examples/python
#
# Prerequisites:
#   - build-tvm.sh has already been run (TVM source is at TVM_HOME).
#   - Cross-compilers for the target are on PATH (installed by the Dockerfile).

set -euo pipefail

TARGET="${1:-x86_64}"
TVM_HOME="${TVM_HOME:-/opt/tvm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/examples/artifacts/runtime/${TARGET}"
BUILD_DIR="${TVM_HOME}/build-runtime-${TARGET}"

case "${TARGET}" in
  x86_64|native)
    CC="gcc"
    CXX="g++"
    CMAKE_SYSTEM_ARGS=()
    ;;
  raspi4_aarch64|raspi5_aarch64)
    CC="aarch64-linux-gnu-gcc"
    CXX="aarch64-linux-gnu-g++"
    CMAKE_SYSTEM_ARGS=(
      -DCMAKE_SYSTEM_NAME=Linux
      -DCMAKE_SYSTEM_PROCESSOR=aarch64
    )
    ;;
  raspi_armv7)
    CC="arm-linux-gnueabihf-gcc"
    CXX="arm-linux-gnueabihf-g++"
    CMAKE_SYSTEM_ARGS=(
      -DCMAKE_SYSTEM_NAME=Linux
      -DCMAKE_SYSTEM_PROCESSOR=arm
    )
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    echo "Known runtime targets: x86_64, native, raspi4_aarch64, raspi_armv7" >&2
    exit 1
    ;;
esac

if [ ! -d "${TVM_HOME}" ]; then
  echo "TVM_HOME does not exist: ${TVM_HOME}" >&2
  echo "Run .devcontainer/scripts/build-tvm.sh first, or set TVM_HOME." >&2
  exit 1
fi

if ! command -v "${CC}" >/dev/null 2>&1; then
  echo "C compiler not found on PATH: ${CC}" >&2
  exit 1
fi

if ! command -v "${CXX}" >/dev/null 2>&1; then
  echo "C++ compiler not found on PATH: ${CXX}" >&2
  exit 1
fi

echo "Building libtvm_runtime.so for ${TARGET} ..."
echo "  TVM source : ${TVM_HOME}"
echo "  Build dir  : ${BUILD_DIR}"
echo "  Output     : ${OUT_DIR}/libtvm_runtime.so"

mkdir -p "${BUILD_DIR}"
rm -f "${BUILD_DIR}/CMakeCache.txt"
rm -rf "${BUILD_DIR}/CMakeFiles"
cp "${TVM_HOME}/cmake/config.cmake" "${BUILD_DIR}/config.cmake"

cat >> "${BUILD_DIR}/config.cmake" <<EOF
set(CMAKE_BUILD_TYPE Release)
set(USE_LLVM OFF)
set(USE_RPC OFF)
set(USE_CPP_RPC OFF)
set(USE_GRAPH_EXECUTOR ON)
set(USE_AOT_EXECUTOR ON)
set(USE_PROFILER OFF)
set(USE_LIBBACKTRACE OFF)
set(BACKTRACE_ON_SEGFAULT OFF)
EOF

cmake -S "${TVM_HOME}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  "${CMAKE_SYSTEM_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel "$(nproc)" --target tvm_runtime

mkdir -p "${OUT_DIR}"
cp "${BUILD_DIR}/libtvm_runtime.so" "${OUT_DIR}/libtvm_runtime.so"

echo "Packaging C++ headers ..."
rm -rf "${OUT_DIR}/include"
mkdir -p "${OUT_DIR}/include"
cp -R "${TVM_HOME}/include/tvm" "${OUT_DIR}/include/"
cp -R "${TVM_HOME}/3rdparty/dlpack/include/dlpack" "${OUT_DIR}/include/"
cp -R "${TVM_HOME}/3rdparty/dmlc-core/include/dmlc" "${OUT_DIR}/include/"

echo "Packaging TVM Python runtime package ..."
rm -rf "${OUT_DIR}/python"
mkdir -p "${OUT_DIR}/python"
cp -R "${TVM_HOME}/python/tvm" "${OUT_DIR}/python/"
if [ -d "${TVM_HOME}/python/tvm.egg-info" ]; then
  cp -R "${TVM_HOME}/python/tvm.egg-info" "${OUT_DIR}/python/"
fi
find "${OUT_DIR}/python" -type d -name "__pycache__" -prune -exec rm -rf {} +

echo ""
echo "Done. Runtime package written to:"
echo "  ${OUT_DIR}"
echo ""
echo "Copy this directory to the target device alongside the compiled model artifacts."
