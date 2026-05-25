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
# The compiled library is written to:
#   examples/artifacts/runtime/<target>/libtvm_runtime.so
#
# Copy this file alongside the compiled model artifacts when deploying to the
# target device or linking the C++ runner.
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
  raspi4_aarch64)
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

echo ""
echo "Done. Runtime library written to:"
echo "  ${OUT_DIR}/libtvm_runtime.so"
echo ""
echo "Copy this file to the target device alongside the compiled model artifacts."
