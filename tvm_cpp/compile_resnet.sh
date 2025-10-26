#!/usr/bin/env bash
set -e

TVM_HOME=${TVM_HOME:-/workspaces/TVM-Prep-guide/tvm}
TVM_BUILD=${TVM_BUILD:-/workspaces/TVM-Prep-guide/tvm/build}

g++ -std=gnu++17 run_resnet.cpp \
  -I"$TVM_HOME/include" \
  -I"$TVM_HOME/3rdparty/dlpack/include" \
  -I"$TVM_HOME/3rdparty/dmlc-core/include" \
  -L"$TVM_BUILD" -ltvm_runtime \
  -Wl,-rpath,"$TVM_BUILD" \
  -O2 -o run_resnet
