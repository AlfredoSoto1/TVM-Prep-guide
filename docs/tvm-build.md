# Apache TVM Build

The devcontainers do not build Apache TVM during image creation. Open the devcontainer first, then run the build manually from inside the running container when you are ready for the full toolchain.

The shared build script is `.devcontainer/scripts/build-tvm.sh`.

Important environment variables:

```bash
TVM_VERSION=v0.19.0
TVM_HOME=/opt/tvm
TVM_BUILD=/opt/tvm/build
PYTHONPATH=/opt/tvm/python:$PYTHONPATH
LD_LIBRARY_PATH=/opt/tvm/build:$LD_LIBRARY_PATH
```

Enabled by default for the CPU build:

- LLVM code generation.
- TVM RPC.
- Graph executor.
- AOT executor.
- Profiler.
- C++ runtime.

The CPU container defaults to `TVM_USE_VULKAN=OFF` because Vulkan pulls in extra shader/SPIR-V headers and has been a common source of CMake configuration failures. Enable it manually only when needed:

```bash
TVM_USE_VULKAN=ON bash .devcontainer/scripts/build-tvm.sh
```

The GPU container sets `TVM_USE_CUDA=ON` and `TVM_USE_VULKAN=ON`, but CUDA workflows are not expanded in this guide.

To build TVM inside the running container:

```bash
bash .devcontainer/scripts/build-tvm.sh
```

To inspect the local TVM configuration inside a running devcontainer:

```bash
python -c "import tvm; print(tvm.__version__); print(tvm.support.libinfo())"
```
