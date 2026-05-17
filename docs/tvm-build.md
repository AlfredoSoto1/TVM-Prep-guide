# Apache TVM Build

The devcontainers build Apache TVM from source using tag `v0.19.0`.

The shared build script is `.devcontainer/scripts/build-tvm.sh`.

Important environment variables:

```bash
TVM_VERSION=v0.19.0
TVM_HOME=/opt/tvm
TVM_BUILD=/opt/tvm/build
PYTHONPATH=/opt/tvm/python:$PYTHONPATH
LD_LIBRARY_PATH=/opt/tvm/build:$LD_LIBRARY_PATH
```

Enabled by default in the CPU image:

- LLVM code generation.
- TVM RPC.
- Graph executor.
- AOT executor.
- Profiler.
- C++ runtime.
- Vulkan support.

The GPU image additionally sets `TVM_USE_CUDA=ON`, but CUDA workflows are not expanded in this guide.

To inspect the local TVM configuration inside a running devcontainer:

```bash
python -c "import tvm; print(tvm.__version__); print(tvm.support.libinfo())"
```
