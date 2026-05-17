# C++ Runtime

Build the C++ runner inside a devcontainer:

```bash
cmake -S examples/cpp -B examples/cpp/build -G Ninja
cmake --build examples/cpp/build
```

Run an x86_64 artifact:

```bash
examples/cpp/build/run_tvm_graph \
  --artifact-dir examples/artifacts/resnet18/x86_64 \
  --image tvm_cpp/images/cat.png
```

## Cross-Compiling The Runner

For Raspberry Pi 4 64-bit, configure CMake with an AArch64 compiler:

```bash
cmake -S examples/cpp -B examples/cpp/build-aarch64 -G Ninja \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DTVM_HOME=/opt/tvm \
  -DTVM_BUILD=/opt/tvm/build
cmake --build examples/cpp/build-aarch64
```

For production deployment, the C++ runner, TVM runtime library, and compiled model library must all match the target architecture and target OS ABI.
