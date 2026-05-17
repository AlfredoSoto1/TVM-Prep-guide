# Devcontainer Setup

The repository provides two VS Code devcontainers.

## CPU Container

Use `.devcontainer/cpu/devcontainer.json` by default. It is the portable setup for Windows, macOS, and Linux hosts.

It installs:

- Python 3.11 and Jupyter.
- PyTorch, TensorFlow, ONNX, TFLite helpers, Pillow, NumPy, and test tools.
- LLVM/Clang, CMake, Ninja, and native build tools.
- ARMv7 and AArch64 Linux cross-compilers.
- Emscripten/WebAssembly tools.
- Vulkan headers/tools.
- Apache TVM built from source at `v0.19.0`.

## GPU Container

Use `.devcontainer/gpu/devcontainer.json` only on hosts with the NVIDIA Container Toolkit configured. It starts from an NVIDIA CUDA development image and builds TVM with CUDA and Vulkan support.

CUDA model compilation is not the focus of this guide. Prefer the `vulkan` target profile for GPU-style examples when the host supports Vulkan.

## Open In VS Code

1. Install Docker Desktop or Docker Engine.
2. Install the VS Code Dev Containers extension.
3. Open this repository folder in VS Code.
4. Run `Dev Containers: Reopen in Container`.
5. Select `TVM Prep Guide CPU` unless you specifically need the GPU image.

The devcontainer build is intentionally heavy because it builds TVM from source. Do not also install `apache-tvm` from pip inside the container; the source build must match the C++ runtime libraries.
