# Devcontainer Setup

The repository provides two VS Code devcontainers.

## CPU Container

Use `.devcontainer/cpu/devcontainer.json` by default. It is the portable setup for Windows, macOS, and Linux hosts.

It installs the baseline operating-system tools:

- Python.
- LLVM/Clang, CMake, Ninja, and native build tools.
- ARMv7 and AArch64 Linux cross-compilers.
- Emscripten/WebAssembly tools.
- Vulkan headers/tools.

It does not install the heavy Python framework dependencies or build Apache TVM during image creation.

## GPU Container

Use `.devcontainer/gpu/devcontainer.json` only on hosts with the NVIDIA Container Toolkit configured. It starts from an NVIDIA CUDA development image and includes the native build tools needed to build TVM with CUDA and Vulkan support after the container is running.

CUDA model compilation is not the focus of this guide. Prefer the `vulkan` target profile for GPU-style examples when the host supports Vulkan.

## Open In VS Code

1. Install Docker Desktop or Docker Engine.
2. Install the VS Code Dev Containers extension.
3. Open this repository folder in VS Code.
4. Run `Dev Containers: Reopen in Container`.
5. Select `TVM Prep Guide CPU` unless you specifically need the GPU image.

The container should open before any heavy software build or framework install starts.

## Manual Setup Inside The Container

After VS Code is connected to the running devcontainer, install Python dependencies only when you need the notebooks and examples:

```bash
bash .devcontainer/scripts/install-python-deps.sh requirements.txt
```

Build TVM from source only when you are ready for the full TVM toolchain:

```bash
bash .devcontainer/scripts/build-tvm.sh
```

The CPU container defaults to `TVM_USE_VULKAN=OFF` to keep the first build path conservative. If you specifically need Vulkan support, run:

```bash
TVM_USE_VULKAN=ON bash .devcontainer/scripts/build-tvm.sh
```

Do not also install `apache-tvm` from pip inside the container; the source build must match the C++ runtime libraries.
