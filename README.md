# TVM Prep Guide

This repository is a practical guide for loading AI models, importing them into Apache TVM, compiling them for selectable targets, and validating the exported artifacts with Python or C++ runtimes.

The primary workflow is now organized around:

- `docs/`: setup, TVM build, target selection, cross-compilation, and deployment documentation.
- `notebooks/`: step-by-step Jupyter notebooks for the learning path.
- `examples/`: reusable Python and C++ code used by the notebooks.
- `tvm_cpp/`: legacy working C++ runner and sample images kept for reference.

## Start Here

1. Open the repository in VS Code.
2. Run `Dev Containers: Reopen in Container`.
3. Select `TVM Prep Guide CPU` unless you specifically need the NVIDIA-based GPU image.
4. Open [docs/README.md](docs/README.md) and follow the notebook sequence.

The devcontainer builds Apache TVM from source at `v0.19.0`. This is slower than installing a pip wheel, but keeps the Python package, headers, and C++ runtime libraries aligned.

## Quick Example

Compile a PyTorch ResNet18 model for native x86_64 Linux:

```bash
python examples/python/compile_model.py \
  --frontend pytorch \
  --model resnet18 \
  --target-profile x86_64
```

Run the exported artifacts with Python:

```bash
python examples/python/run_model.py \
  --artifact-dir examples/artifacts/resnet18/x86_64 \
  --image tvm_cpp/images/cat.png
```

Build and run the C++ graph executor example:

```bash
cmake -S examples/cpp -B examples/cpp/build -G Ninja
cmake --build examples/cpp/build
examples/cpp/build/run_tvm_graph \
  --artifact-dir examples/artifacts/resnet18/x86_64 \
  --image tvm_cpp/images/cat.png
```

## Supported First-Pass Frontends

- PyTorch torchvision models.
- TensorFlow/Keras application models.
- ONNX model files.
- TFLite model files.

## Supported First-Pass Targets

Target profiles are defined in `examples/python/tvm_prep/targets.py`.

- `native`
- `x86_64`
- `raspi4_aarch64`
- `raspi_armv7`
- `wasm32`
- `vulkan`
- `c`

CUDA is not expanded in the current guide. Prefer Vulkan for GPU-oriented examples where possible.

## Legacy Material

Older notebooks and `tvm_cpp/` remain in the repository because they contain working historical examples. New work should use `docs/`, `notebooks/00_*` through `05_*`, and `examples/`.
