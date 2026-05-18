# TVM Prep Guide

This repository is a practical guide for loading AI models, importing them into Apache TVM, compiling them for selectable targets, and validating the exported artifacts with Python or C++ runtimes.

The repository is organized around:

- `docs/`: project orientation, devcontainer setup, and manual dependency installation.
- `notebooks/00_tvm_prep_guide.ipynb`: the main guide for using TVM in this repository.
- `examples/`: reusable Python and C++ code used by the notebook and CLI tools.
- `tvm_cpp/`: legacy working C++ runner and sample images kept for reference.

## Start Here

1. Open the repository in VS Code.
2. Run `Dev Containers: Reopen in Container`.
3. Select `TVM Prep Guide CPU` unless you specifically need the NVIDIA-based GPU image.
4. Run the setup scripts from [docs/devcontainer.md](docs/devcontainer.md).
5. Open [notebooks/00_tvm_prep_guide.ipynb](notebooks/00_tvm_prep_guide.ipynb) and run it top to bottom.
6. Use [notebooks/02_adding_models_and_raspberry_pi.ipynb](notebooks/02_adding_models_and_raspberry_pi.ipynb) when you are ready to add your own models or deploy to Raspberry Pi.
7. Use [notebooks/03_plain_tvm_api_workflow.ipynb](notebooks/03_plain_tvm_api_workflow.ipynb) when you want the same compile/export/run flow written with direct TVM APIs and no project helper abstractions.

The devcontainer builds Apache TVM from source at `v0.19.0`. This is slower than installing a pip wheel, but keeps the Python package, headers, and C++ runtime libraries aligned.

## Supported First-Pass Frontends

The notebook explains how these are used.

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

## Validation

Run the main notebook for the guided workflow. Run the validation notebook for automated compile/runtime checks:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_compile_runtime_tests.ipynb \
  --output /tmp/01_compile_runtime_tests.executed.ipynb \
  --ExecutePreprocessor.timeout=900
```

The validation notebook compiles a small deterministic vision model for `x86_64` from PyTorch and ONNX, runs the artifacts with TVM's Python graph executor, builds the C++ graph runner, and runs the ONNX artifacts from C++.

## Adding Models And Raspberry Pi Deployment

Use [notebooks/02_adding_models_and_raspberry_pi.ipynb](notebooks/02_adding_models_and_raspberry_pi.ipynb) for the detailed workflow:

- what model metadata TVM needs
- how to add reusable loaders
- how to compile ONNX/TFLite/custom models
- how to validate locally on `x86_64`
- how to compile for `raspi4_aarch64`
- how to run artifacts on Raspberry Pi with Python
- how to run artifacts on Raspberry Pi with C++

## Plain TVM API Example

Use [notebooks/03_plain_tvm_api_workflow.ipynb](notebooks/03_plain_tvm_api_workflow.ipynb) to see the workflow without `tvm_prep` helpers. It directly calls TVM APIs such as `relay.frontend.from_pytorch`, `relay.build`, `export_library`, `graph_executor.create`, and the C++ TVM runtime registry.

## Legacy Material

The old exploratory notebooks have been removed. `tvm_cpp/` remains because it contains older C++ material and sample images that are still useful as references. New work should use `docs/`, `notebooks/00_tvm_prep_guide.ipynb`, and `examples/`.
