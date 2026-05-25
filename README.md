# TVM Prep Guide

This repository is a practical guide for loading AI models, importing them into Apache TVM, compiling them for selectable architectures, and running the exported artifacts with Python or C++ runtimes.

The repository is organized around:

- `docs/`: project orientation, devcontainer setup, and manual dependency installation.
- `notebooks/00_tvm_prep_guide.ipynb`: the main TVM preparation guide.
- `notebooks/02_adding_models_and_raspberry_pi.ipynb`: adding models and deploying to Raspberry Pi.
- `compilation/`: flat TVM compilation scripts and target profiles.
- `examples/`: Python and C++ runtime examples plus generated artifacts.
- `tvm_cpp/`: legacy C++ material kept for reference.

## Start Here

1. Open the repository in VS Code.
2. Run `Dev Containers: Reopen in Container`.
3. Select `TVM Prep Guide CPU` unless you specifically need the NVIDIA-based GPU image.
4. Run the setup scripts from [docs/devcontainer.md](docs/devcontainer.md).
5. Open [notebooks/00_tvm_prep_guide.ipynb](notebooks/00_tvm_prep_guide.ipynb) and run it top to bottom.
6. Use [notebooks/02_adding_models_and_raspberry_pi.ipynb](notebooks/02_adding_models_and_raspberry_pi.ipynb) when you are ready to add your own models or deploy to Raspberry Pi.

The devcontainer builds Apache TVM from source at `v0.19.0`. This is slower than installing a pip wheel, but keeps the Python package, headers, and C++ runtime libraries aligned.

## Supported First-Pass Frontends

The notebook explains how these are used.

- PyTorch torchvision models.
- TensorFlow/Keras application models.
- ONNX model files.
- TFLite model files.

## Supported First-Pass Targets

Target profiles are plain dictionaries in `compilation/targets.py`.

- `native`
- `x86_64`
- `raspi4_aarch64`
- `raspi_armv7`
- `wasm32`
- `vulkan`
- `c`

CUDA is not expanded in the current guide. Prefer Vulkan for GPU-oriented examples where possible.

## Core Workflow

The repository keeps the TVM path direct:

1. Select or load a model in Python, usually from a notebook or `compilation/compile.py`.
2. Import the model into Relay with the appropriate TVM frontend.
3. Choose a target architecture from `compilation/targets.py`.
4. Compile and export graph-executor artifacts under `examples/artifacts/<model>/<target>/`.
5. For target-device deployment, build the matching TVM runtime package with `compilation/build_runtime.sh`; the output is `examples/artifacts/runtime/<target>/`.
6. Copy the model artifact directory and runtime package directory to the target device, then run with either the Python or C++ example.

Local Python validation example:

```bash
python compilation/compile.py --frontend pytorch --model resnet18 --target x86_64
python examples/python/run_model.py \
  --artifact-dir examples/artifacts/resnet18/x86_64 \
  --image examples/assets/cat.png
```

Raspberry Pi AArch64 compile example:

```bash
python compilation/compile.py --frontend pytorch --model resnet18 --target raspi4_aarch64
bash compilation/build_runtime.sh raspi4_aarch64
```

## Adding Models And Raspberry Pi Deployment

Use [notebooks/02_adding_models_and_raspberry_pi.ipynb](notebooks/02_adding_models_and_raspberry_pi.ipynb) for the detailed workflow:

- what model metadata TVM needs
- how to add reusable loaders
- how to compile ONNX/TFLite/custom models
- how to validate locally on `x86_64`
- how to compile for `raspi4_aarch64`
- how to run artifacts on Raspberry Pi with Python
- how to run artifacts on Raspberry Pi with C++

## Legacy Material

The old exploratory notebooks have been removed. `tvm_cpp/` remains only as older C++ reference material. New work should use `docs/`, the two maintained notebooks, `compilation/`, and `examples/`.
