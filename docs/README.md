# Setup Documentation

The markdown documentation is intentionally limited to project orientation and setup. The detailed TVM workflow, including model import, target profiles, compilation, artifact export, Python runtime, and C++ runtime usage, lives in `notebooks/00_tvm_prep_guide.ipynb`.

## Setup Path

1. Open the repository in VS Code.
2. Reopen it in the devcontainer: [devcontainer.md](devcontainer.md).
3. Install Python dependencies from inside the running container:

```bash
bash .devcontainer/scripts/install-python-deps.sh requirements.txt
```

4. Build Apache TVM from source:

```bash
bash .devcontainer/scripts/build-tvm.sh
```

5. Open `notebooks/00_tvm_prep_guide.ipynb` and run it top to bottom.
6. Open `notebooks/02_adding_models_and_raspberry_pi.ipynb` when adding your own models or preparing Raspberry Pi deployment.

## Repository Layout

- `docs/`: setup and project-orientation notes.
- `notebooks/00_tvm_prep_guide.ipynb`: the maintained guide for using TVM in this repository.
- `notebooks/02_adding_models_and_raspberry_pi.ipynb`: model onboarding, target compilation, and Raspberry Pi Python/C++ deployment notes.
- `compilation/`: low-abstraction TVM model compilation script, target profiles, and runtime build script.
- `examples/python/`: Python graph-executor runtime and image preprocessing helpers.
- `examples/cpp/`: C++ graph-executor runner for exported artifacts.
- `examples/artifacts/`: generated model artifacts; ignored by git except `.gitkeep`.
- `tvm_cpp/`: legacy C++ material and sample images kept for reference.

Generated models, libraries, parameters, and ONNX exports should go under `examples/artifacts/` and are ignored by git.

## Cross-Compilation Outputs

Model compilation writes:

```text
examples/artifacts/<model>/<target>/
  model.json
  model.params
  model.so
  metadata.json
  labels.txt        # when labels are available
```

C++ deployment also needs the target TVM runtime:

```bash
bash compilation/build_runtime.sh raspi4_aarch64
```

That writes:

```text
examples/artifacts/runtime/raspi4_aarch64/libtvm_runtime.so
```
