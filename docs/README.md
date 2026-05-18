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
7. Open `notebooks/03_plain_tvm_api_workflow.ipynb` when you want to see the same workflow using direct TVM APIs without project helper abstractions.

To run the automated validation notebook:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_compile_runtime_tests.ipynb \
  --output /tmp/01_compile_runtime_tests.executed.ipynb \
  --ExecutePreprocessor.timeout=900
```

## Repository Layout

- `docs/`: setup and project-orientation notes.
- `notebooks/00_tvm_prep_guide.ipynb`: the maintained guide for using TVM in this repository.
- `notebooks/02_adding_models_and_raspberry_pi.ipynb`: model onboarding, target compilation, and Raspberry Pi Python/C++ deployment notes.
- `notebooks/03_plain_tvm_api_workflow.ipynb`: direct TVM API compile/export/run example with Python and C++ runtimes.
- `examples/python/`: reusable model loading, compilation, target, preprocessing, and runtime helpers.
- `examples/cpp/`: C++ graph-executor runner for exported artifacts.
- `examples/artifacts/`: generated model artifacts; ignored by git except `.gitkeep`.
- `tvm_cpp/`: legacy C++ material and sample images kept for reference.

Generated models, libraries, parameters, and ONNX exports should go under `examples/artifacts/` and are ignored by git.
