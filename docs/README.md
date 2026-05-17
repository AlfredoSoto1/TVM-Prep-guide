# TVM Prep Guide Documentation

This documentation is the primary path for using the repository.

## Learning Path

1. Open the repository in a devcontainer: [devcontainer.md](devcontainer.md).
2. Confirm TVM and framework dependencies: `notebooks/00_environment_check.ipynb`.
3. Load representative models: `notebooks/01_load_models.ipynb`.
4. Import models into TVM Relay: `notebooks/02_import_to_tvm.ipynb`.
5. Compile for a target profile: `notebooks/03_compile_for_targets.ipynb`.
6. Export portable artifacts: `notebooks/04_export_artifacts.ipynb`.
7. Validate artifacts with Python and C++: [python-runtime.md](python-runtime.md) and [cpp-runtime.md](cpp-runtime.md).

## Repository Layout

- `docs/`: step-by-step documentation and target/deployment notes.
- `notebooks/`: notebook curriculum for model loading, TVM import, compilation, and validation.
- `examples/`: reusable Python and C++ code used by the notebooks.
- `tvm_cpp/`: legacy working C++ examples kept for reference.

Generated models, libraries, parameters, and ONNX exports should go under `examples/artifacts/` and are ignored by git.
