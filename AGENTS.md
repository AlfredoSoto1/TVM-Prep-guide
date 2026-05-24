# Repository Guidelines

## Project Structure & Module Organization

This repository is a practical Apache TVM preparation guide. Use `docs/` for setup and project-orientation notes. Use `notebooks/00_tvm_prep_guide.ipynb` for the detailed learning path and usage workflow.

Compilation tools live under `compilation/`:
- `compilation/targets.py` — plain dict of TVM target profiles.
- `compilation/compile.py` — flat compilation script and importable `build_and_save` function (no heavy abstractions; mirrors the TVM documentation style).
- `compilation/build_runtime.sh` — cross-compiles `libtvm_runtime.so` for a target architecture.

Runtime and deployment code lives under `examples/`:
- `examples/python/` — Python graph-executor runner (`run_model.py`) and helpers (`tvm_prep/runtime.py`, `tvm_prep/preprocess.py`).
- `examples/cpp/` — C++ graph-executor runner.
- `examples/assets/` — sample images and label files.
- `examples/artifacts/` — generated model outputs (git-ignored except `.gitkeep`). Cross-compiled runtime libraries land in `examples/artifacts/runtime/<target>/`.

Legacy examples are kept in `tvm_cpp/`; prefer `docs/`, `notebooks/00_tvm_prep_guide.ipynb`, `compilation/`, and `examples/` for new work.

## Build, Test, and Development Commands

Open the devcontainer first, then install heavy software manually:

```bash
bash .devcontainer/scripts/install-python-deps.sh requirements.txt
bash .devcontainer/scripts/build-tvm.sh
```

Compile a sample model:

```bash
python compilation/compile.py --frontend pytorch --model resnet18 --target x86_64
```

Run exported artifacts with Python:

```bash
python examples/python/run_model.py --artifact-dir examples/artifacts/resnet18/x86_64 --image tvm_cpp/images/cat.png
```

Cross-compile the TVM runtime for a target device:

```bash
bash compilation/build_runtime.sh raspi4_aarch64
```

Build the C++ runner:

```bash
cmake -S examples/cpp -B examples/cpp/build -G Ninja
cmake --build examples/cpp/build
```

Run tests, where present, with `pytest`.

## Coding Style & Naming Conventions

Write Python with 4-space indentation, type-aware interfaces where useful, and `snake_case` for modules, functions, and variables. Keep CLI scripts small and delegate logic to `examples/python/tvm_prep/`. Use C++17 for C++ examples and keep CMake changes local to the relevant example. Keep the maintained notebook at `notebooks/00_tvm_prep_guide.ipynb`.

## Testing Guidelines

Prefer focused tests for reusable Python modules and smoke checks for CLI workflows. Validate environment changes with `notebooks/00_tvm_prep_guide.ipynb`. For TVM output changes, compile at least one small model and run it through Python runtime validation. Do not commit generated `.so`, `.params`, `.json`, ONNX, or artifact directories.

## Commit & Pull Request Guidelines

Recent history uses short descriptive commit messages, but no strict convention is enforced. Prefer concise imperative messages, for example `Add target profile docs` or `Fix C++ runtime path`. Pull requests should explain the workflow affected, list commands or notebooks used for validation, and call out whether TVM, devcontainer, or artifact-generation behavior changed.

## Agent-Specific Instructions

Keep devcontainer creation separate from heavy TVM and framework installation. Do not add automatic TVM builds or large dependency installs to Docker image creation or post-create hooks.
