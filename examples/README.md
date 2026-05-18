# Working Examples

This folder contains reusable code behind the guided notebook and CLI workflows. The usage walkthrough lives in `notebooks/00_tvm_prep_guide.ipynb`.

## Layout

- `python/`: model loading, TVM import, compilation, artifact export, and Python runtime validation.
- `cpp/`: C++ graph-executor runner for exported TVM artifacts.
- `targets/`: target profile definitions used by the notebook and examples.
- `assets/`: small sample inputs and labels used by examples.
- `artifacts/`: generated model outputs. This folder is intentionally ignored by git.

Generated model outputs belong in `examples/artifacts/` and should stay out of git except for `.gitkeep`.
