# Notebooks

Start with `00_tvm_prep_guide.ipynb`. It is the maintained guided path for:

- environment validation
- TVM and framework import checks
- target profile inspection
- Relay import
- TVM compilation and artifact export
- Python graph-executor validation
- real model CLI workflows
- cross-target compilation
- C++ graph-executor usage

Use `01_compile_runtime_tests.ipynb` as the executable validation notebook. It compiles a deterministic image classifier from PyTorch and ONNX, runs the compiled artifacts with Python, builds the C++ graph runner, and runs the ONNX artifacts from C++.

Use `02_adding_models_and_raspberry_pi.ipynb` when you want to add your own model or deploy compiled artifacts to Raspberry Pi. It explains required model metadata, reusable loaders, ONNX/TFLite compilation, local validation, Raspberry Pi target compilation, and Python/C++ runtime execution on the target.

Use `03_plain_tvm_api_workflow.ipynb` when you want to see the same compile/export/run workflow without the project helper abstractions. It uses direct TVM Python APIs and a minimal generated C++ runtime program.
