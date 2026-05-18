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
