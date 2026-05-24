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

Use `02_adding_models_and_raspberry_pi.ipynb` when you want to add your own model or deploy compiled artifacts to Raspberry Pi. It explains required model metadata, reusable loaders, ONNX/TFLite compilation, local validation, Raspberry Pi target compilation, and Python/C++ runtime execution on the target.

Only these two notebooks are maintained. Generated notebooks, scratch notebooks, and validation artifacts should stay out of git.
