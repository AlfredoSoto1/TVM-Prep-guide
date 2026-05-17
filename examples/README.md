# Working Examples

This folder contains reusable code behind the notebooks.

## Layout

- `python/`: model loading, TVM import, compilation, artifact export, and Python runtime validation.
- `cpp/`: C++ graph-executor runner for exported TVM artifacts.
- `targets/`: target profile definitions shared by docs and examples.
- `assets/`: small sample inputs and labels used by examples.
- `artifacts/`: generated model outputs. This folder is intentionally ignored by git.

## Typical Flow

```bash
python examples/python/compile_model.py \
  --model resnet18 \
  --frontend pytorch \
  --target-profile x86_64 \
  --output-dir examples/artifacts

python examples/python/run_model.py \
  --artifact-dir examples/artifacts/resnet18/x86_64 \
  --image tvm_cpp/images/cat.png
```

The C++ runner consumes the same artifact directory:

```bash
cmake -S examples/cpp -B examples/cpp/build -G Ninja
cmake --build examples/cpp/build
examples/cpp/build/run_tvm_graph \
  --artifact-dir examples/artifacts/resnet18/x86_64 \
  --image tvm_cpp/images/cat.png
```
