# Examples

This folder contains runtime code and generated artifacts. The compilation tools live in `compilation/`. The usage walkthrough lives in `notebooks/00_tvm_prep_guide.ipynb`.

## Layout

- `python/`: Python graph-executor runtime helpers (`run_model.py`, `tvm_prep/runtime.py`, `tvm_prep/preprocess.py`).
- `cpp/`: C++ graph-executor runner for exported TVM artifacts.
- `assets/`: small sample inputs and labels used by examples.
- `artifacts/`: generated model outputs and cross-compiled runtime libraries.
  - `artifacts/<model>/<target>/` - compiled model artifacts.
  - `artifacts/runtime/<target>/libtvm_runtime.so` - cross-compiled TVM runtime (see `compilation/build_runtime.sh`).

Generated model outputs belong in `examples/artifacts/` and should stay out of git except for `.gitkeep`.

## Runtime Flow

Compile first:

```bash
python compilation/compile.py --frontend pytorch --model resnet18 --target x86_64
```

Then run the exported artifact directory:

```bash
python examples/python/run_model.py \
  --artifact-dir examples/artifacts/resnet18/x86_64 \
  --image examples/assets/cat.png
```

For C++ or target-device deployment, also build the matching TVM runtime:

```bash
bash compilation/build_runtime.sh raspi4_aarch64
```
