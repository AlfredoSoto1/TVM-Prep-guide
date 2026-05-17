# Deployment

Compiled artifacts are written to:

```text
examples/artifacts/<model>/<target_profile>/
```

Expected files:

- `model.json`: TVM graph executor graph.
- `model.params`: model parameters.
- `model.so`, `model.tar`, or `model.wasm`: compiled target library.
- `metadata.json`: input name, shape, target profile, and library filename.

## Manual Device Transfer

For an embedded target, manually copy:

- The artifact directory.
- The C++ runner executable built for that target.
- Input assets such as test images and labels.
- Required TVM runtime libraries if using dynamic linking.

## Runtime Options

Use Python runtime when Python and TVM are installed on the target.

Use C++ runtime when you want a smaller target-side executable and direct control over linking. For embedded Linux, this is usually the more realistic deployment path.
