# Python Runtime

Run compiled graph artifacts with Python:

```bash
python examples/python/run_model.py \
  --artifact-dir examples/artifacts/resnet18/x86_64 \
  --image tvm_cpp/images/cat.png
```

The runtime reads `metadata.json` to find:

- Input name.
- Input shape.
- Compiled library filename.

The default image preprocessing matches ImageNet-style NCHW PyTorch models. For TensorFlow/Keras models, pass `--layout NHWC` if the metadata shape does not make that obvious.
