# Model Frontends

TVM supports many import paths. This guide focuses on four practical paths.

## PyTorch

Use TorchScript tracing for pretrained torchvision models:

```bash
python examples/python/compile_model.py \
  --frontend pytorch \
  --model resnet18 \
  --target-profile x86_64
```

Supported starter models in `examples/python/tvm_prep/model_zoo.py`:

- `resnet18`
- `mobilenet_v2`
- `squeezenet1_1`
- `shufflenet_v2_x1_0`

## TensorFlow / Keras

Use Keras applications for initial validation:

```bash
python examples/python/compile_model.py \
  --frontend tensorflow \
  --model mobilenet_v2 \
  --target-profile x86_64
```

## ONNX

Use ONNX when the model can be exported from another framework:

```bash
python examples/python/compile_model.py \
  --frontend onnx \
  --model-path path/to/model.onnx \
  --input-name input0 \
  --input-shape 1,3,224,224 \
  --target-profile x86_64
```

## TFLite

Use TFLite for mobile/embedded TensorFlow models:

```bash
python examples/python/compile_model.py \
  --frontend tflite \
  --model-path path/to/model.tflite \
  --input-name input \
  --input-shape 1,224,224,3 \
  --target-profile x86_64
```

The exact input name and shape must match the model. Inspect them before compiling.
