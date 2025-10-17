import torch
import torchvision

# 1) Load a pretrained model (already trained; we're not training here)
model = torchvision.models.resnet18(weights=True).eval()

# 2) Dummy input shape must match the model's expectation
dummy_input = torch.randn(1, 3, 224, 224)

# 3) Export to ONNX: this freezes the graph + weights into a portable file
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    input_names=["input"],       # name used later in TVM runtime
    output_names=["output"],
    opset_version=13,            # operator set version (compatibility)
    do_constant_folding=True,    # fold constants for small optimizations
)

print("Wrote resnet18.onnx")
