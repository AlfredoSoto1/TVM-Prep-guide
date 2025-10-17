import onnx
import tvm
from tvm import relay
from tvm.contrib import graph_executor

onnx_model = onnx.load("resnet18.onnx")

# Input shape
input_name = "input"
shape_dict = {input_name: (1, 3, 224, 224)}

# Import model to Relay
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# Target: CPU (change to "cuda" for GPU)
target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Save compiled artifacts
lib.export_library("deploy_lib_cpu.so")
with open("deploy_graph.json", "w") as f:
    f.write(lib.get_graph_json())
with open("deploy_params.params", "wb") as f:
    f.write(tvm.runtime.save_param_dict(lib.get_params()))

print("Compiled and exported TVM artifacts:")
print(" - deploy_lib_cpu.so")
print(" - deploy_graph.json")
print(" - deploy_params.params")
