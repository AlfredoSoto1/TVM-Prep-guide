
# Apache TVM Model Compilation inside Docker

## Choose Your Model Format

TVM supports many frontends. Common paths:

| Framework  | Conversion Step     | Import Function                    |
| ---------- | ------------------- | ---------------------------------- |
| PyTorch    | Export to ONNX      | `relay.frontend.from_onnx()`       |
| TensorFlow | Save as FrozenGraph | `relay.frontend.from_tensorflow()` |

For simplicity, this README uses **PyTorch → ONNX → TVM**.

---

## Export a PyTorch Model to ONNX

Create a script called **`export_resnet_onnx.py`**:

```python
import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True).eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    do_constant_folding=True,
)

print("Wrote resnet18.onnx")
```

Run:

```bash
python export_resnet_onnx.py
```

You should now have:

```
resnet18.onnx
```

---

## Compile the ONNX Model with TVM (CPU)

Create **`compile_tvm_cpu.py`**:

```python
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

print("Compiled and exported TVM artifacts")
```

Run:

```bash
python compile_tvm_cpu.py
```

You should now have:

```
deploy_lib_cpu.so
deploy_graph.json
deploy_params.params
```

---

## Run Inference with TVM Runtime

Create **`run_tvm_inference.py`**:

```python
import tvm
from tvm.contrib import graph_executor
import numpy as np

# Load compiled module
lib = tvm.runtime.load_module("deploy_lib_cpu.so")
with open("deploy_graph.json") as f:
    graph_json = f.read()
params = bytearray(open("deploy_params.params", "rb").read())

dev = tvm.cpu(0)
module = graph_executor.create(graph_json, lib, dev)
module.load_params(params)

# Dummy input
input_data = np.random.randn(1, 3, 224, 224).astype("float32")
module.set_input("input", input_data)
module.run()

out = module.get_output(0).asnumpy()
print("✅ Output shape:", out.shape)
```

Run:

```bash
python run_tvm_inference.py
```

You should see something like:

```
Output shape: (1, 1000)
```

---

## GPU Compilation (Optional)

If your Docker container has access to GPU and TVM was built with CUDA:

1. Change the target in your compile script:

   ```python
   target = "cuda"
   ```

2. Update device selection in the runtime script:

   ```python
   dev = tvm.cuda(0)
   ```

3. Run your container with GPU access:

   ```bash
   docker run --gpus all -it <image_name> /bin/bash
   ```

---

## Deploying Artifacts

The three essential files for deployment:

| File                   | Purpose                  |
| ---------------------- | ------------------------ |
| `deploy_graph.json`    | Model graph              |
| `deploy_lib_cpu.so`    | Compiled runtime library |
| `deploy_params.params` | Model weights            |

Copy these to any target device and load with TVM’s runtime (Python, C++, or Java).

---

## 🧪 7️⃣ Optional: Performance Tuning

You can use **AutoScheduler** or **AutoTVM** to find optimal kernel schedules for your target hardware.

Example (basic AutoScheduler):

```python
from tvm import auto_scheduler
```



## Summary

| Step | Action                          |
| ---- | ------------------------------- |
| 0    | Verify TVM setup inside Docker  |
| 1    | Export model to ONNX            |
| 2    | Compile with TVM (Relay → LLVM) |
| 3    | Save compiled artifacts         |
| 4    | Run with graph_executor         |
| 5    | (Optional) Compile for GPU      |
| 6    | (Optional) Tune for speed       |

---



