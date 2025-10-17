
# Apache TVM Model Compilation inside Docker (CPU)

This guide expands the “what and why” behind each step, then walks you through 
export → compile → run using TVM’s Python runtime and shows how to run the 
compiled model from C/C++ and Rust. It assumes your devcontainer already has TVM (CPU/LLVM) installed.

## How does it work?
TVM is a compiler for machine-learning models. You give it a model description (graph + weights), 
and it produces optimized native code for your hardware (e.g., x86 CPU with AVX).

### Compiling ONNX

- ONNX = Open Neural Network Exchange — a portable file format (.onnx) that describes a 
trained model: its operators (Conv, ReLU…), graph connections, and weights.
- Training frameworks (PyTorch, TensorFlow, Keras…) can export their models to ONNX.
- Compilers/runtimes (TVM, ONNX Runtime…) can import ONNX without caring how it was trained.

### What TVM does with ONNX
1. Import: Parse the ONNX graph and weights.
2. Lower to Relay: Convert to TVM’s IR (Relay) so compiler optimizations can run.
3. Optimize: Fuse ops, simplify graph, pick fast kernels, vectorize, etc.
4. Codegen: Generate machine code (via LLVM for CPU).
5. Package artifacts: A shared library `.so`, plus graph/params (for the runtime).

### Three deployment artifacts from compilation
| File                   | What it is (concept)                               |
| ---------------------- | -------------------------------------------------- |
| `deploy_lib_cpu.so`    | The **compiled kernels** (native machine code).    |
| `deploy_graph.json`    | The **execution plan** (nodes, edges, scheduling). |
| `deploy_params.params` | The **learned weights** (binary blob).             |

The runtime takes these three, binds inputs/outputs, and runs inference on your device.

### Choose Your Model Format
TVM supports many frontends. Common paths:

| Framework  | Conversion Step                  | Import Function                    |
| ---------- | -------------------------------- | ---------------------------------- |
| PyTorch    | Export to **ONNX**               | `relay.frontend.from_onnx()`       |
| TensorFlow | Convert (SavedModel/FrozenGraph) | `relay.frontend.from_tensorflow()` |
| TFLite     | Use `.tflite` directly           | `relay.frontend.from_tflite()`     |

For simplicity, this README uses **PyTorch → ONNX → TVM.**

---

## Export a PyTorch Model to ONNX

Create a script called **`export_resnet_onnx.py`**:

```python
import torch
import torchvision

# 1) Load a pretrained model (already trained; we're not training here)
model = torchvision.models.resnet18(pretrained=True).eval()

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

print("Compiled and exported TVM artifacts:")
print(" - deploy_lib_cpu.so")
print(" - deploy_graph.json")
print(" - deploy_params.params")
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

# 1) Load compiled operator library (.so)
lib = tvm.runtime.load_module("deploy_lib_cpu.so")

# 2) Load execution graph (.json) and parameters (.params)
with open("deploy_graph.json") as f:
    graph_json = f.read()
with open("deploy_params.params", "rb") as f:
    param_bytes = f.read()

# 3) Create executor on a device
dev = tvm.cpu(0)
module = graph_executor.create(graph_json, lib, dev)

# 4) Load parameters into the executor
module.load_params(param_bytes)

# 5) Prepare and set input (must match name/shape/dtype)
input_data = np.random.randn(1, 3, 224, 224).astype("float32")
module.set_input("input", input_data)

# 6) Run and fetch output(s)
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
## (Optional) TVMC: No-Code CLI Workflow

```bash
# Compile to a .tar bundle that contains all artifacts
tvmc compile \
  --target "llvm -mcpu=native" \
  --output resnet18_cpu.tar \
  resnet18.onnx

# Run with dummy input; --profile prints timing
tvmc run \
  --device cpu \
  --profile \
  resnet18_cpu.tar

```

## Running Your TVM Build from C/C++

Create `run_tvm_cpp.cpp`:

```cpp
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace tvm::runtime;

static std::string LoadText(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
}

static std::string LoadBinary(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
}

int main() {
    // 1) Load compiled operator library
    Module lib = Module::LoadFromFile("deploy_lib_cpu.so");

    // 2) Load Graph JSON
    std::string graph_json = LoadText("deploy_graph.json");

    // 3) Create a device
    Device dev{kDLCPU, 0};

    // 4) Create Graph Executor via packed function "tvm.graph_executor.create"
    //    This returns a Module that exposes set_input/run/get_output/load_params
    PackedFunc graph_create = (*Registry::Get("tvm.graph_executor.create"));
    Module gmod = graph_create(graph_json, lib, dev);

    // 5) Load parameters (serialized param blob)
    std::string params_data = LoadBinary("deploy_params.params");
    PackedFunc load_params = gmod.GetFunction("load_params");
    TVMByteArray arr;
    arr.data = params_data.data();
    arr.size = params_data.size();
    load_params(arr);

    // 6) Prepare input NDArray
    //    Shape must match (1, 3, 224, 224), dtype float32
    DLTensor* tmp;
    NDArray input = NDArray::Empty({1, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, dev);
    {
        // Fill with random data just for demo
        std::vector<float> host(1 * 3 * 224 * 224);
        std::mt19937 gen(0);
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto &v : host) v = dist(gen);
        input.CopyFromBytes(host.data(), host.size() * sizeof(float));
    }

    // 7) Set input by name
    PackedFunc set_input = gmod.GetFunction("set_input");
    set_input("input", input);

    // 8) Run
    PackedFunc run = gmod.GetFunction("run");
    run();

    // 9) Get output tensor 0
    NDArray out = NDArray::Empty({1, 1000}, DLDataType{kDLFloat, 32, 1}, dev);
    PackedFunc get_output = gmod.GetFunction("get_output");
    get_output(0, out);

    // 10) Pull result to host and print shape
    std::vector<float> host_out(1000);
    out.CopyToBytes(host_out.data(), host_out.size() * sizeof(float));
    std::cout << "OK! output[0:5] = "
              << host_out[0] << ", "
              << host_out[1] << ", "
              << host_out[2] << ", "
              << host_out[3] << ", "
              << host_out[4] << std::endl;
    return 0;
}
```


## Running Your TVM Build from Rust


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

| Step | Action                                    | Why it matters                                    |
| ---- | ----------------------------------------- | ------------------------------------------------- |
| 0    | Verify TVM inside Docker                  | Ensures runtime & headers are available           |
| 1    | **Export to ONNX** (from PyTorch)         | Gives TVM a portable, static model graph          |
| 2    | **Compile with TVM** (Relay → LLVM)       | Produces optimized native code for your CPU       |
| 3    | Save **artifacts** (`.so`, graph, params) | These are your deployable inference package       |
| 4    | **Run** with TVM Python runtime           | Quick validation; good for prototyping            |
| 5    | **Run** from **C/C++**                    | Ship a minimal, Python-free production runtime    |
| 6    | (Optional) **Tune** for speed             | Hardware-aware schedules = real performance gains |
| 7    | (Optional) **GPU** target (`cuda`)        | Compile for NVIDIA GPUs if available              |
---