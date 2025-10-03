# 🚀 TVM Setup with Docker

## 1. Pull a TVM Docker image

**CPU-only (simplest):**

``` bash
docker pull tlcpack/ci-cpu:latest
docker run --rm -it -v $PWD:/workspace -w /workspace tlcpack/ci-cpu:latest bash
```

**GPU (NVIDIA CUDA):** 1. Install [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Run:

``` bash
docker pull tlcpack/ci-gpu:latest
docker run --rm -it --gpus all -v $PWD:/workspace -w /workspace tlcpack/ci-gpu:latest bash
```

------------------------------------------------------------------------

## 2. Clone TVM source (inside the container)

``` bash
git clone --recursive https://github.com/apache/tvm
cd tvm
```

------------------------------------------------------------------------

## 3. Build TVM (optional if you only use `tvmc`)

``` bash
mkdir build && cp cmake/config.cmake build/
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
python3 -m pip install --upgrade pip
python3 -m pip install -r python/requirements.txt
python3 -m pip install -e python
```

------------------------------------------------------------------------

## 4. Compile & Run with `tvmc`

**Compile ONNX model (CPU):**

``` bash
tvmc compile   --target "llvm"   --input-shape "input:1,3,224,224"   --output model.tar   model.onnx
```

**Run & benchmark (CPU):**

``` bash
tvmc run --device cpu --inputs input.npy --output output.npy model.tar
tvmc benchmark --device cpu model.tar
```

**Compile for CUDA:**

``` bash
tvmc compile   --target "cuda"   --input-shape "input:1,3,224,224"   --output model_cuda.tar   model.onnx
```

------------------------------------------------------------------------

## 5. Auto-tuning (Ansor AutoScheduler)

**Start RPC tracker:**

``` bash
python3 -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190
```

**Start RPC server (on target device):**

``` bash
python3 -m tvm.exec.rpc_server --tracker 127.0.0.1:9190 --key linux-x86
```

**Run tuning:**

``` bash
tvmc tune model.onnx   --target "llvm"   --rpc-tracker "127.0.0.1:9190"   --rpc-key "linux-x86"   --output tuning.json
```

**Compile with tuning logs:**

``` bash
tvmc compile   --target "llvm"   --tuning-records tuning.json   --output model_optimized.tar   model.onnx
```

------------------------------------------------------------------------

## 6. Minimal Python API Example

``` python
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import onnx, numpy as np

onnx_model = onnx.load("model.onnx")
shape_dict = {"input": (1,3,224,224)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params=params)

dev = tvm.cpu()
m = graph_executor.GraphModule(lib["default"](dev))
x = np.load("input.npy")
m.set_input("input", tvm.nd.array(x))
m.run()
out = m.get_output(0).numpy()
```

------------------------------------------------------------------------

✅ With this setup: - Use **`ci-cpu`** for CPU workflows.\
- Use **`ci-gpu` + NVIDIA toolkit** for CUDA.\
- Start with `tvmc compile` → `tvmc run` → `tvmc benchmark`.\
- Add **auto-tuning** with RPC tracker + logs.
