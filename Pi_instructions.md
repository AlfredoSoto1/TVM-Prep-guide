# TVM RPC Setup: Cross-Compilation from Laptop to Raspberry Pi / CM5

This guide documents how to set up a Raspberry Pi / Compute Module 5 as a TVM RPC target, compile a model on a laptop/devcontainer, and execute the compiled model remotely on the Raspberry Pi.

---

# 1. System Overview

The workflow is:

```text
Laptop / Devcontainer
    |
    | 1. Compile model with TVM for ARM64
    | 2. Export .so using aarch64 cross-compiler
    | 3. Upload .so through RPC
    v

Raspberry Pi / CM5
    |
    | 4. Load compiled .so
    | 5. Run inference using TVM runtime
    v

Remote inference result returned to laptop
```

The laptop is the host.  
The Raspberry Pi is the target.

---

# 2. Raspberry Pi Setup

## 2.1 Clone TVM

```bash
cd ~
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
git checkout <TVM_VERSION_OR_COMMIT>
git submodule update --init --recursive
```

Replace:

```text
<TVM_VERSION_OR_COMMIT>
```

with the TVM version/commit that works for the project.

---

## 2.2 Build TVM Runtime on the Raspberry Pi

```bash
cd ~/tvm
mkdir build
cp cmake/config.cmake build/
cd build
```



Build:

```bash
cmake ..
cmake --build . --config Release -j
```

Verify:

```bash
ls ~/tvm/build | grep libtvm
```

Expected:

```
libtvm.so
libtvm_runtime.so
```

---

# 3. Configure TVM Python on Raspberry Pi

Run:

```bash
export TVM_HOME=$HOME/tvm
export TVM_LIBRARY_PATH=$TVM_HOME/build
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export LD_LIBRARY_PATH=$TVM_HOME/build:$LD_LIBRARY_PATH
```

If using a newer TVM version that includes `tvm-ffi`, also add:

```bash
export PYTHONPATH=$TVM_HOME/3rdparty/tvm-ffi/python:$PYTHONPATH
```

Test:

```bash
python3 -c "import tvm; print('TVM OK')"
```

If a dependency is missing, install it:

```bash
pip3 install <missing_package> --break-system-packages
```

---

# 4. Make Environment Variables Permanent

Add to `~/.bashrc`:

```bash
echo 'export TVM_HOME=$HOME/tvm' >> ~/.bashrc
echo 'export TVM_LIBRARY_PATH=$TVM_HOME/build' >> ~/.bashrc
echo 'export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/3rdparty/tvm-ffi/python:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$TVM_HOME/build:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

# 5. Start TVM RPC Server on Raspberry Pi

On the Raspberry Pi:

```bash
python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
```

Expected output:

```text
INFO bind to 0.0.0.0:9090
```

Leave this terminal open.

---

# 6. Get Raspberry Pi IP Address

On the Raspberry Pi:

```bash
hostname -I
```

Example:

```
10.34.2.2
```

Use this IP on the laptop.

---

# 7. Laptop / Devcontainer Setup

Install the ARM64 cross-compiler inside the devcontainer:

```bash
sudo apt update
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

Verify:

```bash
which aarch64-linux-gnu-g++
```

Expected:

```text
/usr/bin/aarch64-linux-gnu-g++
```

---

# 8. Compile Model for Raspberry Pi

Example target:

```python
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
```

Compile:

```python
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=target, params=params)
```

Export using the cross-compiler:

```python
from tvm.contrib import cc

lib.export_library(
    "model_tvm.so",
    fcompile=cc.cross_compiler("aarch64-linux-gnu-g++")
)
```

Use `opt_level=0` first for debugging.  
After it works, try `opt_level=3`.

---

# 9. Run Model on Raspberry Pi through RPC

```python
from tvm import rpc
from tvm.contrib import graph_executor

PI_IP = "<RASPBERRY_PI_IP>"
PORT = 9090

remote = rpc.connect(PI_IP, PORT)

remote.upload("model_tvm.so")
rlib = remote.load_module("model_tvm.so")

dev = remote.cpu(0)

module = graph_executor.GraphModule(
    rlib["default"](dev)
)

module.set_input(
    "input0",
    tvm.nd.array(img.astype("float32"))
)

module.run()

output = module.get_output(0)
```

If the Raspberry Pi terminal shows something like:

```text
INFO connected from ...
INFO load_module /tmp/.../model_tvm.so
```

then the model was uploaded and loaded on the Raspberry Pi.

---

# 10. Confirm It Ran on the Raspberry Pi

The strongest indicators are:

1. The RPC server logs show connection and module loading.
2. The `.so` is compiled for `aarch64`, so the x86 laptop cannot execute it locally.
3. The execution uses:

```python
dev = remote.cpu(0)
```

Meaning:

```text
remote.cpu(0) = Raspberry Pi CPU
tvm.cpu(0)    = laptop/devcontainer CPU
```

---



# 11. Final Working Checklist

On Raspberry Pi:

```bash
python3 -c "import tvm; print('TVM OK')"
python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
```

On laptop/devcontainer:

```bash
which aarch64-linux-gnu-g++
```

In Python:

```python
remote = rpc.connect("<RASPBERRY_PI_IP>", 9090)
print("RPC OK")
```

If all three work, the TVM RPC pipeline is ready.