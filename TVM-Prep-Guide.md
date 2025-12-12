# TVM-Prep-guide

# TVM v0.19.0 – Clone & Setup (with all dependencies)

> **Goal:** Get the TVM *runtime* built (for running models from C++/Python) and ensure **all 3rdparty submodules** are pulled correctly for tag `v0.19.0`.

## 0) Prereqs (Ubuntu/Debian minimal)
If you are in the devcontainer you can ignore this.
```bash
sudo apt-get update
sudo apt-get install -y git cmake build-essential python3 python3-venv libopenblas-dev libz-dev
```
*(You don’t need LLVM for the runtime-only build.)*

---

## 1) Clone the repo & checkout **v0.19.0**

> Note: `v0.19.0` is a **tag**, not a branch.

```bash
# Clone (full history is safest for tags/submodules)
git clone https://github.com/apache/tvm.git
cd tvm

# Make sure we have all tags and checkout the v0.19.0 tag
git fetch --tags
git checkout v0.19.0
```

---

## 2) Pull **all submodules** for this tag

```bash
# Sync submodule URLs and init recursively
git submodule sync --recursive
git submodule update --init --recursive --jobs 8
```

**Verify** you have critical submodules (non-empty dirs):
```
tvm/3rdparty/dlpack
tvm/3rdparty/dmlc-core
tvm/3rdparty/libbacktrace
```

> If any of these are missing/empty, rerun the `git submodule …` command.

---

## 3) Build the **runtime only** (recommended for C++ runner)

```bash
mkdir -p build && cd build

cmake ..                  \
  -DUSE_LLVM=OFF          \   
  -DUSE_CUDA=OFF          \   
  -DUSE_VULKAN=OFF        \   
  -DUSE_OPENCL=OFF        \   
  -DUSE_METAL=OFF         \   
  -DUSE_RPC=ON            \   
  -DUSE_GRAPH_EXECUTOR=ON \   
  -DUSE_AOT_EXECUTOR=ON   \

cmake --build . --config Release -j
```

You should now have:
```
tvm/build/libtvm_runtime.so       # (Linux)
```
Once you have the runtime, you can go to `tvm_cpp` and compile the resnet example program (following it's instructions)


(if you are in devcontainer ignore this)
**Optional:** Set env vars for later:
```bash
export TVM_HOME=$(cd .. && pwd)
export TVM_BUILD=$(pwd)
```