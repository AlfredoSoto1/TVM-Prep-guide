# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.11

ENV DEBIAN_FRONTEND=noninteractive

# ---- Build essentials (gcc/g++, make), CMake, Ninja, pkg-config ----
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

# ---- Python deps (pin numpy<2 for broad TVM wheel compat) ----
RUN python -m pip install --upgrade pip && \
  python -m pip install "numpy<2.0" apache-tvm jupyter pillow pytest


# Expose TVM headers & runtime to /usr/local so compilers find them by default
RUN python - <<'PY' \
 && echo "/usr/local/lib" > /etc/ld.so.conf.d/usr-local-lib.conf \
 && ldconfig
import os, pathlib, tvm
pkg = pathlib.Path(tvm.__file__).parent
inc = pkg / "include"
cands = [pkg/"libtvm_runtime.so", pkg/"libtvm_runtime.dylib", pkg/"tvm.dll"]
lib = next((p for p in cands if p.exists()), None)
if not inc.exists(): raise SystemExit(f"TVM include not found at {inc}")
if lib is None:      raise SystemExit("TVM runtime library not found in wheel")
os.makedirs("/usr/local/include", exist_ok=True)
os.makedirs("/usr/local/lib", exist_ok=True)
os.system(f"ln -sfn {inc} /usr/local/include/tvm")
os.system(f"ln -sfn {lib} /usr/local/lib/{lib.name}")
print("[tvm] include -> /usr/local/include/tvm")
print(f"[tvm] lib     -> /usr/local/lib/{lib.name}")
PY

