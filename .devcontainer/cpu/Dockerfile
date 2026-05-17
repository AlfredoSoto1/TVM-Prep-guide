FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

ARG TVM_VERSION=v0.19.0
ARG TVM_HOME=/opt/tvm
ARG TVM_BUILD=/opt/tvm/build

ENV DEBIAN_FRONTEND=noninteractive \
    TVM_VERSION=${TVM_VERSION} \
    TVM_HOME=${TVM_HOME} \
    TVM_BUILD=${TVM_BUILD} \
    TVM_USE_CUDA=OFF \
    TVM_USE_VULKAN=OFF \
    LLVM_CONFIG=/usr/bin/llvm-config-16

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    python3-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libz-dev \
    llvm-16 \
    llvm-16-dev \
    clang-16 \
    lld-16 \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    emscripten \
    wabt \
    binaryen \
    libvulkan-dev \
    vulkan-tools \
    glslang-tools \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "${TVM_HOME}" "${TVM_BUILD}" \
    && chown -R vscode:vscode "${TVM_HOME}"
